"""Muxes the narration onto the recorded walkthrough.

    python tools/build_narrated_video.py raw.webm captions.json narr_dir out.mp4

The recorder flashes an 8px strip at the bottom of the frame when each caption
appears. This finds those flashes, places each caption's wav at its flash, and
crops the strip off the delivered video, so speech and captions stay in step
however the recording's own clock drifted.
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

H, W, STRIP = 900, 1600, 8
RATE = 22050


def marker_times(video: str) -> list[float]:
    """Timestamps where the sync strip turns white."""
    # file=- sends the per-frame values to stdout, independent of the log level
    out = subprocess.run([
        "ffmpeg", "-v", "error", "-i", video,
        "-vf", f"crop={W}:{STRIP}:0:{H - STRIP},scale=1:1,format=gray,"
               "signalstats,metadata=mode=print:key=lavfi.signalstats.YAVG:file=-",
        "-f", "null", "-",
    ], capture_output=True, text=True).stdout

    times, vals = [], []
    t = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            t = float(line.split("pts_time:")[1].split()[0])
        elif "YAVG" in line and t is not None:
            vals.append(float(line.split("=")[1]))
            times.append(t)

    onsets, lit = [], False
    for t, v in zip(times, vals):
        if v > 160 and not lit:
            onsets.append(t)
            lit = True
        elif v < 100:
            lit = False
    # lossy compression can split one flash into two detections, and the very
    # first frame is bright before the overlay exists
    merged = [t for t in onsets if t > 2.0]
    out = [t for i, t in enumerate(merged) if i == 0 or t - merged[i - 1] > 0.6]
    print(f"  {len(onsets)} raw detections -> {len(out)} flashes")
    return out


def align(caps: list[dict], onsets: list[float]) -> list[float]:
    """Map each caption to a flash. The recorder logs wall-clock times and the
    video clock drifts from them by a constant factor, so fit that factor and
    snap to the nearest flash; a caption whose flash was dropped keeps the
    predicted time rather than shifting every caption after it."""
    if not onsets:
        raise SystemExit("no marker flashes found — was the recording made with VW_AUDIO set?")
    wall = [c["t"] / 1000 for c in caps]
    scale = (onsets[-1] - onsets[0]) / (wall[-1] - wall[0]) if len(caps) > 1 else 1.0
    print(f"  video clock runs {scale:.4f}x the recorder's wall clock")
    out, snapped, j = [], 0, 0
    for w in wall:
        pred = onsets[0] + (w - wall[0]) * scale
        while j + 1 < len(onsets) and abs(onsets[j + 1] - pred) < abs(onsets[j] - pred):
            j += 1
        if abs(onsets[j] - pred) < 1.5:
            out.append(onsets[j]); snapped += 1
        else:
            out.append(pred)
    print(f"  {snapped}/{len(caps)} captions matched a flash exactly")
    return out


def main() -> None:
    raw, caps_path, narr_dir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    caps = json.load(open(caps_path))
    narr = json.load(open(narr_dir / "narration.json"))

    onsets = marker_times(raw)
    print(f"{len(caps)} captions, {len(onsets)} marker flashes")
    if len(onsets) == len(caps):
        times = onsets                      # one flash per caption: exact sync
        print("  counts match — pairing one to one")
    else:
        times = align(caps, onsets)

    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", raw], capture_output=True, text=True).stdout)
    track = np.zeros(int((dur + 5) * RATE), dtype=np.float32)

    placed = 0
    for cap, t in zip(caps, times):
        clip = narr.get(cap["key"])
        if not clip:
            continue
        with wave.open(str(narr_dir / clip["file"])) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            assert w.getframerate() == RATE, f"{clip['file']} is {w.getframerate()} Hz"
        i = int(t * RATE)
        end = min(i + len(a), len(track))
        track[i:end] += a[:end - i]
        placed += 1

    peak = float(np.abs(track).max()) or 1.0
    track = np.clip(track / peak * 0.89, -1, 1)
    wav_path = Path(out).with_suffix(".narration.wav")
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((track * 32767).astype(np.int16).tobytes())
    print(f"placed {placed}/{len(caps)} clips -> {wav_path}")

    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", raw, "-i", str(wav_path),
        "-vf", f"crop={W}:{H - STRIP}:0:0,fps=25,format=yuv420p",   # drop the sync strip
        "-c:v", "libx264", "-crf", "24", "-preset", "slow", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "96k", "-ac", "1", "-shortest", "-movflags", "+faststart", out,
    ], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
