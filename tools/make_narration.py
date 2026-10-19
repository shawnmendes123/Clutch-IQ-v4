"""Synthesises the walkthrough narration with Piper, one wav per caption.

    node tools/record_walkthrough.js            # with VW_DUMP=caps.json, to list the captions
    python tools/make_narration.py caps.json out_dir voice.onnx

Writes out_dir/<key>.wav plus out_dir/narration.json {key: {file, dur, text, speech}}.
The key is the sha1 of the caption with its HTML stripped, which is what the
recorder computes at runtime, so pacing and voice stay in step.
"""
import hashlib
import json
import re
import sys
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

# said aloud rather than shown: symbols a text-to-speech voice cannot read
SPOKEN = [
    (r"Q\(S[′']?,\s*·?\)", "Q of S prime"),
    (r"π\(a\|s\)", "pi of a given s"),
    (r"p\(s[′']\|s,\s*a\)", "p of s prime given s and a"),
    (r"q\*", "q star"), (r"V\*", "V star"), (r"Q\*", "Q star"),
    (r"−", " minus "), (r"%", " percent"), (r"\bln\b", "log"),
    (r"π\*", "pi star"), (r"\bπ\b", "pi"), (r"π", "pi"),
    (r"γ", "gamma"), (r"ε", "epsilon"), (r"α", "alpha"), (r"β", "beta"),
    (r"θ", "theta"), (r"ρ", "rho"), (r"Δ", "delta"), (r"Σ", "sum of"),
    (r"√", "square root of"), (r"∈", "in"), (r"≈", "about"), (r"±", "plus or minus"),
    (r"≥", "at least"), (r"≤", "at most"), (r"⁻¹", " inverse"), (r"×", " by "),
    (r"→", " to "), (r"·", " "), (r"—", ", "), (r"–", ", "),
    (r"1e-6", "ten to the minus six"), (r"1e-8", "ten to the minus eight"),
    (r"\b1e-(\d+)\b", r"ten to the minus \1"),
    (r"V\(s\)", "V of s"), (r"Q\(s,a\)", "Q of s a"), (r"Q\(S',·\)", "Q of S prime"),
    (r"p\(s′\|s,a\)", "p of s prime given s and a"),
    (r"G₀|G-nought|G0", "G nought"), (r"G_t|Gt\b", "G t"),
    (r"\bR\s*\+\s*gamma\s*max\b", "R plus gamma times max"),
    (r"\(I − gamma P\)\s*inverse\s*R", "I minus gamma P, inverse, times R"),
    (r"48×3", "48 by 3"), (r"\bs = 12f \+ 3t \+ k\b", "s equals twelve f plus three t plus k"),
]


SUBS = {"t+1": " t plus one", "t": " t", "0": " nought", "k+1": " k plus one", "k": " k"}


def to_speech(caption: str) -> str:
    # subscripts first: "R<sub>t+1</sub>" should be read, not glued into "Rt+1"
    t = re.sub(r"<sub>(.*?)</sub>", lambda m: SUBS.get(m.group(1), " " + m.group(1)), caption)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("&nbsp;", " ").replace("&rarr;", " to ")
    for pat, rep in SPOKEN:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def key_of(caption: str) -> str:
    plain = re.sub(r"<[^>]+>", "", caption)
    plain = re.sub(r"\s+", " ", plain).strip()
    return hashlib.sha1(plain.encode()).hexdigest()[:16]


def main() -> None:
    caps_path, out_dir, voice_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = [c for c in json.load(open(caps_path)) if c.get("text")]

    voice = PiperVoice.load(voice_path)
    cfg = SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8)

    index, total = {}, 0.0
    for n, c in enumerate(caps, 1):
        k = key_of(c["text"])
        if k in index:
            continue
        speech = to_speech(c["text"])
        wav_path = out_dir / f"{k}.wav"
        with wave.open(str(wav_path), "wb") as w:
            voice.synthesize_wav(speech, w, syn_config=cfg)
        with wave.open(str(wav_path)) as w:
            d = w.getnframes() / w.getframerate()
        index[k] = {"file": wav_path.name, "dur": round(d, 3), "text": c["text"], "speech": speech}
        total += d
        if n % 25 == 0:
            print(f"  {n}/{len(caps)} … {total/60:.1f} min of speech", flush=True)

    json.dump(index, open(out_dir / "narration.json", "w"), indent=1)
    print(f"{len(index)} clips, {total/60:.1f} minutes of narration -> {out_dir}")


if __name__ == "__main__":
    main()
