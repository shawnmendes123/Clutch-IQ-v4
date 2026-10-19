"""Generate DEMO datasets (SIMULATED, seeded) for trying Clutch IQ.
Every file here is synthetic: invented gamertags, invented stats. Run:
    python tools/make_sample_datasets.py
"""
import csv, random, datetime as dt
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "sample_datasets"
OUT.mkdir(exist_ok=True)
rng = random.Random(2026)

NAMES = ["Vexen","Ryza","Kairo","Nyx","Orion","Zephyr","Mako","Sable","Juno","Rook","Pixel","Blaze",
         "Echo","Fang","Halo","Ivy","Jett","Koda","Lynx","Moxie","Nova","Onyx","Pyro","Quill"]
MAPS = ["Nuketown","Crash","Summit","Raid","Hijacked","Terminal","Rust","Scrapyard"]

def skill_profile(i):
    return dict(skill=rng.uniform(0.3, 0.9), aggr=rng.uniform(0.2, 0.9), tilt_prone=rng.uniform(0.1, 0.9))
PROFILES = {n: skill_profile(i) for i, n in enumerate(NAMES)}

def match_row(name, p, streak_loss, fatigue):
    # win prob falls with losing streak and fatigue (invented dynamics, SIMULATED)
    pw = min(0.9, max(0.1, 0.35 + 0.4*p["skill"] - 0.05*min(streak_loss,4)*p["tilt_prone"] - 0.02*fatigue))
    win = rng.random() < pw
    perf = p["skill"] + rng.gauss(0, .12) - 0.03*min(streak_loss,4)*p["tilt_prone"] + (0.08 if win else -0.05)
    kills = max(0, round(8 + 22*perf*(0.6+0.6*p["aggr"]) + rng.gauss(0, 3)))
    deaths = max(1, round(18 - 9*perf + rng.gauss(0, 3)))
    assists = max(0, round(3 + 8*(1-p["aggr"]) + rng.gauss(0, 2)))
    score = max(0, round(kills*100 + assists*50 + (300 if win else 0) + rng.gauss(0, 150)))
    return win, kills, deaths, assists, score

# 1) match history: dates, W/L, kills/deaths/assists, several sessions per player -------------
def match_history():
    rows = []; start = dt.datetime(2026, 8, 1, 17, 0)
    for name in NAMES[:16]:
        p = PROFILES[name]; t = start + dt.timedelta(hours=rng.randint(0, 5)); mid = 0
        for day in range(10):
            n = rng.randint(3, 8); sl = 0
            for g in range(n):
                win, k, d, a, s = match_row(name, p, sl, g)
                sl = 0 if win else sl + 1
                rows.append(dict(match_id=f"M{len(rows)+1:05d}", player=name,
                    date=(t + dt.timedelta(days=day, minutes=22*g)).strftime("%Y-%m-%d %H:%M"),
                    map=rng.choice(MAPS), kills=k, deaths=d, assists=a, score=s,
                    result="W" if win else "L"))
    write("match_history_demo.csv", rows)

# 2) scoreline-style results ("13-9") with 12 players -> exercises scoreline closeness -----
def scoreline_history():
    rows = []; t = dt.datetime(2026, 9, 1, 18, 0)
    for i in range(480):
        win = rng.random() < 0.52
        loser = rng.randint(3, 12)
        sc = f"13-{loser}" if win else f"{loser}-13"
        rows.append(dict(gamertag=NAMES[i % 12], timestamp=(t + dt.timedelta(minutes=35*(i//12))).strftime("%Y-%m-%d %H:%M:%S"),
            mode=rng.choice(["Search","Hardpoint","Control"]), result=sc,
            kills=rng.randint(6, 28), deaths=rng.randint(5, 24), assists=rng.randint(0, 12)))
    write("scoreline_history_demo.csv", rows)

# 3) tournament pair: players with stats + maps -----------------------------------------------
def tournament():
    rows = []
    for name in NAMES:
        p = PROFILES[name]
        rows.append(dict(name=name, kills=round(10+22*p["skill"]*(0.6+0.6*p["aggr"])),
            deaths=round(18-9*p["skill"]+rng.gauss(0,1.5)), assists=round(3+8*(1-p["aggr"])),
            score=round(1800+2600*p["skill"]), damage=round(1200+2000*p["skill"]),
            accuracy=round(0.18+0.3*p["skill"]+rng.gauss(0,.02), 3),
            headshot_percentage=round(0.1+0.3*rng.random(), 3)))
    write("tournament_players_demo.csv", rows)
    maps = [
     ("Nuketown","Small","Symmetrical","Team Deathmatch, Frontline","Aggressive Entry","No","Tiny two-lane map for close fights."),
     ("Crash","Medium","Three-lane","Team Deathmatch, Domination","Flex Rotator","No","Balanced three-lane layout."),
     ("Summit","Large","Open","Hardpoint, Control","Sniper/Isolation","Yes","Long sightlines and elevated perches."),
     ("Raid","Medium","Interior","Search, Hardpoint","Objective Anchor","No","Tight interiors around two objectives."),
     ("Hijacked","Small","Linear","Team Deathmatch","Aggressive Entry","No","Compact deck, constant contact."),
     ("Terminal","Large","Hub","Control, Domination","Passive Support","Yes","Central hub with flanking corridors."),
     ("Rust","Medium","Open","Domination","Flex Rotator","Yes","Dusty yard with mixed ranges."),
     ("Scrapyard","Medium","Maze","Search","Objective Anchor","No","Cluttered lanes reward map knowledge."),
    ]
    write("tournament_maps_demo.csv", [dict(rank=i+1, map_name=m[0], size=m[1], layout=m[2], best_modes=m[3],
        recommended_playstyle=m[4], sniper_friendly=m[5], description=m[6]) for i, m in enumerate(maps)])

def write(fn, rows):
    with open(OUT/fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(fn, len(rows), "rows")

if __name__ == "__main__":
    match_history(); scoreline_history(); tournament()
