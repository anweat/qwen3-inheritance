# E6 second opinion: the stage 5 segmental panel (SenseVoice, Paraformer, Whisper; strict majority of centred log odds > 0),
# on items whose two readings differ in initial or final. Reported, not judged (the preset verdict uses the stage 4 detector).
# Also counts how often a judge's top reading leaves the gold/alt pair, a rough flag for garbled or third readings.
#   ../../.venv/bin/python judge_summary.py [whisper|whisper_cuda]
# O and P are always scored by the same Whisper backend.
import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e4_tts_structure"))
from judge_panel import centred, ctype, scores, seg

E6 = os.path.expanduser("~/.cache/qwen-all/e6")
items = json.load(open(f"{E6}/plan.json", encoding="utf-8"))["items"]
PANEL = ["sensevoice", "paraformer", sys.argv[1] if len(sys.argv) > 1 else "whisper"]  # whisper (MLX, Mac) or whisper_cuda (laptop)
O_SRC = dict(hard="tts_out/TTS-0.6B_nonstream", natural="tts_natural/TTS-0.6B_nonstream", natural2="tts_natural2/TTS-0.6B_nonstream")


def vote(per_judge, rid, clip):
    v = [centred(s.get(rid, {}), clip) for s in per_judge]
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return None
    pos = sum(x > 0 for x in v)
    return pos * 2 > len(v)


def left(per_judge, rid, clip, r):
    out = []
    for s in per_judge:
        c = s.get(rid, {}).get(clip)
        if c and c.get("top"):
            out.append(seg(c["top"][0][0]) not in (seg(r["gold"]), seg(r["alt"])))
    return np.mean(out) if out else None


def boot(rows, a, b, n=4000):
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    ks = list(by)
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n):
        s = [x for j in rng.integers(0, len(ks), len(ks)) for x in by[ks[j]]]
        v.append(np.mean([a(x) for x in s]) - np.mean([b(x) for x in s]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


print("==== E6 panel check (segmental contrasts only; strict majority of 3 judges) ====")
for s in ("hard", "natural", "natural2", "dict"):
    newp = [scores(j, f"{E6}/asr/{s}") for j in PANEL]
    oldp = [scores(j, O_SRC[s]) for j in PANEL] if s in O_SRC else newp
    rows = []
    for r in items[s]:
        if ctype(r["gold"], r["alt"]) != "seg":
            continue
        clips = ("P_g2pw", "P_dict") if s == "dict" else ("P",)
        o, ps = vote(oldp, r["id"], "O"), {c: vote(newp, r["id"], c) for c in clips}
        if o is None or any(v is None for v in ps.values()):
            continue
        n3 = all(sum(centred(x.get(r["id"], {}), c) is not None for x in p) == 3 for p, c in [(oldp, "O")] + [(newp, c) for c in clips])
        rows.append(dict(r, O=o, **ps, n3=n3, left_O=left(oldp, r["id"], "O", r), left_P=left(newp, r["id"], clips[-1], r)))
    if not rows:
        print(f"  {s}: no complete rows yet")
        continue
    parts = [("CPP minority", [r for r in rows if r["kind"] == "min"]), ("CPP majority", [r for r in rows if r["kind"] == "maj"])] if s == "hard" else [(s, rows)]
    for label, rs in parts:
        if not rs:
            continue
        print(f"  ({label}: {sum(r['n3'] for r in rs)} of {len(rs)} rows voted by all three judges, the rest by two)")
        lp = [r["left_P"] for r in rs if r["left_P"] is not None]
        lo_ = [r["left_O"] for r in rs if r["left_O"] is not None]
        if s == "dict":
            print(f"  {label:13s} n {len(rs):3d}  TTS {np.mean([r['O'] for r in rs]):.3f}  g2pW only {np.mean([r['P_g2pw'] for r in rs]):.3f}"
                  f"  dictionary {np.mean([r['P_dict'] for r in rs]):.3f}  top reading outside pair: TTS {np.mean(lo_):.2f}, dictionary {np.mean(lp):.2f}")
            continue
        lo, hi = boot(rs, lambda r: r["P"], lambda r: r["O"])
        print(f"  {label:13s} n {len(rs):3d}  TTS {np.mean([r['O'] for r in rs]):.3f}  pipeline {np.mean([r['P'] for r in rs]):.3f}"
              f"  diff {np.mean([r['P'] for r in rs]) - np.mean([r['O'] for r in rs]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"  top reading outside pair: TTS {np.mean(lo_):.2f}, pipeline {np.mean(lp):.2f}")
