# Scores poly_natural.py output with the ASR detector, applying the criterion in its header.
#   python poly_natural_summary.py <jsonl>
import json, os, sys
import numpy as np


def boot(by, fn, n=4000):
    keys = list(by)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        s = [x for j in rng.integers(0, len(keys), len(keys)) for x in by[keys[j]]]
        if s:
            vals.append(fn(s))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(path[:-len(".jsonl")] + "_asr.jsonl", encoding="utf-8"))}
rows = [dict(r, l=asr[r["id"]]) for r in rows if asr.get(r["id"])]
for r in rows:
    l = r["l"]
    mid = (l["R_g"] + l["R_a"]) / 2
    gap = l["R_g"] - l["R_a"]
    r["H"], r["O"], r["gap"] = l["H"] - mid, l["O"] - mid, gap
    r["pH"], r["pO"] = (l["H"] - l["R_a"]) / gap, (l["O"] - l["R_a"]) / gap
print(f"{len(rows)} positions, {len({r['char'] for r in rows})} characters, reference gap median {np.median([r['gap'] for r in rows]):.2f}, gap<=0 share {np.mean([r['gap'] <= 0 for r in rows]):.3f}")
for label, sub in (("all", rows), ("gold is CPP majority", [r for r in rows if r["kind"] == "maj"]), ("gold is CPP minority", [r for r in rows if r["kind"] == "min"])):
    by = {}
    for r in sub:
        by.setdefault(r["char"], []).append(r)
    for k in ("H", "O"):
        lo, hi = boot(by, lambda s: np.mean([x[k] > 0 for x in s]))
        print(f"  {label:22s} {k}: n {len(sub):3d} chars {len(by):3d} says gold {np.mean([r[k] > 0 for r in sub]):.3f} [{lo:.3f}, {hi:.3f}]  position pct {np.percentile([r['p' + k] for r in sub], [10, 50, 90]).round(2)}")
    if label == "all":
        lo, hi = boot(by, lambda s: np.mean([x["O"] > 0 for x in s]) - np.mean([x["H"] > 0 for x in s]))
        print(f"  O minus H {np.mean([r['O'] > 0 for r in sub]) - np.mean([r['H'] > 0 for r in sub]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
h = np.mean([r["H"] > 0 for r in rows])
print(f"criterion: human audio scored as gold >= .90: {h:.3f} -> {'transfers' if h >= .9 else 'FAILS, stage 4 rates may be biased'}")
print("human scored wrong:")
for r in rows:
    if r["H"] <= 0:
        print(f"  {r['char']} {r['gold']}/{r['alt']} H {r['H']:+.1f} O {r['O']:+.1f} gap {r['gap']:.1f} {r['text'][max(0, r['scores']['texts']['R_g'].find(r['text'][0])):][:30]}")
print("TTS scored wrong:")
for r in rows:
    if r["O"] <= 0:
        print(f"  {r['char']} {r['gold']}/{r['alt']} O {r['O']:+.1f} H {r['H']:+.1f} gap {r['gap']:.1f} id {r['id']} {r['text'][:34]}")
