# Applies the criterion in the poly_vector.py header. Group bootstrap for the balanced accuracy.
#   python poly_vector_summary.py <model>
import json, os, re, sys
import numpy as np
m = sys.argv[1]
ROOT = os.path.expanduser("~/.cache/qwen-all/pron/tts_vector")
rows = [json.loads(l) for l in open(f"{ROOT}/{m}_scored.jsonl", encoding="utf-8")]
asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(f"{ROOT}/{m}_scored_asr.jsonl", encoding="utf-8"))}
rows = [dict(r, l=asr[r["id"]]) for r in rows if asr.get(r["id"])]


def seg(r):
    s = lambda x: re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*?)(\d)$", x).groups()[:2]
    return "tone" if s(r["gold"]) == s(r["alt"]) else "seg"


def score(rs, g, a):
    mid = lambda r: (r["l"]["R_g"] + r["l"]["R_a"]) / 2
    return np.mean([r["l"][g] > mid(r) for r in rs]), np.mean([r["l"][a] < mid(r) for r in rs])


def bal_ci(rs, g, a, n=4000):
    by = {}
    for r in rs:
        by.setdefault(r["group"], []).append(r)
    ks = list(by)
    rng = np.random.default_rng(0)
    v = [np.mean(score([x for j in rng.integers(0, len(ks), len(ks)) for x in by[ks[j]]], g, a)) for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


print(f"{m}: {len(rows)} items, {len({r['group'] for r in rows})} groups")
for label, sub in (("all", rows), ("segmental", [r for r in rows if seg(r) == "seg"]), ("tone only", [r for r in rows if seg(r) == "tone"])):
    for name, g, a in (("vector port V", "V_g", "V_a"), ("token swap C", "C_g", "C_a")):
        s, t = score(sub, g, a)
        lo, hi = bal_ci(sub, g, a)
        print(f"  {label:10s} {name:14s} n {len(sub):3d} says-g {s:.3f} says-a {t:.3f} balanced {(s + t) / 2:.3f} [{lo:.3f}, {hi:.3f}]")
s, t = score(rows, "V_g", "V_a")
print("criterion (balanced >= .90, both >= .85):", "port works" if (s + t) / 2 >= .9 and min(s, t) >= .85 else "port fails")
