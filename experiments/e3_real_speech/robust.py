# Exploratory, not part of the pass criterion: how much do runaway outputs and language misdetection explain?
import json, os, re
from collections import defaultdict, Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
by = defaultdict(dict)
for r in map(json.loads, open(os.path.join(HERE, "per_clip.jsonl"), encoding="utf-8")):
    by[r["scheme"]][r["path"]] = r
S = ["intact", "standalone_q4", "base_q4_plus_2bit", "instruct_q4_plus_2bit", "instruct_q4_plus_2bit_with_embedding"]
paths = list(by["intact"])
sets = sorted({by["intact"][p]["set"] for p in paths})
def runaway(r):
    return r["err"] > 1.5 * max(1, r["n"])
def lang(r):
    m = re.match(r"language (\w+)", r["raw"])
    return m.group(1) if m else "none"
print("share of clips by type (runaway: errors > 1.5x reference length; empty: no transcript)")
for s in S:
    rr = [by[s][p] for p in paths]
    print(f"  {s:38s} runaway {np.mean([runaway(r) for r in rr]):.3f} ({sum(runaway(r) for r in rr)})  empty {np.mean([not r['hyp'].strip() for r in rr]):.3f}")
print("\nper-set counts of runaway clips")
print(f"  {'set':20s}" + " ".join(f"{s[:13]:>13s}" for s in S))
for st in sets:
    ps = [p for p in paths if by["intact"][p]["set"] == st]
    print(f"  {st:20s}" + " ".join(f"{sum(runaway(by[s][p]) for p in ps):13d}" for s in S))
print("\nerror rate (%) over clips where NO scheme ran away, same clips for all")
keep = [p for p in paths if not any(runaway(by[s][p]) for s in S)]
print(f"  kept {len(keep)} of {len(paths)}")
def rate(s, ps):
    return 100 * sum(by[s][p]["err"] for p in ps) / max(1, sum(by[s][p]["n"] for p in ps))
print(f"  {'set':20s}{'n':>4s}" + " ".join(f"{s[:13]:>13s}" for s in S))
for st in sets:
    ps = [p for p in keep if by["intact"][p]["set"] == st]
    print(f"  {st:20s}{len(ps):4d}" + " ".join(f"{rate(s, ps):13.2f}" for s in S))
print(f"  {'POOLED':20s}{len(keep):4d}" + " ".join(f"{rate(s, keep):13.2f}" for s in S))
print("\nmedian per-clip error rate (%)")
for s in S:
    print(f"  {s:38s} {100*np.median([by[s][p]['err']/max(1,by[s][p]['n']) for p in paths]):.2f}")
print("\ndetected language differs from intact (share)")
for s in S[1:]:
    print(f"  {s:38s} {np.mean([lang(by[s][p]) != lang(by['intact'][p]) for p in paths]):.3f}")
print("  intact language by set:", {st: Counter(lang(by['intact'][p]) for p in paths if by['intact'][p]['set']==st).most_common(2) for st in sets if st in ('wu','fleurs_yue_hant_hk','fleurs_ru_ru','cantonese_md')})
print("\nexamples of worst clips, cmn under standalone Q4 and ru under Base+2bit:")
for s, st in (("standalone_q4", "fleurs_cmn_hans_cn"), ("base_q4_plus_2bit", "fleurs_ru_ru")):
    w = sorted((p for p in paths if by["intact"][p]["set"] == st), key=lambda p: -by[s][p]["err"])[:2]
    for p in w:
        print(f"  [{s}] err {by[s][p]['err']} n {by[s][p]['n']} (intact err {by['intact'][p]['err']}) hyp: {by[s][p]['hyp'][:90]!r}")
# Paired bootstrap on the kept clips, stratified by set.
rng = np.random.default_rng(0)
print("\npaired difference vs standalone_q4 on kept clips, percentage points, 95% CI (exploratory)")
g = defaultdict(list)
for p in keep:
    g[by["intact"][p]["set"]].append(p)
for s in S[2:]:
    arr = {k: np.array([[by[s][p]["err"], by["standalone_q4"][p]["err"], by[s][p]["n"]] for p in v], float) for k, v in g.items()}
    out = []
    for _ in range(4000):
        a = b = n = 0
        for x in arr.values():
            i = rng.integers(0, len(x), len(x))
            a += x[i, 0].sum(); b += x[i, 1].sum(); n += x[i, 2].sum()
        out.append(100 * (a - b) / n)
    print(f"  {s:38s} {rate(s, keep)-rate('standalone_q4', keep):+.2f} [{np.percentile(out,2.5):+.2f}, {np.percentile(out,97.5):+.2f}]")
