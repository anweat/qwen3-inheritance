# Exploratory: forced-language results next to automatic detection, on the same clips.
import json, os
from collections import defaultdict
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
def load(f):
    by = defaultdict(dict)
    for r in map(json.loads, open(os.path.join(H, f), encoding="utf-8")):
        by[r["scheme"]][r["path"]] = r
    return by
A, F = load("per_clip.jsonl"), load("per_clip_forced.jsonl")
S = ["intact", "standalone_q4", "base_q4_plus_2bit", "instruct_q4_plus_2bit"]
paths = list(F["intact"])
ZH = {"aishell", "fleurs_cmn_hans_cn", "fleurs_yue_hant_hk", "cantonese_md", "sichuan", "northeast", "henan", "wu"}
order = ["aishell", "fleurs_cmn_hans_cn", "fleurs_yue_hant_hk", "cantonese_md", "sichuan", "northeast", "henan", "wu", "fleurs_en_us", "fleurs_ko_kr", "fleurs_ja_jp", "fleurs_vi_vn", "fleurs_th_th", "fleurs_ar_eg", "fleurs_ru_ru"]
def rate(by, s, ps):
    return 100 * sum(by[s][p]["err"] for p in ps) / max(1, sum(by[s][p]["n"] for p in ps))
print("forced language, error rate (%)   [automatic detection in brackets]")
print(f"{'set':20s}{'n':>4s} " + " ".join(f"{s[:21]:>21s}" for s in S))
for st in order + ["ALL", "Chinese+dialect", "other"]:
    ps = paths if st == "ALL" else [p for p in paths if (F['intact'][p]['set'] in ZH if st == "Chinese+dialect" else F['intact'][p]['set'] not in ZH if st == "other" else F['intact'][p]['set'] == st)]
    print(f"{st:20s}{len(ps):4d} " + " ".join(f"{rate(F, s, ps):10.2f} [{rate(A, s, ps):7.2f}] " for s in S))
rng = np.random.default_rng(0)
def boot(by, a, b, ps, n=4000):
    g = defaultdict(list)
    for p in ps:
        g[by['intact'][p]['set']].append(p)
    arr = [np.array([[by[a][p]['err'], by[b][p]['err'], by[a][p]['n']] for p in v], float) for v in g.values()]
    out = []
    for _ in range(n):
        x = y = t = 0
        for m in arr:
            i = rng.integers(0, len(m), len(m))
            x += m[i, 0].sum(); y += m[i, 1].sum(); t += m[i, 2].sum()
        out.append(100 * (x - y) / t)
    return np.percentile(out, [2.5, 97.5])
print("\nforced language, paired difference with 95% CI (percentage points)")
for label, ps in (("all", paths), ("Chinese+dialect", [p for p in paths if F['intact'][p]['set'] in ZH]), ("other", [p for p in paths if F['intact'][p]['set'] not in ZH])):
    print(f"  {label} n={len(ps)}")
    for s in S[1:]:
        lo, hi = boot(F, s, "intact", ps)
        print(f"    {s:26s} vs forced intact     {rate(F,s,ps)-rate(F,'intact',ps):+6.2f} [{lo:+.2f}, {hi:+.2f}]")
    for s in S[2:]:
        lo, hi = boot(F, s, "standalone_q4", ps)
        print(f"    {s:26s} vs forced Q4         {rate(F,s,ps)-rate(F,'standalone_q4',ps):+6.2f} [{lo:+.2f}, {hi:+.2f}]")
print("\nworst per-set difference vs forced Q4:")
for s in S[2:]:
    d = {st: rate(F, s, [p for p in paths if F['intact'][p]['set'] == st]) - rate(F, 'standalone_q4', [p for p in paths if F['intact'][p]['set'] == st]) for st in order}
    w = sorted(d, key=d.get, reverse=True)[:3]
    print(f"  {s:26s} " + " ".join(f"{k}:{d[k]:+.1f}" for k in w))
print("\nforcing language vs automatic on the intact model:")
lo, hi = boot({'intact': F['intact'], 'x': A['intact']}, 'intact', 'x', paths)
print(f"  {rate(F,'intact',paths)-rate(A,'intact',paths):+.2f} [{lo:+.2f}, {hi:+.2f}]")
print("\nempty transcripts:", {s: sum(not F[s][p]['hyp'].strip() for p in paths) for s in S})
print("exact match with forced intact:", {s: round(np.mean([F[s][p]['hyp'] == F['intact'][p]['hyp'] for p in paths]), 3) for s in S[1:]})
