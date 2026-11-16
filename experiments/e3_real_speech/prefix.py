# Exploratory: how much damage comes from clips where the output prefix fails, and how do Chinese sets compare?
import json, os
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
by = defaultdict(dict)
for r in map(json.loads, open(os.path.join(HERE, "per_clip.jsonl"), encoding="utf-8")):
    by[r["scheme"]][r["path"]] = r
S = ["intact", "standalone_q4", "base_q4_plus_2bit", "instruct_q4_plus_2bit"]
paths = list(by["intact"])
ZH = {"aishell", "fleurs_cmn_hans_cn", "fleurs_yue_hant_hk", "cantonese_md", "sichuan", "northeast", "henan", "wu"}
def fail(r):
    return "<asr_text>" not in r["raw"]
def wrong_lang(r, ref):
    return r["raw"].split("<asr_text>")[0] != ref["raw"].split("<asr_text>")[0]
def rate(s, ps):
    return 100 * sum(by[s][p]["err"] for p in ps) / max(1, sum(by[s][p]["n"] for p in ps))
print("clips with no <asr_text> marker, and clips with a different language header than intact")
for s in S:
    print(f"  {s:26s} no marker {sum(fail(by[s][p]) for p in paths):4d}  header differs {sum(wrong_lang(by[s][p], by['intact'][p]) for p in paths):4d}")
print("\nper set: no-marker clips")
print(f"  {'set':20s}" + " ".join(f"{s[:14]:>15s}" for s in S))
for st in sorted({by['intact'][p]['set'] for p in paths}):
    ps = [p for p in paths if by['intact'][p]['set'] == st]
    print(f"  {st:20s}" + " ".join(f"{sum(fail(by[s][p]) for p in ps):15d}" for s in S))
ok = [p for p in paths if not any(fail(by[s][p]) or wrong_lang(by[s][p], by['intact'][p]) for s in S)]
print(f"\nerror rate on clips where every scheme kept the intact header: {len(ok)} of {len(paths)}")
for name, sel in (("all", ok), ("Chinese+dialect", [p for p in ok if by['intact'][p]['set'] in ZH]), ("other languages", [p for p in ok if by['intact'][p]['set'] not in ZH])):
    print(f"  {name:18s} n={len(sel):3d} " + " ".join(f"{s[:14]} {rate(s, sel):6.2f}" for s in S))
rng = np.random.default_rng(0)
def boot(a, b, ps, n=4000):
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
print("\nall clips, Chinese+dialect subset (subset chosen after the criterion was set, so exploratory)")
zh = [p for p in paths if by['intact'][p]['set'] in ZH]
other = [p for p in paths if by['intact'][p]['set'] not in ZH]
for label, ps in (("Chinese+dialect", zh), ("other languages", other)):
    print(f"  {label} n={len(ps)}: " + " ".join(f"{s[:14]} {rate(s, ps):.2f}" for s in S))
    for s in S[2:]:
        lo, hi = boot(s, "standalone_q4", ps)
        print(f"    {s:26s} vs standalone Q4 {rate(s, ps)-rate('standalone_q4', ps):+6.2f} [{lo:+.2f}, {hi:+.2f}]")
    for s in S[1:]:
        lo, hi = boot(s, "intact", ps)
        print(f"    {s:26s} vs intact        {rate(s, ps)-rate('intact', ps):+6.2f} [{lo:+.2f}, {hi:+.2f}]")
