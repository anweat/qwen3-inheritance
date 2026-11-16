# Paired comparison against standalone Q4, per set and pooled, with a stratified clip bootstrap.
import json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "per_clip.jsonl"), encoding="utf-8")]
by = defaultdict(dict)
for r in rows:
    by[r["scheme"]][r["path"]] = r
REF = "standalone_q4"
schemes = [s for s in ["intact", "standalone_q4", "base_q4_plus_2bit", "instruct_q4_plus_2bit", "instruct_q4_plus_2bit_with_embedding"] if s in by]
sets = sorted({r["set"] for r in rows})
order = [s for s in ["aishell", "fleurs_cmn_hans_cn", "fleurs_yue_hant_hk", "cantonese_md", "sichuan", "northeast", "henan", "wu", "fleurs_en_us", "fleurs_ko_kr", "fleurs_ja_jp", "fleurs_vi_vn", "fleurs_th_th", "fleurs_ar_eg", "fleurs_ru_ru"] if s in sets]
rng = np.random.default_rng(0)


def rate(items):
    return sum(i["err"] for i in items) / max(1, sum(i["n"] for i in items))


print("error rate by set (%), WER or CER; clips")
print(f"{'set':20s}{'n':>4s} " + " ".join(f"{s[:22]:>22s}" for s in schemes))
for st in order:
    paths = [p for p, r in by[schemes[0]].items() if r["set"] == st]
    print(f"{st:20s}{len(paths):4d} " + " ".join(f"{100*rate([by[s][p] for p in paths if p in by[s]]):22.2f}" for s in schemes))
all_paths = list(by[schemes[0]])
print(f"{'POOLED':20s}{len(all_paths):4d} " + " ".join(f"{100*rate([by[s][p] for p in all_paths if p in by[s]]):22.2f}" for s in schemes))


def boot(a, b, paths, n=4000):
    groups = defaultdict(list)
    for p in paths:
        groups[by[a][p]["set"]].append(p)
    arrs = {g: np.array([[by[a][p]["err"], by[b][p]["err"], by[a][p]["n"]] for p in ps], dtype=float) for g, ps in groups.items()}
    out = []
    for _ in range(n):
        ea, eb, nn = 0, 0, 0
        for g, x in arrs.items():
            i = rng.integers(0, len(x), len(x))
            ea += x[i, 0].sum(); eb += x[i, 1].sum(); nn += x[i, 2].sum()
        out.append(100 * (ea - eb) / nn)
    return np.percentile(out, [2.5, 97.5])


print(f"\npaired difference vs {REF}, percentage points (positive = worse than {REF}); 95% CI by clip")
for s in schemes:
    if s == REF or len(by[s]) < len(all_paths):
        continue
    d = 100 * (rate([by[s][p] for p in all_paths]) - rate([by[REF][p] for p in all_paths]))
    lo, hi = boot(s, REF, all_paths)
    per = {}
    for st in order:
        ps = [p for p in all_paths if by[s][p]["set"] == st]
        per[st] = 100 * (rate([by[s][p] for p in ps]) - rate([by[REF][p] for p in ps]))
    worst = max(per, key=per.get)
    verdict = "PASS" if hi <= 0.5 and per[worst] <= 2 else "FAIL"
    print(f"  {s:38s} pooled {d:+6.2f} [{lo:+.2f}, {hi:+.2f}]  worst set {worst} {per[worst]:+.2f}  -> {verdict}")
    print("    " + " ".join(f"{k.replace('fleurs_','')[:9]}:{v:+.1f}" for k, v in per.items()))
print("\ndifference vs intact, for scale:")
for s in schemes:
    if s == "intact" or len(by[s]) < len(all_paths):
        continue
    lo, hi = boot(s, "intact", all_paths)
    print(f"  {s:38s} {100*(rate([by[s][p] for p in all_paths])-rate([by['intact'][p] for p in all_paths])):+6.2f} [{lo:+.2f}, {hi:+.2f}]")
print("\nhypotheses that exactly match the intact output:")
for s in schemes:
    if s != "intact":
        same = [by[s][p]["hyp"] == by["intact"][p]["hyp"] for p in by[s] if p in by["intact"]]
        print(f"  {s:38s} {np.mean(same):.3f}")
