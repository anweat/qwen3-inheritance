# E8 scoring, fixed 2026-11-11 before any answers exist (sampling and page: make_page.py).
#   python score.py answers/*.json
# Listeners: native Mandarin speakers, at least 3 included, otherwise no verdict.
#   Inclusion: at least 8 of the 10 catch clips answered with the known reading ("neither" counts as wrong), and at least
#   90% of all clips answered. Excluded listeners are reported, not silently dropped.
# Per clip, the human label is the strict majority of included listeners: gold, alt or neither. "Heard as gold" = gold
#   has a strict majority; ties and neither count as not gold (conservative). Sensitivity: drop clips without a majority.
# Per sentence d = heard_gold(P) - heard_gold(O). Character-group bootstrap 95% CI (4000 resamples, seed 0).
#   H3 (pipeline helps): CPP minority, lower bound > 0.
#   H5 (pipeline is safe): natural2, lower bound > -.05.
# Reported without a threshold: each listener alone; Fleiss' kappa over the three labels; agreement of the human label
#   with the stage 4 ASR detector per clip (gold vs not gold); answers per clip, plays, time; tone-only vs segmental items.
import glob, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = json.load(open(os.path.join(HERE, "key.json"), encoding="utf-8"))["items"]
NEITHER = "都不像 / 听不清"
K = {k["n"]: k for k in KEY}


def label(k, ans):
    if ans == NEITHER:
        return "neither"
    return "gold" if ans == k["opt_labels"][k["opts"].index(k["gold"])] else "alt" if ans == k["opt_labels"][k["opts"].index(k["alt"])] else "neither"


def seg(g, a):
    s = lambda x: re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*?)(\d)$", x).groups()[:2]
    return s(g) != s(a)


def boot(rows, n=4000):
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r["d"])
    ks = list(by)
    rng = np.random.default_rng(0)
    v = [np.mean([x for j in rng.integers(0, len(ks), len(ks)) for x in by[ks[j]]]) for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def fleiss(tab):
    tab = np.array([t for t in tab if sum(t) >= 2], float)
    if len(tab) == 0:
        return float("nan")
    n = tab.sum(1)
    p = tab.sum(0) / tab.sum()
    P = ((tab * (tab - 1)).sum(1) / (n * (n - 1))).mean()
    Pe = (p ** 2).sum()
    return (P - Pe) / (1 - Pe)


files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "answers", "*.json")))
lis = {}
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    lis[d["listener"]] = {a["n"]: label(K[a["n"]], a["answer"]) for a in d["answers"]}
print(f"==== E8 listening test: {len(lis)} listener files ====")
catch = [k for k in KEY if k["part"] == "catch"]
ok = {}
for w, a in lis.items():
    right = sum(a.get(k["n"]) == ("gold" if k["version"] == "R_g" else "alt") for k in catch)
    cover = len(a) / len(KEY)
    ok[w] = right >= 8 and cover >= .9
    print(f"  {w:16s} catch {right}/{len(catch)}  answered {len(a)}/{len(KEY)}  neither {sum(v == 'neither' for v in a.values())}  -> {'included' if ok[w] else 'EXCLUDED'}")
inc = [w for w in lis if ok[w]]


def majority(n, who):
    c = Counter(lis[w][n] for w in who if n in lis[w])
    if not c:
        return None
    top, m = c.most_common(1)[0]
    return top if m * 2 > sum(c.values()) else "tie"


def analyse(who, tag):
    out = {}
    for part, name, rule in (("min", "H3 CPP minority", lambda lo: lo > 0), ("nat2", "H5 natural2", lambda lo: lo > -.05)):
        sents = defaultdict(dict)
        for k in KEY:
            if k["part"] == part:
                sents[(k["set"], k["id"])][k["version"]] = k
        rows, rows_strict = [], []
        for (s, i), v in sents.items():
            mo, mp = majority(v["O"]["n"], who), majority(v["P"]["n"], who)
            if mo is None or mp is None:
                continue
            r = dict(group=v["O"]["group"], o=mo == "gold", p=mp == "gold", seg=seg(v["O"]["gold"], v["O"]["alt"]))
            r["d"] = r["p"] - r["o"]
            rows.append(r)
            if mo in ("gold", "alt") and mp in ("gold", "alt"):
                rows_strict.append(r)
        if not rows:
            continue
        lo, hi = boot(rows)
        out[part] = rule(lo)
        ls, hs = boot(rows_strict) if rows_strict else (float("nan"),) * 2
        print(f"  [{tag}] {name:16s} n {len(rows):2d} chars {len({r['group'] for r in rows}):2d}  O {np.mean([r['o'] for r in rows]):.3f}"
              f"  P {np.mean([r['p'] for r in rows]):.3f}  d {np.mean([r['d'] for r in rows]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"  | clear-majority only n {len(rows_strict)} d {np.mean([r['d'] for r in rows_strict]) if rows_strict else float('nan'):+.3f} [{ls:+.3f}, {hs:+.3f}]")
        for t, sel in (("segmental", True), ("tone only", False)):
            rs = [r for r in rows if r["seg"] == sel]
            if rs:
                print(f"      {t:9s} n {len(rs):2d}  O {np.mean([r['o'] for r in rs]):.3f}  P {np.mean([r['p'] for r in rs]):.3f}")
    return out


if len(inc) < 3:
    print(f"VERDICT: no verdict, only {len(inc)} included listeners (need 3)")
    for w in inc:
        analyse([w], w)
else:
    v = analyse(inc, "panel")
    for w in inc:
        analyse([w], w)
    lab = ("gold", "alt", "neither")
    tab = [[sum(lis[w].get(k["n"]) == l for w in inc) for l in lab] for k in KEY]
    print(f"  Fleiss kappa over {len(inc)} listeners: {fleiss(tab):.3f}")
    agree = [(majority(k["n"], inc) == "gold") == k["detector_gold"] for k in KEY if k["detector_gold"] is not None and k["part"] != "catch"
             and majority(k["n"], inc) in ("gold", "alt")]
    print(f"  human majority vs ASR detector (clips with a clear human gold/alt majority): {np.mean(agree):.3f} (n {len(agree)})")
    print(f"VERDICT: H3 pipeline helps on CPP minority: {v.get('min')};  H5 pipeline safe on natural2: {v.get('nat2')}")
