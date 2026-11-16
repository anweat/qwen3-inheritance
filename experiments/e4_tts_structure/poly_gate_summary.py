# Applies the criteria in the poly_gate.py header.
#   python poly_gate_summary.py <model>
import json, os, re, sys
import numpy as np

m = sys.argv[1]
sys.argv = sys.argv[:1]
from poly_summary import load

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
heads = json.load(open(f"{ROOT}/out/heads_gate.json", encoding="utf-8"))


def boot(rows, a, b, n=4000):
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    ks = list(by)
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n):
        s = [x for j in rng.integers(0, len(ks), len(ks)) for x in by[ks[j]]]
        v.append(np.mean([a(x) for x in s]) - np.mean([b(x) for x in s]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def natural(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(path[:-len(".jsonl")] + "_asr.jsonl", encoding="utf-8"))}
    out = []
    for r in rows:
        l = asr.get(r["id"])
        if l:
            i = next(j for j, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)
            out.append(dict(r, group=r["char"], key=f"{r['text']}|{i}", tts=l["O"] > (l["R_g"] + l["R_a"]) / 2, gap=l["R_g"] - l["R_a"]))
    return out


def stage4():
    out = []
    for r in load(f"{ROOT}/tts_out/{m}_nonstream.jsonl"):
        if "asr_centred" in r["scores"]["O"]:
            i = next(j for j, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)
            out.append(dict(r, key=f"{r['text']}|{i}", tts=r["scores"]["O"]["asr_centred"] > 0))
    return out


def contrast(r):
    s = lambda x: re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*?)(\d)$", x).groups()[:2]
    return "tone" if s(r["gold"]) == s(r["alt"]) else "seg"


hard = stage4()
sets = {"CPP minority": [r for r in hard if r["kind"] == "min"], "CPP majority": [r for r in hard if r["kind"] == "maj"]}
p = f"{ROOT}/tts_natural/{m}_nonstream.jsonl"
if os.path.exists(p):
    sets["natural speech sentences"] = natural(p)
p = f"{ROOT}/tts_natural2/{m}_nonstream.jsonl"
if os.path.exists(p[:-6] + "_asr.jsonl"):
    n2 = natural(p)
    sets["natural2"] = n2
    sets["natural2, segmental only"] = [r for r in n2 if contrast(r) == "seg"]
    sets["natural2, tone only"] = [r for r in n2 if contrast(r) == "tone"]
print(f"==== {m} ====")
verdict = {}
for name, h in heads.items():
    pr, tau = h["pred"], h["tau"]
    print(f" {name}: margin threshold {tau:.4f}")
    plain = lambda r: pr[r["key"]][0] == r["gold"]
    gated = lambda r: (pr[r["key"]][0] == r["gold"]) if pr[r["key"]][1] >= tau else r["tts"]
    for label, rows in sets.items():
        s = [r for r in rows if r["key"] in pr]
        if not s:
            continue
        cov = np.mean([pr[r["key"]][1] >= tau for r in s])
        tts = lambda r: r["tts"]
        lo, hi = boot(s, gated, tts)
        lp, hp = boot(s, plain, tts)
        g, u, t = np.mean([gated(r) for r in s]), np.mean([plain(r) for r in s]), np.mean([r["tts"] for r in s])
        print(f"   {label:26s} n {len(s):3d} chars {len({r['group'] for r in s}):3d} TTS {t:.3f} | gated {g:.3f} {g - t:+.3f} [{lo:+.3f}, {hi:+.3f}] coverage {cov:.2f} | ungated {u:.3f} {u - t:+.3f} [{lp:+.3f}, {hp:+.3f}]")
        verdict[(name, label)] = (lo, hi, lp, hp)
    if (name, "natural2") in verdict:
        safe = verdict[(name, "natural2")][0] > -.05
        useful = verdict[(name, "CPP minority")][0] > 0
        print(f"   gated: safe {safe}, useful {useful} -> {'deployable candidate' if safe and useful else 'not shown deployable'}")
        print(f"   ungated: safe {verdict[(name, 'natural2')][2] > -.05}, useful {verdict[(name, 'CPP minority')][2] > 0}")

# Exploratory, not in the criteria: does the LLM head beat the TTS's own head once both are gated to the same dev precision?
a, b = heads["LLM-0.6B@L8"], heads["TTS-0.6B@L6"]
fa = lambda r: (a["pred"][r["key"]][0] == r["gold"]) if a["pred"][r["key"]][1] >= a["tau"] else r["tts"]
fb = lambda r: (b["pred"][r["key"]][0] == r["gold"]) if b["pred"][r["key"]][1] >= b["tau"] else r["tts"]
print(" exploratory: gated LLM L8 minus gated TTS L6")
for label, rows in sets.items():
    s = [r for r in rows if r["key"] in a["pred"] and r["key"] in b["pred"]]
    if s:
        lo, hi = boot(s, fa, fb)
        print(f"   {label:26s} n {len(s):3d} {np.mean([fa(r) for r in s]) - np.mean([fb(r) for r in s]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
for name, h in heads.items():
    for label in ("CPP minority", "natural2"):
        s = [r for r in sets[label] if r["key"] in h["pred"] and h["pred"][r["key"]][1] >= h["tau"]]
        print(f"   {name} {label}: precision at gated positions {np.mean([h['pred'][r['key']][0] == r['gold'] for r in s]):.3f} on {len(s)}, TTS at those positions {np.mean([r['tts'] for r in s]):.3f}")
