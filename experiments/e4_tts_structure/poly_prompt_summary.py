# Applies the criteria in the poly_prompt.py header.
#   python poly_prompt_summary.py <tts model> [prompt files...]
# Compares zero-shot prompted Qwen3 readings with the TTS's actual reading (detector on O) and with the stage 4 heads in
# heads_gate.json (same candidates, same items).
import json, os, re, sys
import numpy as np

m = sys.argv[1]
files = sys.argv[2:] or ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
sys.argv = sys.argv[:1]
from poly_summary import load

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
heads = json.load(open(f"{ROOT}/out/heads_gate.json", encoding="utf-8"))
P = {}
for f in files:
    p = f"{ROOT}/out/prompt_{f}.json"
    if os.path.exists(p):
        P[f] = json.load(open(p, encoding="utf-8"))


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


def diff_key(r):
    return next(j for j, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)


def natural(path):
    asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(path[:-len(".jsonl")] + "_asr.jsonl", encoding="utf-8"))}
    out = []
    for r in map(json.loads, open(path, encoding="utf-8")):
        l = asr.get(r["id"])
        if l:
            out.append(dict(r, group=r["char"], key=f"{r['text']}|{diff_key(r)}", tts=l["O"] > (l["R_g"] + l["R_a"]) / 2))
    return out


def contrast(r):
    s = lambda x: re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*?)(\d)$", x).groups()[:2]
    return "tone" if s(r["gold"]) == s(r["alt"]) else "seg"


hard = [dict(r, key=f"{r['text']}|{diff_key(r)}", tts=r["scores"]["O"]["asr_centred"] > 0)
        for r in load(f"{ROOT}/tts_out/{m}_nonstream.jsonl") if "asr_centred" in r["scores"]["O"]]
sets = {"CPP minority": [r for r in hard if r["kind"] == "min"], "CPP majority": [r for r in hard if r["kind"] == "maj"]}
p = f"{ROOT}/tts_natural/{m}_nonstream.jsonl"
if os.path.exists(p[:-6] + "_asr.jsonl"):
    sets["natural speech sentences"] = natural(p)
p = f"{ROOT}/tts_natural2/{m}_nonstream.jsonl"
if os.path.exists(p[:-6] + "_asr.jsonl"):
    sets["natural2"] = natural(p)
    sets["natural2, segmental only"] = [r for r in sets["natural2"] if contrast(r) == "seg"]
    sets["natural2, tone only"] = [r for r in sets["natural2"] if contrast(r) == "tone"]

own, probe = heads["TTS-0.6B@L6"]["pred"], heads["LLM-0.6B@L8"]["pred"]
print(f"==== {m}: zero-shot prompted Qwen3, same candidates and items as the stage 4 heads ====")
verdict = {}
for name, h in P.items():
    pr, tau = h["pred"], h["tau"]
    dev = list(h["dev"].values())
    print(f" {name}: dev accuracy {np.mean([v[0] == v[2] for v in dev]):.3f} (n {len(dev)}), margin threshold {tau:.3f}")
    plain = lambda r, pr=pr: pr[r["key"]][0] == r["gold"]
    gated = lambda r, pr=pr, tau=tau: (pr[r["key"]][0] == r["gold"]) if pr[r["key"]][1] >= tau else r["tts"]
    tts = lambda r: r["tts"]
    hown = lambda r: own[r["key"]][0] == r["gold"]
    hprobe = lambda r: probe[r["key"]][0] == r["gold"]
    for label, rows in sets.items():
        s = [r for r in rows if r["key"] in pr and r["key"] in own and r["key"] in probe]
        if not s:
            continue
        u, t, o, q, g = (np.mean([f(r) for r in s]) for f in (plain, tts, hown, hprobe, gated))
        c = [boot(s, plain, tts), boot(s, plain, hown), boot(s, plain, hprobe), boot(s, gated, tts)]
        cov = np.mean([pr[r["key"]][1] >= tau for r in s])
        print(f"   {label:26s} n {len(s):3d} chars {len({r['group'] for r in s}):3d} prompt {u:.3f} | minus TTS ({t:.3f}) {u - t:+.3f} [{c[0][0]:+.3f}, {c[0][1]:+.3f}]"
              f" | minus TTS-L6 head ({o:.3f}) {u - o:+.3f} [{c[1][0]:+.3f}, {c[1][1]:+.3f}] | minus LLM-L8 head ({q:.3f}) {u - q:+.3f} [{c[2][0]:+.3f}, {c[2][1]:+.3f}]"
              f" | gated {g:.3f} {g - t:+.3f} [{c[3][0]:+.3f}, {c[3][1]:+.3f}] cov {cov:.2f}")
        verdict[(name, label)] = c
    if (name, "CPP minority") in verdict:
        c = verdict[(name, "CPP minority")]
        line = f"   verdict: useful {c[0][0] > 0}, knowledge beyond TTS {c[1][0] > 0}, beyond the probe {c[2][0] > 0}"
        if (name, "natural2") in verdict:
            line += f", gated safe on natural2 {verdict[(name, 'natural2')][3][0] > -.05}"
        print(line + ("   (primary)" if name == "Qwen3-4B" else "   (scaling curve, not tested)"))

# Exploratory: positions where the heads cannot answer because the character has no labelled CPP training data.
print("\n exploratory: natural positions whose character has no head (excluded above), prompt accuracy there")
for name, h in P.items():
    for label in ("natural speech sentences", "natural2"):
        s = [r for r in sets.get(label, []) if r["key"] in h["pred"] and r["key"] not in own]
        if s:
            print(f"   {name} {label}: n {len(s)}, prompt {np.mean([h['pred'][r['key']][0] == r['gold'] for r in s]):.3f}, TTS {np.mean([r['tts'] for r in s]):.3f}")
