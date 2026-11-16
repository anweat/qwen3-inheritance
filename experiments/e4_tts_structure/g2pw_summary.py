# Applies the criteria in the g2pw_baseline.py header.
#   python g2pw_summary.py <tts model> [g2pw names...]
import json, os, sys
import numpy as np

m = sys.argv[1]
names = sys.argv[2:] or ["cpp", "release"]
sys.argv = sys.argv[:1]
from poly_summary import load

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
heads = json.load(open(f"{ROOT}/out/heads_gate.json", encoding="utf-8"))
items = {x["key"]: x for x in json.load(open(f"{ROOT}/out/g2p_items.json", encoding="utf-8")) if x["part"] == "eval"}
G = {n: json.load(open(f"{ROOT}/out/g2pw_{n}.json", encoding="utf-8")) for n in names if os.path.exists(f"{ROOT}/out/g2pw_{n}.json")}


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


def dk(r):
    return next(j for j, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)


def natural(path):
    asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(path[:-len(".jsonl")] + "_asr.jsonl", encoding="utf-8"))}
    out = []
    for r in map(json.loads, open(path, encoding="utf-8")):
        l = asr.get(r["id"])
        if l:
            out.append(dict(r, group=r["char"], key=f"{r['text']}|{dk(r)}", tts=l["O"] > (l["R_g"] + l["R_a"]) / 2))
    return out


hard = [dict(r, key=f"{r['text']}|{dk(r)}", tts=r["scores"]["O"]["asr_centred"] > 0)
        for r in load(f"{ROOT}/tts_out/{m}_nonstream.jsonl") if "asr_centred" in r["scores"]["O"]]
sets = {"CPP minority": [r for r in hard if r["kind"] == "min"], "CPP majority": [r for r in hard if r["kind"] == "maj"]}
for label, src in (("natural speech sentences", "tts_natural"), ("natural2", "tts_natural2")):
    p = f"{ROOT}/{src}/{m}_nonstream.jsonl"
    if os.path.exists(p[:-6] + "_asr.jsonl"):
        sets[label] = natural(p)

own, probe = heads["TTS-0.6B@L6"]["pred"], heads["LLM-0.6B@L8"]["pred"]


def g2p_ok(pred, r):
    v = pred.get(r["key"])
    return v == r["gold"] and v in items[r["key"]]["cands"]


print(f"==== {m}: g2pW baseline, same items and candidates as the stage 4 heads ====")
verdict = {}
for n, pred in G.items():
    ev = [items[k] for k in items if k in pred]
    outside = np.mean([pred[x["key"]] not in x["cands"] for x in ev])
    print(f" g2pW-{n}: {len(ev)} items, reading outside the candidate classes {outside:.3f}, accuracy on all eval items {np.mean([pred[x['key']] == x['gold'] for x in ev]):.3f}")
    f = lambda r, pred=pred: g2p_ok(pred, r)
    for label, rows in sets.items():
        s = [r for r in rows if r["key"] in pred and r["key"] in own and r["key"] in probe]
        if not s:
            continue
        a = np.mean([f(r) for r in s])
        cs = {}
        line = f"   {label:24s} n {len(s):3d} chars {len({r['group'] for r in s}):3d} g2pW {a:.3f}"
        for b, fb in (("TTS", lambda r: r["tts"]), ("TTS-L6 head", lambda r: own[r["key"]][0] == r["gold"]),
                      ("LLM-L8 head", lambda r: probe[r["key"]][0] == r["gold"])):
            lo, hi = boot(s, f, fb)
            cs[b] = (lo, hi)
            line += f" | minus {b} ({np.mean([fb(r) for r in s]):.3f}) {a - np.mean([fb(r) for r in s]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(line)
        verdict[(n, label)] = cs

    p = f"{ROOT}/tts_vector/{m}_scored_asr.jsonl"
    if os.path.exists(p):
        asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(p, encoding="utf-8"))}

        def e2e(r, guess):
            l = asr[r["id"]]
            mid = (l["R_g"] + l["R_a"]) / 2
            if guess == r["gold"]:
                return l["V_g"] > mid
            if guess == r["alt"]:
                return l["V_a"] > mid
            return False
        for kind, label in (("min", "CPP minority"), ("maj", "CPP majority")):
            s = [r for r in hard if r["kind"] == kind and asr.get(r["id"]) and r["key"] in pred and r["key"] in probe]
            fg = lambda r: e2e(r, pred[r["key"]] if pred[r["key"]] in items[r["key"]]["cands"] else None)
            fl = lambda r: e2e(r, probe[r["key"]][0])
            lo, hi = boot(s, fg, lambda r: r["tts"])
            l2, h2 = boot(s, fl, fg)
            print(f"   end to end {label:13s} n {len(s):3d} TTS alone {np.mean([r['tts'] for r in s]):.3f} | g2pW+port {np.mean([fg(r) for r in s]):.3f}"
                  f" minus TTS {np.mean([fg(r) for r in s]) - np.mean([r['tts'] for r in s]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f" | LLM-L8+port {np.mean([fl(r) for r in s]):.3f} minus g2pW+port [{l2:+.3f}, {h2:+.3f}]")
            if kind == "min":
                verdict[(n, "e2e")] = (lo, hi)
    if (n, "CPP minority") in verdict:
        c = verdict[(n, "CPP minority")]
        better = lambda k: c[k][0] > 0
        print(f"   verdict: g2pW beats TTS-L6 head {better('TTS-L6 head')}, beats LLM-L8 head {better('LLM-L8 head')}, "
              f"LLM-L8 head beats g2pW {c['LLM-L8 head'][1] < 0}, end to end useful {verdict.get((n, 'e2e'), (0, 0))[0] > 0}, "
              f"safe on natural2 {verdict[(n, 'natural2')]['TTS'][0] > -.05 if (n, 'natural2') in verdict else 'n/a'}"
              + ("   (counts)" if n == "cpp" else "   (reference only)"))
