# Compares reading heads with the TTS on the same sentences, applying the criteria in poly_heads.py and poly_heads_natural.py.
#   python poly_plugin_summary.py <model>
# TTS rate: ASR detector on O, uncorrected. Head rate: prediction equals gold. End to end: the head picks V_g or V_a from
# poly_vector.py, and the same detector scores that audio; a prediction outside the two readings counts as wrong.
import json, os, sys
import numpy as np

sys_argv, sys.argv = sys.argv, sys.argv[:1]
from poly_summary import load

m = sys_argv[1]
ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
heads = json.load(open(f"{ROOT}/out/heads.json", encoding="utf-8"))


def key(r):
    i = next(j for j, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)
    return f"{r['text']}|{i}"


def boot_diff(rows, a, b, n=4000):
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


rows = [r for r in load(f"{ROOT}/tts_out/{m}_nonstream.jsonl") if "asr_centred" in r["scores"]["O"]]
for r in rows:
    r["key"] = key(r)
    r["tts"] = r["scores"]["O"]["asr_centred"] > 0
names = [k for k in heads if k not in ("majority",)]
print(f"==== {m}: head accuracy against the TTS's reading rate, same items ====")
for kind, label in (("min", "CPP minority"), ("maj", "CPP majority")):
    sub = [r for r in rows if r["kind"] == kind and r["key"] in heads["LLM-0.6B@L28"]]
    print(f" {label}: n {len(sub)}, groups {len({r['group'] for r in sub})}, TTS rate {np.mean([r['tts'] for r in sub]):.3f}")
    for h in names:
        ok = lambda r, h=h: heads[h].get(r["key"]) == r["gold"]
        lo, hi = boot_diff(sub, ok, lambda r: r["tts"])
        print(f"   {h:16s} {np.mean([ok(r) for r in sub]):.3f}  minus TTS {np.mean([ok(r) for r in sub]) - np.mean([r['tts'] for r in sub]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    own = "TTS-0.6B@L6"
    for h in [n for n in names if n.startswith("LLM")]:
        a = lambda r, h=h: heads[h].get(r["key"]) == r["gold"]
        b = lambda r: heads[own].get(r["key"]) == r["gold"]
        lo, hi = boot_diff(sub, a, b)
        print(f"   inheritance {h:14s} minus {own}: {np.mean([a(r) for r in sub]) - np.mean([b(r) for r in sub]):+.3f} [{lo:+.3f}, {hi:+.3f}]")

p = f"{ROOT}/tts_vector/{m}_scored_asr.jsonl"
if os.path.exists(p):
    asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(p, encoding="utf-8"))}
    sub = [r for r in rows if asr.get(r["id"]) and r["key"] in heads["LLM-0.6B@L28"]]
    print(f"==== {m}: end to end with the reading-vector port, audio actually synthesized ====")

    def e2e(r, h):
        pred = heads[h].get(r["key"])
        l = asr[r["id"]]
        mid = (l["R_g"] + l["R_a"]) / 2
        if pred == r["gold"]:
            return l["V_g"] > mid
        if pred == r["alt"]:
            return l["V_a"] > mid
        return False
    for kind, label in (("min", "CPP minority"), ("maj", "CPP majority")):
        s = [r for r in sub if r["kind"] == kind]
        print(f" {label}: n {len(s)}, groups {len({r['group'] for r in s})}, TTS alone {np.mean([r['tts'] for r in s]):.3f}")
        for h in names:
            f = lambda r, h=h: e2e(r, h)
            lo, hi = boot_diff(s, f, lambda r: r["tts"])
            print(f"   {h:16s} {np.mean([f(r) for r in s]):.3f}  minus TTS {np.mean([f(r) for r in s]) - np.mean([r['tts'] for r in s]):+.3f} [{lo:+.3f}, {hi:+.3f}]")

p = f"{ROOT}/out/heads_natural.json"
q = f"{ROOT}/tts_natural/{m}_nonstream.jsonl"
if os.path.exists(p) and os.path.exists(q[:-6] + "_asr.jsonl"):
    nat = json.load(open(p, encoding="utf-8"))
    asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(q[:-6] + "_asr.jsonl", encoding="utf-8"))}
    n = [dict(r, group=r["char"], l=asr[r["id"]]) for r in map(json.loads, open(q, encoding="utf-8")) if asr.get(r["id"])]
    for r in n:
        r["tts"] = r["l"]["O"] > (r["l"]["R_g"] + r["l"]["R_a"]) / 2
    print(f"==== {m}: natural sentences, do no harm (lower bound above -.05) ====")
    print(f" TTS alone {np.mean([r['tts'] for r in n]):.3f}, n {len(n)}, characters {len({r['char'] for r in n})}")
    for h, pred in nat.items():
        s = [r for r in n if str(r["id"]) in pred]
        ok = lambda r, pred=pred: pred[str(r["id"])] == r["gold"]
        lo, hi = boot_diff(s, ok, lambda r: r["tts"])
        d = np.mean([ok(r) for r in s]) - np.mean([r["tts"] for r in s])
        print(f"   {h:16s} n {len(s)} head {np.mean([ok(r) for r in s]):.3f}  minus TTS {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {'safe' if lo > -.05 else 'not shown safe'}")
