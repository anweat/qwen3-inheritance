# Exploratory summary of poly_prompt_think.py, same items and comparisons as poly_prompt_summary.py (no preset criteria).
#   python poly_prompt_think_summary.py <tts model> [Qwen3-4B]
import json, os, sys
import numpy as np

m = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-4B"
sys.argv = sys.argv[:1]
from poly_summary import load

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
heads = json.load(open(f"{ROOT}/out/heads_gate.json", encoding="utf-8"))
nothink = json.load(open(f"{ROOT}/out/prompt_{name}.json", encoding="utf-8"))["pred"]
th = {x["key"]: x for x in map(json.loads, open(f"{ROOT}/out/prompt_think_{name}.jsonl", encoding="utf-8"))}


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


rows = []
for r in load(f"{ROOT}/tts_out/{m}_nonstream.jsonl"):
    if "asr_centred" in r["scores"]["O"]:
        i = next(j for j, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)
        rows.append(dict(r, key=f"{r['text']}|{i}", tts=r["scores"]["O"]["asr_centred"] > 0))
own, probe = heads["TTS-0.6B@L6"]["pred"], heads["LLM-0.6B@L8"]["pred"]
print(f"==== {m}: {name} thinking mode, exploratory ====")
print(f" generations {len(th)}, finished thinking {np.mean([x['finished'] for x in th.values()]):.3f}, no candidate found {np.mean([x['pred'] is None for x in th.values()]):.3f}")
fs = {"think": lambda r: th[r["key"]]["pred"] == r["gold"], "no-think": lambda r: nothink[r["key"]][0] == r["gold"],
      "TTS": lambda r: r["tts"], "TTS-L6 head": lambda r: own[r["key"]][0] == r["gold"], "LLM-L8 head": lambda r: probe[r["key"]][0] == r["gold"]}
for kind, label in (("min", "CPP minority"), ("maj", "CPP majority")):
    s = [r for r in rows if r["kind"] == kind and r["key"] in th and r["key"] in own and r["key"] in nothink]
    print(f" {label}: n {len(s)}, groups {len({r['group'] for r in s})}, think {np.mean([fs['think'](r) for r in s]):.3f}")
    for b in ("no-think", "TTS", "TTS-L6 head", "LLM-L8 head"):
        lo, hi = boot(s, fs["think"], fs[b])
        print(f"   minus {b:12s} ({np.mean([fs[b](r) for r in s]):.3f}) {np.mean([fs['think'](r) for r in s]) - np.mean([fs[b](r) for r in s]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
