# Compares one listener's blind answers on 36 TTS-0.6B sentences with the detector that scored the same audio.
# Exploratory: the page was built to test the detector, but no pass mark was fixed in advance.
import json, os, re, sys
import numpy as np
from scipy.stats import beta

sys.argv = sys.argv[:1]
from poly_summary import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
key = json.load(open(os.path.join(HERE, "listening_key.json"), encoding="utf-8"))
ans = json.load(open(os.path.join(HERE, "listening_results.json"), encoding="utf-8"))
rows = {r["id"]: r for r in load(os.path.join(ROOT, "tts_out/TTS-0.6B_nonstream.jsonl")) if "asr_centred" in r["scores"]["O"]}


def typ(gold, alt):
    s = lambda x: re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*?)(\d)$", x).groups()[:2]
    return "tone" if s(gold) == s(alt) else "seg"


def cp(k, n):
    lo = beta.ppf(.025, k, n - k + 1) if k else 0.
    hi = beta.ppf(.975, k + 1, n - k) if k < n else 1.
    return lo, hi


for x in key:
    r = rows[x["id"]]
    a = ans[str(x["n"])]
    x["heard"] = "gold" if a == x["gold"] else "alt" if a == x["alt"] else "neither"
    x["detector"] = "gold" if x["score"] > 0 else "alt"
    x["kind"] = r["kind"]
    x["text"] = r["text"]
    x["type"] = typ(x["gold"], x["alt"])

answered = [x for x in key if x["heard"] != "neither"]
agree = [x for x in answered if x["heard"] == x["detector"]]
print(f"items {len(key)}, answered {len(answered)}, neither {len(key) - len(answered)}")
print(f"agreement on answered {len(agree)}/{len(answered)}, 95% CP {cp(len(agree), len(answered))[0]:.3f}-{cp(len(agree), len(answered))[1]:.3f}")
print(f"distinct characters answered {len({x['char'] for x in answered})}")
for s in ("wrong_seg", "wrong_tone", "right_seg", "right_tone"):
    v = [x for x in key if x["stratum"] == s]
    a = [x for x in v if x["heard"] != "neither"]
    g = [x for x in a if x["heard"] == x["detector"]]
    print(f"  {s:11s} n {len(v):2d} answered {len(a):2d} agree {len(g):2d} neither {len(v) - len(a)}")

print("\nthe listener could lean on the familiar reading or on the sentence. Harder cases for that explanation:")
hard = [x for x in answered if (x["detector"] == "gold" and x["kind"] == "min") or (x["detector"] == "alt" and x["kind"] == "maj")]
print(f"  detector says the rarer reading was spoken: {sum(x['heard'] == x['detector'] for x in hard)}/{len(hard)}")
for x in hard:
    print(f"    #{x['n']:2d} {x['char']} gold {x['gold']} alt {x['alt']} kind {x['kind']} heard {x['heard']} score {x['score']:+.2f}")
easy = [x for x in answered if x not in hard]
print(f"  detector says the commoner reading was spoken: {sum(x['heard'] == x['detector'] for x in easy)}/{len(easy)}")
vs = [x for x in answered if x["detector"] == "alt"]
print(f"  listener picked the reading that does not fit the sentence: {sum(x['heard'] == 'alt' for x in vs)}/{len(vs)}")

print("\nneither:")
for x in key:
    if x["heard"] == "neither":
        print(f"  #{x['n']:2d} id {x['id']} {x['char']} gold {x['gold']} alt {x['alt']} {x['stratum']} kind {x['kind']} score {x['score']:+.2f} | {x['text']}")

# How the reading rate would move if the listener replaces the detector inside each stratum.
print("\nraw rate of the gold reading, detector vs reweighted by the listener")
pop = {}
for r in rows.values():
    v = "right" if r["scores"]["O"]["asr_centred"] > 0 else "wrong"
    pop.setdefault((r["kind"], f"{v}_{typ(r['gold'], r['alt'])}"), []).append(r)
for kind in ("min", "maj"):
    tot = sum(len(v) for (k, _), v in pop.items() if k == kind)
    det = sum(len(v) for (k, s), v in pop.items() if k == kind and s.startswith("right")) / tot
    strict, loose = 0., 0.
    for (k, s), v in pop.items():
        if k != kind:
            continue
        sample = [x for x in key if x["stratum"] == s]
        a = [x for x in sample if x["heard"] != "neither"]
        g = sum(x["heard"] == "gold" for x in sample)
        strict += len(v) / tot * g / len(sample)
        loose += len(v) / tot * g / len(a)
    print(f"  {kind}: n {tot} detector {det:.3f} | listener, neither counted as not gold {strict:.3f} | neither dropped {loose:.3f}")
json.dump(key, open(os.path.join(HERE, "listening_scored.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
