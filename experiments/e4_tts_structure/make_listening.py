# Builds a blind listening page: one TTS-0.6B sentence per item, the listener picks which of two readings the highlighted
# character was given. Items are drawn by the detector's verdict and contrast type, but the page does not show either.
# The key is written separately so the page itself cannot reveal it.
import base64, io, json, os, random, re, sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
sys.argv = sys.argv[:1]
from poly_summary import load

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = dict(wrong_seg=14, wrong_tone=6, right_seg=10, right_tone=6)


def typ(gold, alt):
    s = lambda x: re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*?)(\d)$", x).groups()[:2]
    return "tone" if s(gold) == s(alt) else "seg"


def audio(path):
    w, sr = sf.read(path, dtype="float32")
    w = resample_poly(w, 16000, sr)
    b = io.BytesIO()
    sf.write(b, w, 16000, format="FLAC")
    return "data:audio/flac;base64," + base64.b64encode(b.getvalue()).decode()


rows = [r for r in load(os.path.join(ROOT, "tts_out/TTS-0.6B_nonstream.jsonl")) if "asr_centred" in r["scores"]["O"]]
strata = {k: [] for k in PLAN}
for r in rows:
    v = "right" if r["scores"]["O"]["asr_centred"] > 0 else "wrong"
    strata[f"{v}_{typ(r['gold'], r['alt'])}"].append(r)
rng = random.Random(7)
picked = []
for k, n in PLAN.items():
    rng.shuffle(strata[k])
    seen = {}
    for r in strata[k]:
        if len([x for x in picked if x[1]["char"] == r["char"]]) < 2 and seen.get(k, 0) < n:
            picked.append((k, r))
            seen[k] = seen.get(k, 0) + 1
rng.shuffle(picked)
items, key = [], []
for n, (k, r) in enumerate(picked):
    i = next(j for j, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)
    opts = [r["gold"], r["alt"]]
    rng.shuffle(opts)
    items.append(dict(n=n, a=r["text"][:i], c=r["char"], b=r["text"][i + 1:], opts=opts, audio=audio(os.path.join(ROOT, "tts_out/TTS-0.6B_nonstream", f"{r['id']:04d}_O.flac"))))
    key.append(dict(n=n, id=r["id"], stratum=k, gold=r["gold"], alt=r["alt"], char=r["char"], score=round(r["scores"]["O"]["asr_centred"], 2)))
json.dump(key, open(os.path.join(HERE, "listening_key.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
html = open(os.path.join(HERE, "listening_template.html"), encoding="utf-8").read().replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False))
out = os.path.join(HERE, "..", "..", "reports", "读音盲听.html")
open(out, "w", encoding="utf-8").write(html)
print(len(items), {k: sum(1 for x in key if x["stratum"] == k) for k in PLAN}, f"{os.path.getsize(out)/2**20:.1f} MB")
