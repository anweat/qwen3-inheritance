# E8: multi-listener blind listening test for the E6 on-device reading pipeline (paper H3 and H5).
# Fixed 2026-11-11, before any listener has heard anything:
#   Sentences: 40 CPP minority sentences (at most one per character group) and 30 natural2 sentences (at most two per
#     character), drawn at random (seed 11) from items whose stage 4 O audio and stage 8 P audio both exist.
#   Clips: for every sentence, O (unmodified TTS-0.6B, stage 4) and P (stage 8 pipeline: g2pW int8 + codebook). Plus 10
#     catch clips from other sentences: 5 R_g and 5 R_a (the target replaced by a homophone of the gold or the alt
#     reading, so the spoken reading is known).
#   The page shows the sentence with the target character highlighted and asks which of the two readings was spoken,
#     or "neither / unclear". It never shows the version, the detector verdict or which option is gold.
#     Option order is random per clip (fixed here); clip order is shuffled per listener in the page, with the two
#     clips of a sentence kept at least 20 positions apart.
#   Scoring and hypotheses: see score.py (written now, before any answers exist).
#   python make_page.py   ->   listening.html (send to listeners) and key.json (never send)
import base64, io, json, os, random, re, sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.expanduser("~/.cache/qwen-all/e6")
PRON = os.path.expanduser("~/.cache/qwen-all/pron")
SRC = dict(hard="tts_out", natural2="tts_natural2")
N_MIN, N_NAT2, N_CATCH = 40, 30, 10
plan = json.load(open(f"{E6}/plan.json", encoding="utf-8"))

TONE = {"a": "āáǎà", "e": "ēéěè", "i": "īíǐì", "o": "ōóǒò", "u": "ūúǔù", "ü": "ǖǘǚǜ"}


def mark(py):
    """'lu:4' / 'lv4' / 'zhong1' -> tone-marked pinyin; tone 5 is neutral."""
    m = re.match(r"^([a-z:v]+)([1-5])$", py)
    if not m:
        return py
    s, t = m.group(1).replace("u:", "ü").replace("v", "ü"), int(m.group(2))
    if t == 5:
        return s
    for v in ("a", "e"):
        if v in s:
            return s.replace(v, TONE[v][t - 1], 1)
    if "ou" in s:
        return s.replace("o", TONE["o"][t - 1], 1)
    for k in range(len(s) - 1, -1, -1):
        if s[k] in TONE:
            return s[:k] + TONE[s[k]][t - 1] + s[k + 1:]
    return s


def detector(path):
    out = {}
    for x in map(json.loads, open(path, encoding="utf-8")):
        l = x.get("logodds")
        if l:
            out[int(x["id"])] = {k: bool(v - (l["R_g"] + l["R_a"]) / 2 > 0) for k, v in l.items()}
    return out


def mp3(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    w = resample_poly(w, 16000, sr)
    w = w / max(1e-6, np.abs(w).max()) * 0.9
    b = io.BytesIO()
    sf.write(b, w, 16000, format="MP3")
    return "data:audio/mpeg;base64," + base64.b64encode(b.getvalue()).decode()


def clip_path(s, rid, v):
    if v == "P":
        return f"{E6}/render/{s}/{rid:04d}_P.flac"
    return f"{PRON}/{SRC[s]}/TTS-0.6B_nonstream/{rid:04d}_{v}.flac"


rng = random.Random(11)
det = {s: detector(f"{PRON}/{SRC[s]}/TTS-0.6B_nonstream_asr.jsonl") for s in SRC}
det_p = {s: detector(f"{E6}/asr/{s}_asr.jsonl") for s in SRC}


def pool(s, cond):
    rows = [dict(r, id=int(r["id"]), i=int(r["i"])) for r in plan["items"][s] if cond(r)]
    return [r for r in rows if all(os.path.exists(clip_path(s, r["id"], v)) for v in ("O", "P", "R_g", "R_a"))]


def draw(rows, n, per):
    rng.shuffle(rows)
    out, cnt = [], {}
    for r in rows:
        if cnt.get(r["group"], 0) < per:
            out.append(r)
            cnt[r["group"]] = cnt.get(r["group"], 0) + 1
        if len(out) == n:
            break
    return out


mins = draw(pool("hard", lambda r: r["kind"] == "min"), N_MIN, 1)
nat2 = draw(pool("natural2", lambda r: True), N_NAT2, 2)
used = {("hard", r["id"]) for r in mins} | {("natural2", r["id"]) for r in nat2}
rest = [("hard", r) for r in pool("hard", lambda r: True)] + [("natural2", r) for r in pool("natural2", lambda r: True)]
rest = [x for x in rest if (x[0], x[1]["id"]) not in used]
rng.shuffle(rest)
catch = [(s, r, "R_g" if k < N_CATCH // 2 else "R_a") for k, (s, r) in enumerate(rest[:N_CATCH])]

jobs = [("hard", r, v, "min") for r in mins for v in ("O", "P")] + [("natural2", r, v, "nat2") for r in nat2 for v in ("O", "P")]
jobs += [(s, r, v, "catch") for s, r, v in catch]
items, key = [], []
for n, (s, r, v, part) in enumerate(jobs):
    opts = [r["gold"], r["alt"]]
    rng.shuffle(opts)
    t, i = r["text"], r["i"]
    assert t[i] == r["char"], (s, r["id"])
    items.append(dict(n=n, sent=f"{s}-{r['id']}", a=t[:i], c=t[i], b=t[i + 1:], opts=[mark(o) for o in opts], audio=mp3(clip_path(s, r["id"], v))))
    dv = (det if v != "P" else det_p)[s].get(r["id"], {}).get(v)
    key.append(dict(n=n, set=s, part=part, id=r["id"], version=v, char=r["char"], gold=r["gold"], alt=r["alt"], kind=r["kind"],
                    group=r["group"], opts=opts, opt_labels=[mark(o) for o in opts], detector_gold=dv))
json.dump(dict(fixed="2026-11-11", items=key), open(os.path.join(HERE, "key.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
html = open(os.path.join(HERE, "template.html"), encoding="utf-8").read().replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False))
out = os.path.join(HERE, "listening.html")
open(out, "w", encoding="utf-8").write(html)
print(f"{len(mins)} minority + {len(nat2)} natural2 sentences, {len(catch)} catch -> {len(items)} clips, {os.path.getsize(out) / 2**20:.1f} MB")
print("distinct minority groups", len({r['group'] for r in mins}), "natural2 chars", len({r['group'] for r in nat2}))
