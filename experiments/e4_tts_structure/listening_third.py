# Exploratory, after seeing the blind answers: when the listener says neither offered reading fits, does the ASR also favour a
# third reading? poly_asr.py only compares gold with alt, so it cannot say. Here the next-token distribution is split over
# every reading class, on the same 36 O clips, with the same exclusion of the polyphone and its substitute characters.
import json, os, sys
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2_activation"))
from asr_eval import build  # noqa: E402
from common import DEVICE  # noqa: E402
from poly_asr import reading_index  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
PATH = os.path.join(ROOT, "tts_out/TTS-0.6B_nonstream.jsonl")
key = json.load(open(os.path.join(HERE, "listening_scored.json"), encoding="utf-8"))
rows = {r["id"]: r for r in map(json.loads, open(PATH, encoding="utf-8")) if "id" in r}
mono = json.load(open(os.path.join(ROOT, "mono_read.json"), encoding="utf-8"))
model, proc = build()
model = model.to(DEVICE).eval()
index, first = reading_index(proc.tokenizer, mono)

out = []
for x in key:
    r = rows[x["id"]]
    i = next(n for n, (a, b) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if a != b)
    spoken = {t[i] for t in r["scores"]["texts"].values()}
    w, sr = sf.read(os.path.join(ROOT, "tts_out/TTS-0.6B_nonstream", f"{r['id']:04d}_O.flac"), dtype="float32")
    a = resample_poly(w, 16000, sr).astype(np.float32)
    inp = proc.apply_transcription_request(audio=a, language="Chinese").to(DEVICE, torch.float32)
    extra = proc.tokenizer(r["text"][:i], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)
    inp["input_ids"] = torch.cat([inp["input_ids"], extra], 1)
    inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(extra)], 1)
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0, -1].float(), -1)
    mass = {}
    for reading, ids in index.items():
        keep = torch.tensor([t for t in ids.tolist() if first[t] not in spoken], device=DEVICE)
        if len(keep):
            mass[reading] = float(torch.logsumexp(lp[keep], 0))
    names = list(mass)
    p = torch.softmax(torch.tensor([mass[n] for n in names]), 0).numpy()
    rank = sorted(zip(names, p.tolist()), key=lambda t: -t[1])
    share = sum(q for n, q in rank if n in (x["gold"], x["alt"]))
    top_tokens = [proc.tokenizer.decode([t]) for t in torch.topk(lp, 5).indices.tolist()]
    out.append(dict(n=x["n"], char=x["char"], gold=x["gold"], alt=x["alt"], heard=x["heard"], top=rank[:4], share=share, tokens=top_tokens))
    print(f"#{x['n']:2d} {x['char']} {x['gold']}/{x['alt']} heard {x['heard']:7s} pair share {share:.3f} top " + " ".join(f"{n} {q:.2f}" for n, q in rank[:4]) + f" | {top_tokens}", flush=True)
    if DEVICE == "mps":
        torch.mps.empty_cache()

for label, v in (("neither", [o for o in out if o["heard"] == "neither"]), ("answered", [o for o in out if o["heard"] != "neither"])):
    third = sum(o["top"][0][0] not in (o["gold"], o["alt"]) for o in v)
    print(f"{label}: n {len(v)} median pair share {np.median([o['share'] for o in v]):.3f}, third reading on top {third}/{len(v)}")
json.dump(out, open(os.path.join(HERE, "listening_third.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
