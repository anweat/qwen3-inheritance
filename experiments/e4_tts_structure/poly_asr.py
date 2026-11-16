# Independent reading detector for poly_tts.py, run in the main .venv (transformers 5, ASR weights).
# The ASR is teacher forced with the sentence left of the target; its next-token distribution is split by the reading of
# each token's first character, using only single-reading characters. The score is log P(gold reading) - log P(competing
# reading), so no alignment is needed and the TTS plays no part. The polyphone itself carries both readings and is excluded.
# The substitute characters are excluded too: otherwise calibration audio would be scored with the spoken character inside
# a class, while original audio never is, and calibration would overstate accuracy on the originals.
#   uv-free: ../../.venv/bin/python poly_asr.py <jsonl> [<jsonl> ...]
import json, os, re, sys
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2_activation"))
from asr_eval import build  # noqa: E402
from common import DEVICE  # noqa: E402

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
HAN = re.compile(r"^[一-鿿]")


def reading_index(tokenizer, mono):
    """Reading -> token ids whose text starts with a single-reading character."""
    index, first = {}, {}
    for i in range(len(tokenizer)):
        t = tokenizer.decode([i])
        if HAN.match(t) and t[0] in mono:
            index.setdefault(mono[t[0]], []).append(i)
            first[i] = t[0]
    return {r: torch.tensor(v, device=DEVICE) for r, v in index.items()}, first


def logodds(model, proc, wav, sr, prefix, ig, ia):
    a = resample_poly(wav, 16000, sr).astype(np.float32)
    inp = proc.apply_transcription_request(audio=a, language="Chinese").to(DEVICE, torch.float32)
    extra = proc.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)
    inp["input_ids"] = torch.cat([inp["input_ids"], extra], 1)
    inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(extra)], 1)
    with torch.no_grad():
        lg = model(**inp).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    return float(torch.logsumexp(lp[ig], 0) - torch.logsumexp(lp[ia], 0))


def main():
    mono = json.load(open(os.path.join(ROOT, "mono_read.json"), encoding="utf-8"))
    model, proc = build()
    model = model.to(DEVICE).eval()
    index, first = reading_index(proc.tokenizer, mono)
    print(f"{len(index)} readings indexed", flush=True)
    for path in sys.argv[1:]:
        wavdir, out = path[:-len(".jsonl")], path[:-len(".jsonl")] + "_asr.jsonl"
        done = set()
        if os.path.exists(out):
            done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if r["id"] not in done]
        with open(out, "a", encoding="utf-8") as fo:
            for c, r in enumerate(rows):
                if r["gold"] not in index or r["alt"] not in index:
                    fo.write(json.dumps(dict(id=r["id"], skip=True)) + "\n")
                    continue
                i = next(n for n, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)
                spoken = {t[i] for t in r["scores"]["texts"].values()}
                ig, ia = [torch.tensor([t for t in index[x].tolist() if first[t] not in spoken], device=DEVICE) for x in (r["gold"], r["alt"])]
                if not len(ig) or not len(ia):
                    fo.write(json.dumps(dict(id=r["id"], skip=True)) + "\n")
                    continue
                s = {}
                for k in r["scores"]["texts"]:
                    w, sr = sf.read(os.path.join(wavdir, f"{r['id']:04d}_{k}.flac"), dtype="float32")
                    s[k] = logodds(model, proc, w, sr, r["text"][:i], ig, ia)
                fo.write(json.dumps(dict(id=r["id"], logodds=s)) + "\n")
                fo.flush()
                if c % 50 == 0:
                    print(f"  {os.path.basename(path)} {c}/{len(rows)}", flush=True)
                    if DEVICE == "mps":
                        torch.mps.empty_cache()
    print("ASRDONE", flush=True)


if __name__ == "__main__":
    main()
