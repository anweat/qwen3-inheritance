# Held-out validation of residual coding for Qwen3-ASR-0.6B on real speech.
# Fixed before running. PASS for a scheme, against standalone Q4 on decoder matrices:
#   pooled paired error difference has a 95% upper bound <= +0.5 percentage points,
#   and no single set has a point estimate worse than +2 percentage points.
# Results are written per clip, so the run can resume after an interruption.
import json, os, re, sys, time, unicodedata
import numpy as np
import soundfile as sf
import torch
from opencc import OpenCC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2_activation"))
from asr_eval import L, MATS, build, edits, qdelta, quant
from common import DEVICE, read_decoder

SPEECH = os.path.expanduser("~/.cache/qwen-all/speech")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_clip.jsonl")
CC = OpenCC("t2s")
MAX_NEW = 200


def norm(s, unit):
    s = CC.convert(unicodedata.normalize("NFKC", s)).lower()
    s = "".join(c if unicodedata.category(c)[0] not in "PSCZ" or c == " " else " " for c in s)
    return [c for c in s if c != " "] if unit == "char" else s.split()


def schemes(base, inst, asr, keys):
    def delta(ref, bits):
        q4 = {k: quant(ref[k], 4) for k in keys}
        a4 = {k: float((asr[k] * q4[k]).sum() / (q4[k] * q4[k]).sum()) for k in keys}
        return lambda k: a4[k] * q4[k] + qdelta(asr[k] - a4[k] * q4[k], bits)
    return {
        "intact": (lambda k: asr[k], False),
        "standalone_q4": (lambda k: quant(asr[k], 4), False),
        "base_q4_plus_2bit": (delta(base, 2), False),
        "instruct_q4_plus_2bit": (delta(inst, 2), False),
        "instruct_q4_plus_2bit_with_embedding": (delta(inst, 2), True),
    }


def main():
    rows = [json.loads(l) for l in open(os.path.join(SPEECH, "manifest.jsonl"), encoding="utf-8")]
    done = set()
    if os.path.exists(OUT):
        done = {(r["scheme"], r["path"]) for r in map(json.loads, open(OUT, encoding="utf-8"))}
    model, proc = build()
    params = dict(model.named_parameters())
    base = read_decoder("Qwen3-0.6B-Base")
    inst = read_decoder("Qwen3-0.6B")
    keys = [f"model.layers.{i}.{t}.weight" for i in range(L) for t in MATS]
    asr = {k: params["model.language_" + k].detach().cpu().clone() for k in keys}
    E = "model.language_model.embed_tokens.weight"
    emb = params[E].detach().cpu().clone()
    ei = inst["model.embed_tokens.weight"]
    q_ei = quant(ei, 4)
    a_e = float((emb * q_ei).sum() / (q_ei * q_ei).sum())
    emb_delta = a_e * q_ei + qdelta(emb - a_e * q_ei, 2)
    print(f"embedding residual relative error {float((emb_delta-emb).norm()/emb.norm()):.4f}, lm_head tied: {model.lm_head.weight is params[E]}", flush=True)
    for name, (fn, with_emb) in schemes(base, inst, asr, keys).items():
        todo = [r for r in rows if (name, r["path"]) not in done]
        if not todo:
            continue
        t0 = time.time()
        with torch.no_grad():
            for k in keys:
                params["model.language_" + k].copy_(fn(k).to(DEVICE))
            params[E].copy_((emb_delta if with_emb else emb).to(DEVICE))
        with open(OUT, "a", encoding="utf-8") as f:
            for r in todo:
                a, _ = sf.read(os.path.join(SPEECH, r["path"]), dtype="float32")
                inp = proc.apply_transcription_request(audio=a).to(DEVICE, torch.float32)
                with torch.no_grad():
                    o = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
                raw = proc.decode(o[:, inp["input_ids"].shape[1]:])[0]
                hyp = proc.decode(o[:, inp["input_ids"].shape[1]:], return_format="transcription_only")[0]
                ref_t, hyp_t = norm(r["text"], r["unit"]), norm(hyp, r["unit"])
                f.write(json.dumps(dict(scheme=name, set=r["set"], path=r["path"], err=edits(ref_t, hyp_t), n=len(ref_t), hyp=hyp, raw=raw[:60]), ensure_ascii=False) + "\n")
                f.flush()
        print(f"{name}: {len(todo)} clips in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
