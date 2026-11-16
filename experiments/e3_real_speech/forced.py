# Exploratory follow-up: same schemes and clips, but the language is supplied up front, so the model
# skips the "language X<asr_text>" step. The language per clip is the one the intact model detected.
import json, os, sys, time
import soundfile as sf
import torch
from validate import MAX_NEW, OUT as AUTO_OUT, SPEECH, norm, schemes
from asr_eval import L, MATS, build, edits
from common import DEVICE, read_decoder

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_clip_forced.jsonl")


def main():
    rows = [json.loads(l) for l in open(os.path.join(SPEECH, "manifest.jsonl"), encoding="utf-8")]
    lang = {}
    for r in map(json.loads, open(AUTO_OUT, encoding="utf-8")):
        if r["scheme"] == "intact":
            lang[r["path"]] = r["raw"].split("<asr_text>")[0].replace("language", "").strip()
    done = set()
    if os.path.exists(OUT):
        done = {(r["scheme"], r["path"]) for r in map(json.loads, open(OUT, encoding="utf-8"))}
    model, proc = build()
    print("device", DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    params = dict(model.named_parameters())
    base = read_decoder("Qwen3-0.6B-Base")
    inst = read_decoder("Qwen3-0.6B")
    keys = [f"model.layers.{i}.{t}.weight" for i in range(L) for t in MATS]
    asr = {k: params["model.language_" + k].detach().cpu().clone() for k in keys}
    for name, (fn, with_emb) in schemes(base, inst, asr, keys).items():
        if with_emb:
            continue
        todo = [r for r in rows if (name, r["path"]) not in done]
        if not todo:
            continue
        t0 = time.time()
        with torch.no_grad():
            for k in keys:
                params["model.language_" + k].copy_(fn(k).to(DEVICE))
        with open(OUT, "a", encoding="utf-8") as f:
            for r in todo:
                a, _ = sf.read(os.path.join(SPEECH, r["path"]), dtype="float32")
                inp = proc.apply_transcription_request(audio=a, language=lang[r["path"]]).to(DEVICE, torch.float32)
                with torch.no_grad():
                    o = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
                hyp = proc.tokenizer.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                ref_t, hyp_t = norm(r["text"], r["unit"]), norm(hyp, r["unit"])
                f.write(json.dumps(dict(scheme=name, set=r["set"], path=r["path"], lang=lang[r["path"]], err=edits(ref_t, hyp_t), n=len(ref_t), hyp=hyp), ensure_ascii=False) + "\n")
                f.flush()
        print(f"{name}: {len(todo)} clips in {time.time()-t0:.0f}s", flush=True)
    print("FORCEDDONE", flush=True)


if __name__ == "__main__":
    main()
