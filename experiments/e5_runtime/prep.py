# E5 step 1 (torch, .venv): cache the decoder inputs for the 800 E3 clips, with the language forced to what the intact
# model detected in E3 (as in e3_real_speech/forced.py). The audio tower is not quantized in any scheme, so its output is
# computed once here and shared by every decoder in run_mlx.py. Also times the audio tower per clip (fp32, MPS).
#   python prep.py
import json, os, sys, time
import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e2_activation"))
sys.path.insert(0, os.path.join(HERE, "..", "e3_real_speech"))
from asr_eval import build
from common import DEVICE
from validate import OUT as AUTO_OUT, SPEECH

OUT = os.path.expanduser("~/.cache/qwen-all/e5/prep")


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(SPEECH, "manifest.jsonl"), encoding="utf-8")]
    lang = {}
    for r in map(json.loads, open(AUTO_OUT, encoding="utf-8")):
        if r["scheme"] == "intact":
            lang[r["path"]] = r["raw"].split("<asr_text>")[0].replace("language", "").strip()
    model, proc = build()
    aid = model.config.audio_token_id
    index = []
    for i, r in enumerate(rows):
        a, _ = sf.read(os.path.join(SPEECH, r["path"]), dtype="float32")
        inp = proc.apply_transcription_request(audio=a, language=lang[r["path"]]).to(DEVICE, torch.float32)
        with torch.no_grad():
            sync(); t0 = time.perf_counter()
            emb = model.get_audio_features(inp["input_features"], inp["input_features_mask"], return_dict=True).pooler_output
            sync(); ms = (time.perf_counter() - t0) * 1000
        ids = inp["input_ids"][0].cpu().numpy().astype(np.int32)
        emb = emb.reshape(-1, emb.shape[-1]).float().cpu().numpy()
        assert (ids == aid).sum() == emb.shape[0], (r["path"], (ids == aid).sum(), emb.shape)
        np.savez(os.path.join(OUT, f"{i:04d}.npz"), ids=ids, aud=emb)
        index.append(dict(i=i, set=r["set"], path=r["path"], text=r["text"], unit=r["unit"], seconds=r.get("seconds"),
                          lang=lang[r["path"]], n_prompt=int(len(ids)), n_audio=int(emb.shape[0]), audio_tower_ms=ms))
        if i % 50 == 0:
            print(i, len(ids), f"{ms:.0f} ms", flush=True)
    json.dump(dict(audio_token_id=aid, eos=[151643, 151645], clips=index), open(os.path.join(OUT, "index.json"), "w"), ensure_ascii=False)
    print("PREPDONE", flush=True)


if __name__ == "__main__":
    main()
