# On-device cost of each reading head's feature path, CPU only, single process, float32. Timings are for this machine.
import json, os, time
import numpy as np
import torch
from transformers import Qwen3ForCausalLM
from qwen_tts import Qwen3TTSModel
from shared_structure import ROLE, rp, tok

torch.set_num_threads(4)
gold = json.load(open("natural2_gold.json", encoding="utf-8"))
texts = sorted({x["text"] for x in gold}, key=len)
texts = [t for t in texts if 20 <= len(t) <= 40][:30]
ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
print(f"{len(ids)} sentences, median {int(np.median([len(i) for i in ids]))} tokens")


def params(mods):
    return sum(p.numel() for m in mods for p in m.parameters())


m = Qwen3ForCausalLM.from_pretrained(rp("Qwen/Qwen3-0.6B-Base"), dtype=torch.float32).eval()
layers = m.model.layers
for k in (4, 8, 12, 28):
    m.model.layers = layers[:k]
    t = []
    with torch.no_grad():
        for x in ids:
            s = time.perf_counter()
            m.model(input_ids=torch.tensor([ROLE + x]), use_cache=False)
            t.append(time.perf_counter() - s)
    n = params([m.model.embed_tokens] + list(layers[:k]))
    print(f"LLM-0.6B first {k:2d} layers: {n / 1e6:7.1f}M params, {n * 4 / 2**20:6.0f} MB float32, {n / 2 ** 20:6.0f} MB at 8 bit, median {np.median(t) * 1000:6.0f} ms per sentence")
del m

w = Qwen3TTSModel.from_pretrained(rp("Qwen/Qwen3-TTS-12Hz-0.6B-Base"), device_map="cpu", dtype=torch.float32)
t = w.model.talker
te = t.get_text_embeddings()
print(f"TTS-0.6B talker total {params([t]) / 1e6:.1f}M params; text embedding {te.weight.shape}")
mono = json.load(open(os.path.expanduser("~/.cache/qwen-all/pron/mono_read.json"), encoding="utf-8"))
readings = set(mono.values())
d = te.weight.shape[1]
print(f"reading codebook {len(readings)} x {d} = {len(readings) * d * 2 / 2**20:.1f} MB at 16 bit")
print("COSTDONE")
