# Probe how far Qwen3 function models' LM backbones drift from Qwen3-1.7B-Base,
# using HTTP range reads on safetensors (no full download).
import json, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HF = "https://huggingface.co/Qwen"
MODELS = {
    "base": ("Qwen3-1.7B-Base", ["model.safetensors"], "model."),
    "instruct": ("Qwen3-1.7B", ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"], "model."),
    "asr": ("Qwen3-ASR-1.7B", ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"], "thinker.model."),
    "tts": ("Qwen3-TTS-12Hz-1.7B-Base", ["model.safetensors"], "talker.model."),
}
MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
ROWS = 64
SVD_LAYERS = [2, 14, 26]


RESOLVED = {}


def get(url, a, b):
    for i in range(6):
        try:
            r = urllib.request.Request(RESOLVED.get(url, url), headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(r, timeout=60) as resp:
                RESOLVED.setdefault(url, resp.geturl())
                data = resp.read()
            if len(data) == b - a + 1:
                return data
            err = IOError(f"short read {len(data)}")
        except Exception as e:
            err = e
            RESOLVED.pop(url, None)
    raise err


def headers(repo, files):
    out = {}
    for f in files:
        url = f"{HF}/{repo}/resolve/main/{f}"
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        h = json.loads(get(url, 8, 7 + n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = (url, 8 + n, v)
    return out


def bf16(buf, shape):
    u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape)


def fetch(entry, rows=None, row_off=0, key=None):
    import os, hashlib
    p = "cache/" + hashlib.md5(f"{entry[0]}|{entry[2]['data_offsets']}|{rows}|{row_off}".encode()).hexdigest() + ".npy"
    if os.path.exists(p):
        return np.load(p)
    arr = _fetch(entry, rows, row_off)
    np.save(p, arr)
    return arr


def _fetch(entry, rows=None, row_off=0):
    url, base, v = entry
    assert v["dtype"] == "BF16"
    shape = v["shape"]
    s, e = v["data_offsets"]
    if len(shape) == 1:
        return bf16(get(url, base + s, base + e - 1), shape)
    cols = shape[1]
    r = shape[0] if rows is None else min(rows, shape[0] - row_off)
    a = base + s + row_off * cols * 2
    return bf16(get(url, a, a + r * cols * 2 - 1), (r, cols))


def main():
    import os; os.makedirs("cache", exist_ok=True)
    H = {k: headers(repo, files) for k, (repo, files, _) in MODELS.items()}
    jobs = []
    for m, (_, _, pre) in MODELS.items():
        for L in range(28):
            for t in MATS:
                jobs.append((m, f"L{L}.{t}", H[m][f"{pre}layers.{L}.{t}.weight"], ROWS, 0))
            for t in ["input_layernorm", "post_attention_layernorm"]:
                jobs.append((m, f"L{L}.{t}", H[m][f"{pre}layers.{L}.{t}.weight"], None, 0))
        for L in SVD_LAYERS:
            jobs.append((m, f"FULL.L{L}.q_proj", H[m][f"{pre}layers.{L}.self_attn.q_proj.weight"], None, 0))
        emb = "talker.model.text_embedding" if m == "tts" else pre + "embed_tokens"
        jobs.append((m, "embed.rows20000", H[m][emb + ".weight"], 256, 20000))
    if "thinker.lm_head.weight" in H["asr"]:
        jobs.append(("asr", "lm_head.rows20000", H["asr"]["thinker.lm_head.weight"], 256, 20000))

    T = {}
    with ThreadPoolExecutor(6) as ex:
        for (m, name, _, _, _), arr in zip(jobs, ex.map(lambda j: fetch(j[2], j[3], j[4]), jobs)):
            T[(m, name)] = arr

    def rel(m, name, ref="base"):
        W, B = T[(m, name)], T[(ref, name)]
        d = W - B
        cos = float((W * B).sum() / np.linalg.norm(W) / np.linalg.norm(B))
        return float(np.linalg.norm(d) / np.linalg.norm(B)), cos

    res = {}
    for m in ["instruct", "asr", "tts"]:
        per_layer = []
        for L in range(28):
            row = {t.split(".")[1]: rel(m, f"L{L}.{t}")[0] for t in MATS}
            row["ln"] = rel(m, f"L{L}.input_layernorm")[0]
            per_layer.append(row)
        res[m] = per_layer
    json.dump(res, open("probe_rel.json", "w"), indent=1)

    print("relative ||W-W_base||/||W_base|| per layer (mean over 7 mats, first 64 rows); also vs instruct")
    print("L   instruct   asr     tts    | asr-vs-inst tts-vs-inst")
    for L in range(28):
        f = lambda m, ref="base": np.mean([rel(m, f"L{L}.{t}", ref)[0] for t in MATS])
        print(f"{L:2d}  {f('instruct'):.4f}   {f('asr'):.4f}  {f('tts'):.4f} |  {f('asr','instruct'):.4f}     {f('tts','instruct'):.4f}")
    for m in ["instruct", "asr", "tts"]:
        mats = {t: np.mean([rel(m, f"L{L}.{t}")[0] for L in range(28)]) for t in MATS}
        print(m, "by matrix:", {k.split('.')[1]: round(v, 4) for k, v in mats.items()},
              "embed:", round(rel(m, "embed.rows20000")[0], 4),
              "cos(min over mats):", round(min(rel(m, f"L{L}.{t}")[1] for L in range(28) for t in MATS), 4))
    if ("asr", "lm_head.rows20000") in T:
        a, b = T[("asr", "lm_head.rows20000")], T[("asr", "embed.rows20000")]
        print("asr lm_head vs its embed rel:", float(np.linalg.norm(a - b) / np.linalg.norm(b)))

    print("\nBitDelta proxy (sign * per-matrix mean|d|), per model, first-64-row slabs:")
    for m in ["instruct", "asr", "tts"]:
        e_d, e_w = [], []
        for L in range(28):
            for t in MATS:
                W, B = T[(m, f"L{L}.{t}")], T[("base", f"L{L}.{t}")]
                d = W - B
                q = np.sign(d) * np.abs(d).mean()
                e_d.append(np.linalg.norm(d - q) / np.linalg.norm(d))
                e_w.append(np.linalg.norm(d - q) / np.linalg.norm(W))
        print(f"  {m:8s} delta recon err {np.mean(e_d):.3f}   weight-level err {np.mean(e_w):.5f}")

    print("\nSVD of full q_proj delta: rank needed for 50/90% energy (of 2048), and top-64 energy share")
    for m in ["instruct", "asr", "tts"]:
        for L in SVD_LAYERS:
            d = T[(m, f"FULL.L{L}.q_proj")] - T[("base", f"FULL.L{L}.q_proj")]
            s = np.linalg.svd(d, compute_uv=False) ** 2
            c = np.cumsum(s) / s.sum()
            print(f"  {m:8s} L{L:2d}  r50={int(np.searchsorted(c, .5))+1:4d}  r90={int(np.searchsorted(c, .9))+1:4d}  top64={c[63]:.3f}")


if __name__ == "__main__":
    main()
