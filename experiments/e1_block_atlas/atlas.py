# Parameter-block atlas for same-shaped Qwen3 function models.
# Range-reads full matrices of selected layers and reports drift by block, head and neuron,
# plus lineage (delta cosines), subspace overlap and weight-distribution signatures.
import hashlib, json, os, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import numpy as np

HF = "https://huggingface.co/Qwen"
CACHE = os.environ.get("ATLAS_CACHE", "cache_atlas")
MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
NORMS = ["input_layernorm", "post_attention_layernorm"]
HEAD_DIM = 128

FAMILIES = {
    "0.6B": {
        "layers": [2, 14, 26],
        "base": "Qwen3-0.6B-Base",
        "models": {
            "Qwen3-0.6B-Base": "model.", "Qwen3-0.6B": "model.",
            "Qwen3Guard-Gen-0.6B": "model.", "Qwen3Guard-Stream-0.6B": "model.",
            "Qwen3-Embedding-0.6B": "", "Qwen3-Reranker-0.6B": "model.",
            "Qwen3-ASR-0.6B": "thinker.model.", "Qwen3-ForcedAligner-0.6B": "thinker.model.",
            "Qwen3-TTS-12Hz-0.6B-Base": "talker.model.", "Qwen3-TTS-12Hz-0.6B-CustomVoice": "talker.model.",
        },
    },
    "1.7B": {
        "layers": [14],
        "base": "Qwen3-1.7B-Base",
        "models": {
            "Qwen3-1.7B-Base": "model.", "Qwen3-1.7B": "model.",
            "Qwen3-ASR-1.7B": "thinker.model.", "Qwen3-TTS-12Hz-1.7B-Base": "talker.model.",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice": "talker.model.", "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "talker.model.",
            "Qwen3-VL-2B-Instruct": "model.language_model.", "Qwen3-VL-2B-Thinking": "model.language_model.",
            "Qwen3-VL-Embedding-2B": "model.language_model.", "Qwen3-VL-Reranker-2B": "model.language_model.",
        },
    },
}
RESOLVED = {}


def get(url, a, b):
    for _ in range(6):
        try:
            r = urllib.request.Request(RESOLVED.get(url, url), headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(r, timeout=120) as resp:
                RESOLVED.setdefault(url, resp.geturl())
                data = resp.read()
            if len(data) == b - a + 1:
                return data
            err = IOError(f"short read {len(data)}")
        except Exception as e:
            err = e
            RESOLVED.pop(url, None)
    raise err


def headers(repo):
    d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/Qwen/{repo}"))
    out = {}
    for f in [s["rfilename"] for s in d["siblings"] if s["rfilename"].endswith(".safetensors")]:
        url = f"{HF}/{repo}/resolve/main/{f}"
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        h = json.loads(get(url, 8, 7 + n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = (url, 8 + n, v)
    return out


def decode(buf, dtype, shape):
    if dtype == "BF16":
        u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).reshape(shape)
    raise ValueError(dtype)


def fetch(entry):
    url, base, v = entry
    s, e = v["data_offsets"]
    p = os.path.join(CACHE, hashlib.md5(f"{url}|{s}|{e}".encode()).hexdigest() + ".npy")
    if os.path.exists(p):
        return np.load(p)
    arr = decode(get(url, base + s, base + e - 1), v["dtype"], v["shape"])
    tmp = p + f".{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, p)
    return arr


def load_family(fam):
    H = {m: headers(m) for m in fam["models"]}
    jobs = []
    for m, pre in fam["models"].items():
        for L in fam["layers"]:
            for t in MATS:
                jobs.append(((m, L, t), H[m][f"{pre}layers.{L}.{t}.weight"]))
        for L in range(28):
            for t in NORMS:
                jobs.append(((m, L, t), H[m][f"{pre}layers.{L}.{t}.weight"]))
    T = {}
    with ThreadPoolExecutor(6) as ex:
        for (key, _), arr in zip(jobs, ex.map(lambda j: fetch(j[1]), jobs)):
            T[key] = arr
    return T


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def top_share(x, frac=0.1):
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    return float(x[: max(1, int(len(x) * frac))].sum() / x.sum())


def group_energy(d, t):
    """Energy of a delta grouped by head or by MLP neuron, plus the matching base energy helper."""
    if t in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        n = d.shape[0] // HEAD_DIM
        return (d.reshape(n, HEAD_DIM, -1) ** 2).sum((1, 2))
    if t == "self_attn.o_proj":
        n = d.shape[1] // HEAD_DIM
        return (d.reshape(d.shape[0], n, HEAD_DIM) ** 2).sum((0, 2))
    if t in ("mlp.gate_proj", "mlp.up_proj"):
        return (d ** 2).sum(1)
    if t == "mlp.down_proj":
        return (d ** 2).sum(0)
    raise ValueError(t)


def subspace_overlap(A, B, k=64):
    Va = np.linalg.svd(A, full_matrices=False)[2][:k]
    Vb = np.linalg.svd(B, full_matrices=False)[2][:k]
    return float((Va @ Vb.T).__pow__(2).sum() / k)


def analyze(name, fam, T, out):
    base, layers, models = fam["base"], fam["layers"], list(fam["models"])
    others = [m for m in models if m != base]
    print(f"\n{'='*20} family {name}: reference {base}, layers {layers} {'='*20}")
    res = {}

    print("\n[1] block drift ||W-B||/||B||, cos, alpha (W≈alpha*B), residual ||W-alpha*B||/||W|| (mean over layers/matrices)")
    for m in others:
        rows = []
        for L in layers:
            for t in MATS:
                W, B = T[(m, L, t)], T[(base, L, t)]
                nb, nw = np.linalg.norm(B), np.linalg.norm(W)
                ip = float((W * B).sum())
                alpha = ip / nb ** 2
                rows.append((np.linalg.norm(W - B) / nb, ip / nb / nw, alpha, np.linalg.norm(W - alpha * B) / nw))
        r = np.mean(rows, 0)
        res.setdefault(m, {})["block"] = r.tolist()
        print(f"  {m:34s} drift {r[0]:.3f}  cos {r[1]:.3f}  alpha {r[2]:.3f}  resid {r[3]:.3f}")

    print("\n[2] drift by matrix type (mean over layers)")
    print("  " + " " * 34 + " ".join(f"{t.split('.')[1][:6]:>7s}" for t in MATS) + "  | ln-corr")
    for m in others:
        v = [np.mean([np.linalg.norm(T[(m, L, t)] - T[(base, L, t)]) / np.linalg.norm(T[(base, L, t)]) for L in layers]) for t in MATS]
        ln = np.mean([np.corrcoef(T[(m, L, "input_layernorm")], T[(base, L, "input_layernorm")])[0, 1] for L in range(28)])
        res[m]["by_matrix"] = v
        print(f"  {m:34s}" + " ".join(f"{x:7.3f}" for x in v) + f"  | {ln:.3f}")

    print("\n[3] concentration: Gini and top-10% share of delta energy across heads (q/k/v/o) and MLP neurons (gate/up/down)")
    rng = np.random.default_rng(0)
    L0 = layers[len(layers) // 2]
    for t in ["self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj"]:
        B = T[(base, L0, t)]
        null = group_energy(rng.standard_normal(B.shape, dtype=np.float32), t)
        print(f"  iid-noise null, {t:18s}  gini {gini(null):.3f}  top10% {top_share(null):.3f}")
    for m in others:
        g_h, g_n, t_n, c_gu, c_gd, h_rng = [], [], [], [], [], []
        for L in layers:
            for t in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
                e = group_energy(T[(m, L, t)] - T[(base, L, t)], t)
                b = group_energy(T[(base, L, t)], t)
                g_h.append(gini(e))
                r = np.sqrt(e / b)
                h_rng.append((r.min(), r.max()))
            en = {t: group_energy(T[(m, L, t)] - T[(base, L, t)], t) for t in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
            for v in en.values():
                g_n.append(gini(v))
                t_n.append(top_share(v))
            c_gu.append(np.corrcoef(en["mlp.gate_proj"], en["mlp.up_proj"])[0, 1])
            c_gd.append(np.corrcoef(en["mlp.gate_proj"], en["mlp.down_proj"])[0, 1])
        hr = np.mean(h_rng, 0)
        res[m]["conc"] = dict(head_gini=float(np.mean(g_h)), head_min=float(hr[0]), head_max=float(hr[1]),
                              neuron_gini=float(np.mean(g_n)), neuron_top10=float(np.mean(t_n)),
                              gate_up_corr=float(np.mean(c_gu)), gate_down_corr=float(np.mean(c_gd)))
        print(f"  {m:34s} head gini {np.mean(g_h):.3f} drift[{hr[0]:.2f},{hr[1]:.2f}]  neuron gini {np.mean(g_n):.3f} top10% {np.mean(t_n):.3f}  corr gate~up {np.mean(c_gu):.3f} gate~down {np.mean(c_gd):.3f}")

    print("\n[4] lineage: cosine between deltas from the reference (all sampled matrices concatenated)")
    D = {m: np.concatenate([(T[(m, L, t)] - T[(base, L, t)]).ravel() for L in layers for t in MATS]) for m in others}
    for m in D:
        D[m] = D[m] / np.linalg.norm(D[m])
    short = [m.replace("Qwen3", "").replace("-12Hz", "").replace("-0.6B", "").replace("-1.7B", "").strip("-") or "instruct" for m in others]
    print("  " + " " * 18 + " ".join(f"{s[:9]:>9s}" for s in short))
    lin = np.zeros((len(others), len(others)))
    for i, a in enumerate(others):
        for j, b in enumerate(others):
            lin[i, j] = float(D[a] @ D[b])
        print(f"  {short[i][:18]:18s}" + " ".join(f"{x:9.3f}" for x in lin[i]))
    res["_lineage"] = dict(names=others, matrix=lin.tolist())

    print("\n[5] do the same heads/neurons move across functions? correlation of per-group delta energy (mean over layers)")
    pairs = {}
    for a, b in combinations(others, 2):
        h, n = [], []
        for L in layers:
            for t in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
                h.append(np.corrcoef(group_energy(T[(a, L, t)] - T[(base, L, t)], t), group_energy(T[(b, L, t)] - T[(base, L, t)], t))[0, 1])
            for t in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                n.append(np.corrcoef(group_energy(T[(a, L, t)] - T[(base, L, t)], t), group_energy(T[(b, L, t)] - T[(base, L, t)], t))[0, 1])
        pairs[f"{a}|{b}"] = (float(np.mean(h)), float(np.mean(n)))
    for k, v in sorted(pairs.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"  {k:66s} head {v[0]:.3f}  neuron {v[1]:.3f}")
    print("  ...")
    for k, v in sorted(pairs.items(), key=lambda kv: -kv[1][1])[-4:]:
        print(f"  {k:66s} head {v[0]:.3f}  neuron {v[1]:.3f}")
    res["_pairs"] = pairs

    print("\n[6] top-64 input-subspace overlap with reference (1=same subspace; random≈64/dim), stable rank, weight kurtosis")
    hidden = T[(base, L0, "self_attn.q_proj")].shape[1]
    print(f"  random expectation ≈ {64/hidden:.3f}")
    for m in models:
        ov, sr, ku = [], [], []
        for t in ["self_attn.q_proj", "mlp.gate_proj"]:
            W, B = T[(m, L0, t)], T[(base, L0, t)]
            ov.append(subspace_overlap(W, B))
            s = np.linalg.svd(W, compute_uv=False)
            sr.append(float((s ** 2).sum() / s[0] ** 2))
            x = W.ravel()
            ku.append(float(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3))
        res.setdefault(m, {})["sub"] = dict(overlap_q=ov[0], overlap_gate=ov[1], stable_q=sr[0], stable_gate=sr[1], kurt_q=ku[0], kurt_gate=ku[1])
        print(f"  {m:34s} overlap q {ov[0]:.3f} gate {ov[1]:.3f}  stable-rank q {sr[0]:.0f} gate {sr[1]:.0f}  kurtosis q {ku[0]:.2f} gate {ku[1]:.2f}")

    print("\n[7] RMSNorm depth profile: mean input_layernorm scale per layer (every 3rd layer)")
    print("  " + " " * 34 + " ".join(f"L{L:<5d}" for L in range(0, 28, 3)))
    for m in models:
        v = [float(T[(m, L, "input_layernorm")].mean()) for L in range(0, 28, 3)]
        res[m]["ln_profile"] = v
        print(f"  {m:34s}" + " ".join(f"{x:6.2f}" for x in v))
    return res


def main():
    os.makedirs(CACHE, exist_ok=True)
    all_res = {}
    for name in sys.argv[1:] or FAMILIES:
        T = load_family(FAMILIES[name])
        all_res[name] = analyze(name, FAMILIES[name], T, None)
        del T
    json.dump(all_res, open("atlas.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
