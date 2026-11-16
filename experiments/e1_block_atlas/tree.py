# Second pass over the atlas cache: remove the shrink artifact, measure the model tree directly,
# test clade centroids as a shared base, and compare weight tails / quantization error.
import sys
import numpy as np
import atlas
from atlas import FAMILIES, MATS, gini, top_share, group_energy, load_family

CLADES = {
    "0.6B": {
        "text": ["Qwen3-0.6B", "Qwen3Guard-Stream-0.6B", "Qwen3-Reranker-0.6B"],
        "base+guard+emb": ["Qwen3-0.6B-Base", "Qwen3Guard-Gen-0.6B", "Qwen3-Embedding-0.6B"],
        "all-text": ["Qwen3-0.6B-Base", "Qwen3-0.6B", "Qwen3Guard-Gen-0.6B", "Qwen3Guard-Stream-0.6B", "Qwen3-Embedding-0.6B", "Qwen3-Reranker-0.6B"],
        "audio": ["Qwen3-ASR-0.6B", "Qwen3-ForcedAligner-0.6B"],
        "tts": ["Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS-12Hz-0.6B-CustomVoice"],
    },
    "1.7B": {
        "vl": ["Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Thinking", "Qwen3-VL-Embedding-2B", "Qwen3-VL-Reranker-2B"],
        "vl+asr": ["Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Thinking", "Qwen3-VL-Embedding-2B", "Qwen3-VL-Reranker-2B", "Qwen3-ASR-1.7B"],
        "text+vl+asr": ["Qwen3-1.7B", "Qwen3-VL-2B-Instruct", "Qwen3-ASR-1.7B"],
        "tts": ["Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
    },
}


def short(m):
    return m.replace("Qwen3", "").replace("-12Hz", "").replace("-0.6B", "").replace("-1.7B", "").replace("-2B", "").strip("-") or "instruct"


def flat(T, m, layers):
    return np.concatenate([T[(m, L, t)].ravel() for L in layers for t in MATS])


def one_bit_err(W, B):
    """Weight-level error of a per-matrix sign*mean|d| delta, no scale distillation."""
    d = W - B
    q = np.sign(d) * np.abs(d).mean()
    return np.linalg.norm(d - q) / np.linalg.norm(W)


def quant_err(W, bits=4, group=32):
    """Relative error of symmetric min-max quantization over blocks of `group` weights."""
    x = W.ravel()
    n = len(x) // group * group
    blocks = x[:n].reshape(-1, group)
    scale = np.abs(blocks).max(1, keepdims=True) + 1e-12
    levels = 2 ** (bits - 1) - 1
    q = np.round(blocks / scale * levels) / levels * scale
    return float(np.linalg.norm(blocks - q) / np.linalg.norm(blocks))


def r90(W):
    s = np.linalg.svd(W, compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / s.sum(), 0.9)) + 1


def run(name):
    fam = FAMILIES[name]
    base, layers, models = fam["base"], fam["layers"], list(fam["models"])
    T = load_family(fam)
    F = {m: flat(T, m, layers) for m in models}
    print(f"\n{'='*20} family {name} {'='*20}")

    print("\n[A] direct pairwise distance ||Wa-Wb||/sqrt(||Wa|| ||Wb||)")
    n = len(models)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(F[models[i]] - F[models[j]]) / np.sqrt(np.linalg.norm(F[models[i]]) * np.linalg.norm(F[models[j]]))
    print("  " + " " * 16 + " ".join(f"{short(m)[:7]:>7s}" for m in models))
    for i, m in enumerate(models):
        print(f"  {short(m)[:16]:16s}" + " ".join(f"{x:7.3f}" for x in D[i]))

    print("\n[B] minimum spanning tree (edges by distance)")
    seen, edges = {0}, []
    while len(seen) < n:
        e = min(((D[i, j], i, j) for i in seen for j in range(n) if j not in seen))
        edges.append(e)
        seen.add(e[2])
    for d, i, j in sorted(edges):
        print(f"  {short(models[i]):22s} -- {short(models[j]):22s} {d:.3f}")

    print("\n[C] residual lineage: cosine of R = W - alpha*B (shrink removed)")
    B = F[base]
    others = [m for m in models if m != base]
    R = {}
    for m in others:
        a = (F[m] @ B) / (B @ B)
        r = F[m] - a * B
        R[m] = r / np.linalg.norm(r)
    print("  " + " " * 16 + " ".join(f"{short(m)[:7]:>7s}" for m in others))
    for a in others:
        print(f"  {short(a)[:16]:16s}" + " ".join(f"{float(R[a] @ R[b]):7.3f}" for b in others))

    print("\n[D] concentration on shrink-removed residuals, normalized by base group energy")
    L0 = layers[len(layers) // 2]
    groups = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    neurons = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    for m in others:
        gh, gn, tn, cu, cd = [], [], [], [], []
        for L in layers:
            def rel(t):
                W, Bm = T[(m, L, t)], T[(base, L, t)]
                a = float((W * Bm).sum() / (Bm * Bm).sum())
                return group_energy(W - a * Bm, t) / (group_energy(Bm, t) + 1e-12)
            gh += [gini(rel(t)) for t in groups]
            e = {t: rel(t) for t in neurons}
            gn += [gini(v) for v in e.values()]
            tn += [top_share(v) for v in e.values()]
            cu.append(np.corrcoef(e["mlp.gate_proj"], e["mlp.up_proj"])[0, 1])
            cd.append(np.corrcoef(e["mlp.gate_proj"], e["mlp.down_proj"])[0, 1])
        print(f"  {short(m):22s} head gini {np.mean(gh):.3f}  neuron gini {np.mean(gn):.3f} top10% {np.mean(tn):.3f}  gate~up {np.mean(cu):.3f} gate~down {np.mean(cd):.3f}")

    print("\n[E] same heads/neurons across models after shrink removal (relative residual energy, correlation)")
    prof = {}
    for m in others:
        hs, ns = [], []
        for L in layers:
            for t in groups + neurons:
                W, Bm = T[(m, L, t)], T[(base, L, t)]
                a = float((W * Bm).sum() / (Bm * Bm).sum())
                v = group_energy(W - a * Bm, t) / (group_energy(Bm, t) + 1e-12)
                (hs if t in groups else ns).append(v / v.mean())
        prof[m] = (np.concatenate(hs), np.concatenate(ns))
    print("  neuron profile correlation")
    print("  " + " " * 16 + " ".join(f"{short(m)[:7]:>7s}" for m in others))
    for a in others:
        print(f"  {short(a)[:16]:16s}" + " ".join(f"{np.corrcoef(prof[a][1], prof[b][1])[0, 1]:7.3f}" for b in others))

    print("\n[F] clade centroid as shared base: mean distance to members, 1-bit proxy error, vs best real member and vs Base")
    for cname, members in CLADES[name].items():
        members = [m for m in members if m in models]
        if len(members) < 2:
            continue
        def stats(ref_T):
            d, e = [], []
            for m in members:
                for L in layers:
                    for t in MATS:
                        W, Rf = T[(m, L, t)], ref_T(L, t)
                        d.append(np.linalg.norm(W - Rf) / np.linalg.norm(W))
                        e.append(one_bit_err(W, Rf))
            return float(np.mean(d)), float(np.mean(e))
        cen = stats(lambda L, t: np.mean([T[(m, L, t)] for m in members], 0))
        bas = stats(lambda L, t: T[(base, L, t)])
        best = min(((stats(lambda L, t, r=r: T[(r, L, t)]), r) for r in members), key=lambda x: x[0][0])
        print(f"  {cname:16s} n={len(members)}  centroid d={cen[0]:.3f} 1bit={cen[1]:.3f} | best member {short(best[1])[:14]:14s} d={best[0][0]:.3f} 1bit={best[0][1]:.3f} | Base d={bas[0]:.3f} 1bit={bas[1]:.3f}")

    print("\n[G] weight tails: r90 of W, top-1% energy share, Q4 block-32 relative error (mean of q_proj and gate_proj at mid layer)")
    for m in models:
        rr, tail, qe, q8 = [], [], [], []
        for t in ["self_attn.q_proj", "mlp.gate_proj"]:
            W = T[(m, L0, t)]
            rr.append(r90(W))
            e = np.sort((W ** 2).ravel())[::-1]
            tail.append(float(e[: max(1, len(e) // 100)].sum() / e.sum()))
            qe.append(quant_err(W, 4))
            q8.append(quant_err(W, 8))
        print(f"  {short(m):22s} r90 q {rr[0]:4d} gate {rr[1]:4d}  top1% energy q {tail[0]:.3f} gate {tail[1]:.3f}  Q4 err q {qe[0]:.4f} gate {qe[1]:.4f}  Q8 err q {q8[0]:.5f} gate {q8[1]:.5f}")


if __name__ == "__main__":
    import os
    os.makedirs(atlas.CACHE, exist_ok=True)
    for name in sys.argv[1:] or FAMILIES:
        run(name)
