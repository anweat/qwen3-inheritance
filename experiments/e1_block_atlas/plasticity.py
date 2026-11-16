# Is "which neurons/heads move" a shared property across functions, beyond base-norm effects?
# Compares residual-energy profiles with a partial correlation that controls for log base energy,
# and with shuffled-layer controls.
import sys
import numpy as np
from atlas import FAMILIES, MATS, group_energy, load_family

GROUPS = {"neuron": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
          "head": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]}
PANEL = {"0.6B": ["Qwen3-0.6B", "Qwen3-Reranker-0.6B", "Qwen3-Embedding-0.6B", "Qwen3Guard-Gen-0.6B", "Qwen3-ASR-0.6B", "Qwen3-ForcedAligner-0.6B", "Qwen3-TTS-12Hz-0.6B-Base"],
         "1.7B": ["Qwen3-1.7B", "Qwen3-VL-2B-Instruct", "Qwen3-ASR-1.7B", "Qwen3-TTS-12Hz-1.7B-Base"]}


def short(m):
    return m.replace("Qwen3", "").replace("-12Hz", "").replace("-0.6B", "").replace("-1.7B", "").replace("-2B", "").strip("-") or "instruct"


def residual_energy(T, m, base, L, t):
    W, B = T[(m, L, t)], T[(base, L, t)]
    a = float((W * B).sum() / (B * B).sum())
    return group_energy(W - a * B, t)


def partial(x, y, z):
    def res(v, z):
        A = np.stack([z, np.ones_like(z)], 1)
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return float(np.corrcoef(res(x, z), res(y, z))[0, 1])


for name in sys.argv[1:] or FAMILIES:
    fam = FAMILIES[name]
    T = load_family(fam)
    base, layers = fam["base"], fam["layers"]
    ms = PANEL[name]
    for kind, mats in GROUPS.items():
        print(f"\n=== {name} {kind}: raw corr / partial corr controlling log base energy (log residual energy, per-matrix z-scored, pooled) ===")
        P, Z = {}, []
        for m in ms:
            vs = []
            for L in layers:
                for t in mats:
                    v = np.log(residual_energy(T, m, base, L, t) + 1e-12)
                    vs.append((v - v.mean()) / v.std())
            P[m] = np.concatenate(vs)
        for L in layers:
            for t in mats:
                z = np.log(group_energy(T[(base, L, t)], t) + 1e-12)
                Z.append((z - z.mean()) / z.std())
        Z = np.concatenate(Z)
        print("  corr with log base energy: " + "  ".join(f"{short(m)} {np.corrcoef(P[m], Z)[0,1]:.2f}" for m in ms))
        print("  " + " " * 16 + " ".join(f"{short(m)[:13]:>13s}" for m in ms))
        for a in ms:
            print(f"  {short(a)[:16]:16s}" + " ".join(f"{np.corrcoef(P[a], P[b])[0,1]:6.2f}/{partial(P[a], P[b], Z):6.2f}" for b in ms))
