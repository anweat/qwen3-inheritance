# Where in a matrix does drift live? Singular-value bands of the base ("low" vs "high" frequency)
# and weight-magnitude bands, on shrink-removed residuals, with an iid-noise null.
import sys
import numpy as np
from atlas import FAMILIES, MATS, load_family

BANDS = [(0, .01), (.01, .05), (.05, .2), (.2, .5), (.5, 1)]
PANEL = {"0.6B": ["Qwen3-0.6B", "Qwen3Guard-Gen-0.6B", "Qwen3-Embedding-0.6B", "Qwen3-Reranker-0.6B", "Qwen3-ASR-0.6B", "Qwen3-ForcedAligner-0.6B", "Qwen3-TTS-12Hz-0.6B-Base"],
         "1.7B": ["Qwen3-1.7B", "Qwen3-ASR-1.7B", "Qwen3-VL-2B-Instruct", "Qwen3-VL-Embedding-2B", "Qwen3-TTS-12Hz-1.7B-Base"]}


def short(m):
    return m.replace("Qwen3", "").replace("-12Hz", "").replace("-0.6B", "").replace("-1.7B", "").replace("-2B", "").strip("-") or "instruct"


def residual(W, B):
    return W - float((W * B).sum() / (B * B).sum()) * B


def band_shares(R, U, Vt):
    """Share of residual energy in each singular band, input side and joint (band x band)."""
    C = U.T @ R @ Vt.T
    e = C ** 2
    total = (R ** 2).sum()
    n = e.shape[1]
    idx = [(int(a * n), max(int(b * n), int(a * n) + 1)) for a, b in BANDS]
    inp = [e[:, a:b].sum() / total for a, b in idx]
    out = [e[a:b, :].sum() / total for a, b in idx]
    joint = [e[a:b, a:b].sum() / total for a, b in idx]
    return inp, out, joint


def run(name):
    fam = FAMILIES[name]
    T = load_family(fam)
    base, layers = fam["base"], fam["layers"]
    print(f"\n{'='*20} family {name}, layers {layers} {'='*20}")
    svd = {(L, t): np.linalg.svd(T[(base, L, t)], full_matrices=False) for L in layers for t in MATS}
    hdr = " ".join(f"{int(a*100)}-{int(b*100)}%".rjust(8) for a, b in BANDS)

    print("\n[1] residual energy share by Base singular band (input side). Null = band width (matrix-dependent, shown for square-ish case)")
    print("  " + " " * 18 + hdr + "   | joint top-1% / bottom-50%")
    rng = np.random.default_rng(0)
    rows = [("iid noise", None)] + [(short(m), m) for m in PANEL[name]]
    for label, m in rows:
        I, J = [], []
        for (L, t), (U, S, Vt) in svd.items():
            B = T[(base, L, t)]
            R = rng.standard_normal(B.shape, dtype=np.float32) if m is None else residual(T[(m, L, t)], B)
            i, _, j = band_shares(R, U, Vt)
            I.append(i)
            J.append(j)
        I, J = np.mean(I, 0), np.mean(J, 0)
        print(f"  {label[:18]:18s}" + " ".join(f"{x:8.3f}" for x in I) + f"   | {J[0]:.4f} / {J[-1]:.3f}")

    print("\n[2] relative drift inside each band: ||R restricted to band|| / ||B restricted to band||, input side")
    print("  " + " " * 18 + hdr)
    for m in PANEL[name]:
        out = []
        for (L, t), (U, S, Vt) in svd.items():
            B = T[(base, L, t)]
            R = residual(T[(m, L, t)], B)
            C = (U.T @ R @ Vt.T) ** 2
            n = C.shape[1]
            s2 = S ** 2
            row = []
            for a, b in BANDS:
                i, j = int(a * n), max(int(b * n), int(a * n) + 1)
                row.append(np.sqrt(C[:, i:j].sum() / s2[i:j].sum()))
            out.append(row)
        print(f"  {short(m)[:18]:18s}" + " ".join(f"{x:8.3f}" for x in np.mean(out, 0)))

    print("\n[3] by |B| quantile (0-1% = smallest weights): rms delta / overall rms delta, sign-flip rate, exactly-unchanged rate")
    q = [0, .1, .25, .5, .75, .9, .99, 1]
    print("  " + " " * 18 + " ".join(f"q{int(a*100)}-{int(b*100)}".rjust(8) for a, b in zip(q, q[1:])))
    for m in PANEL[name]:
        rm, fl, un = [], [], []
        for L in layers:
            for t in MATS:
                B, W = T[(base, L, t)].ravel(), T[(m, L, t)].ravel()
                edges = np.quantile(np.abs(B), q)
                d = W - B
                overall = np.sqrt((d ** 2).mean())
                r_, f_, u_ = [], [], []
                for a, b in zip(edges, edges[1:]):
                    s = (np.abs(B) >= a) & (np.abs(B) <= b)
                    r_.append(np.sqrt((d[s] ** 2).mean()) / overall)
                    f_.append((np.sign(W[s]) != np.sign(B[s])).mean())
                    u_.append((W[s] == B[s]).mean())
                rm.append(r_); fl.append(f_); un.append(u_)
        print(f"  {short(m)[:12]:12s} rmsΔ " + " ".join(f"{x:8.2f}" for x in np.mean(rm, 0)))
        print(f"  {'':12s} flip " + " ".join(f"{x:8.3f}" for x in np.mean(fl, 0)))
        print(f"  {'':12s} same " + " ".join(f"{x:8.3f}" for x in np.mean(un, 0)))


if __name__ == "__main__":
    for name in sys.argv[1:] or FAMILIES:
        run(name)
