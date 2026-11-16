# Causal tests on text: spike removal, attention window with/without sink, neuron and head ablation,
# real drift vs isotropic noise of equal size, and interpolation between models.
import argparse, copy, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from common import DEVICE, corpus, load, read_decoder, tokenizer

MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
L = 28
INTER = 3072


def evaluate(model, segs, ref=None, mask_fn=None):
    """Returns per-category CE on true next tokens over positions >= 1, and JS (bits) against ref logits."""
    out = {}
    logs = {}
    for c, ss in segs.items():
        ce, n, js, jn = 0., 0, 0., 0
        for si, ids in enumerate(ss):
            x = torch.tensor([ids], device=DEVICE)
            kw = {}
            if mask_fn is not None:
                kw["attention_mask"] = mask_fn(x.shape[1])
            with torch.no_grad():
                lg = model(x, use_cache=False, **kw).logits[0]
            tgt = x[0, 2:]
            ce += F.cross_entropy(lg[1:-1], tgt, reduction="sum").item()
            n += len(tgt)
            lp = torch.log_softmax(lg[1::4], -1)
            logs[(c, si)] = lp.half().cpu()
            if ref is not None:
                q = ref[(c, si)].to(DEVICE).float()
                pm, qm = lp.exp(), q.exp()
                m = ((pm + qm) / 2).clamp_min(1e-12).log()
                j = 0.5 * ((pm * (lp - m)).sum(-1) + (qm * (q - m)).sum(-1)) / np.log(2)
                js += j.sum().item()
                jn += len(j)
        out[c] = (ce / max(1, n), js / max(1, jn))
    return out, logs


def line(label, r, base=None):
    cats = list(r)
    w = {c: 1 for c in cats}
    print(f"  {label:40s}" + " ".join(f"{r[c][0]:6.3f}" + (f"/{r[c][1]:5.3f}" if r[c][1] else "") for c in cats), flush=True)


def header(segs):
    print("  " + " " * 40 + " ".join(f"{c:>12s}" for c in segs) + "   (CE / JS bits vs intact)", flush=True)


def window_mask(k, keep_sink):
    def fn(T):
        m = torch.full((T, T), float("-inf"), device=DEVICE, dtype=torch.float32)
        for t in range(T):
            m[t, max(0, t - k + 1):t + 1] = 0
            if keep_sink:
                m[t, 0] = 0
        return m[None, None]
    return fn


def spike(args, segs):
    model = load(args.model)
    ref_r, ref = evaluate(model, segs)
    header(segs)
    line("intact", ref_r)
    # Remove the first-token spike: zero the first position's L2 MLP output.
    h = model.model.layers[2].mlp.register_forward_hook(lambda m, a, o: torch.cat([torch.zeros_like(o[:, :1]), o[:, 1:]], 1))
    r, _ = evaluate(model, segs, ref)
    line("no spike (zero L2 MLP at pos 0)", r)
    h.remove()
    # Plant a spike on every token.
    h = model.model.layers[2].mlp.register_forward_hook(lambda m, a, o: o + (o[:, :1] * 0 + o[:, :1]).detach())
    r, _ = evaluate(model, segs, ref)
    line("first-token L2 MLP output added to all", r)
    h.remove()
    # Window attention.
    for k in (64, 32, 16, 8, 4):
        for keep in (True, False):
            r, _ = evaluate(model, segs, ref, window_mask(k, keep))
            line(f"attention window {k}, {'keep' if keep else 'drop'} first token", r)


def ablate(args, segs):
    model = load(args.model)
    ref_r, ref = evaluate(model, segs)
    header(segs)
    line("intact", ref_r)
    d = np.load(os.path.join(args.dir, f"{args.model}.stats.npz"))
    cats = [c for c in segs]
    n = sum(float(d[f"{c}/n_tgt"]) for c in cats)
    dla = sum(d[f"{c}/dla_neuron"] for c in cats) / n
    fire = sum(d[f"{c}/fire"] for c in cats) / n
    plastic = {}
    for p in args.plastic:
        z = np.load(os.path.join(args.dir, f"plastic.{p}.npz"))
        plastic[p] = z
    rng = np.random.default_rng(0)
    groups = {"random": lambda i: rng.permutation(INTER), "top |DLA|": lambda i: np.argsort(-dla[i]), "low |DLA|": lambda i: np.argsort(dla[i]),
              "top firing": lambda i: np.argsort(-fire[i])}
    for p, z in plastic.items():
        groups[f"most plastic ({p[5:]})"] = lambda i, z=z: np.argsort(-z["pn"][i])
        groups[f"least plastic ({p[5:]})"] = lambda i, z=z: np.argsort(z["pn"][i])
    # Plasticity contrast with |DLA| held fixed: pick within each |DLA| decile.
    def stratified(i, z, least):
        q = np.argsort(np.argsort(dla[i])) * 10 // INTER
        rank = np.zeros(INTER)
        for b in range(10):
            m = q == b
            v = z["pn"][i][m] * (1 if least else -1)
            rank[m] = np.argsort(np.argsort(v)) / m.sum()
        return np.argsort(rank)
    for p, z in plastic.items():
        groups[f"least plastic, |DLA|-matched ({p[5:]})"] = lambda i, z=z: stratified(i, z, True)
        groups[f"most plastic, |DLA|-matched ({p[5:]})"] = lambda i, z=z: stratified(i, z, False)
    for frac in (0.1, 0.3):
        for gname, order in groups.items():
            hooks = []
            for i, l in enumerate(model.model.layers):
                idx = torch.tensor(order(i)[: int(INTER * frac)].copy(), device=DEVICE)
                def pre(m, a, idx=idx):
                    x = a[0].clone()
                    x[:, :, idx] = 0
                    return (x,)
                hooks.append(l.mlp.down_proj.register_forward_pre_hook(pre))
            r, _ = evaluate(model, segs, ref)
            line(f"zero {int(frac*100)}% neurons: {gname}", r)
            for h in hooks:
                h.remove()


def noise(args, segs):
    base_sd = read_decoder(args.model)
    model = load(args.model)
    ref_r, ref = evaluate(model, segs)
    header(segs)
    line(f"intact {args.model}", ref_r)
    keys = [f"model.layers.{i}.{t}.weight" for i in range(L) for t in MATS]
    params = dict(model.named_parameters())

    def apply(fn):
        with torch.no_grad():
            for k in keys:
                params[k].copy_(fn(k, base_sd[k]).to(DEVICE))

    drift = {}
    for other in args.others:
        sd = read_decoder(other)
        d = float(np.mean([(sd[k] - base_sd[k]).norm().item() / base_sd[k].norm().item() for k in keys]))
        drift[other] = d
        apply(lambda k, B, sd=sd: sd[k])
        r, _ = evaluate(model, segs, ref)
        line(f"all layer matrices from {other[5:]} d={d:.3f}", r)
        # Same matrices but real drift with shrink removed (only the residual).
        def resid(k, B, sd=sd):
            a = (sd[k] * B).sum() / (B * B).sum()
            return B + (sd[k] - a * B)
        apply(resid)
        r, _ = evaluate(model, segs, ref)
        line(f"  residual only (no shrink)", r)
        def shrink(k, B, sd=sd):
            a = (sd[k] * B).sum() / (B * B).sum()
            return a * B
        apply(shrink)
        r, _ = evaluate(model, segs, ref)
        line(f"  shrink only (alpha*B)", r)
        torch.manual_seed(0)
        def iso(k, B, sd=sd):
            a = (sd[k] * B).sum() / (B * B).sum()
            res = sd[k] - a * B
            nz = torch.randn_like(B)
            return B + nz / nz.norm() * res.norm()
        apply(iso)
        r, _ = evaluate(model, segs, ref)
        line(f"  isotropic noise, same residual norm", r)
        def shuffled(k, B, sd=sd):
            a = (sd[k] * B).sum() / (B * B).sum()
            res = (sd[k] - a * B).flatten()
            return B + res[torch.randperm(res.numel())].reshape(B.shape)
        apply(shuffled)
        r, _ = evaluate(model, segs, ref)
        line(f"  shuffled residual (same values, positions lost)", r)
        def band(k, B, sd=sd, frac=0.2):
            a = (sd[k] * B).sum() / (B * B).sum()
            res = sd[k] - a * B
            U, S, Vh = torch.linalg.svd(B.to(torch.float32), full_matrices=False)
            k_ = max(1, int(len(S) * frac))
            P = U[:, :k_] @ U[:, :k_].T @ res @ Vh[:k_].T @ Vh[:k_]
            return a * B + P
        apply(band)
        r, _ = evaluate(model, segs, ref)
        line(f"  shrink + residual in Base top-20% singular band", r)
        # Control: same perturbation size as the top-band residual, but isotropic, then the complementary band.
        def iso_top(k, B, sd=sd, frac=0.2):
            a = (sd[k] * B).sum() / (B * B).sum()
            res = sd[k] - a * B
            U, S, Vh = torch.linalg.svd(B.to(torch.float32), full_matrices=False)
            k_ = max(1, int(len(S) * frac))
            P = U[:, :k_] @ U[:, :k_].T @ res @ Vh[:k_].T @ Vh[:k_]
            nz = torch.randn_like(B)
            return a * B + nz / nz.norm() * P.norm()
        apply(iso_top)
        r, _ = evaluate(model, segs, ref)
        line(f"  isotropic noise, norm of top-band residual", r)
        def bottom(k, B, sd=sd, frac=0.5):
            a = (sd[k] * B).sum() / (B * B).sum()
            res = sd[k] - a * B
            U, S, Vh = torch.linalg.svd(B.to(torch.float32), full_matrices=False)
            k_ = int(len(S) * frac)
            P = U[:, k_:] @ U[:, k_:].T @ res @ Vh[k_:].T @ Vh[k_:]
            return a * B + P
        apply(bottom)
        r, _ = evaluate(model, segs, ref)
        line(f"  shrink + residual in Base bottom-50% band", r)
        apply(lambda k, B: B)


def interp(args, segs):
    a, b = args.others[0], args.others[1]
    sa, sb = read_decoder(a), read_decoder(b)
    model = load(a)
    params = dict(model.named_parameters())
    keys = [k for k in sa if k in params]
    ref_r, ref = evaluate(model, segs)
    header(segs)
    for t in (0, .1, .25, .5, .75, .9, 1):
        with torch.no_grad():
            for k in keys:
                params[k].copy_(((1 - t) * sa[k] + t * sb[k]).to(DEVICE))
        r, _ = evaluate(model, segs, ref)
        line(f"{t:.2f}: {a[5:]} -> {b[5:]}", r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["spike", "ablate", "noise", "interp"])
    ap.add_argument("--model", default="Qwen3-0.6B-Base")
    ap.add_argument("--others", nargs="*", default=["Qwen3-0.6B", "Qwen3-ASR-0.6B"])
    ap.add_argument("--plastic", nargs="*", default=["Qwen3-0.6B", "Qwen3-ASR-0.6B"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--per-cat", type=int, default=10)
    a = ap.parse_args()
    tok = tokenizer()
    segs = corpus(tok, a.per_cat)
    print(f"\n### {a.test} on {a.model}", flush=True)
    globals()[a.test](a, segs)
