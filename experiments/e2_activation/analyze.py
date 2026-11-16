# Analyze collected runs: Qwen3 structure from Base, attribution vs plasticity, and cross-model divergence.
import argparse, os
import numpy as np
import torch
from common import read_decoder
from scipy.stats import spearmanr

CATS = ["code", "en_doc", "zh_doc", "spoken_en", "spoken_zh", "chat"]
L = 28
INTER, HEADS = 3072, 16
MLP = ["gate_proj", "up_proj", "down_proj"]
ATT = ["q_proj", "k_proj", "v_proj", "o_proj"]


def stats(d, name):
    return dict(np.load(os.path.join(d, f"{name}.stats.npz")))


def agg(S, key, cats=CATS, per="n"):
    tot = sum(S[f"{c}/{key}"] for c in cats)
    return tot / max(1, sum(float(S[f"{c}/{per}"]) for c in cats))


def structure(S):
    print("\n[A1] per-layer structure on Base (all categories). rms=residual stream rms; DLA=mean |logit contribution|")
    n = sum(float(S[f"{c}/n"]) for c in CATS)
    rms = np.sqrt(agg(S, "rms_in"))
    att_n = sum(S[f"{c}/att_n"] for c in CATS)
    ent = sum(S[f"{c}/att_ent"] for c in CATS) / att_n
    dist = sum(S[f"{c}/att_dist"] for c in CATS) / att_n
    self_ = sum(S[f"{c}/att_self"] for c in CATS) / att_n
    dla_l = agg(S, "dla_layer", per="n_tgt")
    fire = agg(S, "fire")
    a2, a4 = agg(S, "a2"), agg(S, "a4")
    kurt = a4 / (a2 ** 2 + 1e-18)
    print("  L   rms   ent(head min/mean/max)  dist(min/mean/max)  self   DLA attn  DLA mlp  fire(p5/50/95)   kurt(p50/p95)")
    for i in range(0, L, 3):
        print(f"  {i:2d} {rms[i]:6.2f}   {ent[i].min():4.2f}/{ent[i].mean():4.2f}/{ent[i].max():4.2f}   {dist[i].min():5.1f}/{dist[i].mean():5.1f}/{dist[i].max():5.1f}"
              f"  {self_[i].mean():5.3f}  {dla_l[i,0]:7.2f} {dla_l[i,1]:7.2f}   {np.percentile(fire[i],5):.3f}/{np.percentile(fire[i],50):.3f}/{np.percentile(fire[i],95):.3f}   {np.percentile(kurt[i],50):5.1f}/{np.percentile(kurt[i],95):6.1f}")
    print(f"  final rms {rms[L]:.2f}; embed DLA {sum(float(S[f'{c}/dla_embed']) for c in CATS)/sum(float(S[f'{c}/n_tgt']) for c in CATS):.2f}; mean |total logit| {sum(float(S[f'{c}/dla_total']) for c in CATS)/sum(float(S[f'{c}/n_tgt']) for c in CATS):.2f}")

    print("\n[A2] attribution by category: share of summed |DLA| by depth third, and attn vs mlp")
    print("  cat          tokens acc    ce   | attn: L0-9 L10-18 L19-27 | mlp: L0-9 L10-18 L19-27 | attn/mlp")
    for c in CATS:
        d = S[f"{c}/dla_layer"] / max(1, float(S[f"{c}/n_tgt"]))
        tot = d.sum()
        third = lambda col: [d[a:b, col].sum() / tot for a, b in ((0, 10), (10, 19), (19, 28))]
        print(f"  {c:11s} {int(S[f'{c}/n']):6d} {float(S[f'{c}/correct'])/max(1,float(S[f'{c}/n_tgt'])):.3f} {float(S[f'{c}/ce'])/max(1,float(S[f'{c}/n_tgt'])):5.2f} | "
              + " ".join(f"{x:5.3f}" for x in third(0)) + " | " + " ".join(f"{x:5.3f}" for x in third(1)) + f" | {d[:,0].sum()/d[:,1].sum():.2f}")

    print("\n[A3] concentration of output attribution: share of per-layer |DLA| from top 1%/10% neurons and top head")
    neur = agg(S, "dla_neuron", per="n_tgt")
    head = agg(S, "dla_head", per="n_tgt")
    for i in (0, 7, 14, 21, 27):
        v = np.sort(neur[i])[::-1]
        h = np.sort(head[i])[::-1]
        print(f"  L{i:2d} neurons top1% {v[:INTER//100].sum()/v.sum():.3f} top10% {v[:INTER//10].sum()/v.sum():.3f} | heads top1 {h[0]/h.sum():.3f} top4 {h[:4].sum()/h.sum():.3f}")

    print("\n[A4] temporal frequency of neuron activation: spectral centroid in cycles/token (0.5 = changes every token), and firing rate; per-layer medians")
    cats128 = [c for c in CATS if float(S[f"{c}/centroid_n"]) > 0]
    cent = sum(S[f"{c}/centroid"] for c in cats128) / sum(float(S[f"{c}/centroid_n"]) for c in cats128)
    for i in (0, 7, 14, 21, 27):
        print(f"  L{i:2d} centroid p5/50/95 {np.percentile(cent[i],5):.3f}/{np.percentile(cent[i],50):.3f}/{np.percentile(cent[i],95):.3f}  corr(centroid, log fire) {spearmanr(cent[i], fire[i])[0]:.3f}  corr(centroid, log dla) {spearmanr(cent[i], neur[i])[0]:.3f}  corr(fire, log dla) {spearmanr(fire[i], neur[i])[0]:.3f}")
    return dict(neur=neur, fire=fire, cent=cent, head=head, ent=ent, dist=dist, kurt=kurt)


def residual_energy(W, B, kind, name):
    a = float((W * B).sum() / (B * B).sum())
    R = W - a * B
    if kind == "neuron":
        return (R ** 2).sum(1 if name != "down_proj" else 0).numpy()
    if name == "o_proj":
        return (R.reshape(R.shape[0], -1, 128) ** 2).sum((0, 2)).numpy()
    return (R.reshape(-1, 128, R.shape[1]) ** 2).sum((1, 2)).numpy()


def plasticity(base_sd, sd):
    pn, ph = np.zeros((L, INTER)), np.zeros((L, HEADS))
    for i in range(L):
        for t in MLP:
            B, W = base_sd[f"model.layers.{i}.mlp.{t}.weight"], sd[f"model.layers.{i}.mlp.{t}.weight"]
            pn[i] += residual_energy(W, B, "neuron", t)
        for t in ATT:
            B, W = base_sd[f"model.layers.{i}.self_attn.{t}.weight"], sd[f"model.layers.{i}.self_attn.{t}.weight"]
            e = residual_energy(W, B, "head", t)
            if len(e) == 8:
                e = np.repeat(e, 2) / 2
            ph[i] += e
    return pn, ph


def link(name, base_sd, sd, A, out_dir):
    pn, ph = plasticity(base_sd, sd)
    np.savez(os.path.join(out_dir, f"plastic.{name}.npz"), pn=pn, ph=ph)
    print(f"\n[B] {name}: does plasticity align with output attribution / activity? per-layer Spearman with neuron residual energy, controlling nothing; rows sampled by depth")
    base_e = np.stack([sum(((base_sd[f"model.layers.{i}.mlp.{t}.weight"] ** 2).sum(1 if t != "down_proj" else 0)).numpy() for t in MLP) for i in range(L)])
    print("  L   vs|DLA|  vs fire  vs centroid vs kurt  vs base energy | head: vs|DLA| vs ent  vs dist")
    rows = []
    for i in range(L):
        r = [spearmanr(pn[i], A["neur"][i])[0], spearmanr(pn[i], A["fire"][i])[0], spearmanr(pn[i], A["cent"][i])[0], spearmanr(pn[i], A["kurt"][i])[0], spearmanr(pn[i], base_e[i])[0],
             spearmanr(ph[i], A["head"][i])[0], spearmanr(ph[i], A["ent"][i])[0], spearmanr(ph[i], A["dist"][i])[0]]
        rows.append(r)
        if i % 4 == 2 or i == 27:
            print(f"  {i:2d} " + " ".join(f"{x:7.3f}" for x in r[:5]) + "   | " + " ".join(f"{x:7.3f}" for x in r[5:]))
    print("  mean" + " ".join(f"{x:7.3f}" for x in np.mean(rows, 0)))
    # Partial correlation of plasticity with attribution, controlling base energy.
    pc = []
    for i in range(L):
        z = np.argsort(np.argsort(base_e[i])).astype(float)
        x = np.argsort(np.argsort(pn[i])).astype(float)
        y = np.argsort(np.argsort(A["neur"][i])).astype(float)
        def res(v):
            X = np.stack([z, np.ones_like(z)], 1)
            return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        pc.append(np.corrcoef(res(x), res(y))[0, 1])
    print(f"  partial rank corr (plasticity, |DLA| | base energy): mean {np.mean(pc):.3f}, min {np.min(pc):.3f}, max {np.max(pc):.3f}")
    return pn, ph


def cross(names, d):
    print("\n[C] cross-model divergence on identical text")
    samples = {n: np.load(os.path.join(d, f"{n}.samples.npz")) for n in names}
    def cka(a, b):
        a = a - a.mean(0); b = b - b.mean(0)
        return float((a.T @ b).__pow__(2).sum() / np.sqrt((a.T @ a).__pow__(2).sum() * (b.T @ b).__pow__(2).sum()))
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    print("  linear CKA by depth (residual stream entering each layer; final=before norm)")
    print("  " + " " * 34 + " ".join(f"L{i:<5d}" for i in (0, 4, 8, 12, 16, 20, 24, 28)))
    for a, b in pairs:
        print(f"  {a[5:]+' vs '+b[5:]:34s}" + " ".join(f"{cka(samples[a][f'res{i}'], samples[b][f'res{i}']):6.3f}" for i in (0, 4, 8, 12, 16, 20, 24, 28)))
    print("  output divergence by category: mean JS (bits) and top-1 agreement")
    tags = samples[names[0]]["tags"]
    print("  " + " " * 34 + " ".join(f"{c:>13s}" for c in CATS))
    for a, b in pairs:
        pa, pb = np.exp(samples[a]["logits"].astype(np.float32)), np.exp(samples[b]["logits"].astype(np.float32))
        m = (pa + pb) / 2 + 1e-12
        js = 0.5 * ((pa * (np.log2(pa + 1e-12) - np.log2(m))).sum(1) + (pb * (np.log2(pb + 1e-12) - np.log2(m))).sum(1))
        ag = pa.argmax(1) == pb.argmax(1)
        print(f"  {a[5:]+' vs '+b[5:]:34s}" + " ".join(f"{js[tags==c].mean():6.3f}/{ag[tags==c].mean():6.3f}" for c in CATS))
    print("  attention: mean JS (bits) per head at L2/14/26, min-max across heads")
    att = {n: np.load(os.path.join(d, f"{n}.att.npy"), allow_pickle=True) for n in names}
    for a, b in pairs:
        J = []
        for x, y in zip(att[a], att[b]):
            p, q = x["A"].astype(np.float32), y["A"].astype(np.float32)
            m = (p + q) / 2 + 1e-9
            js = 0.5 * ((p * (np.log2(p + 1e-9) - np.log2(m))).sum(-1) + (q * (np.log2(q + 1e-9) - np.log2(m))).sum(-1))
            J.append(js.mean(-1))  # layers, heads
        J = np.mean(J, 0)
        print(f"  {a[5:]+' vs '+b[5:]:34s}" + " | ".join(f"L{l}: {J[i].mean():.3f} ({J[i].min():.3f}-{J[i].max():.3f})" for i, l in enumerate((2, 14, 26))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--models", nargs="*", default=["Qwen3-0.6B", "Qwen3-ASR-0.6B"])
    a = ap.parse_args()
    S = stats(a.dir, "Qwen3-0.6B-Base")
    A = structure(S)
    if a.models:
        base_sd = read_decoder("Qwen3-0.6B-Base")
        for m in a.models:
            link(m, base_sd, read_decoder(m), A, a.dir)
        cross(["Qwen3-0.6B-Base"] + a.models, a.dir)
