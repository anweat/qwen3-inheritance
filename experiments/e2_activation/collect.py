# Run one model over the corpus. Collect attention, neuron activation statistics, residual samples,
# output samples, and a full decomposition of each target logit into embedding/heads/neurons.
import argparse, os, sys, time
import numpy as np
import torch
from common import DEVICE, corpus, load, tokenizer

HEADS = 16
HEAD_DIM = 128


def collect(name, out_dir, per_cat=30, check=False):
    tok = tokenizer()
    model = load(name)
    segs = corpus(tok, per_cat)
    layers = model.model.layers
    L, inter, hidden = len(layers), layers[0].mlp.gate_proj.out_features, layers[0].mlp.gate_proj.in_features
    W_U = model.lm_head.weight.detach()
    W_mean = W_U.mean(0)
    O = [l.self_attn.o_proj.weight.detach() for l in layers]
    D = [l.mlp.down_proj.weight.detach() for l in layers]
    norm_w = model.model.norm.weight.detach()

    cap = {}
    hooks = []
    for i, l in enumerate(layers):
        hooks.append(l.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("head_in", i), a[0].detach())))
        hooks.append(l.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("act", i), a[0].detach())))
        hooks.append(l.register_forward_pre_hook(lambda m, a, kw=None, i=i: cap.__setitem__(("in", i), a[0].detach()), with_kwargs=False))
    hooks.append(model.model.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("final", a[0].detach())))

    cats = list(segs)
    S = {}
    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.float64)
    # Per-category accumulators.
    for c in cats:
        S[c] = dict(n=0, n_tgt=0, a1=zeros(L, inter), a2=zeros(L, inter), a4=zeros(L, inter), fire=zeros(L, inter),
                    centroid=zeros(L, inter), centroid_n=0,
                    dla_neuron=zeros(L, inter), dla_neuron_s=zeros(L, inter), dla_head=zeros(L, HEADS), dla_head_s=zeros(L, HEADS),
                    dla_layer=zeros(L, 2), dla_embed=0.0, dla_total=0.0, att_ent=zeros(L, HEADS), att_dist=zeros(L, HEADS),
                    att_self=zeros(L, HEADS), att_n=zeros(L, HEADS), rms_in=zeros(L + 1), logit_err=0.0, correct=0, ce=0.0)
    res_samples = {i: [] for i in range(L + 1)}
    res_tags, logit_samples, att_samples = [], [], []
    rng = np.random.default_rng(0)
    freqs = torch.fft.rfftfreq(128)

    t0 = time.time()
    for c in cats:
        s = S[c]
        for si, ids in enumerate(segs[c]):
            x = torch.tensor([ids], device=DEVICE)
            T = x.shape[1]
            with torch.no_grad():
                out = model(x, output_attentions=True, use_cache=False)
            logits = out.logits[0]
            s["n"] += T
            # Thresholds use the layer-wide rms of neuron activations so firing rates compare within a layer.
            for i in range(L):
                a = cap[("act", i)][0]
                s["a1"][i] += a.abs().sum(0).cpu().double()
                s["a2"][i] += (a ** 2).sum(0).cpu().double()
                s["a4"][i] += (a ** 4).sum(0).cpu().double()
                thr = a.pow(2).mean().sqrt()
                s["fire"][i] += (a.abs() > thr).sum(0).cpu().double()
                if T == 128:
                    spec = torch.fft.rfft((a - a.mean(0, keepdim=True)).cpu(), dim=0).abs() ** 2
                    s["centroid"][i] += ((spec * freqs[:, None]).sum(0) / (spec.sum(0) + 1e-12)).double()
                A = out.attentions[i][0]  # heads, T, T
                ent = -(A * (A + 1e-12).log()).sum(-1)
                pos = torch.arange(T, device=A.device)
                dist = (A * (pos[None, :] - pos[:, None]).abs().float()).sum(-1)
                diag = A.diagonal(dim1=-2, dim2=-1)
                s["att_ent"][i] += ent.sum(-1).cpu().double()
                s["att_dist"][i] += dist.sum(-1).cpu().double()
                s["att_self"][i] += diag.sum(-1).cpu().double()
                s["att_n"][i] += T
                s["rms_in"][i] += cap[("in", i)][0].pow(2).mean(-1).sum().cpu().double()
            s["rms_in"][L] += cap["final"][0].pow(2).mean(-1).sum().cpu().double()
            if T == 128:
                s["centroid_n"] += 1

            # Logit attribution: target is the true next token.
            if T > 1:
                final = cap["final"][0]
                rms = final.pow(2).mean(-1, keepdim=True).add(model.model.norm.variance_epsilon).sqrt()
                tgt = x[0, 1:]
                d = (W_U[tgt] - W_mean) * norm_w / rms[:-1]  # (T-1, H)
                total = (d * final[:-1]).sum(-1)
                recon = (d * model.model.embed_tokens(x)[0, :-1]).sum(-1)
                s["dla_embed"] += recon.abs().sum().item()
                acc = recon.clone()
                for i in range(L):
                    hi = cap[("head_in", i)][0, :-1]
                    dO = d @ O[i]  # (T-1, 2048)
                    per = (dO * hi).reshape(T - 1, HEADS, HEAD_DIM).sum(-1)
                    a = cap[("act", i)][0, :-1]
                    pn = a * (d @ D[i])
                    s["dla_head"][i] += per.abs().sum(0).cpu().double()
                    s["dla_head_s"][i] += per.sum(0).cpu().double()
                    s["dla_neuron"][i] += pn.abs().sum(0).cpu().double()
                    s["dla_neuron_s"][i] += pn.sum(0).cpu().double()
                    s["dla_layer"][i, 0] += per.sum(-1).abs().sum().cpu().double()
                    s["dla_layer"][i, 1] += pn.sum(-1).abs().sum().cpu().double()
                    acc = acc + per.sum(-1) + pn.sum(-1)
                s["logit_err"] += (acc - total).abs().sum().item()
                s["dla_total"] += total.abs().sum().item()
                real = logits[:-1].gather(1, tgt[:, None])[:, 0] - logits[:-1].mean(-1)
                s.setdefault("real_err", 0.0)
                s["real_err"] += (real - total).abs().sum().item()
                s["n_tgt"] += T - 1
                s["correct"] += (logits[:-1].argmax(-1) == tgt).sum().item()
                s["ce"] += torch.nn.functional.cross_entropy(logits[:-1], tgt, reduction="sum").item()

            # Fixed samples for model-to-model comparison.
            if si < 12:
                idx = np.arange(T - 1)[:: max(1, (T - 1) // 16)][:16]
                for i in range(L):
                    res_samples[i].append(cap[("in", i)][0, idx].float().cpu().numpy())
                res_samples[L].append(cap["final"][0, idx].float().cpu().numpy())
                res_tags.extend([c] * len(idx))
                logit_samples.append(torch.log_softmax(logits[idx], -1).half().cpu().numpy())
                if si < 4:
                    att_samples.append(dict(cat=c, seg=si, A=np.stack([out.attentions[i][0].half().cpu().numpy() for i in (2, 14, 26)])))
            cap.clear()
        print(f"  {name} {c}: {len(segs[c])} segs, {s['n']} tokens, acc {s['correct']/max(1,s['n_tgt']):.3f}, "
              f"recon err {s['logit_err']/max(1,s['dla_total']):.2e}, vs real {s.get('real_err',0)/max(1,s['dla_total']):.2e}, {time.time()-t0:.0f}s", flush=True)
        if check:
            break
    for h in hooks:
        h.remove()
    os.makedirs(out_dir, exist_ok=True)
    arrs = {f"{c}/{k}": (v.numpy() if torch.is_tensor(v) else np.array(v)) for c in S for k, v in S[c].items()}
    np.savez_compressed(os.path.join(out_dir, f"{name}.stats.npz"), **arrs)
    np.savez(os.path.join(out_dir, f"{name}.samples.npz"), tags=np.array(res_tags),
             logits=np.concatenate(logit_samples) if logit_samples else np.zeros(0),
             **{f"res{i}": np.concatenate(v) for i, v in res_samples.items() if v})
    np.save(os.path.join(out_dir, f"{name}.att.npy"), np.array(att_samples, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cat", type=int, default=30)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for m in a.models:
        collect(m, a.out, a.per_cat, a.check)
