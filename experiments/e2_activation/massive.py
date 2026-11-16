# Separate the massive-activation tokens from ordinary tokens and locate where the spike is created.
import os, sys
import numpy as np
import torch
from common import DEVICE, corpus, load, tokenizer

name = sys.argv[1]
tok = tokenizer()
model = load(name)
segs = corpus(tok, 8)
cap = {}
hooks = []
for i, l in enumerate(model.model.layers):
    hooks.append(l.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("in", i), a[0].detach())))
    hooks.append(l.self_attn.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("attn", i), o[0].detach())))
    hooks.append(l.mlp.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("mlp", i), o.detach())))
    hooks.append(l.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("act", i), a[0].detach())))

rows = {}
top_tokens = []
per_layer = {}
for c, ss in segs.items():
    for ids in ss:
        x = torch.tensor([ids], device=DEVICE)
        with torch.no_grad():
            model(x, use_cache=False, output_attentions=True)
        s = cap[("in", 8)][0].float().cpu().numpy()
        norm = np.linalg.norm(s, axis=1)
        for t, v in enumerate(norm):
            top_tokens.append((float(v), c, t, tok.decode([ids[t]]), float(s[t, np.argmax(np.abs(s[t]))]), int(np.argmax(np.abs(s[t])))))
        for i in range(28):
            a = cap[("in", i)][0].float().cpu().numpy()
            am = cap[("attn", i)][0].float().cpu().numpy()
            mm = cap[("mlp", i)][0].float().cpu().numpy()
            for t in range(a.shape[0]):
                key = "first" if t == 0 else "rest"
                d = per_layer.setdefault((i, key), [0., 0., 0., 0, 0.])
                d[0] += (a[t] ** 2).mean(); d[1] += (am[t] ** 2).mean(); d[2] += (mm[t] ** 2).mean(); d[3] += 1; d[4] += abs(a[t, 35])
top_tokens.sort(reverse=True, key=lambda r: r[0])
print(f"{name}: stream norm at L8 per token. top tokens (norm, category, position, token, max coord value, dim)")
for r in top_tokens[:12]:
    print("  ", f"{r[0]:9.1f}", r[1], r[2], repr(r[3]), f"{r[4]:9.1f}", r[5])
norms = np.array([r[0] for r in top_tokens])
pos = np.array([r[2] for r in top_tokens])
print(f"  median norm non-first {np.median(norms[pos>0]):.2f}; first-token median {np.median(norms[pos==0]):.1f}; non-first tokens above 10x median: {(norms[pos>0] > 10*np.median(norms[pos>0])).sum()} of {(pos>0).sum()}")
print("  decoded top non-first tokens:", [(r[3], r[1], round(r[0])) for r in top_tokens if r[2] > 0][:12])
print("\n  rms by layer, first token vs rest (stream | attn output | mlp output), |dim35|")
print("  L    first: stream   attn    mlp   dim35 |  rest: stream   attn    mlp   dim35")
for i in [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 26, 27]:
    f, r = per_layer[(i, "first")], per_layer[(i, "rest")]
    print(f"  {i:2d}  {np.sqrt(f[0]/f[3]):12.3f} {np.sqrt(f[1]/f[3]):7.3f} {np.sqrt(f[2]/f[3]):7.3f} {f[4]/f[3]:8.1f} | {np.sqrt(r[0]/r[3]):10.3f} {np.sqrt(r[1]/r[3]):7.3f} {np.sqrt(r[2]/r[3]):7.3f} {r[4]/r[3]:8.2f}")
# Which L2 neurons write into dim 35 on the first token?
x = torch.tensor([segs["code"][0]], device=DEVICE)
with torch.no_grad():
    model(x, use_cache=False, output_attentions=True)
for i in (1, 2, 3):
    D = model.model.layers[i].mlp.down_proj.weight.detach()
    a = cap[("act", i)][0, 0]
    contrib = a * D[35]
    top = torch.topk(contrib.abs(), 5)
    print(f"  L{i} MLP on first token: output dim35 {contrib.sum().item():.1f}; top neurons {top.indices.tolist()} contrib {[round(contrib[j].item(),1) for j in top.indices.tolist()]}")
    print(f"     same neurons on token 50: {[round((cap[('act', i)][0, 50] * D[35])[j].item(),3) for j in top.indices.tolist()]}")
for h in hooks:
    h.remove()
