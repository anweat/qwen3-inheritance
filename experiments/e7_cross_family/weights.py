# E7 stage 1 (descriptive, see PLAN.md): which LLM is each TTS built from, and where did its weights move?
# Per layer: shrink-removed residual R = W - alpha*B (alpha = <W,B>/<B,B> per matrix), pooled over the 7 matrices,
# relative to |B|; plus plain cosine. Embedding: shared text rows only (the TTS vocabularies are extended).
#   python weights.py [models dir]     # CPU only
import glob, json, os, re, sys
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr

M = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.cache/qwen-all/e7/models")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_results.json")
MATS = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
SRC = {"qwen2_base": "qwen2_0.5b_base", "qwen2_inst": "qwen2_0.5b_inst", "qwen_base": "qwen25_0.5b_base", "qwen_inst": "qwen25_0.5b_inst", "cosy_blank": "cosyvoice2_0.5b/CosyVoice-BlankEN",
       "cosy": "cosyvoice2_0.5b/llm.pt", "spark": "spark_tts_0.5b/LLM", "llasa": "llasa_1b",
       "llama_base": "llama32_1b_base", "llama_inst": "llama32_1b_inst"}
# model: candidate bases. The two Instruct-vs-Base rows are the in-family reference.
PAIRS = {"cosy": ["qwen_base", "qwen_inst", "cosy_blank"], "cosy_blank": ["qwen_base", "qwen_inst", "qwen2_base", "qwen2_inst"], "qwen2_inst": ["qwen2_base"],
         "spark": ["qwen_base", "qwen_inst"], "llasa": ["llama_base", "llama_inst"],
         "qwen_inst": ["qwen_base"], "llama_inst": ["llama_base"]}


def load(name):
    p = os.path.join(M, SRC[name])
    if p.endswith(".pt"):
        sd = torch.load(p, map_location="cpu", weights_only=True)
    else:
        sd = {}
        for f in sorted(glob.glob(os.path.join(p, "*.safetensors"))):
            sd.update(load_file(f))
    out = {}
    for k, v in sd.items():
        m = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", k)
        if m and m.group(2) in MATS:
            out[(int(m.group(1)), m.group(2))] = v.float()
        elif k.endswith("embed_tokens.weight"):
            out["embed"] = v.float()
    assert "embed" in out and (0, MATS[0]) in out, f"{name}: unexpected keys, e.g. {list(sd)[:5]}"
    return out


def compare(W, B):
    layers = sorted({l for l, _ in (k for k in B if k != "embed")})
    rel, cos = [], []
    for l in layers:
        r2 = b2 = wb = w2 = 0.0
        for m in MATS:
            w, b = W[(l, m)], B[(l, m)]
            if w.shape != b.shape:
                return None
            d = float((w * b).sum())
            bb, ww = float((b * b).sum()), float((w * w).sum())
            a = d / bb
            r2 += float(((w - a * b) ** 2).sum())
            b2, wb, w2 = b2 + bb, wb + d, w2 + ww
        rel.append((r2 / b2) ** .5)
        cos.append(wb / (w2 * b2) ** .5)
    ew, eb = W["embed"], B["embed"]
    n = min(len(ew), len(eb))
    ew, eb = ew[:n], eb[:n]
    live = eb.norm(dim=1) > 1e-6  # padding rows of the base vocabulary are zero
    a = float((ew[live] * eb[live]).sum() / (eb[live] ** 2).sum())
    row = ((ew[live] - a * eb[live]).norm(dim=1) / eb[live].norm(dim=1)).numpy()
    rho = spearmanr(np.arange(len(rel)), rel).statistic
    return dict(rel=rel, cos=cos, mean_rel=float(np.mean(rel)), mean_cos=float(np.mean(cos)), depth_rho=float(rho),
                embed_rows=int(live.sum()), embed_rel_median=float(np.median(row)), embed_alpha=a,
                vocab=(len(W["embed"]), len(B["embed"])))


cache, res = {}, {}
get = lambda n: cache.setdefault(n, load(n))
for t, bases in PAIRS.items():
    for b in bases:
        r = compare(get(t), get(b))
        if r is None:
            print(f"{t:10s} vs {b:10s}  shapes differ")
            continue
        res[f"{t}|{b}"] = r
        thirds = np.array_split(np.array(r["rel"]), 3)
        print(f"{t:10s} vs {b:10s}  layers {len(r['rel']):2d}  rel resid mean {r['mean_rel']:.3f} (first/mid/last third "
              + "/".join(f"{x.mean():.3f}" for x in thirds) + f")  cos {r['mean_cos']:.3f}  depth rho {r['depth_rho']:+.2f}"
              f"  embed rows {r['embed_rows']} rel {r['embed_rel_median']:.3f}  vocab {r['vocab'][0]} vs {r['vocab'][1]}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
