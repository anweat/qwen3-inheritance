# Do tasks share a thinking structure, independent of weight similarity?
# On identical text, compare LLM, ASR and TTS talker hidden states and attention in non-streaming TTS mode,
# where text tokens pass through the talker's decoder like ordinary self-attention.
# Controls: randomly initialized network and shuffled tokens. Requires transformers 4.57 and qwen-tts.
import gc, json, os, sys, time
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM
from qwen_tts import Qwen3TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e2_activation"))
from common import corpus  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.expanduser("~/.cache/qwen-all/e4")
DTYPE = torch.bfloat16
BATCH = 8
ATT_LAYERS = [2, 6, 10, 14, 18, 22, 26]
GRID = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28]
STEP = 2
NEEDED = {"0.6B": ("Qwen/Qwen3-0.6B", ["Qwen3-0.6B-Base", "Qwen3-0.6B", "ASR-0.6B", "TTS-0.6B", "Random-0.6B"]),
          "1.7B": ("Qwen/Qwen3-1.7B", ["Qwen3-1.7B-Base", "TTS-1.7B", "Random-1.7B"])}
tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.json", "*.txt"]))
ROLE = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]


def rp(repo):
    """Weights fetched by fetch_direct.py live in plain folders; fall back to the Hub cache."""
    p = os.path.expanduser("~/.cache/qwen-all/models/" + repo.split("/")[1])
    return p if os.path.exists(os.path.join(p, "config.json")) else repo


def segments():
    s = corpus(tok, 24)
    out = []
    for c in ("zh_doc", "en_doc", "spoken_en", "spoken_zh", "code"):
        out += [(c, x) for x in s[c] if len(x) == 128]
    return out


def pack(hs, att, offset, segs, res):
    """Store text positions only; attention keeps the text-to-text block and the share going to position 0."""
    b = hs[0].shape[0]
    idx = torch.arange(offset, offset + 128, STEP, device=DEVICE)
    for L in GRID:
        res["hid"].setdefault(L, []).append(hs[L][:, idx].float().cpu().half().numpy().reshape(-1, hs[L].shape[-1]))
    res["norm0"].append(np.array([float(hs[L][:, 0].float().norm(dim=-1).mean()) for L in range(len(hs))]))
    res["normtext"].append(np.array([float(hs[L][:, idx].float().norm(dim=-1).mean()) for L in range(len(hs))]))
    tp = torch.arange(offset, offset + 128, device=DEVICE)
    for j, L in enumerate(ATT_LAYERS):
        A = att[L].float()  # b, heads, T, T
        res["sink"].setdefault(L, []).append(A[:, :, tp, 0].mean((0, 2)).cpu().numpy())
        rows = A[:, :, tp][:, :, :, tp]
        res["within"].setdefault(L, []).append(rows.sum(-1).mean((0, 2)).cpu().numpy())
        if res["n_att"] < 16:
            res["att"].setdefault(L, []).append(rows.cpu().half().numpy())
    res["n_att"] += b


def collect(name, fn, segs):
    t0 = time.time()
    res = dict(hid={}, att={}, sink={}, within={}, norm0=[], normtext=[], n_att=0)
    for i in range(0, len(segs), BATCH):
        ids = torch.tensor([s[1] for s in segs[i:i + BATCH]], device=DEVICE)
        with torch.no_grad():
            hs, att, offset = fn(ids)
        pack(hs, att, offset, segs, res)
    out = {f"hid{L}": np.concatenate(v) for L, v in res["hid"].items()}
    out.update({f"att{L}": np.concatenate(v) for L, v in res["att"].items()})
    out.update({f"sink{L}": np.mean(v, 0) for L, v in res["sink"].items()})
    out.update({f"within{L}": np.mean(v, 0) for L, v in res["within"].items()})
    out["norm0"], out["normtext"] = np.mean(res["norm0"], 0), np.mean(res["normtext"], 0)
    np.savez(os.path.join(OUT, name + ".npz"), **out)
    print(f"{name}: collected in {time.time()-t0:.0f}s; first-token norm by layer 0/4/14/26/27: "
          + "/".join(f"{out['norm0'][k]:.0f}" for k in (0, 4, 14, 26, 27)) + "; text norm: " + "/".join(f"{out['normtext'][k]:.1f}" for k in (0, 4, 14, 26, 27)), flush=True)


def llm_fn(model):
    role = torch.tensor([ROLE], device=DEVICE)

    def fn(ids):
        x = torch.cat([role.expand(ids.shape[0], -1), ids], 1)
        o = model.model(input_ids=x, output_hidden_states=True, output_attentions=True, use_cache=False)
        return o.hidden_states, o.attentions, len(ROLE)
    return fn


def tts_fn(m):
    t, cfg = m.talker, m.config
    tc = cfg.talker_config
    tp = lambda i: t.text_projection(t.get_text_embeddings()(i))
    cd = t.get_input_embeddings()

    def fn(ids):
        b, n = ids.shape
        L = lambda v: torch.tensor([v], device=DEVICE).expand(b, -1)
        bos, eos, pad = [tp(L([x])) for x in (cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id)]
        role = tp(L(ROLE))
        pre = torch.cat([pad, pad, pad, bos], 1) + cd(L([tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id, tc.codec_pad_id]))
        body = torch.cat([tp(ids), eos], 1) + cd(L([tc.codec_pad_id] * (n + 1)))
        tail = pad + cd(L([tc.codec_bos_id]))
        emb = torch.cat([role, pre, body, tail], 1)
        o = t.model(inputs_embeds=emb, output_hidden_states=True, output_attentions=True, use_cache=False)
        return o.hidden_states, o.attentions, role.shape[1] + pre.shape[1]
    return fn


def run_all():
    os.makedirs(OUT, exist_ok=True)
    segs = segments()
    print("segments", len(segs), {c: sum(s[0] == c for s in segs) for c in set(s[0] for s in segs)}, flush=True)
    np.save(os.path.join(OUT, "cats.npy"), np.array([c for c, _ in segs for _ in range(0, 128, STEP)]))

    def done(n):
        return os.path.exists(os.path.join(OUT, n + ".npz"))

    def free(*objs):
        for o in objs:
            del o
        gc.collect()
        torch.cuda.empty_cache()

    for n, repo in (("Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B-Base"), ("Qwen3-0.6B", "Qwen/Qwen3-0.6B"), ("Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B-Base")):
        if not done(n):
            m = Qwen3ForCausalLM.from_pretrained(rp(repo), dtype=DTYPE, attn_implementation="eager").to(DEVICE).eval()
            collect(n, llm_fn(m), segs)
            free(m)
    for n, repo in (("Random-0.6B", "Qwen/Qwen3-0.6B"), ("Random-1.7B", "Qwen/Qwen3-1.7B-Base")):
        if not done(n):
            cfg = AutoConfig.from_pretrained(rp(repo))
            cfg._attn_implementation = "eager"
            torch.manual_seed(0)
            m = Qwen3ForCausalLM(cfg).to(DTYPE).to(DEVICE).eval()
            collect(n, llm_fn(m), segs)
            free(m)
    if not done("ASR-0.6B"):
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        cfg._attn_implementation = "eager"
        m = Qwen3ForCausalLM(cfg).to(DTYPE)
        sd = {}
        with safe_open(os.path.expanduser("~/.cache/qwen-all/Qwen3-ASR-0.6B/model.safetensors"), "pt") as h:
            for k in h.keys():
                if k.startswith("thinker.model."):
                    sd["model." + k[len("thinker.model."):]] = h.get_tensor(k)
        print("ASR load:", m.load_state_dict(sd, strict=False, assign=True), flush=True)
        m.tie_weights()
        m = m.to(DEVICE).eval()
        collect("ASR-0.6B", llm_fn(m), segs)
        free(m)
    for n, repo in (("TTS-0.6B", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"), ("TTS-1.7B", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")):
        if not done(n):
            w = Qwen3TTSModel.from_pretrained(rp(repo), device_map=DEVICE, dtype=DTYPE, attn_implementation="eager")
            m = w.model.eval()
            print(n, "codec ids:", {k: getattr(m.config.talker_config, k) for k in ("codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id", "codec_pad_id", "codec_bos_id")}, "tts:", m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id, flush=True)
            collect(n, tts_fn(m), segs)
            free(m, w)


def gram(x):
    x = torch.as_tensor(x.astype(np.float32), device=DEVICE)
    x = x - x.mean(0, keepdim=True)
    return (x @ x.T).double()


def hsic(K, Lm):
    n = K.shape[0]
    K, Lm = K.clone(), Lm.clone()
    K.fill_diagonal_(0)
    Lm.fill_diagonal_(0)
    a = (K * Lm).sum()
    b = K.sum() * Lm.sum() / ((n - 1) * (n - 2))
    c = 2 * (K.sum(1) * Lm.sum(1)).sum() / (n - 2)
    return (a + b - c) / (n * (n - 3))


def cka(K, Lm):
    return float(hsic(K, Lm) / (hsic(K, K) * hsic(Lm, Lm)).clamp_min(1e-30).sqrt())


def ridge_r2(X, Y, tr, te):
    """Dual ridge from X to Y, lambda chosen on an inner split, R2 pooled over output dims on held-out segments."""
    X = torch.as_tensor(X.astype(np.float32), device=DEVICE).double()
    Y = torch.as_tensor(Y.astype(np.float32), device=DEVICE).double()
    mx, my = X[tr].mean(0), Y[tr].mean(0)
    X, Y = X - mx, Y - my
    sx = X[tr].std().clamp_min(1e-9)
    X = X / sx
    n = len(tr)
    fit, val = tr[: int(n * .8)], tr[int(n * .8):]

    def solve(a, lam):
        K = X[a] @ X[a].T
        K.diagonal().add_(lam)
        return X[a].T @ torch.linalg.solve(K, Y[a])
    best, bl = 1e30, None
    for lam in (1, 10, 100, 1000, 10000, 100000):
        W = solve(fit, lam)
        e = ((X[val] @ W - Y[val]) ** 2).sum().item()
        if e < best:
            best, bl = e, lam
    W = solve(tr, bl)
    return 1 - ((X[te] @ W - Y[te]) ** 2).sum().item() / ((Y[te] - Y[tr].mean(0)) ** 2).sum().item()


def analyze(family):
    repo, names = NEEDED[family]
    D = {n: np.load(os.path.join(OUT, n + ".npz")) for n in names}
    base = names[0]
    print(f"\n{'='*20} family {family}, reference {base} {'='*20}", flush=True)
    cats = np.load(os.path.join(OUT, "cats.npy"))
    n = len(cats)
    G = {nm: {L: gram(D[nm][f"hid{L}"]) for L in GRID} for nm in names}
    perm = torch.as_tensor(np.random.default_rng(0).permutation(n), device=DEVICE)
    others = names[1:]
    print(f"\n[1] debiased linear CKA at the same depth, text tokens only (n={n})")
    print("  " + " " * 22 + " ".join(f"L{L:<5d}" for L in GRID))
    for o in others:
        print(f"  {o:22s}" + " ".join(f"{cka(G[base][L], G[o][L]):6.3f}" for L in GRID), flush=True)
    print(f"  {'shuffled tokens (TTS)':22s}" + " ".join(f"{cka(G[base][L], G[others[-2]][L][perm][:, perm]):6.3f}" for L in GRID), flush=True)
    print("\n[2] CKA across depths, reference (rows) vs each model (columns), selected depths")
    sel = [2, 6, 10, 14, 18, 22, 26, 27]
    for o in others:
        print(f"  {o}: " + " ".join(f"L{L:<5d}" for L in sel))
        for r in sel:
            print(f"    L{r:<3d}" + " ".join(f"{cka(G[base][r], G[o][c]):6.3f}" for c in sel))
    print("\n[3] CKA by language at the same depth (zh_doc / en_doc), TTS vs reference")
    tts = others[-2]
    for c in ("zh_doc", "en_doc", "code"):
        i = torch.as_tensor(np.where(cats == c)[0], device=DEVICE)
        print(f"  {c:8s} n={len(i)} " + " ".join(f"L{L}:{cka(G[base][L][i][:, i], G[tts][L][i][:, i]):.3f}" for L in (2, 8, 14, 20, 26, 27)), flush=True)
    # Linear reconstruction, split by segment.
    seg = np.repeat(np.arange(n // 64), 64)
    rng = np.random.default_rng(0)
    sid = rng.permutation(n // 64)
    te_s = set(sid[: len(sid) // 4].tolist())
    te = torch.as_tensor(np.where([s in te_s for s in seg])[0], device=DEVICE)
    tr = torch.as_tensor(rng.permutation(np.where([s not in te_s for s in seg])[0]), device=DEVICE)
    R = [4, 8, 12, 16, 20, 24, 27]
    for src, dst, label in ((base, tts, f"{base} -> {tts}"), (tts, base, f"{tts} -> {base}"), (others[-1], tts, f"{others[-1]} -> {tts} (control)")):
        print(f"\n[4] held-out R2 predicting {label}; rows source depth, columns target depth")
        print("      " + " ".join(f"L{c:<5d}" for c in R))
        for r in R:
            print(f"  L{r:<3d}" + " ".join(f"{ridge_r2(D[src][f'hid{r}'], D[dst][f'hid{c}'], tr, te):6.3f}" for c in R), flush=True)
    print("\n[5] attention on text tokens: JS (bits) to reference; mean over heads, and with heads matched one to one; sink = share going to position 0; within = share staying on text")
    def js(p, q):
        p = p / np.clip(p.sum(-1, keepdims=True), 1e-9, None)
        q = q / np.clip(q.sum(-1, keepdims=True), 1e-9, None)
        m = (p + q) / 2 + 1e-9
        return 0.5 * ((p * (np.log2(p + 1e-9) - np.log2(m))).sum(-1) + (q * (np.log2(q + 1e-9) - np.log2(m))).sum(-1))
    for o in others:
        print(f"  {o}")
        for L in ATT_LAYERS:
            a, b = D[base][f"att{L}"].astype(np.float32), D[o][f"att{L}"].astype(np.float32)
            mean_js = js(a.mean(1), b.mean(1))[:, 8:].mean()
            cost = np.zeros((16, 16))
            for i in range(16):
                for j in range(16):
                    cost[i, j] = js(a[:, i], b[:, j])[:, 8:].mean()
            ri, ci = linear_sum_assignment(cost)
            null = cost[np.arange(16), np.random.default_rng(L).permutation(16)].mean()
            print(f"    L{L:2d} mean-head JS {mean_js:.3f} | matched JS {cost[ri, ci].mean():.3f} (random pairing {null:.3f}) | sink ref {D[base][f'sink{L}'].mean():.3f} model {D[o][f'sink{L}'].mean():.3f} | within ref {D[base][f'within{L}'].mean():.3f} model {D[o][f'within{L}'].mean():.3f}", flush=True)
    print("\n[6] stream norm by layer: position 0 and text tokens")
    for nm in names:
        print(f"  {nm:18s} pos0 " + " ".join(f"{D[nm]['norm0'][k]:7.0f}" for k in (0, 2, 3, 4, 14, 26, 27, 28)) + "  text " + " ".join(f"{D[nm]['normtext'][k]:6.1f}" for k in (0, 2, 3, 4, 14, 26, 27, 28)), flush=True)
    for L in list(G):
        G[L] = None
    del G
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_all()
    analyze("0.6B")
    analyze("1.7B")
    print("STRUCTUREDONE", flush=True)
