# Stage 2: can LLM hidden states causally replace the text path inside the TTS talker?
# At layer k, text positions are overwritten and later (codec) rows are blocked from reading text in layers below k,
# so text can only reach the audio side through the supplied states. Target: first-codebook prediction on real speech
# codes, teacher forced. Maps are linear ridge fits; no gradient training. Controls: native text at k, train-mean,
# shuffled order, another clip's text, and a random network.
import argparse, gc, hashlib, json, os, sys, time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoConfig, Qwen3ForCausalLM
from qwen_tts import Qwen3TTSModel
from shared_structure import DEVICE, DTYPE, ROLE, rp, segments, tok

SPEECH = os.path.expanduser("~/.cache/qwen-all/speech")
OUT = os.path.expanduser("~/.cache/qwen-all/e4_stitch")
SETS = ["aishell", "fleurs_cmn_hans_cn", "fleurs_en_us", "sichuan", "northeast", "henan"]
# Extra text-only sequences for fitting maps; these clips are never scored.
TEXT_SETS = SETS + ["wu", "cantonese_md", "fleurs_yue_hant_hk"]
KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20]
CROSS_K = [2, 4, 6, 8]
CROSS_J = [4, 12, 20, 27]
FAMILIES = {
    "0.6B": dict(tts="Qwen/Qwen3-TTS-12Hz-0.6B-Base", llms={"base": "Qwen/Qwen3-0.6B-Base", "instruct": "Qwen/Qwen3-0.6B"}, cfg="Qwen/Qwen3-0.6B"),
    "1.7B": dict(tts="Qwen/Qwen3-TTS-12Hz-1.7B-Base", llms={"base": "Qwen/Qwen3-1.7B-Base"}, cfg="Qwen/Qwen3-1.7B-Base"),
}
GRID = sorted(set(KS + CROSS_J))


def free():
    gc.collect()
    torch.cuda.empty_cache()


def load_clips(n_test):
    rows = [json.loads(l) for l in open(os.path.join(SPEECH, "manifest.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["set"] in TEXT_SETS]
    rows.sort(key=lambda r: hashlib.md5(r["path"].encode()).hexdigest())
    for r in rows:
        r["ids"] = tok(r["text"], add_special_tokens=False)["input_ids"][:120]
    rows = [r for r in rows if len(r["ids"]) >= 4]
    test = [r for r in rows if r["set"] in SETS][:n_test]
    chosen = {r["path"] for r in test}
    return test, [r for r in rows if r["path"] not in chosen]


def llm_states(model, id_lists):
    role = torch.tensor([ROLE], device=DEVICE)
    out = []
    for ids in id_lists:
        x = torch.cat([role, torch.tensor([ids], device=DEVICE)], 1)
        with torch.no_grad():
            hs = model.model(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states
        out.append({L: hs[L][0, len(ROLE):].float().cpu() for L in GRID})
    return out


class Talker:
    def __init__(self, repo):
        self.w = Qwen3TTSModel.from_pretrained(rp(repo), device_map=DEVICE, dtype=DTYPE, attn_implementation="eager")
        self.m = self.w.model.eval()
        self.t = self.m.talker
        self.cfg = self.m.config
        self.tc = self.cfg.talker_config

    def tp(self, ids):
        return self.t.text_projection(self.t.get_text_embeddings()(torch.tensor([ids], device=DEVICE)))

    def cd(self, ids):
        return self.t.get_input_embeddings()(torch.tensor([ids], device=DEVICE))

    def prompt(self, ids):
        """role + tags + text + eos + tail, matching non-streaming generation."""
        c, t = self.cfg, self.tc
        pad, bos, eos = self.tp([c.tts_pad_token_id]), self.tp([c.tts_bos_token_id]), self.tp([c.tts_eos_token_id])
        role = self.tp(ROLE)
        pre = torch.cat([pad, pad, pad, bos], 1) + self.cd([t.codec_nothink_id, t.codec_think_bos_id, t.codec_think_eos_id, t.codec_pad_id])
        body = torch.cat([self.tp(ids), eos], 1) + self.cd([t.codec_pad_id] * (len(ids) + 1))
        tail = pad + self.cd([t.codec_bos_id])
        return torch.cat([role, pre, body, tail], 1), role.shape[1] + pre.shape[1], pad

    def text_states(self, id_lists):
        out = []
        for ids in id_lists:
            emb, start, _ = self.prompt(ids)
            with torch.no_grad():
                hs = self.t.model(inputs_embeds=emb, output_hidden_states=True, use_cache=False).hidden_states
            out.append({L: hs[L][0, start:start + len(ids)].float().cpu() for L in GRID})
        return out

    def frames(self, codes, pad):
        """Sum of all codebook embeddings per frame plus the padding text embedding."""
        c = torch.as_tensor(codes, device=DEVICE)[None]
        e = self.t.get_input_embeddings()(c[:, :, 0])
        pe = self.t.code_predictor.get_input_embeddings()
        for i in range(1, c.shape[2]):
            e = e + pe[i - 1](c[:, :, i])
        return e + pad

    def score(self, clip, k=None, rep=None, block=True, att=False):
        """Per-frame CE and accuracy for the first codebook. rep overwrites text at layer k; block masks text below k."""
        ids, codes = clip["ids"], clip["codes"]
        emb, start, pad = self.prompt(ids)
        tail = emb.shape[1] - 1
        end = start + len(ids)
        x = torch.cat([emb, self.frames(codes[:-1], pad)], 1)
        hooks = []
        if k is not None and block:
            def mask_hook(mod, args, kwargs):
                m = kwargs["attention_mask"]
                assert m is not None
                m = m.clone()
                m[:, :, end:, start:end] = torch.finfo(m.dtype).min
                kwargs["attention_mask"] = m
                return args, kwargs
            for i in range(k):
                hooks.append(self.t.model.layers[i].register_forward_pre_hook(mask_hook, with_kwargs=True))
        if rep is not None:
            def rep_hook(mod, args, kwargs):
                h = args[0].clone()
                h[:, start:end] = rep.to(h.device, h.dtype)
                return (h,) + args[1:], kwargs
            hooks.append(self.t.model.layers[k].register_forward_pre_hook(rep_hook, with_kwargs=True))
        with torch.no_grad():
            o = self.t.model(inputs_embeds=x, use_cache=False, output_attentions=att)
        for h in hooks:
            h.remove()
        lg = self.t.codec_head(o.last_hidden_state[0, tail:]).float()
        tgt = torch.as_tensor(codes[:, 0], device=DEVICE)
        res = (F.cross_entropy(lg, tgt, reduction="sum").item(), (lg.argmax(-1) == tgt).sum().item(), len(tgt))
        if att:
            shares = [float(a[0][:, tail:, start:end].sum(-1).mean()) for a in o.attentions]
            return res, shares
        return res


def fit_ridge(Xs, Ys):
    """Primal ridge X -> Y with lambda picked on the last 15% of sequences."""
    n = len(Xs)
    cut = int(n * .85)
    X = torch.cat(Xs[:cut]).to(DEVICE).double()
    Y = torch.cat(Ys[:cut]).to(DEVICE).double()
    Xv = torch.cat(Xs[cut:]).to(DEVICE).double()
    Yv = torch.cat(Ys[cut:]).to(DEVICE).double()
    mx, my = X.mean(0), Y.mean(0)
    s = (X - mx).std().clamp_min(1e-9)

    def solve(X_, Y_, lam):
        A = (X_ - mx) / s
        C = A.T @ A
        C.diagonal().add_(lam * C.diagonal().mean())
        return torch.linalg.solve(C, A.T @ (Y_ - my))
    best, bl = 1e300, None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1):
        W = solve(X, Y, lam)
        e = (((Xv - mx) / s @ W + my - Yv) ** 2).sum().item()
        if e < best:
            best, bl = e, lam
    W = solve(torch.cat([X, Xv]), torch.cat([Y, Yv]), bl)
    return dict(W=W.float(), mx=mx.float(), my=my.float(), s=float(s))


def apply_map(m, x):
    x = x.to(DEVICE)
    return ((x - m["mx"]) / m["s"]) @ m["W"] + m["my"]


def r2(m, Xs, Ys):
    a = torch.cat([apply_map(m, x) for x in Xs]).double()
    y = torch.cat(Ys).to(DEVICE).double()
    return 1 - ((a - y) ** 2).sum().item() / ((y - m["my"].double()) ** 2).sum().item()


def rec_str(cm, cn, ce):
    """Recovered fraction; n/a where text has no effect at this k and the denominator is near zero."""
    return f"{(cm - ce) / (cm - cn):.2f}" if cm - cn >= .02 else "n/a"


def boot(diff, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array(diff)
    return np.percentile([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)], [2.5, 97.5])


def run_family(name, n_test, ks):
    fam = FAMILIES[name]
    test, train = load_clips(n_test)
    corpus_ids = [ids for _, ids in segments()]
    train_ids = corpus_ids + [c["ids"] for c in train]
    test_ids = [c["ids"] for c in test]
    print(f"\n{'='*16} family {name}: {len(test)} test clips, {len(train_ids)} training sequences {'='*16}", flush=True)
    states = {}
    for label, repo in fam["llms"].items():
        m = Qwen3ForCausalLM.from_pretrained(rp(repo), dtype=DTYPE, attn_implementation="eager").to(DEVICE).eval()
        states[label] = (llm_states(m, train_ids), llm_states(m, test_ids))
        del m
        free()
    cfg = AutoConfig.from_pretrained(rp(fam["cfg"]))
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(DTYPE).to(DEVICE).eval()
    states["random"] = (llm_states(m, train_ids), llm_states(m, test_ids))
    del m
    free()
    T = Talker(fam["tts"])
    tts_train, tts_test = T.text_states(train_ids), T.text_states(test_ids)
    t0 = time.time()
    for c in test:
        a, sr = sf.read(os.path.join(SPEECH, c["path"]), dtype="float32")
        codes = T.m.speech_tokenizer.encode(a, sr=sr).audio_codes[0].cpu().numpy()
        c["codes"] = codes[:300]
    print(f"encoded speech in {time.time()-t0:.0f}s; frames per clip mean {np.mean([len(c['codes']) for c in test]):.0f}; codes shape {test[0]['codes'].shape}", flush=True)

    rec = {}
    base_scores, shares = [], []
    for c in test:
        r, s = T.score(c, att=True)
        base_scores.append(r)
        shares.append(s)
    shares = np.mean(shares, 0)
    print("share of codec-row attention on text, by layer: " + " ".join(f"L{i}:{shares[i]:.3f}" for i in range(28)), flush=True)
    tot = lambda rs: (sum(r[0] for r in rs) / sum(r[2] for r in rs), sum(r[1] for r in rs) / sum(r[2] for r in rs))
    u = tot(base_scores)
    print(f"native talker, nothing masked: CE {u[0]:.3f} acc {u[1]:.3f}", flush=True)
    rec["native"] = base_scores
    print("\nCE by replacement layer k. recovered = (mean - cond) / (mean - native_k): 1 means as good as native text at k, 0 means no better than no text")
    labels = list(fam["llms"]) + ["random"]
    print(f"{'k':>3s} {'native_k':>9s} {'mean':>8s} {'shuffle':>8s} {'other':>8s} " + " ".join(f"{l:>8s} {'rec':>5s} {'R2':>6s}" for l in labels) + "   base-native_k [95%]", flush=True)
    for k in ks:
        Xt, Yt = None, [s[k] for s in tts_train]
        mean_vec = torch.cat(Yt).mean(0)
        native = [T.score(c, k=k) for c in test]
        meanr = [T.score(c, k=k, rep=mean_vec) for c in test]
        rng = np.random.default_rng(k)
        shuf = []
        other = []
        for i, c in enumerate(test):
            h = tts_test[i][k]
            shuf.append(T.score(c, k=k, rep=h[torch.as_tensor(rng.permutation(len(h)))]))
            o = tts_test[(i + 1) % len(test)][k]
            shuf_len = len(h)
            other.append(T.score(c, k=k, rep=o[torch.arange(shuf_len) % len(o)]))
        rec[f"k{k}"] = dict(native=native, mean=meanr, shuffle=shuf, other=other)
        cm, cn = tot(meanr)[0], tot(native)[0]
        cols = []
        for l in labels:
            mp = fit_ridge([s[k] for s in states[l][0]], Yt)
            rs = [T.score(c, k=k, rep=apply_map(mp, states[l][1][i][k])) for i, c in enumerate(test)]
            rec[f"k{k}"][l] = rs
            ce = tot(rs)[0]
            cols.append(f"{ce:8.3f} {rec_str(cm, cn, ce):>5s} {r2(mp, [s[k] for s in states[l][1]], [s[k] for s in tts_test]):6.3f}")
        d = [(a[0] / a[2]) - (b[0] / b[2]) for a, b in zip(rec[f'k{k}']['base'], native)]
        lo, hi = boot(d)
        print(f"{k:3d} {cn:9.3f} {cm:8.3f} {tot(shuf)[0]:8.3f} {tot(other)[0]:8.3f} " + " ".join(cols) + f"   {np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
        free()
    print("\nReading a fixed LLM depth J into talker layer k (base LLM): CE and recovered", flush=True)
    for k in [x for x in CROSS_K if x in ks]:
        cm, cn = tot(rec[f"k{k}"]["mean"])[0], tot(rec[f"k{k}"]["native"])[0]
        cells = []
        for J in CROSS_J:
            mp = fit_ridge([s[J] for s in states["base"][0]], [s[k] for s in tts_train])
            rs = [T.score(c, k=k, rep=apply_map(mp, states["base"][1][i][J])) for i, c in enumerate(test)]
            ce = tot(rs)[0]
            cells.append(f"J{J}: {ce:.3f} rec {rec_str(cm, cn, ce)}")
        print(f"  k{k:<3d} native_k {cn:.3f} mean {cm:.3f} | " + " | ".join(cells), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, f"stitch_{name}.json"), "w"))
    del T
    free()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.smoke:
        run_family("0.6B", 4, [4])
    else:
        run_family("0.6B", 100, KS)
        run_family("1.7B", 100, KS)
    print("STITCHDONE", flush=True)
