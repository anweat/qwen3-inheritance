# E7 stage 2 (primary, criteria in PLAN.md): do LLM-initialized TTS models share the LLM's deep text representations?
# Same corpus builder as E4 stage 1 (e2_activation/common.py, 24 segments x 5 categories, 128 tokens, every 2nd position).
# Each TTS gets its own native text prompt; only the text positions are compared, with the base LLM fed the same text ids.
# Round 2 (PLAN.md): robust() removes massive-activation tokens and dims; Instruct vs Base must reach .9 at every layer.
# Controls: a randomly initialized copy of the base architecture (seed 0) against the base; shuffled row pairing.
#   python cka.py [models dir]
import json, os, sys
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e2_activation"))
from common import corpus  # noqa: E402

M = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.cache/qwen-all/e7/models")
OUT = os.path.join(HERE, "cka2_results.json")  # round 1 (raw CKA only) is cka_results.json
FEATS = os.path.join(os.path.dirname(M), "feats")
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
STEP = 2
FAM = {
    "qwen": dict(base="qwen25_0.5b_base", inst="qwen25_0.5b_inst", tts=["cosy", "spark"]),
    "llama": dict(base="llama32_1b_base", inst="llama32_1b_inst", tts=["llasa"]),
}


def segments(tok, inst):
    if not tok.chat_template:  # corpus() also builds a chat category (unused here) and needs a template
        tok.chat_template = AutoTokenizer.from_pretrained(os.path.join(M, inst)).chat_template
    s = corpus(tok, 24)
    return [x for c in ("zh_doc", "en_doc", "spoken_en", "spoken_zh", "code") for x in s[c] if len(x) == 128]


def run(model, seqs, embed=None):
    """seqs: list of (input ids or embeddings, text start). Returns {layer: [n*64, d] float32} for text positions."""
    out = {}
    model.eval()
    with torch.no_grad():
        for x, a in seqs:
            if embed is None:
                h = model(input_ids=torch.tensor([x], device=DEV), output_hidden_states=True).hidden_states
            else:
                h = model(inputs_embeds=embed(x), output_hidden_states=True).hidden_states
            idx = torch.arange(a, a + 128, STEP, device=DEV)
            for L in range(1, len(h)):
                out.setdefault(L, []).append(h[L][0, idx].float().cpu().numpy())
    return {L: np.concatenate(v) for L, v in out.items()}


def lm(name, random_init=False):
    p = os.path.join(M, name)
    if random_init:
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(p), torch_dtype=DT).to(DEV)
    return AutoModelForCausalLM.from_pretrained(p, torch_dtype=DT).to(DEV)


def base_seqs(fam, tok, segs):
    if fam == "llama":
        return [([tok.bos_token_id] + s, 1) for s in segs]
    return [(s, 0) for s in segs]


def tts_seqs(name, segs):
    """Native prompt around the base tokenizer's text ids; checks that the TTS tokenizer encodes the text the same way."""
    if name == "cosy":
        return [(s, 1) for s in segs]  # [sos_eos embedding] + text embeddings, built in cosy_model()
    p = os.path.join(M, "spark_tts_0.5b/LLM" if name == "spark" else "llasa_1b")
    tok = AutoTokenizer.from_pretrained(p)
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    if name == "spark":
        pre, post = enc("<|task_tts|><|start_content|>"), enc("<|end_content|><|start_global_token|>")
    else:
        u = [{"role": "user", "content": "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>@@@<|TEXT_UNDERSTANDING_END|>"},
             {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}]
        full = tok.apply_chat_template(u, tokenize=False, continue_final_message=True)
        a, b = full.split("@@@")
        pre, post = tok(a, add_special_tokens=False)["input_ids"], enc(b)
    seqs = [(pre + s + post, len(pre)) for s in segs]
    return seqs


def cosy_model():
    """CosyVoice2's Qwen2 LM from llm.pt on the CosyVoice-BlankEN config, plus its sos_eos embedding."""
    cfg = AutoConfig.from_pretrained(os.path.join(M, "cosyvoice2_0.5b/CosyVoice-BlankEN"))
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    sd = torch.load(os.path.join(M, "cosyvoice2_0.5b/llm.pt"), map_location="cpu", weights_only=True)
    lm_sd = {k[len("llm.model."):]: v for k, v in sd.items() if k.startswith("llm.model.")}
    missing, unexpected = model.load_state_dict(lm_sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model = model.to(DEV, DT)
    sos = sd["llm_embedding.weight"][0].to(DEV, DT)
    emb = model.get_input_embeddings()

    def embed(ids):
        e = emb(torch.tensor([ids], device=DEV))
        return torch.cat([sos.view(1, 1, -1), e], 1)
    return model, embed


def gram(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    x = x - x.mean(0, keepdim=True)
    return x @ x.T


def hsic(K, Lm):
    n = K.shape[0]
    K, Lm = K.clone(), Lm.clone()
    K.fill_diagonal_(0)
    Lm.fill_diagonal_(0)
    return ((K * Lm).sum() + K.sum() * Lm.sum() / ((n - 1) * (n - 2)) - 2 * (K.sum(1) * Lm.sum(1)).sum() / (n - 2)) / (n * (n - 3))


def cka(X, Y):
    K, Lm = gram(X), gram(Y)
    return float(hsic(K, Lm) / (hsic(K, K) * hsic(Lm, Lm)).clamp_min(1e-30).sqrt())


def robust(X, Y, k=10.0):
    """Round 2 (PLAN.md): drop tokens whose norm exceeds k x the layer median in either model (jointly),
    then each model's dims whose RMS exceeds k x its median dim RMS."""
    nx, ny = np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1)
    keep = (nx <= k * np.median(nx)) & (ny <= k * np.median(ny))
    X, Y = X[keep], Y[keep]
    rx, ry = np.sqrt((X ** 2).mean(0)), np.sqrt((Y ** 2).mean(0))
    dx, dy = rx <= k * np.median(rx), ry <= k * np.median(ry)
    return X[:, dx], Y[:, dy], (int((~keep).sum()), int((~dx).sum()), int((~dy).sum()))


def thirds(d):
    ls = sorted(d)
    parts = np.array_split(np.array(ls), 3)
    return [float(np.mean([d[l] for l in p])) for p in parts]


res, valid = {}, {}
for fam, f in FAM.items():
    tok = AutoTokenizer.from_pretrained(os.path.join(M, f["base"]))
    segs = segments(tok, f["inst"])
    bs = base_seqs(fam, tok, segs)
    print(f"[{fam}] {len(segs)} segments, device {DEV}", flush=True)
    H = {"base": run(lm(f["base"]), bs)}
    H["inst"] = run(lm(f["inst"]), bs)
    H["random"] = run(lm(f["base"], random_init=True), bs)
    for t in f["tts"]:
        if t == "cosy":
            model, embed = cosy_model()
            H[t] = run(model, tts_seqs(t, segs), embed)
        else:
            H[t] = run(lm("spark_tts_0.5b/LLM" if t == "spark" else "llasa_1b"), tts_seqs(t, segs))
        torch.cuda.empty_cache() if DEV == "cuda" else None
    os.makedirs(FEATS, exist_ok=True)
    for m, h in H.items():
        np.savez(os.path.join(FEATS, f"{fam}_{m}.npz"), **{f"L{L}": v.astype(np.float16) for L, v in h.items()})
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(H["base"][1]))
    for m in ["inst", "random"] + f["tts"]:
        raw = {L: cka(H[m][L], H["base"][L]) for L in H["base"]}
        rob, drop = {}, {}
        for L in H["base"]:
            X, Y, info = robust(H[m][L], H["base"][L])
            rob[L], drop[L] = cka(X, Y), info
        sh = {L: cka(H[m][L][perm], H["base"][L]) for L in (1, max(H["base"]))}
        res[m] = dict(family=fam, cka_raw=raw, cka=rob, dropped=drop, thirds=thirds(rob), thirds_raw=thirds(raw), shuffled=sh)
        print(f"  {m:7s} robust " + " ".join(f"L{L}:{v:.2f}" for L, v in rob.items()) + "  thirds S/mid/D " + "/".join(f"{x:.2f}" for x in res[m]["thirds"]), flush=True)
        print(f"  {m:7s} raw    " + " ".join(f"L{L}:{v:.2f}" for L, v in raw.items()) + "  thirds S/mid/D " + "/".join(f"{x:.2f}" for x in res[m]["thirds_raw"])
              + "  shuffled " + " ".join(f"{v:.2f}" for v in sh.values()), flush=True)
        print(f"  {m:7s} dropped tokens/dims(model)/dims(base) " + " ".join(f"L{L}:{a}/{b}/{c}" for L, (a, b, c) in drop.items()), flush=True)
    valid[fam] = min(res["inst"]["cka"].values()) >= .9
    print(f"  POSITIVE CONTROL {fam}: min Instruct-vs-Base robust CKA {min(res['inst']['cka'].values()):.2f} -> {'ok' if valid[fam] else 'FAILED'}", flush=True)
    rd = res["random"]["thirds"]
    for t in f["tts"]:
        S, _, D = res[t]["thirds"]
        ok = S - rd[0] >= .2
        cls = ("deep divergence" if D <= .5 and S - D >= .2 else "deep sharing" if D >= .8 else "intermediate") if ok else "uninterpretable"
        res[t]["class"] = cls
        print(f"  CLASS {t}: S {S:.2f} D {D:.2f} random S {rd[0]:.2f} -> {cls}", flush=True)
if not all(valid.values()):
    print("VERDICT: measurement invalid (positive control failed):", valid)
else:
    cl = [res[t]["class"] for f in FAM.values() for t in f["tts"] if res[t]["class"] != "uninterpretable"]
    verdict = ("general deep divergence" if cl and all(c == "deep divergence" for c in cl) else
               "depends on design" if "deep sharing" in cl else "mixed")
    print("VERDICT:", verdict, cl)
json.dump(res, open(OUT, "w"), indent=1)
