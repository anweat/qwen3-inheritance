# Qwen3-ASR-0.6B on real audio: which parts of the decoder drift are needed for transcription?
#   graft: restore or replace decoder weights, measure WER/CER on speech.
#   trace: teacher-forced decomposition of transcript logits, and attention on audio vs text positions.
import argparse, glob, io, json, os, re, subprocess, sys, time, unicodedata
import numpy as np
import soundfile as sf
import torch
import librosa
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, Qwen3ASRForConditionalGeneration
from common import DEVICE, SPOKEN_EN, SPOKEN_ZH, read_decoder

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "asr_data")
WEIGHTS = os.path.expanduser("~/.cache/qwen-all/Qwen3-ASR-0.6B/model.safetensors")
MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
L = 28
INTER = 3072


def build():
    p = snapshot_download("Qwen/Qwen3-ASR-0.6B-hf", allow_patterns=["*.json", "*.jinja"])
    proc = AutoProcessor.from_pretrained(p)
    cfg = AutoConfig.from_pretrained(p)
    model = Qwen3ASRForConditionalGeneration(cfg).float()
    sd = {}
    with safe_open(WEIGHTS, "pt") as h:
        for k in h.keys():
            v = h.get_tensor(k).float()
            if k.startswith("thinker.model."):
                sd["model.language_model." + k[len("thinker.model."):]] = v
            elif k.startswith("thinker.audio_tower.proj"):
                n = k[len("thinker.audio_tower.proj"):]
                sd[f"model.multi_modal_projector.linear_{n}"] = v
            elif k.startswith("thinker.audio_tower."):
                sd["model.audio_tower." + k[len("thinker.audio_tower."):]] = v
            elif k.startswith("thinker.lm_head."):
                sd["lm_head." + k[len("thinker.lm_head."):]] = v
            else:
                sd[k] = v
    want = set(model.state_dict())
    missing, extra = want - set(sd), set(sd) - want
    if missing or extra:
        print("key mismatch: missing", sorted(missing)[:8], len(missing), "extra", sorted(extra)[:8], len(extra), flush=True)
    model.load_state_dict({k: v for k, v in sd.items() if k in want}, strict=False, assign=True)
    model.tie_weights()
    return model.to(DEVICE).eval(), proc


def clips(limit_ls=30):
    os.makedirs(DATA, exist_ok=True)
    out = []
    q = hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet", repo_type="dataset")
    import pyarrow.parquet as pq
    rows = pq.read_table(q).to_pylist()
    for r in rows[:limit_ls]:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        out.append(dict(kind="librispeech", audio=a.astype(np.float32), ref=r["text"], lang="en"))
    for kind, text, voices, lang in (("say_en", SPOKEN_EN, ["Samantha", "Daniel"], "en"), ("say_zh", SPOKEN_ZH, ["Flo (中文（中国大陆）)", "Reed (中文（中国大陆）)"], "zh")):
        for i, s in enumerate(text.strip().split("\n")):
            wav = os.path.join(DATA, f"{kind}_{i}.wav")
            if not os.path.exists(wav):
                aiff = wav[:-4] + ".aiff"
                subprocess.run(["say", "-v", voices[i % 2], "-o", aiff, s], check=True)
                subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", aiff, wav], check=True)
                os.remove(aiff)
            a, sr = sf.read(wav)
            out.append(dict(kind=kind, audio=a.astype(np.float32), ref=s, lang=lang))
    return out


def normalize(s, lang):
    s = unicodedata.normalize("NFKC", s).lower()
    if lang == "zh":
        return [c for c in s if "一" <= c <= "鿿" or c.isalnum()]
    return re.sub(r"[^\w' ]+", " ", s).split()


def edits(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def transcribe(model, proc, cs, max_new=128):
    res = {}
    samples = []
    for c in cs:
        inp = proc.apply_transcription_request(audio=c["audio"]).to(DEVICE, torch.float32)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        hyp = proc.decode(out[:, inp["input_ids"].shape[1]:], return_format="transcription_only")[0]
        e, n = edits(normalize(c["ref"], c["lang"]), normalize(hyp, c["lang"])), len(normalize(c["ref"], c["lang"]))
        r = res.setdefault(c["kind"], [0, 0])
        r[0] += e
        r[1] += n
        samples.append((c["kind"], c["ref"][:60], hyp[:60]))
    return {k: v[0] / max(1, v[1]) for k, v in res.items()}, samples


def graft(a):
    model, proc = build()
    cs = clips(a.ls)
    params = dict(model.named_parameters())
    base = read_decoder("Qwen3-0.6B-Base")
    inst = read_decoder("Qwen3-0.6B")
    asr = {k: params["model.language_" + k].detach().cpu().clone() for k in base if "model.language_" + k in params}
    keys = [f"model.layers.{i}.{t}.weight" for i in range(L) for t in MATS]
    z = np.load(os.path.join(a.dir, "plastic.Qwen3-ASR-0.6B.npz"))
    pn, ph = z["pn"], z["ph"]
    total = sum(asr[k].numel() for k in keys)
    rng = np.random.default_rng(0)

    def setw(fn, label, frac=None):
        t0 = time.time()
        with torch.no_grad():
            for k in keys:
                params["model.language_" + k].copy_(fn(k).to(DEVICE))
        r, samp = transcribe(model, proc, cs)
        f = "" if frac is None else f"  shared {frac:5.1%}"
        print(f"  {label:52s}" + " ".join(f"{k} {v:6.3f}" for k, v in r.items()) + f"{f}  [{time.time()-t0:.0f}s]", flush=True)
        return samp

    def alpha(k):
        return float((asr[k] * base[k]).sum() / (base[k] * base[k]).sum())

    print("### graft: error rate (WER for English, CER for Chinese); 'shared' = share of decoder matrix weights equal to Base", flush=True)
    samp = setw(lambda k: asr[k], "intact ASR", 0)
    for s in samp[:3] + samp[-3:]:
        print("     ", s, flush=True)
    setw(lambda k: base[k], "all matrices = Base", 1)
    setw(lambda k: inst[k], "all matrices = Instruct")
    setw(lambda k: alpha(k) * base[k], "alpha*Base (shrink only)")

    def iso(k):
        res = asr[k] - alpha(k) * base[k]
        n = torch.randn_like(res)
        return alpha(k) * base[k] + n / n.norm() * res.norm()
    torch.manual_seed(0)
    setw(iso, "alpha*Base + isotropic noise, same residual norm")

    def shuf(k):
        res = (asr[k] - alpha(k) * base[k]).flatten()
        return alpha(k) * base[k] + res[torch.randperm(res.numel())].reshape(res.shape if False else asr[k].shape)
    setw(shuf, "alpha*Base + shuffled residual")

    for t in (.9, .75, .5, .25):
        setw(lambda k, t=t: base[k] + t * (asr[k] - base[k]), f"interpolate Base -> ASR t={t}")

    def bandp(k, lo, hi):
        res = asr[k] - alpha(k) * base[k]
        U, S, Vh = torch.linalg.svd(base[k], full_matrices=False)
        a_, b_ = int(len(S) * lo), max(int(len(S) * hi), int(len(S) * lo) + 1)
        return alpha(k) * base[k] + U[:, a_:b_] @ U[:, a_:b_].T @ res @ Vh[a_:b_].T @ Vh[a_:b_]
    setw(lambda k: bandp(k, 0, .2), "alpha*Base + residual in top-20% band")
    setw(lambda k: bandp(k, .5, 1), "alpha*Base + residual in bottom-50% band")

    print("  -- restore parts of ASR to Base", flush=True)
    def layers_to_base(rng_):
        return lambda k: base[k] if int(k.split(".")[2]) in rng_ else asr[k]
    for name, rg in (("L0-2", range(0, 3)), ("L0-9", range(0, 10)), ("L10-18", range(10, 19)), ("L19-27", range(19, 28)), ("L27", range(27, 28))):
        fr = sum(asr[k].numel() for k in keys if int(k.split(".")[2]) in rg) / total
        setw(layers_to_base(rg), f"layers {name} = Base", fr)
    fr = sum(asr[k].numel() for k in keys if "self_attn" in k) / total
    setw(lambda k: base[k] if "self_attn" in k else asr[k], "attention matrices = Base", fr)
    setw(lambda k: base[k] if "mlp" in k else asr[k], "MLP matrices = Base", 1 - fr)

    def neurons(order, frac):
        def fn(k):
            if ".mlp." not in k:
                return asr[k]
            i = int(k.split(".")[2])
            idx = torch.as_tensor(order(i)[: int(INTER * frac)].copy())
            W = asr[k].clone()
            if "down_proj" in k:
                W[:, idx] = base[k][:, idx]
            else:
                W[idx] = base[k][idx]
            return W
        return fn
    mlp_share = 1 - fr
    orders = {"least plastic": lambda i: np.argsort(pn[i]), "most plastic": lambda i: np.argsort(-pn[i]), "random": lambda i: rng.permutation(INTER)}
    for frac in (.25, .5, .75):
        for name, order in orders.items():
            setw(neurons(order, frac), f"{int(frac*100)}% MLP neurons = Base: {name}", frac * mlp_share)

    def heads(order, frac):
        def fn(k):
            if "self_attn" not in k:
                return asr[k]
            i = int(k.split(".")[2])
            sel = order(i)[: int(16 * frac)]
            W = asr[k].clone()
            for h in sel:
                h = int(h)
                if "o_proj" in k:
                    W[:, h * 128:(h + 1) * 128] = base[k][:, h * 128:(h + 1) * 128]
                elif "q_proj" in k:
                    W[h * 128:(h + 1) * 128] = base[k][h * 128:(h + 1) * 128]
            return W
        return fn
    for frac in (.5,):
        setw(heads(lambda i: np.argsort(ph[i]), frac), f"{int(frac*100)}% heads (q,o) = Base: least plastic")
        setw(heads(lambda i: np.argsort(-ph[i]), frac), f"{int(frac*100)}% heads (q,o) = Base: most plastic")
    # Embeddings too.
    setw(lambda k: asr[k], "(matrices back to ASR)", 0)
    e = "model.language_model.embed_tokens.weight"
    keep = params[e].detach().clone()
    with torch.no_grad():
        params[e].copy_(base["model.embed_tokens.weight"].to(DEVICE))
    r, _ = transcribe(model, proc, cs)
    print(f"  {'embedding table = Base':52s}" + " ".join(f"{k} {v:6.3f}" for k, v in r.items()), flush=True)
    with torch.no_grad():
        params[e].copy_(keep)


def trace(a):
    model, proc = build()
    lm = model.model.language_model
    lm.config._attn_implementation = "eager"
    for l in lm.layers:
        l.self_attn.config._attn_implementation = "eager"
    cs = clips(a.ls)
    aid = model.config.audio_token_id
    W_U = model.lm_head.weight.detach()
    W_mean = W_U.mean(0)
    norm_w = lm.norm.weight.detach()
    O = [l.self_attn.o_proj.weight.detach() for l in lm.layers]
    D = [l.mlp.down_proj.weight.detach() for l in lm.layers]
    cap = {}
    for i, l in enumerate(lm.layers):
        l.self_attn.o_proj.register_forward_pre_hook(lambda m, x, i=i: cap.__setitem__(("head_in", i), x[0].detach()))
        l.mlp.down_proj.register_forward_pre_hook(lambda m, x, i=i: cap.__setitem__(("act", i), x[0].detach()))
        l.register_forward_pre_hook(lambda m, x, i=i: cap.__setitem__(("in", i), x[0].detach()))
    lm.norm.register_forward_pre_hook(lambda m, x: cap.__setitem__("final", x[0].detach()))
    dla_n, dla_h, fire, att_audio, att_first, att_rest, att_n = [np.zeros(s) for s in ((L, INTER), (L, 16), (L, INTER), (L, 16), (L, 16), (L, 16), (L, 16))]
    n_tok = 0
    stream = {"audio": np.zeros(L + 1), "prompt": np.zeros(L + 1), "first": np.zeros(L + 1), "text": np.zeros(L + 1)}
    stream_n = {k: 0 for k in stream}
    err, tot = 0., 0.
    spikes = []
    for c in cs:
        inp = proc.apply_transcription_request(audio=c["audio"]).to(DEVICE, torch.float32)
        prompt = inp["input_ids"]
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=128, do_sample=False)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=False, **kw)
        T, P = ids.shape[1], prompt.shape[1]
        final = cap["final"][0]
        rms = final.pow(2).mean(-1, keepdim=True).add(lm.norm.variance_epsilon).sqrt()
        pos = torch.arange(P - 1, T - 1, device=DEVICE)
        d = (W_U[ids[0, pos + 1]] - W_mean) * norm_w / rms[pos]
        total = (d * final[pos]).sum(-1)
        recon = torch.zeros_like(total)
        is_audio = (ids[0] == aid)
        for i in range(L):
            dO = d @ O[i]
            hi = cap[("head_in", i)][0, pos]
            per = (dO * hi).reshape(len(pos), 16, 128).sum(-1)
            act = cap[("act", i)][0, pos]
            pn = act * (d @ D[i])
            dla_h[i] += per.abs().sum(0).cpu().numpy()
            dla_n[i] += pn.abs().sum(0).cpu().numpy()
            thr = act.pow(2).mean().sqrt()
            fire[i] += (act.abs() > thr).sum(0).float().cpu().numpy()
            recon = recon + per.sum(-1) + pn.sum(-1)
            A = out.attentions[i][0][:, pos]  # heads, tgt positions, T
            att_audio[i] += A[:, :, is_audio].sum(-1).sum(-1).float().cpu().numpy()
            att_first[i] += A[:, :, 0].sum(-1).float().cpu().numpy()
            att_rest[i] += A[:, :, ~is_audio].sum(-1).sum(-1).float().cpu().numpy() - A[:, :, 0].sum(-1).float().cpu().numpy()
            att_n[i] += len(pos)
            s = cap[("in", i)][0]
            for name, mask in (("audio", is_audio), ("first", torch.arange(T, device=DEVICE) == 0)):
                stream[name][i] += s[mask].pow(2).mean().item() * int(mask.sum())
                stream_n[name] += int(mask.sum()) if i == 0 else 0
            mask = torch.zeros(T, dtype=torch.bool, device=DEVICE); mask[P:] = True
            stream["text"][i] += s[mask].pow(2).mean().item() * int(mask.sum())
            stream_n["text"] += int(mask.sum()) if i == 0 else 0
        spikes.append(cap[("in", 8)][0, 0, 35].item())
        recon = recon + (d * lm.embed_tokens(ids)[0, pos]).sum(-1)
        err += (recon - total).abs().sum().item()
        tot += total.abs().sum().item()
        n_tok += len(pos)
        cap.clear()
    print(f"### trace: {len(cs)} clips, {n_tok} transcript tokens; attribution recon error {err/tot:.2e}; mean dim35 at first token {np.mean(spikes):.1f}", flush=True)
    print("  stream rms by layer (audio positions | first token | transcript)")
    for i in (0, 2, 3, 8, 14, 20, 24, 27):
        print(f"   L{i:2d}  {np.sqrt(stream['audio'][i]/max(1,stream_n['audio'])):8.3f} | {np.sqrt(stream['first'][i]/max(1,stream_n['first'])):8.2f} | {np.sqrt(stream['text'][i]/max(1,stream_n['text'])):8.3f}")
    print("  where transcript-token attention goes (share, mean over heads; min-max across heads for audio)")
    print("   L    audio(min-max)        first-token  other-text")
    for i in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27):
        au = att_audio[i] / att_n[i]
        print(f"   {i:2d}  {au.mean():.3f} ({au.min():.3f}-{au.max():.3f})   {(att_first[i]/att_n[i]).mean():.3f}       {(att_rest[i]/att_n[i]).mean():.3f}")
    np.savez(os.path.join(a.dir, "asr_trace.npz"), dla_n=dla_n / n_tok, dla_h=dla_h / n_tok, fire=fire / n_tok, att_audio=att_audio / att_n, att_first=att_first / att_n)
    from scipy.stats import spearmanr
    z = np.load(os.path.join(a.dir, "plastic.Qwen3-ASR-0.6B.npz"))
    s = np.load(os.path.join(a.dir, "Qwen3-ASR-0.6B.stats.npz"))
    cats = ["code", "en_doc", "zh_doc", "spoken_en", "spoken_zh", "chat"]
    n_txt = sum(float(s[f"{c}/n_tgt"]) for c in cats)
    text_n = sum(s[f"{c}/dla_neuron"] for c in cats) / n_txt
    text_h = sum(s[f"{c}/dla_head"] for c in cats) / n_txt
    print("  link to plasticity (Spearman per layer): neuron plasticity vs audio-mode |DLA|, vs audio/text |DLA| ratio; head plasticity vs audio attention, vs audio-mode |DLA|")
    rows = []
    for i in range(L):
        ratio = (dla_n[i] / n_tok + 1e-9) / (text_n[i] + 1e-9)
        rows.append([spearmanr(z["pn"][i], dla_n[i])[0], spearmanr(z["pn"][i], ratio)[0], spearmanr(z["ph"][i], att_audio[i] / att_n[i])[0], spearmanr(z["ph"][i], dla_h[i])[0], spearmanr(dla_n[i], text_n[i])[0]])
        if i % 4 == 2 or i == 27:
            print(f"   L{i:2d} " + " ".join(f"{x:7.3f}" for x in rows[-1]))
    print("   mean " + " ".join(f"{x:7.3f}" for x in np.mean(rows, 0)) + "   (last column: audio-mode vs text-mode |DLA| correlation)")
    top = np.argsort(-(dla_n / n_tok).ravel())[:10]
    print("   top audio-mode neurons (layer, neuron, |DLA|, text-mode |DLA|, plasticity pct):")
    for t in top:
        i, j = np.unravel_index(t, dla_n.shape)
        pct = (z["pn"][i] < z["pn"][i, j]).mean()
        print(f"     L{i} n{j}  {dla_n[i,j]/n_tok:.3f}  {text_n[i,j]:.3f}  {pct:.2f}")


def quant(x, bits, group=32):
    """Block-wise symmetric quantization; 1 bit means sign times the block mean magnitude."""
    shape = x.shape
    f = x.flatten()
    n = f.numel() // group * group
    b = f[:n].reshape(-1, group)
    if bits == 1:
        q = torch.sign(b) * b.abs().mean(1, keepdim=True)
    else:
        levels = 2 ** (bits - 1) - 1
        scale = b.abs().amax(1, keepdim=True).clamp_min(1e-12)
        q = torch.round(b / scale * levels) / levels * scale
    out = f.clone()
    out[:n] = q.flatten()
    return out.reshape(shape)


def tf_targets(model, proc, cs):
    """The intact model's own raw output for each clip: a target that needs no text normalization."""
    out = []
    for c in cs:
        inp = proc.apply_transcription_request(audio=c["audio"]).to(DEVICE, torch.float32)
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=128, do_sample=False)
        out.append((inp, ids, inp["input_ids"].shape[1]))
    return out


def tf_ce(model, targets):
    tot, n, agree = 0., 0, 0
    for inp, ids, P in targets:
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        with torch.no_grad():
            lg = model(input_ids=ids, use_cache=False, **kw).logits[0]
        tgt = ids[0, P:]
        pred = lg[P - 1:-1]
        tot += torch.nn.functional.cross_entropy(pred, tgt, reduction="sum").item()
        agree += (pred.argmax(-1) == tgt).sum().item()
        n += len(tgt)
    return tot / n, agree / n


def graft2(a):
    model, proc = build()
    cs = clips(a.ls)
    params = dict(model.named_parameters())
    base = read_decoder("Qwen3-0.6B-Base")
    inst = read_decoder("Qwen3-0.6B")
    asr = {k: params["model.language_" + k].detach().cpu().clone() for k in base if "model.language_" + k in params}
    keys = [f"model.layers.{i}.{t}.weight" for i in range(L) for t in MATS]
    total = sum(asr[k].numel() for k in keys)
    z = np.load(os.path.join(a.dir, "plastic.Qwen3-ASR-0.6B.npz"))
    pn = z["pn"]
    rng = np.random.default_rng(0)
    targets = tf_targets(model, proc, cs)
    alpha = {k: float((asr[k] * base[k]).sum() / (base[k] * base[k]).sum()) for k in keys}

    def setw(fn):
        with torch.no_grad():
            for k in keys:
                params["model.language_" + k].copy_(fn(k).to(DEVICE))

    def run(fn, label, wer=True, note=""):
        t0 = time.time()
        setw(fn)
        ce, ag = tf_ce(model, targets)
        out = f"  {label:50s} CE {ce:7.3f} agree {ag:5.3f}"
        if wer:
            r, _ = transcribe(model, proc, cs, max_new=64)
            out += "  " + " ".join(f"{k[:6]} {v:5.3f}" for k, v in r.items())
        print(out + f"  {note} [{time.time()-t0:.0f}s]", flush=True)

    print("### graft2: teacher-forced CE and agreement against the intact ASR's own output, plus error rates", flush=True)
    run(lambda k: asr[k], "intact ASR")
    print("  -- how far back toward alpha*Base before it breaks (scale kept)", flush=True)
    for t in (.95, .9, .85, .8, .75, .7, .6, .5):
        run(lambda k, t=t: alpha[k] * base[k] + t * (asr[k] - alpha[k] * base[k]), f"residual scaled by t={t}", wer=t in (.9, .8, .7))
    print("  -- quantization of ASR itself (standalone baseline), decoder matrices only", flush=True)
    for b in (8, 6, 5, 4, 3, 2):
        run(lambda k, b=b: quant(asr[k], b), f"standalone Q{b}", note=f"{b} bits/weight")
    print("  -- delta against a Q4 base: alpha*Q4(ref) + Qb(W - alpha*Q4(ref)); cost = b bits on top of the shared base", flush=True)
    for ref_name, ref in (("Base", base), ("Instruct", inst)):
        q4 = {k: quant(ref[k], 4) for k in keys}
        a4 = {k: float((asr[k] * q4[k]).sum() / (q4[k] * q4[k]).sum()) for k in keys}
        for b in (4, 3, 2, 1):
            run(lambda k, b=b: a4[k] * q4[k] + quant(asr[k] - a4[k] * q4[k], b), f"delta from Q4 {ref_name}, Q{b} residual", note=f"{b} bits/weight + base")
    print("  -- restore parts to alpha*Base (scale kept)", flush=True)
    def scaled(k):
        return alpha[k] * base[k]
    for name, rg in (("L0-2", range(0, 3)), ("L0-9", range(0, 10)), ("L10-18", range(10, 19)), ("L19-27", range(19, 28)), ("L27", range(27, 28))):
        fr = sum(asr[k].numel() for k in keys if int(k.split(".")[2]) in rg) / total
        run(lambda k, rg=rg: scaled(k) if int(k.split(".")[2]) in rg else asr[k], f"layers {name} = alpha*Base", note=f"shared {fr:.1%}")
    fr = sum(asr[k].numel() for k in keys if "self_attn" in k) / total
    run(lambda k: scaled(k) if "self_attn" in k else asr[k], "attention = alpha*Base", note=f"shared {fr:.1%}")
    run(lambda k: scaled(k) if ".mlp." in k else asr[k], "MLP = alpha*Base", note=f"shared {1-fr:.1%}")
    def neurons(order, frac):
        def fn(k):
            if ".mlp." not in k:
                return asr[k]
            i = int(k.split(".")[2])
            idx = torch.as_tensor(order(i)[: int(INTER * frac)].copy())
            W = asr[k].clone()
            s_ = scaled(k)
            if "down_proj" in k:
                W[:, idx] = s_[:, idx]
            else:
                W[idx] = s_[idx]
            return W
        return fn
    orders = {"least plastic": lambda i: np.argsort(pn[i]), "most plastic": lambda i: np.argsort(-pn[i]), "random": lambda i: rng.permutation(INTER)}
    for frac in (.1, .25, .5, .75):
        for name, order in orders.items():
            run(neurons(order, frac), f"{int(frac*100)}% neurons = alpha*Base: {name}", wer=False, note=f"shared {frac*(1-fr):.1%}")
    print("  -- single-layer scan: only this layer = alpha*Base (matrices), others intact", flush=True)
    for i in range(L):
        run(lambda k, i=i: scaled(k) if int(k.split(".")[2]) == i else asr[k], f"layer {i}", wer=False)
    print("  -- single-block scan over all layers: one matrix type = alpha*Base", flush=True)
    for t in MATS:
        run(lambda k, t=t: scaled(k) if t in k else asr[k], f"all {t}", wer=False)
    setw(lambda k: asr[k])
    e = "model.language_model.embed_tokens.weight"
    keep = params[e].detach().clone()
    n_real = 151643
    print("  -- embedding table", flush=True)
    with torch.no_grad():
        mixed = base["model.embed_tokens.weight"].to(DEVICE).clone()
        mixed[n_real:] = keep[n_real:]
        params[e].copy_(mixed)
    ce, ag = tf_ce(model, targets)
    r, _ = transcribe(model, proc, cs, max_new=64)
    print(f"  {'ordinary rows = Base, special rows = ASR':50s} CE {ce:7.3f} agree {ag:5.3f}  " + " ".join(f"{k[:6]} {v:5.3f}" for k, v in r.items()), flush=True)
    with torch.no_grad():
        a_e = float((keep[:n_real] * mixed[:n_real]).sum() / (mixed[:n_real] * mixed[:n_real]).sum())
        mixed[:n_real] *= a_e
        params[e].copy_(mixed)
    ce, ag = tf_ce(model, targets)
    r, _ = transcribe(model, proc, cs, max_new=64)
    print(f"  {f'same, ordinary rows scaled by alpha={a_e:.3f}':50s} CE {ce:7.3f} agree {ag:5.3f}  " + " ".join(f"{k[:6]} {v:5.3f}" for k, v in r.items()), flush=True)
    with torch.no_grad():
        params[e].copy_(keep)


def qdelta(x, bits, group=32):
    """Residual quantizer: 4-level midrise for 2 bits (no zero level), otherwise as quant()."""
    if bits != 2:
        return quant(x, bits, group)
    shape = x.shape
    f = x.flatten()
    n = f.numel() // group * group
    b = f[:n].reshape(-1, group)
    s = b.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    q = torch.where(b.abs() > .9674 * s, 1.5104 * s, .4528 * s) * torch.sign(b)
    out = f.clone()
    out[:n] = q.flatten()
    return out.reshape(shape)


def graft3(a):
    model, proc = build()
    lm = model.model.language_model
    cs = clips(a.ls)
    params = dict(model.named_parameters())
    base = read_decoder("Qwen3-0.6B-Base")
    asr = {k: params["model.language_" + k].detach().cpu().clone() for k in base if "model.language_" + k in params}
    keys = [f"model.layers.{i}.{t}.weight" for i in range(L) for t in MATS]
    total = sum(asr[k].numel() for k in keys)
    z = np.load(os.path.join(a.dir, "plastic.Qwen3-ASR-0.6B.npz"))
    pn = z["pn"]
    scan = {}
    for ln in open(os.path.join(ROOT, "asr_graft2.txt"), encoding="utf-8"):
        m = re.match(r"\s+layer (\d+)\s+CE\s+([\d.]+)", ln)
        if m:
            scan[int(m.group(1))] = float(m.group(2))
    order = sorted((i for i in scan if i not in (0, 27)), key=lambda i: scan[i])
    print("layers by single-restore damage (low to high):", order, flush=True)
    targets = tf_targets(model, proc, cs)
    q4 = {k: quant(base[k], 4) for k in keys}
    a4 = {k: float((asr[k] * q4[k]).sum() / (q4[k] * q4[k]).sum()) for k in keys}
    s4 = {k: a4[k] * q4[k] for k in keys}
    res = {k: asr[k] - s4[k] for k in keys}

    def setw(fn):
        with torch.no_grad():
            for k in keys:
                params["model.language_" + k].copy_(fn(k).to(DEVICE))

    def run(fn, label, bits=None, wer=True):
        t0 = time.time()
        setw(fn)
        ce, ag = tf_ce(model, targets)
        out = f"  {label:58s} CE {ce:7.3f} agree {ag:5.3f}"
        if wer:
            r, _ = transcribe(model, proc, cs, max_new=64)
            out += "  " + " ".join(f"{k[:6]} {v:5.3f}" for k, v in r.items())
        if bits is not None:
            out += f"  avg {bits:.2f} bits"
        print(out + f"  [{time.time()-t0:.0f}s]", flush=True)

    def avg(bits_of):
        return sum(bits_of(k) * asr[k].numel() for k in keys) / total

    print("### graft3: residual coding on a Q4 Base. avg bits = residual bits per decoder matrix weight, on top of the shared base; scale side info not counted", flush=True)
    run(lambda k: asr[k], "intact ASR")
    run(lambda k: quant(asr[k], 4), "standalone Q4 (fair baseline, no shared base)", 4.0)
    run(lambda k: s4[k] + qdelta(res[k], 3), "all residuals Q3", 3.0)
    run(lambda k: s4[k] + qdelta(res[k], 2), "all residuals 2-bit midrise", 2.0)
    print("  -- spend bits by importance", flush=True)

    def neuron_mix(frac, b_keep=3, attn_bits=3):
        def fn(k):
            i = int(k.split(".")[2])
            if "self_attn" in k:
                return s4[k] + qdelta(res[k], attn_bits)
            W = s4[k] + qdelta(res[k], b_keep)
            idx = torch.as_tensor(np.argsort(-pn[i])[: int(INTER * frac)].copy())
            if "down_proj" in k:
                W[:, idx] = s4[k][:, idx]
            else:
                W[idx] = s4[k][idx]
            return W
        return fn
    for frac in (.25, .5, .75):
        run(neuron_mix(frac), f"most-plastic {int(frac*100)}% neurons 0 bits, rest Q3", avg(lambda k: 3 if "self_attn" in k else 3 * (1 - frac)))
        run(neuron_mix(frac, attn_bits=2), f"same, attention at 2-bit", avg(lambda k: 2 if "self_attn" in k else 3 * (1 - frac)))
    def random_mix(frac, seed=0):
        rng = np.random.default_rng(seed)
        perm = {i: rng.permutation(INTER) for i in range(L)}
        def fn(k):
            i = int(k.split(".")[2])
            if "self_attn" in k:
                return s4[k] + qdelta(res[k], 3)
            W = s4[k] + qdelta(res[k], 3)
            idx = torch.as_tensor(perm[i][: int(INTER * frac)].copy())
            if "down_proj" in k:
                W[:, idx] = s4[k][:, idx]
            else:
                W[idx] = s4[k][idx]
            return W
        return fn
    run(random_mix(.5), "control: random 50% neurons 0 bits, rest Q3", avg(lambda k: 3 if "self_attn" in k else 1.5))
    print("  -- spend bits by layer: k least-damaging layers restored together (0 bits), others Q3", flush=True)
    for n in (2, 4, 6, 8, 10, 12, 14, 18, 22, 26):
        sel = set(order[:n])
        run(lambda k, sel=sel: s4[k] if int(k.split(".")[2]) in sel else s4[k] + qdelta(res[k], 3), f"{n} layers 0 bits", avg(lambda k: 0 if int(k.split(".")[2]) in sel else 3), wer=n in (4, 8, 12))
    odd = set(range(1, 27, 2))
    run(lambda k: s4[k] if int(k.split(".")[2]) in odd else s4[k] + qdelta(res[k], 3), "every other layer (odd 1-25) 0 bits", avg(lambda k: 0 if int(k.split(".")[2]) in odd else 3))
    print("  -- text- and audio-specific ablation, excluding the spike layer L2 and output layer L27", flush=True)
    setw(lambda k: asr[k])
    s_ = np.load(os.path.join(a.dir, "Qwen3-ASR-0.6B.stats.npz"))
    cats = ["code", "en_doc", "zh_doc", "spoken_en", "spoken_zh", "chat"]
    text = sum(s_[f"{c}/dla_neuron"] for c in cats) / sum(float(s_[f"{c}/n_tgt"]) for c in cats)
    audio = np.load(os.path.join(a.dir, "asr_trace.npz"))["dla_n"]
    def specific(i, x, y):
        ratio = (x[i] + 1e-6) / (y[i] + 1e-6)
        ratio[x[i] < np.median(x[i])] = 0
        return np.argsort(-ratio)
    rng = np.random.default_rng(1)
    for name, order_fn in (("audio-specific", lambda i: specific(i, audio, text)), ("text-specific", lambda i: specific(i, text, audio)), ("random", lambda i: rng.permutation(INTER))):
        hooks = []
        for i, l in enumerate(lm.layers):
            if i in (2, 27):
                continue
            idx = torch.tensor(order_fn(i)[: int(INTER * .01)].copy(), device=DEVICE)
            def pre(m, x, idx=idx):
                v = x[0].clone()
                v[:, :, idx] = 0
                return (v,)
            hooks.append(l.mlp.down_proj.register_forward_pre_hook(pre))
        ce, ag = tf_ce(model, targets)
        r, _ = transcribe(model, proc, cs, max_new=64)
        print(f"  {f'zero 1% per layer, no L2/L27: {name}':58s} CE {ce:7.3f} agree {ag:5.3f}  " + " ".join(f"{k[:6]} {v:5.3f}" for k, v in r.items()), flush=True)
        for h in hooks:
            h.remove()


def ablate(a):
    """Zero decoder MLP neurons during transcription: does plasticity or audio specificity predict damage?"""
    model, proc = build()
    lm = model.model.language_model
    cs = clips(a.ls)
    z = np.load(os.path.join(a.dir, "plastic.Qwen3-ASR-0.6B.npz"))
    tr = np.load(os.path.join(a.dir, "asr_trace.npz"))
    s = np.load(os.path.join(a.dir, "Qwen3-ASR-0.6B.stats.npz"))
    cats = ["code", "en_doc", "zh_doc", "spoken_en", "spoken_zh", "chat"]
    text = sum(s[f"{c}/dla_neuron"] for c in cats) / sum(float(s[f"{c}/n_tgt"]) for c in cats)
    audio = tr["dla_n"]
    rng = np.random.default_rng(0)

    def stratified(i, least):
        q = np.argsort(np.argsort(audio[i])) * 10 // INTER
        rank = np.zeros(INTER)
        for b in range(10):
            m = q == b
            v = z["pn"][i][m] * (1 if least else -1)
            rank[m] = np.argsort(np.argsort(v)) / m.sum()
        return np.argsort(rank)

    def specific(i, a_, b_):
        # High on one mode and low on the other, restricted to neurons above the layer's median on the first mode.
        ratio = (a_[i] + 1e-6) / (b_[i] + 1e-6)
        ratio[a_[i] < np.median(a_[i])] = 0
        return np.argsort(-ratio)

    conds = [("random", .1, lambda i: rng.permutation(INTER)),
             ("top audio-mode |DLA|", .01, lambda i: np.argsort(-audio[i])),
             ("random", .01, lambda i: rng.permutation(INTER)),
             ("audio-specific (audio/text |DLA| ratio)", .01, lambda i: specific(i, audio, text)),
             ("text-specific (text/audio |DLA| ratio)", .01, lambda i: specific(i, text, audio)),
             ("least plastic", .1, lambda i: np.argsort(z["pn"][i])),
             ("most plastic", .1, lambda i: np.argsort(-z["pn"][i])),
             ("least plastic, audio |DLA|-matched", .1, lambda i: stratified(i, True)),
             ("most plastic, audio |DLA|-matched", .1, lambda i: stratified(i, False))]
    print("### ablate decoder neurons during transcription: error rate (WER en, CER zh)", flush=True)
    r, _ = transcribe(model, proc, cs)
    print(f"  {'intact':52s}" + " ".join(f"{k} {v:6.3f}" for k, v in r.items()), flush=True)
    for name, frac, order in conds:
        hooks = []
        for i, l in enumerate(lm.layers):
            idx = torch.tensor(order(i)[: int(INTER * frac)].copy(), device=DEVICE)
            def pre(m, x, idx=idx):
                v = x[0].clone()
                v[:, :, idx] = 0
                return (v,)
            hooks.append(l.mlp.down_proj.register_forward_pre_hook(pre))
        t0 = time.time()
        r, _ = transcribe(model, proc, cs)
        print(f"  {f'zero {frac:.0%} per layer: {name}':52s}" + " ".join(f"{k} {v:6.3f}" for k, v in r.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
        for h in hooks:
            h.remove()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["graft", "graft2", "graft3", "trace", "smoke", "ablate"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ls", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "smoke":
        model, proc = build()
        cs = clips(2)
        c = cs[:2] + cs[2:4] + cs[-2:]
        r, s = transcribe(model, proc, c)
        print(r)
        for x in s:
            print(x)
    else:
        globals()[a.cmd](a)
