# Are "audio-specific" neurons about audio, or about the transcription format? Split damage by token type,
# inspect outputs, and test the same ablation on plain text.
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from asr_eval import build, clips, tf_targets, INTER, L
from common import DEVICE, corpus, tokenizer

d = sys.argv[1]
model, proc = build()
lm = model.model.language_model
cs = clips(30)
targets = tf_targets(model, proc, cs)
s_ = np.load(os.path.join(d, "Qwen3-ASR-0.6B.stats.npz"))
cats = ["code", "en_doc", "zh_doc", "spoken_en", "spoken_zh", "chat"]
text = sum(s_[f"{c}/dla_neuron"] for c in cats) / sum(float(s_[f"{c}/n_tgt"]) for c in cats)
audio = np.load(os.path.join(d, "asr_trace.npz"))["dla_n"]
tok = proc.tokenizer
marker = tok.convert_tokens_to_ids("<asr_text>")
segs = corpus(tokenizer(), 8)


def specific(i, x, y):
    ratio = (x[i] + 1e-6) / (y[i] + 1e-6)
    ratio[x[i] < np.median(x[i])] = 0
    return np.argsort(-ratio)


def split_eval():
    pre, body, eos = [0., 0, 0], [0., 0, 0], [0., 0, 0]
    for inp, ids, P in targets:
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        with torch.no_grad():
            lg = model(input_ids=ids, use_cache=False, **kw).logits[0]
        tgt = ids[0, P:]
        pred = lg[P - 1:-1]
        ce = F.cross_entropy(pred, tgt, reduction="none")
        ok = pred.argmax(-1) == tgt
        m = (tgt == marker).nonzero()
        cut = int(m[0]) + 1 if len(m) else 0
        for r, sel in ((pre, range(0, cut)), (body, range(cut, len(tgt) - 1)), (eos, range(len(tgt) - 1, len(tgt)))):
            for j in sel:
                r[0] += ce[j].item(); r[1] += int(ok[j]); r[2] += 1
    return [(r[0] / max(1, r[2]), r[1] / max(1, r[2]), r[2]) for r in (pre, body, eos)]


def text_eval():
    tot, n = {}, {}
    for c, ss in segs.items():
        for ids in ss:
            x = torch.tensor([ids], device=DEVICE)
            with torch.no_grad():
                lg = model(input_ids=x, use_cache=False).logits[0]
            tot[c] = tot.get(c, 0) + F.cross_entropy(lg[1:-1], x[0, 2:], reduction="sum").item()
            n[c] = n.get(c, 0) + x.shape[1] - 2
    return {c: tot[c] / n[c] for c in tot}


def report(label):
    (pc, pa, pn), (bc, ba, bn), (ec, ea, en_) = split_eval()
    print(f"\n{label}\n  prefix CE {pc:6.3f} agree {pa:.3f} (n={pn}) | transcript CE {bc:6.3f} agree {ba:.3f} (n={bn}) | end CE {ec:6.3f} agree {ea:.3f} (n={en_})")
    print("  text CE: " + " ".join(f"{c} {v:.3f}" for c, v in text_eval().items()), flush=True)
    for c in (cs[0], cs[31], cs[-1]):
        inp = proc.apply_transcription_request(audio=c["audio"]).to(DEVICE, torch.float32)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=40, do_sample=False)
        print("   gen:", repr(proc.decode(o[:, inp["input_ids"].shape[1]:])[0][:110]), flush=True)


rng = np.random.default_rng(1)
report("intact")
for name, fn in (("audio-specific 1% (no L2/L27)", lambda i: specific(i, audio, text)), ("text-specific 1% (no L2/L27)", lambda i: specific(i, text, audio)), ("random 1% (no L2/L27)", lambda i: rng.permutation(INTER))):
    hooks = []
    sel = {}
    for i, l in enumerate(lm.layers):
        if i in (2, 27):
            continue
        idx = fn(i)[: int(INTER * .01)]
        sel[i] = idx
        t = torch.tensor(idx.copy(), device=DEVICE)
        def pre(m, x, t=t):
            v = x[0].clone()
            v[:, :, t] = 0
            return (v,)
        hooks.append(l.mlp.down_proj.register_forward_pre_hook(pre))
    report(name)
    for h in hooks:
        h.remove()
    if "audio" in name:
        by_layer = sorted(((audio[i][sel[i]].sum() / audio[i].sum(), i) for i in sel), reverse=True)[:6]
        print("  share of the layer's audio |DLA| held by these neurons, top layers:", [(f"L{i}", round(float(v), 3)) for v, i in by_layer])
