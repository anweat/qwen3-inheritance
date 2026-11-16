# E5: does "shared Q4 LLM base + 2-bit ASR residual" survive real kernels on an on-device runtime (Apple M4, MLX)?
# E2/E3 only simulated the quantizers in fp32 torch (symmetric Q4, Lloyd-Max 2-bit, scale side info not counted, runtime cost
# not measured). Here the Qwen3-ASR-0.6B text decoder runs in MLX with packed weights and mx.quantized_matmul; the audio
# tower output and the prompt (language forced as in E3) come from prep.py and are the same for every scheme.
#   python run_mlx.py run      # 800 clips, all schemes interleaved per clip (rotating order), greedy, max 200 new tokens
#   python run_mlx.py bench    # decoder-only and lm_head timing per generated token, interleaved, at a fixed context
# Then score with summary.py (.venv, needs the tokenizer).
#
# Schemes (decoder matrices q,k,v,o,gate,up,down only; embedding / tied lm_head and norms stay bf16 in every scheme, as in E3):
#   intact      bf16 dense
#   q4_gG       standalone MLX affine 4-bit, group G (mx.quantize: per-group min/max scale and bias)
#   res_gG      Qwen3-0.6B (Instruct) quantized as q4_gG = the shared LLM base, plus the ASR residual
#               R = W_asr - a * deq(Q4(W_inst)) (a: one least-squares scalar per matrix, as in E3) coded with 2 bits per
#               weight, group G: uniform 4-level midrise (q - 1.5) * d, d per group picked from c * RMS(group),
#               c in 0.70..1.40 by minimum squared error. Run as y = a * qmm4(x, base) + qmm2(x, R): two kernels, no fusion.
#   G in {32, 64}; 32 matches E3, 64 is the MLX default and halves the side information.
#
# Criteria, fixed before any MLX scheme decoded a clip:
#   Accuracy (each G): res_gG against q4_gG on the 800 clips, paired by clip: PASS if the pooled error-rate difference has a
#     95% bootstrap upper bound <= +0.5 percentage points and no set has a point estimate worse than +2 points (E3's rule).
#   Runtime (each G, batch 1, median over clips of the interleaved run): cost ACCEPTABLE if the end-to-end time per clip
#     (audio tower from prep.py + decoder prefill + decode) of res_gG is <= 1.25 x q4_gG. Decode ms per token and the
#     decoder-only ratio from `bench` are reported, not judged (a pure bandwidth model predicts about (6 + 2s) / (4 + s),
#     s = side bits per weight, for the decoder alone).
#   Memory: bytes of decoder matrices including scales and biases, counted from the arrays; reported for "LLM Q4 + separate
#     ASR Q4" vs "LLM Q4 shared + ASR residual". Not judged.
#   Sanity (reported): intact MLX bf16 against E3's torch fp32 forced-language intact run, error rate and identical outputs.
import json, os, sys, time
import numpy as np
import mlx.core as mx

PREP = os.path.expanduser("~/.cache/qwen-all/e5/prep")
OUT = os.path.expanduser("~/.cache/qwen-all/e5")
ASR = os.path.expanduser("~/.cache/qwen-all/Qwen3-ASR-0.6B/model.safetensors")
INST = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
L, H, KV, HD, EPS, THETA = 28, 16, 8, 128, 1e-6, 1e6
MAX_NEW = 200
GROUPS = (32, 64)


def quant2(r, G):
    """2-bit uniform midrise with a per-group step chosen by minimum squared error; returns packed, scales, biases."""
    o, i = r.shape
    b = r.astype(mx.float32).reshape(o, i // G, G)
    rms = mx.sqrt(mx.mean(b * b, axis=-1, keepdims=True)) + 1e-12
    best_e, best_d = None, None
    for c in np.arange(0.70, 1.401, 0.05):
        d = (float(c) * rms).astype(mx.bfloat16).astype(mx.float32)
        q = mx.clip(mx.floor(b / d) + 2, 0, 3)
        e = mx.sum(((q - 1.5) * d - b) ** 2, axis=-1, keepdims=True)
        if best_e is None:
            best_e, best_d = e, d
        else:
            m = e < best_e
            best_e, best_d = mx.where(m, e, best_e), mx.where(m, d, best_d)
    d = best_d
    q = mx.clip(mx.floor(b / d) + 2, 0, 3).astype(mx.uint32).reshape(o, i // 16, 16)
    packed = mx.sum(q << (2 * mx.arange(16, dtype=mx.uint32)), axis=-1).astype(mx.uint32)
    scales = d.reshape(o, i // G).astype(mx.bfloat16)
    biases = (-1.5 * d).reshape(o, i // G).astype(mx.bfloat16)
    # check the packing against MLX's own dequantizer
    ref = ((q.reshape(o, i // G, G).astype(mx.float32) - 1.5) * d).reshape(o, i)
    got = mx.dequantize(packed, scales, biases, group_size=G, bits=2).astype(mx.float32)
    assert float(mx.max(mx.abs(ref - got))) < 1e-2 * float(mx.max(mx.abs(ref))) + 1e-6
    return packed, scales, biases


def load_weights():
    a = mx.load(ASR)
    pre = "thinker.model."
    asr = {k[len(pre):]: v for k, v in a.items() if k.startswith(pre)}
    snap = os.path.join(INST, os.listdir(INST)[0], "model.safetensors")
    inst = {k[len("model."):]: v for k, v in mx.load(snap).items() if k.startswith("model.")}
    return asr, inst


def build_schemes(asr, inst):
    keys = [f"layers.{i}.{t}.weight" for i in range(L) for t in MATS]
    S = {"intact": {k: ("dense", asr[k]) for k in keys}}
    stats = {}
    for G in GROUPS:
        q4, res = {}, {}
        e_q4 = e_res = n2 = 0.0
        for k in keys:
            w = asr[k]
            wq = mx.quantize(w, group_size=G, bits=4)
            q4[k] = ("q4", wq)
            bq = mx.quantize(inst[k], group_size=G, bits=4)
            db = mx.dequantize(*bq, group_size=G, bits=4).astype(mx.float32)
            wf = w.astype(mx.float32)
            alpha = float(mx.sum(wf * db) / mx.sum(db * db))
            rq = quant2(wf - alpha * db, G)
            res[k] = ("res", bq, alpha, rq)
            dq = mx.dequantize(*wq, group_size=G, bits=4).astype(mx.float32)
            dr = alpha * db + mx.dequantize(*rq, group_size=G, bits=2).astype(mx.float32)
            e_q4 += float(mx.sum((dq - wf) ** 2)); e_res += float(mx.sum((dr - wf) ** 2)); n2 += float(mx.sum(wf * wf))
            mx.eval(q4[k][1], rq)
        S[f"q4_g{G}"], S[f"res_g{G}"] = q4, res
        stats[G] = dict(rel_err_q4=(e_q4 / n2) ** .5, rel_err_res=(e_res / n2) ** .5)
    return S, stats


def nbytes(entry):
    kind = entry[0]
    if kind == "dense":
        return entry[1].nbytes
    if kind == "q4":
        return sum(x.nbytes for x in entry[1])
    return dict(base=sum(x.nbytes for x in entry[1]), residual=sum(x.nbytes for x in entry[3]))


def make_lin(entry, G):
    kind = entry[0]
    if kind == "dense":
        w = entry[1]
        return lambda x: x @ w.T
    if kind == "q4":
        wq, s, b = entry[1]
        return lambda x: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=G, bits=4)
    (bw, bs, bb), alpha, (rw, rs, rb) = entry[1], entry[2], entry[3]
    a = mx.array(alpha, dtype=mx.bfloat16)
    return lambda x: a * mx.quantized_matmul(x, bw, bs, bb, transpose=True, group_size=G, bits=4) + \
        mx.quantized_matmul(x, rw, rs, rb, transpose=True, group_size=G, bits=2)


class Decoder:
    def __init__(self, name, mats, asr):
        G = int(name.split("_g")[1]) if "_g" in name else None
        self.lin = [{t: make_lin(mats[f"layers.{i}.{t}.weight"], G) for t in MATS} for i in range(L)]
        self.n = [{t: asr[f"layers.{i}.{t}.weight"] for t in
                   ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")} for i in range(L)]
        self.norm = asr["norm.weight"]
        self.emb = asr["embed_tokens.weight"]
        # Decode steps (T = 1) run the fixed-shape parts through mx.compile, which fuses elementwise ops and cuts per-op
        # dispatch; the cache update, rope offset and attention stay outside. Prefill (varying T) runs uncompiled.
        self.cpre = [mx.compile(lambda x, i=i: self._pre(i, x)) for i in range(L)]
        self.cpost = [mx.compile(lambda x, o, i=i: self._post(i, x, o)) for i in range(L)]

    def _pre(self, i, x):
        lin, n = self.lin[i], self.n[i]
        B, T, _ = x.shape
        h = mx.fast.rms_norm(x, n["input_layernorm"], EPS)
        q = lin["self_attn.q_proj"](h).reshape(B, T, H, HD)
        k = lin["self_attn.k_proj"](h).reshape(B, T, KV, HD)
        v = lin["self_attn.v_proj"](h).reshape(B, T, KV, HD)
        q = mx.fast.rms_norm(q, n["self_attn.q_norm"], EPS).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(k, n["self_attn.k_norm"], EPS).transpose(0, 2, 1, 3)
        return q, k, v.transpose(0, 2, 1, 3)

    def _post(self, i, x, o):
        lin, n = self.lin[i], self.n[i]
        B, _, T, _ = o.shape
        x = x + lin["self_attn.o_proj"](o.transpose(0, 2, 1, 3).reshape(B, T, H * HD))
        h = mx.fast.rms_norm(x, n["post_attention_layernorm"], EPS)
        g = lin["mlp.gate_proj"](h)
        return x + lin["mlp.down_proj"](mx.sigmoid(g) * g * lin["mlp.up_proj"](h))

    def layers(self, x, cache, off):
        T = x.shape[1]
        pre, post = (self.cpre, self.cpost) if T == 1 else ([lambda x, i=i: self._pre(i, x) for i in range(L)],
                                                            [lambda x, o, i=i: self._post(i, x, o) for i in range(L)])
        for i in range(L):
            q, k, v = pre[i](x)
            q = mx.fast.rope(q, HD, traditional=False, base=THETA, scale=1.0, offset=off)
            k = mx.fast.rope(k, HD, traditional=False, base=THETA, scale=1.0, offset=off)
            ck, cv = cache[i]
            ck[:, :, off:off + T, :] = k
            cv[:, :, off:off + T, :] = v
            o = mx.fast.scaled_dot_product_attention(q, ck[:, :, :off + T, :], cv[:, :, :off + T, :], scale=HD ** -.5,
                                                     mask="causal" if T > 1 else None)
            x = post[i](x, o)
        return x

    def head(self, x):
        return mx.fast.rms_norm(x[:, -1:, :], self.norm, EPS) @ self.emb.T

    def new_cache(self, n):
        return [(mx.zeros((1, KV, n, HD), dtype=mx.bfloat16), mx.zeros((1, KV, n, HD), dtype=mx.bfloat16)) for _ in range(L)]

    def embed(self, ids, aud, aid):
        x = self.emb[mx.array(ids)]
        pos = np.where(ids == aid)[0]
        x[mx.array(pos)] = mx.array(aud).astype(mx.bfloat16)
        return x[None]

    def transcribe(self, ids, aud, aid, eos):
        cache = self.new_cache(len(ids) + MAX_NEW + 1)
        t0 = time.perf_counter()
        x = self.embed(ids, aud, aid)
        tok = mx.argmax(self.head(self.layers(x, cache, 0)), axis=-1)
        mx.eval(tok)
        t1 = time.perf_counter()
        # mlx_lm-style pipelining: build and launch step n+1 before reading token n, so graph construction overlaps the GPU.
        # One step past the end is computed and thrown away, the same for every scheme.
        out, off = [], len(ids)
        while True:
            nxt = mx.argmax(self.head(self.layers(self.emb[tok.reshape(1, 1)], cache, off)), axis=-1)
            mx.async_eval(nxt)
            off += 1
            t = int(tok.item())
            if t in eos or len(out) >= MAX_NEW:
                break
            out.append(t)
            tok = nxt
        mx.eval(nxt)
        t2 = time.perf_counter()
        return out, (t1 - t0) * 1000, (t2 - t1) * 1000


def setup():
    asr, inst = load_weights()
    S, stats = build_schemes(asr, inst)
    keys = list(S["intact"])
    mem = {}
    for name, mats in S.items():
        tot = {}
        for k in keys:
            b = nbytes(mats[k])
            for part, v in (b.items() if isinstance(b, dict) else [("all", b)]):
                tot[part] = tot.get(part, 0) + v
        mem[name] = tot
    mem["embedding_bf16"] = asr["embed_tokens.weight"].nbytes
    dec = {name: Decoder(name, mats, asr) for name, mats in S.items()}
    return dec, stats, mem


def main():
    mode = sys.argv[1]
    idx = json.load(open(os.path.join(PREP, "index.json")))
    aid, eos = idx["audio_token_id"], set(idx["eos"])
    dec, stats, mem = setup()
    names = list(dec)
    print("quantization relative error", json.dumps(stats), "\nbytes", json.dumps(mem), flush=True)
    json.dump(dict(stats=stats, mem=mem), open(os.path.join(OUT, "weights_info.json"), "w"))
    clips = idx["clips"]
    z = lambda c: np.load(os.path.join(PREP, f"{c['i']:04d}.npz"))
    for c in clips[:2]:  # warm-up: kernels compile on first use
        d = z(c)
        for n in names:
            dec[n].transcribe(d["ids"], d["aud"], aid, eos)
    if mode == "run":
        path = os.path.join(OUT, "run.jsonl")
        done = set()
        if os.path.exists(path):
            done = {(r["scheme"], r["i"]) for r in map(json.loads, open(path))}
        with open(path, "a") as f:
            for c in clips:
                d = z(c)
                order = names[c["i"] % len(names):] + names[:c["i"] % len(names)]
                for n in order:
                    if (n, c["i"]) in done:
                        continue
                    toks, pre, dms = dec[n].transcribe(d["ids"], d["aud"], aid, eos)
                    f.write(json.dumps(dict(scheme=n, i=c["i"], toks=toks, prefill_ms=pre, decode_ms=dms)) + "\n")
                f.flush()
                if c["i"] % 50 == 0:
                    print(c["i"], flush=True)
        print("RUNDONE", flush=True)
    elif mode == "bench":
        # one generated token at a fixed context: decoder layers only, and lm_head only; interleaved, 300 rounds
        d = z(clips[0])
        caches, xs = {}, {}
        for n in names:
            caches[n] = dec[n].new_cache(len(d["ids"]) + 8)
            mx.eval(dec[n].layers(dec[n].embed(d["ids"], d["aud"], aid), caches[n], 0))
        off = len(d["ids"])
        x1 = dec["intact"].emb[mx.array([[1000]])]
        t = {n: dict(layers=[], head=[]) for n in names}
        for r in range(300):
            for n in (names[r % len(names):] + names[:r % len(names)]):
                t0 = time.perf_counter(); y = dec[n].layers(x1, caches[n], off); mx.eval(y)
                t1 = time.perf_counter(); mx.eval(dec[n].head(y)); t2 = time.perf_counter()
                t[n]["layers"].append((t1 - t0) * 1000); t[n]["head"].append((t2 - t1) * 1000)
        res = {n: {k: float(np.median(v)) for k, v in t[n].items()} for n in names}
        print(json.dumps(res, indent=1), flush=True)
        json.dump(dict(context=off, ms_per_token=res), open(os.path.join(OUT, "bench.json"), "w"), indent=1)
        print("BENCHDONE", flush=True)


if __name__ == "__main__":
    main()
