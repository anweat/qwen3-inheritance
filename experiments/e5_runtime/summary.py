# Applies the criteria in the run_mlx.py header. Run with .venv (tokenizer, E3 text normalization).
#   python summary.py > results.txt
import json, os, sys
from collections import defaultdict
import numpy as np
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e3_real_speech"))
sys.path.insert(0, os.path.join(HERE, "..", "e2_activation"))
from validate import norm
from asr_eval import edits

E5 = os.path.expanduser("~/.cache/qwen-all/e5")
idx = json.load(open(os.path.join(E5, "prep", "index.json")))
clips = {c["i"]: c for c in idx["clips"]}
tk = Tokenizer.from_file(os.path.join(snapshot_download("Qwen/Qwen3-ASR-0.6B-hf", allow_patterns=["*.json"]), "tokenizer.json"))
R = defaultdict(dict)
for r in map(json.loads, open(os.path.join(E5, "run.jsonl"))):
    c = clips[r["i"]]
    hyp = tk.decode(r["toks"], skip_special_tokens=True)
    ref_t, hyp_t = norm(c["text"], c["unit"]), norm(hyp, c["unit"])
    R[r["scheme"]][r["i"]] = dict(r, hyp=hyp, err=edits(ref_t, hyp_t), n=len(ref_t), set=c["set"], n_new=len(r["toks"]))
S = ["intact", "q4_g32", "res_g32", "q4_g64", "res_g64"]
ids = sorted(set.intersection(*[set(R[s]) for s in S]))
print(f"clips complete for all schemes: {len(ids)} of {len(clips)}")
ZH = {"aishell", "fleurs_cmn_hans_cn", "fleurs_yue_hant_hk", "cantonese_md", "sichuan", "northeast", "henan", "wu"}
order = ["aishell", "fleurs_cmn_hans_cn", "fleurs_yue_hant_hk", "cantonese_md", "sichuan", "northeast", "henan", "wu",
         "fleurs_en_us", "fleurs_ko_kr", "fleurs_ja_jp", "fleurs_vi_vn", "fleurs_th_th", "fleurs_ar_eg", "fleurs_ru_ru"]


def rate(s, ps):
    return 100 * sum(R[s][p]["err"] for p in ps) / max(1, sum(R[s][p]["n"] for p in ps))


rng = np.random.default_rng(0)


def boot(a, b, ps, n=4000):
    g = defaultdict(list)
    for p in ps:
        g[clips[p]["set"]].append(p)
    arr = [np.array([[R[a][p]["err"], R[b][p]["err"], R[a][p]["n"]] for p in v], float) for v in g.values()]
    out = []
    for _ in range(n):
        x = y = t = 0
        for m in arr:
            i = rng.integers(0, len(m), len(m))
            x += m[i, 0].sum(); y += m[i, 1].sum(); t += m[i, 2].sum()
        out.append(100 * (x - y) / t)
    return np.percentile(out, [2.5, 97.5])


print("\nerror rate (%), MLX on M4, language forced")
print(f"{'set':20s}{'n':>4s} " + " ".join(f"{s:>9s}" for s in S))
for st in order + ["ALL", "Chinese+dialect", "other"]:
    ps = ids if st == "ALL" else [p for p in ids if (clips[p]["set"] in ZH if st == "Chinese+dialect" else clips[p]["set"] not in ZH if st == "other" else clips[p]["set"] == st)]
    print(f"{st:20s}{len(ps):4d} " + " ".join(f"{rate(s, ps):9.2f}" for s in S))

print("\nACCURACY (preset): res_gG minus q4_gG, pooled, 95% CI; worst set point estimate")
verdict = {}
for G in (32, 64):
    a, b = f"res_g{G}", f"q4_g{G}"
    lo, hi = boot(a, b, ids)
    d = {st: rate(a, [p for p in ids if clips[p]["set"] == st]) - rate(b, [p for p in ids if clips[p]["set"] == st]) for st in order}
    w = sorted(d, key=d.get, reverse=True)[:3]
    ok = hi <= .5 and max(d.values()) <= 2
    verdict[G] = dict(acc=ok)
    print(f"  G={G}: {rate(a, ids) - rate(b, ids):+.2f} [{lo:+.2f}, {hi:+.2f}]  worst sets " + " ".join(f"{k}:{d[k]:+.1f}" for k in w) + f"  -> {'PASS' if ok else 'FAIL'}")
    for label, ps in (("Chinese+dialect", [p for p in ids if clips[p]["set"] in ZH]), ("other", [p for p in ids if clips[p]["set"] not in ZH])):
        lo2, hi2 = boot(a, b, ps)
        print(f"        {label:16s} {rate(a, ps) - rate(b, ps):+.2f} [{lo2:+.2f}, {hi2:+.2f}]")
print("  context, each vs intact (MLX bf16):")
for s in S[1:]:
    lo, hi = boot(s, "intact", ids)
    print(f"    {s:8s} {rate(s, ids) - rate('intact', ids):+.2f} [{lo:+.2f}, {hi:+.2f}]")

print("\nRUNTIME (preset): median per clip, interleaved; end to end = audio tower (prep.py, torch MPS fp32) + prefill + decode")
tower = {p: clips[p]["audio_tower_ms"] for p in ids}
med = lambda v: float(np.median(v))
for s in S:
    pre = [R[s][p]["prefill_ms"] for p in ids]
    dec = [R[s][p]["decode_ms"] / (R[s][p]["n_new"] + 1) for p in ids]
    e2e = [tower[p] + R[s][p]["prefill_ms"] + R[s][p]["decode_ms"] for p in ids]
    print(f"  {s:8s} prefill {med(pre):6.1f} ms  decode {med(dec):5.2f} ms/token  end to end {med(e2e):6.1f} ms  (mean tokens {np.mean([R[s][p]['n_new'] for p in ids]):.1f})")
for G in (32, 64):
    a, b = f"res_g{G}", f"q4_g{G}"
    ratio = [(tower[p] + R[a][p]["prefill_ms"] + R[a][p]["decode_ms"]) / (tower[p] + R[b][p]["prefill_ms"] + R[b][p]["decode_ms"]) for p in ids]
    # same-length comparison: per-token decode ratio and prefill ratio, paired by clip
    dr = [(R[a][p]["decode_ms"] / (R[a][p]["n_new"] + 1)) / (R[b][p]["decode_ms"] / (R[b][p]["n_new"] + 1)) for p in ids]
    pr = [R[a][p]["prefill_ms"] / R[b][p]["prefill_ms"] for p in ids]
    q = np.percentile(ratio, [25, 75])
    ok = med(ratio) <= 1.25
    verdict[G]["rt"] = ok
    print(f"  G={G}: end-to-end ratio res/q4 median {med(ratio):.3f} (IQR {q[0]:.2f}-{q[1]:.2f}) -> {'ACCEPTABLE' if ok else 'NOT ACCEPTABLE'};"
          f" decode per token {med(dr):.3f}, prefill {med(pr):.3f}")

wi = json.load(open(os.path.join(E5, "weights_info.json")))
m = wi["mem"]
print("\nMEMORY: decoder matrices incl. scales/biases (MB); embedding bf16", round(m["embedding_bf16"] / 1e6, 1), "MB in every scheme")
print(f"  intact bf16 {m['intact']['all'] / 1e6:.1f}")
for G in (32, 64):
    q4, base, res = m[f"q4_g{G}"]["all"], m[f"res_g{G}"]["base"], m[f"res_g{G}"]["residual"]
    print(f"  G={G}: LLM Q4 {base / 1e6:.1f} + separate ASR Q4 {q4 / 1e6:.1f} = {(base + q4) / 1e6:.1f};  shared: LLM Q4 {base / 1e6:.1f} + residual {res / 1e6:.1f} = {(base + res) / 1e6:.1f}"
          f"  -> ASR-specific decoder bytes {res / q4:.2f} x; bits/weight q4 {8 * q4 / 440401920:.2f}, residual {8 * res / 440401920:.2f}")
print("  weight relative error (Frobenius):", json.dumps(wi["stats"]))
if os.path.exists(os.path.join(E5, "bench.json")):
    b = json.load(open(os.path.join(E5, "bench.json")))
    print(f"\nBENCH (reported): one token at context {b['context']}, medians of 300 interleaved rounds (ms)")
    for s, v in b["ms_per_token"].items():
        print(f"  {s:8s} layers {v['layers']:.2f}  lm_head {v['head']:.2f}")
    for G in (32, 64):
        print(f"  G={G}: decoder layers res/q4 {b['ms_per_token'][f'res_g{G}']['layers'] / b['ms_per_token'][f'q4_g{G}']['layers']:.3f}")

print("\nSANITY: MLX bf16 intact vs E3 torch fp32 forced intact")
e3 = {r["path"]: r for r in map(json.loads, open(os.path.join(HERE, "..", "e3_real_speech", "per_clip_forced.jsonl"), encoding="utf-8")) if r["scheme"] == "intact"}
ps = [p for p in ids if clips[p]["path"] in e3]
e3r = 100 * sum(e3[clips[p]["path"]]["err"] for p in ps) / sum(e3[clips[p]["path"]]["n"] for p in ps)
print(f"  n {len(ps)}  E3 torch {e3r:.2f}  MLX {rate('intact', ps):.2f}  identical transcripts {np.mean([R['intact'][p]['hyp'] == e3[clips[p]['path']]['hyp'] for p in ps]):.3f}")
print("  hit the 200-token cap:", {s: sum(R[s][p]["n_new"] >= 200 for p in ids) for s in S})
print("\nVERDICT:", {f"G={G}": ("accuracy PASS" if v["acc"] else "accuracy FAIL") + ", runtime " + ("ACCEPTABLE" if v.get("rt") else "NOT ACCEPTABLE") for G, v in verdict.items()})
