# Stage 2 controls: is the failed LLM stitch a fact about the LLM, or about ridge maps at this R2?
# Positive controls map the talker's own earlier layer and the other TTS family into layer k.
# No-LLM controls map token embeddings only. Calibration damages native text to the base map's R2.
# Position decoding asks whether text states carry their own sequence position in their content.
import argparse, json, os
import numpy as np
import torch
import soundfile as sf
from transformers import Qwen3ForCausalLM
import stitch as S
from stitch import DEVICE, DTYPE, FAMILIES, OUT, SPEECH, Talker, apply_map, boot, fit_ridge, free, llm_states, load_clips, r2, rp, segments

KS = {"0.6B": [1, 3, 6], "1.7B": [1, 2, 3]}
OTHER = {"0.6B": "1.7B", "1.7B": "0.6B"}


def tot(rs):
    return sum(r[0] for r in rs) / sum(r[2] for r in rs)


def rec(cm, cn, ce):
    return (cm - ce) / (cm - cn) if cm - cn > .02 else float("nan")


def pca_truncate(Y_train, Y_tests, target_r2):
    """Keep the top principal components of native text states until test R2 reaches the base map's R2."""
    Y = torch.cat(Y_train).to(DEVICE).double()
    mu = Y.mean(0)
    _, _, V = torch.linalg.svd(Y - mu, full_matrices=False)
    Yt = torch.cat(Y_tests).to(DEVICE).double()
    total = ((Yt - mu) ** 2).sum().item()
    lo, hi = 1, V.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        P = V[:mid]
        e = (((Yt - mu) @ P.T @ P - (Yt - mu)) ** 2).sum().item()
        if 1 - e / total >= target_r2:
            hi = mid
        else:
            lo = mid + 1
    P = V[:lo]
    return lo, [((y.to(DEVICE).double() - mu) @ P.T @ P + mu).float() for y in Y_tests]


def position_r2(train_states, test_states):
    def target(s):
        n = len(s)
        i = torch.arange(n, dtype=torch.float32)
        return torch.stack([i, n - 1 - i], 1)
    m = fit_ridge(train_states, [target(s) for s in train_states])
    return r2(m, test_states, [target(s) for s in test_states])


def run_family(name, n_test, ks):
    fam = FAMILIES[name]
    S.GRID = sorted({0} | set(ks) | {k - 1 for k in ks})
    test, train = load_clips(n_test)
    train_ids = [ids for _, ids in segments()] + [c["ids"] for c in train]
    test_ids = [c["ids"] for c in test]
    print(f"\n{'='*12} family {name}: {len(test)} test clips, {len(train_ids)} training sequences {'='*12}", flush=True)
    m = Qwen3ForCausalLM.from_pretrained(rp(fam["llms"]["base"]), dtype=DTYPE, attn_implementation="eager").to(DEVICE).eval()
    llm_tr, llm_te = llm_states(m, train_ids), llm_states(m, test_ids)
    del m
    free()
    O = Talker(FAMILIES[OTHER[name]]["tts"])
    oth_tr, oth_te = O.text_states(train_ids), O.text_states(test_ids)
    del O
    free()
    T = Talker(fam["tts"])
    tr, te = T.text_states(train_ids), T.text_states(test_ids)
    for c in test:
        a, sr = sf.read(os.path.join(SPEECH, c["path"]), dtype="float32")
        c["codes"] = T.m.speech_tokenizer.encode(a, sr=sr).audio_codes[0].cpu().numpy()[:300]
    out = {}
    print("k cond: CE, recovered [95% CI of CE gain over mean], R2", flush=True)
    for k in ks:
        Y_tr, Y_te = [s[k] for s in tr], [s[k] for s in te]
        native = [T.score(c, k=k) for c in test]
        mean_vec = torch.cat(Y_tr).mean(0)
        mean_r = [T.score(c, k=k, rep=mean_vec) for c in test]
        cn, cm = tot(native), tot(mean_r)
        print(f"-- k={k} native_k {cn:.3f} mean {cm:.3f}", flush=True)
        sources = {
            "llm_same_depth": ([s[k] for s in llm_tr], [s[k] for s in llm_te]),
            "llm_embedding": ([s[0] for s in llm_tr], [s[0] for s in llm_te]),
            "own_embedding": ([s[0] for s in tr], [s[0] for s in te]),
            "own_prev_layer": ([s[k - 1] for s in tr], [s[k - 1] for s in te]),
            "other_tts_same_depth": ([s[k] for s in oth_tr], [s[k] for s in oth_te]),
        }
        out[k] = dict(native=native, mean=mean_r)
        base_r2 = None

        def report(label, reps, score_r2=None, extra=""):
            rs = [T.score(c, k=k, rep=reps[i]) for i, c in enumerate(test)]
            out[k][label] = rs
            ce = tot(rs)
            lo, hi = boot([b[0] / b[2] - a[0] / a[2] for a, b in zip(rs, mean_r)])
            r2s = f" R2 {score_r2:.3f}" if score_r2 is not None else ""
            print(f"  {label:22s} CE {ce:.3f} rec {rec(cm, cn, ce):5.2f} gain [{lo:+.3f}, {hi:+.3f}]{r2s}{extra}", flush=True)

        for label, (X_tr, X_te) in sources.items():
            mp = fit_ridge(X_tr, Y_tr)
            val = r2(mp, X_te, Y_te)
            if label == "llm_same_depth":
                base_r2 = val
            report(label, [apply_map(mp, x) for x in X_te], val)
        # Calibration: damage native text to the LLM map's R2, without any map.
        r, reps = pca_truncate(Y_tr, Y_te, base_r2)
        report("native_pca_matched", reps, base_r2, f" ({r} components)")
        Y = torch.cat(Y_tr)
        sigma = ((1 - base_r2) * ((Y - Y.mean(0)) ** 2).mean()).sqrt().item()
        torch.manual_seed(k)
        report("native_noise_matched", [y + sigma * torch.randn_like(y) for y in Y_te], base_r2)
        free()
    print("\nLinear decoding of token position (index from start and from end), test R2:", flush=True)
    for k in ks:
        print(f"  k={k}: talker {position_r2([s[k] for s in tr], [s[k] for s in te]):.3f}   LLM {position_r2([s[k] for s in llm_tr], [s[k] for s in llm_te]):.3f}", flush=True)
    json.dump(out, open(os.path.join(OUT, f"stitch_control_{name}.json"), "w"))
    del T
    free()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.smoke:
        run_family("0.6B", 4, [6])
    else:
        run_family("0.6B", 100, KS["0.6B"])
        run_family("1.7B", 100, KS["1.7B"])
    print("CONTROLDONE", flush=True)
