# A head that overrules the TTS everywhere may harm text the TTS already reads well. Gate: intervene only when the ridge margin
# (top score minus second score) reaches a threshold; otherwise leave the TTS alone.
#
# Criteria, fixed before any comparison with audio:
#   Threshold per head: the smallest margin at which precision among positions at or above it is >= .97 on CPP dev.
#   Head rate with the gate: at gated positions, correct if the prediction equals gold; elsewhere the TTS's own verdict from the detector.
#   Safe if, on natural2_gold.json, gated head minus TTS has a character bootstrap CI with lower bound above -.05.
#   Useful if, on CPP minority items of stage 4, gated head minus TTS has a group bootstrap CI above 0.
#   Deployable candidate only if both hold. The same head is also reported without the gate.
import json, os
import numpy as np
import torch
import pron_probe as P
from pron_probe import LAMBDAS, OUT, build, capture, free, load_data, make_groups
from shared_structure import tok
from stitch import SPEECH

HEADS = [("LLM-0.6B", "llm", "Qwen/Qwen3-0.6B-Base", [8]), ("TTS-0.6B", "tts", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", [6])]


def fit_raw(X_tr, Y, X_ev):
    """Same dual ridge as pron_probe.fit_scores, returning scores rather than the argmax."""
    mu = X_tr.mean(0)
    A = X_tr - mu
    S, U = torch.linalg.eigh(A @ A.T)
    S = S.clamp_min(0)
    md = A.shape[1]
    ym = Y.mean(0)
    uy = U.T @ (Y - ym)
    Ks = [(X - mu) @ A.T for X in X_ev]
    return {l: [K @ (U @ (uy / (S + l * md + 1e-9)[:, None])) + ym for K in Ks] for l in LAMBDAS}


def extra_positions():
    here = os.path.dirname(os.path.abspath(__file__))
    text = {r["path"]: r["text"] for r in map(json.loads, open(os.path.join(SPEECH, "manifest.jsonl"), encoding="utf-8"))}
    rows = [dict(x, text=text[x["path"]], src="natural1") for x in json.load(open(os.path.join(here, "natural_gold.json"), encoding="utf-8"))]
    rows += [dict(x, src="natural2") for x in json.load(open(os.path.join(here, "natural2_gold.json"), encoding="utf-8"))]
    out = []
    for k, x in enumerate(rows):
        e = tok(x["text"], add_special_tokens=False, return_offsets_mapping=True)
        hit = [j for j, (s, t) in enumerate(e["offset_mapping"]) if s <= x["i"] < t]
        if len(hit) == 1 and e["offset_mapping"][hit[0]][1] - e["offset_mapping"][hit[0]][0] == 1:
            out.append(dict(split="extra", cat="single", char=x["char"], gold=x["gold"], text=x["text"], i=x["i"], ids=e["input_ids"], pos=hit[0], src=x["src"]))
    print(f"extra positions: {len(out)} of {len(rows)} are single-character tokens", flush=True)
    return out


def margin(s):
    v = s.topk(2, dim=1).values
    return (v[:, 0] - v[:, 1]).numpy()


def main():
    torch.set_num_threads(4)
    samples, _, _, _ = load_data()
    samples = [x for x in samples if x["cat"] == "single"]
    extra = extra_positions()
    all_s = samples + extra
    groups = make_groups(all_s)
    key = lambda x: f"{x['text']}|{x['i']}"
    res = {}
    for name, kind, repo, layers in HEADS:
        P.LAYERS = layers
        model = build(kind, repo)
        poly, _ = capture(model, all_s, [])
        del model
        free()
        arr = poly[:, 0]
        scale = max(float(arr[::7].astype(np.float64).std()), 1e-9)
        get = lambda ix: torch.from_numpy(arr[ix].astype(np.float64)).reshape(len(ix), -1) / scale
        by = {}
        for n, x in enumerate(all_s):
            if x["split"] == "extra":
                by.setdefault(x["char"], []).append(n)
        dev = {l: [] for l in LAMBDAS}
        out = {l: {} for l in LAMBDAS}
        for g in groups:
            names = {i: r for r, i in g["cls"].items()}
            ev = g["te"] + by.get(g["key"][0], [])
            Y = torch.nn.functional.one_hot(torch.tensor(g["y_tr"]), len(names)).double()
            pr = fit_raw(get(g["tr"]), Y, [get(g["dv"]), get(ev)])
            for l in LAMBDAS:
                m, p = margin(pr[l][0]), pr[l][0].argmax(1).numpy()
                dev[l] += list(zip(m.tolist(), (p == np.array(g["y_dv"])).tolist()))
                m, p = margin(pr[l][1]), pr[l][1].argmax(1).numpy()
                for n, a, b in zip(ev, m, p):
                    out[l][key(all_s[n])] = [names[int(b)], round(float(a), 4)]
        lam = max(LAMBDAS, key=lambda l: np.mean([c for _, c in dev[l]]))
        d = sorted(dev[lam], key=lambda x: -x[0])
        ok = np.cumsum([c for _, c in d]) / np.arange(1, len(d) + 1)
        passing = np.where(ok >= .97)[0]
        tau = d[int(passing[-1])][0] if len(passing) else float("inf")
        print(f"{name} L{layers[0]} lambda {lam}: dev accuracy {np.mean([c for _, c in d]):.4f}, threshold {tau:.4f}, dev coverage {np.mean([m >= tau for m, _ in d]):.3f}", flush=True)
        res[f"{name}@L{layers[0]}"] = dict(tau=tau, pred=out[lam])
        del poly
        free()
    json.dump(res, open(os.path.join(OUT, "heads_gate.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("GATEDONE", flush=True)


if __name__ == "__main__":
    main()
