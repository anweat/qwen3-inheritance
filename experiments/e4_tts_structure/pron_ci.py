# Finishes criterion B of pron_probe.py, whose run failed at the neighbour baseline before saving per-character rows.
# Reruns only single-character tokens at the layers dev-selected in that run, so point estimates should reproduce
# pron_results.txt. Comparing the LLM at L8 is exploratory: L8 was noticed after seeing the test curves.
import json, os
import numpy as np
import torch
from transformers import Qwen3ForCausalLM
import pron_probe as P
from pron_probe import DEVICE, DTYPE, MODELS, OUT, boot_diff, build, capture, check_padding, free, load_data, make_groups, probe_b, probe_b_features, rp, summary, tally, LAMBDAS

SELECTED = {"LLM-0.6B": [8, 20, 28], "Random-0.6B": [1], "TTS-0.6B": [5, 6], "LLM-1.7B": [8, 12, 16], "TTS-1.7B": [4, 16]}


def main():
    torch.set_num_threads(4)
    samples, _, _, dict_test = load_data()
    samples = [x for x in samples if x["cat"] == "single"]
    groups = make_groups(samples)
    print(f"{len(samples)} samples, {len(groups)} groups, test samples {sum(len(g['te']) for g in groups)}", flush=True)
    majority = [tally([g["maj"]] * len(g["y_te"]), g["y_te"], g["maj"]) for g in groups]
    dictionary = [tally([g["cls"].get(dict_test[samples[n]["line"]], -1) for n in g["te"]], g["y_te"], g["maj"]) for g in groups]
    print(f"majority acc {summary(majority)[0]:.4f}; dictionary acc {summary(dictionary)[0]:.4f} recall {summary(dictionary)[1]:.4f}", flush=True)

    table = {}
    for name, kind, repo in MODELS:
        print(f"==== {name} ====", flush=True)
        P.LAYERS = SELECTED[name]
        model = build(kind, repo)
        poly, _ = capture(model, samples, [])
        del model
        free()
        probe_b(name, poly, groups, table, samples)
        if 8 in SELECTED[name]:
            P.LAYERS = [8]
            probe_b(name + "@L8", poly[:, [SELECTED[name].index(8)]], groups, table, samples)
        del poly
        free()

    m = Qwen3ForCausalLM.from_pretrained(rp("Qwen/Qwen3-0.6B-Base"), dtype=DTYPE).to(DEVICE).eval()
    emb = m.get_input_embeddings().weight.detach().float().cpu().numpy()
    del m
    free()
    emb = emb / emb.std()
    win = np.zeros((len(samples), 5), dtype=np.int64)
    for n, x in enumerate(samples):
        for o in range(5):
            j = x["pos"] + o - 2
            win[n, o] = x["ids"][j] if 0 <= j < len(x["ids"]) else 0
    res = probe_b_features(lambda ix: torch.from_numpy(emb[win[ix]].astype(np.float64)).reshape(len(ix), -1), groups)
    lam = max(LAMBDAS, key=lambda l: summary(res[l]["dv"])[0])
    neighbours = res[lam]["te"]
    print(f"neighbour tokens t-2..t+2 embeddings: acc {summary(neighbours)[0]:.4f} recall {summary(neighbours)[1]:.4f}", flush=True)

    competitors = {"TTS-0.6B": table[("TTS-0.6B", "t+2")]["rows"], "TTS-1.7B": table[("TTS-1.7B", "t+2")]["rows"],
                   "dictionary": dictionary, "neighbours": neighbours, "random": table[("Random-0.6B", "t+2")]["rows"]}
    for llm in ("LLM-0.6B", "LLM-1.7B", "LLM-0.6B@L8", "LLM-1.7B@L8"):
        for feat in ("t+2", "t"):
            a = table[(llm, feat)]
            print(f"{llm} feature {feat}, L{a['layer']}: accuracy {a['acc']:.4f} recall {a['recall']:.4f}", flush=True)
            for label, rows in competitors.items():
                lo, hi = boot_diff(a["rows"], rows)
                ac, rc = summary(rows)
                print(f"   vs {label:10s} accuracy {ac:.4f} diff {a['acc']-ac:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]; recall {rc:.4f} diff {a['recall']-rc:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(table={"|".join(k): v for k, v in table.items()}, majority=majority, dictionary=dictionary, neighbours=neighbours),
              open(os.path.join(OUT, "pron_ci.json"), "w"))


if __name__ == "__main__":
    main()
    print("PRONCIDONE", flush=True)
