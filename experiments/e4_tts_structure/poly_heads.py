# Which reading head would an adapter need? Saves per-sample predictions so heads can be compared with the TTS's own reading
# on exactly the sentences of stage 4. Same per-character ridge as pron_probe.py (feature t+2, lambda chosen on dev accuracy).
# Layers are fixed in advance: LLM-0.6B 2, 4, 8, 12, 28 for a cost curve, TTS-0.6B 6 and TTS-1.7B 16 as dev-selected earlier.
#
# Criteria, fixed before comparing heads with audio:
#   An adapter gains over the TTS alone if head accuracy minus the TTS's reading rate has a group-bootstrap CI above 0 on the same items.
#   Inheritance from the LLM adds value if LLM head minus the TTS's own head (a head on TTS L6 needs no LLM) has a CI above 0.
import json, os
import numpy as np
import torch
import pron_probe as P
from pron_probe import DEVICE, LAMBDAS, OUT, build, capture, fit_scores, free, load_data, make_groups, summary, tally

HEADS = [("LLM-0.6B", "llm", "Qwen/Qwen3-0.6B-Base", [2, 4, 8, 12, 28]), ("TTS-0.6B", "tts", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", [6]),
         ("TTS-1.7B", "tts", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", [16])]


def main():
    torch.set_num_threads(4)
    samples, _, _, dict_test = load_data()
    samples = [x for x in samples if x["cat"] == "single"]
    groups = make_groups(samples)
    key = lambda n: f"{samples[n]['text']}|{samples[n]['i']}"
    res = {}
    res["dictionary"] = {key(n): dict_test[samples[n]["line"]] for g in groups for n in g["te"]}
    res["majority"] = {key(n): {i: r for r, i in g["cls"].items()}[g["maj"]] for g in groups for n in g["te"]}
    for name, kind, repo, layers in HEADS:
        P.LAYERS = layers
        model = build(kind, repo)
        poly, _ = capture(model, samples, [])
        del model
        free()
        for li, lay in enumerate(layers):
            arr = poly[:, li]
            scale = max(float(arr[::7].astype(np.float64).std()), 1e-9)
            get = lambda ix: torch.from_numpy(arr[ix].astype(np.float64)).reshape(len(ix), -1) / scale
            preds = {l: {} for l in LAMBDAS}
            dev = {l: [] for l in LAMBDAS}
            test = {l: [] for l in LAMBDAS}
            for g in groups:
                names = {i: r for r, i in g["cls"].items()}
                Y = torch.nn.functional.one_hot(torch.tensor(g["y_tr"]), len(names)).double()
                pr = fit_scores(get(g["tr"]), Y, [get(g["dv"]), get(g["te"])])
                for l in LAMBDAS:
                    dev[l].append(tally(pr[l][0].numpy(), g["y_dv"], g["maj"]))
                    test[l].append(tally(pr[l][1].numpy(), g["y_te"], g["maj"]))
                    for n, p in zip(g["te"], pr[l][1].numpy()):
                        preds[l][key(n)] = names[int(p)]
            lam = max(LAMBDAS, key=lambda l: summary(dev[l])[0])
            res[f"{name}@L{lay}"] = preds[lam]
            a, r = summary(test[lam])
            print(f"{name} L{lay} lambda {lam}: test accuracy {a:.4f} minority recall {r:.4f}", flush=True)
        del poly
        free()
    json.dump(res, open(os.path.join(OUT, "heads.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("HEADSDONE", flush=True)


if __name__ == "__main__":
    main()
