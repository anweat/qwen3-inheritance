# Do no harm: on natural sentences the TTS is mostly right already, so an adapter must not overrule it wrongly.
# Same per-character heads as poly_heads.py, trained on CPP train and dev, applied to the positions in natural_gold.json.
# Criterion, fixed before comparison with audio: a head is safe on natural text if its accuracy is not below the TTS's rate on O
# at those positions, judged by a character bootstrap CI of the difference whose lower bound is above -.05.
import json, os
import numpy as np
import torch
import pron_probe as P
from pron_probe import LAMBDAS, OUT, build, capture, fit_scores, free, load_data, make_groups, summary, tally
from shared_structure import tok
from stitch import SPEECH

HEADS = [("LLM-0.6B", "llm", "Qwen/Qwen3-0.6B-Base", [2, 4, 8, 28]), ("TTS-0.6B", "tts", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", [6])]


def main():
    torch.set_num_threads(4)
    samples, _, _, _ = load_data()
    gold = json.load(open("natural_gold.json", encoding="utf-8"))
    text = {r["path"]: r["text"] for r in map(json.loads, open(os.path.join(SPEECH, "manifest.jsonl"), encoding="utf-8"))}
    chars = {x["char"] for x in gold}
    subset = [x for x in samples if x["cat"] == "single" and x["char"] in chars]
    natural = []
    for k, x in enumerate(gold):
        e = tok(text[x["path"]], add_special_tokens=False, return_offsets_mapping=True)
        hit = [j for j, (s, t) in enumerate(e["offset_mapping"]) if s <= x["i"] < t]
        if len(hit) == 1 and e["offset_mapping"][hit[0]][1] - e["offset_mapping"][hit[0]][0] == 1:
            natural.append(dict(split="natural", cat="single", char=x["char"], gold=x["gold"], text=text[x["path"]], i=x["i"], ids=e["input_ids"], pos=hit[0], k=k))
    print(f"{len(subset)} CPP samples for {len(chars)} characters, {len(natural)} natural positions", flush=True)
    all_s = subset + natural
    groups = make_groups(all_s)
    res = {}
    for name, kind, repo, layers in HEADS:
        P.LAYERS = layers
        model = build(kind, repo)
        poly, _ = capture(model, all_s, [])
        del model
        free()
        for li, lay in enumerate(layers):
            arr = poly[:, li]
            scale = max(float(arr[::7].astype(np.float64).std()), 1e-9)
            get = lambda ix: torch.from_numpy(arr[ix].astype(np.float64)).reshape(len(ix), -1) / scale
            dev = {l: [] for l in LAMBDAS}
            preds = {l: {} for l in LAMBDAS}
            for g in groups:
                names = {i: r for r, i in g["cls"].items()}
                nat = [n for n, x in enumerate(all_s) if x["split"] == "natural" and x["char"] == g["key"][0]]
                if not nat:
                    continue
                Y = torch.nn.functional.one_hot(torch.tensor(g["y_tr"]), len(names)).double()
                pr = fit_scores(get(g["tr"]), Y, [get(g["dv"]), get(nat)])
                for l in LAMBDAS:
                    dev[l].append(tally(pr[l][0].numpy(), g["y_dv"], g["maj"]))
                    for n, p in zip(nat, pr[l][1].numpy()):
                        preds[l][all_s[n]["k"]] = names[int(p)]
            lam = max(LAMBDAS, key=lambda l: summary(dev[l])[0])
            res[f"{name}@L{lay}"] = preds[lam]
            ok = [preds[lam][x["k"]] == x["gold"] for x in natural if x["k"] in preds[lam]]
            print(f"{name} L{lay} lambda {lam}: natural accuracy {np.mean(ok):.3f} on {len(ok)}", flush=True)
        del poly
        free()
    json.dump(res, open(os.path.join(OUT, "heads_natural.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("HEADSNATDONE", flush=True)


if __name__ == "__main__":
    main()
