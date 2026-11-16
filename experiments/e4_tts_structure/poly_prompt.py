# Does the LLM's own reasoning and knowledge, used as a language model rather than through a trained probe, read polyphones
# better than a head trained on the TTS? Stage 4 heads are per-character ridge probes on hidden states and need labelled
# sentences for every character. Here a Qwen3 chat model is asked, zero shot, which reading the marked character has, and each
# candidate reading is scored by its log probability as the answer. No training, no prompt tuning: the prompt below was fixed
# before any model scored any item.
#   python poly_prompt.py Qwen/Qwen3-4B [--limit N]
#
# Candidates are the same classes the heads choose among (make_groups on CPP plus the natural positions, as in poly_gate.py),
# so prompt and head answers are comparable item by item. Margin = log probability of the best minus the second candidate.
#
# Criteria, fixed before any model scored any item (applied by poly_prompt_summary.py on the stage 4 items and audio verdicts):
#   Primary model is Qwen3-4B, the largest size considered for a device; 0.6B and 1.7B are a scaling curve, reported, not tested.
#   Useful: ungated prompt accuracy minus the TTS's actual reading rate on CPP minority, group bootstrap CI above 0.
#   Knowledge beyond the TTS: ungated prompt minus the TTS-0.6B@L6 head (needs no LLM) on CPP minority, CI above 0.
#   Beyond the probe: ungated prompt minus the LLM-0.6B@L8 head on CPP minority, CI above 0 (does using the LLM as a language
#   model add to reading its hidden states).
#   Safe: gated prompt (threshold = smallest margin with precision >= .97 on CPP dev, as poly_gate.py) minus TTS on natural2,
#   character bootstrap CI lower bound above -.05.
#   All three CPP comparisons are also reported on CPP majority and natural sets without a threshold.
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pron_probe as P
from pron_probe import OUT, load_data, make_groups
from poly_gate import extra_positions

PROMPT = ("请判断下面句子中【】里的汉字在这句话里的读音。\n"
          "句子：{left}【{char}】{right}\n"
          "候选读音：{cands}\n"
          "只回答一个候选读音，用拼音加数字声调表示。")


def build_items():
    samples, _, _, _ = load_data()
    samples = [x for x in samples if x["cat"] == "single"]
    extra = extra_positions()
    all_s = samples + extra
    groups = make_groups(all_s)
    key = lambda x: f"{x['text']}|{x['i']}"
    by = {}
    for n, x in enumerate(all_s):
        if x["split"] == "extra":
            by.setdefault(x["char"], []).append(n)
    items = []
    for g in groups:
        names = sorted(g["cls"])
        for part, ix in (("dev", g["dv"]), ("eval", g["te"] + by.get(g["key"][0], []))):
            for n in ix:
                x = all_s[n]
                items.append(dict(part=part, key=key(x), text=x["text"], i=x["i"], char=x["char"], gold=x["gold"], cands=names))
    # Natural positions whose character has no head (too few labelled CPP sentences): candidates are gold and alt from the hand
    # gold. Exploratory only, since heads cannot be compared there.
    here = os.path.dirname(os.path.abspath(__file__))
    alt = {}
    for f in ("natural_gold.json", "natural2_gold.json"):
        for x in json.load(open(os.path.join(here, f), encoding="utf-8")):
            alt[(x["char"], x["gold"])] = x["alt"]
    covered = {g["key"][0] for g in groups}
    for x in extra:
        if x["char"] not in covered and (x["char"], x["gold"]) in alt:
            items.append(dict(part="eval", key=key(x), text=x["text"], i=x["i"], char=x["char"], gold=x["gold"],
                              cands=sorted({x["gold"], alt[(x["char"], x["gold"])]})))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = build_items()
    if a.limit:
        items = items[:a.limit]
    print(f"{len(items)} items, {sum(x['part'] == 'dev' for x in items)} dev", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.repo)
    model = AutoModelForCausalLM.from_pretrained(a.repo, dtype=torch.bfloat16).to(dev).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out = {"dev": {}, "eval": {}}
    t0 = time.time()
    for k, x in enumerate(items):
        msg = PROMPT.format(left=x["text"][:x["i"]], char=x["char"], right=x["text"][x["i"] + 1:], cands="、".join(x["cands"]))
        prefix = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        p = tok(prefix, add_special_tokens=False)["input_ids"]
        conts = [tok(c, add_special_tokens=False)["input_ids"] + [end] for c in x["cands"]]
        L = max(len(c) for c in conts)
        ids = torch.full((len(conts), len(p) + L), tok.pad_token_id or end, dtype=torch.long)
        att = torch.zeros_like(ids)
        for j, c in enumerate(conts):
            ids[j, :len(p) + len(c)] = torch.tensor(p + c)
            att[j, :len(p) + len(c)] = 1
        with torch.no_grad():
            lp = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float().log_softmax(-1).cpu()
        sc = []
        for j, c in enumerate(conts):
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(c))
            sc.append(float(lp[j, pos, torch.tensor(c)].sum()))
        order = np.argsort(sc)[::-1]
        out[x["part"]][x["key"]] = [x["cands"][order[0]], round(sc[order[0]] - sc[order[1]], 4), x["gold"]]
        if k % 200 == 0:
            print(f"{k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    d = sorted(out["dev"].values(), key=lambda v: -v[1])
    ok = np.cumsum([v[0] == v[2] for v in d]) / np.arange(1, len(d) + 1)
    passing = np.where(ok >= .97)[0]
    tau = d[int(passing[-1])][1] if len(passing) else float("inf")
    acc = np.mean([v[0] == v[2] for v in d]) if d else float("nan")
    print(f"{a.repo}: dev accuracy {acc:.4f}, threshold {tau:.4f}, dev coverage {np.mean([v[1] >= tau for v in d]):.3f}", flush=True)
    name = a.repo.split("/")[-1] + ("_limit" if a.limit else "")
    json.dump(dict(tau=tau, pred={k: v[:2] for k, v in out["eval"].items()}, dev=out["dev"]),
              open(os.path.join(OUT, f"prompt_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("PROMPTDONE", flush=True)


if __name__ == "__main__":
    main()
