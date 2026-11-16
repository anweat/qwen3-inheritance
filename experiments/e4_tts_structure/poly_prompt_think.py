# Exploratory, added after the non-thinking prompt (poly_prompt.py) came out near chance at every size: does Qwen3's thinking
# mode, i.e. reasoning before answering, read polyphones? Same prompt and candidates, stage 4 CPP items only, greedy decoding.
# The answer is the first candidate found after </think>, with tone marks stripped; no candidate found counts as wrong.
#   python poly_prompt_think.py Qwen/Qwen3-4B
import json, os, re, sys, time, unicodedata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pron_probe import OUT
from poly_prompt import PROMPT, build_items

sys_argv, sys.argv = sys.argv, sys.argv[:1]
from poly_summary import load

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
BATCH, MAX_NEW = 8, 768


def plain(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c)).lower()


def answer(text, cands):
    tail = plain(text.split("</think>")[-1]) if "</think>" in text else ""
    hits = [(m.start(), c) for c in cands for m in re.finditer(re.escape(c) + r"(?!\d)", tail)]
    return min(hits)[1] if hits else None


def main():
    repo = sys_argv[1]
    want = set()
    for r in load(f"{ROOT}/tts_out/TTS-0.6B_nonstream.jsonl"):
        if "asr_centred" in r["scores"]["O"]:
            i = next(j for j, (x, y) in enumerate(zip(r["text"], r["scores"]["texts"]["R_g"])) if x != y)
            want.add(f"{r['text']}|{i}")
    items = [x for x in build_items() if x["part"] == "eval" and x["key"] in want]
    print(f"{len(items)} items", flush=True)
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("mps").eval()
    path = os.path.join(OUT, f"prompt_think_{repo.split('/')[-1]}.jsonl")
    done = {json.loads(l)["key"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    items = [x for x in items if x["key"] not in done]
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for b in range(0, len(items), BATCH):
            chunk = items[b:b + BATCH]
            texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(
                left=x["text"][:x["i"]], char=x["char"], right=x["text"][x["i"] + 1:], cands="、".join(x["cands"]))}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True) for x in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
            for x, seq in zip(chunk, o[:, enc.input_ids.shape[1]:]):
                out = tok.decode(seq, skip_special_tokens=True)
                f.write(json.dumps(dict(key=x["key"], gold=x["gold"], cands=x["cands"], pred=answer(out, x["cands"]),
                                        finished="</think>" in out, text=out), ensure_ascii=False) + "\n")
            f.flush()
            print(f"{b + len(chunk)}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    print("THINKDONE", flush=True)


if __name__ == "__main__":
    main()
