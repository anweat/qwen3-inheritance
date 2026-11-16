# Shared helpers: corpus, loading any 0.6B-shaped decoder as a plain Qwen3ForCausalLM (no remote code).
import glob, json, os, random, re
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
PREFIX = {"Qwen3-0.6B-Base": "model.", "Qwen3-0.6B": "model.", "Qwen3-ASR-0.6B": "thinker.model."}
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEG = 128

SPOKEN_EN = """so I was thinking we could meet up around three o'clock if that works for you.
the weather has been pretty strange this week, it rained on monday and then it got really hot.
could you please send me the report before the end of the day, I need to review it tonight.
my brother just moved to a new apartment near the train station and he really likes it.
we should probably check the numbers again because something does not look right to me.
I went to the store this morning but they were out of milk so I got some juice instead.
the meeting was moved to next tuesday because half of the team is traveling this week.
honestly I think the first option is better, it is cheaper and it will take less time.
she told me that the new restaurant downtown is great but you need to book a table early.
can you turn the music down a little bit, I am trying to make a phone call right now.
the flight leaves at seven in the morning so we need to get to the airport by five.
I have been learning to play the guitar for about two years and I still cannot play very well.
they announced that the bridge will be closed for repairs during the whole month of march.
when I was a kid we used to spend every summer at my grandmother's house near the lake.
thanks for calling, I will look into the problem and get back to you as soon as I can."""
SPOKEN_ZH = """我觉得我们可以下午三点左右见面，如果你那个时候方便的话。
这个星期的天气有点奇怪，周一下了一天雨，然后突然就热起来了。
麻烦你今天下班之前把报告发给我，我晚上要看一下。
我弟弟刚搬到火车站附近的新房子，他还挺喜欢那边的。
这些数字我们最好再核对一遍，我总觉得有哪里不太对。
我今天早上去了超市，牛奶卖完了，所以就买了点果汁。
会议改到下周二了，因为这周团队有一半的人都在出差。
说实话我觉得第一个方案更好，又便宜，花的时间也少。
她跟我说市中心新开的那家餐厅很不错，不过要早点订位子。
你能不能把音乐关小一点，我现在要打个电话。
飞机早上七点起飞，所以我们五点之前就要赶到机场。
我学吉他学了差不多两年了，到现在还是弹得不太好。
他们通知说那座桥整个三月份都要封闭维修。
小时候我们每年夏天都会去湖边的外婆家住一段时间。
谢谢你打电话过来，我查一下这个问题，尽快给你回复。"""
CHAT = ["What is the capital of Australia, and why is it not Sydney?",
        "Write a Python function that checks whether a string is a palindrome.",
        "请用三句话解释一下什么是通货膨胀。",
        "If a train travels 120 km in 1.5 hours, what is its average speed?",
        "给我推荐一个适合周末带孩子去的活动。",
        "Explain the difference between a list and a tuple in Python.",
        "把这句话翻译成英文：今天的会议推迟到明天上午十点。",
        "What are three common causes of a car battery dying?",
        "一个长方形长 8 厘米宽 5 厘米，它的面积和周长是多少？",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "How do I reverse a linked list? Show the idea briefly.",
        "为什么天空是蓝色的？请简单说明。"]


def corpus(tok, per_cat=30):
    """Token segments per category, built deterministically from local text."""
    # Corpus text ships with the scripts so the laptop and the Mac tokenize identical input.
    zh = ""
    for f in ("report.md", "research_brief.md"):
        zh += open(os.path.join(ROOT, "corpus_data", f), encoding="utf-8").read() + "\n"
    topics = open(os.path.join(ROOT, "corpus_data", "en_doc.txt"), encoding="utf-8").read()
    code = open(os.path.join(ROOT, "corpus_data", "code.txt"), encoding="utf-8").read()
    zh = re.sub(r"https?://\S+|[|#*`>\[\]()]+", " ", zh)
    cats = {"code": code, "en_doc": topics, "zh_doc": zh, "spoken_en": SPOKEN_EN.replace("\n", " "), "spoken_zh": SPOKEN_ZH.replace("\n", "")}
    rng = random.Random(0)
    segs = {}
    for c, text in cats.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= SEG:
            segs[c] = [ids]
            continue
        starts = list(range(0, len(ids) - SEG, SEG))
        rng.shuffle(starts)
        segs[c] = [ids[s:s + SEG] for s in starts[:per_cat]]
    segs["chat"] = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, return_dict=False) for q in CHAT]
    return segs


def tokenizer():
    return AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.json", "*.txt"]))


def read_decoder(name, base_cfg="Qwen/Qwen3-0.6B"):
    """Return {'model.layers...': tensor, 'model.embed_tokens.weight':...} for one model."""
    # Weights fetched by fetch_parallel.py live outside the Hub cache.
    local = os.path.expanduser(f"~/.cache/qwen-all/{name}")
    path = local if os.path.exists(os.path.join(local, "model.safetensors")) else snapshot_download(f"Qwen/{name}", allow_patterns=["*.safetensors", "*.json", "*.txt"])
    pre = PREFIX[name]
    sd = {}
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.startswith(pre):
                    sd["model." + k[len(pre):]] = h.get_tensor(k).float()
    return sd


def load(name, device=DEVICE, attn="eager"):
    cfg = AutoConfig.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.json"]))
    cfg._attn_implementation = attn
    cfg.tie_word_embeddings = True
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)
    sd = read_decoder(name)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    model.load_state_dict(sd, strict=True, assign=True)
    model.tie_weights()
    model.rotary_emb_fix = None
    # Non-persistent buffers on meta must be rebuilt.
    model.model.rotary_emb = type(model.model.rotary_emb)(cfg)
    return model.to(device).eval()
