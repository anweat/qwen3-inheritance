# E6: reading pipeline text -> g2pW (CPP-retrained) -> reading codebook -> unchanged Qwen3-TTS-0.6B. Builds the synthesis plan
# on the Mac (.venv-judge: g2pw, onnxruntime, tokenizer); pipeline_synth.py renders it on the laptop; score.py judges it.
#   python build_plan.py <fp32|int8> [v1|v2]     (v2 writes plan_v2.json; see the v2 note below)
#
# Pipeline rule, fixed before any pipeline audio exists:
#   Automatic edit at position j when all hold: g2pW treats the character as polyphonic (it is in the CPP polyphonic list) and
#   returns a toned pinyin reading; the reading is not neutral tone and the character is not 一 or 不 (tone sandhi); the reading
#   has a codebook vector (at least 3 single-reading, single-token characters with that reading); the character is a token of
#   its own in the sentence's tokenization (a polyphone inside a multi-character token is left alone, since the word token
#   carries word identity and splitting it changes the text the TTS sees).
#   An edit replaces the character with a placeholder token whose embedding row holds the reading vector (stage 4 port:
#   mean embedding of the reading's single-reading characters, rescaled to their mean norm). Every edit in the sentence is
#   applied, not only the evaluated position. The TTS, its voices and generation settings are those of stage 4.
#   User dictionary: entries (word, offset, reading) override g2pW at matching positions and are applied even inside a
#   multi-character token, since the user asked for that reading explicitly.
#
# Rule v2 (word guard), added 2026-11-10 after neighbour.py, before any v2 audio exists: the v1 rule, then the sentence is
#   segmented with jieba (0.42.1, default dictionary); in a word of two or more characters that holds a CPP polyphone the v1
#   rule leaves unedited (no vector, inside a multi-character token, neutral tone or sandhi), every automatic edit is dropped,
#   so a placeholder never splits a word whose other polyphone the TTS must read from context. Dictionary entries still apply.
#   Criteria for v2 are in wordrule.py.
#
# Criteria (TTS-0.6B; the stage 4 ASR detector, log odds centred on the same item's R_g and R_a references, says gold if > 0;
# P = pipeline audio, O = the existing stage 4 audio of the unmodified sentence, paired by item):
#   Useful: CPP minority (stage 4 hard items), P minus O, character-group bootstrap 95% CI lower bound > 0.
#   Safe: natural2 (131 natural-frequency positions with hand gold), P minus O, character bootstrap lower bound > -.05.
#   Reported without a threshold: CPP majority, the natural speech sentences, share of positions edited, edits per sentence,
#   and P against g2pW+port from stage 4 (the target alone edited).
#   Dictionary set (exploratory demonstration; sentences and readings written for this test, mostly names and places):
#     reading rate of O, P with g2pW only, and P with the dictionary.
#   Limits: the detector is a forced choice between two readings, so a third reading or a garbled syllable is not seen; only
#   listening can rule that out. Sampling differs between P and O.
import json, os, sys
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from g2pw import G2PWConverter
import jieba

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
OUT = os.path.expanduser("~/.cache/qwen-all/e6")
SANDHI = {"一", "不"}

# (sentence, word, offset of the target in the word, dictionary reading, competing reading the TTS is likely to use)
DICT_SET = [
    ("单雄信是隋末唐初的一员猛将。", "单雄信", 0, "shan4", "dan1"),
    ("明代才子解缙主持编修了永乐大典。", "解缙", 0, "xie4", "jie3"),
    ("明代画家仇英擅长工笔人物画。", "仇英", 0, "qiu2", "chou2"),
    ("作家查良镛的笔名是金庸。", "查良镛", 0, "zha1", "cha2"),
    ("曾巩是唐宋八大家之一。", "曾巩", 0, "zeng1", "ceng2"),
    ("尉迟恭是唐朝的开国功臣。", "尉迟", 0, "yu4", "wei4"),
    ("朴先生来自韩国首尔。", "朴先生", 0, "piao2", "pu3"),
    ("纪晓岚是清代的大学士。", "纪晓岚", 0, "ji3", "ji4"),
    ("召公是周朝初年的重要大臣。", "召公", 0, "shao4", "zhao4"),
    ("任先生是我们的数学老师。", "任先生", 0, "ren2", "ren4"),
    ("番禺区位于广州市南部。", "番禺", 0, "pan1", "fan1"),
    ("蚌埠是安徽省的一座城市。", "蚌埠", 0, "beng4", "bang4"),
    ("铅山县隶属于江西省上饶市。", "铅山", 0, "yan2", "qian1"),
    ("东阿县隶属于山东省聊城市。", "东阿", 1, "e1", "a1"),
    ("洪洞县有一棵著名的大槐树。", "洪洞", 1, "tong2", "dong4"),
    ("浚县隶属于河南省鹤壁市。", "浚县", 0, "xun4", "jun4"),
    ("莘县隶属于山东省聊城市。", "莘县", 0, "shen1", "xin1"),
    ("黄陂区位于武汉市北部。", "黄陂", 1, "pi2", "bei1"),
    ("涡阳县隶属于安徽省亳州市。", "涡阳", 0, "guo1", "wo1"),
    ("高句丽是古代东北亚的一个政权。", "高句丽", 1, "gou1", "ju4"),
    ("龟兹是古代西域的一个国家。", "龟兹", 0, "qiu1", "gui1"),
    ("牟平区隶属于山东省烟台市。", "牟平", 0, "mu4", "mou2"),
    ("乐亭县隶属于河北省唐山市。", "乐亭", 0, "lao4", "le4"),
    ("秘鲁位于南美洲西部。", "秘鲁", 0, "bi4", "mi4"),
    ("华山是五岳中的西岳。", "华山", 0, "hua4", "hua2"),
    ("万俟卨是南宋初年的大臣。", "万俟", 1, "qi2", "si4"),
]


def load_sets():
    sets = {}
    for name, src in (("hard", "tts_out"), ("natural", "tts_natural"), ("natural2", "tts_natural2")):
        rows = []
        for r in map(json.loads, open(f"{ROOT}/{src}/TTS-0.6B_nonstream.jsonl", encoding="utf-8")):
            rg = r["scores"]["texts"]["R_g"]
            i = next(j for j, (x, y) in enumerate(zip(r["text"], rg)) if x != y)
            rows.append(dict(id=r["id"], text=r["text"], i=i, char=r["char"], gold=r["gold"], alt=r["alt"], kind=r["kind"],
                             group=r["group"], texts=r["scores"]["texts"]))
        sets[name] = rows
    homs = json.load(open(f"{ROOT}/homophones.json", encoding="utf-8"))
    rows = []
    for k, (s, w, off, g, a) in enumerate(DICT_SET):
        i = s.index(w) + off
        hg, ha = [h for h in homs.get(g, []) if h != s[i]], [h for h in homs.get(a, []) if h != s[i]]
        if not hg or not ha:
            print("dictionary item without homophone references, dropped:", s, g, a)
            continue
        ha = [h for h in ha if h not in hg]
        sw = lambda h: s[:i] + h + s[i + 1:]
        rows.append(dict(id=k, text=s, i=i, char=s[i], gold=g, alt=a, kind="dict", group=s[i], word=w, off=off,
                         texts=dict(O=s, R_g=sw(hg[0]), R_a=sw(ha[0]))))
    sets["dict"] = rows
    return sets


def main():
    variant = sys.argv[1]
    rule = sys.argv[2] if len(sys.argv) > 2 else "v1"
    model_dir = os.path.expanduser("~/.cache/qwen-all/g2pw/G2PWModel-cpp" + ("-int8" if variant == "int8" else ""))
    poly = {l.split("\t")[0] for l in open(os.path.join(model_dir, "POLYPHONIC_CHARS.txt"), encoding="utf-8").read().strip().split("\n")}
    tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.json", "*.txt"]))
    mono = json.load(open(f"{ROOT}/mono_read.json", encoding="utf-8"))
    members = {}
    for ch, r in mono.items():
        if len(tok(ch, add_special_tokens=False)["input_ids"]) == 1:
            members[r] = members.get(r, 0) + 1
    has_vec = {r for r, n in members.items() if n >= 3}
    conv = G2PWConverter(model_dir=model_dir, style="bopomofo", enable_non_tradional_chinese=False, num_workers=0)
    conv.num_workers = 0  # g2pw treats 0 as unset and would spawn 2 DataLoader processes per call
    sets = load_sets()
    texts = sorted({r["text"] for rows in sets.values() for r in rows})
    g = {}
    for b in range(0, len(texts), 64):
        for t, p in zip(texts[b:b + 64], conv(texts[b:b + 64])):
            g[t] = p

    def auto_edits(t):
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        own = {s for s, e in enc["offset_mapping"] if e - s == 1}
        out, skipped = {}, dict(not_own_token=0, no_vector=0, neutral_or_sandhi=0, word_guard=0)
        held = set()
        for j, (c, r) in enumerate(zip(t, g[t])):
            if c not in poly or not r or not r[-1].isdigit() or not r[0].isascii():
                continue
            if r.endswith("5") or c in SANDHI:
                skipped["neutral_or_sandhi"] += 1
                held.add(j)
            elif r not in has_vec:
                skipped["no_vector"] += 1
                held.add(j)
            elif j not in own:
                skipped["not_own_token"] += 1
                held.add(j)
            else:
                out[j] = r
        if rule == "v2":
            for w, a, b in jieba.tokenize(t):
                if b - a >= 2 and any(a <= j < b for j in held):
                    for j in [j for j in out if a <= j < b]:
                        del out[j]
                        skipped["word_guard"] += 1
        return out, skipped

    jobs, stats = [], {}
    for name, rows in sets.items():
        st = dict(sentences=len(rows), edits=0, target_edited=0, target_reading_gold=0, not_own_token=0, no_vector=0, neutral_or_sandhi=0, word_guard=0)
        for r in rows:
            e, sk = auto_edits(r["text"])
            for k, v in sk.items():
                st[k] += v
            st["edits"] += len(e)
            st["target_edited"] += r["i"] in e
            st["target_reading_gold"] += g[r["text"]][r["i"]] == r["gold"]
            base = dict(set=name, id=r["id"], text=r["text"], voice=r["id"] % 2)
            if name == "dict":
                for k in ("O", "R_g", "R_a"):
                    jobs.append(dict(base, kind=k, text=r["texts"][k], edits={}))
                jobs.append(dict(base, kind="P_g2pw", edits=e))
                d = dict(e)
                if r["gold"] in has_vec:
                    d[r["i"]] = r["gold"]
                jobs.append(dict(base, kind="P_dict", edits=d))
            else:
                jobs.append(dict(base, kind="P", edits=e))
        stats[name] = st
    for j in jobs:
        j["edits"] = {str(k): v for k, v in sorted(j["edits"].items())}
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(variant=variant, rule=rule, jobs=jobs, items={n: rows for n, rows in sets.items()}),
              open(f"{OUT}/plan{'' if rule == 'v1' else '_' + rule}.json", "w", encoding="utf-8"), ensure_ascii=False)
    for n, st in stats.items():
        print(n, st)
    print(len(jobs), "synthesis jobs; PLANDONE")


if __name__ == "__main__":
    main()
