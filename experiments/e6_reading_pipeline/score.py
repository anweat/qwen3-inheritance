# E6 scoring, criteria in build_plan.py.
#   python score.py prep        # (any venv) lay out rendered audio + references for ../e4_tts_structure/poly_asr.py
#   ../../.venv/bin/python ../e4_tts_structure/poly_asr.py ~/.cache/qwen-all/e6/asr/{hard,natural,natural2,dict}.jsonl
#   python score.py summary     # (.venv-tts or .venv) verdicts
import json, os, shutil, sys
from collections import defaultdict
import numpy as np

E6 = os.path.expanduser("~/.cache/qwen-all/e6")
ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
SRC = dict(hard="tts_out", natural="tts_natural", natural2="tts_natural2")
plan = json.load(open(f"{E6}/plan.json", encoding="utf-8"))
items = plan["items"]
rendered = {(r["set"], r["id"], r["kind"]): r for r in map(json.loads, open(f"{E6}/render/rendered.jsonl", encoding="utf-8"))}


def prep():
    os.makedirs(f"{E6}/asr", exist_ok=True)
    for s, rows in items.items():
        d = f"{E6}/asr/{s}"
        os.makedirs(d, exist_ok=True)
        out = []
        for r in rows:
            kinds = [k for (s2, i, k) in rendered if s2 == s and i == r["id"]]
            if not kinds:
                continue
            tx = {k: rendered[(s, r["id"], k)]["text"] for k in kinds}
            for k in kinds:
                shutil.copy(f"{E6}/render/{s}/{r['id']:04d}_{k}.flac", f"{d}/{r['id']:04d}_{k}.flac")
            if s != "dict":
                for k in ("R_g", "R_a"):
                    tx[k] = r["texts"][k]
                    shutil.copy(f"{ROOT}/{SRC[s]}/TTS-0.6B_nonstream/{r['id']:04d}_{k}.flac", f"{d}/{r['id']:04d}_{k}.flac")
            out.append(dict(id=r["id"], gold=r["gold"], alt=r["alt"], text=r["text"], char=r["char"], scores=dict(texts=tx)))
        with open(f"{d}.jsonl", "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(s, len(out))


def centred(path):
    out = {}
    for x in map(json.loads, open(path, encoding="utf-8")):
        l = x.get("logodds")
        if l:
            out[x["id"]] = {k: v - (l["R_g"] + l["R_a"]) / 2 for k, v in l.items()}
    return out


def boot(rows, a, b, n=4000):
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    ks = list(by)
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n):
        s = [x for j in rng.integers(0, len(ks), len(ks)) for x in by[ks[j]]]
        v.append(np.mean([a(x) for x in s]) - np.mean([b(x) for x in s]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def summary():
    print(f"==== E6 reading pipeline (g2pW {plan['variant']} -> codebook -> TTS-0.6B), ASR detector centred on R_g/R_a ====")
    verdict = {}
    for s in ("hard", "natural", "natural2"):
        old = centred(f"{ROOT}/{SRC[s]}/TTS-0.6B_nonstream_asr.jsonl")
        new = centred(f"{E6}/asr/{s}_asr.jsonl")
        rows = []
        for r in items[s]:
            if r["id"] in old and r["id"] in new and "O" in old[r["id"]] and "P" in new[r["id"]]:
                app = rendered[(s, r["id"], "P")]["applied"]
                rows.append(dict(r, O=old[r["id"]]["O"] > 0, P=new[r["id"]]["P"] > 0, edited=str(r["i"]) in app, n_edits=len(app)))
        parts = [("CPP minority", [r for r in rows if r["kind"] == "min"]), ("CPP majority", [r for r in rows if r["kind"] == "maj"])] if s == "hard" else [(s, rows)]
        for label, rs in parts:
            if not rs:
                continue
            lo, hi = boot(rs, lambda r: r["P"], lambda r: r["O"])
            o, p = np.mean([r["O"] for r in rs]), np.mean([r["P"] for r in rs])
            ed = np.mean([r["edited"] for r in rs])
            print(f"  {label:14s} n {len(rs):3d} chars {len({r['group'] for r in rs}):3d}  TTS alone {o:.3f}  pipeline {p:.3f}  "
                  f"diff {p - o:+.3f} [{lo:+.3f}, {hi:+.3f}]  target edited {ed:.2f}  edits/sentence {np.mean([r['n_edits'] for r in rs]):.2f}")
            e = [r for r in rs if r["edited"]]
            if e and len(e) < len(rs):
                print(f"      edited targets only n {len(e)}: TTS {np.mean([r['O'] for r in e]):.3f} pipeline {np.mean([r['P'] for r in e]):.3f};"
                      f" unedited targets n {len(rs) - len(e)}: TTS {np.mean([r['O'] for r in rs if not r['edited']]):.3f} pipeline {np.mean([r['P'] for r in rs if not r['edited']]):.3f}")
            flips = sum(r["O"] and not r["P"] for r in rs), sum(r["P"] and not r["O"] for r in rs)
            print(f"      right->wrong {flips[0]}, wrong->right {flips[1]}")
            verdict[label] = (lo, hi)
    print("VERDICT: useful", verdict.get("CPP minority", (0, 0))[0] > 0, " safe on natural2", verdict.get("natural2", (-1, 0))[0] > -.05)

    d = centred(f"{E6}/asr/dict_asr.jsonl")
    rows = [dict(r, **{k: d[r["id"]][k] > 0 for k in ("O", "P_g2pw", "P_dict")}) for r in items["dict"] if r["id"] in d]
    print(f"\nDictionary set (exploratory), n {len(rows)}:  TTS alone {np.mean([r['O'] for r in rows]):.3f}  g2pW only {np.mean([r['P_g2pw'] for r in rows]):.3f}"
          f"  with dictionary {np.mean([r['P_dict'] for r in rows]):.3f}")
    for r in rows:
        tag = lambda b: "ok" if b else "--"
        print(f"   {r['word']:5s} {r['gold']:6s} vs {r['alt']:6s} TTS {tag(r['O'])} g2pW {tag(r['P_g2pw'])} dict {tag(r['P_dict'])}  g2pW edited target: {str(r['i']) in rendered[('dict', r['id'], 'P_g2pw')]['applied']}")


if __name__ == "__main__":
    {"prep": prep, "summary": summary}[sys.argv[1]]()
