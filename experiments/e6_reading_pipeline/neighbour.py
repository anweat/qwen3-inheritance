# E6 follow-up: do edits elsewhere in a sentence change how the TTS reads a polyphone the pipeline left alone?
# Prompted by the stage 8 panel check (exploratory there): on natural2, unedited targets went .913 -> .739 under the panel
# while edited targets went .800 -> .978. That came from one O and one P sample per sentence, so it may be sampling noise.
#   python neighbour.py plan      # (any venv) writes ~/.cache/qwen-all/e6/nb/plan.json
#   ../../.venv-tts/bin/python ../e4_tts_structure/pipeline_synth.py ~/.cache/qwen-all/e6/nb/plan.json ~/.cache/qwen-all/e6/nb/render
#     (run from ../e4_tts_structure)
#   python neighbour.py prep      # lays out audio + stage 4 references for poly_asr.py and judge_score.py
#   ../../.venv/bin/python neighbour.py summary
#
# Design, fixed before any of this audio exists:
#   Sentences: every hard, natural and natural2 item whose target the stage 8 pipeline did not edit, but where it edited at
#   least one other position (86 sentences; 63 with a segmental gold/alt contrast). Unedited targets with no other edit are
#   left out, since there P and O are the same text.
#   Fresh audio: O1, O2 = the unmodified sentence; P1, P2 = the sentence with exactly the edits stage 8 applied. Same TTS,
#   voice and settings; new sampling seeds. The stage 8 O and P samples are not used for the verdict (the hypothesis came
#   from them); they are reported for reference.
#   Panel: SenseVoice, Paraformer, Whisper, strict majority, segmental items only (as in stage 5 and 8). Detector: the stage 4
#   ASR detector, all items.
#   Per sentence d = mean(P1, P2 says gold) - mean(O1, O2 says gold).
#   Primary: panel, all three sets pooled, mean d with character-group bootstrap 95% CI.
#     "Neighbour edits harm unedited targets" if the upper bound < 0. Otherwise no harm is shown, and the report also says
#     whether a harm larger than 5 points is excluded (lower bound > -.05).
#   Secondary, reported without a threshold: natural2 alone (where the drop was seen); the detector pooled and per set.
#   Noise baseline, reported: how often two samples of the same text disagree (O1 vs O2, P1 vs P2), against O vs P.
import json, os, shutil, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e4_tts_structure"))
E6 = os.path.expanduser("~/.cache/qwen-all/e6")
NB = f"{E6}/nb"
ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
SRC = dict(hard="tts_out", natural="tts_natural", natural2="tts_natural2")
SETS = ("hard", "natural", "natural2")
KINDS = ("O1", "O2", "P1", "P2")
plan = json.load(open(f"{E6}/plan.json", encoding="utf-8"))
rendered = {(r["set"], r["id"], r["kind"]): r for r in map(json.loads, open(f"{E6}/render/rendered.jsonl", encoding="utf-8"))}


def chosen():
    out = {}
    for s in SETS:
        out[s] = []
        for r in plan["items"][s]:
            app = rendered[(s, r["id"], "P")]["applied"]
            if str(r["i"]) not in app and app:
                out[s].append(r)
    return out


def make_plan():
    jobs = []
    for s, rows in chosen().items():
        for r in rows:
            app = rendered[(s, r["id"], "P")]["applied"]
            for k, off in zip(KINDS, (1000, 2000, 1000, 2000)):
                jobs.append(dict(set=s, id=r["id"], text=r["text"], voice=r["id"] % 2, kind=k,
                                 edits=dict(app) if k[0] == "P" else {}, seed_offset=off))
    os.makedirs(NB, exist_ok=True)
    json.dump(dict(variant=plan["variant"], jobs=jobs), open(f"{NB}/plan.json", "w", encoding="utf-8"), ensure_ascii=False)
    print({s: len(v) for s, v in chosen().items()}, len(jobs), "jobs")


def prep():
    rend = {(r["set"], r["id"], r["kind"]): r for r in map(json.loads, open(f"{NB}/render/rendered.jsonl", encoding="utf-8"))}
    for s, rows in chosen().items():
        d = f"{NB}/asr/{s}"
        os.makedirs(d, exist_ok=True)
        out = []
        for r in rows:
            tx = {}
            for k in KINDS:
                tx[k] = rend[(s, r["id"], k)]["text"]
                shutil.copy(f"{NB}/render/{s}/{r['id']:04d}_{k}.flac", f"{d}/{r['id']:04d}_{k}.flac")
            for k in ("R_g", "R_a"):
                tx[k] = r["texts"][k]
                shutil.copy(f"{ROOT}/{SRC[s]}/TTS-0.6B_nonstream/{r['id']:04d}_{k}.flac", f"{d}/{r['id']:04d}_{k}.flac")
            out.append(dict(id=r["id"], gold=r["gold"], alt=r["alt"], text=r["text"], char=r["char"], scores=dict(texts=tx)))
        with open(f"{d}.jsonl", "w", encoding="utf-8") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(s, len(out))


def boot(rows, f, n=4000):
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    ks = list(by)
    rng = np.random.default_rng(0)
    v = [np.mean([f(x) for j in rng.integers(0, len(ks), len(ks)) for x in by[ks[j]]]) for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def summary():
    from judge_panel import centred, ctype, scores
    PANEL = ["sensevoice", "paraformer", "whisper"]

    def vote(per, rid, clip):
        v = [x for x in (centred(p.get(rid, {}), clip) for p in per) if x is not None]
        return None if len(v) < 2 else sum(x > 0 for x in v) * 2 > len(v)

    def det(path):
        out = {}
        for x in map(json.loads, open(path, encoding="utf-8")):
            l = x.get("logodds")
            if l:
                out[x["id"]] = {k: v - (l["R_g"] + l["R_a"]) / 2 > 0 for k, v in l.items()}
        return out

    res = {}
    for s, rows in chosen().items():
        newp = [scores(j, f"{NB}/asr/{s}") for j in PANEL]
        oldp = [scores(j, f"{SRC[s]}/TTS-0.6B_nonstream") for j in PANEL]
        e6p = [scores(j, f"{E6}/asr/{s}") for j in PANEL]
        dn, do, de = det(f"{NB}/asr/{s}_asr.jsonl"), det(f"{ROOT}/{SRC[s]}/TTS-0.6B_nonstream_asr.jsonl"), det(f"{E6}/asr/{s}_asr.jsonl")
        for r in rows:
            x = dict(set=s, group=r["group"], seg=ctype(r["gold"], r["alt"]) == "seg")
            pv = {k: vote(newp, r["id"], k) for k in KINDS}
            if x["seg"] and None not in pv.values():
                x["panel"] = pv
                x["panel_old"] = (vote(oldp, r["id"], "O"), vote(e6p, r["id"], "P"))
            if r["id"] in dn and all(k in dn[r["id"]] for k in KINDS):
                x["det"] = {k: dn[r["id"]][k] for k in KINDS}
                if r["id"] in do and r["id"] in de:
                    x["det_old"] = (do[r["id"]]["O"], de[r["id"]]["P"])
            res[(s, r["id"])] = x
    rows = list(res.values())
    d = lambda v: (v["P1"] + v["P2"]) / 2 - (v["O1"] + v["O2"]) / 2
    print("==== E6 neighbour-edit check: unedited targets in sentences with other edits; fresh samples O1, O2, P1, P2 ====")
    verdict = None
    for judge in ("panel", "det"):
        print(f"-- {'panel (segmental, strict majority of 3)' if judge == 'panel' else 'detector (all items)'}")
        for label, sel in (("pooled", SETS), ("hard", ("hard",)), ("natural", ("natural",)), ("natural2", ("natural2",))):
            rs = [r for r in rows if judge in r and r["set"] in sel]
            if not rs:
                continue
            f = lambda r: d(r[judge])
            lo, hi = boot(rs, f)
            o = np.mean([(r[judge]["O1"] + r[judge]["O2"]) / 2 for r in rs])
            p = np.mean([(r[judge]["P1"] + r[judge]["P2"]) / 2 for r in rs])
            oo = np.mean([r[judge]["O1"] != r[judge]["O2"] for r in rs])
            pp = np.mean([r[judge]["P1"] != r[judge]["P2"] for r in rs])
            op = np.mean([np.mean([r[judge][a] != r[judge][b] for a in ("O1", "O2") for b in ("P1", "P2")]) for r in rs])
            old = [r[judge + "_old"] for r in rs if r.get(judge + "_old") and None not in r[judge + "_old"]]
            olds = f"  stage 8 single samples O {np.mean([a for a, _ in old]):.3f} P {np.mean([b for _, b in old]):.3f} (n {len(old)})" if old else ""
            print(f"  {label:9s} n {len(rs):3d} chars {len({r['group'] for r in rs}):3d}  O {o:.3f}  P {p:.3f}  d {p - o:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"  disagree O1/O2 {oo:.2f} P1/P2 {pp:.2f} O/P {op:.2f}{olds}")
            if judge == "panel" and label == "pooled":
                verdict = (lo, hi)
    if verdict:
        print(f"VERDICT: neighbour edits harm unedited targets: {verdict[1] < 0};  harm larger than 5 points excluded: {verdict[0] > -.05}")


if __name__ == "__main__":
    {"plan": make_plan, "prep": prep, "summary": summary}[sys.argv[1]]()
