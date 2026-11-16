# E6 rule v2 (word guard, build_plan.py): does it repair the neighbour harm without losing the stage 8 gains?
#   python wordrule.py plan       # ~/.cache/qwen-all/e6/v2/plan.json: only sentences whose edits change from v1 to v2
#   (from ../e4_tts_structure) ../../.venv-tts/bin/python pipeline_synth.py ~/.cache/qwen-all/e6/v2/plan.json ~/.cache/qwen-all/e6/v2/render
#   python wordrule.py prep       # audio + stage 4 references for poly_asr.py and judge_score.py
#   ../../.venv/bin/python wordrule.py summary
#
# Fixed before any v2 audio exists (2026-11-10). v2 changes the edits of 65 hard, 5 natural, 34 natural2 and 3 dictionary
# sentences; every other sentence keeps its v1 text, so its stage 8 P audio is reused as the v2 audio.
# Audio: V = v2 text for every changed sentence, one new sample. V1, V2 = two more samples for the 28 changed sentences of
# the neighbour set (neighbour.py), to compare with its P1, P2 (v1 text) and O1, O2.
# v2 replaces v1 in the pipeline only if all three hold:
#   1 Repair: neighbour set, changed sentences, panel (segmental, strict majority of SenseVoice, Paraformer, Whisper),
#     d = mean(V1, V2) - mean(P1, P2) per sentence, character-group bootstrap 95% CI lower bound > 0.
#   2 Still useful: detector (stage 4 ASR, centred on R_g/R_a), CPP minority, v2 pipeline minus O, lower bound > 0.
#   3 Still safe: detector, natural2, v2 pipeline minus O, lower bound > -.05.
# Reported without a threshold: v2 minus v1 on each set (detector and panel, changed sentences only and all), the panel
# versions of 2 and 3, V1/V2 against O1/O2 on the neighbour set, and the dictionary set.
import json, os, shutil, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e4_tts_structure"))
import neighbour as N  # noqa: E402

E6 = N.E6
V2 = f"{E6}/v2"
ROOT = N.ROOT
SRC = N.SRC
v2plan = json.load(open(f"{E6}/plan_v2.json", encoding="utf-8"))
items = v2plan["items"]
r1 = N.rendered
PK = dict(P="V", P_g2pw="V_g2pw", P_dict="V_dict")


def changed():
    """(set, id) -> {v1 kind: v2 edits} for the jobs whose edits differ."""
    out = defaultdict(dict)
    for j in v2plan["jobs"]:
        if j["kind"] in PK and r1[(j["set"], j["id"], j["kind"])]["applied"] != j["edits"]:
            out[(j["set"], j["id"])][j["kind"]] = j
    return out


def nb_ids():
    return {(s, r["id"]) for s, rows in N.chosen().items() for r in rows}


def make_plan():
    jobs, nb = [], nb_ids()
    for key, kinds in changed().items():
        for k, j in kinds.items():
            jobs.append(dict(j, kind=PK[k], seed_offset=3000))
            if key in nb and k == "P":
                for kk, off in (("V1", 1000), ("V2", 2000)):
                    jobs.append(dict(j, kind=kk, seed_offset=off))
    os.makedirs(V2, exist_ok=True)
    json.dump(dict(variant=v2plan["variant"], rule="v2", jobs=jobs), open(f"{V2}/plan.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(len(changed()), "changed sentences,", len(jobs), "jobs")


def prep():
    rend = {(r["set"], r["id"], r["kind"]): r for r in map(json.loads, open(f"{V2}/render/rendered.jsonl", encoding="utf-8"))}
    by = defaultdict(list)
    for (s, i, k) in rend:
        by[(s, i)].append(k)
    for s in ("hard", "natural", "natural2", "dict"):
        d = f"{V2}/asr/{s}"
        os.makedirs(d, exist_ok=True)
        out = []
        for r in items[s]:
            ks = by.get((s, r["id"]))
            if not ks:
                continue
            tx = {}
            for k in ks:
                tx[k] = rend[(s, r["id"], k)]["text"]
                shutil.copy(f"{V2}/render/{s}/{r['id']:04d}_{k}.flac", f"{d}/{r['id']:04d}_{k}.flac")
            src = f"{E6}/asr/dict" if s == "dict" else f"{ROOT}/{SRC[s]}/TTS-0.6B_nonstream"
            for k in ("R_g", "R_a"):
                tx[k] = r["texts"][k]
                shutil.copy(f"{src}/{r['id']:04d}_{k}.flac", f"{d}/{r['id']:04d}_{k}.flac")
            out.append(dict(id=r["id"], gold=r["gold"], alt=r["alt"], text=r["text"], char=r["char"], scores=dict(texts=tx)))
        with open(f"{d}.jsonl", "w", encoding="utf-8") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(s, len(out))


def summary():
    from judge_panel import centred, ctype, scores
    PANEL = ["sensevoice", "paraformer", "whisper"]

    def vote(per, rid, clip):
        v = [x for x in (centred(p.get(rid, {}), clip) for p in per) if x is not None]
        return None if len(v) < 2 else sum(x > 0 for x in v) * 2 > len(v)

    def det(path):
        out = {}
        if os.path.exists(path):
            for x in map(json.loads, open(path, encoding="utf-8")):
                l = x.get("logodds")
                if l:
                    out[x["id"]] = {k: v - (l["R_g"] + l["R_a"]) / 2 > 0 for k, v in l.items()}
        return out

    ch = changed()
    print("==== E6 rule v2 (word guard) ====")
    # 1 repair on the neighbour set
    rows = []
    for s, rs in N.chosen().items():
        nbp = [scores(j, f"{N.NB}/asr/{s}") for j in PANEL]
        v2p = [scores(j, f"{V2}/asr/{s}") for j in PANEL]
        dv, dn = det(f"{V2}/asr/{s}_asr.jsonl"), det(f"{N.NB}/asr/{s}_asr.jsonl")
        for r in rs:
            if (s, r["id"]) not in ch:
                continue
            x = dict(set=s, group=r["group"])
            if ctype(r["gold"], r["alt"]) == "seg":
                a = {k: vote(nbp, r["id"], k) for k in N.KINDS}
                b = {k: vote(v2p, r["id"], k) for k in ("V1", "V2")}
                if None not in a.values() and None not in b.values():
                    x["panel"] = dict(a, **b)
            if r["id"] in dn and r["id"] in dv and all(k in dn[r["id"]] for k in N.KINDS) and all(k in dv[r["id"]] for k in ("V1", "V2")):
                x["det"] = dict(dn[r["id"]], **dv[r["id"]])
            rows.append(x)
    m = lambda v, a, b: (v[a] + v[b]) / 2
    verdict = {}
    print("-- 1 repair: neighbour set, sentences changed by v2, fresh samples")
    for judge in ("panel", "det"):
        rs = [r for r in rows if judge in r]
        if not rs:
            continue
        for lab, f in (("V - P", lambda r: m(r[judge], "V1", "V2") - m(r[judge], "P1", "P2")),
                       ("V - O", lambda r: m(r[judge], "V1", "V2") - m(r[judge], "O1", "O2"))):
            lo, hi = N.boot(rs, f)
            print(f"  {judge:5s} {lab}  n {len(rs):2d} chars {len({r['group'] for r in rs}):2d}  O {np.mean([m(r[judge], 'O1', 'O2') for r in rs]):.3f}"
                  f"  P(v1) {np.mean([m(r[judge], 'P1', 'P2') for r in rs]):.3f}  V(v2) {np.mean([m(r[judge], 'V1', 'V2') for r in rs]):.3f}"
                  f"  d {np.mean([f(r) for r in rs]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
            if judge == "panel" and lab == "V - P":
                verdict["repair"] = lo > 0
    # 2, 3 full sets, single samples: v2 pipeline = V where changed, else stage 8 P
    print("-- 2, 3 full sets (v2 = new V where the edits changed, stage 8 P elsewhere; O = stage 4 audio)")
    for judge in ("det", "panel"):
        for s in ("hard", "natural", "natural2"):
            if judge == "det":
                o, p, v = det(f"{ROOT}/{SRC[s]}/TTS-0.6B_nonstream_asr.jsonl"), det(f"{E6}/asr/{s}_asr.jsonl"), det(f"{V2}/asr/{s}_asr.jsonl")
                get = lambda d, i, k: d.get(i, {}).get(k)
            else:
                po = [scores(j, f"{SRC[s]}/TTS-0.6B_nonstream") for j in PANEL]
                pp = [scores(j, f"{E6}/asr/{s}") for j in PANEL]
                pv = [scores(j, f"{V2}/asr/{s}") for j in PANEL]
                o, p, v = po, pp, pv
                get = lambda d, i, k: vote(d, i, k)
            out = []
            for r in items[s]:
                if judge == "panel" and ctype(r["gold"], r["alt"]) != "seg":
                    continue
                c = "P" in ch.get((s, r["id"]), {})
                x = dict(r, O=get(o, r["id"], "O"), P1=get(p, r["id"], "P"), changed=c)
                x["P2"] = get(v, r["id"], "V") if c else x["P1"]
                if None not in (x["O"], x["P1"], x["P2"]):
                    out.append(x)
            parts = [("CPP minority", [r for r in out if r["kind"] == "min"]), ("CPP majority", [r for r in out if r["kind"] == "maj"])] if s == "hard" else [(s, out)]
            for lab, rs in parts:
                lo, hi = N.boot(rs, lambda r: r["P2"] - r["O"])
                l2, h2 = N.boot(rs, lambda r: r["P2"] - r["P1"])
                cc = [r for r in rs if r["changed"]]
                print(f"  {judge:5s} {lab:13s} n {len(rs):3d}  O {np.mean([r['O'] for r in rs]):.3f}  v1 {np.mean([r['P1'] for r in rs]):.3f}"
                      f"  v2 {np.mean([r['P2'] for r in rs]):.3f}  v2-O {np.mean([r['P2'] - r['O'] for r in rs]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
                      f"  v2-v1 {np.mean([r['P2'] - r['P1'] for r in rs]):+.3f} [{l2:+.3f}, {h2:+.3f}]"
                      f"  changed n {len(cc)}: v1 {np.mean([r['P1'] for r in cc]) if cc else float('nan'):.3f} v2 {np.mean([r['P2'] for r in cc]) if cc else float('nan'):.3f}")
                if judge == "det" and lab == "CPP minority":
                    verdict["useful"] = lo > 0
                if judge == "det" and lab == "natural2":
                    verdict["safe"] = lo > -.05
    d = det(f"{V2}/asr/dict_asr.jsonl")
    if d:
        for i, ks in d.items():
            print(f"  dict id {i}: " + " ".join(f"{k}={'ok' if v else '--'}" for k, v in ks.items() if k.startswith("V")))
    print(f"VERDICT: repair {verdict.get('repair')}  still useful {verdict.get('useful')}  still safe {verdict.get('safe')}"
          f"  -> adopt v2: {all(verdict.get(k) for k in ('repair', 'useful', 'safe'))}")


if __name__ == "__main__":
    {"plan": make_plan, "prep": prep, "summary": summary}[sys.argv[1]]()
