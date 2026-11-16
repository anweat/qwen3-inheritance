# Validates a panel of independent speech recognizers as judges of which reading a polyphone was given, before any of them is
# used to mine or optimize anything. Scores come from judge_score.py; nothing here needs a model.
#   python judge_panel.py
#
# Judges: Charsiu (frame classifier over tonal phones, no language model, sentence aligned around a free slot), SenseVoiceSmall
# and Paraformer-zh (non-autoregressive), Whisper large-v3-turbo (teacher forced with the left context, so it has a strong
# language prior). The existing Qwen3-ASR detector is the baseline; it shares a family with the system under test.
#
# Each judge gives every clip a gold-minus-alt log odds, centred on R_g and R_a of the same sentence as in poly_summary.py,
# and an uncentred top reading among all readings, which lets it say that neither offered reading was spoken.
#
# Validation data, all with a known spoken reading:
#   TTS audio: C_g and C_a from TTS-0.6B, a second homophone of each reading, not used for centring.
#   Human audio: H from poly_natural.py, gold reviewed by hand, centred on TTS-0.6B references of the same sentence.
#
# Criteria, fixed before any of these judges scored any audio. All on point estimates, with group bootstrap intervals reported:
#   Segmental contrasts (initial or final differs) qualify a judge if TTS balanced >= .90, both directions >= .85, and human
#   positions scored as gold >= .90.
#   Tone-only contrasts qualify it separately, by the same three numbers on tone-only items.
#   A judge can report a third reading if, on C_g, C_a, R_g, R_a and on H separately, its top reading matches the spoken initial
#   and final at >= .80, and it leaves the offered pair (initial and final differ from both) on <= .10 of clips. Tones are not
#   required because they are the known weak point.
#   The panel is established for a contrast type if at least two judges outside the Qwen family qualify for it.
#
# Panel rule, fixed with the criteria, used by later stages and applied here to the listening clips:
#   gold or alt: strict majority of the judges qualified for that contrast type, Qwen3-ASR not counted.
#   third: a third-reading judge leaves the pair and no strict majority of qualified judges says otherwise with a positive margin.
#   disputed: everything else.
#
# Reported without a threshold, because the answers were seen first: agreement with the 36 blind listening clips, including
# whether the six "neither" clips are flagged. Also pairwise agreement between judges on O, as a rough check of independence.
#
# Limits: the human set is small and from a hand gold by one labeler, and its speakers may read some characters differently
# from the dictionary. C_g and C_a test homophone substitutes, not the polyphone itself. Judges score a reading, not audio quality.
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
JUDGES = os.environ.get("JUDGES", "charsiu,sensevoice,paraformer,whisper,qwen").split(",")  # e.g. whisper_cuda in place of whisper
PANEL = [j for j in JUDGES if j != "qwen"]


def seg(x):
    return re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*?)(\d)$", x).groups()[:2]


def ctype(gold, alt):
    return "tone" if seg(gold) == seg(alt) else "seg"


def rows_of(name):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, name + ".jsonl"), encoding="utf-8"))}


def scores(judge, source):
    """id -> clip -> {lo, top}. Qwen comes from the stage 4 files and has no top reading."""
    if judge == "qwen":
        out = {}
        for x in map(json.loads, open(os.path.join(ROOT, source + "_asr.jsonl"), encoding="utf-8")):
            if x.get("logodds"):
                out[x["id"]] = {k: dict(lo=v) for k, v in x["logodds"].items()}
        return out
    p = os.path.join(ROOT, "judges", judge, source.replace("/", "_") + ".jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    for x in map(json.loads, open(p, encoding="utf-8")):
        out.setdefault(x["id"], {})[x["clip"]] = x
    return out


def centred(s, clip):
    c = s.get(clip)
    if not c or c.get("lo") is None or s.get("R_g", {}).get("lo") is None or s.get("R_a", {}).get("lo") is None:
        return None
    return c["lo"] - (s["R_g"]["lo"] + s["R_a"]["lo"]) / 2


def boot(sub, fn, n=2000):
    by = {}
    for x in sub:
        by.setdefault(x["group"], []).append(x)
    keys = list(by)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n):
        s = [x for j in rng.integers(0, len(keys), len(keys)) for x in by[keys[j]]]
        v.append(fn(s))
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def binary(judge, kind):
    """Rows of (group, says_gold, wanted) for one contrast type."""
    tts, hum = [], []
    S = scores(judge, "tts_out/TTS-0.6B_nonstream")
    for i, r in T.items():
        if r["calibrate"] and ctype(r["gold"], r["alt"]) == kind and i in S:
            for clip, want in (("C_g", True), ("C_a", False)):
                c = centred(S[i], clip)
                if c is not None:
                    tts.append(dict(group=r["group"], ok=(c > 0) == want, want=want))
    S = scores(judge, "tts_natural/TTS-0.6B_nonstream")
    for i, r in N.items():
        if ctype(r["gold"], r["alt"]) == kind and i in S:
            c = centred(S[i], "H")
            if c is not None:
                hum.append(dict(group=r["group"], ok=c > 0))
    return tts, hum


def third(judge):
    """Top reading on clips with a known spoken reading: does it match, and does it wrongly leave the pair?"""
    out = {}
    for label, source, table, clips in (("TTS", "tts_out/TTS-0.6B_nonstream", T, {"C_g": "gold", "C_a": "alt", "R_g": "gold", "R_a": "alt"}),
                                        ("human", "tts_natural/TTS-0.6B_nonstream", N, {"H": "gold"})):
        S, v = scores(judge, source), []
        for i, r in table.items():
            for clip, w in clips.items():
                t = S.get(i, {}).get(clip, {}).get("top")
                if t:
                    top = seg(t[0][0])
                    v.append(dict(group=r["group"], hit=top == seg(r[w]), left=top not in (seg(r["gold"]), seg(r["alt"]))))
        out[label] = v
    return out


def main():
    verdict = {}
    for j in JUDGES:
        verdict[j] = {}
        print(f"\n==== {j} ====")
        for kind in ("seg", "tone"):
            tts, hum = binary(j, kind)
            if not tts or not hum:
                print(f"  {kind}: no scores (TTS {len(tts)}, human {len(hum)})")
                verdict[j][kind] = False
                continue
            sg = np.mean([x["ok"] for x in tts if x["want"]])
            sa = np.mean([x["ok"] for x in tts if not x["want"]])
            bal = (sg + sa) / 2
            h = np.mean([x["ok"] for x in hum])
            lo, hi = boot(tts, lambda s: np.mean([x["ok"] for x in s]))
            hl, hh = boot(hum, lambda s: np.mean([x["ok"] for x in s]))
            ok = bal >= .90 and min(sg, sa) >= .85 and h >= .90
            verdict[j][kind] = ok
            print(f"  {kind:4s} TTS n {len(tts):3d} says-g {sg:.3f} says-a {sa:.3f} balanced {bal:.3f} [{lo:.3f}, {hi:.3f}] | human n {len(hum):3d} gold {h:.3f} [{hl:.3f}, {hh:.3f}] -> {'QUALIFIES' if ok else 'no'}")
        t = third(j)
        if t["TTS"] and t["human"]:
            ok = True
            for label, v in t.items():
                hit, left = np.mean([x["hit"] for x in v]), np.mean([x["left"] for x in v])
                ok &= hit >= .80 and left <= .10
                print(f"  top reading on {label:5s} n {len(v):4d} matches spoken {hit:.3f}, leaves pair {left:.3f}")
            verdict[j]["third"] = bool(ok)
            print(f"  third reading: {'CAN REPORT' if ok else 'no'}")
        else:
            verdict[j]["third"] = False
            print("  third reading: no top readings")

    print("\n==== panel ====")
    for kind in ("seg", "tone"):
        q = [j for j in PANEL if verdict[j][kind]]
        print(f"  {kind}: qualified outside Qwen {q} -> {'ESTABLISHED' if len(q) >= 2 else 'not established'}")
    print(f"  third reading judges: {[j for j in PANEL if verdict[j]['third']]}")

    src = "tts_out/TTS-0.6B_nonstream"
    S = {j: scores(j, src) for j in JUDGES}
    print("\nagreement between judges on O (share of clips with the same sign, exploratory)")
    for a in range(len(JUDGES)):
        for b in range(a + 1, len(JUDGES)):
            v = []
            for i in T:
                x, y = [centred(S[j].get(i, {}), "O") for j in (JUDGES[a], JUDGES[b])]
                if x is not None and y is not None:
                    v.append((x > 0) == (y > 0))
            if v:
                print(f"  {JUDGES[a]:10s} {JUDGES[b]:10s} n {len(v):3d} {np.mean(v):.3f}")

    key = os.path.join(HERE, "listening_scored.json")
    if os.path.exists(key):
        print("\nblind listening, exploratory (answers were seen before these judges ran)")
        L = json.load(open(key, encoding="utf-8"))
        table = []
        for x in L:
            row, r = {}, T[x["id"]]
            for j in JUDGES:
                s = S[j].get(x["id"], {})
                c = centred(s, "O")
                row[j] = dict(says=None if c is None else ("gold" if c > 0 else "alt"), top=(s.get("O", {}).get("top") or [[None]])[0][0])
            table.append((x, row))
        for j in JUDGES:
            ans = [(x, row[j]) for x, row in table if x["heard"] != "neither" and row[j]["says"]]
            non = [(x, row[j]) for x, row in table if x["heard"] == "neither"]
            if ans:
                agree = np.mean([x["heard"] == v["says"] for x, v in ans])
                tp = [x for x, v in ans if v["top"]]
                left_ans = sum(seg(v["top"]) not in (seg(x["gold"]), seg(x["alt"])) for x, v in ans if v["top"])
                left_non = sum(seg(v["top"]) not in (seg(x["gold"]), seg(x["alt"])) for x, v in non if v["top"])
                print(f"  {j:10s} agrees on answered {agree:.3f} (n {len(ans)}), top reading leaves the pair: answered {left_ans}/{len(tp)}, neither {left_non}/{len(non)}")
        print("  neither clips, top reading by judge:")
        for x, row in table:
            if x["heard"] == "neither":
                print(f"    #{x['n']:2d} {x['char']} {x['gold']}/{x['alt']} " + " ".join(f"{j}={row[j]['top']}" for j in PANEL))


T = rows_of("tts_out/TTS-0.6B_nonstream")
N = rows_of("tts_natural/TTS-0.6B_nonstream")
if __name__ == "__main__":
    main()
