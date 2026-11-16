# Positive control for the reading detector, and a reading rate on natural text. CPP over-samples rare readings and has no
# audio with a known reading of the real polyphone, so stage 4 could only validate the detector on homophone substitutes.
# Here: natural Mandarin sentences (AISHELL, FLEURS) with human recordings, gold readings reviewed by hand (natural_gold.json).
# The human is assumed to read correctly. For each polyphone position, synthesize O, R_g and R_a as in poly_tts.py, and score
# H (the human clip) and O with the ASR detector, centred on R_g and R_a of the same sentence.
#
# Criteria, fixed before any audio from these sentences was scored:
#   The detector transfers to real polyphones if at least 90% of human positions are scored as gold.
#   Below that, stage 4 rates may be biased and must be reported as such.
# Reported without a threshold: rate on O by model, and for positions where gold is the CPP minority vs majority.
# Limits: references use the same two TTS voices, so human clips differ in voice from their references. Positions with a
# neutral tone, 一, 不, or no homophone are excluded, which removes many everyday cases such as 地/de5, 过/guo5, 子/zi5.
import argparse, json, os
import soundfile as sf
import poly_tts as P
from stitch import SPEECH


def items():
    gold = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_gold.json"), encoding="utf-8"))
    text = {}
    for l in open(os.path.join(SPEECH, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        text[r["path"]] = r["text"]
    out = []
    for k, x in enumerate(gold):
        out.append(dict(id=k, group=x["char"], char=x["char"], kind="min" if x["gold"] != x["maj"] else "maj", gold=x["gold"], alt=x["alt"],
                        text=text[x["path"]], i=x["i"], hg=x["hg"], ha=x["ha"], calibrate=False, path=x["path"]))
        assert text[x["path"]][x["i"]] == x["char"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TTS-0.6B")
    ap.add_argument("--batch", type=int, default=9)
    a = ap.parse_args()
    P.OUT = os.path.join(P.ROOT, "tts_natural")
    rows = items()
    print(f"{len(rows)} positions", flush=True)
    P.run_model(a.model, P.MODELS[a.model], rows, a.batch, False)
    print("NATURALDONE", flush=True)
