# Larger natural-frequency sample for the do-no-harm test. Polyphones are taken as they occur in CPP test sentences, with up to
# 8 positions per character, single-character tokens only, gold readings reviewed by hand before any audio (natural2_gold.json).
# Encyclopedic text with many proper names, so this is not everyday speech. No human audio here: only O, R_g, R_a.
# Criteria are in poly_gate.py.
import argparse, json, os
import poly_tts as P

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TTS-0.6B")
    ap.add_argument("--batch", type=int, default=9)
    a = ap.parse_args()
    gold = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural2_gold.json"), encoding="utf-8"))
    rows = [dict(id=k, group=x["char"], char=x["char"], kind="min" if x["gold"] != x["maj"] else "maj", gold=x["gold"], alt=x["alt"], text=x["text"],
                 i=x["i"], hg=x["hg"], ha=x["ha"], calibrate=False) for k, x in enumerate(gold)]
    P.OUT = os.path.join(P.ROOT, "tts_natural2")
    print(f"{len(rows)} positions", flush=True)
    P.run_model(a.model, P.MODELS[a.model], rows, a.batch, False)
    print("NATURAL2DONE", flush=True)
