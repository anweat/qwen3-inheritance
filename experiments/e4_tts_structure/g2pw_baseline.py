# Supervised G2P baseline: g2pW (Chen et al., Interspeech 2022, Apache-2.0) on the same items as the stage 4 heads and the
# stage 6 prompts (out/g2p_items.json, written from poly_prompt.build_items). Needs .venv-judge (onnxruntime, g2pw 0.1.2).
#   python g2pw_baseline.py release                      # released G2PWModel-v2-onnx, bopomofo labels, simplified -> traditional
#   python g2pw_baseline.py cpp <model_dir>              # g2pW retrained on CPP train with the official config_cpp recipe
# Writes out/g2pw_<name>.json: key -> predicted reading (None if g2pW does not treat the character as polyphonic).
#
# Criteria, fixed before g2pW scored any item (applied by g2pw_summary.py):
#   The comparison that counts is the CPP-retrained model, trained on the same CPP train data as the heads (the heads saw a
#   capped subset, so g2pW has at least as much data). The released model was trained on a Taiwanese-standard dataset and is
#   reported as a reference only, with how often it gives a reading outside the gold/alt pair.
#   Head competitiveness: on CPP minority (stage 4 items), g2pW minus TTS-0.6B@L6 head and minus LLM-0.6B@L8 head, group
#   bootstrap CI. If g2pW is better with the CI above 0, the head is not a competitive G2P and a paper must claim the interface
#   (reading head output -> codebook -> TTS), not the head.
#   LLM inheritance against a real baseline: LLM-0.6B@L8 head minus g2pW on CPP minority, CI above 0 would mean the LLM adds
#   something a dedicated G2P does not have.
#   End to end: g2pW readings through the reading-vector port (existing V_g/V_a audio, scored as in poly_plugin_summary.py)
#   minus the TTS alone on CPP minority, CI above 0 = useful as the front end of the same interface.
#   Safe: g2pW minus TTS on natural2, character bootstrap lower bound above -.05.
#   A g2pW reading outside the head's candidate classes counts as wrong, the same rule as for heads and prompts.
#
# Retraining (added after the first 200 iterations showed 13 min per 200 on the Mac MPS, before any test item was scored; the
# laptop GPU was busy): official configs/config_cpp.py unchanged except the device line (cuda -> mps). Early stop: keep the
# checkpoint with the best dev accuracy (the script already does), stop when 5 consecutive evaluations (1000 iterations) bring
# no improvement, at most 10000. An early-stopped g2pW may be slightly weaker than the paper's (CPP test .9908), which favours
# the heads; reported with its own CPP test accuracy.
import json, os, sys
from g2pw import G2PWConverter

OUT = os.path.expanduser("~/.cache/qwen-all/pron/out")


def main():
    name = sys.argv[1]
    model_dir = os.path.expanduser("~/.cache/qwen-all/g2pw/G2PWModel") if name == "release" else sys.argv[2]
    # The released model has bopomofo labels; the CPP model's labels are already pinyin, so its "bopomofo" style is identity.
    conv = G2PWConverter(model_dir=model_dir, style="pinyin" if name == "release" else "bopomofo",
                         enable_non_tradional_chinese=(name == "release"), num_workers=0)
    items = json.load(open(os.path.join(OUT, "g2p_items.json"), encoding="utf-8"))
    texts = sorted({x["text"] for x in items})
    res = {}
    B = 64
    for b in range(0, len(texts), B):
        chunk = texts[b:b + B]
        for t, r in zip(chunk, conv(chunk)):
            res[t] = r
        print(f"{b + len(chunk)}/{len(texts)}", flush=True)
    pred = {x["key"]: res[x["text"]][x["i"]] for x in items}
    json.dump(pred, open(os.path.join(OUT, f"g2pw_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{sum(v is None for v in pred.values())} of {len(pred)} items without a reading", flush=True)
    print("G2PWDONE", flush=True)


if __name__ == "__main__":
    main()
