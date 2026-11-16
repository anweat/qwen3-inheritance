# E6 step 1: can the CPP-retrained g2pW front end shrink for the device? Run with .venv-judge.
#   python g2pw_quant.py quantize      # writes ~/.cache/qwen-all/g2pw/G2PWModel-cpp-int8 (onnxruntime dynamic int8 weights)
#   python g2pw_quant.py test          # CPP test subset accuracy, fp32 and int8, plus CPU latency per sentence and file sizes
#   (stage 4 items: ../e4_tts_structure/g2pw_baseline.py cpp_int8 <dir>, then compare out/g2pw_cpp.json and g2pw_cpp_int8.json)
#
# Criterion, fixed before int8 scored anything: the pipeline uses int8 if, against fp32,
#   (a) accuracy on a seeded 2000-sentence subset of the CPP test set drops by at most .005, and
#   (b) accuracy on the 230 stage 4 CPP minority positions drops by at most .02 (about 5 positions).
# Otherwise the pipeline uses fp32. Latency (single sentence, M4 CPU, onnxruntime default threads) and size are reported.
import json, os, random, shutil, sys, time
import numpy as np

FP = os.path.expanduser("~/.cache/qwen-all/g2pw/G2PWModel-cpp")
Q8 = FP + "-int8"
CPP = os.path.expanduser("~/.cache/qwen-all/g2pw/repo/cpp_dataset")
OUT = os.path.expanduser("~/.cache/qwen-all/e6")


def quantize():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    os.makedirs(Q8, exist_ok=True)
    for f in os.listdir(FP):
        if f not in ("g2pw.onnx", "best_accuracy.pth"):
            shutil.copy(os.path.join(FP, f), os.path.join(Q8, f))
    quantize_dynamic(os.path.join(FP, "g2pw.onnx"), os.path.join(Q8, "g2pw.onnx"), weight_type=QuantType.QInt8)
    print({d: round(os.path.getsize(os.path.join(d, "g2pw.onnx")) / 1e6, 1) for d in (FP, Q8)})


def cpp_test(n=2000):
    sents = open(os.path.join(CPP, "test.sent"), encoding="utf-8").read().strip().split("\n")
    labels = open(os.path.join(CPP, "test.lb"), encoding="utf-8").read().strip().split("\n")
    rows = []
    for s, l in zip(sents, labels):
        i = s.index("▁")
        rows.append((s.replace("▁", ""), i, l))
    random.Random(0).shuffle(rows)
    return rows[:n]


def test():
    from g2pw import G2PWConverter
    os.makedirs(OUT, exist_ok=True)
    rows = cpp_test()
    nat = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e4_tts_structure", "natural2_gold.json"), encoding="utf-8"))
    lat_texts = sorted({x["text"] for x in nat if 20 <= len(x["text"]) <= 40})[:30]
    res = {}
    for name, d in (("fp32", FP), ("int8", Q8)):
        conv = G2PWConverter(model_dir=d, style="bopomofo", enable_non_tradional_chinese=False, num_workers=0)
        texts = [t for t, _, _ in rows]
        pred = []
        for b in range(0, len(texts), 64):
            pred += conv(texts[b:b + 64])
        acc = float(np.mean([p[i] == l for p, (_, i, l) in zip(pred, rows)]))
        conv(lat_texts[:3])  # warm-up
        t = []
        for s in lat_texts:
            t0 = time.perf_counter(); conv([s]); t.append((time.perf_counter() - t0) * 1000)
        res[name] = dict(cpp_test_subset_acc=acc, median_ms_per_sentence=float(np.median(t)),
                         onnx_mb=os.path.getsize(os.path.join(d, "g2pw.onnx")) / 1e6, pred=[p[i] for p, (_, i, _) in zip(pred, rows)])
        print(name, {k: v for k, v in res[name].items() if k != "pred"}, flush=True)
    res["agreement"] = float(np.mean([a == b for a, b in zip(res["fp32"]["pred"], res["int8"]["pred"])]))
    json.dump(res, open(os.path.join(OUT, "g2pw_quant.json"), "w"))
    print("agreement", res["agreement"], "\nQUANTDONE", flush=True)


def latency():
    """Re-measure latency only. g2pw 0.1.2 treats num_workers=0 as unset and falls back to config.num_workers (2), so every
    call in `test` spawned two DataLoader processes (about 8.6 s per call on macOS). Accuracy is unaffected by this."""
    from g2pw import G2PWConverter
    nat = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e4_tts_structure", "natural2_gold.json"), encoding="utf-8"))
    lat_texts = sorted({x["text"] for x in nat if 20 <= len(x["text"]) <= 40})[:30]
    res = json.load(open(os.path.join(OUT, "g2pw_quant.json")))
    for name, d in (("fp32", FP), ("int8", Q8)):
        conv = G2PWConverter(model_dir=d, style="bopomofo", enable_non_tradional_chinese=False, num_workers=0)
        conv.num_workers = 0
        conv(lat_texts[:3])
        t = []
        for s in lat_texts * 3:
            t0 = time.perf_counter(); conv([s]); t.append((time.perf_counter() - t0) * 1000)
        res[name]["median_ms_per_sentence"] = float(np.median(t))
        res[name]["p90_ms_per_sentence"] = float(np.percentile(t, 90))
        print(name, f"median {np.median(t):.1f} ms, p90 {np.percentile(t, 90):.1f} ms per sentence (2 onnxruntime threads)", flush=True)
    json.dump(res, open(os.path.join(OUT, "g2pw_quant.json"), "w"))


if __name__ == "__main__":
    {"quantize": quantize, "test": test, "latency": latency}[sys.argv[1]]()
