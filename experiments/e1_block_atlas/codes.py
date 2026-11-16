# Lossless cost of storing one model's quantized codes given another's: H(code_W | code_Ref) in bits/weight.
# This compares directly with standalone storage H(code_W), at the same quantization (and so the same Q error).
import sys
import numpy as np
from atlas import FAMILIES, MATS, load_family

PAIRS = {
    "0.6B": [("Qwen3-0.6B", "Qwen3-0.6B-Base"), ("Qwen3Guard-Stream-0.6B", "Qwen3-0.6B"), ("Qwen3-Reranker-0.6B", "Qwen3-0.6B"),
             ("Qwen3Guard-Gen-0.6B", "Qwen3-0.6B-Base"), ("Qwen3-Embedding-0.6B", "Qwen3-0.6B-Base"), ("Qwen3-ASR-0.6B", "Qwen3-0.6B"),
             ("Qwen3-ForcedAligner-0.6B", "Qwen3-ASR-0.6B"), ("Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen3-TTS-12Hz-0.6B-Base"),
             ("Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-0.6B")],
    "1.7B": [("Qwen3-1.7B", "Qwen3-1.7B-Base"), ("Qwen3-ASR-1.7B", "Qwen3-1.7B"), ("Qwen3-ASR-1.7B", "Qwen3-VL-2B-Instruct"),
             ("Qwen3-VL-2B-Instruct", "Qwen3-1.7B"), ("Qwen3-VL-Embedding-2B", "Qwen3-VL-2B-Instruct"), ("Qwen3-VL-2B-Thinking", "Qwen3-VL-2B-Instruct"),
             ("Qwen3-VL-Reranker-2B", "Qwen3-VL-2B-Instruct"), ("Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-12Hz-1.7B-Base"),
             ("Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen3-TTS-12Hz-1.7B-Base"), ("Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-1.7B")],
}
GROUP = 32


def short(m):
    return m.replace("Qwen3", "").replace("-12Hz", "").replace("-0.6B", "").replace("-1.7B", "").replace("-2B", "").strip("-") or "instruct"


def codes(W, bits, group=GROUP):
    """Symmetric per-block codes, matching the block-wise error measured in tree.py."""
    x = W.ravel()
    n = len(x) // group * group
    b = x[:n].reshape(-1, group)
    levels = 2 ** (bits - 1) - 1
    scale = np.abs(b).max(1, keepdims=True) + 1e-12
    return (np.round(b / scale * levels).astype(np.int16) + levels).ravel(), levels * 2 + 1


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cond_entropy(a, b, k):
    """H(a | b) from the empirical joint of two code streams."""
    joint = np.bincount(a.astype(np.int64) * k + b, minlength=k * k).reshape(k, k).astype(np.float64)
    joint /= joint.sum()
    return entropy(joint.ravel()) - entropy(joint.sum(0))


def run(name):
    fam = FAMILIES[name]
    T = load_family(fam)
    layers = fam["layers"]
    print(f"\n=== {name}, layers {layers}: bits/weight, weighted over all sampled matrices; 'saved' = 1 - H(W|Ref)/H(W) ===")
    print(f"  {'model':16s} {'given':14s} | Q4: H(W) H(W|Ref) saved  same | Q8: H(W) H(W|Ref) saved  same")
    for m, ref in PAIRS[name]:
        out = []
        for bits in (4, 8):
            hw, hc, same, n = 0., 0., 0., 0
            for L in layers:
                for t in MATS:
                    a, k = codes(T[(m, L, t)], bits)
                    b, _ = codes(T[(ref, L, t)], bits)
                    pa = np.bincount(a, minlength=k) / len(a)
                    hw += entropy(pa) * len(a)
                    hc += cond_entropy(a, b, k) * len(a)
                    same += (a == b).sum()
                    n += len(a)
            out.append((hw / n, hc / n, 1 - hc / hw, same / n))
        (a4, c4, s4, e4), (a8, c8, s8, e8) = out
        print(f"  {short(m)[:16]:16s} {short(ref)[:14]:14s} | {a4:7.2f} {c4:7.2f} {s4:6.1%} {e4:5.2f} | {a8:7.2f} {c8:7.2f} {s8:6.1%} {e8:5.2f}")


if __name__ == "__main__":
    for name in sys.argv[1:] or FAMILIES:
        run(name)
