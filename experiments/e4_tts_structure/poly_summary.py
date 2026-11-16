# Applies the criteria in the poly_tts.py header to saved scores. No model needed.
#   python poly_summary.py [--calibration-only] <jsonl> [<jsonl> ...]
import json, os, sys
import numpy as np

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
MAX_NEW = 400
WINDOW = 7


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    asr = {}
    p = path[:-len(".jsonl")] + "_asr.jsonl"
    if os.path.exists(p):
        asr = {x["id"]: x.get("logodds") for x in map(json.loads, open(p, encoding="utf-8"))}
    for r in rows:
        s, l = r["scores"], asr.get(r["id"])
        for k in s["texts"]:
            if k not in s:
                s[k] = {}
            if "ce_diff" in s[k]:
                d = np.array(s[k]["ce_diff"])
                m = np.convolve(d, np.ones(WINDOW), mode="same")
                s[k]["ce_peak"] = float(m[np.argmax(np.abs(m))])
            if l:
                s[k]["asr_raw"] = l[k]
                # Removes the language-model prior shared by this sentence's audio: position between the two references.
                s[k]["asr_centred"] = l[k] - (l["R_g"] + l["R_a"]) / 2
    return rows


def rates(rows, det):
    cal = [r for r in rows if r["calibrate"] and det in r["scores"]["C_g"]]
    if not cal:
        return float("nan"), float("nan"), 0
    return np.mean([r["scores"]["C_g"][det] > 0 for r in cal]), np.mean([r["scores"]["C_a"][det] < 0 for r in cal]), len(cal)


def theta(rows, det, sens, spec):
    denom = sens + spec - 1
    if denom <= 0 or not rows:
        return None
    return (np.mean([r["scores"]["O"][det] > 0 for r in rows]) + spec - 1) / denom


def boot(rows, fn, n=4000):
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    keys = list(by)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        s = [x for j in rng.integers(0, len(keys), len(keys)) for x in by[keys[j]]]
        v = fn(s)
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def summary(rows):
    a = np.array(rows, dtype=float).sum(0)
    return a[0] / a[1], a[2] / max(1, a[3])


def analyse(path, cal_only):
    rows = load(path)
    n = len(rows)
    med = np.median([r["scores"]["O"]["frames"] / max(1, len(r["text"])) for r in rows])
    bad = {r["id"] for r in rows if max(r["scores"]["O"]["frames"], *r["scores"]["ref_frames"]) >= MAX_NEW - 1
           or r["scores"]["O"]["frames"] / max(1, len(r["text"])) > 2.5 * med}
    rows = [r for r in rows if r["id"] not in bad]
    print(f"\n==== {os.path.basename(path)}: {n} items, dropped {len(bad)} with runaway audio; median {med:.2f} frames per character ====")
    table = []
    for det in ("dtw_total", "dtw_window", "ce_ratio", "ce_peak", "asr_raw", "asr_centred"):
        sens, spec, m = rates(rows, det)
        if m:
            table.append(((sens + spec) / 2, det, sens, spec))
            print(f"  calibration {det:12s} n {m:4d}  says-g {sens:.3f}  says-a {spec:.3f}  balanced {(sens + spec) / 2:.3f}")
    bal, best, sens, spec = max(table)
    valid = bal >= .90 and min(sens, spec) >= .85
    print(f"  primary detector {best}: {'passes validity' if valid else 'FAILS validity, result not interpretable'}")
    if cal_only:
        return
    rows = [r for r in rows if best in r["scores"]["O"]]
    for kind, label in (("min", "minority"), ("maj", "majority")):
        sub = [r for r in rows if r["kind"] == kind]
        if sub:
            th = theta(sub, best, sens, spec)
            lo, hi = boot(sub, lambda s: theta(s, best, sens, spec))
            print(f"  {label}: n {len(sub)}, groups {len({r['group'] for r in sub})}, observed gold rate {np.mean([r['scores']['O'][best] > 0 for r in sub]):.3f}, "
                  f"corrected {th:.3f} [{lo:.3f}, {hi:.3f}]")
    probe = os.path.join(ROOT, "out", "pron_ci.json")
    sub = [r for r in rows if r["kind"] == "min"]
    if sub and os.path.exists(probe):
        t = json.load(open(probe))["table"]
        cov = sorted({r["group"] for r in sub})
        for name in ("LLM-0.6B|t+2", "TTS-0.6B|t+2", "TTS-1.7B|t+2"):
            print(f"  probe recall on covered groups, {name}: {summary([t[name]['rows'][g] for g in cov])[1]:.3f}")
        T = summary([t["LLM-0.6B|t+2"]["rows"][g] for g in cov])[1] - .05
        lo, hi = boot(sub, lambda s: theta(s, best, sens, spec))
        verdict = "not interpretable" if not valid else "no room" if lo >= T else "room" if hi < T else "inconclusive"
        print(f"  decision threshold T {T:.3f}: {verdict}")


if __name__ == "__main__":
    args = sys.argv[1:]
    cal = "--calibration-only" in args
    for p in args:
        if p != "--calibration-only":
            analyse(p, cal)
