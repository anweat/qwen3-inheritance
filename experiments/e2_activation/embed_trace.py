# Trace training-data footprints through embedding rows: which tokens did each function's training move?
# Tied embeddings get gradient mainly for tokens seen or predicted, so row drift hints at the data used.
import json, os, re, struct, sys, time, unicodedata, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from common import PREFIX, read_decoder, tokenizer

CACHE = os.environ["EMBED_CACHE"]
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HF = f"{ENDPOINT}/Qwen"
MODELS = {"Qwen3-0.6B-Base": "model.embed_tokens.weight", "Qwen3-0.6B": "model.embed_tokens.weight",
          "Qwen3Guard-Gen-0.6B": "model.embed_tokens.weight", "Qwen3Guard-Stream-0.6B": "model.embed_tokens.weight",
          "Qwen3-Embedding-0.6B": "embed_tokens.weight", "Qwen3-Reranker-0.6B": "model.embed_tokens.weight",
          "Qwen3-ASR-0.6B": "thinker.model.embed_tokens.weight", "Qwen3-ForcedAligner-0.6B": "thinker.model.embed_tokens.weight",
          "Qwen3-TTS-12Hz-0.6B-Base": "talker.model.text_embedding.weight"}


def get(url, a, b, tries=6):
    for _ in range(tries):
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(r, timeout=300) as resp:
                d = resp.read()
            if len(d) == b - a + 1:
                return d
        except Exception as e:
            print("retry", e, flush=True)
    raise IOError("failed")


def api(repo, tries=8):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(f"{ENDPOINT}/api/models/Qwen/{repo}", timeout=60))
        except Exception as err:
            print("api retry", err, flush=True)
            time.sleep(5)
    raise IOError("api failed")


def table(repo, key):
    p = os.path.join(CACHE, repo + ".npy")
    if os.path.exists(p):
        return np.load(p)
    # Prefer weights already on disk.
    if repo in PREFIX:
        arr = read_decoder(repo)["model.embed_tokens.weight"].numpy()
        np.save(p, arr)
        print("local", repo, arr.shape, flush=True)
        return arr
    d = api(repo)
    for f in [s["rfilename"] for s in d["siblings"] if s["rfilename"].endswith(".safetensors")]:
        url = f"{HF}/{repo}/resolve/main/{f}"
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        h = json.loads(get(url, 8, 7 + n))
        if key in h:
            s, e = h[key]["data_offsets"]
            base = 8 + n + s
            size = e - s
            step = 8 << 20
            ranges = [(a, min(size, a + step) - 1) for a in range(0, size, step)]
            with ThreadPoolExecutor(8) as ex:
                buf = b"".join(ex.map(lambda r: get(url, base + r[0], base + r[1]), ranges))
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32).reshape(h[key]["shape"])
            np.save(p, arr)
            print("fetched", repo, arr.shape, flush=True)
            return arr
    raise KeyError(key)


def token_class(s):
    if not s.strip():
        return "space"
    cats = {"zh": 0, "latin": 0, "digit": 0, "punct": 0, "other": 0}
    for ch in s:
        if "一" <= ch <= "鿿":
            cats["zh"] += 1
        elif ch.isdigit():
            cats["digit"] += 1
        elif ch.isalpha() and ord(ch) < 0x250:
            cats["latin"] += 1
        elif unicodedata.category(ch)[0] in "PSZ":
            cats["punct"] += 1
        else:
            cats["other"] += 1
    return max(cats, key=cats.get)


def main():
    os.makedirs(CACHE, exist_ok=True)
    tok = tokenizer()
    base = table("Qwen3-0.6B-Base", MODELS["Qwen3-0.6B-Base"])
    V = base.shape[0]
    n_real = 151643
    text = tok.batch_decode([[i] for i in range(n_real)])
    cls = np.array([token_class(t) for t in text])
    id_bands = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 60000), (60000, n_real), (n_real, V)]
    print("rows", V, "class sizes", {c: int((cls == c).sum()) for c in sorted(set(cls))})
    base_n = np.linalg.norm(base, axis=1)
    print("\nBase row norm by id band:", {f"{a}-{b}": round(float(base_n[a:b].mean()), 3) for a, b in id_bands})
    print("\n[1] relative row drift (rms over rows) by token id band (lower ids ≈ more frequent merges); shrink removed")
    print("  " + " " * 18 + " ".join(f"{a//1000}k-{b//1000}k".rjust(10) for a, b in id_bands) + "    unchanged%")
    res = {}
    for m, key in MODELS.items():
        if m == "Qwen3-0.6B-Base":
            continue
        W = table(m, key)
        if W.shape[1] != base.shape[1]:
            print(f"  {m}: width {W.shape[1]} differs from Base {base.shape[1]}, not comparable row by row", flush=True)
            continue
        if W.shape != base.shape:
            print(f"  {m}: shape {W.shape}, comparing shared rows only")
            n = min(V, W.shape[0])
            padded = base.copy()
            padded[:n] = W[:n]
            W = padded
        a = float((W * base).sum() / (base * base).sum())
        R = W - a * base
        r = (R ** 2).sum(1)
        res[m] = (r, W, a)
        bands = [np.sqrt(r[x:y].sum() / (base_n[x:y] ** 2).sum()) for x, y in id_bands]
        same = (W == base).all(1).mean()
        print(f"  {m.replace('Qwen3','').replace('-0.6B','').replace('-12Hz','').strip('-')[:18] or 'instruct':18s}" + " ".join(f"{x:10.3f}" for x in bands) + f"    {same:8.3%}  alpha {a:.3f}")
    print("\n[2] relative row drift by token class")
    classes = ["zh", "latin", "digit", "punct", "space", "other"]
    print("  " + " " * 18 + " ".join(f"{c:>8s}" for c in classes))
    for m, (r, _, _) in res.items():
        vals = [np.sqrt(r[:n_real][cls == c].sum() / (base_n[:n_real][cls == c] ** 2).sum()) for c in classes]
        print(f"  {m.replace('Qwen3','').replace('-0.6B','').replace('-12Hz','').strip('-')[:18] or 'instruct':18s}" + " ".join(f"{x:8.3f}" for x in vals))
    print("\n[3] most-moved tokens per model (relative residual, ordinary tokens with Base row norm above median), specials separately")
    med = np.median(base_n[:n_real])
    for m, (r, _, _) in res.items():
        rel = np.sqrt(r[:n_real]) / (base_n[:n_real] + 1e-9)
        rel[base_n[:n_real] < med] = 0
        top = np.argsort(-rel)[:25]
        sp = np.argsort(-np.sqrt(r[n_real:]) / (base_n[n_real:] + 1e-9))[:6] + n_real
        print(f"  {m}:\n    " + " ".join(repr(text[i])[:14] for i in top))
        print("    specials: " + " ".join(tok.decode([int(i)]) for i in sp))
    print("\n[4] rank correlation between models' row-drift profiles (log residual)")
    names = list(res)
    logs = {m: np.log(res[m][0][:n_real] + 1e-12) for m in names}
    def rankcorr(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    sh = [m.replace("Qwen3", "").replace("-0.6B", "").replace("-12Hz", "").strip("-")[:9] or "instruct" for m in names]
    print("  " + " " * 10 + " ".join(f"{s:>9s}" for s in sh))
    for i, a in enumerate(names):
        print(f"  {sh[i]:10s}" + " ".join(f"{rankcorr(logs[a], logs[b]):9.3f}" for b in names))


if __name__ == "__main__":
    main()
