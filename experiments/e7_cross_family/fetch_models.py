# E7: download only what stages 1-2 need (LM weights, configs, tokenizers) from ModelScope with curl.
# Same script on the Mac and on AutoDL; resumable (curl -C -). Speeds measured from the Mac on 2026-11-11:
# ModelScope 13.5 MB/s, hf-mirror 5.6, huggingface.co 4.7 (30 MB range reads).
#   python fetch_models.py [dest]     # default ~/.cache/qwen-all/e7/models
import fnmatch, json, os, subprocess, sys, time, urllib.request

DST = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.cache/qwen-all/e7/models")
# name: (ModelScope id, include patterns)
MODELS = {
    "qwen25_0.5b_base": ("Qwen/Qwen2.5-0.5B", ["*"]),
    "qwen25_0.5b_inst": ("Qwen/Qwen2.5-0.5B-Instruct", ["*"]),
    "qwen2_0.5b_base": ("Qwen/Qwen2-0.5B", ["*"]),  # round 2: is CosyVoice-BlankEN a Qwen2?
    "qwen2_0.5b_inst": ("Qwen/Qwen2-0.5B-Instruct", ["*"]),
    "cosyvoice2_0.5b": ("iic/CosyVoice2-0.5B", ["llm.pt", "*.yaml", "CosyVoice-BlankEN/*", "*.md", "*.json"]),
    "spark_tts_0.5b": ("SparkAudio/Spark-TTS-0.5B", ["LLM/*", "config.yaml", "*.md"]),
    "llasa_1b": ("HKUSTAudio/Llasa-1B", ["*"]),
    "llama32_1b_base": ("LLM-Research/Llama-3.2-1B", ["*.json", "*.safetensors", "*.md"]),
    "llama32_1b_inst": ("LLM-Research/Llama-3.2-1B-Instruct", ["*.json", "*.safetensors", "*.md"]),
}


def files(mid):
    url = f"https://modelscope.cn/api/v1/models/{mid}/repo/files?Recursive=true"
    d = json.load(urllib.request.urlopen(url, timeout=60))
    return [(f["Path"], f["Size"]) for f in d["Data"]["Files"] if f["Type"] != "tree"]


manifest = []
for name, (mid, pats) in MODELS.items():
    t0, tot = time.time(), 0
    for path, size in files(mid):
        if not any(fnmatch.fnmatch(path, p) for p in pats):
            continue
        out = os.path.join(DST, name, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == size:
            tot += size
            continue
        url = f"https://modelscope.cn/models/{mid}/resolve/master/{path}"
        for attempt in range(5):
            r = subprocess.run(["curl", "-sSfL", "-C", "-", "--retry", "3", "-o", out, url])
            if os.path.exists(out) and os.path.getsize(out) == size:
                break
            print(f"  retry {attempt} {name}/{path} rc {r.returncode}", flush=True)
        else:
            print(f"FAILED {name}/{path}", flush=True)
        tot += size
    dt = time.time() - t0
    line = f"{name:18s} {mid:36s} {tot / 1e9:5.2f} GB {dt:6.0f} s"
    print(line, flush=True)
    manifest.append(line)
open(os.path.join(DST, "manifest.txt"), "w").write("\n".join(manifest) + "\n")
print("FETCHDONE")
