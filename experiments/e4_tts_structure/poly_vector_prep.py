# Builds an input for poly_asr.py from poly_vector.py output: the V audio plus the stage 4 references and calibration audio.
#   python poly_vector_prep.py <model>
import json, os, shutil, sys
m = sys.argv[1]
ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
base = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/tts_out/{m}_nonstream.jsonl", encoding="utf-8"))}
d = f"{ROOT}/tts_vector/{m}"
out = []
for x in map(json.loads, open(d + ".jsonl", encoding="utf-8")):
    if x.get("skip"):
        continue
    r = base[x["id"]]
    tx = {k: r["scores"]["texts"][k] for k in ("R_g", "R_a", "C_g", "C_a")}
    tx.update(x["texts"])
    for k in ("R_g", "R_a", "C_g", "C_a"):
        shutil.copy(f"{ROOT}/tts_out/{m}_nonstream/{x['id']:04d}_{k}.flac", f"{d}/{x['id']:04d}_{k}.flac")
    out.append(dict(id=x["id"], group=r["group"], char=r["char"], kind=r["kind"], gold=r["gold"], alt=r["alt"], text=r["text"], scores=dict(texts=tx)))
with open(f"{ROOT}/tts_vector/{m}_scored.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
os.makedirs(f"{ROOT}/tts_vector/{m}_scored", exist_ok=True)
for fn in os.listdir(d):
    shutil.copy(f"{d}/{fn}", f"{ROOT}/tts_vector/{m}_scored/{fn}")
print(len(out))
