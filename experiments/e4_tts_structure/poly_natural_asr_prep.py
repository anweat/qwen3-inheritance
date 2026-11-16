# Adds the human clip as kind H to a poly_natural.py output, so poly_asr.py scores it like any other audio.
#   python poly_natural_asr_prep.py <jsonl>
import json, os, sys
import soundfile as sf
SPEECH = os.path.expanduser("~/.cache/qwen-all/speech")
gold = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_gold.json"), encoding="utf-8"))
path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
for r in rows:
    w, sr = sf.read(os.path.join(SPEECH, gold[r["id"]]["path"]), dtype="float32")
    assert r["text"][gold[r["id"]]["i"]] == r["char"]
    sf.write(os.path.join(path[:-len(".jsonl")], f"{r['id']:04d}_H.flac"), w, sr)
    r["scores"]["texts"]["H"] = r["text"]
with open(path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows))
