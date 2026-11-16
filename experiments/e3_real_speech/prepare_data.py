# Build a held-out real-speech test set in a local cache. Audio is used for evaluation only and stays out of the project.
# Tar members are read as bytes and never extracted to disk by their own paths.
import hashlib, io, json, os, random, tarfile, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

OUT = os.path.expanduser("~/.cache/qwen-all/speech")
HF = "https://huggingface.co/datasets"
SR = 16000
PER_SET = 50
DIALECTS = {
    "sichuan": ("MagicDataTech/magicdata-dialect-sichuanese-tts-lite", "Sichuan Dialect Speech Corpus for TTS"),
    "northeast": ("MagicDataTech/magicdata-dialect-northeastern-chinese-tts-lite", "Dongbei Dialect Speech Corpus for TTS"),
    "wu": ("MagicHub/magicdata-dialect-wu-chinese-tts-lite", "Jiangsu Dialect Speech Corpus for TTS"),
    "henan": ("MagicDataTech/magicdata-dialect-henan-dialect-tts-lite", "Henan Dialect Speech Corpus for TTS"),
    "cantonese_md": ("MagicDataTech/magicdata-dialect-cantonese-tts-lite", "Guangzhou Cantonese Speech Corpus for TTS"),
}
# unit: char for scripts without word spacing, word otherwise.
FLEURS = {"cmn_hans_cn": "char", "yue_hant_hk": "char", "ko_kr": "word", "ja_jp": "char", "vi_vn": "word", "th_th": "char", "ar_eg": "word", "ru_ru": "word", "en_us": "word"}


def fetch(url, a=None, b=None, tries=8):
    last = None
    for _ in range(tries):
        try:
            h = {"Range": f"bytes={a}-{b}"} if a is not None else {}
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90) as r:
                return r.read()
        except Exception as e:
            last = e
    raise last


def save(set_name, key, audio, sr, text, unit):
    d = os.path.join(OUT, set_name)
    os.makedirs(d, exist_ok=True)
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    name = hashlib.md5(key.encode()).hexdigest()[:16] + ".wav"
    sf.write(os.path.join(d, name), audio, SR, subtype="PCM_16")
    return dict(set=set_name, path=os.path.join(set_name, name), text=text, unit=unit, seconds=round(len(audio) / SR, 2))


def dialects():
    out = []
    for name, (repo, root) in DIALECTS.items():
        base = f"{HF}/{repo}/resolve/main/" + urllib.parse.quote(root)
        rows = [l.split("\t", 1) for l in fetch(base + "/ProsodyLabeling/txt.txt").decode("utf-8-sig").splitlines() if "\t" in l]
        rows.sort(key=lambda r: hashlib.md5(r[0].encode()).hexdigest())
        rows = rows[:PER_SET]

        def one(r):
            try:
                a, sr = sf.read(io.BytesIO(fetch(f"{base}/wav/{urllib.parse.quote(r[0].strip())}")))
                return save(name, r[0], a, sr, r[1].strip(), "char")
            except Exception as e:
                print("skip", name, r[0], e, flush=True)
        with ThreadPoolExecutor(6) as ex:
            got = [x for x in ex.map(one, rows) if x]
        print(name, len(got), flush=True)
        out += got
    return out


def aishell():
    url = f"{HF}/AudioLLMs/aishell_1_zh_test/resolve/main/data/test-00000-of-00003.parquet"
    path = os.path.join(OUT, "aishell_shard0.parquet")
    if not os.path.exists(path):
        size = 364047728
        step = 8 << 20
        ranges = [(a, min(size, a + step) - 1) for a in range(0, size, step)]
        with ThreadPoolExecutor(8) as ex:
            data = b"".join(ex.map(lambda r: fetch(url, r[0], r[1]), ranges))
        with open(path, "wb") as f:
            f.write(data)
    rows = pq.read_table(path).to_pylist()
    random.Random(0).shuffle(rows)
    out = []
    for i, r in enumerate(rows[:100]):
        a, sr = sf.read(io.BytesIO(r["context"]["bytes"]))
        out.append(save("aishell", f"aishell{i}", a, sr, r["answer"].replace(" ", ""), "char"))
    os.remove(path)
    print("aishell", len(out), "columns sample", rows[0]["instruction"][:40], "|", rows[0]["answer"][:30], flush=True)
    return out


def fleurs():
    out = []
    for lang, unit in FLEURS.items():
        base = f"{HF}/google/fleurs/resolve/main/data/{lang}"
        text = {}
        for line in fetch(base + "/test.tsv").decode("utf-8").splitlines():
            p = line.split("\t")
            if len(p) >= 3:
                text[p[1]] = p[2].strip().strip('"').replace('""', '"')
        got = []
        for attempt in range(4):
            try:
                with urllib.request.urlopen(base + "/audio/test.tar.gz", timeout=120) as resp:
                    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                        seen = set(g["key"] for g in got)
                        for m in tar:
                            if not m.isfile() or len(got) >= PER_SET:
                                if len(got) >= PER_SET:
                                    break
                                continue
                            key = os.path.basename(m.name)
                            if key not in text or key in seen:
                                continue
                            buf = tar.extractfile(m).read()
                            a, sr = sf.read(io.BytesIO(buf))
                            e = save("fleurs_" + lang, key, a, sr, text[key], unit)
                            e["key"] = key
                            got.append(e)
                break
            except Exception as e:
                print("retry", lang, len(got), e, flush=True)
        print("fleurs", lang, len(got), flush=True)
        out += got
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = dialects() + aishell() + fleurs()
    with open(os.path.join(OUT, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("TOTAL", len(rows), "clips,", round(sum(r["seconds"] for r in rows) / 60, 1), "minutes", flush=True)
