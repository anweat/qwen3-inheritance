# Reading table for poly_tts.py. Runs on the Mac:
#   uv run --no-project --with pypinyin python prepare_poly.py
# Writes ~/.cache/qwen-all/pron/homophones.json: for each CPP reading, single-reading characters most frequent in CPP train.
import json, os, re, collections
from pypinyin import Style, pinyin

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
HAN = re.compile(r"[一-鿿]")
# Substitutes must be common enough for the TTS to know them; calibration checks whether they are read as listed.
MIN_COUNT = 10


def main():
    freq = collections.Counter()
    for line in open(os.path.join(ROOT, "cpp", "train.sent"), encoding="utf-8"):
        freq.update(HAN.findall(line))
    table = collections.defaultdict(list)
    mono = {}
    for code in range(0x4E00, 0xA000):
        ch = chr(code)
        reads = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
        if len(reads) != 1 or not reads[0][-1:].isdigit():
            continue
        # CPP writes ü as "u:", pypinyin as "v".
        mono[ch] = reads[0].replace("v", "u:")
        if freq[ch] >= MIN_COUNT:
            table[mono[ch]].append(ch)
    table = {r: sorted(c, key=lambda x: -freq[x])[:6] for r, c in table.items()}
    json.dump(table, open(os.path.join(ROOT, "homophones.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(mono, open(os.path.join(ROOT, "mono_read.json"), "w", encoding="utf-8"), ensure_ascii=False)
    labels = set()
    for split in ("train", "dev", "test"):
        labels |= set(open(os.path.join(ROOT, "cpp", f"{split}.lb"), encoding="utf-8").read().split())
    two = {r for r in labels if len(table.get(r, [])) >= 2}
    print(f"{len(table)} readings have a frequent single-reading character; CPP uses {len(labels)} readings, {len(two)} of them have two or more")
    print("missing:", sorted(labels - two)[:60])
    print("sample:", {r: table[r] for r in ("hang2", "xing2", "le5", "lu:4", "yue4") if r in table})


if __name__ == "__main__":
    main()
