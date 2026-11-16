# Pronunciation labels for pron_probe.py. Runs on the Mac:
#   uv run --no-project --with pypinyin python prepare_pron.py
# Writes ~/.cache/qwen-all/pron/labels.json with single-reading characters and a dictionary baseline on CPP test.
import json, os, re
from pypinyin import Style, lazy_pinyin, pinyin

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
HAN = re.compile(r"[一-鿿]+")


def mono():
    out = {}
    for code in range(0x4E00, 0xA000):
        ch = chr(code)
        reads = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
        if len(reads) != 1 or not reads[0][-1:].isdigit():
            continue
        ini = lazy_pinyin(ch, style=Style.INITIALS, strict=True)[0]
        fin = lazy_pinyin(ch, style=Style.FINALS, strict=True)[0]
        out[ch] = [ini, fin, reads[0][-1]]
    return out


def dict_predict(line):
    """pypinyin's phrase dictionary on the Han run holding the target character."""
    a, c, b = line.split("▁")
    plain, i = a + c + b, len(a)
    for m in HAN.finditer(plain):
        if m.start() <= i < m.end():
            reads = lazy_pinyin(m.group(), style=Style.TONE3, neutral_tone_with_five=True)
            # CPP writes ü as "u:", pypinyin as "v".
            return reads[i - m.start()].replace("v", "u:") if len(reads) == m.end() - m.start() else None
    return None


def main():
    lines = open(os.path.join(ROOT, "cpp", "test.sent"), encoding="utf-8").read().splitlines()
    labels = open(os.path.join(ROOT, "cpp", "test.lb"), encoding="utf-8").read().splitlines()
    pred = [dict_predict(l) if l.count("▁") == 2 else None for l in lines]
    m = mono()
    json.dump(dict(mono=m, dict_test=pred), open(os.path.join(ROOT, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False)
    hit = sum(p == g for p, g in zip(pred, labels))
    print(f"{len(m)} single-reading characters; dictionary on CPP test: {hit}/{len(labels)} = {hit/len(labels):.4f}")
    print("sample labels", labels[:8], "predictions", pred[:8])


if __name__ == "__main__":
    main()
