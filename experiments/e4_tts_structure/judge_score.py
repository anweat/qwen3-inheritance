# Scores TTS and human clips with an independent recognizer, for the criteria in judge_panel.py. Run in .venv-judge.
#   python judge_score.py <charsiu|sensevoice|paraformer|whisper|whisper_cuda> [--limit N] [--sources a,b]
#   (whisper_cuda: laptop GPU, openai-whisper; see WhisperCuda)
# For each clip: lo, the gold-minus-alt log odds, and top, the best readings among all readings.
# As in poly_asr.py, characters-based judges split their distribution by the reading of single-reading characters and exclude
# the polyphone and its substitutes, so the character written in the text cannot decide the score.
import argparse, difflib, json, os, re, time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.expanduser(os.environ.get("QWEN_PRON_ROOT", "~/.cache/qwen-all/pron"))
HAN = re.compile(r"[一-鿿]")
SOURCES = ["tts_out/TTS-0.6B_nonstream", "tts_natural/TTS-0.6B_nonstream"]
mono = json.load(open(os.path.join(ROOT, "mono_read.json"), encoding="utf-8"))
READINGS = sorted(set(mono.values()))


def snapshot(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


def top_list(s, k=5):
    return [[r, round(float(v), 3)] for r, v in sorted(s.items(), key=lambda t: -t[1])[:k]]


def result(s, gold, alt):
    lo = float(s[gold] - s[alt]) if gold in s and alt in s else None
    return dict(lo=lo, top=top_list(s))


def locate(ref, hyp, i):
    """Span of hyp positions covering ref[i]: its match, or the gap between the neighbouring matches."""
    m = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    before, after = -1, len(hyp)
    for a, b, n in m.get_matching_blocks():
        if a <= i < a + n:
            k = b + i - a
            return k, k
        if a + n <= i:
            before = max(before, b + n - 1)
        elif a > i:
            after = min(after, b)
    return before + 1, after - 1


# ---- Charsiu: frame posteriors over tonal phones, no language model. The sentence is aligned with a free slot at the target.
class Charsiu:
    GARBAGE = 1.0

    def __init__(self):
        import torch
        from transformers import Wav2Vec2ForCTC
        self.torch = torch
        self.model = Wav2Vec2ForCTC.from_pretrained(snapshot("charsiu/zh_w2v2_tiny_fc_10ms")).eval()
        self.vocab = json.load(open(os.path.join(snapshot("charsiu/tokenizer_zh_pinyin"), "vocab.json")))
        self.sil = self.vocab["[SIL]"]
        self.cands = {r: p for r in READINGS if (p := self.phones(r))}
        self.names = list(self.cands)
        self.first = np.array([self.cands[r][0] for r in self.names])
        self.last = np.array([self.cands[r][-1] for r in self.names])
        self.two = np.array([len(self.cands[r]) == 2 for r in self.names])
        print(f"charsiu: {len(self.names)}/{len(READINGS)} readings mapped", flush=True)

    def phones(self, reading):
        m = re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*?)(\d)$", reading)
        if not m:
            return None
        ini, fin, tone = m.groups()
        whole = {"ou": "iou", "an": "ian", "ang": "iang", "ong": "iong", "e": "ie", "a": "ia", "ao": "iao", "o": "iou",
                 "u": "v", "ue": "ve", "uan": "van", "un": "vn"}
        if ini == "y":
            ini, fin = "", whole.get(fin, fin)
        elif ini == "w":
            ini, fin = "", dict(u="u", a="ua", o="uo", ai="uai", ei="ui", an="uan", en="un", ang="uang", eng="ong").get(fin, fin)
        elif ini in ("j", "q", "x") and fin.startswith("u"):
            fin = "v" + fin[1:]
        elif ini in ("z", "c", "s") and fin == "i":
            fin = "ii"
        elif ini in ("zh", "ch", "sh", "r") and fin == "i":
            fin = "iii"
        elif ini in ("n", "l") and fin == "ue":
            fin = "ve"
        if fin == "iu":
            fin = "iou"
        out = ([ini] if ini else []) + [fin + tone]
        return [self.vocab[p] for p in out] if all(p in self.vocab for p in out) else None

    def states(self, text):
        """Phone ids in order, with a skippable silence between syllables and a skippable garbage state for non-Han runs."""
        from pypinyin import Style, lazy_pinyin
        seq = [("sil", True)]
        for run in re.findall(r"[一-鿿]+|[^一-鿿]+", text):
            if HAN.match(run):
                for y in lazy_pinyin(run, style=Style.TONE3, neutral_tone_with_five=True, errors=lambda s: ["?"] * len(s)):
                    p = self.phones(y) if y != "?" else None
                    seq += [(q, False) for q in p] if p else [("any", True)]
                    seq.append(("sil", True))
            elif re.search(r"[0-9A-Za-z]", run):
                seq += [("any", True), ("sil", True)]
        return seq

    def sweep(self, F, seq):
        """Best score of covering frames 0..t-1 with seq, for t = 0..T. seq may be fully skipped."""
        T = len(F)
        L = np.full(T + 1, -np.inf)
        if not seq:
            L[0] = 0.
            return L
        garbage = F.max(1) - self.GARBAGE
        E = np.stack([garbage if p == "any" else F[:, self.sil if p == "sil" else p] for p, _ in seq])
        skip = np.array([s for _, s in seq])
        S = len(seq)
        reach = np.full(S, -np.inf)  # best score of having finished everything before state s, at the previous frame
        D = np.full(S, -np.inf)
        start = np.zeros(S, bool)
        for s in range(S):
            start[s] = s == 0 or (start[s - 1] and skip[s - 1])
        tail = np.zeros(S, bool)
        for s in range(S - 1, -1, -1):
            tail[s] = s == S - 1 or (tail[s + 1] and skip[s + 1])
        L[0] = 0. if skip.all() else -np.inf
        for t in range(T):
            prev = D
            reach = np.empty(S)
            reach[0] = -np.inf
            for s in range(1, S):
                reach[s] = max(prev[s - 1], reach[s - 1] if skip[s - 1] else -np.inf)
            best = np.maximum(prev, reach)
            if t == 0:
                best = np.where(start, 0., best)
            D = best + E[:, t]
            L[t + 1] = D[tail].max()
        return L

    def score(self, wav, text, i, gold, alt, spoken):
        torch = self.torch
        x = (wav - wav.mean()) / (wav.std() + 1e-7)
        with torch.no_grad():
            F = torch.log_softmax(self.model(torch.tensor(x)[None]).logits[0], -1).numpy().astype(np.float64)
        left = self.states(text[:i])
        right = self.states(text[i + 1:])
        L = self.sweep(F, left)
        R = self.sweep(F[::-1], right[::-1])[::-1]
        T = len(F)
        e1 = np.full(len(self.names), -np.inf)
        e2 = np.full(len(self.names), -np.inf)
        best = np.full(len(self.names), -np.inf)
        for t in range(T):
            f1, f2 = F[t, self.first], F[t, self.last]
            n2 = f2 + np.maximum(e2, np.where(self.two, e1, -np.inf))
            n1 = f1 + np.maximum(e1, L[t])
            # A one-phone reading lives in e1 and also finishes there.
            end = np.where(self.two, n2, n1)
            best = np.maximum(best, end + R[t + 1])
            e1, e2 = n1, n2
        s = dict(zip(self.names, best))
        return result({k: v for k, v in s.items() if np.isfinite(v)}, gold, alt)


# ---- Shared by the two character-output judges.
class CharClasses:
    def build(self, pieces):
        """pieces: token id -> text. Keeps tokens starting with a single-reading character."""
        self.by = {}
        for t, p in pieces.items():
            if p and p[0] in mono and HAN.match(p[0]):
                self.by.setdefault(mono[p[0]], []).append(t)
        self.first = {t: pieces[t][0] for v in self.by.values() for t in v}
        print(f"{len(self.by)} readings indexed", flush=True)

    def classes(self, spoken):
        return {r: [t for t in v if self.first[t] not in spoken] for r, v in self.by.items()}


class SenseVoice(CharClasses):
    def __init__(self):
        import torch
        from funasr import AutoModel
        from funasr.utils.load_utils import extract_fbank
        self.torch, self.extract = torch, extract_fbank
        am = AutoModel(model=snapshot("FunAudioLLM/SenseVoiceSmall"), device="cpu", disable_update=True)
        self.m, self.frontend, self.tok = am.model.eval(), am.kwargs["frontend"], am.kwargs["tokenizer"]
        self.pieces = {t: self.tok.sp.IdToPiece(t).replace("▁", "") for t in range(self.tok.sp.GetPieceSize())}
        self.build(self.pieces)

    def score(self, wav, text, i, gold, alt, spoken):
        torch, m = self.torch, self.m
        speech, lens = self.extract([torch.tensor(wav)], data_type="sound", frontend=self.frontend)
        with torch.no_grad():
            lang = m.embed(torch.LongTensor([[m.lid_dict["zh"]]]))
            norm = m.embed(torch.LongTensor([[m.textnorm_dict["woitn"]]]))
            ev = m.embed(torch.LongTensor([[1, 2]]))
            sp = torch.cat((lang, ev, norm, speech), 1)
            enc, el = m.encoder(sp, lens + 4)
            if isinstance(enc, tuple):
                enc = enc[0]
            lp = m.ctc.log_softmax(enc)[0, 4:int(el[0])]
        path = lp.argmax(-1).tolist()
        tokens = []
        for t, y in enumerate(path):
            if y != m.blank_id and (t == 0 or path[t - 1] != y):
                tokens.append([y, t, t])
            elif y != m.blank_id:
                tokens[-1][2] = t
        hyp, owner = "", []
        for k, (y, _, _) in enumerate(tokens):
            p = self.pieces.get(y, "")
            hyp += p
            owner += [k] * len(p)
        a, b = locate(text, hyp, i)
        if a <= b and owner:
            ks = [owner[j] for j in range(max(a, 0), min(b + 1, len(owner)))]
            t0, t1 = (tokens[ks[0]][1], tokens[ks[-1]][2]) if ks else (0, len(lp) - 1)
        else:
            t0 = tokens[owner[a - 1]][2] if 0 < a <= len(owner) else 0
            t1 = tokens[owner[a]][1] if 0 <= a < len(owner) else len(lp) - 1
        t0, t1 = max(0, t0 - 1), min(len(lp) - 1, t1 + 1)
        win = lp[t0:t1 + 1]
        s = {}
        for r, ids in self.classes(spoken).items():
            if ids:
                s[r] = float(torch.logsumexp(win[:, ids], 1).max())
        return dict(result(s, gold, alt), hyp=hyp)


class Paraformer(CharClasses):
    def __init__(self):
        import torch
        from funasr import AutoModel
        from funasr.utils.load_utils import extract_fbank
        self.torch, self.extract = torch, extract_fbank
        am = AutoModel(model=snapshot("funasr/paraformer-zh"), device="cpu", disable_update=True)
        self.m, self.frontend, self.tok = am.model.eval(), am.kwargs["frontend"], am.kwargs["tokenizer"]
        self.list = self.tok.token_list
        self.build({t: p for t, p in enumerate(self.list)})

    def score(self, wav, text, i, gold, alt, spoken):
        torch, m = self.torch, self.m
        speech, lens = self.extract([torch.tensor(wav)], data_type="sound", frontend=self.frontend)
        with torch.no_grad():
            enc, el = m.encode(speech, lens)
            if isinstance(enc, tuple):
                enc = enc[0]
            emb, n, _, _ = m.calc_predictor(enc, el)
            n = n.round().long()
            if int(n[0]) < 1:
                return dict(lo=None, top=[], hyp="")
            out, _ = m.cal_decoder_with_predictor(enc, el, emb, n)
        lp = out[0, :int(n[0])]
        ids = lp.argmax(-1).tolist()
        hyp, owner = "", []
        for k, y in enumerate(ids):
            p = self.list[y]
            if p.startswith("<") or p.endswith(">"):
                continue
            p = p.replace("@@", "")
            hyp += p
            owner += [k] * len(p)
        a, b = locate(text, hyp, i)
        if a <= b and owner:
            ks = sorted({owner[j] for j in range(max(a, 0), min(b + 1, len(owner)))}) or [min(len(lp) - 1, owner[min(a, len(owner) - 1)])]
        else:
            ks = sorted({owner[j] for j in (a - 1, a) if 0 <= j < len(owner)}) or [0]
        win = lp[ks]
        s = {}
        for r, c in self.classes(spoken).items():
            if c:
                s[r] = float(torch.logsumexp(win[:, c], 1).max())
        return dict(result(s, gold, alt), hyp=hyp)


class Whisper(CharClasses):
    def __init__(self):
        self.load()
        self.index()

    def load(self):
        import mlx.core as mx
        from mlx_whisper import load_models
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.tokenizer import get_tokenizer
        self.mx, self.mel, self.pad, self.N = mx, log_mel_spectrogram, pad_or_trim, N_SAMPLES
        self.m = load_models.load_model("mlx-community/whisper-large-v3-turbo")
        self.tok = get_tokenizer(True, num_languages=self.m.num_languages, language="zh", task="transcribe")

    def encode(self, wav):
        mx = self.mx
        a = self.pad(mx.array(wav), self.N)
        return self.m.encoder(self.mel(a, n_mels=self.m.dims.n_mels)[None])

    def step(self, batch, enc):
        mx = self.mx
        out = self.m.decoder(mx.array(batch), mx.repeat(enc, len(batch), 0))
        logits = out[0] if isinstance(out, tuple) else out
        lp = logits[:, -1]
        return np.array(lp - mx.logsumexp(lp, axis=-1, keepdims=True), dtype=np.float64)

    def index(self):
        pieces = {}
        for t in range(self.tok.eot):
            p = self.tok.decode([t])
            if p and "�" not in p:
                pieces[t] = p.strip()
        self.build(pieces)
        # Most Han characters take two tokens here, the first being a partial byte sequence. Those need a second decoder step.
        self.pairs = {}
        for c, r in mono.items():
            if HAN.match(c):
                e = self.tok.encode(c)
                if len(e) == 2:
                    self.pairs.setdefault(r, []).append((c, e[0], e[1]))
        self.sot = list(self.tok.sot_sequence_including_notimestamps)
        print(f"whisper: {len(set(self.by) | set(self.pairs))} readings with one- or two-token characters", flush=True)

    TOP_PARTIAL = 48

    def score(self, wav, text, i, gold, alt, spoken):
        enc = self.encode(wav)
        ids = self.sot + self.tok.encode(text[:i])
        step = lambda batch: self.step(batch, enc)
        lp = step([ids])[0]
        pairs = {r: [(t1, t2) for c, t1, t2 in v if c not in spoken] for r, v in self.pairs.items()}
        # Gold and alt classes are always expanded; other readings only through the likeliest partial tokens.
        must = {t1 for r in (gold, alt) for t1, _ in pairs.get(r, [])}
        partial = {t1 for v in pairs.values() for t1, _ in v}
        order = sorted(partial, key=lambda t: -lp[t])[:self.TOP_PARTIAL]
        firsts = sorted(must | set(order))
        second = {}
        for b in range(0, len(firsts), 32):
            part = firsts[b:b + 32]
            for t1, row in zip(part, step([ids + [t1] for t1 in part])):
                second[t1] = row
        s = {}
        single = self.classes(spoken)
        for r in set(single) | set(pairs):
            v = list(lp[single[r]]) if single.get(r) else []
            v += [lp[t1] + second[t1][t2] for t1, t2 in pairs.get(r, []) if t1 in second]
            if v:
                v = np.array(v)
                m = v.max()
                s[r] = float(m + np.log(np.exp(v - m).sum()))
        return result(s, gold, alt)


class WhisperCuda(Whisper):
    """Same judge on the laptop GPU: openai-whisper (the code mlx-whisper was ported from), same checkpoint, float16 like
    mlx-whisper's default. Scored under its own name; agreement with the MLX judge is checked on clips both scored."""

    def load(self):
        import torch, whisper
        from whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from whisper.tokenizer import get_tokenizer
        torch.set_grad_enabled(False)
        self.torch, self.mel, self.pad, self.N = torch, log_mel_spectrogram, pad_or_trim, N_SAMPLES
        self.m = whisper.load_model("large-v3-turbo", device="cuda").half().eval()
        self.tok = get_tokenizer(True, num_languages=self.m.num_languages, language="zh", task="transcribe")

    def encode(self, wav):
        a = self.pad(self.torch.from_numpy(wav), self.N)
        mel = self.mel(a, n_mels=self.m.dims.n_mels).to("cuda", self.torch.float16)
        return self.m.encoder(mel[None])

    def step(self, batch, enc):
        torch = self.torch
        logits = self.m.decoder(torch.tensor(batch, device="cuda"), enc.expand(len(batch), -1, -1))
        return torch.log_softmax(logits[:, -1].double(), dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judge")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sources", default=",".join(SOURCES))
    a = ap.parse_args()
    J = dict(charsiu=Charsiu, sensevoice=SenseVoice, paraformer=Paraformer, whisper=Whisper, whisper_cuda=WhisperCuda)[a.judge]()
    os.makedirs(os.path.join(ROOT, "judges", a.judge), exist_ok=True)
    for source in a.sources.split(","):
        path = os.path.join(ROOT, source + ".jsonl")
        out = os.path.join(ROOT, "judges", a.judge, source.replace("/", "_") + ".jsonl")
        done = set()
        if os.path.exists(out):
            done = {(x["id"], x["clip"]) for x in map(json.loads, open(out, encoding="utf-8"))}
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if a.limit:
            rows = rows[:a.limit]
        t0, n = time.time(), 0
        with open(out, "a", encoding="utf-8") as fo:
            for r in rows:
                texts = r["scores"]["texts"]
                i = next(j for j, (x, y) in enumerate(zip(r["text"], texts["R_g"])) if x != y)
                spoken = {t[i] for t in texts.values()}
                for clip in texts:
                    if (r["id"], clip) in done:
                        continue
                    f = os.path.join(ROOT, source, f"{r['id']:04d}_{clip}.flac")
                    w, sr = sf.read(f, dtype="float32")
                    if w.ndim > 1:
                        w = w.mean(1)
                    w = resample_poly(w, 16000, sr).astype(np.float32) if sr != 16000 else w
                    res = J.score(w, r["text"], i, r["gold"], r["alt"], spoken)
                    fo.write(json.dumps(dict(id=r["id"], clip=clip, **res), ensure_ascii=False) + "\n")
                    fo.flush()
                    n += 1
                    if n % 100 == 0:
                        print(f"  {source} {n} clips, {(time.time() - t0) / n:.2f} s each", flush=True)
        print(f"{source}: {n} clips in {time.time() - t0:.0f} s", flush=True)
    print("JUDGEDONE", flush=True)


if __name__ == "__main__":
    main()
