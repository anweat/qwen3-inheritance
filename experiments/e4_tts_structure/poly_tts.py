# Does the TTS actually read polyphones correctly in its audio? pron_probe.py only read text rows.
#
# For a CPP test sentence with polyphone c, gold reading g and competing reading a, synthesize:
#   O    the original sentence
#   R_g  c replaced by a single-reading homophone of g
#   R_a  c replaced by a single-reading homophone of a
# A detector then decides whether O is closer to R_g or R_a.
#
# Calibration uses the same sentences and reading pairs, with audio of known reading:
#   C_g  c replaced by a second homophone of g (detector should say g)
#   C_a  c replaced by a second homophone of a (detector should say a)
#
# Detectors: total DTW over log-mel; DTW restricted to the region where R_g and R_a differ; a likelihood ratio, the talker's
# teacher-forced first-codebook CE on O's codes under text R_a minus under text R_g, over all frames (ce_ratio) or its peak
# 7-frame window (ce_peak); and an independent ASR's log odds of the two readings at the target position (poly_asr.py),
# raw and centred on the two references. A pilot showed both DTW variants at chance, which is why the last ones were added.
# Scoring is in poly_summary.py. Reading calibration first does not reveal O scores.
#
# Criteria, fixed before any full-data result:
#   Validity: the primary detector is the one with the highest balanced accuracy on calibration, chosen per model before
#     looking at O scores. It must reach balanced accuracy >= .90 with both directions >= .85. Otherwise the result is
#     reported as not interpretable.
#   Corrected rate: theta = (observed + spec - 1) / (sens + spec - 1), sens = P(says g | reading g), spec = P(says a | reading a).
#   Decision per model, on minority readings, with T = LLM-0.6B probe recall on the covered groups minus 5 points:
#     no room for an adapter if the theta 95% CI lower bound >= T;
#     room if the upper bound < T;
#     otherwise inconclusive.
# Limits: forced choice between two readings, so a third reading or a mumble still counts as one of them. Readings with a
# neutral tone or without a frequent single-reading character cannot be tested. Substitutes change the sentence's meaning,
# which may alter prosody around the target.
import argparse, json, os, random, time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa
from scipy.spatial.distance import cdist
from pron_probe import ROOT, load_data, make_groups
from shared_structure import DEVICE, DTYPE, rp, tok
from stitch import SPEECH, Talker, free
from qwen_tts import Qwen3TTSModel

OUT = os.path.join(ROOT, "tts_out")
MODELS = {"TTS-0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "TTS-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
# Tone changes with context, so a substitute with the citation tone is not a true homophone reference.
SANDHI = {"一", "不"}
MAX_CHARS = 40
MAX_NEW = 400
HOP = 480
KINDS_CAL = ("C_g", "C_a")


def build_items(n_min, n_maj, per_group):
    samples, _, _, _ = load_data()
    samples = [x for x in samples if x["cat"] == "single"]
    groups = make_groups(samples)
    homs = json.load(open(os.path.join(ROOT, "homophones.json"), encoding="utf-8"))
    rng = random.Random(0)
    total = dict(min=0, maj=0)
    pools = dict(min=[], maj=[])
    for gi, g in enumerate(groups):
        c = g["key"][0]
        names = {i: r for r, i in g["cls"].items()}
        counts = np.bincount(g["y_tr"], minlength=len(names))
        others = [i for i in np.argsort(-counts) if i != g["maj"]]
        kept = dict(min=[], maj=[])
        for n, y in zip(g["te"], g["y_te"]):
            kind = "maj" if y == g["maj"] else "min"
            total[kind] += 1
            x = samples[n]
            alt = g["maj"] if kind == "min" else int(others[0])
            rg, ra = names[y], names[alt]
            hg = [h for h in homs.get(rg, []) if h != c]
            ha = [h for h in homs.get(ra, []) if h != c and h not in hg]
            if c in SANDHI or rg.endswith("5") or ra.endswith("5") or not hg or not ha or len(x["text"]) > MAX_CHARS:
                continue
            kept[kind].append(dict(group=gi, char=c, kind=kind, gold=rg, alt=ra, text=x["text"], i=x["i"], line=x["line"],
                                   hg=hg[:2], ha=ha[:2]))
        for kind, rows in kept.items():
            rng.shuffle(rows)
            pools[kind] += rows[:per_group]
    rng.shuffle(pools["min"])
    rng.shuffle(pools["maj"])
    print(f"test samples: minority {total['min']}, majority {total['maj']}; usable after filters and per-group cap: "
          f"minority {len(pools['min'])} in {len({r['group'] for r in pools['min']})} groups, "
          f"majority {len(pools['maj'])} in {len({r['group'] for r in pools['maj']})} groups", flush=True)
    items = pools["min"][:n_min] + pools["maj"][:n_maj]
    for k, r in enumerate(items):
        r["id"] = k
        r["calibrate"] = len(r["hg"]) == 2 and len(r["ha"]) == 2
    return items, groups


def swap(r, h):
    return r["text"][:r["i"]] + h + r["text"][r["i"] + 1:]


def texts_for(r):
    out = dict(O=r["text"], R_g=swap(r, r["hg"][0]), R_a=swap(r, r["ha"][0]))
    if r["calibrate"]:
        out.update(C_g=swap(r, r["hg"][1]), C_a=swap(r, r["ha"][1]))
    return out


def refs(wrapper):
    rows = [json.loads(l) for l in open(os.path.join(SPEECH, "manifest.jsonl"), encoding="utf-8")]
    rows = sorted((r for r in rows if r["set"] == "aishell" and 4 <= r["seconds"] <= 8), key=lambda r: r["path"])[:2]
    print("reference clips:", [r["path"] for r in rows], flush=True)
    return [wrapper.create_voice_clone_prompt(ref_audio=os.path.join(SPEECH, r["path"]), ref_text=r["text"])[0] for r in rows]


def feats(wav, sr):
    m = librosa.feature.melspectrogram(y=wav.astype(np.float32), sr=sr, n_fft=1024, hop_length=HOP, n_mels=64)
    x = np.log(m + 1e-6).T
    return (x - x.mean(0)) / (x.std(0) + 1e-5)


def dtw(a, b):
    C = cdist(a, b)
    D, wp = librosa.sequence.dtw(C=C, backtrack=True)
    wp = wp[::-1]
    return D[-1, -1] / len(wp), wp, C[wp[:, 0], wp[:, 1]]


def locate(fg, fa, half=15):
    """Peak of smoothed frame mismatch along the alignment of the two references."""
    _, wp, cost = dtw(fg, fa)
    smooth = np.convolve(cost, np.ones(2 * half + 1) / (2 * half + 1), mode="same")
    p = wp[int(np.argmax(smooth))]
    return int(p[0]), int(p[1])


def window_cost(path, cost, centre, half=15):
    """Mean alignment cost over path steps whose reference frame falls near centre."""
    m = np.abs(path[:, 1] - centre) <= half
    return float(cost[m].mean()) if m.any() else float(cost.mean())


def acoustic(f, fg, fa, centre):
    dg, pg, cg = dtw(f, fg)
    da, pa, ca = dtw(f, fa)
    return dict(dtw_total=float(da - dg), dtw_window=window_cost(pa, ca, centre[1]) - window_cost(pg, cg, centre[0]))


class Runner:
    def __init__(self, repo):
        self.t = Talker.__new__(Talker)
        self.t.w = Qwen3TTSModel.from_pretrained(rp(repo), device_map=DEVICE, dtype=DTYPE, attn_implementation="sdpa")
        self.t.m = self.t.w.model.eval()
        self.t.t = self.t.m.talker
        self.t.cfg = self.t.m.config
        self.t.tc = self.t.cfg.talker_config
        self.refs = refs(self.t.w)
        self.codes = []
        original = self.t.m.generate

        def wrapped(*a, **k):
            r = original(*a, **k)
            self.codes.extend(c.detach().cpu() for c in r[0])
            return r
        self.t.m.generate = wrapped

    def synth(self, texts, voices, streaming):
        self.codes = []
        wavs, sr = self.t.w.generate_voice_clone(text=texts, language=["chinese"] * len(texts), voice_clone_prompt=[self.refs[v] for v in voices],
                                                 non_streaming_mode=not streaming, max_new_tokens=MAX_NEW)
        return wavs, sr, self.codes

    def ce_frames(self, text, codes):
        """Teacher-forced first-codebook CE per frame, same prompt as Talker.score."""
        t, ids = self.t, tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            emb, _, pad = t.prompt(ids)
            x = torch.cat([emb, t.frames(codes[:-1].to(DEVICE), pad)], 1)
            h = t.t.model(inputs_embeds=x, use_cache=False).last_hidden_state
            lg = t.t.codec_head(h[0, emb.shape[1] - 1:]).float()
            return F.cross_entropy(lg, codes[:, 0].to(DEVICE), reduction="none").cpu().numpy()


def score_item(run, r, wavs, sr, codes):
    fs = {k: feats(wavs[k], sr) for k in wavs}
    centre = locate(fs["R_g"], fs["R_a"])
    out = {}
    tx = texts_for(r)
    out["texts"] = tx
    for k in ("O",) + (KINDS_CAL if r["calibrate"] else ()):
        s = acoustic(fs[k], fs["R_g"], fs["R_a"], centre)
        diff = run.ce_frames(tx["R_a"], codes[k]) - run.ce_frames(tx["R_g"], codes[k])
        s["ce_ratio"] = float(diff.sum())
        s["ce_diff"] = [round(float(v), 3) for v in diff]
        s["frames"] = int(codes[k].shape[0])
        out[k] = s
    out["ref_frames"] = [int(codes["R_g"].shape[0]), int(codes["R_a"].shape[0])]
    return out


def run_model(name, repo, items, batch, streaming):
    tag = f"{name}_{'stream' if streaming else 'nonstream'}"
    path = os.path.join(OUT, tag + ".jsonl")
    wavdir = os.path.join(OUT, tag)
    os.makedirs(wavdir, exist_ok=True)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
    todo = sorted((r for r in items if r["id"] not in done), key=lambda r: len(r["text"]))
    print(f"==== {tag}: {len(todo)} items to do, {len(done)} done ====", flush=True)
    if not todo:
        return
    run = Runner(repo)
    t0, audio, count = time.time(), 0.0, 0
    q = list(todo)
    with open(path, "a", encoding="utf-8") as fo:
        while q:
            chunk, n = [], 0
            while q and (not chunk or n + len(texts_for(q[0])) <= batch):
                chunk.append(q.pop(0))
                n += len(texts_for(chunk[-1]))
            texts, voices, owner = [], [], []
            for r in chunk:
                for k, s in texts_for(r).items():
                    texts.append(s)
                    voices.append(r["id"] % len(run.refs))
                    owner.append((r["id"], k))
            torch.manual_seed(chunk[0]["id"])
            wavs, sr, codes = run.synth(texts, voices, streaming)
            byid = {r["id"]: (dict(), dict()) for r in chunk}
            for (i, k), w, c in zip(owner, wavs, codes):
                byid[i][0][k] = w
                byid[i][1][k] = c
                audio += len(w) / sr
            for r in chunk:
                w, c = byid[r["id"]]
                res = score_item(run, r, w, sr, c)
                fo.write(json.dumps(dict(id=r["id"], group=r["group"], char=r["char"], kind=r["kind"], gold=r["gold"], alt=r["alt"], text=r["text"],
                                         calibrate=r["calibrate"], scores=res), ensure_ascii=False) + "\n")
                for k in w:
                    sf.write(os.path.join(wavdir, f"{r['id']:04d}_{k}.flac"), w[k], sr)
            fo.flush()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            count += len(chunk)
            el = time.time() - t0
            print(f"  {count}/{len(todo)} items, {el:.0f}s, audio {audio:.0f}s, realtime factor {el/max(audio,1):.2f}, "
                  f"GPU {(torch.cuda.max_memory_allocated() if DEVICE == 'cuda' else torch.mps.driver_allocated_memory())/2**30:.1f} GiB", flush=True)
    del run
    free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=300)
    ap.add_argument("--maj", type=int, default=100)
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--models", default="TTS-0.6B,TTS-1.7B")
    ap.add_argument("--stream", action="store_true", help="use the streaming-text mode instead of the non-streaming mode")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mod", type=int, default=1, help="keep only items whose id is a multiple of this")
    a = ap.parse_args()
    global OUT
    if a.smoke:
        OUT = os.path.join(ROOT, "tts_smoke")
    torch.set_num_threads(4)
    items, _ = build_items(a.min, a.maj, a.per_group)
    print(f"{len(items)} items, {sum(r['calibrate'] for r in items)} with calibration", flush=True)
    items = [r for r in items if r["id"] % a.mod == 0]
    for r in items[:4]:
        print("  ", r["char"], r["gold"], r["alt"], r["hg"], r["ha"], r["text"], flush=True)
    for name in a.models.split(","):
        run_model(name, MODELS[name], items, a.batch, a.stream)


if __name__ == "__main__":
    main()
    print("POLYDONE", flush=True)
