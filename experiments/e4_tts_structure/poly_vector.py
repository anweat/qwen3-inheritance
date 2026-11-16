# Can reading be set through the talker's text embedding without choosing a substitute character?
# A reading vector is the mean text embedding of single-token, single-reading characters of that reading, rescaled to their mean
# norm, excluding the characters used as references in that item. A rare token's embedding row is overwritten with the vector and
# that token replaces the polyphone, so generation code is untouched. If this works, a head needs only an 841-row codebook as its
# port into the talker, covers characters without a good homophone, and carries no substitute's identity.
#
# Criterion, fixed before scoring: on stage 4 items with calibration, audio V_g and V_a are scored by the ASR detector centred on
# the stage 4 references R_g and R_a. The port works if balanced accuracy >= .90 and both directions >= .85. The comparison is
# token substitution (C_g, C_a) on the same items.
# Also reported: share of embedding variance explained by reading, against shuffled readings.
import argparse, json, os, time
import numpy as np
import soundfile as sf
import torch
import poly_tts as P
from shared_structure import tok

KINDS = ("V_g", "V_a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TTS-0.6B")
    ap.add_argument("--n-min", type=int, default=100)
    ap.add_argument("--n-maj", type=int, default=40)
    ap.add_argument("--batch", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(4)
    items, _ = P.build_items(300, 100, 8)
    items = [r for r in items if r["calibrate"]]
    items = [r for r in items if r["kind"] == "min"][:a.n_min] + [r for r in items if r["kind"] == "maj"][:a.n_maj]
    mono = json.load(open(os.path.join(P.ROOT, "mono_read.json"), encoding="utf-8"))
    seen = set("".join(r["text"] for r in items))
    members, spare = {}, []
    for ch, r in sorted(mono.items()):
        ids = tok(ch, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            members.setdefault(r, []).append((ch, ids[0]))
            if ch not in seen:
                spare.append((ch, ids[0]))
    used = {h for r in items for h in r["hg"] + r["ha"]}
    spare = [x for x in spare if x[0] not in used][-a.batch:]
    print(f"{len(items)} items, {len(members)} readings, placeholders {''.join(c for c, _ in spare)}", flush=True)

    run = P.Runner(P.MODELS[a.model])
    weight = run.t.t.get_text_embeddings().weight
    W0 = weight.data.float().clone()

    ids = [i for v in members.values() for _, i in v]
    labels = np.array([k for k, v in enumerate(members.values()) for _ in v])
    X = W0[ids].cpu().numpy()
    def between(y):
        total = ((X - X.mean(0)) ** 2).sum()
        within = sum(((X[y == k] - X[y == k].mean(0)) ** 2).sum() for k in np.unique(y))
        return 1 - within / total
    print(f"embedding variance explained by reading {between(labels):.3f}, shuffled {between(np.random.default_rng(0).permutation(labels)):.3f}", flush=True)

    def vector(reading, exclude):
        m = [i for ch, i in members[reading] if ch not in exclude]
        if len(m) < 3:
            return None
        rows = W0[m]
        v = rows.mean(0)
        return (v * rows.norm(dim=1).mean() / v.norm(), len(m))

    out = os.path.join(P.ROOT, "tts_vector", a.model)
    os.makedirs(out, exist_ok=True)
    path = out + ".jsonl"
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    q = sorted((r for r in items if r["id"] not in done), key=lambda r: len(r["text"]))
    t0, audio, count = time.time(), 0.0, 0
    with open(path, "a", encoding="utf-8") as fo:
        while q:
            chunk = [q.pop(0) for _ in range(min(a.batch // 2, len(q)))]
            texts, voices, owner = [], [], []
            pl = iter(spare)
            for r in chunk:
                ex = set(r["hg"] + r["ha"] + [r["char"]])
                vg, va = vector(r["gold"], ex), vector(r["alt"], ex)
                if vg is None or va is None:
                    fo.write(json.dumps(dict(id=r["id"], skip=True)) + "\n")
                    continue
                for k, (v, _) in zip(KINDS, (vg, va)):
                    ch, tid = next(pl)
                    with torch.no_grad():
                        weight.data[tid] = v.to(weight.dtype)
                    texts.append(P.swap(r, ch))
                    voices.append(r["id"] % len(run.refs))
                    owner.append((r, k, ch))
            if not texts:
                continue
            torch.manual_seed(chunk[0]["id"])
            wavs, sr, _ = run.synth(texts, voices, False)
            for (r, k, ch), w in zip(owner, wavs):
                sf.write(os.path.join(out, f"{r['id']:04d}_{k}.flac"), w, sr)
                audio += len(w) / sr
            for r in chunk:
                got = {k: P.swap(r, ch) for (x, k, ch) in owner if x is r}
                if got:
                    fo.write(json.dumps(dict(id=r["id"], texts=got), ensure_ascii=False) + "\n")
            fo.flush()
            if P.DEVICE == "mps":
                torch.mps.empty_cache()
            count += len(chunk)
            print(f"  {count} items, {time.time()-t0:.0f}s, audio {audio:.0f}s", flush=True)
    print("VECTORDONE", flush=True)


if __name__ == "__main__":
    main()
