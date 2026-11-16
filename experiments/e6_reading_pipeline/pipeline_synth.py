# E6: render plan.json (build_plan.py) with Qwen3-TTS-0.6B. Runs on the laptop from qwen-all/e4_tts_structure (.venv-tts), so it
# imports poly_tts.Runner and the stage 4 voices unchanged. Each edit puts a placeholder character into the text whose
# embedding row holds the reading vector (as in poly_vector.py, without exclusions: the pipeline sees no references).
#   python pipeline_synth.py <plan.json> <out dir> [--batch 10]
import argparse, json, os, time
import soundfile as sf
import torch
import poly_tts as P
from shared_structure import tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(4)
    plan = json.load(open(a.plan, encoding="utf-8"))
    jobs = plan["jobs"]
    mono = json.load(open(os.path.join(P.ROOT, "mono_read.json"), encoding="utf-8"))
    seen = set("".join(j["text"] for j in jobs))
    members, spare = {}, []
    for ch, r in sorted(mono.items()):
        ids = tok(ch, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            members.setdefault(r, []).append((ch, ids[0]))
            if ch not in seen:
                spare.append((ch, ids[0]))
    spare = spare[-300:]
    spare_set = {c for c, _ in spare}
    run = P.Runner(P.MODELS["TTS-0.6B"])
    weight = run.t.t.get_text_embeddings().weight
    W0 = weight.data.float().clone()

    def vector(reading):
        m = [i for ch, i in members.get(reading, []) if ch not in spare_set]
        if len(m) < 3:
            return None
        rows = W0[m]
        v = rows.mean(0)
        return v * rows.norm(dim=1).mean() / v.norm()

    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "rendered.jsonl")
    done = {(r["set"], r["id"], r["kind"]) for r in map(json.loads, open(log, encoding="utf-8"))} if os.path.exists(log) else set()
    q = sorted((j for j in jobs if (j["set"], j["id"], j["kind"]) not in done), key=lambda j: len(j["text"]))
    print(f"{len(q)} jobs to render, {len(spare)} placeholders", flush=True)
    t0, audio, count = time.time(), 0.0, 0
    with open(log, "a", encoding="utf-8") as fo:
        while q:
            chunk = [q.pop(0) for _ in range(min(a.batch, len(q)))]
            readings = sorted({r for j in chunk for r in j["edits"].values()})
            slot, pool = {}, iter(spare)
            for r in readings:
                v = vector(r)
                if v is None:
                    continue
                ch, tid = next(pool)
                with torch.no_grad():
                    weight.data[tid] = v.to(weight.dtype)
                slot[r] = ch
            texts, voices, rec = [], [], []
            for j in chunk:
                t = list(j["text"])
                applied = {}
                for pos, r in j["edits"].items():
                    if r in slot:
                        t[int(pos)] = slot[r]
                        applied[pos] = r
                texts.append("".join(t))
                voices.append(j["voice"])
                rec.append(dict(set=j["set"], id=j["id"], kind=j["kind"], text="".join(t), applied=applied))
            torch.manual_seed(chunk[0]["id"] + chunk[0].get("seed_offset", 0))
            wavs, sr, _ = run.synth(texts, voices, False)
            for r, w in zip(rec, wavs):
                d = os.path.join(a.out, r["set"])
                os.makedirs(d, exist_ok=True)
                sf.write(os.path.join(d, f"{r['id']:04d}_{r['kind']}.flac"), w, sr)
                r["seconds"] = len(w) / sr
                audio += r["seconds"]
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()
            with torch.no_grad():
                weight.data.copy_(W0.to(weight.dtype))
            if P.DEVICE == "mps":
                torch.mps.empty_cache()
            count += len(chunk)
            el = time.time() - t0
            print(f"  {count} jobs, {el:.0f}s, audio {audio:.0f}s, rtf {el / max(audio, 1):.2f}", flush=True)
    print("PIPEDONE", flush=True)


if __name__ == "__main__":
    main()
