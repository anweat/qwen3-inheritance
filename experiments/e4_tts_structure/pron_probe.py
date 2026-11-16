# Does the LLM hold what a TTS needs: pronunciation, and context that resolves polyphones?
#
# Probe A, character sound: ridge from hidden state to pinyin initial, final and tone, tested on characters never seen
# in training, so a probe can succeed only if the representation is organized by sound.
# Probe B, polyphones (CPP, human labels): per-character ridge to the reading in context. The main number is recall on
# readings other than the character's training majority, on single-character tokens, whose embedding is identical for
# all readings. Features: target state (left context only, these models are causal) and target + state two tokens later
# (reaches two tokens of right context). Layer and regularization are picked on dev accuracy, because recall alone can
# be raised by predicting minority readings more often. Revised after the smoke run, before any full-data result.
#
# Criteria fixed before running:
# A. "LLM is not organized by sound" holds if LLM embedding accuracy is within 5 points of the random network on
#    initial, final and tone, while TTS embedding is at least 20 points above random on at least two of them.
# B. "Worth building the adapter" needs the LLM, at its dev-selected layer, to beat every TTS text path and the
#    pypinyin dictionary on minority recall by at least 5 points, with the paired 95% CI excluding zero, and without
#    a significant loss of overall accuracy (accuracy CI upper bound above zero).
# Limit: this reads text rows only. A TTS may also resolve readings later, in its audio rows.
import argparse, gc, json, os, random
import numpy as np
import torch
from transformers import AutoConfig, Qwen3ForCausalLM
from stitch import DEVICE, DTYPE, Talker, free, rp, tok, ROLE

ROOT = os.path.expanduser("~/.cache/qwen-all/pron")
OUT = os.path.join(ROOT, "out")
LAYERS = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 28]
CAPS = {"train": 100, "dev": 20, "test": 100}
MODELS = [
    ("LLM-0.6B", "llm", "Qwen/Qwen3-0.6B-Base"),
    ("Random-0.6B", "random", "Qwen/Qwen3-0.6B-Base"),
    ("TTS-0.6B", "tts", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
    ("LLM-1.7B", "llm", "Qwen/Qwen3-1.7B-Base"),
    ("TTS-1.7B", "tts", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
]
LLMS = ["LLM-0.6B", "LLM-1.7B"]
TTSS = ["TTS-0.6B", "TTS-1.7B"]
LAMBDAS = (1e-3, 1e-2, 1e-1, 1)
BATCH = 16


def load_data(max_chars=None):
    labels = json.load(open(os.path.join(ROOT, "labels.json"), encoding="utf-8"))
    mono = labels["mono"]
    rng = random.Random(0)
    samples, seen = [], {}
    for split in ("train", "dev", "test"):
        sents = open(os.path.join(ROOT, "cpp", f"{split}.sent"), encoding="utf-8").read().splitlines()
        golds = open(os.path.join(ROOT, "cpp", f"{split}.lb"), encoding="utf-8").read().splitlines()
        assert len(sents) == len(golds)
        rows = [(i, s, g) for i, (s, g) in enumerate(zip(sents, golds)) if s.count("▁") == 2]
        rng.shuffle(rows)
        for line, s, g in rows:
            a, c, b = s.split("▁")
            if len(c) != 1:
                continue
            key = (split, c)
            if seen.get(key, 0) >= CAPS[split]:
                continue
            seen[key] = seen.get(key, 0) + 1
            samples.append(dict(split=split, line=line, text=a + c + b, char=c, gold=g, i=len(a)))
    if max_chars:
        keep = set(sorted({x["char"] for x in samples})[:max_chars])
        samples = [x for x in samples if x["char"] in keep]
    kept = []
    for x in samples:
        e = tok(x["text"], add_special_tokens=False, return_offsets_mapping=True)
        hit = [j for j, (s, t) in enumerate(e["offset_mapping"]) if s <= x["i"] < t]
        if len(hit) != 1 or len(e["input_ids"]) > 160:
            continue
        s, t = e["offset_mapping"][hit[0]]
        x.update(ids=e["input_ids"], pos=hit[0], cat="single" if t - s == 1 else "multi",
                 offsets=e["offset_mapping"])
        kept.append(x)
    # Probe A positions: single-character tokens of single-reading characters, at most 10 per character.
    count, static = {}, []
    order = list(range(len(kept)))
    rng.shuffle(order)
    for n in order:
        x = kept[n]
        cands = [j for j, (s, t) in enumerate(x["offsets"]) if t - s == 1 and x["text"][s] in mono and j != x["pos"]]
        rng.shuffle(cands)
        taken = 0
        for j in cands:
            ch = x["text"][x["offsets"][j][0]]
            if count.get(ch, 0) >= 10 or taken >= 2:
                continue
            count[ch] = count.get(ch, 0) + 1
            taken += 1
            static.append((n, j, ch))
    for x in kept:
        del x["offsets"]
    return kept, static, mono, labels["dict_test"]


class Llm:
    def __init__(self, repo, random_init):
        if random_init:
            cfg = AutoConfig.from_pretrained(rp(repo))
            cfg._attn_implementation = "eager"
            torch.manual_seed(0)
            self.m = Qwen3ForCausalLM(cfg).to(DTYPE).to(DEVICE).eval()
        else:
            self.m = Qwen3ForCausalLM.from_pretrained(rp(repo), dtype=DTYPE, attn_implementation="eager").to(DEVICE).eval()
        self.start = len(ROLE)
        self.d = self.m.config.hidden_size

    def run(self, ids):
        x = torch.cat([torch.tensor([ROLE], device=DEVICE).expand(len(ids), -1), ids], 1)
        return self.m.model(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states


class Tts:
    def __init__(self, repo):
        self.T = Talker(repo)
        self.start = len(ROLE) + 4
        self.d = self.T.t.config.hidden_size if hasattr(self.T.t.config, "hidden_size") else self.T.tc.hidden_size

    def run(self, ids):
        T = self.T
        c, t = T.cfg, T.tc
        pad, bos = T.tp([c.tts_pad_token_id]), T.tp([c.tts_bos_token_id])
        pre = torch.cat([T.tp(ROLE), torch.cat([pad, pad, pad, bos], 1) + T.cd([t.codec_nothink_id, t.codec_think_bos_id, t.codec_think_eos_id, t.codec_pad_id])], 1)
        codec_pad = T.t.get_input_embeddings()(torch.full_like(ids, t.codec_pad_id))
        body = T.t.text_projection(T.t.get_text_embeddings()(ids)) + codec_pad
        emb = torch.cat([pre.expand(len(ids), -1, -1), body], 1)
        return T.t.model(inputs_embeds=emb, output_hidden_states=True, use_cache=False).hidden_states


def build(kind, repo):
    return Tts(repo) if kind == "tts" else Llm(repo, kind == "random")


@torch.no_grad()
def capture(model, samples, static):
    """Returns poly[n, layer, {t, t+2}, d] and stat[m, layer, d] as float16."""
    poly = np.zeros((len(samples), len(LAYERS), 2, model.d), dtype=np.float16)
    stat = np.zeros((len(static), len(LAYERS), model.d), dtype=np.float16)
    by_sent = {}
    for k, (n, j, _) in enumerate(static):
        by_sent.setdefault(n, []).append((k, j))
    order = sorted(range(len(samples)), key=lambda n: len(samples[n]["ids"]))
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        L = max(len(samples[n]["ids"]) for n in idx)
        # Right padding is safe: the text rows are causal, so they never see the padding.
        ids = torch.tensor([samples[n]["ids"] + [0] * (L - len(samples[n]["ids"])) for n in idx], device=DEVICE)
        hs = model.run(ids)
        for li, lay in enumerate(LAYERS):
            h = hs[lay][:, model.start:].float().half().cpu().numpy()
            for row, n in enumerate(idx):
                x = samples[n]
                poly[n, li, 0] = h[row, x["pos"]]
                if x["pos"] + 2 < len(x["ids"]):
                    poly[n, li, 1] = h[row, x["pos"] + 2]
                for k, j in by_sent.get(n, []):
                    stat[k, li] = h[row, j]
        if (b // BATCH) % 500 == 0:
            print(f"    {b}/{len(order)}", flush=True)
    return poly, stat


@torch.no_grad()
def check_padding(model, samples):
    """Batched, right-padded states must match a sequence run alone."""
    a, b = samples[0]["ids"], samples[-1]["ids"]
    L = max(len(a), len(b))
    both = model.run(torch.tensor([a + [0] * (L - len(a)), b + [0] * (L - len(b))], device=DEVICE))[6]
    one = model.run(torch.tensor([a], device=DEVICE))[6]
    n = model.start + len(a)
    err = ((both[0, :n] - one[0, :n]).float().norm() / one[0, :n].float().norm()).item()
    assert err < .02, f"padding changes text states: relative error {err:.4f}"
    print(f"  padding check: relative error {err:.5f}", flush=True)


def fit_scores(X_tr, Y, X_ev):
    """Dual ridge over the lambda grid on globally scaled features. Lambda is absolute, so a character whose features
    barely vary (the embedding of a single token, up to float noise) falls back to its class prior instead of fitting noise.
    Returns predicted class index per lambda for each evaluation matrix."""
    mu = X_tr.mean(0)
    A = X_tr - mu
    K = A @ A.T
    S, U = torch.linalg.eigh(K)
    S = S.clamp_min(0)
    md = A.shape[1]
    ym = Y.mean(0)
    uy = U.T @ (Y - ym)
    out = {l: [] for l in LAMBDAS}
    Ks = [(X - mu) @ A.T for X in X_ev]
    for lam in LAMBDAS:
        a = U @ (uy / (S + lam * md + 1e-9)[:, None])
        for KX in Ks:
            out[lam].append((KX @ a + ym).argmax(1))
    return out


def make_groups(samples):
    grouped = {}
    for n, x in enumerate(samples):
        grouped.setdefault((x["char"], x["cat"]), []).append(n)
    groups = []
    for key, members in grouped.items():
        tr = [n for n in members if samples[n]["split"] == "train"]
        cnt = {}
        for n in tr:
            cnt[samples[n]["gold"]] = cnt.get(samples[n]["gold"], 0) + 1
        classes = sorted(g for g, c in cnt.items() if c >= 3)
        if len(classes) < 2:
            continue
        tr = [n for n in tr if samples[n]["gold"] in classes]
        dv = [n for n in members if samples[n]["split"] == "dev" and samples[n]["gold"] in classes]
        te = [n for n in members if samples[n]["split"] == "test" and samples[n]["gold"] in classes]
        if not dv or not te:
            continue
        cls = {g: i for i, g in enumerate(classes)}
        maj = cls[max(classes, key=lambda g: cnt[g])]
        groups.append(dict(key=key, cat=key[1], tr=tr, dv=dv, te=te, cls=cls, maj=maj,
                           y_tr=[cls[samples[n]["gold"]] for n in tr],
                           y_dv=[cls[samples[n]["gold"]] for n in dv],
                           y_te=[cls[samples[n]["gold"]] for n in te]))
    return groups


def tally(pred, gold, maj):
    """[correct, n, correct on minority, n minority]"""
    p, g = np.asarray(pred), np.asarray(gold)
    m = g != maj
    return [int((p == g).sum()), len(g), int((p[m] == g[m]).sum()), int(m.sum())]


def summary(rows):
    a = np.array(rows, dtype=float).sum(0)
    return a[0] / a[1], a[2] / max(1, a[3])


def probe_b_features(get, groups):
    """get(indices) returns a feature tensor. Returns per-lambda tallies on dev and test per group."""
    res = {l: dict(dv=[], te=[]) for l in LAMBDAS}
    for g in groups:
        Y = torch.nn.functional.one_hot(torch.tensor(g["y_tr"]), len(g["cls"])).double()
        pr = fit_scores(get(g["tr"]), Y, [get(g["dv"]), get(g["te"])])
        for l in LAMBDAS:
            res[l]["dv"].append(tally(pr[l][0].numpy(), g["y_dv"], g["maj"]))
            res[l]["te"].append(tally(pr[l][1].numpy(), g["y_te"], g["maj"]))
    return res


def probe_b(name, poly, groups, table, samples):
    for feat, sel in (("t", [0]), ("t+2", [0, 1])):
        best = None
        curve = []
        for li, lay in enumerate(LAYERS):
            arr = poly[:, li][:, sel]
            scale = max(float(arr[::7].astype(np.float64).std()), 1e-9)
            def get(ix):
                return torch.from_numpy(arr[ix].astype(np.float64)).reshape(len(ix), -1) / scale
            res = probe_b_features(get, groups)
            lam = max(LAMBDAS, key=lambda l: summary(res[l]["dv"])[0])
            dvs, tes = summary(res[lam]["dv"]), summary(res[lam]["te"])
            curve.append(f"L{lay}:{tes[0]:.3f}/{tes[1]:.3f}")
            if best is None or dvs[0] > best["dev_acc"]:
                best = dict(layer=lay, lam=lam, dev_acc=dvs[0], acc=tes[0], recall=tes[1], rows=res[lam]["te"])
        table[(name, feat)] = best
        print(f"  B {name:12s} {feat:4s} dev-selected L{best['layer']} acc {best['acc']:.4f} minority recall {best['recall']:.4f}", flush=True)
        print("     test accuracy/minority recall by layer: " + " ".join(curve), flush=True)


def probe_a(name, stat, static, mono, results):
    chars = sorted({c for _, _, c in static})
    random.Random(1).shuffle(chars)
    n = len(chars)
    te_chars, val_chars = set(chars[:n // 5]), set(chars[n // 5:n // 5 + n // 10])
    tags = np.array([0 if c in te_chars else 1 if c in val_chars else 2 for _, _, c in static])
    tr, val, te = np.where(tags == 2)[0], np.where(tags == 1)[0], np.where(tags == 0)[0]
    char_te = [static[i][2] for i in te]
    for li, lay in enumerate(LAYERS):
        Xn = stat[:, li].astype(np.float64)
        mu = Xn[tr].mean(0)
        s = max((Xn[tr] - mu).std(), 1e-9)
        Xs = torch.from_numpy((Xn - mu) / s).to(DEVICE)
        cov = Xs[tr].T @ Xs[tr]
        S, U = torch.linalg.eigh(cov)
        S = S.clamp_min(0)
        md = S.sum() / len(S)
        row = []
        for ti, tname in enumerate(("initial", "final", "tone")):
            lab = [mono[c][ti] for _, _, c in static]
            cls = {g: i for i, g in enumerate(sorted(set(lab)))}
            yi = np.array([cls[g] for g in lab])
            Y = torch.nn.functional.one_hot(torch.from_numpy(yi[tr]), len(cls)).double().to(DEVICE)
            ym = Y.mean(0)
            c = U.T @ (Xs[tr].T @ (Y - ym))
            best = (-1, None)
            for lam in (1e-3, 1e-2, 1e-1, 1, 10):
                W = U @ (c / (S + lam * md + 1e-9)[:, None])
                v = ((Xs[val] @ W + ym).argmax(1).cpu().numpy() == yi[val]).mean()
                if v > best[0]:
                    best = (v, W)
            hit = ((Xs[te] @ best[1] + ym).argmax(1).cpu().numpy() == yi[te])
            per = {}
            for c_, h in zip(char_te, hit):
                per.setdefault(c_, []).append(h)
            means = np.array([np.mean(v) for v in per.values()])
            rng = np.random.default_rng(0)
            lo, hi = np.percentile([means[rng.integers(0, len(means), len(means))].mean() for _ in range(2000)], [2.5, 97.5])
            maj_te = (yi[te] == np.bincount(yi[tr]).argmax()).mean()
            results[(name, lay, tname)] = float(means.mean())
            row.append(f"{tname} {means.mean():.3f} [{lo:.3f},{hi:.3f}] (maj {maj_te:.3f})")
        print(f"  A {name:12s} L{lay:<2d} " + "  ".join(row), flush=True)
        del Xs, cov, U, S
        free()


def boot_diff(a_rows, b_rows, n=4000):
    """95% CIs of the paired difference, resampling characters: accuracy, then minority recall."""
    a, b = np.array(a_rows, float), np.array(b_rows, float)
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        ds.append([a[i, 0].sum() / max(1, a[i, 1].sum()) - b[i, 0].sum() / max(1, b[i, 1].sum()),
                   a[i, 2].sum() / max(1, a[i, 3].sum()) - b[i, 2].sum() / max(1, b[i, 3].sum())])
    return np.percentile(np.array(ds), [2.5, 97.5], axis=0)


def main(smoke):
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    samples, static, mono, dict_test = load_data(40 if smoke else None)
    all_groups = make_groups(samples)
    print(f"{len(samples)} polyphone samples, {len(static)} static positions over {len({c for _,_,c in static})} characters", flush=True)
    for cat in ("single", "multi"):
        gs = [g for g in all_groups if g["cat"] == cat]
        print(f"  {cat}: {len(gs)} groups, test samples {sum(len(g['te']) for g in gs)}", flush=True)

    # Baselines needing no hidden states
    base = {}
    for cat in ("single", "multi"):
        gs = [g for g in all_groups if g["cat"] == cat]
        rows_m, rows_d = [], []
        for g in gs:
            rows_m.append(tally([g["maj"]] * len(g["y_te"]), g["y_te"], g["maj"]))
            pred = [g["cls"].get(dict_test[samples[n]["line"]], -1) for n in g["te"]]
            rows_d.append(tally(pred, g["y_te"], g["maj"]))
        base[cat] = dict(majority=rows_m, dictionary=rows_d)
        print(f"  baseline {cat}: majority acc {summary(rows_m)[0]:.4f}; dictionary acc {summary(rows_d)[0]:.4f} minority recall {summary(rows_d)[1]:.4f}", flush=True)

    models = MODELS[:3] if smoke else MODELS
    tables = {cat: {} for cat in ("single", "multi")}
    a_res = {}
    for name, kind, repo in models:
        print(f"\n==== {name} ====", flush=True)
        model = build(kind, repo)
        check_padding(model, samples)
        poly, stat = capture(model, samples, static)
        del model
        free()
        probe_a(name, stat, static, mono, a_res)
        for cat in ("single", "multi"):
            print(f" polyphones, {cat}-character tokens:", flush=True)
            probe_b(name, poly, [g for g in all_groups if g["cat"] == cat], tables[cat], samples)
        del poly, stat
        gc.collect()

    # Neighbour-token baseline: LLM embeddings of t-2..t+2, no processing.
    if not smoke:
        m = Qwen3ForCausalLM.from_pretrained(rp("Qwen/Qwen3-0.6B-Base"), dtype=DTYPE).to(DEVICE).eval()
        emb = m.get_input_embeddings().weight.detach().float().cpu().numpy()
        emb = emb / emb.std()
        del m
        free()
        win = np.zeros((len(samples), 5), dtype=np.int64)
        for n, x in enumerate(samples):
            for o in range(5):
                j = x["pos"] + o - 2
                win[n, o] = x["ids"][j] if 0 <= j < len(x["ids"]) else 0
        for cat in ("single", "multi"):
            res = probe_b_features(lambda ix: torch.from_numpy(emb[win[ix]].astype(np.float64)).reshape(len(ix), -1), [g for g in all_groups if g["cat"] == cat])
            lam = max(LAMBDAS, key=lambda l: summary(res[l]["dv"])[0])
            tes = summary(res[lam]["te"])
            tables[cat][("Neighbours", "t+2")] = dict(layer=0, lam=lam, acc=tes[0], recall=tes[1], rows=res[lam]["te"])
            print(f"  neighbour-token baseline {cat}: acc {tes[0]:.4f} minority recall {tes[1]:.4f}", flush=True)

    print("\nCriterion B on single-character tokens, feature t+2, LLM minus competitor:", flush=True)
    t = tables["single"]
    for llm in [x for x in LLMS if (x, "t+2") in t]:
        a = t[(llm, "t+2")]
        print(f" {llm}: L{a['layer']} accuracy {a['acc']:.4f} recall {a['recall']:.4f}", flush=True)
        comp = {tts: t[(tts, "t+2")]["rows"] for tts in TTSS if (tts, "t+2") in t}
        comp["dictionary"] = base["single"]["dictionary"]
        if ("Neighbours", "t+2") in t:
            comp["neighbours"] = t[("Neighbours", "t+2")]["rows"]
        for label, rows in comp.items():
            lo, hi = boot_diff(a["rows"], rows)
            ac, rc = summary(rows)
            print(f"   vs {label:12s} accuracy {ac:.4f} diff {a['acc']-ac:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]; recall {rc:.4f} diff {a['recall']-rc:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]", flush=True)
    json.dump(dict(a={"|".join(map(str, k)): v for k, v in a_res.items()},
                   b={cat: {"|".join(k): v for k, v in tb.items()} for cat, tb in tables.items()},
                   base=base), open(os.path.join(OUT, "pron_probe.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
    print("PRONDONE", flush=True)
