# Shared Backbone, Separate Minds?

Code, per-item outputs and LaTeX source for the paper **“Shared Backbone, Separate Minds?
What Qwen3 Speech Models Inherit from Their LLM”**.

The paper asks what the ASR and TTS models of the Qwen3 family actually inherit from the LLM they
were built from — at the level of weights (shrink-corrected lineage), representations (debiased linear
CKA with random-network controls), causal substitutability (model stitching), and behaviour (what the
TTS actually says, judged by independent ASR models). It also tests whether one piece of LLM knowledge,
the context-dependent reading of Mandarin polyphones, can be handed to the TTS through a narrow
discrete interface.

- Paper PDF: [`paper/main.pdf`](paper/main.pdf)
- LaTeX source: [`paper/main.tex`](paper/main.tex)
- arXiv: not yet posted; the ID will be added here.

## What is in this repository

| Path | Contents |
|---|---|
| `paper/` | LaTeX source, bibliography, vector figures, compiled PDF, arXiv bundle (`arxiv_source.tar.gz`) |
| `paper/figures/make_figures.py` | Regenerates all five figures from the result files in `experiments/` |
| `experiments/e0_delta_probe/` | Weight-difference cosine probe: how a shared shrink toward the base inflates delta cosines |
| `experiments/e1_block_atlas/` | Shrink-corrected weight atlas, pairwise distances, 4-bit code entropy, plasticity bands |
| `experiments/e2_activation/` | Massive activation and attention sink, layer/neuron ablations, transcription-vs-text double dissociation |
| `experiments/e3_real_speech/` | ASR evaluation on 800 held-out utterances, per-clip outputs |
| `experiments/e4_tts_structure/` | CKA profiles, the talker's text read-out window, causal stitching, G2P baselines |
| `experiments/e5_runtime/` | On-device measurement with packed weights and real kernels (Apple M4, MLX) |
| `experiments/e6_reading_pipeline/` | Reading probes, pronunciation codebook, int8 g2pW pipeline, judge panel and scoring |
| `experiments/e7_cross_family/` | Llasa-1B, Spark-TTS-0.5B and CosyVoice 2 lineage and CKA (run on a remote GPU instance) |
| `experiments/e8_listening/` | Blind multi-listener study: page generator and scoring scripts |
| `reports/` | Working reports, in Chinese, with the numbers each paper claim was taken from |

Each `experiments/*/` directory contains the scripts and the raw or summarised outputs (`*.txt`,
`*.json`, `*.jsonl`) that the corresponding numbers in the paper come from. Criterion definitions are
inside the scripts, which were written before the outputs they judge were produced.

## Not included

- Model weights. All models are public: the Qwen3 family from the Qwen team (Base, Instruct, ASR,
  TTS, ForcedAligner, Guard, Embedding, Reranker, VL), and Llasa-1B, Spark-TTS-0.5B and CosyVoice 2
  for the cross-family check.
- AISHELL-1 and FLEURS audio, which are distributed by their own providers under their own terms.
  Only derived per-clip scores and text-level metadata are released here.
- Python environments. `.venv*` is git-ignored; the scripts state their imports.

## Reproducing

Hardware used: an Apple M4 for the representation, stitching and on-device runtime experiments
(float32 on MPS for analysis, MLX for the quantised kernels), and a rented GPU instance for the
cross-family check.

```bash
# figures + PDF (tectonic uses the XeTeX path with xeCJK for the Chinese characters)
cd paper
../.venv-paper/bin/python figures/make_figures.py
tectonic -X compile main.tex
```

The paper is written so that it compiles on arXiv with pdfLaTeX (CJK via `CJKutf8`) and locally with
XeTeX (CJK via `xeCJK`); `iftex` selects the path. See [`paper/SUBMISSION.md`](paper/SUBMISSION.md)
for the arXiv submission checklist.

## Citation

The paper is not on arXiv yet. Until the arXiv ID exists:

```bibtex
@misc{zhao2026sharedbackbone,
  title  = {Shared Backbone, Separate Minds? {What} {Qwen3} Speech Models Inherit from Their {LLM}},
  author = {Zhao, Jiexiong},
  year   = {2026},
  note   = {Preprint. Code: \url{https://github.com/anweat/qwen3-inheritance}}
}
```

## License

Code and scripts: MIT (see [`LICENSE`](LICENSE)). The paper text and figures are released under
CC BY 4.0.

## Contact

Jiexiong Zhao, Beijing Jiaotong University — anweat@outlook.com
