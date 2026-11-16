# arXiv submission checklist

Status: metadata complete (author, affiliation, e-mail, code URL, acknowledgements). What remains is
the upload itself.

## 1. Metadata to paste into the form

| Field | Value |
|---|---|
| Title | Shared Backbone, Separate Minds? What Qwen3 Speech Models Inherit from Their LLM |
| Authors | Jiexiong Zhao |
| Affiliation (author field) | Beijing Jiaotong University |
| E-mail | anweat@outlook.com |
| Primary category | `cs.CL` (Computation and Language) |
| Cross-lists | `eess.AS` (Audio and Speech Processing), `cs.SD` (Sound) — optional, drop them if the endorsement for `eess.AS` is missing |
| Comments | `14 pages, 5 figures. Code and per-item results: https://github.com/anweat/qwen3-inheritance` |
| License | CC BY 4.0 (default) — required to post; pick another only if you want to restrict reuse |
| Journal reference / DOI | leave empty |

The abstract goes into the form as **plain text**: no LaTeX, no line breaks, no `\citep`. The running
text of `main.tex` is already plain ASCII apart from a handful of Chinese characters — 赵杰雄 in the
author block, and 行, 番, 禺, 一, 不 in §7 and Appendix C — the abstract contains none, so copy it from
the compiled PDF, not from the `.tex`.

## 2. Files to upload

`paper/arxiv_source.tar.gz` (built from `paper/arxiv_source/`) contains exactly what arXiv needs:

```
main.tex      # the paper; \ifXeTeX selects pdfLaTeX + CJKutf8 on arXiv
main.bbl      # pre-built bibliography: arXiv does not have to run BibTeX
refs.bib      # kept for the record; harmless if arXiv runs BibTeX anyway
figures/      # five vector PDFs, referenced as figures/fig_*.pdf
```

Regenerate the bundle after any edit:

```bash
cd paper
cp main.tex arxiv/main.tex
rm -rf arxiv_source && mkdir arxiv_source
cp main.tex refs.bib main.bbl arxiv_source/
cp -R figures arxiv_source/figures && rm -f arxiv_source/figures/make_figures.py
tar czf arxiv_source.tar.gz -C arxiv_source .
```

Do **not** upload `main.pdf` (arXiv rejects PDFs in the source bundle), `make_figures.py`, or the
`.venv*` directories.

## 3. Endorsement (the gate that matters)

Since **21 January 2026** an institutional e-mail alone no longer qualifies a first-time submitter
([policy update](https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/)).
There are two paths, and a new submitter needs one of them:

| Path | Requirement |
|---|---|
| 1. Automatic | An **institutional e-mail** on the arXiv account **and** ownership of a paper already on arXiv in the same *endorsement domain* |
| 2. Personal | A **personal endorsement** from an established arXiv author in that domain |

This author has neither arXiv authorship nor an institutional account yet, so the paper needs **path 2**.
The good news: for Computer Science the endorsement domain is *all of cs.\**, not the single category.
An endorser qualifies by having submitted papers in **any** of `cs.CL, cs.LG, cs.CV, cs.SD, cs.SE, …`
(three or more within the last five years, per the CS rule), so anyone active in CS — an advisor, a
senior labmate, an author of a cited paper — can endorse a `cs.CL` submission. `eess.AS` is a
*different* domain; do not count on an `eess.AS`-only author.

How to proceed:

1. Register with the university address (`@bjtu.edu.cn`), not the outlook one. The paper's corresponding
   e-mail stays `anweat@outlook.com`; the two are separate fields. The institutional account also makes
   path 1 automatic later, once any arXiv paper with this author on it exists.
2. Start a submission and select `cs.CL`. arXiv replies with an **endorsement request e-mail**
   containing a six-character code. Save it — it is what the endorser types in.
3. Find a candidate and check eligibility before writing: open an arXiv abstract page of a recent paper
   in the field → click **"Which authors of this paper are endorsers?"** → the page lists
   "Can endorse for cs.AI, cs.CL, …". Only ask people whose list contains `cs.CL`.
4. Send a short mail with the code, the title, one paragraph of contribution and the PDF link
   (repository or arXiv preview). Suggested text:

   > Subject: arXiv endorsement request for one submission (code: XXXXXX)
   >
   > Dear Prof. X,
   >
   > I am Jiexiong Zhao, an undergraduate at Beijing Jiaotong University. I would like to submit a
   > preprint to arXiv cs.CL and need an endorsement. The paper is "Shared Backbone, Separate Minds?
   > What Qwen3 Speech Models Inherit from Their LLM": using the Qwen3 family, where the LLM, ASR and TTS
   > decoders have identical shapes, it separates weight lineage (after removing a shared shrink term),
   > layerwise representation similarity with random controls, and causal substitutability — ASR reuses
   > the LLM backbone at every depth, TTS only its first layers, and linearly mapped LLM states cannot
   > replace the talker's text path. It also reports a negative on-device result for shared-base
   > quantisation and a pronunciation codebook that lets any G2P control a character-input TTS.
   > Manuscript: https://github.com/anweat/qwen3-inheritance . Endorsement code: XXXXXX
   > (enter at https://arxiv.org/auth/endorse). Thank you for your time.
   >
   > Best regards, Jiexiong Zhao

5. One positive endorsement per domain is enough, and the endorsement is private — a refusal or a
   non-answer is not visible to anyone else. Do not mass-mail or re-mail the same person; that is
   explicitly discouraged by arXiv.
6. If no endorser appears within a week, fall back in this order: (a) post the preprint with a DOI on
   Zenodo (free, no gate) and keep the GitHub repository as the canonical link; (b) submit to the venue
   first and let the camera-ready carry the arXiv version later, once an endorsement exists; (c) ask
   again after co-authoring any arXiv paper — path 1 then triggers automatically.

arXiv staff cannot waive the requirement or endorse on request; there is an
[endorsement feedback form](https://cornell.ca1.qualtrics.com/jfe/form/SV_9mBSosmZCoHNmFE) for policy
questions only.

## 4. Order of operations

1. Sign in at <https://arxiv.org/submit> (or start at <https://arxiv.org/user/register>); complete §3
   first, since the code is only issued once a submission is started.
2. **Start submission**, accept the submission agreement, pick the license.
3. Paste the metadata from §1. Abstract as one paragraph of plain text.
4. Upload `arxiv_source.tar.gz` (or the individual files). arXiv processes it with pdfLaTeX and shows
   the log and any errors.
5. **Inspect the generated PDF**, page 1 in particular: the Chinese name 赵杰雄 in the author block,
   the `行/番禺/禺` characters in §7, the footnote URL in §1, and that the five figures are not blank.
   If the CJK characters come out as boxes, the `CJK` package is missing from arXiv's TeX Live build
   and the fallback is to transliterate those six characters (only six, all in the author name and §7).
6. Set the **submission category** (`cs.CL` is covered by a CS endorsement; `eess.AS` cross-lists may
   require an endorsement in the `eess` domain — drop the cross-list if the form complains).
7. Review the "view paper" preview one more time, then hit **Submit**. The paper is announced at
   20:00 US Eastern on the next business day if it is submitted before 14:00 US Eastern.

## 5. Common rejections and how this source avoids them

- **Missing `.bbl`** — included, so a BibTeX failure cannot block the build.
- **Undefined control sequences from CJK** — only `\zh{}` and the standard `CJK` package are used; no
  `fontspec` on the pdfLaTeX path.
- **Absolute paths in `\includegraphics`** — all figures are relative (`figures/fig_*.pdf`).
- **`.DS_Store` / `__pycache__` in the tarball** — the build command above copies only the five files
  and the figure directory.
- **Title with a line break** — the form gets a single line; the `\\` exists only in `\title{}` for
  typesetting.

## 6. After the announcement

- Put the arXiv ID into `README.md` (top, and in the BibTeX entry) and into `comments`
  (as `arXiv:XXXX.XXXXX`) only via a new version.
- Replace the `paper/` copy of `main.pdf` with the arXiv build if you want the repository PDF to match
  what readers see.
- Changes after announcement are posted as new versions (`v2`, `v3`, …); the metadata can be edited
  until the paper is announced, then only with a new version for the abstract.
