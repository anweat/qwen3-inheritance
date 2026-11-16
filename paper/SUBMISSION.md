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

## 3. Order of operations

1. Sign in at <https://arxiv.org/submit> (or start at <https://arxiv.org/user/register>). A first
   submission to `cs.CL` may need **endorsement** from an established author; if the form asks for an
   endorsement code, request one before going further — the rest of the form can wait.
2. **Start submission**, accept the submission agreement, pick the license.
3. Paste the metadata from §1. Abstract as one paragraph of plain text.
4. Upload `arxiv_source.tar.gz` (or the individual files). arXiv processes it with pdfLaTeX and shows
   the log and any errors.
5. **Inspect the generated PDF**, page 1 in particular: the Chinese name 赵杰雄 in the author block,
   the `行/番禺/禺` characters in §7, the footnote URL in §1, and that the five figures are not blank.
   If the CJK characters come out as boxes, the `CJK` package is missing from arXiv's TeX Live build
   and the fallback is to transliterate those six characters (only six, all in the author name and §7).
6. Set the **submission category** and confirm the cross-lists; add the comments line last, since it is
   truncated at 400 characters in some listings.
7. Review the "view paper" preview one more time, then hit **Submit**. The paper is announced at
   20:00 US Eastern on the next business day if it is submitted before 14:00 US Eastern.

## 4. Common rejections and how this source avoids them

- **Missing `.bbl`** — included, so a BibTeX failure cannot block the build.
- **Undefined control sequences from CJK** — only `\zh{}` and the standard `CJK` package are used; no
  `fontspec` on the pdfLaTeX path.
- **Absolute paths in `\includegraphics`** — all figures are relative (`figures/fig_*.pdf`).
- **`.DS_Store` / `__pycache__` in the tarball** — the build command above copies only the five files
  and the figure directory.
- **Title with a line break** — the form gets a single line; the `\\` exists only in `\title{}` for
  typesetting.

## 5. After the announcement

- Put the arXiv ID into `README.md` (top, and in the BibTeX entry) and into `comments`
  (as `arXiv:XXXX.XXXXX`) only via a new version.
- Replace the `paper/` copy of `main.pdf` with the arXiv build if you want the repository PDF to match
  what readers see.
- Changes after announcement are posted as new versions (`v2`, `v3`, …); the metadata can be edited
  until the paper is announced, then only with a new version for the abstract.
