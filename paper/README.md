# 论文：Shared Backbone, Separate Minds?

2026-11-16 初稿。数字全部来自 `reports/Qwen3 结构与功能溯源实验.md` 和 `experiments/*/results` 文件；图里的每个数都在 `figures/make_figures.py` 中注明了来源文件。

代码与逐条结果：<https://github.com/anweat/qwen3-inheritance>

## 文件

| 文件 | 内容 |
|---|---|
| `main.tex` | 正文（英文），**这是唯一源文件**；`arxiv/main.tex` 由它复制 |
| `refs.bib` | 40 条参考文献，都已对照 arXiv、Crossref 或官方页面核实（jieba 只标版本号，不写年份） |
| `figures/make_figures.py` | 生成 5 张图（矢量 PDF） |
| `main.pdf` | 本地预览，14 页 |
| `arxiv_source/`、`arxiv_source.tar.gz` | 提交 arXiv 用的源码包，含 `main.tex`、`main.bbl`、`refs.bib`、`figures/` |
| `SUBMISSION.md` | arXiv 投稿清单：元数据、分类、上传文件和检查点 |

## 编译

```bash
cd paper && ../.venv-paper/bin/python figures/make_figures.py && tectonic -X compile main.tex
```

本地用 tectonic，引擎是 XeTeX，中文走 xeCJK 和宋体。arXiv 用 pdfLaTeX 编译，中文走 CJKutf8（gbsn 字体）。pdfLaTeX 这条路径在本机没有测过，上传后要看 arXiv 生成的 PDF 里中文字符是否正常；正文里只有作者名和 6 个汉字。

改完 `main.tex` 后要同步两处，否则会和 `main.tex` 不一致（2026-11-16 修过一次：`arxiv_source/` 里还是旧稿）：

```bash
cp main.tex arxiv/main.tex
rm -rf arxiv_source && mkdir arxiv_source
cp main.tex refs.bib main.bbl arxiv_source/ && cp -R figures arxiv_source/figures
rm -f arxiv_source/figures/make_figures.py
tar czf arxiv_source.tar.gz -C arxiv_source .
```

## 已处理（2026-11-16）

1. 作者 Jiexiong Zhao（赵杰雄）、单位北京交通大学、邮箱 anweat@outlook.com。
2. 代码仓库地址写进第 1 节脚注：<https://github.com/anweat/qwen3-inheritance>。
3. 致谢：开源模型（Qwen3 家族、Llasa、Spark-TTS、CosyVoice 2、Llama/Qwen2/Qwen2.5、SenseVoice、Paraformer、Whisper）、开源数据（AISHELL-1、FLEURS、CPP/g2pM、g2pW）、开源软件（MLX、jieba），以及 Claude（Anthropic）的协助。

## 还要做的

- E8 多人听测（`experiments/e8_listening/`）：结果出来后补进第 7 节和局限。
- 第二标注者复核金标（局限里现在写的是“一名标注者（AI 助手）”）。
- 跨家族 CKA：换一个能通过正对照的度量。E7 的特征保存在 AutoDL 实例盘上。
- 拿到 arXiv 编号后，把它填进 `README.md` 和 `SUBMISSION.md` 的 comments 字段。
