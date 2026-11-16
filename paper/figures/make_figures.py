# Figures for the paper. Every number is copied from a results file named next to it; nothing is recomputed here.
#   ../../.venv-paper/bin/python make_figures.py      -> fig_*.pdf in this folder
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Palette: categorical slots 1-5 of the dataviz reference instance (validated: CVD and normal-vision checks pass;
# contrast warns for aqua and yellow, so every series also gets a marker or a direct label).
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY, INK, MUTED, GRID = "#8a8984", "#1f1e1b", "#6c6a64", "#e6e4de"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.6,
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


# ---- Figure 1: weight lineage of the Qwen3-0.6B family (experiments/e1_block_atlas/tree_results.txt, [A] and [C])
names = ["Base", "Instruct", "Guard-Gen", "Guard-Stream", "Embedding", "Reranker", "ASR", "ForcedAligner", "TTS-Base", "TTS-CustomVoice"]
D = np.array([
    [0.000, 0.123, 0.005, 0.124, 0.078, 0.128, 0.365, 0.661, 1.504, 1.504],
    [0.123, 0.000, 0.123, 0.013, 0.146, 0.036, 0.365, 0.661, 1.500, 1.500],
    [0.005, 0.123, 0.000, 0.124, 0.078, 0.128, 0.365, 0.661, 1.504, 1.504],
    [0.124, 0.013, 0.124, 0.000, 0.147, 0.038, 0.365, 0.661, 1.500, 1.500],
    [0.078, 0.146, 0.078, 0.147, 0.000, 0.150, 0.374, 0.666, 1.505, 1.505],
    [0.128, 0.036, 0.128, 0.038, 0.150, 0.000, 0.367, 0.662, 1.500, 1.500],
    [0.365, 0.365, 0.365, 0.365, 0.374, 0.367, 0.000, 0.586, 1.461, 1.461],
    [0.661, 0.661, 0.661, 0.661, 0.666, 0.662, 0.586, 0.000, 1.457, 1.457],
    [1.504, 1.500, 1.504, 1.500, 1.505, 1.500, 1.461, 1.457, 0.000, 0.014],
    [1.504, 1.500, 1.504, 1.500, 1.505, 1.500, 1.461, 1.457, 0.014, 0.000]])
rn = names[1:]
C = np.array([
    [1.000, 0.002, 0.995, 0.001, 0.959, 0.135, 0.055, 0.000, 0.000],
    [0.002, 1.000, 0.007, 0.001, 0.002, 0.001, 0.000, 0.000, 0.000],
    [0.995, 0.007, 1.000, 0.001, 0.954, 0.135, 0.055, 0.000, 0.000],
    [0.001, 0.001, 0.001, 1.000, 0.001, 0.001, 0.002, 0.000, 0.000],
    [0.959, 0.002, 0.954, 0.001, 1.000, 0.130, 0.053, 0.000, 0.000],
    [0.135, 0.001, 0.135, 0.001, 0.130, 1.000, 0.392, 0.000, 0.000],
    [0.055, 0.000, 0.055, 0.002, 0.053, 0.392, 1.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000]])
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(wspace=.55))
for a, M, lab, title, vmax in ((ax[0], D, names, "(a) weight distance $\\|W_a-W_b\\|/\\sqrt{\\|W_a\\|\\|W_b\\|}$", 1.6),
                               (ax[1], C, rn, "(b) cosine of residuals $W-\\alpha B$ (shrink removed)", 1.0)):
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
    a.set_xticks(range(len(lab)), lab, rotation=55, ha="right", fontsize=7)
    a.set_yticks(range(len(lab)), lab, fontsize=7)
    a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title(title, fontsize=8, color=INK)
    for i in range(len(lab)):
        for j in range(len(lab)):
            v = M[i, j]
            a.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else f"{v:.1f}", ha="center", va="center", fontsize=5,
                   color="white" if v > .55 * vmax else INK)
save(fig, "fig_lineage.pdf")

# ---- Figure 2: layerwise CKA to the same-size Base (experiments/e4_tts_structure/results.txt, block [1])
L = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28]
c06 = {
    "Instruct-0.6B": [0.997, 0.983, 0.969, 0.963, 0.753, 0.932, 0.747, 0.740, 0.933, 0.924, 0.921, 0.927, 0.919, 0.929, 0.924, 0.940],
    "ASR-0.6B": [0.973, 0.977, 0.970, 0.967, 0.919, 0.958, 0.918, 0.913, 0.961, 0.956, 0.961, 0.958, 0.951, 0.952, 0.947, 0.963],
    "TTS-0.6B": [0.577, 0.634, 0.610, 0.460, 0.293, 0.343, 0.243, 0.169, 0.052, 0.049, 0.055, 0.050, 0.037, 0.033, 0.027, 0.019],
    "Random init": [0.456, 0.112, 0.128, 0.132, 0.130, 0.166, 0.156, 0.165, 0.192, 0.177, 0.194, 0.192, 0.167, 0.202, 0.198, 0.127],
}
c17 = {
    "TTS-1.7B": [0.629, 0.703, 0.664, 0.557, 0.452, 0.413, 0.338, 0.341, 0.332, 0.197, 0.204, 0.136, 0.104, 0.123, 0.019, 0.032],
    "Random init": [0.402, 0.125, 0.143, 0.147, 0.152, 0.189, 0.181, 0.205, 0.229, 0.207, 0.197, 0.189, 0.161, 0.182, 0.169, 0.108],
}
style = {"Instruct-0.6B": (VIOLET, "s"), "ASR-0.6B": (BLUE, "o"), "TTS-0.6B": (ORANGE, "^"), "TTS-1.7B": (ORANGE, "^"), "Random init": (GRAY, "x")}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True, gridspec_kw=dict(wspace=.08))
for a, data, title in ((ax[0], c06, "(a) Qwen3-0.6B family (reference: 0.6B-Base)"), (ax[1], c17, "(b) Qwen3-1.7B family (reference: 1.7B-Base)")):
    for k, v in data.items():
        col, mk = style[k]
        a.plot(L, v, color=col, marker=mk, markersize=4, lw=1.4 if k != "Random init" else 1.1,
               ls="-" if k != "Random init" else "--", label=k)
    a.set_title(title, fontsize=8, color=INK, loc="left")
    a.set_xlabel("layer")
    a.set_xlim(-1, 29)
    a.set_xticks([0, 4, 8, 12, 16, 20, 24, 28])
ax[0].set_ylabel("debiased linear CKA")
ax[0].set_ylim(-.02, 1.02)
ax[0].legend(loc="center right", fontsize=7, ncol=2, bbox_to_anchor=(1, .55))
ax[1].legend(loc="upper right", fontsize=7)
save(fig, "fig_cka.pdf")

# ---- Figure 3: causal stitching into the TTS talker (experiments/e4_tts_structure/stitch_results.txt and stitch_control_results.txt; report table)
cols = ["0.6B\nk=1", "0.6B\nk=3", "0.6B\nk=6", "1.7B\nk=1", "1.7B\nk=2", "1.7B\nk=3"]
rows = [("LLM state, same depth", [0.10, -0.03, 0.02, 0.30, 0.15, 0.01], ORANGE),
        ("LLM token embedding", [0.14, 0.08, 0.01, 0.36, 0.27, 0.05], YELLOW),
        ("other TTS, same depth (+ control)", [0.97, 0.93, 0.44, 0.85, 0.57, 0.07], BLUE),
        ("own previous layer (+ control)", [0.99, 0.99, 0.47, 0.80, 0.86, 0.67], VIOLET)]
fig, a = plt.subplots(figsize=(7.0, 2.2))
x = np.arange(len(cols))
w = .2
for i, (lab, v, col) in enumerate(rows):
    a.bar(x + (i - 1.5) * w, v, w * .9, color=col, label=lab)
a.axhline(0, color=MUTED, lw=.8)
a.set_xticks(x, cols)
a.set_ylabel("text effect recovered")
a.set_ylim(-.1, 1.05)
a.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(.5, 1.22))
a.grid(axis="x", visible=False)
save(fig, "fig_stitch.pdf")

# ---- Figure 4: CPP minority readings (experiments/e4_tts_structure/poly_prompt_results.txt, poly_prompt_think_results.txt,
#      poly_heads results, g2pw_results.txt; experiments/e6_reading_pipeline/results.txt and judge_results.txt)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=.25))
a = ax[0]
labs = ["TTS\naudio", "prompt\n0.6B", "prompt\n1.7B", "prompt\n4B", "4B\nthink", "TTS-L6\nhead", "LLM-L8\nhead", "g2pW"]
vals = [.522, .461, .478, .461, .517, .748, .774, .900]
cols4 = [GRAY, YELLOW, YELLOW, YELLOW, YELLOW, AQUA, BLUE, VIOLET]
a.bar(range(len(vals)), vals, .7, color=cols4)
for i, v in enumerate(vals):
    a.text(i, v + .015, f"{v:.3f}".lstrip("0"), ha="center", fontsize=6.5, color=INK)
a.set_xticks(range(len(vals)), labs, fontsize=6.5)
a.set_ylim(0, 1)
a.set_ylabel("accuracy, CPP minority readings")
a.set_title("(a) who knows the reading (n = 230)", fontsize=8, color=INK, loc="left")
a.grid(axis="x", visible=False)
a = ax[1]
labs = ["TTS\nalone", "TTS-L6\n+ cb", "LLM-L8\n+ cb", "g2pW\n+ cb"]
vals = [.410, .795, .795, .872]
a.bar(range(4), vals, .62, color=[GRAY, AQUA, BLUE, VIOLET])
for i, v in enumerate(vals):
    a.text(i, v + .015, f"{v:.3f}".lstrip("0"), ha="center", fontsize=6.5, color=INK)
a.set_xticks(range(4), labs, fontsize=6.5)
a.set_ylim(0, 1)
a.set_title("(b) what the TTS then says (synthesized, n = 78)", fontsize=8, color=INK, loc="left")
a.grid(axis="x", visible=False)
save(fig, "fig_pron.pdf")

# ---- Figure 5: on-device pipeline, paired differences with 95% CI (experiments/e6_reading_pipeline/results.txt, judge_results.txt)
rowsF = [("CPP minority", "detector", .326, .209, .447, 230), ("CPP minority", "3-judge panel", .344, .192, .497, 163),
         ("CPP majority", "detector", .222, .120, .333, 90), ("CPP majority", "3-judge panel", .161, .066, .270, 62),
         ("natural2", "detector", .122, .034, .221, 131), ("natural2", "3-judge panel", .059, -.060, .206, 68),
         ("real-speech sentences", "detector", .013, -.024, .070, 76)]
fig, a = plt.subplots(figsize=(3.4, 2.4))
y = np.arange(len(rowsF))[::-1]
for yi, (s, j, m, lo, hi, n) in zip(y, rowsF):
    col, mk = (BLUE, "o") if j == "detector" else (ORANGE, "D")
    a.plot([lo, hi], [yi, yi], color=col, lw=1.6)
    a.plot(m, yi, mk, color=col, markersize=5, markeredgecolor="white", markeredgewidth=.8)
a.axvline(0, color=MUTED, lw=.8)
a.axvline(-.05, color=MUTED, lw=.8, ls=":")
a.set_yticks(y, [f"{s}, {j} (n={n})" for s, j, *_ , n in rowsF], fontsize=6.5)
a.set_xlabel("pipeline minus TTS alone (accuracy)")
a.grid(axis="y", visible=False)
a.text(-.05, y[0] + .7, "safety\nmargin", fontsize=6, color=MUTED, ha="center", va="bottom")
a.set_ylim(-.6, len(rowsF) + .2)
save(fig, "fig_pipeline.pdf")
print("ok")
