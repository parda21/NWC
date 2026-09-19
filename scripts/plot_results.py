"""Renders the headline chart in the README (docs/img/headline.png) from the measured numbers in docs/results.md.
usage: python scripts/plot_results.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img", "headline.png")
NATIVE, NWC, FP8, DF11 = "#9aa5b1", "#1f6feb", "#2da44e", "#d29922"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#6e7781", "axes.labelcolor": "#24292f", "xtick.color": "#24292f", "ytick.color": "#24292f"})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=180, gridspec_kw={"width_ratios": [1.15, 1.15, 1.3]})


def bars(ax, groups, labels, colors, ylabel, title, fmt, ylim=None):
    n = len(labels); width = 0.8 / n
    for i, (lab, col) in enumerate(zip(labels, colors)):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        vals = [v[i] for _, v in groups]
        b = ax.bar(xs, vals, width * 0.92, color=col, label=lab, zorder=3)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt(v), ha="center", va="bottom", fontsize=9.5, color="#24292f")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11.5, loc="left", color="#24292f")
    ax.grid(axis="y", color="#eaeef2", zorder=0)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=3, columnspacing=0.6, handlelength=1.0)


# Qwen3-4B, greedy decoding as a CUDA graph (docs/results.md, section 1)
# NWC fp8 = weight-only fp8 e4m3 with the fp8 values stored lossless (docs/results.md, section 8)
bars(axes[0], [("RTX 4070", (45.0, 55.2, 73.6)), ("A16 (vGPU 16Q)", (16.8, 18.2, 20.1))], ["native BF16", "NWC BF16", "NWC fp8"], [NATIVE, NWC, FP8],
     "tokens / s", "Decoding speed, Qwen3-4B (CUDA graph)", lambda v: f"{v:.1f}", (0, 96))
bars(axes[1], [("RTX 4070", (8.10, 5.67, 3.61)), ("A16 (vGPU 16Q)", (8.10, 5.67, 3.61))], ["native BF16", "NWC BF16", "NWC fp8"], [NATIVE, NWC, FP8],
     "GB", "VRAM in use, Qwen3-4B", lambda v: f"{v:.2f}", (0, 11.5))
# RTX 4070, GPU time per token, HF eager, torch.profiler (docs/results.md, section 2)
ax = axes[2]
names, vals, cols = ["native BF16", "DFloat11", "NWC"], [20.4, 56.1, 18.9], [NATIVE, DF11, NWC]
b = ax.barh(names[::-1], vals[::-1], color=cols[::-1], zorder=3, height=0.62)
for r, v in zip(b, vals[::-1]):
    ax.text(v + 0.8, r.get_y() + r.get_height() / 2, f"{v:.1f} ms", va="center", fontsize=9.5, color="#24292f")
ax.set_xlim(0, 66); ax.set_xlabel("GPU time per token (ms), lower is better"); ax.grid(axis="x", color="#eaeef2", zorder=0)
ax.set_title("vs DFloat11, RTX 4070: same size, 3× faster", fontsize=11.5, loc="left", color="#24292f")
ax.text(0.99, 0.02, "VRAM: 8.10 / 5.73 / 5.67 GB", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#57606a")

fig.tight_layout(w_pad=2.0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor="white")
print("written:", OUT)
