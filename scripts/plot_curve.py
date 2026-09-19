"""Size-vs-quality curve (docs/img/curve_qwen3_8b.png): one base model, every point measured with the same perplexity
method (scripts/ppl_curve.py for BF16 / fp8 / NWC, llama-perplexity for GGUF models, same text, n_ctx 2048).
usage: python scripts/plot_curve.py [docs/data/ppl_curve_qwen3_8b.json]"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "data", "ppl_curve_qwen3_8b.json")
d = json.load(open(SRC))
OUT = os.path.join(ROOT, "docs", "img", d.get("image", "curve_qwen3_8b.png"))
COL = {"native": "#9aa5b1", "nwc": "#1f6feb", "fp8": "#2da44e", "other": "#d29922"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#6e7781", "axes.labelcolor": "#24292f", "xtick.color": "#24292f", "ytick.color": "#24292f"})
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=180)
pts = d["points"]
base = next(p for p in pts if p.get("reference"))
for p in pts:
    c = COL[p.get("kind", "other")]
    ax.scatter(p["gb"], p["ppl"], s=110 if p.get("kind") == "nwc" else 80, color=c, zorder=4, edgecolor="white", linewidth=1.2)
    dx, dy = p.get("offset", (10, 0))                      # label offset in points
    txt = p["label"] + (f"\n{p['gb']:.1f} GB, ppl {p['ppl']:.2f}" if p.get("detail", True) else "")
    ax.annotate(txt, (p["gb"], p["ppl"]), xytext=(dx, dy), textcoords="offset points", fontsize=9.2, color="#24292f",
                ha=p.get("ha", "left"), va=p.get("va", "center"), zorder=5)
ax.axhline(base["ppl"], color=COL["native"], lw=1, ls=(0, (4, 3)), zorder=2)
ax.text(0.4, base["ppl"], "BF16 quality", fontsize=8.5, color="#57606a", ha="left", va="bottom")
ax.set_xlim(0, d.get("xmax", 18)); ax.set_ylim(*d.get("ylim", (base["ppl"] - 0.6, base["ppl"] + 2.2)))
ax.set_xlabel("weights in memory (GB)"); ax.set_ylabel("perplexity, WikiText-2 (lower is better)")
ax.set_title(d.get("title", "Qwen3-8B: size vs quality, every point measured the same way"), fontsize=11.5, loc="left", color="#24292f")
ax.grid(color="#eaeef2", zorder=0)
fig.text(0.99, 0.01, d.get("note", ""), ha="right", va="bottom", fontsize=8.2, color="#57606a")
fig.tight_layout(rect=(0, 0.035, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor="white")
print("written:", OUT)
