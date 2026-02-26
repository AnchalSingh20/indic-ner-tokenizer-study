"""Render results/fertility_vs_f1.png.

Left: the relationship the study is about -- how badly a tokenizer fragments a
script, against how far behind that model's NER falls on it. A scatter, because
the claim is about a relationship between two continuous quantities.
Right: the underlying per-language F1, so the reader can see the raw magnitudes
rather than only the derived gap.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHI, QWEN = "#2a78d6", "#eb6834"      # categorical slots 1 and 2
POINT = "#4a3aa7"                      # slot 7, distinct from both series

NAMES = {"hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ta": "Tamil",
         "te": "Telugu", "pa": "Punjabi", "gu": "Gujarati"}
ORDER = ["hi", "mr", "bn", "pa", "ta", "gu", "te"]


def main(out="results"):
    fert = {r["lang"]: r for r in json.load(open(f"{out}/fertility.json"))}
    df = pd.read_csv(f"{out}/ner_raw.csv")

    x, gap, phi_f1, qwen_f1, err_p, err_q = [], [], [], [], [], []
    for l in ORDER:
        s = df[df.lang == l]
        p = s[s.model == "phi-4-mini"]["f1"] * 100
        q = s[s.model == "qwen2.5-3b"]["f1"] * 100
        x.append(fert[l]["qwen2.5-3b_fertility"])
        gap.append(p.mean() - q.mean())
        phi_f1.append(p.mean()); qwen_f1.append(q.mean())
        err_p.append(p.std(ddof=1)); err_q.append(q.std(ddof=1))
    rho, pv = spearmanr(x, gap)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # ---- left: fertility vs gap ----
    ax1.set_facecolor(SURFACE)
    m, b = np.polyfit(x, gap, 1)
    xs = np.linspace(min(x) - 0.3, max(x) + 0.4, 50)
    ax1.plot(xs, m * xs + b, color=INK_2, linewidth=1.4, linestyle="--",
             zorder=2, alpha=0.7)
    ax1.scatter(x, gap, s=110, color=POINT, zorder=3, linewidth=0)
    for xi, yi, l in zip(x, gap, ORDER):
        ax1.annotate(NAMES[l], (xi, yi), textcoords="offset points",
                     xytext=(9, 4), fontsize=9.5, color=INK)
    ax1.axhline(0, color=GRID, linewidth=1.2, zorder=1)
    ax1.set_xlabel("Qwen2.5 tokenizer fertility (tokens per word)",
                   fontsize=10.5, color=INK_2)
    ax1.set_ylabel("Phi-4-mini advantage (F1 points)", fontsize=10.5, color=INK_2)
    ax1.set_title(f"Worse tokenization, wider gap  (Spearman rho = {rho:.2f}, p = {pv:.3f})",
                  fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=10)

    # ---- right: per-language F1 ----
    ax2.set_facecolor(SURFACE)
    idx = np.arange(len(ORDER)); bw = 0.36
    ax2.bar(idx - bw / 2 - 0.01, phi_f1, bw, yerr=err_p, capsize=3, color=PHI,
            linewidth=0, zorder=3, error_kw=dict(ecolor=INK_2, elinewidth=1, capthick=1))
    ax2.bar(idx + bw / 2 + 0.01, qwen_f1, bw, yerr=err_q, capsize=3, color=QWEN,
            linewidth=0, zorder=3, error_kw=dict(ecolor=INK_2, elinewidth=1, capthick=1))
    ax2.set_xticks(idx)
    ax2.set_xticklabels([NAMES[l] + ("\n(trained)" if l == "hi" else "")
                         for l in ORDER], fontsize=9, color=INK, rotation=20, ha="right")
    ax2.set_ylabel("Entity F1 (%)", fontsize=10.5, color=INK_2)
    ax2.set_title("Per-language NER, trained on Hindi only",
                  fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=10)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=PHI),
               plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=QWEN)]
    leg = ax2.legend(handles, ["Phi-4-mini", "Qwen2.5-3B"], loc="upper right",
                     frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK_2)

    for ax in (ax1, ax2):
        ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=9.5, colors=INK_2)

    fig.tight_layout()
    fig.savefig(f"{out}/fertility_vs_f1.png", facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}/fertility_vs_f1.png")


if __name__ == "__main__":
    main()
