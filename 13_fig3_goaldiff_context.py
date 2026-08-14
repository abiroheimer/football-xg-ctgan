"""
13_fig3_goaldiff_context.py — Step 4 Fig 1: paper Fig 3a/3b equivalent —
shot volume and conversion rate by goal-differential context, with the rare
zone (goal_diff <= -2) highlighted, from the final shots_cache.csv.

Plotting only — no analysis/model code touched, reads only what Step 1
already produced.

Run standalone:
    python 13_fig3_goaldiff_context.py

Input : data/shots_cache.csv
Output: figures/fig3_goaldiff_context.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C_ALL, C_RARE, FIG_DIR
from config import DATA_DIR

CACHE_FILE = os.path.join(DATA_DIR, "shots_cache.csv")

if __name__ == "__main__":
    shots = pd.read_csv(CACHE_FILE)

    gd = shots.groupby("goal_diff").agg(shots_n=("is_goal", "size"), goals=("is_goal", "sum")).reset_index()
    gd["conv_pct"] = 100 * gd["goals"] / gd["shots_n"]
    gd["is_rare_zone"] = gd["goal_diff"] <= -2
    colors = [C_RARE if r else C_ALL for r in gd["is_rare_zone"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 3 — Shot Volume & Conversion by Goal Differential", fontsize=13, fontweight="bold")

    axes[0].bar(gd["goal_diff"].astype(str), gd["shots_n"], color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_title("3a — Shot Volume by Goal Differential", fontweight="bold")
    axes[0].set_xlabel("Goal Differential (shooting team's perspective)")
    axes[0].set_ylabel("Shot Count")
    for i, v in enumerate(gd["shots_n"]):
        axes[0].text(i, v + max(gd["shots_n"]) * 0.01, f"{v:,}", ha="center", fontsize=8, rotation=90)

    axes[1].bar(gd["goal_diff"].astype(str), gd["conv_pct"], color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_title("3b — Conversion Rate by Goal Differential", fontweight="bold")
    axes[1].set_xlabel("Goal Differential (shooting team's perspective)")
    axes[1].set_ylabel("Conversion Rate (%)")
    for i, v in enumerate(gd["conv_pct"]):
        axes[1].text(i, v + max(gd["conv_pct"]) * 0.01, f"{v:.1f}%", ha="center", fontsize=8)

    for ax in axes:
        ax.axvspan(-0.5, list(gd["goal_diff"]).index(-2) + 0.5 if -2 in gd["goal_diff"].values else 0,
                    color=C_RARE, alpha=0.06)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C_RARE, alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color=C_ALL, alpha=0.85)]
    fig.legend(handles, ["Rare zone (goal_diff <= -2)", "Non-rare"],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(FIG_DIR, "fig3_goaldiff_context.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✔ Saved -> {out_path}")

    rare_mask = shots["goal_diff"] <= -2
    print(f"\nRare-zone (goal_diff<=-2) shots: {rare_mask.sum():,} ({100*rare_mask.mean():.2f}%)  "
          f"conv={100*shots.loc[rare_mask,'is_goal'].mean():.2f}%")
    print(f"Non-rare-zone shots: {(~rare_mask).sum():,}  conv={100*shots.loc[~rare_mask,'is_goal'].mean():.2f}%")
    print("\nPer-goal_diff breakdown:")
    print(gd.to_string(index=False))
