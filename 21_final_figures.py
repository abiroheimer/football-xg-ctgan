"""
21_final_figures.py — the two new figures for the rewritten paper
(football.tex currently has placeholder \\fbox boxes for both; this script
only produces the PNGs, no .tex edits, no compile).

Plotting only — reads already-computed results/*.csv, no retraining. Style
matches 16_fig8_compact_for_paper.py (same config.py color constants,
errorbar/marker conventions, IEEEtran single-column sizing) so these sit
visually consistent with the project's other output figures even though
the ones from 16_ were cut from the current paper draft.

Output: figures/fig_loco_per_fold.png
        figures/fig_splitdist_ctgan_smote.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C_ALL, C_C, C_SYNTH, FIG_DIR, RESULTS_DIR, SEED

RNG = np.random.default_rng(SEED)
N_BOOT = 10_000

PRIMARY_FOLDS = ["England", "France", "Germany", "Italy", "Spain"]
HEADLINE_RATIO = 0.05

# ════════════════════════════════════════════════════════════════════════
# Figure 1 — LOCO per-fold calibration improvement, headline ratio (5%)
# ════════════════════════════════════════════════════════════════════════
def bootstrap_ci_1d(vals, n_boot=N_BOOT):
    vals = np.asarray(vals)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, len(vals), size=len(vals))
        boot[b] = vals[idx].mean()
    return np.percentile(boot, [2.5, 97.5])


def make_fig1():
    raw = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_E_loco_raw_runs.csv"))
    per_fold = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_E_loco_per_fold.csv"))
    supp = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_E_loco_supplementary.csv"))
    macro_ci = pd.read_csv(os.path.join(RESULTS_DIR, "bootstrap_ci_loco_calib_5pct.csv")).iloc[0]

    baseA_by_fold = per_fold.drop_duplicates("fold").set_index("fold")["baseA_calib_err"]

    labels, points, lo_err, hi_err, colors = [], [], [], [], []
    for f in PRIMARY_FOLDS:
        runs = raw[(raw.fold == f) & (raw.ratio == HEADLINE_RATIO)]["calib"].values
        rel_improve_runs = 100 * (1.0 - runs / baseA_by_fold[f])
        point = rel_improve_runs.mean()
        lo, hi = bootstrap_ci_1d(rel_improve_runs)
        labels.append(f)
        points.append(point)
        lo_err.append(point - lo)
        hi_err.append(hi - point)
        colors.append(C_C)

    # Macro aggregate (two-stage bootstrap CI already computed)
    labels.append("Macro\n(5 folds)")
    points.append(float(macro_ci["point_estimate_pct"]))
    lo_err.append(float(macro_ci["point_estimate_pct"]) - float(macro_ci["ci_2.5pct"]))
    hi_err.append(float(macro_ci["ci_97.5pct"]) - float(macro_ci["point_estimate_pct"]))
    colors.append("#333333")

    # Supplementary WC+Euro fold (low confidence, visually distinct)
    supp_row = supp[supp.ratio == HEADLINE_RATIO].iloc[0]
    supp_runs_raw = None  # supplementary raw runs live in the same loco raw file, tagged by fold name
    supp_fold_name = supp_row["fold"]
    supp_raw = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_E_loco_raw_runs.csv"))
    supp_runs = supp_raw[(supp_raw.fold == supp_fold_name) & (supp_raw.ratio == HEADLINE_RATIO)]["calib"].values
    supp_rel = 100 * (1.0 - supp_runs / float(supp_row["baseA_calib_err"]))
    supp_point = supp_rel.mean()
    supp_lo, supp_hi = bootstrap_ci_1d(supp_rel)
    labels.append("WC+Euro\n(suppl., n=61)")
    points.append(supp_point)
    lo_err.append(supp_point - supp_lo)
    hi_err.append(supp_hi - supp_point)
    colors.append("#AAAAAA")

    y_pos = np.arange(len(labels))[::-1]  # top-to-bottom in list order

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    for i, (y, pt, le, he, c, lbl) in enumerate(zip(y_pos, points, lo_err, hi_err, colors, labels)):
        is_macro = "Macro" in lbl
        is_supp = "suppl" in lbl
        marker = "D" if is_macro else ("s" if is_supp else "o")
        ms = 7 if is_macro else 5.5
        alpha = 0.55 if is_supp else 0.95
        ax.errorbar(pt, y, xerr=[[le], [he]], fmt=marker, color=c, ecolor=c,
                    capsize=3, capthick=1.1, elinewidth=1.3, ms=ms, alpha=alpha,
                    markeredgecolor="black" if is_macro else "none", markeredgewidth=0.8 if is_macro else 0)

    ax.axvline(0, color="black", ls="-", lw=0.8)
    # Separator line between primary folds+macro and the supplementary fold
    ax.axhline(0.5, color="#cccccc", ls=":", lw=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Relative Calibration-Error Improvement vs.\nFair Baseline (%, 5% Synthetic Ratio)", fontsize=6.8)
    ax.set_title("LOCO Cross-Competition Cold-Start\n(95% Bootstrap CI)", fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_loco_per_fold.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ════════════════════════════════════════════════════════════════════════
# Figure 2 — CTGAN vs SMOTE final test log_loss, 8 match-grouped splits
# ════════════════════════════════════════════════════════════════════════
def make_fig2():
    per_split = pd.read_csv(os.path.join(RESULTS_DIR, "multiseed_valtest_per_split_matchgrouped.csv"))
    head = pd.read_csv(os.path.join(RESULTS_DIR, "multiseed_valtest_headtohead_matchgrouped.csv"))

    splits = sorted(per_split["split_id"].unique())
    baseline = per_split[per_split.arm == "Baseline A"].set_index("split_id")["test_log_loss_mean"]
    ctgan = per_split[per_split.arm == "CTGAN"].set_index("split_id")["test_log_loss_mean"]
    smote = per_split[per_split.arm == "SMOTE"].set_index("split_id")["test_log_loss_mean"]
    winner = head.set_index("split_id")["winner"]

    x = np.arange(1, len(splits) + 1)
    xlabels = [f"S{s}" if s != 1 else "S1\n(committed)" for s in splits]

    fig, ax = plt.subplots(figsize=(3.6, 3.4))

    for i, s in enumerate(splits):
        xi = x[i]
        c_val, s_val, b_val = ctgan[s], smote[s], baseline[s]
        line_color = C_C if winner[s] == "CTGAN" else C_SYNTH
        ax.plot([xi, xi], [c_val, s_val], color=line_color, lw=1.4, alpha=0.6, zorder=1)
        ax.scatter(xi, b_val, marker="_", s=180, color="#999999", linewidths=1.6, zorder=2)

    ax.scatter(x, [ctgan[s] for s in splits], marker="o", s=34, color=C_C, label="CTGAN (val.-selected)",
               edgecolor="black", linewidth=0.4, zorder=3)
    ax.scatter(x, [smote[s] for s in splits], marker="s", s=34, color=C_SYNTH, label="SMOTE (val.-selected)",
               edgecolor="black", linewidth=0.4, zorder=3)
    ax.scatter([], [], marker="_", s=180, color="#999999", linewidths=1.6, label="Baseline A")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6.3)
    ax.set_ylabel("Held-Out Test Log Loss", fontsize=7.5)
    ax.set_xlabel("Match-Grouped Split (1 = originally-committed)", fontsize=7)
    ax.set_title("CTGAN vs. SMOTE Across 8 Independent\nMatch-Grouped Splits", fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=5.8, loc="upper left")
    ax.grid(axis="y", color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    n_ctgan_wins = int((winner == "CTGAN").sum())
    n_smote_wins = int((winner == "SMOTE").sum())
    ax.text(0.98, 0.02,
            f"CTGAN wins {n_ctgan_wins}/8, SMOTE wins {n_smote_wins}/8\n"
            f"paired $t$-test across splits: $p=0.79$ (n.s.)",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.7))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_splitdist_ctgan_smote.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
