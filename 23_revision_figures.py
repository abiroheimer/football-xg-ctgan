"""
23_revision_figures.py — Blocks B and E of the revision pass. Regenerates
all four of the paper's figures so they share one style definition:

  B1 (new)  figures/fig_loco_dose_response.png    six-ratio LOCO dose-response
  B2 (new)  figures/fig_loco_pipeline_diagram.png LOCO study-design schematic
  E  (fix)  figures/fig_splitdist_ctgan_smote.png revision-history labels removed,
                                                  fonts up, mean-diff annotation
  E  (fix)  figures/fig_loco_per_fold.png         explicit % axis label, direct
                                                  "supplementary" annotation, retitled

Also writes results/bootstrap_ci_loco_all_ratios.csv -- the two-stage
(fold-then-run) bootstrap CI at every tested ratio, not just 5%, which B1
needs for its macro CI band and which the paper's dose-response claim
("significant 5-50%, not 100%") should be able to cite at each point.

All text here is plain matplotlib (no usetex), so no LaTeX escapes.

Plotting only -- reads already-computed results/*.csv, no retraining.

Run standalone:
    python 23_revision_figures.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import C_C, C_SYNTH, FIG_DIR, RESULTS_DIR, SEED

RNG = np.random.default_rng(SEED)
N_BOOT = 10_000
PRIMARY_FOLDS = ["England", "France", "Germany", "Italy", "Spain"]
HEADLINE_RATIO = 0.05
ALL_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

FOLD_COLORS = {
    "England": "#4C72B0", "France": "#C44E52", "Germany": "#DD8452",
    "Italy": "#55A868", "Spain": "#8172B3",
}


def _load_loco():
    raw_all = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_E_loco_raw_runs.csv"))
    per_fold = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_E_loco_per_fold.csv"))
    folds = sorted(per_fold["fold"].unique())
    raw = raw_all[raw_all["fold"].isin(folds)].copy()
    baseA = per_fold.drop_duplicates("fold").set_index("fold")["baseA_calib_err"]
    return raw_all, raw, per_fold, baseA


def two_stage_bootstrap_rel_improve(raw, baseA, ratio, folds, n_boot=N_BOOT):
    """Resample folds with replacement, then runs within each resampled fold --
    runs within a fold share that fold's eval set and generator and are not
    independent draws of the competition population (same rationale as
    20_statistical_rigor.py)."""
    fold_runs = {f: raw[(raw.fold == f) & (raw.ratio == ratio)]["calib"].values for f in folds}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        resampled = RNG.choice(folds, size=len(folds), replace=True)
        vals = np.empty(len(folds))
        for i, f in enumerate(resampled):
            runs = fold_runs[f]
            vals[i] = 1.0 - runs[RNG.integers(0, len(runs), size=len(runs))].mean() / baseA[f]
        boot[b] = vals.mean()
    return np.percentile(boot, [2.5, 97.5]) * 100


def bootstrap_ci_1d(vals, n_boot=N_BOOT):
    vals = np.asarray(vals)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = vals[RNG.integers(0, len(vals), size=len(vals))].mean()
    return np.percentile(boot, [2.5, 97.5])


# ════════════════════════════════════════════════════════════════════════
# B1 — six-ratio LOCO dose-response
# ════════════════════════════════════════════════════════════════════════
def make_dose_response():
    raw_all, raw, per_fold, baseA = _load_loco()
    holm = pd.read_csv(os.path.join(RESULTS_DIR, "loco_ratio_significance_holm.csv"))
    holm_sig = holm.set_index("ratio")["significant_after_holm"]

    ci_rows, macro_pts, macro_lo, macro_hi = [], [], [], []
    for ratio in ALL_RATIOS:
        fold_means = raw[raw.ratio == ratio].groupby("fold")["calib"].mean().reindex(PRIMARY_FOLDS)
        rel = 100 * (1.0 - fold_means.values / baseA.reindex(PRIMARY_FOLDS).values)
        pt = rel.mean()
        lo, hi = two_stage_bootstrap_rel_improve(raw, baseA, ratio, PRIMARY_FOLDS)
        macro_pts.append(pt)
        macro_lo.append(lo)
        macro_hi.append(hi)
        ci_rows.append({
            "ratio": ratio, "macro_rel_improve_pct": round(pt, 2),
            "ci_2.5pct": round(lo, 2), "ci_97.5pct": round(hi, 2),
            "significant_after_holm": bool(holm_sig.get(ratio, False)),
        })

    ci_df = pd.DataFrame(ci_rows)
    out_csv = os.path.join(RESULTS_DIR, "bootstrap_ci_loco_all_ratios.csv")
    ci_df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")
    print(ci_df.to_string(index=False))

    x = np.arange(len(ALL_RATIOS))
    fig, ax = plt.subplots(figsize=(3.5, 3.1))

    ax.fill_between(x, macro_lo, macro_hi, color="#555555", alpha=0.13, zorder=1,
                    label="Macro 95% CI")
    for f in PRIMARY_FOLDS:
        ys = [100 * (1.0 - raw[(raw.fold == f) & (raw.ratio == r)]["calib"].mean() / baseA[f])
              for r in ALL_RATIOS]
        ax.plot(x, ys, marker="o", ms=3.1, lw=1.0, color=FOLD_COLORS[f], alpha=0.75,
                label=f, zorder=2)

    ax.plot(x, macro_pts, marker="D", ms=5.2, lw=2.2, color="#222222",
            label="Macro (5 folds)", zorder=4, markeredgecolor="white", markeredgewidth=0.6)
    # Thin reference line at 0%: above it a ratio helps, below it it hurts.
    ax.axhline(0, color="#444444", lw=0.9, ls="--", zorder=1)
    ax.text(len(ALL_RATIOS) - 0.55, 0.6, "no improvement", fontsize=5.3, color="#444444",
            ha="right", va="bottom", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in ALL_RATIOS], fontsize=7.5)
    ax.set_xlabel("Synthetic ratio (% of six-competition real rare-training count)", fontsize=6.4)
    ax.set_ylabel("Relative calibration-error\nimprovement (%)", fontsize=7.5)
    ax.set_title("Rare-scenario cold-start dose-response\nby held-out competition", fontsize=8.5,
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.5, loc="lower left", ncol=2, framealpha=0.92)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.20)
    sig_marks = ["*" if holm_sig.get(r, False) else "n.s." for r in ALL_RATIOS]
    for i, m in enumerate(sig_marks):
        ax.text(x[i], ymax + (ymax - ymin) * 0.055, m, ha="center", va="bottom",
                fontsize=6.4, color="#333333", fontweight="bold" if m == "*" else "normal")
    # Holm row caption sits above the marks, inside the axes, so it cannot
    # collide with the y-axis tick labels.
    ax.text(0.5, 0.985, "Holm-corrected significance vs. fair baseline",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.6,
            color="#333333", style="italic")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_loco_dose_response.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ════════════════════════════════════════════════════════════════════════
# B2 — LOCO study-design pipeline diagram
# ════════════════════════════════════════════════════════════════════════
def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(3.5, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.0, 12.3)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fontsize=6.0, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                zorder=3, weight=weight, linespacing=1.3)

    def arrow(x1, y1, x2, y2, color="#555555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.9, color=color, zorder=1, shrinkA=1.5, shrinkB=1.5))

    box(2.1, 11.15, 5.8, 0.95, "7 competitions\n(43,078 shots; 1,345 rare)", "#eef2f7", "#7f8fa6",
        fontsize=6.3, weight="bold")
    arrow(3.6, 11.15, 2.55, 10.2)
    arrow(6.4, 11.15, 7.45, 10.2)

    box(0.15, 9.1, 4.55, 1.1, "Other 6 competitions:\nreal rare shots", "#e3f0e6", "#55A868", fontsize=6.1)
    box(5.3, 9.1, 4.55, 1.1, "Held-out competition G:\nreal rare shots", "#fbe6e6", "#C44E52", fontsize=6.1)

    arrow(2.4, 9.1, 2.4, 8.15)
    box(0.15, 7.0, 4.55, 1.15, "CTGAN generator\n(never sees G's rare shots)", "#e3f0e6", "#55A868",
        fontsize=6.0)
    arrow(2.4, 7.0, 2.4, 6.1)
    box(0.15, 5.0, 4.55, 1.1, "Synthetic rare shots", "#e3f0e6", "#55A868", fontsize=6.1)

    ax.plot([7.58, 7.58], [9.1, 2.7], color="#C44E52", ls=(0, (3, 2)), lw=1.15, zorder=1)
    ax.text(7.75, 6.0, "held out,\nuntouched by\nany training", fontsize=5.6, color="#C44E52",
            ha="left", va="center", style="italic", linespacing=1.3)
    arrow(7.58, 2.9, 7.58, 2.5, color="#C44E52")

    box(0.15, 3.15, 4.55, 1.25, "Normal (non-rare) shots\nfrom ALL 7 competitions\n(including G)",
        "#eef2f7", "#7f8fa6", fontsize=5.9)
    # "+" rather than a bare arrow: the two boxes are the two UNION components
    # of the classifier's training set, not sequential stages.
    ax.text(2.4, 4.70, "+", ha="center", va="center", fontsize=10, color="#555555",
            fontweight="bold", zorder=3)
    arrow(2.4, 3.15, 2.4, 2.5)

    box(0.15, 1.25, 4.55, 1.25, "XGBoost classifier\nZERO real rare shots\nof any kind", "#f3eefa",
        "#8172B3", fontsize=5.9)
    box(5.3, 1.25, 4.55, 1.25, "Evaluate on G's\nreal rare shots", "#fbe6e6", "#C44E52",
        fontsize=6.2, weight="bold")
    arrow(4.7, 1.87, 5.3, 1.87)

    ax.text(5.0, 0.45, "Repeated for each of the 5 primary held-out competitions",
            ha="center", va="center", fontsize=5.7, color="#444444", style="italic")

    ax.set_title("Leave-one-competition-out (LOCO)\nrare-scenario cold-start design",
                 fontsize=8.3, fontweight="bold", pad=4)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_loco_pipeline_diagram.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ════════════════════════════════════════════════════════════════════════
# E — cleanup: LOCO per-fold forest plot
# ════════════════════════════════════════════════════════════════════════
def make_per_fold_forest():
    raw_all, raw, per_fold, baseA = _load_loco()
    supp = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_E_loco_supplementary.csv"))
    macro_ci = pd.read_csv(os.path.join(RESULTS_DIR, "bootstrap_ci_loco_calib_5pct.csv")).iloc[0]

    labels, points, lo_err, hi_err, colors = [], [], [], [], []
    for f in PRIMARY_FOLDS:
        runs = raw[(raw.fold == f) & (raw.ratio == HEADLINE_RATIO)]["calib"].values
        rel = 100 * (1.0 - runs / baseA[f])
        pt = rel.mean()
        lo, hi = bootstrap_ci_1d(rel)
        labels.append(f)
        points.append(pt)
        lo_err.append(pt - lo)
        hi_err.append(hi - pt)
        colors.append(C_C)

    labels.append("Macro\n(5 folds)")
    points.append(float(macro_ci["point_estimate_pct"]))
    lo_err.append(float(macro_ci["point_estimate_pct"]) - float(macro_ci["ci_2.5pct"]))
    hi_err.append(float(macro_ci["ci_97.5pct"]) - float(macro_ci["point_estimate_pct"]))
    colors.append("#222222")

    supp_row = supp[supp.ratio == HEADLINE_RATIO].iloc[0]
    supp_runs = raw_all[(raw_all.fold == supp_row["fold"]) &
                        (raw_all.ratio == HEADLINE_RATIO)]["calib"].values
    supp_rel = 100 * (1.0 - supp_runs / float(supp_row["baseA_calib_err"]))
    supp_pt = supp_rel.mean()
    supp_lo, supp_hi = bootstrap_ci_1d(supp_rel)
    labels.append("WC+Euro")
    points.append(supp_pt)
    lo_err.append(supp_pt - supp_lo)
    hi_err.append(supp_hi - supp_pt)
    colors.append("#9e9e9e")

    y_pos = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    for y, pt, le, he, c, lbl in zip(y_pos, points, lo_err, hi_err, colors, labels):
        is_macro = "Macro" in lbl
        is_supp = lbl == "WC+Euro"
        marker = "D" if is_macro else ("s" if is_supp else "o")
        ax.errorbar(pt, y, xerr=[[le], [he]], fmt=marker, color=c, ecolor=c,
                    capsize=3, capthick=1.1, elinewidth=1.3,
                    ms=7.2 if is_macro else 5.8, alpha=0.5 if is_supp else 0.95,
                    markeredgecolor="black" if is_macro else "none",
                    markeredgewidth=0.8 if is_macro else 0)

    ax.axvline(0, color="black", ls="-", lw=0.9)
    ax.axhline(0.5, color="#bbbbbb", ls=":", lw=1.0)

    ax.text(points[-1] + hi_err[-1] + 4.0, -0.30,
            "Supplementary - excluded\nfrom primary aggregate\n(n=61, 4 goals)",
            fontsize=5.4, color="#666666", va="bottom", ha="left", style="italic",
            linespacing=1.25)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_xlabel("Relative calibration-error improvement (%)", fontsize=7.8)
    ax.set_title("Rare-scenario cold-start calibration\nimprovement by competition (5% ratio)",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7.2)
    ax.grid(axis="x", color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(right=max(np.array(points) + np.array(hi_err)) + 26)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_loco_per_fold.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ════════════════════════════════════════════════════════════════════════
# E — cleanup: 8-split CTGAN-vs-SMOTE
# ════════════════════════════════════════════════════════════════════════
def make_splitdist():
    per_split = pd.read_csv(os.path.join(RESULTS_DIR, "multiseed_valtest_per_split_matchgrouped.csv"))
    head = pd.read_csv(os.path.join(RESULTS_DIR, "multiseed_valtest_headtohead_matchgrouped.csv"))
    perm = pd.read_csv(os.path.join(RESULTS_DIR, "multisplit_permutation_test.csv"))
    perm_cs = perm[perm.comparison == "CTGAN vs SMOTE"].iloc[0]

    splits = sorted(per_split["split_id"].unique())
    baseline = per_split[per_split.arm == "Baseline A"].set_index("split_id")["test_log_loss_mean"]
    ctgan = per_split[per_split.arm == "CTGAN"].set_index("split_id")["test_log_loss_mean"]
    smote = per_split[per_split.arm == "SMOTE"].set_index("split_id")["test_log_loss_mean"]
    winner = head.set_index("split_id")["winner"]

    x = np.arange(1, len(splits) + 1)

    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    for i, s in enumerate(splits):
        line_color = C_C if winner[s] == "CTGAN" else C_SYNTH
        ax.plot([x[i], x[i]], [ctgan[s], smote[s]], color=line_color, lw=1.5, alpha=0.62, zorder=1)
        ax.scatter(x[i], baseline[s], marker="_", s=190, color="#999999", linewidths=1.7, zorder=2)

    ax.scatter(x, [ctgan[s] for s in splits], marker="o", s=38, color=C_C,
               label="CTGAN (val.-selected)", edgecolor="black", linewidth=0.4, zorder=3)
    ax.scatter(x, [smote[s] for s in splits], marker="s", s=38, color=C_SYNTH,
               label="SMOTE (val.-selected)", edgecolor="black", linewidth=0.4, zorder=3)
    ax.scatter([], [], marker="_", s=190, color="#999999", linewidths=1.7, label="Baseline A")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in splits], fontsize=7.5)
    ax.set_ylabel("Held-out test log loss", fontsize=8)
    ax.set_xlabel("Independent match-grouped split", fontsize=8)
    ax.set_title("CTGAN vs. SMOTE across 8 independent\nmatch-grouped splits", fontsize=8.5,
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=7.2)
    ax.grid(axis="y", color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, loc="upper left", framealpha=0.93)

    n_c = int((winner == "CTGAN").sum())
    n_s = int((winner == "SMOTE").sum())
    mean_diff = float(perm_cs["mean_log_loss_diff"])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - (ymax - ymin) * 0.16, ymax)
    ax.text(0.985, 0.02,
            f"CTGAN {n_c}/8 vs. SMOTE {n_s}/8    mean diff. {mean_diff:+.4f}\n"
            f"paired t-test p=0.79    exact permutation p={perm_cs['exact_permutation_p']:.2f}",
            transform=ax.transAxes, fontsize=5.8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc", lw=0.7))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_splitdist_ctgan_smote.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    plt.rcParams["mathtext.default"] = "regular"
    make_dose_response()
    make_pipeline_diagram()
    make_per_fold_forest()
    make_splitdist()
