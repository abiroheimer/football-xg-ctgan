"""
14_fig8_smote_comparison.py — Step 4 Fig 3: Ablation D, CTGAN vs SMOTE as
separate series on the same axes, across all 18 ratios and all four metrics
(log loss, Brier, AUC, calibration error), mean +/- std over N=5 runs each.

Plotting only — reads results/ablation_D_mean_std.csv and
results/smote_ablation_D_mean_std.csv, both already produced by Step 3.

Run standalone:
    python 14_fig8_smote_comparison.py

Input : results/ablation_D_mean_std.csv, results/smote_ablation_D_mean_std.csv,
        results/baseline_metrics.pkl
Output: figures/fig8_smote_vs_ctgan.png
"""
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from config import ABLATION_D_FILE, BASELINE_METRICS_FILE, C_ALL, C_C, C_SYNTH, FIG_DIR, RESULTS_DIR

SMOTE_D_FILE = os.path.join(RESULTS_DIR, "smote_ablation_D_mean_std.csv")

if __name__ == "__main__":
    ctgan_df = pd.read_csv(ABLATION_D_FILE)
    smote_df = pd.read_csv(SMOTE_D_FILE)
    with open(BASELINE_METRICS_FILE, "rb") as f:
        bmetrics = pickle.load(f)
    res_A_rare = bmetrics["res_A_rare"]

    pct = [f"{r:.0%}" for r in ctgan_df["ratio"]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Figure 8 — Ablation D: CTGAN vs SMOTE (mean±std, n=5 runs, 269-shot rare test)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()
    plot_cfg = [
        ("log_loss_mean", "Log Loss (↓)", res_A_rare["log_loss"], "min"),
        ("brier_mean", "Brier Score (↓)", res_A_rare["brier"], "min"),
        ("auc_mean", "AUC-ROC (↑)", res_A_rare["auc"], "max"),
        ("calib_err_mean", "Calibration Error (↓)", res_A_rare["calib_err"], "min"),
    ]
    for ax, (mc_col, title, base_val, rule) in zip(axes, plot_cfg):
        std_col = mc_col.replace("_mean", "_std")
        ax.errorbar(pct, ctgan_df[mc_col], yerr=ctgan_df[std_col], marker="o", color=C_C, lw=2,
                     capsize=3, capthick=1.2, ms=6, label="CTGAN (Model C)", alpha=0.9)
        ax.errorbar(pct, smote_df[mc_col], yerr=smote_df[std_col], marker="s", color=C_SYNTH, lw=2,
                     capsize=3, capthick=1.2, ms=6, label="SMOTE", alpha=0.9)
        ax.axhline(base_val, color=C_ALL, ls="--", lw=1.5, label="Baseline A")
        best_c = ctgan_df[mc_col].idxmin() if rule == "min" else ctgan_df[mc_col].idxmax()
        best_s = smote_df[mc_col].idxmin() if rule == "min" else smote_df[mc_col].idxmax()
        ax.axvline(pct[best_c], color=C_C, ls=":", lw=1, alpha=0.6)
        ax.axvline(pct[best_s], color=C_SYNTH, ls=":", lw=1, alpha=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Synthetic Ratio")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig8_smote_vs_ctgan.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✔ Saved -> {out_path}")

    print("\nBest ratio per metric:")
    for mc_col, title, base_val, rule in plot_cfg:
        bc = ctgan_df.loc[ctgan_df[mc_col].idxmin() if rule == "min" else ctgan_df[mc_col].idxmax()]
        bs = smote_df.loc[smote_df[mc_col].idxmin() if rule == "min" else smote_df[mc_col].idxmax()]
        print(f"  {title:<22} CTGAN best: {bc['ratio']:.0%} ({bc[mc_col]:.4f})   "
              f"SMOTE best: {bs['ratio']:.0%} ({bs[mc_col]:.4f})   Baseline A: {base_val:.4f}")
