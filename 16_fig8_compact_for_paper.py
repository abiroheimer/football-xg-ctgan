"""
16_fig8_compact_for_paper.py — compact single-column (2-panel: log loss +
calibration error) versions of the SMOTE-vs-CTGAN and Ablation E figures,
sized for IEEEtran single-column placement in football.tex. Does not modify
or replace the existing full 4-panel/3-panel versions (fig8_smote_vs_ctgan.png,
fig8E_ablation_E.png) — those stay available for supplementary material.

Plotting only — reads already-computed results/*.csv and results/baseline
pickles, no retraining.

Run standalone:
    python 16_fig8_compact_for_paper.py

Output: figures/fig8_smote_vs_ctgan_compact.png
        figures/fig8E_ablation_E_compact.png
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (ABLATION_D_FILE, ABLATION_E_FILE, BASELINE_MODELS_FILE, C_ALL, C_C, C_E,
                     C_SYNTH, FIG_DIR, RESULTS_DIR, SPLIT_FILE, XG_FEATURES)
from model_utils import evaluate

SMOTE_D_FILE = os.path.join(RESULTS_DIR, "smote_ablation_D_mean_std.csv")

if __name__ == "__main__":
    # ── SMOTE vs CTGAN, compact (log loss + calibration error only) ──
    ctgan_df = pd.read_csv(ABLATION_D_FILE)
    smote_df = pd.read_csv(SMOTE_D_FILE)

    with open(BASELINE_MODELS_FILE, "rb") as f:
        baselines = pickle.load(f)
    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)

    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    res_A_rare = evaluate(baselines["model_A"], X_rare_test, y_rare_test, "Baseline A")

    pct = [f"{r:.0%}" for r in ctgan_df["ratio"]]

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.4))
    plot_cfg = [
        ("log_loss_mean", "Log Loss (↓)", res_A_rare["log_loss"], "min"),
        ("calib_err_mean", "Calibration Error (↓)", res_A_rare["calib_err"], "min"),
    ]
    for ax, (mc_col, title, base_val, rule) in zip(axes, plot_cfg):
        std_col = mc_col.replace("_mean", "_std")
        ax.errorbar(pct, ctgan_df[mc_col], yerr=ctgan_df[std_col], marker="o", color=C_C, lw=1.3,
                     capsize=2, capthick=1, ms=3.5, label="CTGAN", alpha=0.9)
        ax.errorbar(pct, smote_df[mc_col], yerr=smote_df[std_col], marker="s", color=C_SYNTH, lw=1.3,
                     capsize=2, capthick=1, ms=3.5, label="SMOTE", alpha=0.9)
        ax.axhline(base_val, color=C_ALL, ls="--", lw=1, label="Baseline A")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.tick_params(axis="x", rotation=60, labelsize=5.5)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(fontsize=5.5, loc="best")
    axes[-1].set_xlabel("Synthetic Ratio", fontsize=7)
    plt.tight_layout()
    out1 = os.path.join(FIG_DIR, "fig8_smote_vs_ctgan_compact.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out1}")

    # ── Ablation E, compact (log loss + calibration error only) ──
    abl_E_df = pd.read_csv(ABLATION_E_FILE)
    model_df = split["model_df"]
    all_rare_df = model_df[model_df["is_rare"] == 1]
    X_all_rare = all_rare_df[XG_FEATURES].values
    y_all_rare = all_rare_df["is_goal"].values
    res_A_allrare = evaluate(baselines["model_A"], X_all_rare, y_all_rare, "Baseline A (all rare)")

    pct_e = [f"{r:.0%}" for r in abl_E_df["ratio"]]

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.4))
    plot_cfg_e = [
        ("log_loss_mean", "Log Loss (↓)", res_A_allrare["log_loss"]),
        ("calib_err_mean", "Calibration Error (↓)", res_A_allrare["calib_err"]),
    ]
    for ax, (mc_col, title, base_val) in zip(axes, plot_cfg_e):
        std_col = mc_col.replace("_mean", "_std")
        ax.errorbar(pct_e, abl_E_df[mc_col], yerr=abl_E_df[std_col], marker="s", color=C_E, lw=1.3,
                     capsize=2, capthick=1, ms=3.5, label="Zero-shot", alpha=0.9)
        ax.axhline(base_val, color=C_ALL, ls="--", lw=1, label="Baseline A")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.tick_params(axis="x", rotation=0, labelsize=6.5)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(fontsize=5.5, loc="best")
    axes[-1].set_xlabel("Synthetic Ratio (% of rare_train)", fontsize=7)
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "fig8E_ablation_E_compact.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out2}")
