"""
10_results_and_figures.py — Stage 10: final results tables + every figure
(calibration curves, Ablation D/E plots, tradeoff curve, model comparison,
feature importance). Faithful to the original notebook's cells 14 & 15
(fig7-fig10, plus the fuller fig8/8B/9 versions).

Run standalone:
    python 10_results_and_figures.py

Input : everything produced by stages 03-09
Output: results/main_results.csv, results/main_results_final.csv,
        figures/fig7_calibration.png, fig8_ablation_D.png,
        fig8_ablation_D_full.png, fig8B_tradeoff.png, fig8E_ablation_E.png,
        fig9_comparison.png, fig9_best_models.png, fig10_importance.png
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from config import (ABLATION_D_FILE, ABLATION_E_FILE, BASELINE_METRICS_FILE, BASELINE_MODELS_FILE,
                     C_ALL, C_B, C_C, C_E, C_RARE, FIG_DIR, MODEL_C_FILE, MODEL_C_METRICS_FILE,
                     RESULTS_DIR, SPLIT_FILE, XG_FEATURES)
from model_utils import evaluate


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


if __name__ == "__main__":
    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)
    with open(BASELINE_MODELS_FILE, "rb") as f:
        baselines = pickle.load(f)
    with open(BASELINE_METRICS_FILE, "rb") as f:
        bmetrics = pickle.load(f)
    with open(MODEL_C_FILE, "rb") as f:
        model_C = pickle.load(f)
    with open(MODEL_C_METRICS_FILE, "rb") as f:
        cmetrics = pickle.load(f)

    model_A, model_B = baselines["model_A"], baselines["model_B"]
    res_A_full, res_A_rare, res_B_rare = bmetrics["res_A_full"], bmetrics["res_A_rare"], bmetrics["res_B_rare"]
    res_C_full, res_C_rare, DEFAULT_RATIO = cmetrics["res_C_full"], cmetrics["res_C_rare"], cmetrics["ratio"]

    X_test, y_test = split["X_test"], split["y_test"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    model_df = split["model_df"]

    all_rare_df = model_df[model_df["is_rare"] == 1].copy()
    X_all_rare = all_rare_df[XG_FEATURES].values
    y_all_rare = all_rare_df["is_goal"].values
    res_A_allrare = evaluate(model_A, X_all_rare, y_all_rare, "A — all rare")

    abl_df = pd.read_csv(ABLATION_D_FILE)
    abl_E_df = pd.read_csv(ABLATION_E_FILE)
    N_RUNS, N_RUNS_E = 5, 5  # fixed by stages 08/09

    # ═══════════════════════════════════════════════════════════
    # Main results table (all conditions)
    # ═══════════════════════════════════════════════════════════
    results = pd.DataFrame([
        {"Model": "Baseline A (full test)", **{k: res_A_full[k] for k in ["log_loss", "brier", "auc", "calib_err", "n"]}},
        {"Model": "Baseline A (rare 20%)", **{k: res_A_rare[k] for k in ["log_loss", "brier", "auc", "calib_err", "n"]}},
        {"Model": "Baseline A (all rare)", **{k: res_A_allrare[k] for k in ["log_loss", "brier", "auc", "calib_err", "n"]}},
        {"Model": "Baseline B (rare 20%)", **{k: res_B_rare[k] for k in ["log_loss", "brier", "auc", "calib_err", "n"]}},
        {"Model": f"Model C {DEFAULT_RATIO:.0%} (rare 20%)", **{k: res_C_rare[k] for k in ["log_loss", "brier", "auc", "calib_err", "n"]}},
        {"Model": f"Model C {DEFAULT_RATIO:.0%} (full)", **{k: res_C_full[k] for k in ["log_loss", "brier", "auc", "calib_err", "n"]}},
    ])
    results.to_csv(os.path.join(RESULTS_DIR, "main_results.csv"), index=False)
    print("MAIN RESULTS TABLE")
    print("=" * 70)
    print(results[["Model", "log_loss", "brier", "auc", "calib_err", "n"]].to_string(index=False))
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # Figure 7 — Calibration curves
    # ═══════════════════════════════════════════════════════════
    def plot_cal(ax, model, X, y, label, color, ls="-", n_bins=6):
        if len(y) < 10 or len(np.unique(y)) < 2:
            return
        p = model.predict_proba(X)[:, 1]
        fp, mp = calibration_curve(y, p, n_bins=min(n_bins, max(3, len(y) // 15)), strategy="quantile")
        ax.plot(mp, fp, marker="o", color=color, lw=2, ls=ls, label=label, ms=5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 7 — Calibration Curves", fontsize=13, fontweight="bold")
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    plot_cal(axes[0], model_A, X_test, y_test, "Baseline A", C_ALL)
    plot_cal(axes[0], model_C, X_test, y_test, f"Model C ({DEFAULT_RATIO:.0%})", C_C, ls="--")
    axes[0].set_title("Full Test Set", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Mean Predicted xG")
    axes[0].set_ylabel("Actual Conversion")
    plot_cal(axes[1], model_A, X_rare_test, y_rare_test, "Baseline A", C_ALL, n_bins=5)
    plot_cal(axes[1], model_B, X_rare_test, y_rare_test, "Baseline B", C_B, n_bins=5)
    plot_cal(axes[1], model_C, X_rare_test, y_rare_test, f"Model C ({DEFAULT_RATIO:.0%})", C_C, ls="--", n_bins=5)
    axes[1].set_title("Rare Test Slice (20%)", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Mean Predicted xG")
    axes[1].set_ylabel("Actual Conversion")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_calibration.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Figure 8 — Ablation D, mean±std (1x3 quick version)
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Figure 8 — Ablation D: Synthetic Ratio vs Metrics (mean±std, n={N_RUNS} runs)",
                 fontsize=12, fontweight="bold")
    pct = [f"{r:.0%}" for r in abl_df["ratio"]]
    for ax, (mc_col, base_val, ylabel) in zip(axes, [
        ("log_loss_mean", res_A_rare["log_loss"], "Log Loss (↓)"),
        ("brier_mean", res_A_rare["brier"], "Brier Score (↓)"),
        ("calib_err_mean", res_A_rare["calib_err"], "Calibration Error (↓)"),
    ]):
        std_col = mc_col.replace("_mean", "_std")
        ax.errorbar(pct, abl_df[mc_col], yerr=abl_df[std_col], marker="o", color=C_C, lw=2,
                     capsize=4, capthick=1.5, ms=7, label="Model C")
        ax.axhline(base_val, color=C_ALL, ls="--", lw=1.5, label="Baseline A")
        best = abl_df[mc_col].idxmin()
        ax.axvline(pct[best], color=C_RARE, ls=":", lw=1.5)
        ax.set_title(ylabel, fontweight="bold")
        ax.set_xlabel("Synthetic Ratio")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_ablation_D.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Figure 8 (full) — Ablation D, all four metrics, 2x2
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Figure 8 — Ablation D: Synthetic Ratio vs Metrics (mean±std, n={N_RUNS} runs)",
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
        ax.errorbar(pct, abl_df[mc_col], yerr=abl_df[std_col], marker="o", color=C_C, lw=2,
                     capsize=4, capthick=1.5, ms=7, label="Model C")
        ax.axhline(base_val, color=C_ALL, ls="--", lw=1.5, label="Baseline A")
        best = abl_df[mc_col].idxmin() if rule == "min" else abl_df[mc_col].idxmax()
        ax.axvline(pct[best], color=C_RARE, ls=":", lw=1.5, label=f"Best {pct[best]}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Synthetic Ratio")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_ablation_D_full.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Figure 8B — Tradeoff curve
    # ═══════════════════════════════════════════════════════════
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(abl_df))
    labels = [f"{r:.0%}" for r in abl_df["ratio"]]
    ax1.plot(x, abl_df["log_loss_mean"], marker="o", lw=2, label="Log Loss ↓")
    ax1.plot(x, abl_df["calib_err_mean"], marker="s", lw=2, label="Calibration ↓")
    ax1.set_ylabel("Loss / Calibration")
    ax1.set_xlabel("Synthetic Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(x, abl_df["auc_mean"], marker="^", lw=2, linestyle="--", label="AUC ↑")
    ax2.set_ylabel("AUC")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("Figure 8B — Tradeoff Between Predictive Performance and Calibration", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8B_tradeoff.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Figure 8E — Ablation E
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Figure 8E — Ablation E: Zero-Shot Rare Eval (ALL {len(X_all_rare)} rare shots, n={N_RUNS_E} runs)",
                 fontsize=11, fontweight="bold")
    pct_e = [f"{r:.0%}" for r in abl_E_df["ratio"]]
    for ax, (mc_col, base_val, ylabel) in zip(axes, [
        ("log_loss_mean", res_A_allrare["log_loss"], "Log Loss (↓)"),
        ("brier_mean", res_A_allrare["brier"], "Brier Score (↓)"),
        ("auc_mean", res_A_allrare["auc"], "AUC-ROC (↑)"),
    ]):
        std_col = mc_col.replace("_mean", "_std")
        ax.errorbar(pct_e, abl_E_df[mc_col], yerr=abl_E_df[std_col], marker="s", color=C_E, lw=2,
                     capsize=4, capthick=1.5, ms=7, label="Model E (normal+synth)")
        ax.axhline(base_val, color=C_ALL, ls="--", lw=1.5, label="Baseline A")
        ax.set_title(ylabel, fontweight="bold")
        ax.set_xlabel("Synthetic Ratio (% of rare_train)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8E_ablation_E.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Figure 9 — Bar comparison (A vs B vs C)
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Figure 9 — Model Comparison on Rare Test Slice", fontsize=13, fontweight="bold")
    mdl_labels = ["Baseline A", "Baseline B", f"Model C\n({DEFAULT_RATIO:.0%})"]
    colors_bar = [C_ALL, C_B, C_C]
    for ax, (vals, ylabel) in zip(axes, [
        ([res_A_rare["log_loss"], res_B_rare["log_loss"], res_C_rare["log_loss"]], "Log Loss (↓)"),
        ([res_A_rare["brier"], res_B_rare["brier"], res_C_rare["brier"]], "Brier Score (↓)"),
        ([res_A_rare["calib_err"], res_B_rare["calib_err"], res_C_rare["calib_err"]], "Calibration Error (↓)"),
    ]):
        bars = ax.bar(mdl_labels, vals, color=colors_bar, alpha=0.85, edgecolor="white")
        bars[vals.index(min(vals))].set_edgecolor("black")
        bars[vals.index(min(vals))].set_linewidth(2.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(ylabel, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Figure 9 (best-model variant) — representative augmentation regimes
    # ═══════════════════════════════════════════════════════════
    def abl_val(ratio_val, col):
        rows = abl_df.loc[abl_df["ratio"] == ratio_val, col].values
        return rows[0] if len(rows) else np.nan

    compare_models = {
        "Baseline A": {"log_loss": res_A_rare["log_loss"], "brier": res_A_rare["brier"],
                        "auc": res_A_rare["auc"], "calib": res_A_rare["calib_err"]},
        "80%": {"log_loss": abl_val(0.80, "log_loss_mean"), "brier": abl_val(0.80, "brier_mean"),
                 "auc": abl_val(0.80, "auc_mean"), "calib": abl_val(0.80, "calib_err_mean")},
        "500%": {"log_loss": abl_val(5.00, "log_loss_mean"), "brier": abl_val(5.00, "brier_mean"),
                  "auc": abl_val(5.00, "auc_mean"), "calib": abl_val(5.00, "calib_err_mean")},
        "1000%": {"log_loss": abl_val(10.00, "log_loss_mean"), "brier": abl_val(10.00, "brier_mean"),
                   "auc": abl_val(10.00, "auc_mean"), "calib": abl_val(10.00, "calib_err_mean")},
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 9 — Comparison of Representative Augmentation Regimes", fontsize=13, fontweight="bold")
    axes = axes.flatten()
    metric_titles = [("log_loss", "Log Loss (↓)"), ("brier", "Brier Score (↓)"),
                      ("auc", "AUC-ROC (↑)"), ("calib", "Calibration Error (↓)")]
    for ax, (metric, title) in zip(axes, metric_titles):
        labels = list(compare_models.keys())
        vals = [compare_models[m][metric] for m in labels]
        bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        bars = ax.bar(labels, vals, color=bar_colors[:len(labels)], alpha=0.85)
        best_idx = int(np.argmax(vals)) if metric == "auc" else int(np.argmin(vals))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                     f"{v:.4f}", ha="center", fontsize=9)
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_best_models.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Figure 10 — Feature importance
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 10 — Feature Importance", fontsize=13, fontweight="bold")
    for ax, (model, label, color) in zip(axes, [(model_A, "Baseline A", C_ALL),
                                                   (model_C, f"Model C ({DEFAULT_RATIO:.0%})", C_C)]):
        imp = pd.Series(model.feature_importances_, index=XG_FEATURES).sort_values()
        imp.plot(kind="barh", ax=ax, color=color, alpha=0.85)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig10_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Final paper-style summary table (best ratio per metric from Ablation D)
    # ═══════════════════════════════════════════════════════════
    best_ll_row = abl_df.loc[abl_df["log_loss_mean"].idxmin()]
    best_brier_row = abl_df.loc[abl_df["brier_mean"].idxmin()]
    best_auc_row = abl_df.loc[abl_df["auc_mean"].idxmax()]
    best_calib_row = abl_df.loc[abl_df["calib_err_mean"].idxmin()]

    main_results = pd.DataFrame([
        {"Model": "Baseline A (Rare Test)", "LogLoss": f"{res_A_rare['log_loss']:.4f}",
         "Brier": f"{res_A_rare['brier']:.4f}", "AUC": f"{res_A_rare['auc']:.4f}",
         "Calib": f"{res_A_rare['calib_err']:.4f}", "Notes": "Full real-data baseline"},
        {"Model": "Baseline B (Rare-Only)", "LogLoss": f"{res_B_rare['log_loss']:.4f}",
         "Brier": f"{res_B_rare['brier']:.4f}", "AUC": f"{res_B_rare['auc']:.4f}",
         "Calib": f"{res_B_rare['calib_err']:.4f}", "Notes": "Trained only on real rare shots"},
        {"Model": f"Model C ({best_calib_row['ratio']:.0%})",
         "LogLoss": fmt(best_calib_row["log_loss_mean"], best_calib_row["log_loss_std"]),
         "Brier": fmt(best_calib_row["brier_mean"], best_calib_row["brier_std"]),
         "AUC": fmt(best_calib_row["auc_mean"], best_calib_row["auc_std"]),
         "Calib": fmt(best_calib_row["calib_err_mean"], best_calib_row["calib_err_std"]),
         "Notes": "Best calibration"},
        {"Model": f"Model C ({best_ll_row['ratio']:.0%})",
         "LogLoss": fmt(best_ll_row["log_loss_mean"], best_ll_row["log_loss_std"]),
         "Brier": fmt(best_ll_row["brier_mean"], best_ll_row["brier_std"]),
         "AUC": fmt(best_ll_row["auc_mean"], best_ll_row["auc_std"]),
         "Calib": fmt(best_ll_row["calib_err_mean"], best_ll_row["calib_err_std"]),
         "Notes": "Best probabilistic accuracy"},
        {"Model": f"Model C ({best_auc_row['ratio']:.0%})",
         "LogLoss": fmt(best_auc_row["log_loss_mean"], best_auc_row["log_loss_std"]),
         "Brier": fmt(best_auc_row["brier_mean"], best_auc_row["brier_std"]),
         "AUC": fmt(best_auc_row["auc_mean"], best_auc_row["auc_std"]),
         "Calib": fmt(best_auc_row["calib_err_mean"], best_auc_row["calib_err_std"]),
         "Notes": "Best discrimination"},
    ])
    main_results.to_csv(os.path.join(RESULTS_DIR, "main_results_final.csv"), index=False)

    print("\n" + "=" * 90)
    print("FINAL MAIN RESULTS TABLE")
    print("=" * 90)
    print(main_results[["Model", "LogLoss", "Brier", "AUC", "Calib", "Notes"]].to_string(index=False))
    print("=" * 90)

    print(f"\n✔ All figures saved to {FIG_DIR}")
    print(f"✔ All CSVs saved to {RESULTS_DIR}")
    print("\nPipeline complete.")
