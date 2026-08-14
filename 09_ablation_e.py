"""
09_ablation_e.py — Stage 9: Ablation E — train on normal shots + CTGAN
synthetic rare shots ONLY (zero real rare shots in training), evaluate on
ALL 1,345 rare shots for maximum statistical power.

Run standalone:
    python 09_ablation_e.py

Input : data/split_data.pkl, models/ctgan_model.pkl, models/baseline_models.pkl
Output: results/ablation_E_zero_shot.csv, results/ablation_E_raw_runs.csv
        (raw per-run values — needed by 12_significance_tests.py)
"""
import os
import pickle

import numpy as np
import pandas as pd

from config import (ABLATION_E_FILE, BASELINE_MODELS_FILE, CTGAN_MODEL_FILE, RESULTS_DIR,
                     SEED, SPLIT_FILE, XG_FEATURES)
from model_utils import evaluate, load_ctgan, sample_synthetic, train_xgb

ABLATION_E_RAW_FILE = os.path.join(RESULTS_DIR, "ablation_E_raw_runs.csv")

# 6 ratios (Section 4.2 says "five" in the paper text — see Step 3 Fix 5
# errata; the code and results have always had 6, including 30%).
E_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]  # ratio = n_synthetic / len(rare_train_df)
N_RUNS_E = 5

if __name__ == "__main__":
    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)
    with open(BASELINE_MODELS_FILE, "rb") as f:
        baselines = pickle.load(f)
    ctgan_model = load_ctgan(CTGAN_MODEL_FILE)

    model_A = baselines["model_A"]
    model_df = split["model_df"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    X_norm_train, y_norm_train = split["X_norm_train"], split["y_norm_train"]
    rare_train_df = split["rare_train_df"]

    all_rare_df = model_df[model_df["is_rare"] == 1].copy()
    X_all_rare = all_rare_df[XG_FEATURES].values
    y_all_rare = all_rare_df["is_goal"].values
    print(f"Ablation E test set: ALL {len(X_all_rare):,} rare shots  "
          f"({y_all_rare.sum()} goals = {100 * y_all_rare.mean():.1f}% conv)")
    print(f"  vs standard rare test: {len(X_rare_test)} shots  → "
          f"{len(X_all_rare) / len(X_rare_test):.1f}x more statistical power\n")

    res_A_allrare = evaluate(model_A, X_all_rare, y_all_rare, "A — all rare")
    print(f"Baseline A on all rare: log_loss={res_A_allrare['log_loss']}  "
          f"brier={res_A_allrare['brier']}  AUC={res_A_allrare['auc']}  calib={res_A_allrare['calib_err']}")

    run_results_E = {r: [] for r in E_RATIOS}

    print(f"\nRunning Ablation E: {len(E_RATIOS)} ratios × {N_RUNS_E} runs")
    print(f"Train = normal shots ({len(X_norm_train):,}) + synthetic rare")
    print(f"Test  = ALL {len(X_all_rare):,} rare shots\n")
    print(f"{'Ratio':<8} {'Run':<5} {'LogLoss':<10} {'Brier':<10} {'AUC':<8} {'Calib'}")
    print("-" * 52)

    for ratio in E_RATIOS:
        for run in range(N_RUNS_E):
            np.random.seed(SEED + run + 100)
            try:
                import torch
                torch.manual_seed(SEED + run + 100)
            except Exception:
                pass

            n_synth_e = max(10, int(len(rare_train_df) * ratio))
            s = sample_synthetic(ctgan_model, n_synth_e, rare_train_df)
            Xs = s[XG_FEATURES].values
            ys = s["is_goal"].values

            Xe = np.vstack([X_norm_train, Xs])
            ye = np.concatenate([y_norm_train, ys])

            me = train_xgb(Xe, ye, "")
            re = evaluate(me, X_all_rare, y_all_rare, "")
            if re:
                run_results_E[ratio].append({
                    "ll": re["log_loss"], "brier": re["brier"],
                    "auc": re["auc"], "calib": re["calib_err"], "ece": re["ece"],
                })
                print(f"{ratio:<8.1%} {run + 1:<5} {re['log_loss']:<10.4f} {re['brier']:<10.4f} "
                      f"{re['auc']:<8.4f} {re['calib_err']:.4f}")

    np.random.seed(SEED)  # reset

    raw_rows = []
    for ratio, runs in run_results_E.items():
        for i, r in enumerate(runs):
            raw_rows.append({"ratio": ratio, "run": i, **r})
    pd.DataFrame(raw_rows).to_csv(ABLATION_E_RAW_FILE, index=False)
    print(f"\n✔ Saved raw per-run results → {ABLATION_E_RAW_FILE}")

    print("\n" + "=" * 72)
    print(f"ABLATION E SUMMARY  (n={N_RUNS_E} runs, test=ALL {len(X_all_rare)} rare shots)")
    print("=" * 72)
    print(f"{'Ratio':<8} {'N_synth':<9} {'LogLoss':>14} {'Brier':>14} {'AUC':>14} {'Calib':>14}")
    print("-" * 72)
    print(f"{'BaseA':<8} {'0':<9} "
          f"{res_A_allrare['log_loss']:>6.4f}{'±0':>8}  "
          f"{res_A_allrare['brier']:>6.4f}{'±0':>8}  "
          f"{res_A_allrare['auc']:>6.4f}{'±0':>8}  "
          f"{res_A_allrare['calib_err']:>6.4f}{'±0':>8}")

    abl_E_summary = []
    for ratio in E_RATIOS:
        runs = run_results_E[ratio]
        if not runs:
            continue
        ns = max(10, int(len(rare_train_df) * ratio))
        row = {"ratio": ratio, "n_synth": ns}
        for metric, col in [("ll", "log_loss"), ("brier", "brier"), ("auc", "auc"),
                             ("calib", "calib_err"), ("ece", "ece")]:
            vals = [r[metric] for r in runs]
            row[f"{col}_mean"] = round(np.mean(vals), 4)
            row[f"{col}_std"] = round(np.std(vals), 4)
        abl_E_summary.append(row)
        print(f"{ratio:<8.1%} {ns:<9} "
              f"{row['log_loss_mean']:.4f}±{row['log_loss_std']:.4f}  "
              f"{row['brier_mean']:.4f}±{row['brier_std']:.4f}  "
              f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}  "
              f"{row['calib_err_mean']:.4f}±{row['calib_err_std']:.4f}")

    abl_E_df = pd.DataFrame(abl_E_summary)
    abl_E_df.to_csv(ABLATION_E_FILE, index=False)

    print(f"\n✔ Saved → {ABLATION_E_FILE}")
    print("Next: python 10_results_and_figures.py")
