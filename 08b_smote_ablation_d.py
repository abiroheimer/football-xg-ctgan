"""
08b_smote_ablation_d.py — Stage 8b (Step 3 Fix 1): SMOTE comparison arm for
Ablation D. Same 18 ratios, N=5 runs each, same XGBoost hyperparameters, same
269-shot rare test slice as 08_ablation_d.py — the only thing that changes is
the augmentation method (SMOTENC instead of CTGAN), so this is directly
comparable to results/ablation_D_mean_std.csv.

Run standalone:
    python 08b_smote_ablation_d.py

Input : data/split_data.pkl
Output: results/smote_ablation_D_mean_std.csv, results/smote_ablation_D_raw_runs.csv
"""
import os
import pickle

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SEED, SPLIT_FILE
from model_utils import build_smote_aug, evaluate, train_xgb

SMOTE_D_FILE = os.path.join(RESULTS_DIR, "smote_ablation_D_mean_std.csv")
SMOTE_D_RAW_FILE = os.path.join(RESULTS_DIR, "smote_ablation_D_raw_runs.csv")

N_RUNS = 5
RATIOS = [0.01, 0.05, 0.10, 0.20, 0.40, 0.50, 0.80, 1.00,
          1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 10.00]

if __name__ == "__main__":
    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)

    X_train, y_train = split["X_train"], split["y_train"]
    X_rare_train, y_rare_train = split["X_rare_train"], split["y_rare_train"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]

    run_results = {r: [] for r in RATIOS}

    print(f"Running SMOTE Ablation D: {len(RATIOS)} ratios x {N_RUNS} runs = {len(RATIOS) * N_RUNS} models\n")
    print(f"{'Ratio':<7} {'Run':<5} {'LogLoss':<10} {'Brier':<10} {'AUC':<8} {'Calib'}")
    print("-" * 52)

    for ratio in RATIOS:
        for run in range(N_RUNS):
            seed_r = SEED + run
            np.random.seed(seed_r)

            Xa, ya, ns = build_smote_aug(X_train, y_train, X_rare_train, y_rare_train, ratio, seed=seed_r)
            mr = train_xgb(Xa, ya, "")
            rr = evaluate(mr, X_rare_test, y_rare_test, "")
            if rr:
                run_results[ratio].append({
                    "ll": rr["log_loss"], "brier": rr["brier"],
                    "auc": rr["auc"], "calib": rr["calib_err"], "ece": rr["ece"],
                })
                print(f"{ratio:<7.0%} {run + 1:<5} {rr['log_loss']:<10.4f} {rr['brier']:<10.4f} "
                      f"{rr['auc']:<8.4f} {rr['calib_err']:.4f}")

    np.random.seed(SEED)  # reset

    raw_rows = []
    for ratio, runs in run_results.items():
        for i, r in enumerate(runs):
            raw_rows.append({"ratio": ratio, "run": i, **r})
    pd.DataFrame(raw_rows).to_csv(SMOTE_D_RAW_FILE, index=False)
    print(f"\n✔ Saved raw per-run results → {SMOTE_D_RAW_FILE}")

    print("\n" + "=" * 72)
    print(f"SMOTE ABLATION D SUMMARY  (n={N_RUNS} runs each)  — test set: {len(y_rare_test)} rare shots")
    print("=" * 72)
    print(f"{'Ratio':<8} {'N_synth':<9} {'LogLoss':>14} {'Brier':>14} {'AUC':>14} {'Calib':>14}")
    print("-" * 72)

    summary = []
    for ratio in RATIOS:
        runs = run_results[ratio]
        if not runs:
            continue
        ns = max(10, int(len(X_rare_train) * ratio))
        row = {"ratio": ratio, "n_synth": ns}
        for metric, col in [("ll", "log_loss"), ("brier", "brier"), ("auc", "auc"),
                             ("calib", "calib_err"), ("ece", "ece")]:
            vals = [r[metric] for r in runs]
            row[f"{col}_mean"] = round(np.mean(vals), 4)
            row[f"{col}_std"] = round(np.std(vals), 4)
        summary.append(row)
        print(f"{ratio:<8.0%} {ns:<9} "
              f"{row['log_loss_mean']:.4f}±{row['log_loss_std']:.4f}  "
              f"{row['brier_mean']:.4f}±{row['brier_std']:.4f}  "
              f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}  "
              f"{row['calib_err_mean']:.4f}±{row['calib_err_std']:.4f}")

    df = pd.DataFrame(summary)
    df.to_csv(SMOTE_D_FILE, index=False)

    best = df.loc[df["log_loss_mean"].idxmin()]
    print(f"\nBest ratio (mean log_loss): {best['ratio']:.0%}  ({best['log_loss_mean']:.4f} ± {best['log_loss_std']:.4f})")

    print(f"\n✔ Saved → {SMOTE_D_FILE}")
