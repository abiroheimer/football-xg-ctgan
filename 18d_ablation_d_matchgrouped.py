"""
18d_ablation_d_matchgrouped.py — Ablation D (CTGAN arm) re-run on the
match-grouped split. Mirrors 08_ablation_d.py exactly (18 ratios, N=5 runs
each, identical hyperparameters); only the input split/CTGAN model change.

Run standalone:
    python 18d_ablation_d_matchgrouped.py

Input : data/split_data_matchgrouped.pkl, models/ctgan_model_matchgrouped.pkl
Output: results/ablation_D_mean_std_matchgrouped.csv, results/ablation_D_raw_runs_matchgrouped.csv
"""
import os
import pickle

import numpy as np
import pandas as pd

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED
from model_utils import build_aug, evaluate, load_ctgan, train_xgb

SPLIT_MG_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped.pkl")
CTGAN_MODEL_MG_FILE = os.path.join(MODELS_DIR, "ctgan_model_matchgrouped.pkl")
ABLATION_D_MG_FILE = os.path.join(RESULTS_DIR, "ablation_D_mean_std_matchgrouped.csv")
ABLATION_D_RAW_MG_FILE = os.path.join(RESULTS_DIR, "ablation_D_raw_runs_matchgrouped.csv")

N_RUNS = 5
RATIOS = [0.01, 0.05, 0.10, 0.20, 0.40, 0.50, 0.80, 1.00,
          1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 10.00]

if __name__ == "__main__":
    with open(SPLIT_MG_FILE, "rb") as f:
        split = pickle.load(f)
    ctgan_model = load_ctgan(CTGAN_MODEL_MG_FILE)

    X_train, y_train = split["X_train"], split["y_train"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    rare_train_df = split["rare_train_df"]

    run_results = {r: [] for r in RATIOS}

    print(f"Running Ablation D [matchgrouped]: {len(RATIOS)} ratios × {N_RUNS} runs = {len(RATIOS) * N_RUNS} models\n")
    print(f"{'Ratio':<7} {'Run':<5} {'LogLoss':<10} {'Brier':<10} {'AUC':<8} {'Calib'}")
    print("-" * 52)

    for ratio in RATIOS:
        for run in range(N_RUNS):
            np.random.seed(SEED + run)
            try:
                import torch
                torch.manual_seed(SEED + run)
            except Exception:
                pass

            Xa, ya, ns = build_aug(X_train, y_train, rare_train_df, ctgan_model, ratio)
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
    pd.DataFrame(raw_rows).to_csv(ABLATION_D_RAW_MG_FILE, index=False)
    print(f"\n✔ Saved raw per-run results → {ABLATION_D_RAW_MG_FILE}")

    print("\n" + "=" * 72)
    print(f"ABLATION D SUMMARY [matchgrouped]  (n={N_RUNS} runs each)  — test set: {len(y_rare_test)} rare shots")
    print("=" * 72)
    print(f"{'Ratio':<8} {'N_synth':<9} {'LogLoss':>14} {'Brier':>14} {'AUC':>14} {'Calib':>14}")
    print("-" * 72)

    abl_summary = []
    for ratio in RATIOS:
        runs = run_results[ratio]
        if not runs:
            continue
        ns = max(10, int(len(rare_train_df) * ratio))
        row = {"ratio": ratio, "n_synth": ns}
        for metric, col in [("ll", "log_loss"), ("brier", "brier"), ("auc", "auc"),
                             ("calib", "calib_err"), ("ece", "ece")]:
            vals = [r[metric] for r in runs]
            row[f"{col}_mean"] = round(np.mean(vals), 4)
            row[f"{col}_std"] = round(np.std(vals), 4)
        abl_summary.append(row)
        print(f"{ratio:<8.0%} {ns:<9} "
              f"{row['log_loss_mean']:.4f}±{row['log_loss_std']:.4f}  "
              f"{row['brier_mean']:.4f}±{row['brier_std']:.4f}  "
              f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}  "
              f"{row['calib_err_mean']:.4f}±{row['calib_err_std']:.4f}")

    abl_df = pd.DataFrame(abl_summary)
    abl_df.to_csv(ABLATION_D_MG_FILE, index=False)

    best = abl_df.loc[abl_df["log_loss_mean"].idxmin()]
    print(f"\nBest ratio (mean log_loss): {best['ratio']:.0%}  ({best['log_loss_mean']:.4f} ± {best['log_loss_std']:.4f})")

    print(f"\n✔ Saved → {ABLATION_D_MG_FILE}")
    print("Next: python 18e_smote_ablation_d_matchgrouped.py")
