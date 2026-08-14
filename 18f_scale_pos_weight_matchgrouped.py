"""
18f_scale_pos_weight_matchgrouped.py — scale_pos_weight fixed-vs-variable
sensitivity check (Section V-F) re-run on the match-grouped split, at the
same representative ratios (80%, 300%, 1000%) for both CTGAN and SMOTE arms.
Mirrors Part A of 15_scale_pos_weight_and_pressure_parity.py exactly; only
the input split/CTGAN model change. (Part B, pressure_index parity, is not
split-dependent in a way the task asked to re-run, so it's omitted here.)

Run standalone:
    python 18f_scale_pos_weight_matchgrouped.py

Input : data/split_data_matchgrouped.pkl, models/ctgan_model_matchgrouped.pkl,
        results/baseline_metrics_matchgrouped.pkl
Output: results/scale_pos_weight_sensitivity_matchgrouped.csv,
        results/scale_pos_weight_sensitivity_raw_matchgrouped.csv
"""
import os
import pickle

import numpy as np
import pandas as pd

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED
from model_utils import build_aug, build_smote_aug, evaluate, load_ctgan, train_xgb

SPLIT_MG_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped.pkl")
CTGAN_MODEL_MG_FILE = os.path.join(MODELS_DIR, "ctgan_model_matchgrouped.pkl")
BASELINE_MODELS_MG_FILE = os.path.join(MODELS_DIR, "baseline_models_matchgrouped.pkl")
SENS_MG_FILE = os.path.join(RESULTS_DIR, "scale_pos_weight_sensitivity_matchgrouped.csv")
SENS_RAW_MG_FILE = os.path.join(RESULTS_DIR, "scale_pos_weight_sensitivity_raw_matchgrouped.csv")

RATIOS = [0.80, 3.00, 10.00]  # 80%, 300%, 1000%
N_RUNS = 5

if __name__ == "__main__":
    with open(SPLIT_MG_FILE, "rb") as f:
        split = pickle.load(f)
    with open(BASELINE_MODELS_MG_FILE, "rb") as f:
        baselines = pickle.load(f)
    ctgan_model = load_ctgan(CTGAN_MODEL_MG_FILE)

    X_train, y_train = split["X_train"], split["y_train"]
    X_rare_train, y_rare_train = split["X_rare_train"], split["y_rare_train"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    rare_train_df = split["rare_train_df"]
    res_A_rare = evaluate(baselines["model_A"], X_rare_test, y_rare_test, "Baseline A")

    baseline_pw = float(max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    print(f"Baseline A exact scale_pos_weight [matchgrouped]: {baseline_pw:.4f}  "
          f"(neg={int((y_train == 0).sum())}, pos={int((y_train == 1).sum())})")

    raw_rows = [{
        "arm": "Baseline A", "ratio": 0.0, "weight_mode": "natural (=fixed)", "run": 0,
        "pw_used": round(baseline_pw, 4),
        "log_loss": res_A_rare["log_loss"], "brier": res_A_rare["brier"],
        "auc": res_A_rare["auc"], "calib_err": res_A_rare["calib_err"], "ece": res_A_rare["ece"],
    }]

    print(f"\nRunning {len(RATIOS)} ratios x 2 arms (CTGAN/SMOTE) x 2 weight modes x {N_RUNS} runs "
          f"= {len(RATIOS) * 2 * 2 * N_RUNS} models\n")
    print(f"{'Arm':<8} {'Ratio':<7} {'Weight':<10} {'Run':<4} {'pw':<7} {'LogLoss':<9} {'Brier':<9} {'AUC':<8} {'Calib'}")
    print("-" * 72)

    for ratio in RATIOS:
        for run in range(N_RUNS):
            seed_r = SEED + run
            np.random.seed(seed_r)

            Xc, yc, _ = build_aug(X_train, y_train, rare_train_df, ctgan_model, ratio)
            Xs, ys, _ = build_smote_aug(X_train, y_train, X_rare_train, y_rare_train, ratio, seed=seed_r)

            for arm_name, (Xa, ya) in [("CTGAN", (Xc, yc)), ("SMOTE", (Xs, ys))]:
                natural_pw = max(1.0, (ya == 0).sum() / max((ya == 1).sum(), 1))
                for weight_mode, pw in [("variable", None), ("fixed", baseline_pw)]:
                    m = train_xgb(Xa, ya, "", scale_pos_weight=pw)
                    r = evaluate(m, X_rare_test, y_rare_test, "")
                    pw_used = natural_pw if pw is None else pw
                    raw_rows.append({
                        "arm": arm_name, "ratio": ratio, "weight_mode": weight_mode, "run": run,
                        "pw_used": round(float(pw_used), 4),
                        "log_loss": r["log_loss"], "brier": r["brier"],
                        "auc": r["auc"], "calib_err": r["calib_err"], "ece": r["ece"],
                    })
                    print(f"{arm_name:<8} {ratio:<7.0%} {weight_mode:<10} {run + 1:<4} {pw_used:<7.2f} "
                          f"{r['log_loss']:<9.4f} {r['brier']:<9.4f} {r['auc']:<8.4f} {r['calib_err']:.4f}")

    np.random.seed(SEED)

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(SENS_RAW_MG_FILE, index=False)
    print(f"\n✔ Saved raw per-run results -> {SENS_RAW_MG_FILE}")

    summary_rows = []
    for (arm, ratio, weight_mode), g in raw_df.groupby(["arm", "ratio", "weight_mode"]):
        summary_rows.append({
            "arm": arm, "ratio": ratio, "weight_mode": weight_mode, "n_runs": len(g),
            "pw_used_mean": round(g["pw_used"].mean(), 4),
            "log_loss_mean": round(g["log_loss"].mean(), 4), "log_loss_std": round(g["log_loss"].std(ddof=0), 4),
            "brier_mean": round(g["brier"].mean(), 4), "brier_std": round(g["brier"].std(ddof=0), 4),
            "auc_mean": round(g["auc"].mean(), 4), "auc_std": round(g["auc"].std(ddof=0), 4),
            "calib_err_mean": round(g["calib_err"].mean(), 4), "calib_err_std": round(g["calib_err"].std(ddof=0), 4),
            "ece_mean": round(g["ece"].mean(), 4), "ece_std": round(g["ece"].std(ddof=0), 4),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["arm", "ratio", "weight_mode"]).reset_index(drop=True)
    summary_df.to_csv(SENS_MG_FILE, index=False)

    print("\n" + "=" * 100)
    print("SCALE_POS_WEIGHT SENSITIVITY SUMMARY [matchgrouped]  (mean over N=5 runs; Baseline A is n=1, deterministic)")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print("\n" + "-" * 100)
    print("Ratio-by-ratio: does SMOTE still beat CTGAN, and is 80% still the calib sweet spot, under FIXED weight?")
    print("-" * 100)
    for ratio in RATIOS:
        row_c_var = summary_df[(summary_df.arm == "CTGAN") & (summary_df.ratio == ratio) & (summary_df.weight_mode == "variable")].iloc[0]
        row_c_fix = summary_df[(summary_df.arm == "CTGAN") & (summary_df.ratio == ratio) & (summary_df.weight_mode == "fixed")].iloc[0]
        row_s_var = summary_df[(summary_df.arm == "SMOTE") & (summary_df.ratio == ratio) & (summary_df.weight_mode == "variable")].iloc[0]
        row_s_fix = summary_df[(summary_df.arm == "SMOTE") & (summary_df.ratio == ratio) & (summary_df.weight_mode == "fixed")].iloc[0]
        print(f"\nRatio {ratio:.0%}:")
        print(f"  CTGAN  variable-w: ll={row_c_var.log_loss_mean:.4f} brier={row_c_var.brier_mean:.4f} calib={row_c_var.calib_err_mean:.4f}")
        print(f"  CTGAN  fixed-w   : ll={row_c_fix.log_loss_mean:.4f} brier={row_c_fix.brier_mean:.4f} calib={row_c_fix.calib_err_mean:.4f}")
        print(f"  SMOTE  variable-w: ll={row_s_var.log_loss_mean:.4f} brier={row_s_var.brier_mean:.4f} calib={row_s_var.calib_err_mean:.4f}")
        print(f"  SMOTE  fixed-w   : ll={row_s_fix.log_loss_mean:.4f} brier={row_s_fix.brier_mean:.4f} calib={row_s_fix.calib_err_mean:.4f}")
        gap_var = row_c_var.log_loss_mean - row_s_var.log_loss_mean
        gap_fix = row_c_fix.log_loss_mean - row_s_fix.log_loss_mean
        print(f"  SMOTE-beats-CTGAN log_loss gap: variable-w={gap_var:+.4f}  fixed-w={gap_fix:+.4f}  "
              f"({'holds' if gap_fix > 0 else 'reverses'} under fixed weight)")

    print(f"\n✔ Saved -> {SENS_MG_FILE}")
    print("Next: python 18g_ece_reliability_matchgrouped.py")
