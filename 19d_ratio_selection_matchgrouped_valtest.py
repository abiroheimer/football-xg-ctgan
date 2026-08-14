"""
19d_ratio_selection_matchgrouped_valtest.py — Step B: fix Point 6 (implicit
test-set tuning). Sweeps all 18 ratios x N=5 runs for both CTGAN and SMOTE
on the train/val/test match-grouped split (19b). Selection metric is mean
validation-set log loss (justification below) — the best ratio for EACH arm
is chosen using ONLY the validation set. The already-trained models for
every ratio are also scored on the held-out test set for full transparency,
but that test-set sweep is NEVER consulted when picking the ratio — only
the ratio selected from validation is reported as the "final" test number.

Selection metric = log loss (not Brier/AUC/calib): log loss is XGBoost's own
training objective (matches what's actually being optimized), is a strictly
proper scoring rule (unlike accuracy or raw calibration-error, it can't be
gamed by a systematically over/under-confident model), and is what Ablation
D has used as the primary "best ratio" metric throughout the paper so far —
using it for selection keeps this consistent with everything else.

Run standalone:
    python 19d_ratio_selection_matchgrouped_valtest.py

Input : data/split_data_matchgrouped_valtest.pkl, models/ctgan_model_matchgrouped_valtest.pkl
Output: results/ratio_sweep_val_matchgrouped_valtest.csv   (selection basis, per ratio, VAL)
        results/ratio_sweep_test_matchgrouped_valtest.csv  (same runs scored on TEST, NOT used for selection)
        results/final_selected_matchgrouped_valtest.csv    (selected ratio + final held-out TEST headline numbers)
        results/significance_final_matchgrouped_valtest.csv
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED
from model_utils import build_aug, build_smote_aug, evaluate, load_ctgan, train_xgb

SPLIT_VT_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped_valtest.pkl")
CTGAN_MODEL_VT_FILE = os.path.join(MODELS_DIR, "ctgan_model_matchgrouped_valtest.pkl")

VAL_SWEEP_FILE = os.path.join(RESULTS_DIR, "ratio_sweep_val_matchgrouped_valtest.csv")
TEST_SWEEP_FILE = os.path.join(RESULTS_DIR, "ratio_sweep_test_matchgrouped_valtest.csv")
FINAL_FILE = os.path.join(RESULTS_DIR, "final_selected_matchgrouped_valtest.csv")
SIG_FINAL_FILE = os.path.join(RESULTS_DIR, "significance_final_matchgrouped_valtest.csv")

N_RUNS = 5
RATIOS = [0.01, 0.05, 0.10, 0.20, 0.40, 0.50, 0.80, 1.00,
          1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 10.00]
ALPHA = 0.05

if __name__ == "__main__":
    with open(SPLIT_VT_FILE, "rb") as f:
        split = pickle.load(f)
    ctgan_model = load_ctgan(CTGAN_MODEL_VT_FILE)

    X_train, y_train = split["X_train"], split["y_train"]
    X_rare_train, y_rare_train = split["X_rare_train"], split["y_rare_train"]
    X_rare_val, y_rare_val = split["X_rare_val"], split["y_rare_val"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    rare_train_df = split["rare_train_df"]

    # ── Baseline A: trained on TRAIN only ──
    model_A = train_xgb(X_train, y_train, "Baseline A [valtest]")
    res_A_val = evaluate(model_A, X_rare_val, y_rare_val, "A-rare-val")
    res_A_test = evaluate(model_A, X_rare_test, y_rare_test, "A-rare-test")
    print(f"Baseline A  |  VAL: log_loss={res_A_val['log_loss']} calib={res_A_val['calib_err']}  "
          f"|  TEST: log_loss={res_A_test['log_loss']} calib={res_A_test['calib_err']}")

    val_rows, test_rows = [], []

    print(f"\nSweeping {len(RATIOS)} ratios x {N_RUNS} runs x 2 arms "
          f"= {len(RATIOS) * N_RUNS * 2} models (scored on both VAL and TEST)\n")

    for ratio in RATIOS:
        for arm in ["CTGAN", "SMOTE"]:
            for run in range(N_RUNS):
                seed_r = SEED + run
                np.random.seed(seed_r)
                try:
                    import torch
                    torch.manual_seed(seed_r)
                except Exception:
                    pass

                if arm == "CTGAN":
                    Xa, ya, ns = build_aug(X_train, y_train, rare_train_df, ctgan_model, ratio)
                else:
                    Xa, ya, ns = build_smote_aug(X_train, y_train, X_rare_train, y_rare_train,
                                                  ratio, seed=seed_r)
                m = train_xgb(Xa, ya, "")

                rv = evaluate(m, X_rare_val, y_rare_val, "")
                rt = evaluate(m, X_rare_test, y_rare_test, "")
                if rv:
                    val_rows.append({"arm": arm, "ratio": ratio, "run": run, "n_synth": ns,
                                      "ll": rv["log_loss"], "brier": rv["brier"],
                                      "auc": rv["auc"], "calib": rv["calib_err"]})
                if rt:
                    test_rows.append({"arm": arm, "ratio": ratio, "run": run, "n_synth": ns,
                                       "ll": rt["log_loss"], "brier": rt["brier"],
                                       "auc": rt["auc"], "calib": rt["calib_err"]})
        print(f"  ratio={ratio:.0%} done")

    np.random.seed(SEED)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    val_df.to_csv(VAL_SWEEP_FILE, index=False)
    test_df.to_csv(TEST_SWEEP_FILE, index=False)
    print(f"\n✔ Saved → {VAL_SWEEP_FILE}")
    print(f"✔ Saved → {TEST_SWEEP_FILE}  (kept for transparency; NOT used for ratio selection)")

    # ── Selection: best ratio per arm by mean VAL log_loss ONLY ──
    val_summary = val_df.groupby(["arm", "ratio"])["ll"].mean().reset_index()
    selected = {}
    for arm in ["CTGAN", "SMOTE"]:
        sub = val_summary[val_summary.arm == arm]
        best_row = sub.loc[sub["ll"].idxmin()]
        selected[arm] = float(best_row["ratio"])
        print(f"\nSelected ratio for {arm} (by VAL log_loss): {best_row['ratio']:.0%}  "
              f"(val log_loss={best_row['ll']:.4f})")

    # ── Final headline numbers: TEST performance at the VAL-selected ratio ──
    final_rows = [{
        "config": "Baseline A", "ratio": None, "n_runs": 1,
        "test_log_loss_mean": res_A_test["log_loss"], "test_log_loss_std": 0.0,
        "test_brier_mean": res_A_test["brier"], "test_brier_std": 0.0,
        "test_auc_mean": res_A_test["auc"], "test_auc_std": 0.0,
        "test_calib_err_mean": res_A_test["calib_err"], "test_calib_err_std": 0.0,
    }]
    sig_rows = []
    for arm, ratio in selected.items():
        g = test_df[(test_df.arm == arm) & (test_df.ratio == ratio)]
        final_rows.append({
            "config": f"{arm} (val-selected {ratio:.0%})", "ratio": ratio, "n_runs": len(g),
            "test_log_loss_mean": round(g["ll"].mean(), 4), "test_log_loss_std": round(g["ll"].std(), 4),
            "test_brier_mean": round(g["brier"].mean(), 4), "test_brier_std": round(g["brier"].std(), 4),
            "test_auc_mean": round(g["auc"].mean(), 4), "test_auc_std": round(g["auc"].std(), 4),
            "test_calib_err_mean": round(g["calib"].mean(), 4), "test_calib_err_std": round(g["calib"].std(), 4),
        })
        t, p = ttest_1samp(g["ll"].values, res_A_test["log_loss"])
        sig_rows.append({
            "arm": arm, "selected_ratio": ratio, "test_log_loss_mean": round(g["ll"].mean(), 4),
            "baseline_test_log_loss": res_A_test["log_loss"],
            "t": round(t, 3), "p": round(p, 4),
            "significant_improvement": bool(p < ALPHA and g["ll"].mean() < res_A_test["log_loss"]),
        })

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(FINAL_FILE, index=False)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(SIG_FINAL_FILE, index=False)

    print(f"\n{'=' * 90}\nFINAL HEADLINE (validation-selected ratio, evaluated once on held-out TEST)\n{'=' * 90}")
    print(final_df.to_string(index=False))
    print(f"\n{'=' * 90}\nSIGNIFICANCE vs Baseline A (test set, one-sample t-test, N=5)\n{'=' * 90}")
    print(sig_df.to_string(index=False))

    # ── CTGAN vs SMOTE head-to-head at their own selected ratios ──
    ctgan_g = test_df[(test_df.arm == "CTGAN") & (test_df.ratio == selected["CTGAN"])]
    smote_g = test_df[(test_df.arm == "SMOTE") & (test_df.ratio == selected["SMOTE"])]
    gap = ctgan_g["ll"].mean() - smote_g["ll"].mean()
    print(f"\nCTGAN (best {selected['CTGAN']:.0%}) vs SMOTE (best {selected['SMOTE']:.0%}) on TEST: "
          f"log_loss gap = {gap:+.4f}  ({'CTGAN wins' if gap < 0 else 'SMOTE wins'})")

    print(f"\n✔ Saved → {FINAL_FILE}")
    print(f"✔ Saved → {SIG_FINAL_FILE}")
