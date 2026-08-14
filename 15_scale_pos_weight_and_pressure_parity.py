"""
15_scale_pos_weight_and_pressure_parity.py — Step 5: two validity checks on
the SMOTE-vs-CTGAN comparison, requested before the paper's narrative is
finalized around it.

Part A — scale_pos_weight confound (Fix 7's optional sensitivity check,
now run because it directly bears on the SMOTE-beats-CTGAN finding):
XGBoost's scale_pos_weight is recomputed per training set in every ablation
run, and it drifts as augmentation ratio changes class balance. This reruns
Baseline A's weight (~8.6, computed exactly below) as a FIXED weight across
three representative ratios (80%, 300%, 1000%) for both CTGAN and SMOTE,
N=5 runs each, and compares against the variable-weight numbers already in
results/ablation_D_mean_std.csv / smote_ablation_D_mean_std.csv — recomputed
fresh here (not reused) so both arms of the comparison run through the exact
same code path and seeds.

Part B — pressure_index parity: SMOTE's synthetic pressure_index is plain
continuous interpolation (no recompute-from-formula fix exists for it, since
SMOTE's feature space has no "minute" column — see build_smote_aug's
docstring). This quantifies that asymmetry: JS divergence + a domain-
validity check for SMOTE's pressure_index, using the same methodology and
QC sample size as CTGAN's stage-06 check, so the two numbers are directly
comparable.

Run standalone:
    python 15_scale_pos_weight_and_pressure_parity.py

Input : data/split_data.pkl, models/ctgan_model.pkl, results/baseline_metrics.pkl
Output: results/scale_pos_weight_sensitivity.csv,
        results/scale_pos_weight_sensitivity_raw.csv,
        results/pressure_index_parity.csv
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from config import BASELINE_MODELS_FILE, RESULTS_DIR, SEED, SPLIT_FILE, CTGAN_MODEL_FILE, XG_FEATURES
from model_utils import (SMOTE_CAT_COLS, build_aug, build_smote_aug, evaluate, load_ctgan,
                          sample_synthetic, train_xgb)
from imblearn.over_sampling import SMOTENC

SENS_FILE = os.path.join(RESULTS_DIR, "scale_pos_weight_sensitivity.csv")
SENS_RAW_FILE = os.path.join(RESULTS_DIR, "scale_pos_weight_sensitivity_raw.csv")
PARITY_FILE = os.path.join(RESULTS_DIR, "pressure_index_parity.csv")

RATIOS = [0.80, 3.00, 10.00]  # 80%, 300%, 1000%
N_RUNS = 5

if __name__ == "__main__":
    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)
    with open(BASELINE_MODELS_FILE, "rb") as f:
        baselines = pickle.load(f)
    ctgan_model = load_ctgan(CTGAN_MODEL_FILE)

    X_train, y_train = split["X_train"], split["y_train"]
    X_rare_train, y_rare_train = split["X_rare_train"], split["y_rare_train"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    rare_train_df = split["rare_train_df"]
    # Recomputed fresh (not read from results/baseline_metrics.pkl, which
    # predates the "ece" field added to evaluate() in Step 3) — pure
    # inference on the already-trained Baseline A model, no retraining.
    res_A_rare = evaluate(baselines["model_A"], X_rare_test, y_rare_test, "Baseline A")

    # ════════════════════════════════════════════════════════════════
    # Part A — scale_pos_weight sensitivity
    # ════════════════════════════════════════════════════════════════
    baseline_pw = float(max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    print(f"Baseline A exact scale_pos_weight: {baseline_pw:.4f}  "
          f"(neg={int((y_train == 0).sum())}, pos={int((y_train == 1).sum())})")

    # Baseline A itself is deterministic (one model, no synthetic sampling) —
    # its own natural weight already equals the "fixed" reference weight, so
    # there is nothing to compare for Baseline A specifically. It's included
    # as a reference row (n_runs=1) rather than re-run 5x for an identical model.
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
    raw_df.to_csv(SENS_RAW_FILE, index=False)
    print(f"\n✔ Saved raw per-run results -> {SENS_RAW_FILE}")

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
    summary_df.to_csv(SENS_FILE, index=False)

    print("\n" + "=" * 100)
    print("SCALE_POS_WEIGHT SENSITIVITY SUMMARY  (mean over N=5 runs; Baseline A is n=1, deterministic)")
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

    print(f"\n✔ Saved -> {SENS_FILE}")

    # ════════════════════════════════════════════════════════════════
    # Part B — pressure_index parity (SMOTE plain-interpolated vs CTGAN fixed)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("PRESSURE_INDEX PARITY: SMOTE (plain interpolation) vs CTGAN (recomputed from formula)")
    print("=" * 100)

    real_pi = rare_train_df["pressure_index"].dropna().values
    pi_idx = XG_FEATURES.index("pressure_index")

    # CTGAN side — same methodology as 06_synthetic_quality.py, computed fresh
    # here for a same-run apples-to-apples number (06's saved value was 0.1095).
    N_QC = min(5000, len(rare_train_df) * 3)
    ctgan_qc = sample_synthetic(ctgan_model, N_QC, rare_train_df, clip=True)
    ctgan_pi = ctgan_qc["pressure_index"].dropna().values

    # SMOTE side — generate a pure-synthetic batch of the same size, same
    # categorical/continuous column split as build_smote_aug, but return ALL
    # synthetic rows directly (not merged with real) so pressure_index is
    # isolated for QC. This is plain SMOTENC interpolation — no recompute.
    cat_idx = [XG_FEATURES.index(c) for c in SMOTE_CAT_COLS]
    n_pos_exist = int((y_rare_train == 1).sum())
    n_neg_exist = int((y_rare_train == 0).sum())
    target_rate = float(y_rare_train.mean())
    n_pos = max(1, round(N_QC * target_rate))
    n_neg = N_QC - n_pos
    k = max(1, min(5, n_pos_exist - 1, n_neg_exist - 1))
    sm = SMOTENC(categorical_features=cat_idx,
                 sampling_strategy={0: n_neg_exist + n_neg, 1: n_pos_exist + n_pos},
                 k_neighbors=k, random_state=SEED)
    X_res, y_res = sm.fit_resample(X_rare_train, y_rare_train)
    smote_pi = X_res[len(X_rare_train):, pi_idx]

    def js_div(real, synth, nbins=60):
        bins = np.linspace(min(real.min(), synth.min()), max(real.max(), synth.max()), nbins)
        rh, _ = np.histogram(real, bins=bins, density=True)
        sh, _ = np.histogram(synth, bins=bins, density=True)
        rh = rh / rh.sum() + 1e-10
        sh = sh / sh.sum() + 1e-10
        return float(jensenshannon(rh, sh))

    real_min, real_max = real_pi.min(), real_pi.max()
    parity_rows = []
    for name, synth in [("CTGAN (recomputed)", ctgan_pi), ("SMOTE (plain interpolation)", smote_pi)]:
        _, ks_p = ks_2samp(real_pi, synth)
        js = js_div(real_pi, synth)
        domain_valid = float(((synth >= 0) & (synth <= real_max)).mean())
        nonneg = float((synth >= 0).mean())
        parity_rows.append({
            "method": name, "n_synth": len(synth),
            "real_range": f"[{real_min:.3f}, {real_max:.3f}]",
            "synth_range": f"[{synth.min():.3f}, {synth.max():.3f}]",
            "ks_pvalue": round(ks_p, 4), "js_divergence": round(js, 4),
            "pct_nonnegative": round(100 * nonneg, 2),
            "pct_within_real_range": round(100 * domain_valid, 2),
        })
        print(f"\n{name}:")
        print(f"  n_synth={len(synth)}  real_range=[{real_min:.3f},{real_max:.3f}]  "
              f"synth_range=[{synth.min():.3f},{synth.max():.3f}]")
        print(f"  KS p-value={ks_p:.4f}  JS divergence={js:.4f}")
        print(f"  % non-negative: {100*nonneg:.2f}%   % within real [min,max] range: {100*domain_valid:.2f}%")

    parity_df = pd.DataFrame(parity_rows)
    parity_df.to_csv(PARITY_FILE, index=False)
    print(f"\n✔ Saved -> {PARITY_FILE}")
