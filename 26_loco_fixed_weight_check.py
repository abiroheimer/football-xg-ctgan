"""
26_loco_fixed_weight_check.py — targeted scale_pos_weight sensitivity check
for the LOCO B_G / N / C_G comparison.

Why this is needed: the three LOCO conditions have different training-set
class compositions (B_G adds real rare shots, C_G adds synthetic rare
shots, N adds nothing), and train_xgb() recomputes scale_pos_weight from
each training set's own class balance. The N -> C_G calibration improvement
could therefore be partly a reweighting artifact rather than an effect of
the synthetic rare-event information. This isolates that confound.

Design (deliberately narrow -- 5% ratio only, same five folds, same held-out
test sets, existing seeds; nothing else is re-run):

  fixed_pw = natural scale_pos_weight of the normal-only (N) training set.
  N is the common substrate of all three conditions and its training set is
  identical across folds, so its weight is the natural single reference
  value. All three conditions are then retrained with scale_pos_weight held
  at that value and re-evaluated on the same held-out rare shots.

Both weight modes are computed inside THIS script from the same fold
generators and the same synthetic draws, so the variable-vs-fixed contrast
is perfectly paired. The variable-weight numbers are additionally
cross-checked against the original artifacts from 17/25 to confirm this run
reproduces the published values.

Run standalone:
    python 26_loco_fixed_weight_check.py

Input : data/shots_features.csv
Output: results/loco_fixed_weight_check.csv
        results/loco_fixed_weight_verdict.txt
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from config import FEATURES_FILE, RESULTS_DIR, SEED, XG_FEATURES
from matchgrouped_utils import train_ctgan_on_rare
from model_utils import evaluate, sample_synthetic, train_xgb

OUT_FILE = os.path.join(RESULTS_DIR, "loco_fixed_weight_check.csv")
VERDICT_FILE = os.path.join(RESULTS_DIR, "loco_fixed_weight_verdict.txt")

PRIMARY_FOLDS = ["England", "France", "Germany", "Italy", "Spain"]
RATIO = 0.05
N_RUNS = 5

if __name__ == "__main__":
    shots = pd.read_csv(FEATURES_FILE)
    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)
    is_rare = model_df["is_rare"] == 1

    normal_all = model_df[~is_rare]
    X_norm = normal_all[XG_FEATURES].values
    y_norm = normal_all["is_goal"].values

    # Fixed reference weight = natural weight of the normal-only training set.
    fixed_pw = float(max(1.0, (y_norm == 0).sum() / max((y_norm == 1).sum(), 1)))
    print(f"Normal-only (N) training set: {len(X_norm):,} shots")
    print(f"Fixed reference scale_pos_weight = {fixed_pw:.4f}\n")

    rows = []
    for fold in PRIMARY_FOLDS:
        eval_df = model_df[(model_df["competition"] == fold) & is_rare]
        X_eval = eval_df[XG_FEATURES].values
        y_eval = eval_df["is_goal"].values

        rare_other_df = model_df[(model_df["competition"] != fold) & is_rare].copy()
        X_rare_other = rare_other_df[XG_FEATURES].values
        y_rare_other = rare_other_df["is_goal"].values

        print(f"--- {fold}: eval {len(y_eval)} rare ({int(y_eval.sum())} goals), "
              f"generator pool {len(rare_other_df)} rare ---")

        # ---- B_G : normal + real rare from other 6 ----
        X_B = np.vstack([X_norm, X_rare_other])
        y_B = np.concatenate([y_norm, y_rare_other])
        nat_pw_B = float(max(1.0, (y_B == 0).sum() / max((y_B == 1).sum(), 1)))

        # ---- N : normal only ----
        nat_pw_N = fixed_pw  # by construction

        for mode, pw in [("variable", None), ("fixed", fixed_pw)]:
            np.random.seed(SEED)
            mB = train_xgb(X_B, y_B, "", scale_pos_weight=pw)
            rB = evaluate(mB, X_eval, y_eval, "")
            np.random.seed(SEED)
            mN = train_xgb(X_norm, y_norm, "", scale_pos_weight=pw)
            rN = evaluate(mN, X_eval, y_eval, "")
            rows.append({"fold": fold, "cond": "B_G", "weight_mode": mode, "run": 0,
                         "pw_used": nat_pw_B if pw is None else pw,
                         "calib_err": rB["calib_err"], "log_loss": rB["log_loss"]})
            rows.append({"fold": fold, "cond": "N", "weight_mode": mode, "run": 0,
                         "pw_used": nat_pw_N if pw is None else pw,
                         "calib_err": rN["calib_err"], "log_loss": rN["log_loss"]})

        # ---- C_G : normal + synthetic rare (N_RUNS draws, both weight modes per draw) ----
        ctgan = train_ctgan_on_rare(rare_other_df, label=f"[{fold}]")
        n_synth = max(10, int(len(rare_other_df) * RATIO))
        for run in range(N_RUNS):
            np.random.seed(SEED + run + 100)          # same seeds as 17_ablation_e_loco.py
            try:
                import torch
                torch.manual_seed(SEED + run + 100)
            except Exception:
                pass
            s = sample_synthetic(ctgan, n_synth, rare_other_df)
            Xs, ys = s[XG_FEATURES].values, s["is_goal"].values
            X_C = np.vstack([X_norm, Xs])
            y_C = np.concatenate([y_norm, ys])
            nat_pw_C = float(max(1.0, (y_C == 0).sum() / max((y_C == 1).sum(), 1)))
            # Same synthetic draw feeds both weight modes -> perfectly paired contrast
            for mode, pw in [("variable", None), ("fixed", fixed_pw)]:
                m = train_xgb(X_C, y_C, "", scale_pos_weight=pw)
                r = evaluate(m, X_eval, y_eval, "")
                rows.append({"fold": fold, "cond": "C_G", "weight_mode": mode, "run": run,
                             "pw_used": nat_pw_C if pw is None else pw,
                             "calib_err": r["calib_err"], "log_loss": r["log_loss"]})
        np.random.seed(SEED)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)

    # ── Per-fold means, then unweighted macro (paper's convention) ──
    fold_means = df.groupby(["fold", "cond", "weight_mode"])["calib_err"].mean().reset_index()

    def macro(cond, mode):
        return fold_means[(fold_means.cond == cond) &
                          (fold_means.weight_mode == mode)].set_index("fold")["calib_err"].reindex(PRIMARY_FOLDS)

    print(f"\n{'=' * 88}\nMACRO CALIBRATION ERROR (unweighted mean over 5 folds), 5% ratio\n{'=' * 88}")
    print(f"{'Condition':<10} {'variable-w':<14} {'fixed-w':<14}")
    print("-" * 88)
    res = {}
    for cond in ["B_G", "N", "C_G"]:
        v, f_ = macro(cond, "variable").mean(), macro(cond, "fixed").mean()
        res[cond] = (v, f_)
        print(f"{cond:<10} {v:<14.4f} {f_:<14.4f}")

    print(f"\n{'Comparison':<26} {'variable-w':<14} {'fixed-w':<14}")
    print("-" * 88)
    for name, a, b in [("B_G -> N", "B_G", "N"), ("N -> C_G", "N", "C_G"), ("B_G -> C_G", "B_G", "C_G")]:
        rv = 100 * (1 - res[b][0] / res[a][0])
        rf = 100 * (1 - res[b][1] / res[a][1])
        print(f"{name:<26} {rv:<14.1f} {rf:<14.1f}")

    # ── Paired test on N -> C_G under fixed weight (per-fold, N=5) ──
    n_fix = macro("N", "fixed").values
    c_fix = macro("C_G", "fixed").values
    d_fix = n_fix - c_fix                        # positive = C_G better calibrated
    t_f, p_f = ttest_1samp(d_fix, 0.0)
    n_var = macro("N", "variable").values
    c_var = macro("C_G", "variable").values
    d_var = n_var - c_var
    t_v, p_v = ttest_1samp(d_var, 0.0)

    print(f"\nPer-fold N -> C_G relative improvement (%):")
    print(f"{'fold':<10} {'variable-w':<14} {'fixed-w':<14}")
    for i, fold in enumerate(PRIMARY_FOLDS):
        print(f"{fold:<10} {100*(1-c_var[i]/n_var[i]):<14.1f} {100*(1-c_fix[i]/n_fix[i]):<14.1f}")

    n_pos_fix = int((d_fix > 0).sum())
    print(f"\nN -> C_G positive in {n_pos_fix}/5 folds under FIXED weight "
          f"(variable: {int((d_var > 0).sum())}/5)")
    print(f"paired t-test on fold-level diffs: variable t={t_v:.3f} p={p_v:.4f} | "
          f"fixed t={t_f:.3f} p={p_f:.4f}")

    macro_var = 100 * (1 - res['C_G'][0] / res['N'][0])
    macro_fix = 100 * (1 - res['C_G'][1] / res['N'][1])
    if macro_fix > 0 and n_pos_fix >= 4:
        verdict = (f"N->C_G REMAINS POSITIVE under fixed class weighting: "
                   f"{macro_fix:.1f}% macro (vs {macro_var:.1f}% variable), "
                   f"positive in {n_pos_fix}/5 folds.")
    elif macro_fix > 0:
        verdict = (f"N->C_G positive but WEAKENED/MIXED under fixed weighting: "
                   f"{macro_fix:.1f}% macro (vs {macro_var:.1f}% variable), "
                   f"positive in only {n_pos_fix}/5 folds.")
    else:
        verdict = (f"N->C_G DOES NOT SURVIVE fixed weighting: {macro_fix:.1f}% macro "
                   f"(vs {macro_var:.1f}% variable). The improvement is attributable to "
                   f"the class-weight change, not synthetic information.")
    print(f"\n{verdict}")

    with open(VERDICT_FILE, "w", encoding="utf-8") as f:
        f.write(verdict + "\n")
        f.write(f"fixed_pw={fixed_pw:.4f}\n")
        for cond in ["B_G", "N", "C_G"]:
            f.write(f"{cond}: variable={res[cond][0]:.4f} fixed={res[cond][1]:.4f}\n")
        f.write(f"N->C_G macro: variable={macro_var:.1f}% fixed={macro_fix:.1f}%\n")
        f.write(f"paired t (fixed): t={t_f:.3f} p={p_f:.4f}\n")

    print(f"\n✔ Saved → {OUT_FILE}")
    print(f"✔ Saved → {VERDICT_FILE}")
