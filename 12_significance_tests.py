"""
12_significance_tests.py — Step 3 Fix 4: significance testing for Ablation D,
the new SMOTE arm, and Ablation E, using the N=5 raw per-run results each
stage already saves.

Test choice (flagged per Step 3 instructions): one-sample t-test
(scipy.stats.ttest_1samp), not a two-sample/paired t-test and not Wilcoxon.

  - It's one-sample, not paired, because Baseline A is a single fixed value
    per evaluation protocol (one model, trained once, no run-to-run
    variance) — there is no second sample of Baseline-A runs to pair against.
    The correct test for "does this set of 5 run values differ from a fixed
    reference constant" is a one-sample t-test against that constant.
  - Wilcoxon signed-rank was considered and rejected: at N=5, the exact
    Wilcoxon signed-rank test's smallest achievable two-sided p-value is
    1/16 = 0.0625 (all 5 differences must share the same sign, and even then
    that's the floor) — it can NEVER reach the p<0.05 threshold at this
    sample size, regardless of effect size. Using it here would make every
    single ratio "non-significant" by construction, which is a property of
    the test, not a finding about the data. The one-sample t-test has no
    such floor and is the only choice that can distinguish ratios by
    evidence strength at N=5, at the cost of assuming approximate normality
    of the 5 per-run values (a standard, if imperfect, assumption at this
    sample size).

Only log_loss and Brier are tested — AUC significance testing is skipped
per Step 3 Fix 6 (the 269-shot slice's 24 positive samples make AUC variance
estimates unreliable; see also the caveat text in ablation_D errata).

Run standalone:
    python 12_significance_tests.py

Input : results/ablation_D_raw_runs.csv, results/smote_ablation_D_raw_runs.csv,
        results/ablation_E_raw_runs.csv, results/baseline_metrics.pkl
Output: results/significance_ablation_D.csv, results/significance_smote_D.csv,
        results/significance_ablation_E.csv
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from config import RESULTS_DIR

ABLATION_D_RAW_FILE = os.path.join(RESULTS_DIR, "ablation_D_raw_runs.csv")
SMOTE_D_RAW_FILE = os.path.join(RESULTS_DIR, "smote_ablation_D_raw_runs.csv")
ABLATION_E_RAW_FILE = os.path.join(RESULTS_DIR, "ablation_E_raw_runs.csv")
BASELINE_METRICS_FILE = os.path.join(RESULTS_DIR, "baseline_metrics.pkl")

SIG_D_FILE = os.path.join(RESULTS_DIR, "significance_ablation_D.csv")
SIG_SMOTE_D_FILE = os.path.join(RESULTS_DIR, "significance_smote_D.csv")
SIG_E_FILE = os.path.join(RESULTS_DIR, "significance_ablation_E.csv")

ALPHA = 0.05


def test_arm(raw_df, baseline_ll, baseline_brier, ratio_col="ratio"):
    rows = []
    for ratio, g in raw_df.groupby(ratio_col):
        ll_vals = g["ll"].values
        brier_vals = g["brier"].values

        t_ll, p_ll = ttest_1samp(ll_vals, baseline_ll)
        t_br, p_br = ttest_1samp(brier_vals, baseline_brier)

        rows.append({
            "ratio": ratio, "n_runs": len(g),
            "log_loss_mean": round(ll_vals.mean(), 4), "log_loss_vs_baseline": round(ll_vals.mean() - baseline_ll, 4),
            "log_loss_t": round(t_ll, 3), "log_loss_p": round(p_ll, 4),
            "log_loss_significant_improvement": bool(p_ll < ALPHA and ll_vals.mean() < baseline_ll),
            "brier_mean": round(brier_vals.mean(), 4), "brier_vs_baseline": round(brier_vals.mean() - baseline_brier, 4),
            "brier_t": round(t_br, 3), "brier_p": round(p_br, 4),
            "brier_significant_improvement": bool(p_br < ALPHA and brier_vals.mean() < baseline_brier),
        })
    return pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)


if __name__ == "__main__":
    with open(BASELINE_METRICS_FILE, "rb") as f:
        bm = pickle.load(f)
    baseline_A_rare = bm["res_A_rare"]  # 269-shot slice baseline — for D & SMOTE-D

    print(f"Baseline A (269-shot rare test): log_loss={baseline_A_rare['log_loss']}  brier={baseline_A_rare['brier']}")
    print(f"NOTE: AUC significance is skipped throughout (Fix 6) — the 269-shot slice's 24 "
          f"positive samples make AUC variance estimates unreliable.\n")

    # ── Ablation D (CTGAN) vs Baseline A ──
    d_raw = pd.read_csv(ABLATION_D_RAW_FILE)
    sig_d = test_arm(d_raw, baseline_A_rare["log_loss"], baseline_A_rare["brier"])
    sig_d.to_csv(SIG_D_FILE, index=False)
    print("=" * 100)
    print("ABLATION D (CTGAN) — significance vs Baseline A  (one-sample t-test, N=5, alpha=0.05)")
    print("=" * 100)
    print(sig_d[["ratio", "log_loss_mean", "log_loss_p", "log_loss_significant_improvement",
                 "brier_mean", "brier_p", "brier_significant_improvement"]].to_string(index=False))
    n_sig_ll = sig_d["log_loss_significant_improvement"].sum()
    print(f"\n{n_sig_ll}/{len(sig_d)} ratios show significant log_loss improvement over Baseline A")

    # ── SMOTE Ablation D vs Baseline A ──
    s_raw = pd.read_csv(SMOTE_D_RAW_FILE)
    sig_s = test_arm(s_raw, baseline_A_rare["log_loss"], baseline_A_rare["brier"])
    sig_s.to_csv(SIG_SMOTE_D_FILE, index=False)
    print("\n" + "=" * 100)
    print("SMOTE ABLATION D — significance vs Baseline A  (one-sample t-test, N=5, alpha=0.05)")
    print("=" * 100)
    print(sig_s[["ratio", "log_loss_mean", "log_loss_p", "log_loss_significant_improvement",
                 "brier_mean", "brier_p", "brier_significant_improvement"]].to_string(index=False))
    n_sig_ll_s = sig_s["log_loss_significant_improvement"].sum()
    print(f"\n{n_sig_ll_s}/{len(sig_s)} ratios show significant log_loss improvement over Baseline A")

    # ── Ablation E vs Baseline A (on the full 1345-rare-shot set) ──
    # baseline_metrics.pkl only has the 269-shot Baseline A; Ablation E's own
    # script prints/derives Baseline A on the full rare set but doesn't save
    # it standalone, so recompute it the same way 09_ablation_e.py does.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import BASELINE_MODELS_FILE, SPLIT_FILE, XG_FEATURES
    from model_utils import evaluate

    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)
    with open(BASELINE_MODELS_FILE, "rb") as f:
        baselines = pickle.load(f)
    model_df = split["model_df"]
    all_rare_df = model_df[model_df["is_rare"] == 1]
    X_all_rare = all_rare_df[XG_FEATURES].values
    y_all_rare = all_rare_df["is_goal"].values
    res_A_allrare = evaluate(baselines["model_A"], X_all_rare, y_all_rare, "A-allrare")

    e_raw = pd.read_csv(ABLATION_E_RAW_FILE)
    sig_e = test_arm(e_raw, res_A_allrare["log_loss"], res_A_allrare["brier"])
    sig_e.to_csv(SIG_E_FILE, index=False)
    print("\n" + "=" * 100)
    print("ABLATION E (zero-shot, full 1345-rare-shot test) — significance vs Baseline A  (N=5, alpha=0.05)")
    print("=" * 100)
    print(sig_e[["ratio", "log_loss_mean", "log_loss_p", "log_loss_significant_improvement",
                 "brier_mean", "brier_p", "brier_significant_improvement"]].to_string(index=False))
    n_sig_ll_e = sig_e["log_loss_significant_improvement"].sum()
    print(f"\n{n_sig_ll_e}/{len(sig_e)} ratios show significant log_loss improvement over Baseline A")

    print(f"\n✔ Saved → {SIG_D_FILE}")
    print(f"✔ Saved → {SIG_SMOTE_D_FILE}")
    print(f"✔ Saved → {SIG_E_FILE}")
