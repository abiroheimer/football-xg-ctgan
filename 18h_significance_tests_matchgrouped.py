"""
18h_significance_tests_matchgrouped.py — significance testing (one-sample
t-test, N=5, vs the match-grouped Baseline A) for the CTGAN and SMOTE arms
across all 18 ratios. Mirrors 12_significance_tests.py's methodology exactly
(same test choice/rationale — see that file's docstring); only the input
split/baseline/raw-run files change. Ablation E is not re-tested here (the
LOCO cold-start redesign already superseded/replaced that comparison).

Run standalone:
    python 18h_significance_tests_matchgrouped.py

Input : results/ablation_D_raw_runs_matchgrouped.csv,
        results/smote_ablation_D_raw_runs_matchgrouped.csv,
        results/baseline_metrics_matchgrouped.pkl
Output: results/significance_ablation_D_matchgrouped.csv,
        results/significance_smote_D_matchgrouped.csv
"""
import os
import pickle

import pandas as pd
from scipy.stats import ttest_1samp

from config import RESULTS_DIR

ABLATION_D_RAW_MG_FILE = os.path.join(RESULTS_DIR, "ablation_D_raw_runs_matchgrouped.csv")
SMOTE_D_RAW_MG_FILE = os.path.join(RESULTS_DIR, "smote_ablation_D_raw_runs_matchgrouped.csv")
BASELINE_METRICS_MG_FILE = os.path.join(RESULTS_DIR, "baseline_metrics_matchgrouped.pkl")

SIG_D_MG_FILE = os.path.join(RESULTS_DIR, "significance_ablation_D_matchgrouped.csv")
SIG_SMOTE_D_MG_FILE = os.path.join(RESULTS_DIR, "significance_smote_D_matchgrouped.csv")

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
    with open(BASELINE_METRICS_MG_FILE, "rb") as f:
        bm = pickle.load(f)
    baseline_A_rare = bm["res_A_rare"]

    print(f"Baseline A [matchgrouped] (rare test slice): log_loss={baseline_A_rare['log_loss']}  "
          f"brier={baseline_A_rare['brier']}")
    print("NOTE: AUC significance is skipped throughout, same as the original split "
          "(small rare-test positive count makes AUC variance estimates unreliable).\n")

    d_raw = pd.read_csv(ABLATION_D_RAW_MG_FILE)
    sig_d = test_arm(d_raw, baseline_A_rare["log_loss"], baseline_A_rare["brier"])
    sig_d.to_csv(SIG_D_MG_FILE, index=False)
    print("=" * 100)
    print("ABLATION D (CTGAN) [matchgrouped] — significance vs Baseline A  (one-sample t-test, N=5, alpha=0.05)")
    print("=" * 100)
    print(sig_d[["ratio", "log_loss_mean", "log_loss_p", "log_loss_significant_improvement",
                 "brier_mean", "brier_p", "brier_significant_improvement"]].to_string(index=False))
    n_sig_ll = sig_d["log_loss_significant_improvement"].sum()
    print(f"\n{n_sig_ll}/{len(sig_d)} ratios show significant log_loss improvement over Baseline A")

    s_raw = pd.read_csv(SMOTE_D_RAW_MG_FILE)
    sig_s = test_arm(s_raw, baseline_A_rare["log_loss"], baseline_A_rare["brier"])
    sig_s.to_csv(SIG_SMOTE_D_MG_FILE, index=False)
    print("\n" + "=" * 100)
    print("SMOTE ABLATION D [matchgrouped] — significance vs Baseline A  (one-sample t-test, N=5, alpha=0.05)")
    print("=" * 100)
    print(sig_s[["ratio", "log_loss_mean", "log_loss_p", "log_loss_significant_improvement",
                 "brier_mean", "brier_p", "brier_significant_improvement"]].to_string(index=False))
    n_sig_ll_s = sig_s["log_loss_significant_improvement"].sum()
    print(f"\n{n_sig_ll_s}/{len(sig_s)} ratios show significant log_loss improvement over Baseline A")

    print(f"\n✔ Saved → {SIG_D_MG_FILE}")
    print(f"✔ Saved → {SIG_SMOTE_D_MG_FILE}")
