"""
22_permutation_tests.py — Block A: exact paired permutation tests as a
non-parametric sensitivity analysis alongside (not replacing) the existing
paired t-tests and bootstrap CIs.

Rationale: both headline analyses rest on small paired samples -- N=5 LOCO
folds and N=8 match-grouped splits -- where the t-test's normality
assumption is doing real work. The exact paired permutation test makes no
distributional assumption: under the null of no systematic difference, each
paired difference's sign is exchangeable, so enumerating all 2^N sign-flip
assignments gives the exact permutation-null distribution of the mean
difference. At N=5 (32 permutations) and N=8 (256 permutations) full
enumeration is trivially cheap, so no sampling approximation is needed.

Note on resolution floors: the two-sided exact permutation p-value cannot
go below 2/2^N -- 0.0625 at N=5, 0.0078 at N=8. At N=5 that floor sits
ABOVE 0.05, so a LOCO ratio can never reach p<0.05 by this test regardless
of effect size; that is a property of the test at this sample size, not
evidence against the effect. This is exactly why the permutation test is
reported as a supporting sensitivity analysis and the t-test/bootstrap
remain the primary inference -- the same reasoning already documented in
12_significance_tests.py for rejecting Wilcoxon at N=5.

Run standalone:
    python 22_permutation_tests.py

Input : results/ablation_E_loco_raw_runs.csv, results/ablation_E_loco_per_fold.csv,
        results/multiseed_valtest_per_split_matchgrouped.csv
Output: results/loco_permutation_test.csv
        results/multisplit_permutation_test.csv
"""
import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from config import RESULTS_DIR

LOCO_RAW_FILE = os.path.join(RESULTS_DIR, "ablation_E_loco_raw_runs.csv")
LOCO_PER_FOLD_FILE = os.path.join(RESULTS_DIR, "ablation_E_loco_per_fold.csv")
PER_SPLIT_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_per_split_matchgrouped.csv")

LOCO_PERM_FILE = os.path.join(RESULTS_DIR, "loco_permutation_test.csv")
MULTISPLIT_PERM_FILE = os.path.join(RESULTS_DIR, "multisplit_permutation_test.csv")


def exact_paired_permutation_p(diffs):
    """Exact two-sided paired permutation test on N paired differences.

    Enumerates all 2^N sign assignments (the exchangeability group under the
    null that each pair's difference is equally likely to have either sign),
    and returns the proportion of permuted mean differences at least as
    extreme in absolute value as the observed mean difference.
    Returns (p_value, n_permutations, observed_mean).
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    observed = diffs.mean()
    signs = np.array(list(product([1, -1], repeat=n)))   # (2^n, n)
    perm_means = (signs * diffs).mean(axis=1)
    p = float(np.mean(np.abs(perm_means) >= abs(observed) - 1e-12))
    return p, len(signs), float(observed)


# ════════════════════════════════════════════════════════════════════════
# A1 — LOCO, 5 folds, all 6 ratios
# ════════════════════════════════════════════════════════════════════════
def loco_permutation():
    raw_all = pd.read_csv(LOCO_RAW_FILE)
    per_fold = pd.read_csv(LOCO_PER_FOLD_FILE)
    # Primary folds only; the supplementary WC+Euro fold is never part of
    # the primary aggregate (see 17_ablation_e_loco.py).
    folds = sorted(per_fold["fold"].unique())
    raw = raw_all[raw_all["fold"].isin(folds)]
    baseA_by_fold = per_fold.drop_duplicates("fold").set_index("fold")["baseA_calib_err"]

    rows = []
    for ratio in sorted(raw["ratio"].unique()):
        fold_means = raw[raw.ratio == ratio].groupby("fold")["calib"].mean().reindex(folds)
        # diffs: baseline - LOCO, so POSITIVE = LOCO improves calibration
        diffs = baseA_by_fold.reindex(folds).values - fold_means.values
        rel_improve = 100 * (diffs / baseA_by_fold.reindex(folds).values).mean()

        p_perm, n_perm, obs = exact_paired_permutation_p(diffs)
        t, p_t = ttest_1samp(diffs, 0.0)

        rows.append({
            "ratio": ratio, "n_folds": len(folds),
            "mean_rel_improve_pct": round(rel_improve, 1),
            "mean_calib_diff_base_minus_loco": round(obs, 5),
            "paired_t": round(float(t), 3), "ttest_p": round(float(p_t), 5),
            "n_permutations": n_perm, "exact_permutation_p": round(p_perm, 5),
            "perm_p_floor": round(2 / n_perm, 5),
            "perm_at_floor": bool(abs(p_perm - 2 / n_perm) < 1e-9),
        })

    df = pd.DataFrame(rows)
    df.to_csv(LOCO_PERM_FILE, index=False)
    print("=" * 100)
    print("A1 — LOCO exact paired permutation test (N=5 folds, 2^5=32 permutations) vs paired t-test")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nNote: two-sided exact permutation p floor at N=5 is 2/32 = {2/32:.4f} (> 0.05 by construction).")
    print(f"✔ Saved → {LOCO_PERM_FILE}")
    return df


# ════════════════════════════════════════════════════════════════════════
# A2 — 8-split main effects
# ════════════════════════════════════════════════════════════════════════
def multisplit_permutation():
    per_split = pd.read_csv(PER_SPLIT_FILE)
    splits = sorted(per_split["split_id"].unique())

    baseline = per_split[per_split.arm == "Baseline A"].set_index("split_id")["test_log_loss_mean"].reindex(splits)
    ctgan = per_split[per_split.arm == "CTGAN"].set_index("split_id")["test_log_loss_mean"].reindex(splits)
    smote = per_split[per_split.arm == "SMOTE"].set_index("split_id")["test_log_loss_mean"].reindex(splits)

    comparisons = [
        ("CTGAN vs Baseline A", (ctgan - baseline).values),   # negative = CTGAN better
        ("SMOTE vs Baseline A", (smote - baseline).values),
        ("CTGAN vs SMOTE", (ctgan - smote).values),           # negative = CTGAN better
    ]

    rows = []
    for name, diffs in comparisons:
        p_perm, n_perm, obs = exact_paired_permutation_p(diffs)
        t, p_t = ttest_1samp(diffs, 0.0)
        rows.append({
            "comparison": name, "n_splits": len(splits),
            "mean_log_loss_diff": round(obs, 5),
            "paired_t": round(float(t), 3), "ttest_p": round(float(p_t), 5),
            "n_permutations": n_perm, "exact_permutation_p": round(p_perm, 5),
            "perm_p_floor": round(2 / n_perm, 5),
        })

    df = pd.DataFrame(rows)
    df.to_csv(MULTISPLIT_PERM_FILE, index=False)
    print("\n" + "=" * 100)
    print("A2 — Eight-split exact paired permutation test (N=8 splits, 2^8=256 permutations) vs paired t-test")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nNote: two-sided exact permutation p floor at N=8 is 2/256 = {2/256:.4f} (below 0.05, so this test CAN reach significance).")
    print(f"✔ Saved → {MULTISPLIT_PERM_FILE}")
    return df


if __name__ == "__main__":
    loco_permutation()
    multisplit_permutation()
