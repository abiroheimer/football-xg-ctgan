"""
20_statistical_rigor.py — Point 7: bootstrap CIs + effect sizes for the two
claims that survive after the CTGAN-vs-SMOTE comparison came back null
(8-split paired t-test, p=0.79 — see multiseed_valtest_headtohead_matchgrouped.csv).
That null result is already correctly characterized by the paired t-test and
gets NO additional machinery here (bootstrapping a null doesn't strengthen
anything). This script only touches the two claims that still have a
direction to quantify:

  Claim 1 — augmentation (CTGAN, SMOTE) modestly beats Baseline A, averaged
  across the 8 match-grouped/validation-selected splits.
  Resampling unit: the 8 splits themselves (paired bootstrap — each
  resample redraws which splits contribute, keeping each split's own
  (arm, baseline) pair together since they share that split's train/test
  partition and Baseline A model).

  Claim 2 — the LOCO cold-start calibration improvement at the 5% ratio
  (and, with multiple-comparisons correction, across all 6 ratios).
  Resampling unit: a TWO-STAGE bootstrap — resample the 5 folds with
  replacement (primary unit: folds are the genuinely independent draws,
  since they're different held-out competitions with different generators
  and different eval sets), then WITHIN each resampled fold also resample
  its 5 runs with replacement (secondary unit: captures run-to-run
  stochasticity too, without pretending the 25 raw run values are 25
  independent samples of the underlying population of competitions, which
  they are not — runs within a fold share the same held-out eval set and
  generator, so treating them as independent would understate the true CI
  width). This is the standard hierarchical/cluster bootstrap prescription
  for clustered data.

Multiple-comparisons correction (LOCO's 6 ratios): Holm-Bonferroni, not
Benjamini-Hochberg. With only 6 tests and these being the ratios the paper
uses to justify its ratio-selection story, family-wise error control
(Holm) is the more conservative, defensible choice; Holm is uniformly more
powerful than plain Bonferroni while keeping the same FWER guarantee, so
there's no cost to preferring it over BH here.

Run standalone:
    python 20_statistical_rigor.py

Input : results/multiseed_valtest_per_split_matchgrouped.csv
        results/ablation_E_loco_raw_runs.csv, results/ablation_E_loco_per_fold.csv
Output: results/bootstrap_ci_augmentation_vs_baseline.csv
        results/bootstrap_ci_loco_calib_5pct.csv
        results/loco_ratio_significance_holm.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from config import RESULTS_DIR, SEED

PER_SPLIT_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_per_split_matchgrouped.csv")
LOCO_RAW_FILE = os.path.join(RESULTS_DIR, "ablation_E_loco_raw_runs.csv")
LOCO_PER_FOLD_FILE = os.path.join(RESULTS_DIR, "ablation_E_loco_per_fold.csv")

CI_AUG_FILE = os.path.join(RESULTS_DIR, "bootstrap_ci_augmentation_vs_baseline.csv")
CI_LOCO_FILE = os.path.join(RESULTS_DIR, "bootstrap_ci_loco_calib_5pct.csv")
LOCO_HOLM_FILE = os.path.join(RESULTS_DIR, "loco_ratio_significance_holm.csv")

N_BOOT = 10_000
RNG = np.random.default_rng(SEED)


def holm_bonferroni(pvals, alpha=0.05):
    """Returns (adjusted_pvals in original order, reject in original order)."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw_adj = (m - rank) * pvals[idx]
        running_max = max(running_max, raw_adj)
        adj[idx] = min(running_max, 1.0)
    reject = adj < alpha
    return adj, reject


def cohens_dz(diffs):
    """Paired-samples effect size: mean difference / std of differences."""
    return float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else np.nan


# ════════════════════════════════════════════════════════════════════════
# Claim 1 — augmentation vs Baseline A, 8-split paired bootstrap
# ════════════════════════════════════════════════════════════════════════
def bootstrap_claim1():
    per_split = pd.read_csv(PER_SPLIT_FILE)
    splits = sorted(per_split["split_id"].unique())
    n_splits = len(splits)

    baseline = per_split[per_split.arm == "Baseline A"].set_index("split_id")["test_log_loss_mean"]
    ctgan = per_split[per_split.arm == "CTGAN"].set_index("split_id")["test_log_loss_mean"]
    smote = per_split[per_split.arm == "SMOTE"].set_index("split_id")["test_log_loss_mean"]

    diff_ctgan = (ctgan - baseline).reindex(splits).values   # negative = CTGAN better
    diff_smote = (smote - baseline).reindex(splits).values

    rows = []
    for name, diffs in [("CTGAN vs Baseline A", diff_ctgan), ("SMOTE vs Baseline A", diff_smote)]:
        boot_means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = RNG.integers(0, n_splits, size=n_splits)
            boot_means[b] = diffs[idx].mean()
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        dz = cohens_dz(diffs)
        t, p = ttest_1samp(diffs, 0.0)
        rows.append({
            "comparison": name, "n_splits": n_splits,
            "mean_log_loss_diff": round(float(diffs.mean()), 4),
            "ci_2.5pct": round(float(lo), 4), "ci_97.5pct": round(float(hi), 4),
            "cohens_dz": round(dz, 3),
            "t": round(float(t), 3), "p": round(float(p), 4),
        })
        print(f"{name}: mean Δlog_loss={diffs.mean():+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"Cohen's dz={dz:.3f}  (paired t={t:.3f}, p={p:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(CI_AUG_FILE, index=False)
    print(f"\n✔ Saved → {CI_AUG_FILE}")
    return df


# ════════════════════════════════════════════════════════════════════════
# Claim 2 — LOCO calibration improvement, two-stage (fold + run) bootstrap
# ════════════════════════════════════════════════════════════════════════
def bootstrap_claim2():
    raw_all = pd.read_csv(LOCO_RAW_FILE)
    per_fold = pd.read_csv(LOCO_PER_FOLD_FILE)
    # per_fold only ever held the 5 primary folds (England/France/Germany/Italy/Spain);
    # the raw-runs file also contains the supplementary World_Cup+European_Championship
    # fold, which must be excluded here — it's reported separately, low-confidence, and
    # never part of the primary aggregate (see 17_ablation_e_loco.py).
    folds = sorted(per_fold["fold"].unique())
    raw = raw_all[raw_all["fold"].isin(folds)].copy()
    n_folds = len(folds)

    baseA_by_fold = per_fold.drop_duplicates("fold").set_index("fold")["baseA_calib_err"]

    # ── 5% ratio: two-stage bootstrap on relative calibration improvement ──
    ratio = 0.05
    fold_runs = {}
    for f in folds:
        vals = raw[(raw.fold == f) & (raw.ratio == ratio)]["calib"].values
        fold_runs[f] = vals  # length-5 array of calib_err per run

    boot_rel_improve = np.empty(N_BOOT)
    for b in range(N_BOOT):
        resampled_folds = RNG.choice(folds, size=n_folds, replace=True)
        fold_rel_improves = np.empty(n_folds)
        for i, f in enumerate(resampled_folds):
            runs = fold_runs[f]
            resampled_runs = runs[RNG.integers(0, len(runs), size=len(runs))]
            fold_mean_calib = resampled_runs.mean()
            fold_rel_improves[i] = 1.0 - fold_mean_calib / baseA_by_fold[f]
        boot_rel_improve[b] = fold_rel_improves.mean()  # macro-average across resampled folds

    point_estimate = per_fold[per_fold.ratio == ratio].apply(
        lambda r: 1.0 - r["calib_err_mean"] / r["baseA_calib_err"], axis=1).mean()
    lo, hi = np.percentile(boot_rel_improve, [2.5, 97.5])
    print(f"\nLOCO macro calibration improvement @ 5% ratio: point estimate={100 * point_estimate:.1f}%  "
          f"95% CI=[{100 * lo:.1f}%, {100 * hi:.1f}%]  (two-stage bootstrap, N={N_BOOT}, "
          f"resample folds then within-fold runs)")

    # Fold-level paired effect size at 5%: fold-mean calib_err vs fold baseA_calib_err (N=5 folds)
    fold_means_5pct = raw[raw.ratio == ratio].groupby("fold")["calib"].mean().reindex(folds)
    paired_diffs_5pct = (fold_means_5pct.values - baseA_by_fold.reindex(folds).values)
    dz_5pct = cohens_dz(paired_diffs_5pct)
    t5, p5 = ttest_1samp(paired_diffs_5pct, 0.0)
    print(f"Fold-level paired effect size @ 5%: Cohen's dz={dz_5pct:.3f}  (paired t={t5:.3f}, p={p5:.4f}, N=5 folds)")

    ci_row = {
        "ratio": ratio, "point_estimate_pct": round(100 * point_estimate, 1),
        "ci_2.5pct": round(100 * lo, 1), "ci_97.5pct": round(100 * hi, 1),
        "cohens_dz_fold_level": round(dz_5pct, 3), "paired_t": round(float(t5), 3), "paired_p": round(float(p5), 4),
        "bootstrap_method": "two-stage: resample 5 folds, then resample 5 runs within each resampled fold",
    }
    pd.DataFrame([ci_row]).to_csv(CI_LOCO_FILE, index=False)
    print(f"✔ Saved → {CI_LOCO_FILE}")

    # ── All 6 ratios: fold-level paired t-test + Holm correction ──
    all_ratios = sorted(raw["ratio"].unique())
    holm_rows = []
    pvals = []
    for r in all_ratios:
        fold_means = raw[raw.ratio == r].groupby("fold")["calib"].mean().reindex(folds)
        diffs = fold_means.values - baseA_by_fold.reindex(folds).values
        t, p = ttest_1samp(diffs, 0.0)
        rel_improve = 1.0 - fold_means.values / baseA_by_fold.reindex(folds).values
        holm_rows.append({
            "ratio": r, "mean_rel_improve_pct": round(100 * rel_improve.mean(), 1),
            "paired_t": round(float(t), 3), "raw_p": round(float(p), 5),
        })
        pvals.append(p)

    adj_p, reject = holm_bonferroni(pvals, alpha=0.05)
    holm_df = pd.DataFrame(holm_rows)
    holm_df["holm_adjusted_p"] = np.round(adj_p, 5)
    holm_df["significant_after_holm"] = reject
    holm_df.to_csv(LOCO_HOLM_FILE, index=False)

    print(f"\n{'=' * 90}\nLOCO — fold-level paired t-test per ratio (N=5 folds), Holm-Bonferroni corrected (alpha=0.05)\n{'=' * 90}")
    print(holm_df.to_string(index=False))
    print(f"\n✔ Saved → {LOCO_HOLM_FILE}")


if __name__ == "__main__":
    print("=" * 90)
    print("CLAIM 1 — Augmentation (CTGAN/SMOTE) vs Baseline A, 8-split paired bootstrap")
    print("=" * 90)
    bootstrap_claim1()

    print("\n" + "=" * 90)
    print("CLAIM 2 — LOCO cold-start calibration improvement, two-stage (fold+run) bootstrap")
    print("=" * 90)
    bootstrap_claim2()
