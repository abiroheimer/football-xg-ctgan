"""
19e_multiseed_valtest_matchgrouped.py — Step B, revised: 19a's seed-
robustness check showed the CTGAN-vs-SMOTE winner flips depending on which
match-grouped split is drawn (CTGAN won cleanly on the originally-committed
split, but only 4/12 seed x ratio combinations favored CTGAN across 3 more
seeds). A single split's headline number is exactly the kind of fragile
result that prompted this whole re-split investigation, so per user
decision this replaces "pick one split" with averaging across N_SPLITS=8
independently-drawn match-grouped train/val/test splits — same logic as the
paper's existing N=5-per-ratio decision: enough draws to report a believable
mean±std, not just enough to notice instability exists.

For EACH of the 8 splits (train ~70% / val ~15% / test ~15%, grouped by
match_id, each independently searched for closest is_rare x is_goal
stratification — same procedure as 19b, just repeated per seed):
  1. Verify zero match_id overlap across all 3 partitions (hard check).
  2. Train Baseline A on TRAIN, evaluate on VAL and TEST.
  3. Train a CTGAN generator on TRAIN's rare shots only.
  4. Sweep all 18 ratios x N=5 runs for CTGAN and SMOTE, scoring every
     trained model on both VAL and TEST (TEST is recorded but never
     consulted for selection).
  5. Select each arm's best ratio using ONLY mean VAL log_loss.
  6. Report that split's final TEST numbers at the VAL-selected ratio.

Then aggregates the 8 splits' final-TEST numbers into a mean±std headline
per arm, reports the distribution of which ratio got selected each time,
and a head-to-head CTGAN-vs-SMOTE win count across splits.

Run standalone (long-running — ~8 x (CTGAN train + 180 XGB fits x2 eval sets)):
    python 19e_multiseed_valtest_matchgrouped.py

Input : data/shots_features.csv
Output: results/multiseed_valtest_per_split_matchgrouped.csv   (one row per split x arm: selected ratio + final test numbers)
        results/multiseed_valtest_raw_matchgrouped.csv          (every run, every split, val+test scores)
        results/multiseed_valtest_aggregate_matchgrouped.csv    (mean±std across splits, the paper's headline)
        results/multiseed_valtest_headtohead_matchgrouped.csv   (per-split CTGAN-vs-SMOTE winner)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from config import FEATURES_FILE, RESULTS_DIR, SEED, XG_FEATURES
from matchgrouped_utils import assert_no_overlap, search_best_group_split, train_ctgan_on_rare
from model_utils import build_aug, build_smote_aug, evaluate, train_xgb

PER_SPLIT_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_per_split_matchgrouped.csv")
RAW_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_raw_matchgrouped.csv")
AGGREGATE_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_aggregate_matchgrouped.csv")
HEAD_TO_HEAD_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_headtohead_matchgrouped.csv")

# Split 1 = the originally-committed split (SEED=42 base, matches 18a/18b's
# search range) so the paper's "canonical" split is one of the 8 draws, not
# discarded. Splits 2-8 are 7 new, well-separated base seeds.
BASE_SEEDS = [SEED, 10042, 20042, 30042, 40042, 50042, 60042, 70042]
N_CANDIDATES = 500
TEST_FRAC = 0.15
VAL_FRAC_OF_REMAINDER = 0.15 / 0.85

N_RUNS = 5
RATIOS = [0.01, 0.05, 0.10, 0.20, 0.40, 0.50, 0.80, 1.00,
          1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 10.00]
ALPHA = 0.05


def build_three_way_split(model_df, base_seed):
    idx_rest, idx_test, seed1, score1 = search_best_group_split(model_df, TEST_FRAC, base_seed, N_CANDIDATES)
    rest_df = model_df.iloc[idx_rest].reset_index(drop=True)
    idx_train_rel, idx_val_rel, seed2, score2 = search_best_group_split(
        rest_df, VAL_FRAC_OF_REMAINDER, base_seed + 100000, N_CANDIDATES)
    idx_train = idx_rest[idx_train_rel]
    idx_val = idx_rest[idx_val_rel]

    matches_train = set(model_df.iloc[idx_train]["match_id"])
    matches_val = set(model_df.iloc[idx_val]["match_id"])
    matches_test = set(model_df.iloc[idx_test]["match_id"])
    assert_no_overlap(("train", matches_train), ("val", matches_val), ("test", matches_test))

    X = model_df[XG_FEATURES].values
    y = model_df["is_goal"].values
    is_rare_all = (model_df["is_rare"] == 1).values

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    rare_train_mask = is_rare_all[idx_train]
    rare_val_mask = is_rare_all[idx_val]
    rare_test_mask = is_rare_all[idx_test]

    rare_train_df = model_df.iloc[idx_train][rare_train_mask].copy()
    X_rare_train, y_rare_train = X_train[rare_train_mask], y_train[rare_train_mask]
    X_rare_val, y_rare_val = X_val[rare_val_mask], y_val[rare_val_mask]
    X_rare_test, y_rare_test = X_test[rare_test_mask], y_test[rare_test_mask]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_rare_train": X_rare_train, "y_rare_train": y_rare_train,
        "X_rare_val": X_rare_val, "y_rare_val": y_rare_val,
        "X_rare_test": X_rare_test, "y_rare_test": y_rare_test,
        "rare_train_df": rare_train_df,
        "n_matches_train": len(matches_train), "n_matches_val": len(matches_val), "n_matches_test": len(matches_test),
        "seed1": seed1, "seed2": seed2,
    }


if __name__ == "__main__":
    print(f"Loading {FEATURES_FILE} ...")
    shots = pd.read_csv(FEATURES_FILE)
    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)

    all_raw_rows = []
    per_split_rows = []
    head_to_head_rows = []

    for split_id, base_seed in enumerate(BASE_SEEDS, start=1):
        tag = "ORIGINAL-COMMITTED" if split_id == 1 else f"seed{split_id}"
        print(f"\n{'#' * 72}\nSPLIT {split_id}/{len(BASE_SEEDS)}  ({tag}, base_seed={base_seed})\n{'#' * 72}")

        split = build_three_way_split(model_df, base_seed)
        print(f"✔ 0 match_id overlap  |  matches: train={split['n_matches_train']} "
              f"val={split['n_matches_val']} test={split['n_matches_test']}")
        print(f"  Rare train={len(split['X_rare_train'])}  Rare val={len(split['X_rare_val'])} "
              f"(goals={int(split['y_rare_val'].sum())})  Rare test={len(split['X_rare_test'])} "
              f"(goals={int(split['y_rare_test'].sum())})")

        X_train, y_train = split["X_train"], split["y_train"]
        X_rare_train, y_rare_train = split["X_rare_train"], split["y_rare_train"]
        X_rare_val, y_rare_val = split["X_rare_val"], split["y_rare_val"]
        X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
        rare_train_df = split["rare_train_df"]

        np.random.seed(SEED)
        model_A = train_xgb(X_train, y_train, f"Baseline A [split {split_id}]")
        res_A_val = evaluate(model_A, X_rare_val, y_rare_val, "A-val")
        res_A_test = evaluate(model_A, X_rare_test, y_rare_test, "A-test")
        print(f"  Baseline A: val ll={res_A_val['log_loss']}  test ll={res_A_test['log_loss']}")

        ctgan_model = train_ctgan_on_rare(rare_train_df, label=f"[split {split_id}]")

        val_rows, test_rows = [], []
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
                        row_v = {"split_id": split_id, "base_seed": base_seed, "set": "val",
                                  "arm": arm, "ratio": ratio, "run": run,
                                  "ll": rv["log_loss"], "brier": rv["brier"], "auc": rv["auc"], "calib": rv["calib_err"]}
                        val_rows.append(row_v)
                        all_raw_rows.append(row_v)
                    if rt:
                        row_t = {"split_id": split_id, "base_seed": base_seed, "set": "test",
                                  "arm": arm, "ratio": ratio, "run": run,
                                  "ll": rt["log_loss"], "brier": rt["brier"], "auc": rt["auc"], "calib": rt["calib_err"]}
                        test_rows.append(row_t)
                        all_raw_rows.append(row_t)
            print(f"    ratio={ratio:.0%} done")
        np.random.seed(SEED)

        val_df = pd.DataFrame(val_rows)
        test_df = pd.DataFrame(test_rows)

        val_summary = val_df.groupby(["arm", "ratio"])["ll"].mean().reset_index()
        selected = {}
        for arm in ["CTGAN", "SMOTE"]:
            sub = val_summary[val_summary.arm == arm]
            best_row = sub.loc[sub["ll"].idxmin()]
            selected[arm] = float(best_row["ratio"])

        per_split_rows.append({
            "split_id": split_id, "tag": tag, "base_seed": base_seed,
            "arm": "Baseline A", "selected_ratio": None,
            "test_log_loss_mean": res_A_test["log_loss"], "test_log_loss_std": 0.0,
            "test_auc_mean": res_A_test["auc"], "test_calib_err_mean": res_A_test["calib_err"],
        })
        for arm in ["CTGAN", "SMOTE"]:
            g = test_df[(test_df.arm == arm) & (test_df.ratio == selected[arm])]
            per_split_rows.append({
                "split_id": split_id, "tag": tag, "base_seed": base_seed,
                "arm": arm, "selected_ratio": selected[arm],
                "test_log_loss_mean": round(g["ll"].mean(), 4), "test_log_loss_std": round(g["ll"].std(), 4),
                "test_auc_mean": round(g["auc"].mean(), 4), "test_calib_err_mean": round(g["calib"].mean(), 4),
            })

        ctgan_final = test_df[(test_df.arm == "CTGAN") & (test_df.ratio == selected["CTGAN"])]["ll"]
        smote_final = test_df[(test_df.arm == "SMOTE") & (test_df.ratio == selected["SMOTE"])]["ll"]
        winner = "CTGAN" if ctgan_final.mean() < smote_final.mean() else "SMOTE"
        head_to_head_rows.append({
            "split_id": split_id, "tag": tag, "base_seed": base_seed,
            "ctgan_selected_ratio": selected["CTGAN"], "smote_selected_ratio": selected["SMOTE"],
            "ctgan_test_ll": round(ctgan_final.mean(), 4), "smote_test_ll": round(smote_final.mean(), 4),
            "gap_ctgan_minus_smote": round(ctgan_final.mean() - smote_final.mean(), 4),
            "winner": winner,
        })
        print(f"  -> Selected: CTGAN@{selected['CTGAN']:.0%} (test ll={ctgan_final.mean():.4f})  "
              f"SMOTE@{selected['SMOTE']:.0%} (test ll={smote_final.mean():.4f})  winner={winner}")

    raw_df = pd.DataFrame(all_raw_rows)
    raw_df.to_csv(RAW_FILE, index=False)
    per_split_df = pd.DataFrame(per_split_rows)
    per_split_df.to_csv(PER_SPLIT_FILE, index=False)
    head_df = pd.DataFrame(head_to_head_rows)
    head_df.to_csv(HEAD_TO_HEAD_FILE, index=False)

    print(f"\n{'=' * 90}\nPER-SPLIT FINAL (validation-selected ratio, evaluated on that split's held-out TEST)\n{'=' * 90}")
    print(per_split_df.to_string(index=False))

    print(f"\n{'=' * 90}\nHEAD-TO-HEAD ACROSS ALL {len(BASE_SEEDS)} SPLITS\n{'=' * 90}")
    print(head_df.to_string(index=False))
    n_ctgan_wins = (head_df["winner"] == "CTGAN").sum()
    print(f"\nCTGAN wins {n_ctgan_wins}/{len(head_df)} splits.")

    # ── Aggregate across splits ──
    agg_rows = []
    for arm in ["Baseline A", "CTGAN", "SMOTE"]:
        sub = per_split_df[per_split_df.arm == arm]
        row = {
            "arm": arm, "n_splits": len(sub),
            "test_log_loss_mean_of_means": round(sub["test_log_loss_mean"].mean(), 4),
            "test_log_loss_std_across_splits": round(sub["test_log_loss_mean"].std(), 4),
            "test_auc_mean_of_means": round(sub["test_auc_mean"].mean(), 4),
            "test_auc_std_across_splits": round(sub["test_auc_mean"].std(), 4),
            "test_calib_mean_of_means": round(sub["test_calib_err_mean"].mean(), 4),
            "test_calib_std_across_splits": round(sub["test_calib_err_mean"].std(), 4),
        }
        if arm in ("CTGAN", "SMOTE"):
            row["selected_ratio_values"] = sorted(sub["selected_ratio"].tolist())
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(AGGREGATE_FILE, index=False)

    print(f"\n{'=' * 90}\nAGGREGATE ACROSS {len(BASE_SEEDS)} SPLITS (mean of per-split means ± std across splits)\n{'=' * 90}")
    print(agg_df.to_string(index=False))

    # Paired t-test across splits: is CTGAN's per-split final test log_loss
    # systematically different from SMOTE's, across the 8 draws?
    ctgan_vals = per_split_df[per_split_df.arm == "CTGAN"].sort_values("split_id")["test_log_loss_mean"].values
    smote_vals = per_split_df[per_split_df.arm == "SMOTE"].sort_values("split_id")["test_log_loss_mean"].values
    t, p = ttest_1samp(ctgan_vals - smote_vals, 0.0)
    print(f"\nAcross-split paired t-test (CTGAN final ll - SMOTE final ll, N={len(ctgan_vals)} splits): "
          f"mean diff={np.mean(ctgan_vals - smote_vals):+.4f}  t={t:.3f}  p={p:.4f}")

    print(f"\n✔ Saved → {RAW_FILE}")
    print(f"✔ Saved → {PER_SPLIT_FILE}")
    print(f"✔ Saved → {HEAD_TO_HEAD_FILE}")
    print(f"✔ Saved → {AGGREGATE_FILE}")
