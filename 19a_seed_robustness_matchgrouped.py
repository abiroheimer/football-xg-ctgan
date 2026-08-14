"""
19a_seed_robustness_matchgrouped.py — Step A: verify the CTGAN/SMOTE
reversal found under the match-grouped split (18d/18e/18h) is not an
artifact of one lucky/unlucky split. Re-runs the match-grouped split with 3
additional, well-separated base seeds, and for each, checks whether CTGAN
still significantly beats SMOTE at a reduced ratio set: 80%, 200%, 300%
(SMOTE's and CTGAN's previous best ratios) and 1000%.

Each seed gets its own independent 500-candidate GroupShuffleSplit search
(same procedure as 18a), its own Baseline A, and its own CTGAN generator
trained only on that split's rare training shots — nothing is shared across
seeds so this is a genuine independent robustness check, not a re-analysis
of the same split.

Run standalone:
    python 19a_seed_robustness_matchgrouped.py

Input : data/shots_features.csv
Output: results/seed_robustness_matchgrouped_raw.csv
        results/seed_robustness_matchgrouped_summary.csv
        results/seed_robustness_matchgrouped_significance.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from config import FEATURES_FILE, RESULTS_DIR, SEED, XG_FEATURES
from matchgrouped_utils import assert_no_overlap, build_split_dict, search_best_group_split, train_ctgan_on_rare
from model_utils import build_aug, build_smote_aug, evaluate, train_xgb

RAW_FILE = os.path.join(RESULTS_DIR, "seed_robustness_matchgrouped_raw.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "seed_robustness_matchgrouped_summary.csv")
SIG_FILE = os.path.join(RESULTS_DIR, "seed_robustness_matchgrouped_significance.csv")

# Additional base seeds for the candidate search, well-separated from 18a's
# search range (SEED..SEED+499 = 42..541) so these are genuinely different splits.
ADDITIONAL_BASE_SEEDS = [10042, 20042, 30042]
N_CANDIDATES = 500
TEST_SIZE = 0.20
RATIOS = [0.80, 2.00, 3.00, 10.00]
N_RUNS = 5
ALPHA = 0.05

if __name__ == "__main__":
    print(f"Loading {FEATURES_FILE} ...")
    shots = pd.read_csv(FEATURES_FILE)
    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)

    all_raw_rows = []
    all_summary_rows = []
    all_sig_rows = []

    for seed_id, base_seed in enumerate(ADDITIONAL_BASE_SEEDS, start=1):
        print(f"\n{'#' * 72}\nSEED {seed_id}  (candidate search base_seed={base_seed})\n{'#' * 72}")

        idx_train, idx_test, winning_seed, score = search_best_group_split(
            model_df, TEST_SIZE, base_seed, N_CANDIDATES)
        print(f"Winning candidate random_state={winning_seed}  (score={score:.5f})")

        split = build_split_dict(model_df, idx_train, idx_test)
        assert_no_overlap(("train", split["matches_train"]), ("test", split["matches_test"]))
        print(f"✔ Verified: 0 match_id overlap "
              f"(train={len(split['matches_train'])} matches, test={len(split['matches_test'])} matches)")

        X_train, y_train = split["X_train"], split["y_train"]
        X_rare_train, y_rare_train = split["X_rare_train"], split["y_rare_train"]
        X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
        rare_train_df = split["rare_train_df"]

        print(f"Rare train: {len(X_rare_train)}  |  Rare test: {len(X_rare_test)} "
              f"goals={int(y_rare_test.sum())}")

        # ── Baseline A for this seed's split ──
        np.random.seed(SEED)
        model_A = train_xgb(X_train, y_train, f"Baseline A [seed {seed_id}]")
        res_A = evaluate(model_A, X_rare_test, y_rare_test, "A-rare")
        print(f"Baseline A: log_loss={res_A['log_loss']}  brier={res_A['brier']}  "
              f"AUC={res_A['auc']}  calib={res_A['calib_err']}")

        # ── CTGAN generator for this seed's split ──
        ctgan_model = train_ctgan_on_rare(rare_train_df, label=f"[seed {seed_id}]")

        seed_raw = []
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
                    r = evaluate(m, X_rare_test, y_rare_test, "")
                    if r:
                        row = {"seed_id": seed_id, "base_seed": base_seed, "arm": arm, "ratio": ratio,
                               "run": run, "ll": r["log_loss"], "brier": r["brier"],
                               "auc": r["auc"], "calib": r["calib_err"]}
                        seed_raw.append(row)
                        all_raw_rows.append(row)
            print(f"  ratio={ratio:.0%} done")
        np.random.seed(SEED)

        seed_raw_df = pd.DataFrame(seed_raw)
        print(f"\n{'Arm':<8} {'Ratio':<7} {'LogLoss mean±std':<20} {'vs BaseA':<10}")
        print("-" * 50)
        for ratio in RATIOS:
            for arm in ["CTGAN", "SMOTE"]:
                g = seed_raw_df[(seed_raw_df.ratio == ratio) & (seed_raw_df.arm == arm)]
                ll_mean, ll_std = g["ll"].mean(), g["ll"].std()
                t, p = ttest_1samp(g["ll"].values, res_A["log_loss"])
                sig = p < ALPHA and ll_mean < res_A["log_loss"]
                print(f"{arm:<8} {ratio:<7.0%} {ll_mean:.4f}±{ll_std:.4f}      "
                      f"{ll_mean - res_A['log_loss']:+.4f}  p={p:.4f}  sig={sig}")
                all_summary_rows.append({
                    "seed_id": seed_id, "base_seed": base_seed, "arm": arm, "ratio": ratio,
                    "log_loss_mean": round(ll_mean, 4), "log_loss_std": round(ll_std, 4),
                    "baseA_log_loss": res_A["log_loss"],
                    "log_loss_vs_baseline": round(ll_mean - res_A["log_loss"], 4),
                    "log_loss_p": round(p, 4), "log_loss_significant_improvement": bool(sig),
                })

        # ── Head-to-head: does CTGAN beat SMOTE at each ratio, this seed? ──
        for ratio in RATIOS:
            ctgan_ll = seed_raw_df[(seed_raw_df.ratio == ratio) & (seed_raw_df.arm == "CTGAN")]["ll"].values
            smote_ll = seed_raw_df[(seed_raw_df.ratio == ratio) & (seed_raw_df.arm == "SMOTE")]["ll"].values
            t, p = ttest_1samp(ctgan_ll - smote_ll, 0.0)  # paired: same run indices, same seed_r per run
            ctgan_wins = ctgan_ll.mean() < smote_ll.mean()
            all_sig_rows.append({
                "seed_id": seed_id, "base_seed": base_seed, "ratio": ratio,
                "ctgan_ll_mean": round(ctgan_ll.mean(), 4), "smote_ll_mean": round(smote_ll.mean(), 4),
                "gap_ctgan_minus_smote": round(ctgan_ll.mean() - smote_ll.mean(), 4),
                "ctgan_wins": bool(ctgan_wins),
                "paired_t": round(t, 3), "paired_p": round(p, 4),
            })

    raw_df = pd.DataFrame(all_raw_rows)
    raw_df.to_csv(RAW_FILE, index=False)
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    sig_df = pd.DataFrame(all_sig_rows)
    sig_df.to_csv(SIG_FILE, index=False)

    print(f"\n{'=' * 90}\nCTGAN vs SMOTE HEAD-TO-HEAD ACROSS ALL {len(ADDITIONAL_BASE_SEEDS)} ADDITIONAL SEEDS\n{'=' * 90}")
    print(sig_df.to_string(index=False))

    n_ctgan_wins = sig_df["ctgan_wins"].sum()
    n_total = len(sig_df)
    print(f"\nCTGAN beats SMOTE (lower mean log_loss) in {n_ctgan_wins}/{n_total} "
          f"(seed x ratio) combinations across {len(ADDITIONAL_BASE_SEEDS)} additional seeds.")

    print(f"\n✔ Saved → {RAW_FILE}")
    print(f"✔ Saved → {SUMMARY_FILE}")
    print(f"✔ Saved → {SIG_FILE}")
    print("\nReview before proceeding to Step B (19b_train_val_test_split_matchgrouped.py).")
