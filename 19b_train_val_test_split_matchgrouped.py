"""
19b_train_val_test_split_matchgrouped.py — Step B: three-way match-grouped
split (train ~70% / val ~15% / test ~15%) so ratio selection (19d) can use
the validation set instead of the test set, closing the implicit
test-set-tuning gap (Point 6).

Uses the SAME candidate-search procedure and base seed (SEED=42) as the
paper's committed two-way match-grouped split (18a) — no split-shopping
across seeds here; Step A (19a) already checked robustness across seeds
separately, and this locks in the canonical split the paper's numbers use
from here forward.

Two-stage split, both stages grouped by match_id:
  1. Carve off TEST (~15% of shots) from the full set.
  2. Split the remaining ~85% into TRAIN (~70/85 of remainder) and
     VAL (~15/85 of remainder).
Each stage searches candidate GroupShuffleSplit random states and keeps the
one closest to its target is_rare x is_goal stratification (same scoring
rule as 18a), then reports the ACTUAL resulting composition.

Run standalone:
    python 19b_train_val_test_split_matchgrouped.py

Input : data/shots_features.csv
Output: data/split_data_matchgrouped_valtest.pkl
        results/split_valtest_composition_matchgrouped.csv
"""
import os
import pickle

import numpy as np
import pandas as pd

from config import DATA_DIR, FEATURES_FILE, RESULTS_DIR, SEED, XG_FEATURES
from matchgrouped_utils import assert_no_overlap, search_best_group_split

SPLIT_VT_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped_valtest.pkl")
COMPOSITION_FILE = os.path.join(RESULTS_DIR, "split_valtest_composition_matchgrouped.csv")

N_CANDIDATES = 500
TEST_FRAC = 0.15
VAL_FRAC_OF_REMAINDER = 0.15 / 0.85   # so val ends up ~15% of the ORIGINAL total


def compose(name, X, y, is_rare_mask, n_matches):
    return {
        "split": name, "n_shots": len(y), "n_matches": n_matches,
        "n_rare": int(is_rare_mask.sum()), "conv_pct": round(100 * y.mean(), 2),
        "rare_conv_pct": round(100 * y[is_rare_mask].mean(), 2) if is_rare_mask.sum() else float("nan"),
    }


if __name__ == "__main__":
    print(f"Loading {FEATURES_FILE} ...")
    shots = pd.read_csv(FEATURES_FILE)
    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)
    print(f"Total shots: {len(model_df):,}  |  total matches: {model_df['match_id'].nunique():,}  "
          f"|  total rare: {(model_df['is_rare'] == 1).sum():,}")

    # ── Stage 1: carve off TEST ──
    print(f"\nStage 1: searching {N_CANDIDATES} candidates for TEST split (~{TEST_FRAC:.0%})...")
    idx_rest, idx_test, seed1, score1 = search_best_group_split(model_df, TEST_FRAC, SEED, N_CANDIDATES)
    print(f"Winning candidate random_state={seed1}  (score={score1:.5f})")

    rest_df = model_df.iloc[idx_rest].reset_index(drop=True)

    # ── Stage 2: split remainder into TRAIN / VAL ──
    print(f"\nStage 2: searching {N_CANDIDATES} candidates for VAL split within remainder "
          f"(~{VAL_FRAC_OF_REMAINDER:.2%} of remainder)...")
    idx_train_rel, idx_val_rel, seed2, score2 = search_best_group_split(
        rest_df, VAL_FRAC_OF_REMAINDER, SEED + 100000, N_CANDIDATES)
    print(f"Winning candidate random_state={seed2}  (score={score2:.5f})")

    # Map back to original model_df indices
    idx_train = idx_rest[idx_train_rel]
    idx_val = idx_rest[idx_val_rel]

    matches_train = set(model_df.iloc[idx_train]["match_id"])
    matches_val = set(model_df.iloc[idx_val]["match_id"])
    matches_test = set(model_df.iloc[idx_test]["match_id"])
    assert_no_overlap(("train", matches_train), ("val", matches_val), ("test", matches_test))
    print(f"\n✔ Verified: 0 match_id overlap across all 3 splits "
          f"(train={len(matches_train)}, val={len(matches_val)}, test={len(matches_test)} matches)")

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

    X_norm_train, y_norm_train = X_train[~rare_train_mask], y_train[~rare_train_mask]

    comp_rows = [
        compose("train", X_train, y_train, rare_train_mask, len(matches_train)),
        compose("val", X_val, y_val, rare_val_mask, len(matches_val)),
        compose("test", X_test, y_test, rare_test_mask, len(matches_test)),
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(COMPOSITION_FILE, index=False)

    print(f"\n{'=' * 72}\nTRAIN / VAL / TEST COMPOSITION (match-grouped)\n{'=' * 72}")
    print(comp_df.to_string(index=False))

    split_vt = {
        "model_df": model_df,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_rare_train": X_rare_train, "y_rare_train": y_rare_train,
        "X_rare_val": X_rare_val, "y_rare_val": y_rare_val,
        "X_rare_test": X_rare_test, "y_rare_test": y_rare_test,
        "X_norm_train": X_norm_train, "y_norm_train": y_norm_train,
        "rare_train_df": rare_train_df,
        "matches_train": matches_train, "matches_val": matches_val, "matches_test": matches_test,
    }
    with open(SPLIT_VT_FILE, "wb") as f:
        pickle.dump(split_vt, f)
    print(f"\n✔ Saved → {SPLIT_VT_FILE}")
    print(f"✔ Saved → {COMPOSITION_FILE}")
    print("\nNext: python 19c_ctgan_train_matchgrouped_valtest.py")
