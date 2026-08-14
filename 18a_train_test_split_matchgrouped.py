"""
18a_train_test_split_matchgrouped.py — Match-grouped re-split (external review
fix): the original 03_train_test_split.py did an ordinary shot-level random
split stratified on is_rare x is_goal. Shots from the same match are not
independent (same score trajectory, same two teams, same tactical context),
so a random shot-level split can place correlated shots from one match on
both sides of train/test, inflating apparent performance vs a genuinely
unseen-match evaluation.

This re-splits at the MATCH level via GroupShuffleSplit (group = match_id),
targeting the same 80/20 ratio. Perfect is_rare x is_goal stratification is
not achievable under a group constraint (groups are indivisible blocks of
shots), so instead we search many candidate random_state values and keep the
split whose resulting composition (test-set size fraction, rare-shot
fraction, and goal-rate balance) is closest to the un-grouped target — then
report the ACTUAL resulting composition rather than forcing it.

Run standalone:
    python 18a_train_test_split_matchgrouped.py

Input : data/shots_features.csv
Output: data/split_data_matchgrouped.pkl
        results/split_comparison_matchgrouped.csv   (old vs new composition)
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import DATA_DIR, FEATURES_FILE, RESULTS_DIR, SEED, SPLIT_FILE, XG_FEATURES

SPLIT_MG_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped.pkl")
SPLIT_COMPARISON_FILE = os.path.join(RESULTS_DIR, "split_comparison_matchgrouped.csv")

N_CANDIDATES = 500   # candidate random_state values searched for closest-to-target stratification
TEST_SIZE = 0.20


def score_split(model_df, idx_train, idx_test):
    """Lower is better. Penalizes deviation of (a) test-set shot fraction
    from 0.20, (b) test-set rare-shot fraction from 0.20, (c) rare goal-rate
    gap between train/test, (d) normal goal-rate gap between train/test."""
    n_total = len(model_df)
    is_rare = (model_df["is_rare"] == 1).values
    is_goal = model_df["is_goal"].values

    test_frac = len(idx_test) / n_total
    rare_total = is_rare.sum()
    rare_test_frac = is_rare[idx_test].sum() / rare_total if rare_total else 0.0

    rare_train_mask = is_rare[idx_train]
    rare_test_mask = is_rare[idx_test]
    if rare_train_mask.sum() == 0 or rare_test_mask.sum() == 0:
        return np.inf

    rare_goal_train = is_goal[idx_train][rare_train_mask].mean()
    rare_goal_test = is_goal[idx_test][rare_test_mask].mean()
    norm_goal_train = is_goal[idx_train][~rare_train_mask].mean()
    norm_goal_test = is_goal[idx_test][~rare_test_mask].mean()

    return (
        abs(test_frac - TEST_SIZE)
        + abs(rare_test_frac - TEST_SIZE)
        + 0.5 * abs(rare_goal_train - rare_goal_test)
        + 0.5 * abs(norm_goal_train - norm_goal_test)
    )


if __name__ == "__main__":
    print(f"Loading {FEATURES_FILE} ...")
    shots = pd.read_csv(FEATURES_FILE)
    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)
    groups = model_df["match_id"].values
    n_matches = model_df["match_id"].nunique()
    print(f"Total shots: {len(model_df):,}  |  total matches: {n_matches:,}  "
          f"|  total rare: {(model_df['is_rare'] == 1).sum():,}")

    print(f"\nSearching {N_CANDIDATES} candidate group-shuffle splits for closest-to-target "
          f"stratification (test_size={TEST_SIZE:.0%}, group=match_id)...")
    best_score = np.inf
    best_idx = None
    for trial in range(N_CANDIDATES):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED + trial)
        idx_train, idx_test = next(gss.split(model_df, groups=groups))
        s = score_split(model_df, idx_train, idx_test)
        if s < best_score:
            best_score = s
            best_idx = (idx_train, idx_test, SEED + trial)

    idx_train, idx_test, best_seed = best_idx
    print(f"Best candidate: random_state={best_seed}  (score={best_score:.5f})")

    # ── Hard correctness check: no match_id in both train and test ──
    matches_train = set(model_df.iloc[idx_train]["match_id"])
    matches_test = set(model_df.iloc[idx_test]["match_id"])
    overlap = matches_train & matches_test
    assert len(overlap) == 0, f"FAIL: {len(overlap)} match_id(s) leak across train/test: {overlap}"
    print(f"✔ Verified: 0 match_id overlap between train ({len(matches_train)} matches) "
          f"and test ({len(matches_test)} matches)")

    X = model_df[XG_FEATURES].values
    y = model_df["is_goal"].values
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    rare_test_mask = model_df.iloc[idx_test]["is_rare"].values == 1
    X_rare_test = X_test[rare_test_mask]
    y_rare_test = y_test[rare_test_mask]

    rare_train_mask = model_df.iloc[idx_train]["is_rare"].values == 1
    rare_train_df = model_df.iloc[idx_train][rare_train_mask].copy()
    X_rare_train = X_train[rare_train_mask]
    y_rare_train = y_train[rare_train_mask]

    norm_train_mask = ~rare_train_mask
    X_norm_train = X_train[norm_train_mask]
    y_norm_train = y_train[norm_train_mask]

    print(f"\nTrain      : {len(X_train):,}  ({rare_train_mask.sum()} rare)")
    print(f"Test (std) : {len(X_test):,}   ({rare_test_mask.sum()} rare = {len(X_rare_test)} shots)")
    print(f"Rare train : {len(X_rare_train):,}  goals={y_rare_train.sum()} ({100 * y_rare_train.mean():.1f}% conv)")
    print(f"Rare test  : {len(X_rare_test):,}   goals={y_rare_test.sum()} ({100 * y_rare_test.mean():.1f}% conv)")

    split_mg = {
        "model_df": model_df,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_rare_train": X_rare_train, "y_rare_train": y_rare_train,
        "X_rare_test": X_rare_test, "y_rare_test": y_rare_test,
        "X_norm_train": X_norm_train, "y_norm_train": y_norm_train,
        "rare_train_df": rare_train_df,
        "match_split_random_state": best_seed,
        "matches_train": matches_train, "matches_test": matches_test,
    }
    with open(SPLIT_MG_FILE, "wb") as f:
        pickle.dump(split_mg, f)
    print(f"\n✔ Saved → {SPLIT_MG_FILE}")

    # ── Composition comparison vs the old shot-level split ──
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "rb") as f:
            split_old = pickle.load(f)

        def compose(split, matches=None):
            return {
                "n_train": len(split["X_train"]), "n_test": len(split["X_test"]),
                "n_rare_train": len(split["X_rare_train"]), "n_rare_test": len(split["X_rare_test"]),
                "rare_train_conv_pct": round(100 * split["y_rare_train"].mean(), 2),
                "rare_test_conv_pct": round(100 * split["y_rare_test"].mean(), 2),
                "train_conv_pct": round(100 * split["y_train"].mean(), 2),
                "test_conv_pct": round(100 * split["y_test"].mean(), 2),
                "n_matches_train": len(matches[0]) if matches else "n/a",
                "n_matches_test": len(matches[1]) if matches else "n/a",
            }

        old_comp = compose(split_old)
        new_comp = compose(split_mg, matches=(matches_train, matches_test))

        comparison_rows = []
        for k in old_comp:
            comparison_rows.append({"metric": k, "old_shot_level": old_comp[k], "new_match_grouped": new_comp[k]})
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(SPLIT_COMPARISON_FILE, index=False)

        print(f"\n{'=' * 72}\nSPLIT COMPOSITION: OLD (shot-level) vs NEW (match-grouped)\n{'=' * 72}")
        print(comp_df.to_string(index=False))
        print(f"\n✔ Saved → {SPLIT_COMPARISON_FILE}")
    else:
        print(f"\n⚠ {SPLIT_FILE} not found — skipping old-vs-new composition comparison")

    print("\nNext: python 18b_baseline_models_matchgrouped.py")
