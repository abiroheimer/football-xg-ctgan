"""
matchgrouped_utils.py — Shared helpers for the match-grouped-split family of
scripts (18a-18h, 19a-19e). Not a standalone stage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import XG_FEATURES


def two_way_score(model_df, idx_train, idx_test, test_size):
    """Lower is better. Same scoring rule as 18a: penalizes deviation of
    (a) test-set shot fraction from target, (b) test-set rare-shot fraction
    from target, (c) rare goal-rate train/test gap, (d) normal goal-rate
    train/test gap."""
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
        abs(test_frac - test_size)
        + abs(rare_test_frac - test_size)
        + 0.5 * abs(rare_goal_train - rare_goal_test)
        + 0.5 * abs(norm_goal_train - norm_goal_test)
    )


def search_best_group_split(model_df, test_size, base_seed, n_candidates=500):
    """Search n_candidates GroupShuffleSplit(random_state=base_seed+trial)
    candidates (group=match_id) and return the (idx_train, idx_test,
    winning_seed) closest to target stratification on is_rare x is_goal."""
    groups = model_df["match_id"].values
    best_score = np.inf
    best = None
    for trial in range(n_candidates):
        seed = base_seed + trial
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_train, idx_test = next(gss.split(model_df, groups=groups))
        s = two_way_score(model_df, idx_train, idx_test, test_size)
        if s < best_score:
            best_score = s
            best = (idx_train, idx_test, seed)
    return best[0], best[1], best[2], best_score


def build_split_dict(model_df, idx_train, idx_test):
    """Build the standard split dict (X/y train/test/rare/normal + rare_train_df)
    used throughout the pipeline, from a set of positional row indices."""
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

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_rare_train": X_rare_train, "y_rare_train": y_rare_train,
        "X_rare_test": X_rare_test, "y_rare_test": y_rare_test,
        "X_norm_train": X_norm_train, "y_norm_train": y_norm_train,
        "rare_train_df": rare_train_df,
        "matches_train": set(model_df.iloc[idx_train]["match_id"]),
        "matches_test": set(model_df.iloc[idx_test]["match_id"]),
    }


def assert_no_overlap(*match_sets_with_names):
    """match_sets_with_names: (name, set) pairs. Hard fail on any pairwise overlap."""
    for i in range(len(match_sets_with_names)):
        for j in range(i + 1, len(match_sets_with_names)):
            name_i, set_i = match_sets_with_names[i]
            name_j, set_j = match_sets_with_names[j]
            overlap = set_i & set_j
            assert len(overlap) == 0, (
                f"FAIL: {len(overlap)} match_id(s) leak between {name_i} and {name_j}: {overlap}")


def train_ctgan_on_rare(rare_train_df, label=""):
    """Train a CTGAN generator with the pipeline's standard hyperparameters
    on a given rare_train_df. Identical methodology to 05_ctgan_train.py."""
    import time
    from ctgan import CTGAN

    CTGAN_COLS = ["distance", "angle_deg", "is_header", "is_freekick",
                  "goal_diff_clipped", "minute_bucket", "is_goal"]
    CTGAN_DISC = ["is_header", "is_freekick", "goal_diff_clipped", "minute_bucket", "is_goal"]

    ctgan_df = rare_train_df[CTGAN_COLS].copy()
    for c in ["is_header", "is_freekick", "is_goal"]:
        ctgan_df[c] = ctgan_df[c].astype(int)
    ctgan_df["goal_diff_clipped"] = ctgan_df["goal_diff_clipped"].astype(int)
    ctgan_df = ctgan_df.dropna()

    batch_sz = min(500, max(50, len(ctgan_df) // 2))
    batch_sz -= batch_sz % 10  # CTGAN requires batch_size % 2 == 0 and, with default pac=10, % 10 == 0
    print(f"  Training CTGAN on {len(ctgan_df)} rare shots {label} | batch={batch_sz} | epochs=500")
    m = CTGAN(
        epochs=500, batch_size=batch_sz,
        generator_dim=(256, 256), discriminator_dim=(256, 256),
        generator_lr=2e-4, discriminator_lr=2e-4,
        discriminator_steps=1, log_frequency=True, verbose=False, cuda=True,
    )
    t0 = time.time()
    m.fit(ctgan_df, discrete_columns=CTGAN_DISC)
    print(f"  -> trained in {time.time() - t0:.0f}s")
    return m
