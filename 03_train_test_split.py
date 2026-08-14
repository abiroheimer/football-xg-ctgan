"""
03_train_test_split.py — Stage 3: 80/20 split stratified on is_rare x is_goal.

Run standalone:
    python 03_train_test_split.py

Input : data/shots_features.csv   (stage 02 output)
Output: data/split_data.pkl        (dict consumed by every later stage)
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FEATURES_FILE, SEED, SPLIT_FILE, XG_FEATURES

if __name__ == "__main__":
    print(f"Loading {FEATURES_FILE} ...")
    shots = pd.read_csv(FEATURES_FILE)

    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)
    X = model_df[XG_FEATURES].values
    y = model_df["is_goal"].values
    idx = np.arange(len(model_df))
    strat = model_df["is_rare"].astype(str) + "_" + model_df["is_goal"].astype(str)

    idx_train, idx_test = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=strat.values)
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    rare_test_mask = model_df.iloc[idx_test]["is_rare"].values == 1
    X_rare_test = X_test[rare_test_mask]
    y_rare_test = y_test[rare_test_mask]

    rare_train_mask = model_df.iloc[idx_train]["is_rare"].values == 1
    rare_train_df = model_df.iloc[idx_train][rare_train_mask].copy()
    X_rare_train = X_train[rare_train_mask]
    y_rare_train = y_train[rare_train_mask]

    # Normal-only training subset (used by Ablation E)
    norm_train_mask = ~rare_train_mask
    X_norm_train = X_train[norm_train_mask]
    y_norm_train = y_train[norm_train_mask]

    print(f"Train      : {len(X_train):,}  ({rare_train_mask.sum()} rare)")
    print(f"Test (std) : {len(X_test):,}   ({rare_test_mask.sum()} rare = {len(X_rare_test)} shots)")
    print(f"Rare train : {len(X_rare_train):,}  goals={y_rare_train.sum()} ({100 * y_rare_train.mean():.1f}% conv)")
    print(f"Rare test  : {len(X_rare_test):,}   goals={y_rare_test.sum()} ({100 * y_rare_test.mean():.1f}% conv)")
    print(f"\nNOTE: Ablation E will test on ALL {(model_df['is_rare'] == 1).sum()} rare shots for more power.")

    split = {
        "model_df": model_df,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_rare_train": X_rare_train, "y_rare_train": y_rare_train,
        "X_rare_test": X_rare_test, "y_rare_test": y_rare_test,
        "X_norm_train": X_norm_train, "y_norm_train": y_norm_train,
        "rare_train_df": rare_train_df,
    }
    with open(SPLIT_FILE, "wb") as f:
        pickle.dump(split, f)
    print(f"\n✔ Saved → {SPLIT_FILE}")
    print("Next: python 04_baseline_models.py")
