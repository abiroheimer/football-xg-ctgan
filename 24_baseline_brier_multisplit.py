"""
24_baseline_brier_multisplit.py — Block C support: the eight-split study
(19e) saved Baseline A's log loss / AUC / calibration error per split but
not its Brier score, which Table I now needs. Baseline A is deterministic
(one XGBoost fit per split, seed 42, no synthetic sampling), so this
recomputes it exactly rather than approximating -- rebuilding each of the 8
splits with the identical procedure and base seeds used in 19e, then
refitting Baseline A and recording all four metrics.

CTGAN/SMOTE Brier scores are NOT recomputed here -- they are already present
per-run in results/multiseed_valtest_raw_matchgrouped.csv and are pulled
from there.

Run standalone:
    python 24_baseline_brier_multisplit.py

Input : data/shots_features.csv
Output: results/multiseed_valtest_baseline_brier.csv
        results/multiseed_valtest_table1.csv   (assembled Table I source, all arms)
"""
import os

import numpy as np
import pandas as pd

from config import FEATURES_FILE, RESULTS_DIR, SEED, XG_FEATURES
from model_utils import evaluate, train_xgb

# Reuse 19e's exact split construction so these are the same 8 splits.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "multiseed_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "19e_multiseed_valtest_matchgrouped.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BASELINE_BRIER_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_baseline_brier.csv")
TABLE1_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_table1.csv")
PER_SPLIT_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_per_split_matchgrouped.csv")
RAW_FILE = os.path.join(RESULTS_DIR, "multiseed_valtest_raw_matchgrouped.csv")

if __name__ == "__main__":
    shots = pd.read_csv(FEATURES_FILE)
    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)

    rows = []
    for split_id, base_seed in enumerate(_mod.BASE_SEEDS, start=1):
        split = _mod.build_three_way_split(model_df, base_seed)
        np.random.seed(SEED)
        model_A = train_xgb(split["X_train"], split["y_train"], f"Baseline A [split {split_id}]")
        r = evaluate(model_A, split["X_rare_test"], split["y_rare_test"], "")
        rows.append({
            "split_id": split_id, "base_seed": base_seed,
            "test_log_loss": r["log_loss"], "test_brier": r["brier"],
            "test_auc": r["auc"], "test_calib_err": r["calib_err"],
        })
        print(f"  split {split_id}: ll={r['log_loss']:.4f} brier={r['brier']:.4f} "
              f"auc={r['auc']:.4f} calib={r['calib_err']:.4f}")

    base_df = pd.DataFrame(rows)
    base_df.to_csv(BASELINE_BRIER_FILE, index=False)
    print(f"\n✔ Saved → {BASELINE_BRIER_FILE}")

    # ── Cross-check: recomputed log loss must match what 19e already saved ──
    per_split = pd.read_csv(PER_SPLIT_FILE)
    saved_ll = (per_split[per_split.arm == "Baseline A"]
                .set_index("split_id")["test_log_loss_mean"].reindex(base_df["split_id"]).values)
    max_dev = float(np.max(np.abs(saved_ll - base_df["test_log_loss"].values)))
    print(f"Cross-check vs 19e's saved Baseline A log loss: max abs deviation = {max_dev:.6f}")
    assert max_dev < 1e-6, "Recomputed Baseline A does not reproduce 19e's saved values"
    print("✔ Baseline A reproduces exactly — splits and model are deterministic as expected")

    # ── Assemble the full Table I source (all three arms, incl. Brier) ──
    raw = pd.read_csv(RAW_FILE)
    sel = per_split[per_split.arm.isin(["CTGAN", "SMOTE"])][["split_id", "arm", "selected_ratio"]]
    arm_rows = []
    for _, r in sel.iterrows():
        g = raw[(raw.split_id == r.split_id) & (raw.arm == r.arm) &
                (raw.set == "test") & (raw.ratio == r.selected_ratio)]
        arm_rows.append({"split_id": r.split_id, "arm": r.arm,
                         "test_log_loss": g["ll"].mean(), "test_brier": g["brier"].mean(),
                         "test_auc": g["auc"].mean(), "test_calib_err": g["calib"].mean()})
    arm_df = pd.DataFrame(arm_rows)
    base_for_concat = base_df.assign(arm="Baseline A")[
        ["split_id", "arm", "test_log_loss", "test_brier", "test_auc", "test_calib_err"]]
    allarms = pd.concat([base_for_concat, arm_df], ignore_index=True)

    summary = []
    for arm in ["Baseline A", "CTGAN", "SMOTE"]:
        sub = allarms[allarms.arm == arm]
        row = {"arm": arm, "n_splits": len(sub)}
        for col in ["test_log_loss", "test_brier", "test_auc", "test_calib_err"]:
            row[f"{col}_mean"] = round(sub[col].mean(), 4)
            row[f"{col}_std"] = round(sub[col].std(), 4)
        summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(TABLE1_FILE, index=False)

    print(f"\n{'=' * 90}\nTABLE I SOURCE (mean of per-split means ± std across 8 splits)\n{'=' * 90}")
    print(summary_df.to_string(index=False))
    print(f"\n✔ Saved → {TABLE1_FILE}")
