"""
18b_baseline_models_matchgrouped.py — Baseline A/B re-trained on the
match-grouped split (18a). Mirrors 04_baseline_models.py exactly; only the
input split changes.

Run standalone:
    python 18b_baseline_models_matchgrouped.py

Input : data/split_data_matchgrouped.pkl
Output: models/baseline_models_matchgrouped.pkl, results/baseline_metrics_matchgrouped.pkl
"""
import os
import pickle

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from model_utils import evaluate, train_xgb

SPLIT_MG_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped.pkl")
BASELINE_MODELS_MG_FILE = os.path.join(MODELS_DIR, "baseline_models_matchgrouped.pkl")
BASELINE_METRICS_MG_FILE = os.path.join(RESULTS_DIR, "baseline_metrics_matchgrouped.pkl")

if __name__ == "__main__":
    with open(SPLIT_MG_FILE, "rb") as f:
        split = pickle.load(f)

    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    X_rare_train, y_rare_train = split["X_rare_train"], split["y_rare_train"]

    # ── Baseline A: full real data ──────────────────────────────
    model_A = train_xgb(X_train, y_train, "Baseline A [matchgrouped]")
    res_A_full = evaluate(model_A, X_test, y_test, "A-full")
    res_A_rare = evaluate(model_A, X_rare_test, y_rare_test, "A-rare")
    print(f"\nBaseline A on rare test: log_loss={res_A_rare['log_loss']}  "
          f"brier={res_A_rare['brier']}  AUC={res_A_rare['auc']}  calib={res_A_rare['calib_err']}")

    # ── Baseline B: rare shots only ─────────────────────────────
    if len(X_rare_train) >= 30:
        model_B = train_xgb(X_rare_train, y_rare_train, "Baseline B — rare only [matchgrouped]")
        res_B_rare = evaluate(model_B, X_rare_test, y_rare_test, "B-rare")
        print(f"Baseline B: log_loss={res_B_rare['log_loss']}  brier={res_B_rare['brier']}  "
              f"AUC={res_B_rare['auc']}  calib={res_B_rare['calib_err']}")
    else:
        model_B = model_A
        res_B_rare = res_A_rare.copy()
        print(f"⚠ Only {len(X_rare_train)} rare training shots — using Baseline A as B")

    with open(BASELINE_MODELS_MG_FILE, "wb") as f:
        pickle.dump({"model_A": model_A, "model_B": model_B}, f)
    with open(BASELINE_METRICS_MG_FILE, "wb") as f:
        pickle.dump({
            "res_A_full": res_A_full, "res_A_rare": res_A_rare, "res_B_rare": res_B_rare,
        }, f)

    print(f"\n✔ Saved → {BASELINE_MODELS_MG_FILE}")
    print(f"✔ Saved → {BASELINE_METRICS_MG_FILE}")
    print("Next: python 18c_ctgan_train_matchgrouped.py")
