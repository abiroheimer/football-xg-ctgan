"""
07_model_c.py — Stage 7: Model C — real training data + CTGAN synthetic
rare shots at a single default ratio (20%).

Run standalone:
    python 07_model_c.py

Input : data/split_data.pkl, models/ctgan_model.pkl
Output: models/model_c.pkl, results/model_c_metrics.pkl
"""
import pickle

from config import CTGAN_MODEL_FILE, MODEL_C_FILE, MODEL_C_METRICS_FILE, SPLIT_FILE
from model_utils import build_aug, evaluate, load_ctgan, train_xgb

DEFAULT_RATIO = 0.20

if __name__ == "__main__":
    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)
    ctgan_model = load_ctgan(CTGAN_MODEL_FILE)

    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    rare_train_df = split["rare_train_df"]

    X_aug, y_aug, n_C = build_aug(X_train, y_train, rare_train_df, ctgan_model, DEFAULT_RATIO)
    print(f"Augmented: {len(X_train):,} real + {n_C} synthetic = {len(X_aug):,} total")

    model_C = train_xgb(X_aug, y_aug, f"Model C ({DEFAULT_RATIO:.0%})")
    res_C_full = evaluate(model_C, X_test, y_test, "C-full")
    res_C_rare = evaluate(model_C, X_rare_test, y_rare_test, "C-rare")
    print(f"Model C rare: log_loss={res_C_rare['log_loss']}  brier={res_C_rare['brier']}  "
          f"AUC={res_C_rare['auc']}  calib={res_C_rare['calib_err']}")

    with open(MODEL_C_FILE, "wb") as f:
        pickle.dump(model_C, f)
    with open(MODEL_C_METRICS_FILE, "wb") as f:
        pickle.dump({"res_C_full": res_C_full, "res_C_rare": res_C_rare, "ratio": DEFAULT_RATIO}, f)

    print(f"\n✔ Saved → {MODEL_C_FILE}")
    print(f"✔ Saved → {MODEL_C_METRICS_FILE}")
    print("Next: python 08_ablation_d.py")
