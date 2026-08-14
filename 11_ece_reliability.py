"""
11_ece_reliability.py — Step 3 Fix 2: Expected Calibration Error (ECE) +
reliability diagrams, computed alongside (not replacing) the paper's original
aggregate calibration-error metric.

Compares, all on the 269-shot rare test slice:
    Baseline A, Model C (CTGAN 80%), Model C (CTGAN 500%),
    SMOTE 80%, SMOTE 500%

Run standalone:
    python 11_ece_reliability.py

Input : data/split_data.pkl, models/baseline_models.pkl, models/ctgan_model.pkl
Output: results/ece_reliability.csv, figures/fig_ece_reliability.png
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (BASELINE_MODELS_FILE, C_ALL, C_B, C_RARE, CTGAN_MODEL_FILE,
                     FIG_DIR, RESULTS_DIR, SEED, SPLIT_FILE)
from model_utils import build_aug, build_smote_aug, evaluate, load_ctgan, reliability_bins, train_xgb

ECE_FILE = os.path.join(RESULTS_DIR, "ece_reliability.csv")
FIG_PATH = os.path.join(FIG_DIR, "fig_ece_reliability.png")
N_BINS = 10

if __name__ == "__main__":
    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)
    with open(BASELINE_MODELS_FILE, "rb") as f:
        baselines = pickle.load(f)
    ctgan_model = load_ctgan(CTGAN_MODEL_FILE)

    X_train, y_train = split["X_train"], split["y_train"]
    X_rare_train, y_rare_train = split["X_rare_train"], split["y_rare_train"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    rare_train_df = split["rare_train_df"]

    model_A = baselines["model_A"]

    np.random.seed(SEED)
    Xc80, yc80, _ = build_aug(X_train, y_train, rare_train_df, ctgan_model, 0.80)
    model_C80 = train_xgb(Xc80, yc80, "Model C (CTGAN 80%)")

    np.random.seed(SEED)
    Xc500, yc500, _ = build_aug(X_train, y_train, rare_train_df, ctgan_model, 5.00)
    model_C500 = train_xgb(Xc500, yc500, "Model C (CTGAN 500%)")

    Xs80, ys80, _ = build_smote_aug(X_train, y_train, X_rare_train, y_rare_train, 0.80, seed=SEED)
    model_S80 = train_xgb(Xs80, ys80, "SMOTE 80%")

    Xs500, ys500, _ = build_smote_aug(X_train, y_train, X_rare_train, y_rare_train, 5.00, seed=SEED)
    model_S500 = train_xgb(Xs500, ys500, "SMOTE 500%")

    models = {
        "Baseline A": model_A,
        "Model C (CTGAN 80%)": model_C80,
        "Model C (CTGAN 500%)": model_C500,
        "SMOTE 80%": model_S80,
        "SMOTE 500%": model_S500,
    }

    print(f"\n{'Model':<24} {'LogLoss':>9} {'Brier':>9} {'AUC':>9} {'CalibErr':>10} {'ECE':>9}")
    print("-" * 72)
    rows = []
    proba = {}
    for name, m in models.items():
        r = evaluate(m, X_rare_test, y_rare_test, name)
        proba[name] = r["proba"]
        rows.append({"model": name, "log_loss": r["log_loss"], "brier": r["brier"],
                      "auc": r["auc"], "calib_err": r["calib_err"], "ece": r["ece"]})
        print(f"{name:<24} {r['log_loss']:>9.4f} {r['brier']:>9.4f} {r['auc']:>9.4f} "
              f"{r['calib_err']:>10.4f} {r['ece']:>9.4f}")

    pd.DataFrame(rows).to_csv(ECE_FILE, index=False)
    print(f"\n✔ Saved → {ECE_FILE}")

    # ── Reliability diagram: Baseline A vs Model C (80%) vs Model C (500%) ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    colors = {"Baseline A": C_ALL, "Model C (CTGAN 80%)": C_RARE, "Model C (CTGAN 500%)": C_B}
    for name in ["Baseline A", "Model C (CTGAN 80%)", "Model C (CTGAN 500%)"]:
        bins = reliability_bins(y_rare_test, proba[name], n_bins=N_BINS)
        xs = [b["mean_pred"] for b in bins if b["n"] > 0]
        ys = [b["actual_rate"] for b in bins if b["n"] > 0]
        ns = [b["n"] for b in bins if b["n"] > 0]
        ax.plot(xs, ys, "o-", color=colors[name], label=name, markersize=5)
    ax.set_xlabel("Mean predicted xG (bin)")
    ax.set_ylabel("Actual conversion rate (bin)")
    ax.set_title(f"Reliability Diagram — {N_BINS}-bin ECE, 269-shot rare test slice",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"✔ Saved → {FIG_PATH}")
