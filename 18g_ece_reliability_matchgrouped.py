"""
18g_ece_reliability_matchgrouped.py — ECE/reliability numbers re-run on the
match-grouped split, for Baseline A, CTGAN 80%, and CTGAN 500% (the three
models the task asked to re-check here; SMOTE 80%/500% ECE was not
requested for this pass — see 11_ece_reliability.py for the full 5-model
version on the original split).

Run standalone:
    python 18g_ece_reliability_matchgrouped.py

Input : data/split_data_matchgrouped.pkl, models/baseline_models_matchgrouped.pkl,
        models/ctgan_model_matchgrouped.pkl
Output: results/ece_reliability_matchgrouped.csv, figures/fig_ece_reliability_matchgrouped.png
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import C_ALL, C_B, C_RARE, DATA_DIR, FIG_DIR, MODELS_DIR, RESULTS_DIR, SEED
from model_utils import build_aug, evaluate, load_ctgan, reliability_bins, train_xgb

SPLIT_MG_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped.pkl")
BASELINE_MODELS_MG_FILE = os.path.join(MODELS_DIR, "baseline_models_matchgrouped.pkl")
CTGAN_MODEL_MG_FILE = os.path.join(MODELS_DIR, "ctgan_model_matchgrouped.pkl")
ECE_MG_FILE = os.path.join(RESULTS_DIR, "ece_reliability_matchgrouped.csv")
FIG_MG_PATH = os.path.join(FIG_DIR, "fig_ece_reliability_matchgrouped.png")
N_BINS = 10

if __name__ == "__main__":
    with open(SPLIT_MG_FILE, "rb") as f:
        split = pickle.load(f)
    with open(BASELINE_MODELS_MG_FILE, "rb") as f:
        baselines = pickle.load(f)
    ctgan_model = load_ctgan(CTGAN_MODEL_MG_FILE)

    X_train, y_train = split["X_train"], split["y_train"]
    X_rare_test, y_rare_test = split["X_rare_test"], split["y_rare_test"]
    rare_train_df = split["rare_train_df"]

    model_A = baselines["model_A"]

    np.random.seed(SEED)
    Xc80, yc80, _ = build_aug(X_train, y_train, rare_train_df, ctgan_model, 0.80)
    model_C80 = train_xgb(Xc80, yc80, "Model C (CTGAN 80%) [matchgrouped]")

    np.random.seed(SEED)
    Xc500, yc500, _ = build_aug(X_train, y_train, rare_train_df, ctgan_model, 5.00)
    model_C500 = train_xgb(Xc500, yc500, "Model C (CTGAN 500%) [matchgrouped]")

    models = {
        "Baseline A": model_A,
        "Model C (CTGAN 80%)": model_C80,
        "Model C (CTGAN 500%)": model_C500,
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

    pd.DataFrame(rows).to_csv(ECE_MG_FILE, index=False)
    print(f"\n✔ Saved → {ECE_MG_FILE}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    colors = {"Baseline A": C_ALL, "Model C (CTGAN 80%)": C_RARE, "Model C (CTGAN 500%)": C_B}
    for name in ["Baseline A", "Model C (CTGAN 80%)", "Model C (CTGAN 500%)"]:
        bins = reliability_bins(y_rare_test, proba[name], n_bins=N_BINS)
        xs = [b["mean_pred"] for b in bins if b["n"] > 0]
        ys = [b["actual_rate"] for b in bins if b["n"] > 0]
        ax.plot(xs, ys, "o-", color=colors[name], label=name, markersize=5)
    ax.set_xlabel("Mean predicted xG (bin)")
    ax.set_ylabel("Actual conversion rate (bin)")
    ax.set_title(f"Reliability Diagram [match-grouped split] — {N_BINS}-bin ECE, "
                 f"{len(y_rare_test)}-shot rare test slice", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_MG_PATH, dpi=150, bbox_inches="tight")
    print(f"✔ Saved → {FIG_MG_PATH}")
    print("Next: python 18h_significance_tests_matchgrouped.py")
