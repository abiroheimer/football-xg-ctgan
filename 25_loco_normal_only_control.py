"""
25_loco_normal_only_control.py — Block 1: the normal-only LOCO control
(Model N), a third condition alongside the fair baseline B_G and the
cold-start model C_G.

Model N trains on normal (non-rare) shots from all 7 competitions ONLY --
zero rare shots of any kind, real or synthetic. It is the control that
separates two competing explanations for C_G's calibration gain:

  (a) the synthetic rare shots carry genuine rare-event information, or
  (b) C_G simply benefits from having *any* extra training signal near the
      rare-scenario region, in which case a model with no rare shots at all
      would do about as well.

Hoped-for ordering in calibration error (lower is better):
      C_G  <  B_G   and   C_G  <  Model N
i.e. Model N worst. Whatever the data says is what gets reported.

Model N is deterministic (one XGBoost fit per fold, seed 42, no synthetic
sampling), so it is a single run per fold, matching how the fold baselines
are already handled in 17_ablation_e_loco.py.

Run standalone:
    python 25_loco_normal_only_control.py

Input : data/shots_features.csv
Output: results/loco_normal_only_control.csv
"""
import os

import numpy as np
import pandas as pd

from config import FEATURES_FILE, RESULTS_DIR, SEED, XG_FEATURES
from model_utils import evaluate, train_xgb

OUT_FILE = os.path.join(RESULTS_DIR, "loco_normal_only_control.csv")
PER_FOLD_FILE = os.path.join(RESULTS_DIR, "ablation_E_loco_per_fold.csv")

PRIMARY_FOLDS = ["England", "France", "Germany", "Italy", "Spain"]
HEADLINE_RATIO = 0.05

if __name__ == "__main__":
    print(f"Loading {FEATURES_FILE} ...")
    shots = pd.read_csv(FEATURES_FILE)
    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)

    is_rare = model_df["is_rare"] == 1
    normal_all_df = model_df[~is_rare]
    X_norm_all = normal_all_df[XG_FEATURES].values
    y_norm_all = normal_all_df["is_goal"].values
    print(f"Model N training set (identical for every fold): {len(X_norm_all):,} normal shots, "
          f"0 rare shots of any kind\n")

    # Existing B_G / C_G numbers at the headline ratio, for the ordering check.
    per_fold = pd.read_csv(PER_FOLD_FILE)
    ref = per_fold[per_fold.ratio == HEADLINE_RATIO].set_index("fold")

    # Model N's training set does not depend on the fold (normal shots from all
    # 7 competitions, no rare shots) -- only the evaluation set changes. It is
    # therefore fit once and evaluated per fold, which is exactly equivalent to
    # refitting per fold with a fixed seed, and avoids implying otherwise.
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
    except Exception:
        pass
    model_N = train_xgb(X_norm_all, y_norm_all, "Model N (normal-only)")

    rows = []
    print(f"\n{'Fold':<10} {'n(goals)':<12} {'N calib':<10} {'C_G calib':<11} {'B_G calib':<11} {'ordering'}")
    print("-" * 74)
    for fold in PRIMARY_FOLDS:
        eval_df = model_df[(model_df["competition"] == fold) & is_rare]
        X_eval = eval_df[XG_FEATURES].values
        y_eval = eval_df["is_goal"].values

        r = evaluate(model_N, X_eval, y_eval, f"N-{fold}")
        c_calib = float(ref.loc[fold, "calib_err_mean"])
        b_calib = float(ref.loc[fold, "baseA_calib_err"])
        c_ll = float(ref.loc[fold, "log_loss_mean"])
        b_ll = float(ref.loc[fold, "baseA_log_loss"])

        # Does the hoped-for ordering hold for this fold? (C_G better than N)
        cg_beats_n = c_calib < r["calib_err"]
        rows.append({
            "fold": fold, "n_eval": len(y_eval), "n_eval_goals": int(y_eval.sum()),
            "N_log_loss": r["log_loss"], "N_brier": r["brier"],
            "N_auc": r["auc"], "N_calib_err": r["calib_err"], "N_ece": r["ece"],
            "CG_log_loss": c_ll, "CG_calib_err": c_calib,
            "BG_log_loss": b_ll, "BG_calib_err": b_calib,
            "CG_better_than_N_calib": bool(cg_beats_n),
            "N_calib_vs_BG": round(r["calib_err"] - b_calib, 4),
        })
        order = "C_G < N (as hoped)" if cg_beats_n else "N <= C_G  <-- CONTRADICTS"
        print(f"{fold:<10} {len(y_eval)} ({int(y_eval.sum())}){'':<4} "
              f"{r['calib_err']:<10.4f} {c_calib:<11.4f} {b_calib:<11.4f} {order}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)

    # ── Macro (unweighted across the 5 competitions, consistent with the paper) ──
    macro_N_calib = df["N_calib_err"].mean()
    macro_N_ll = df["N_log_loss"].mean()
    macro_C_calib = df["CG_calib_err"].mean()
    macro_C_ll = df["CG_log_loss"].mean()
    macro_B_calib = df["BG_calib_err"].mean()
    macro_B_ll = df["BG_log_loss"].mean()

    print(f"\n{'=' * 74}\nMACRO (unweighted mean across the 5 primary folds)\n{'=' * 74}")
    print(f"{'Condition':<34} {'Calib err':<12} {'Log loss'}")
    print("-" * 74)
    print(f"{'Model N (normal-only, 0 rare)':<34} {macro_N_calib:<12.4f} {macro_N_ll:.4f}")
    print(f"{'C_G (CTGAN cold-start)':<34} {macro_C_calib:<12.4f} {macro_C_ll:.4f}")
    print(f"{'B_G (fair real-rare baseline)':<34} {macro_B_calib:<12.4f} {macro_B_ll:.4f}")

    n_folds_cg_wins = int(df["CG_better_than_N_calib"].sum())
    print(f"\nC_G better-calibrated than Model N in {n_folds_cg_wins}/{len(df)} folds.")
    print(f"Macro calibration: C_G improves on Model N by "
          f"{100 * (1 - macro_C_calib / macro_N_calib):.1f}% relative.")

    if macro_C_calib < macro_N_calib < macro_B_calib:
        verdict = ("ORDERING: C_G < N < B_G. C_G beats both; the normal-only control is "
                   "itself better calibrated than the real-rare baseline.")
    elif macro_C_calib < macro_N_calib:
        verdict = ("ORDERING: C_G < N. Synthetic rare shots add calibration value beyond "
                   "the normal-only control -- the hoped-for direction.")
    else:
        verdict = ("ORDERING CONTRADICTED: Model N is at least as well calibrated as C_G. "
                   "The cold-start gain cannot be attributed to synthetic rare-event "
                   "information on this evidence.")
    print(f"\n{verdict}")

    with open(os.path.join(RESULTS_DIR, "loco_normal_only_verdict.txt"), "w", encoding="utf-8") as f:
        f.write(verdict + "\n")
        f.write(f"macro_N_calib={macro_N_calib:.4f} macro_CG_calib={macro_C_calib:.4f} "
                f"macro_BG_calib={macro_B_calib:.4f}\n")
        f.write(f"macro_N_ll={macro_N_ll:.4f} macro_CG_ll={macro_C_ll:.4f} macro_BG_ll={macro_B_ll:.4f}\n")

    print(f"\n✔ Saved → {OUT_FILE}")
