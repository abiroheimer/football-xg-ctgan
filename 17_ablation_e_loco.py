"""
17_ablation_e_loco.py — Leave-One-Competition-Out (LOCO) cold-start redesign
of Ablation E, replacing the leaky "zero-shot" framing flagged by external
review (Points 2 & 3):

  Point 2 (naming): the old Ablation E's CTGAN generator was trained on real
  rare shots from the SAME competitions it was evaluated on, so "zero-shot"
  overstated what was tested. Here, for each held-out competition G, the
  generator sees zero real rare shots from G, at any point.

  Point 3 (leakage): the old Baseline A was trained on all real data,
  including ~80% of the rare shots it was then evaluated against. Here each
  fold gets its own "fair" Baseline A trained on all real data EXCEPT the
  held-out competition's rare shots.

Design (confirmed with user):
  - Primary folds (5): England, France, Germany, Italy, Spain — each has
    enough rare shots (208-282) and rare goals (18-30) for a stable
    held-out AUC/calibration estimate.
  - Excluded from primary folds: World_Cup (28 rare / 2 goals) and
    European_Championship (33 rare / 2 goals) — too few positives for a
    trustworthy per-fold estimate (Step A audit). They still contribute to
    the OTHER folds' generator/baseline training pools as normal.
  - Supplementary fold (reported separately, NOT in the primary aggregate):
    World_Cup + European_Championship merged into one held-out fold (61
    rare shots / 4 goals combined) — still thin, explicit low-confidence
    caveat in the output.
  - Held-out competition's NORMAL (non-rare) shots stay in the downstream
    classifier's training set for every fold (matches original Ablation E
    scope: the claim is about rare-scenario generalization specifically,
    not full-competition generalization).

Per fold G:
  - CTGAN generator: trained only on real rare shots from competitions
    OTHER than G (or, for the merged fold, other than {World_Cup, Euro}).
  - Downstream classifier training: normal shots from ALL 7 competitions +
    CTGAN-synthetic rare shots sampled from G's generator. Zero real rare
    shots of any kind.
  - Fold's fair Baseline A: normal shots from all 7 + REAL rare shots from
    the other competitions (G's real rare shots excluded). Deterministic,
    single run, seed=42.
  - Evaluation set: G's real rare shots (held out, untouched by training).

Run standalone (after stages 01-04 have produced data/split_data.pkl,
models/baseline_models.pkl is NOT reused — this script trains its own
fold-specific baselines and generators from scratch):
    python 17_ablation_e_loco.py

Input : data/shots_features.csv
Output: results/ablation_E_loco_per_fold.csv       (primary 5-fold summary)
        results/ablation_E_loco_raw_runs.csv        (raw per-run values, all folds incl. supplementary)
        results/ablation_E_loco_macro.csv           (macro-averaged headline number, primary folds only)
        results/ablation_E_loco_supplementary.csv   (WC+Euro merged fold, reported separately)
"""
import os
import time

import numpy as np
import pandas as pd
from ctgan import CTGAN

from config import FEATURES_FILE, RESULTS_DIR, SEED, XG_FEATURES
from model_utils import evaluate, sample_synthetic, train_xgb

PER_FOLD_FILE   = os.path.join(RESULTS_DIR, "ablation_E_loco_per_fold.csv")
RAW_FILE        = os.path.join(RESULTS_DIR, "ablation_E_loco_raw_runs.csv")
MACRO_FILE      = os.path.join(RESULTS_DIR, "ablation_E_loco_macro.csv")
SUPP_FILE       = os.path.join(RESULTS_DIR, "ablation_E_loco_supplementary.csv")

PRIMARY_FOLDS = ["England", "France", "Germany", "Italy", "Spain"]
SUPP_FOLD_NAME = "World_Cup+European_Championship"
SUPP_FOLD_COMPS = ["World_Cup", "European_Championship"]

E_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
N_RUNS_E = 5

CTGAN_COLS = ["distance", "angle_deg", "is_header", "is_freekick",
              "goal_diff_clipped", "minute_bucket", "is_goal"]
CTGAN_DISC = ["is_header", "is_freekick", "goal_diff_clipped", "minute_bucket", "is_goal"]


def make_ctgan_train_df(rare_df):
    df = rare_df[CTGAN_COLS].copy()
    for c in ["is_header", "is_freekick", "is_goal"]:
        df[c] = df[c].astype(int)
    df["goal_diff_clipped"] = df["goal_diff_clipped"].astype(int)
    return df.dropna()


def train_ctgan(rare_df, label=""):
    ctgan_df = make_ctgan_train_df(rare_df)
    batch_sz = min(500, max(50, len(ctgan_df) // 2))
    print(f"  Training CTGAN generator on {len(ctgan_df)} rare shots "
          f"(excludes held-out fold) | batch={batch_sz} | epochs=500 {label}")
    m = CTGAN(
        epochs=500, batch_size=batch_sz,
        generator_dim=(256, 256), discriminator_dim=(256, 256),
        generator_lr=2e-4, discriminator_lr=2e-4,
        discriminator_steps=1, log_frequency=True, verbose=False, cuda=True,
    )
    t0 = time.time()
    m.fit(ctgan_df, discrete_columns=CTGAN_DISC)
    print(f"  -> CTGAN fold generator trained in {time.time() - t0:.0f}s")
    return m


def run_fold(fold_name, holdout_comps, model_df):
    """holdout_comps: list of competition names excluded from generator +
    fair-baseline training, and whose rare shots form the eval set."""
    is_holdout = model_df["competition"].isin(holdout_comps)
    is_rare = model_df["is_rare"] == 1

    rare_holdout_df = model_df[is_holdout & is_rare].copy()          # eval set
    rare_other_df   = model_df[(~is_holdout) & is_rare].copy()       # generator pool + fair-baseline rare
    normal_all_df   = model_df[~is_rare].copy()                      # normal shots, ALL 7 competitions

    X_eval = rare_holdout_df[XG_FEATURES].values
    y_eval = rare_holdout_df["is_goal"].values

    X_norm_all = normal_all_df[XG_FEATURES].values
    y_norm_all = normal_all_df["is_goal"].values

    X_rare_other = rare_other_df[XG_FEATURES].values
    y_rare_other = rare_other_df["is_goal"].values

    print(f"\n{'=' * 72}\nFOLD: {fold_name}  (held out: {holdout_comps})")
    print(f"  Eval (real rare, held out)      : {len(X_eval):,} shots, {int(y_eval.sum())} goals")
    print(f"  Generator train (real rare,other): {len(X_rare_other):,} shots, {int(y_rare_other.sum())} goals")
    print(f"  Downstream normal-shot pool      : {len(X_norm_all):,} shots (all 7 competitions)")

    if len(X_eval) == 0 or len(np.unique(y_eval)) < 2:
        print(f"  ⚠ Fold {fold_name}: eval set unusable (no positive/negative variety), skipping")
        return None

    # ── Fold-specific fair Baseline A: all real data except this fold's rare shots ──
    Xa = np.vstack([X_norm_all, X_rare_other])
    ya = np.concatenate([y_norm_all, y_rare_other])
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
    except Exception:
        pass
    model_A_fold = train_xgb(Xa, ya, f"Fair Baseline A [{fold_name}]")
    res_A_fold = evaluate(model_A_fold, X_eval, y_eval, f"A-fold-{fold_name}")
    print(f"  Fair Baseline A: log_loss={res_A_fold['log_loss']}  brier={res_A_fold['brier']}  "
          f"AUC={res_A_fold['auc']}  calib={res_A_fold['calib_err']}")

    # ── CTGAN generator trained only on the other competitions' real rare shots ──
    ctgan_model = train_ctgan(rare_other_df, label=f"[{fold_name}]")

    run_results = {r: [] for r in E_RATIOS}
    print(f"\n  Running {len(E_RATIOS)} ratios x {N_RUNS_E} runs for fold {fold_name}")
    print(f"  {'Ratio':<8} {'Run':<5} {'LogLoss':<10} {'Brier':<10} {'AUC':<8} {'Calib'}")
    print("  " + "-" * 50)

    for ratio in E_RATIOS:
        for run in range(N_RUNS_E):
            np.random.seed(SEED + run + 100)
            try:
                import torch
                torch.manual_seed(SEED + run + 100)
            except Exception:
                pass

            n_synth = max(10, int(len(rare_other_df) * ratio))
            s = sample_synthetic(ctgan_model, n_synth, rare_other_df)
            Xs = s[XG_FEATURES].values
            ys = s["is_goal"].values

            Xe = np.vstack([X_norm_all, Xs])
            ye = np.concatenate([y_norm_all, ys])

            me = train_xgb(Xe, ye, "")
            re = evaluate(me, X_eval, y_eval, "")
            if re:
                run_results[ratio].append({
                    "ll": re["log_loss"], "brier": re["brier"],
                    "auc": re["auc"], "calib": re["calib_err"], "ece": re["ece"],
                })
                print(f"  {ratio:<8.1%} {run + 1:<5} {re['log_loss']:<10.4f} {re['brier']:<10.4f} "
                      f"{re['auc']:<8.4f} {re['calib_err']:.4f}")

    np.random.seed(SEED)  # reset

    fold_rows = []
    raw_rows = []
    for ratio in E_RATIOS:
        runs = run_results[ratio]
        if not runs:
            continue
        for i, r in enumerate(runs):
            raw_rows.append({"fold": fold_name, "ratio": ratio, "run": i, **r})
        ns = max(10, int(len(rare_other_df) * ratio))
        row = {"fold": fold_name, "ratio": ratio, "n_synth": ns,
               "n_eval": len(X_eval), "n_eval_goals": int(y_eval.sum()),
               "baseA_log_loss": res_A_fold["log_loss"], "baseA_brier": res_A_fold["brier"],
               "baseA_auc": res_A_fold["auc"], "baseA_calib_err": res_A_fold["calib_err"]}
        for metric, col in [("ll", "log_loss"), ("brier", "brier"), ("auc", "auc"),
                             ("calib", "calib_err"), ("ece", "ece")]:
            vals = [r[metric] for r in runs]
            row[f"{col}_mean"] = round(np.mean(vals), 4)
            row[f"{col}_std"] = round(np.std(vals), 4)
        fold_rows.append(row)

    return {"fold_rows": fold_rows, "raw_rows": raw_rows, "res_A_fold": res_A_fold,
            "n_eval": len(X_eval), "n_eval_goals": int(y_eval.sum())}


if __name__ == "__main__":
    print(f"Loading {FEATURES_FILE} ...")
    shots = pd.read_csv(FEATURES_FILE)
    model_df = shots.dropna(subset=XG_FEATURES + ["is_goal"]).reset_index(drop=True)
    print(f"Total usable shots: {len(model_df):,}  |  total rare: {(model_df['is_rare'] == 1).sum():,}")

    all_fold_rows = []
    all_raw_rows = []
    supp_fold_rows = []
    supp_raw_rows = []

    for comp in PRIMARY_FOLDS:
        result = run_fold(comp, [comp], model_df)
        if result is None:
            continue
        all_fold_rows.extend(result["fold_rows"])
        all_raw_rows.extend(result["raw_rows"])

    # ── Supplementary low-confidence fold: World_Cup + European_Championship merged ──
    print(f"\n{'#' * 72}\nSUPPLEMENTARY FOLD (low-confidence, NOT in primary aggregate): "
          f"{SUPP_FOLD_NAME}")
    supp_result = run_fold(SUPP_FOLD_NAME, SUPP_FOLD_COMPS, model_df)
    if supp_result is not None:
        supp_fold_rows.extend(supp_result["fold_rows"])
        supp_raw_rows.extend(supp_result["raw_rows"])

    # ── Save raw runs (primary + supplementary, clearly tagged by fold) ──
    raw_df = pd.DataFrame(all_raw_rows + supp_raw_rows)
    raw_df.to_csv(RAW_FILE, index=False)
    print(f"\n✔ Saved raw per-run results (all folds) → {RAW_FILE}")

    # ── Per-fold summary (primary 5 folds only) ──
    per_fold_df = pd.DataFrame(all_fold_rows)
    per_fold_df.to_csv(PER_FOLD_FILE, index=False)
    print(f"✔ Saved per-fold summary (primary 5 folds) → {PER_FOLD_FILE}")

    # ── Supplementary fold summary, saved separately ──
    supp_df = pd.DataFrame(supp_fold_rows)
    supp_df.to_csv(SUPP_FILE, index=False)
    print(f"✔ Saved supplementary (WC+Euro merged) fold summary → {SUPP_FILE}")

    # ── Macro-average across primary folds, per ratio ──
    print(f"\n{'=' * 72}\nMACRO-AVERAGED HEADLINE (5 primary folds: {PRIMARY_FOLDS})\n{'=' * 72}")
    macro_rows = []
    for ratio in E_RATIOS:
        sub = per_fold_df[per_fold_df["ratio"] == ratio]
        if sub.empty:
            continue
        row = {"ratio": ratio, "n_folds": len(sub)}
        for col in ["log_loss_mean", "brier_mean", "auc_mean", "calib_err_mean", "ece_mean",
                    "baseA_log_loss", "baseA_brier", "baseA_auc", "baseA_calib_err"]:
            row[f"macro_{col}"] = round(sub[col].mean(), 4)
            row[f"macro_{col}_across_fold_std"] = round(sub[col].std(), 4)
        macro_rows.append(row)
        print(f"Ratio {ratio:.1%}: LOCO logloss={row['macro_log_loss_mean']:.4f} "
              f"(±{row['macro_log_loss_mean_across_fold_std']:.4f} across folds)  "
              f"AUC={row['macro_auc_mean']:.4f}  calib={row['macro_calib_err_mean']:.4f}  "
              f"| fair-baseA calib={row['macro_baseA_calib_err']:.4f}")

    macro_df = pd.DataFrame(macro_rows)
    macro_df.to_csv(MACRO_FILE, index=False)
    print(f"\n✔ Saved macro-averaged headline → {MACRO_FILE}")

    # ── Flag folds diverging sharply from the others ──
    print(f"\n{'=' * 72}\nDIVERGENCE CHECK (per fold, ratio=100% as representative)\n{'=' * 72}")
    rep = per_fold_df[per_fold_df["ratio"] == 1.00]
    if not rep.empty:
        auc_mean, auc_std = rep["auc_mean"].mean(), rep["auc_mean"].std()
        for _, r in rep.iterrows():
            z = (r["auc_mean"] - auc_mean) / auc_std if auc_std > 0 else 0.0
            flag = "  <-- DIVERGES (|z|>1.5)" if abs(z) > 1.5 else ""
            print(f"  {r['fold']:<10} n_eval={int(r['n_eval']):>4} goals={int(r['n_eval_goals']):>3} "
                  f"AUC={r['auc_mean']:.4f} z={z:+.2f}{flag}")

    print("\nDone. Review results/ablation_E_loco_*.csv before any paper-text changes.")
