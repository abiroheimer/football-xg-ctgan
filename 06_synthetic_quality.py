"""
06_synthetic_quality.py — Stage 6: quality-check the trained CTGAN's
synthetic output against the real rare-training distribution.

Run standalone:
    python 06_synthetic_quality.py

Input : models/ctgan_model.pkl, data/ctgan_train_df.csv   (stage 05 output)
Output: figures/fig6_quality.png
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from config import C_RARE, C_SYNTH, CTGAN_MODEL_FILE, CTGAN_TRAIN_FILE, FIG_DIR, SPLIT_FILE
from model_utils import build_bucket_minute_samples, compute_pressure_index, load_ctgan

if __name__ == "__main__":
    ctgan_model = load_ctgan(CTGAN_MODEL_FILE)
    ctgan_train_cols = pd.read_csv(CTGAN_TRAIN_FILE).columns.tolist()  # CTGAN_COLS from stage 05

    # pressure_index is no longer a CTGAN generation column (Step 3 Fix 3) —
    # it's recomputed deterministically post-sample by sample_synthetic(), so
    # QC needs the real reference values from split_data.pkl (which still
    # carries the true pressure_index column). Rebuilt with the same dropna
    # scope stage 05 used (over CTGAN_COLS only) rather than a positional
    # join, so this stays correct even if row-dropping behavior ever changes.
    with open(SPLIT_FILE, "rb") as f:
        rare_train_df = pickle.load(f)["rare_train_df"]
    ctgan_df = rare_train_df[ctgan_train_cols + ["pressure_index"]].dropna(subset=ctgan_train_cols).reset_index(drop=True)

    N_QC = min(5000, len(ctgan_df) * 3)
    # Deliberately UNCALIBRATED (no calibrate_goal_rate) — this QC step exists
    # to measure the goal-rate/domain-validity gap BEFORE post-processing
    # (paper: "synthetic goal rate before post-processing was 13.8% versus
    # the real rate of 8.8%"). Only domain clipping + the pressure_index
    # recompute (Fix 3) are applied, matching what sample_synthetic() does
    # up to (but not including) its calibrate_goal_rate() call.
    qc = ctgan_model.sample(N_QC)
    qc["distance"] = qc["distance"].clip(1, 65)
    qc["angle_deg"] = qc["angle_deg"].clip(0, 70)
    for c in ["is_goal", "is_header", "is_freekick"]:
        qc[c] = qc[c].round().clip(0, 1).astype(int)
    qc["goal_diff_clipped"] = qc["goal_diff_clipped"].round().clip(-5, 5).astype(int)
    bucket_minute_samples = build_bucket_minute_samples(rare_train_df)
    qc["pressure_index"] = compute_pressure_index(qc["goal_diff_clipped"], qc["minute_bucket"], bucket_minute_samples)
    print(f"Sampled {N_QC} synthetic shots for QC (pre-calibration; pressure_index recomputed post-sample)\n")

    checks = {
        "distance 1-65m": ((qc["distance"] >= 1) & (qc["distance"] <= 65)).mean(),
        "angle 0-70°": ((qc["angle_deg"] >= 0) & (qc["angle_deg"] <= 70)).mean(),
        "is_goal {0,1}": qc["is_goal"].isin([0, 1]).mean(),
    }
    for k, v in checks.items():
        print(f"  {'✔' if v > 0.95 else '⚠'}  {k:<20}  {100 * v:.1f}%")

    print("\n  Feature  |  KS p-val  |  JS div")
    ks_res = {}
    for feat in ["distance", "angle_deg", "pressure_index"]:
        r = ctgan_df[feat].dropna().values
        s = qc[feat].dropna().values
        _, p = ks_2samp(r, s)
        bins = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 60)
        rh, _ = np.histogram(r, bins=bins, density=True)
        sh, _ = np.histogram(s, bins=bins, density=True)
        rh = rh / rh.sum() + 1e-10
        sh = sh / sh.sum() + 1e-10
        js = jensenshannon(rh, sh)
        ks_res[feat] = {"ks_p": p, "js": js}
        print(f"  {'✔' if p > 0.05 else '⚠'}  {feat:<18}  {p:.4f}      {js:.4f}")

    real_gr = ctgan_df["is_goal"].mean()
    s_gr = qc["is_goal"].astype(float).mean()
    print(f"\n  Goal rate: real={100 * real_gr:.1f}%  synth={100 * s_gr:.1f}%  diff={100 * abs(real_gr - s_gr):.1f}pp")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Fig 6 — Synthetic vs Real Distributions", fontsize=12, fontweight="bold")
    for ax, feat in zip(axes, ["distance", "angle_deg", "pressure_index"]):
        ax.hist(ctgan_df[feat], bins=30, color=C_RARE, alpha=0.55, density=True, label="Real rare")
        ax.hist(qc[feat], bins=30, color=C_SYNTH, alpha=0.55, density=True, label="Synthetic")
        ax.set_title(f"{feat}  JS={ks_res[feat]['js']:.3f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig6_quality.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"\n✔ Saved → {out_path}")
    print("Next: python 07_model_c.py")
