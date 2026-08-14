"""
05_ctgan_train.py — Stage 5: train CTGAN exclusively on real rare-scenario
training shots.

Run standalone:
    python 05_ctgan_train.py

Input : data/split_data.pkl        (stage 03 output, needs rare_train_df)
Output: models/ctgan_model.pkl, data/ctgan_train_df.csv
"""
import pickle
import time

from ctgan import CTGAN

from config import CTGAN_MODEL_FILE, CTGAN_TRAIN_FILE, SPLIT_FILE

# pressure_index is intentionally NOT a CTGAN generation column (Step 3 Fix 3):
# it's mathematically derived from goal_diff and minute, both of which CTGAN
# already conditions on via goal_diff_clipped/minute_bucket. Sampling it as an
# independent column could produce internally-inconsistent synthetic rows
# (e.g. goal_diff_clipped=-2, minute_bucket="75-90", pressure_index=0.5, which
# no real shot could have). model_utils.sample_synthetic() recomputes it
# deterministically post-generation from the sampled goal_diff_clipped and
# minute_bucket instead — see compute_pressure_index() there.
CTGAN_COLS = ["distance", "angle_deg", "is_header", "is_freekick",
              "goal_diff_clipped", "minute_bucket", "is_goal"]
CTGAN_DISC = ["is_header", "is_freekick", "goal_diff_clipped", "minute_bucket", "is_goal"]

if __name__ == "__main__":
    with open(SPLIT_FILE, "rb") as f:
        split = pickle.load(f)
    rare_train_df = split["rare_train_df"]

    ctgan_df = rare_train_df[CTGAN_COLS].copy()
    for c in ["is_header", "is_freekick", "is_goal"]:
        ctgan_df[c] = ctgan_df[c].astype(int)
    ctgan_df["goal_diff_clipped"] = ctgan_df["goal_diff_clipped"].astype(int)
    ctgan_df = ctgan_df.dropna()
    ctgan_df.to_csv(CTGAN_TRAIN_FILE, index=False)

    batch_sz = min(500, max(50, len(ctgan_df) // 2))
    print(f"CTGAN training on {len(ctgan_df)} rare shots | batch={batch_sz} | epochs=500\n")

    ctgan_model = CTGAN(
        epochs=500, batch_size=batch_sz,
        generator_dim=(256, 256), discriminator_dim=(256, 256),
        generator_lr=2e-4, discriminator_lr=2e-4,
        discriminator_steps=1, log_frequency=True, verbose=True, cuda=True,
    )

    t0 = time.time()
    ctgan_model.fit(ctgan_df, discrete_columns=CTGAN_DISC)
    print(f"\n✔ CTGAN trained in {time.time() - t0:.0f}s")

    # ctgan's CTGAN objects support .save()/CTGAN.load(); more robust than raw pickle
    try:
        ctgan_model.save(CTGAN_MODEL_FILE)
    except Exception:
        with open(CTGAN_MODEL_FILE, "wb") as f:
            pickle.dump(ctgan_model, f)

    print(f"✔ Saved → {CTGAN_MODEL_FILE}")
    print(f"✔ Saved → {CTGAN_TRAIN_FILE}")
    print("Next: python 06_synthetic_quality.py")
