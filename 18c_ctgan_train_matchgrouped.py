"""
18c_ctgan_train_matchgrouped.py — CTGAN generator re-trained on the
match-grouped split's rare_train_df (18a). Mirrors 05_ctgan_train.py exactly
(same hyperparameters/methodology); only the input split changes.

Run standalone:
    python 18c_ctgan_train_matchgrouped.py

Input : data/split_data_matchgrouped.pkl
Output: models/ctgan_model_matchgrouped.pkl, data/ctgan_train_df_matchgrouped.csv
"""
import os
import pickle
import time

from ctgan import CTGAN

from config import DATA_DIR, MODELS_DIR

SPLIT_MG_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped.pkl")
CTGAN_MODEL_MG_FILE = os.path.join(MODELS_DIR, "ctgan_model_matchgrouped.pkl")
CTGAN_TRAIN_MG_FILE = os.path.join(DATA_DIR, "ctgan_train_df_matchgrouped.csv")

CTGAN_COLS = ["distance", "angle_deg", "is_header", "is_freekick",
              "goal_diff_clipped", "minute_bucket", "is_goal"]
CTGAN_DISC = ["is_header", "is_freekick", "goal_diff_clipped", "minute_bucket", "is_goal"]

if __name__ == "__main__":
    with open(SPLIT_MG_FILE, "rb") as f:
        split = pickle.load(f)
    rare_train_df = split["rare_train_df"]

    ctgan_df = rare_train_df[CTGAN_COLS].copy()
    for c in ["is_header", "is_freekick", "is_goal"]:
        ctgan_df[c] = ctgan_df[c].astype(int)
    ctgan_df["goal_diff_clipped"] = ctgan_df["goal_diff_clipped"].astype(int)
    ctgan_df = ctgan_df.dropna()
    ctgan_df.to_csv(CTGAN_TRAIN_MG_FILE, index=False)

    batch_sz = min(500, max(50, len(ctgan_df) // 2))
    print(f"CTGAN training on {len(ctgan_df)} rare shots [matchgrouped] | batch={batch_sz} | epochs=500\n")

    ctgan_model = CTGAN(
        epochs=500, batch_size=batch_sz,
        generator_dim=(256, 256), discriminator_dim=(256, 256),
        generator_lr=2e-4, discriminator_lr=2e-4,
        discriminator_steps=1, log_frequency=True, verbose=True, cuda=True,
    )

    t0 = time.time()
    ctgan_model.fit(ctgan_df, discrete_columns=CTGAN_DISC)
    print(f"\n✔ CTGAN trained in {time.time() - t0:.0f}s")

    try:
        ctgan_model.save(CTGAN_MODEL_MG_FILE)
    except Exception:
        with open(CTGAN_MODEL_MG_FILE, "wb") as f:
            pickle.dump(ctgan_model, f)

    print(f"✔ Saved → {CTGAN_MODEL_MG_FILE}")
    print(f"✔ Saved → {CTGAN_TRAIN_MG_FILE}")
    print("Next: python 18d_ablation_d_matchgrouped.py")
