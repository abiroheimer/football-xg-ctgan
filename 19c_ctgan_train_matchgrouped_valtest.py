"""
19c_ctgan_train_matchgrouped_valtest.py — CTGAN generator trained on the
train/val/test split's (19b) rare_train_df. Same hyperparameters/methodology
as 05_ctgan_train.py / 18c.

Run standalone:
    python 19c_ctgan_train_matchgrouped_valtest.py

Input : data/split_data_matchgrouped_valtest.pkl
Output: models/ctgan_model_matchgrouped_valtest.pkl
"""
import os
import pickle

from config import DATA_DIR, MODELS_DIR
from matchgrouped_utils import train_ctgan_on_rare

SPLIT_VT_FILE = os.path.join(DATA_DIR, "split_data_matchgrouped_valtest.pkl")
CTGAN_MODEL_VT_FILE = os.path.join(MODELS_DIR, "ctgan_model_matchgrouped_valtest.pkl")

if __name__ == "__main__":
    with open(SPLIT_VT_FILE, "rb") as f:
        split = pickle.load(f)

    ctgan_model = train_ctgan_on_rare(split["rare_train_df"], label="[matchgrouped valtest]")

    try:
        ctgan_model.save(CTGAN_MODEL_VT_FILE)
    except Exception:
        with open(CTGAN_MODEL_VT_FILE, "wb") as f:
            pickle.dump(ctgan_model, f)
    print(f"✔ Saved → {CTGAN_MODEL_VT_FILE}")
    print("Next: python 19d_ratio_selection_matchgrouped_valtest.py")
