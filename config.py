"""
config.py — Shared configuration for the CTGAN xG augmentation pipeline.

Every stage script (01_... through 10_...) imports this first. Only place
you should need to edit paths is the block marked below.
"""
import os
import random
import warnings

import numpy as np
import matplotlib

warnings.filterwarnings("ignore")

# ── Full seed fix ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print("  ✔ PyTorch seed fixed")
except ImportError:
    print("  ℹ PyTorch not found — CTGAN will use CPU defaults")

matplotlib.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ════════════════════════════════════════════════════════════════
# ✏  EDIT THIS ONE PATH FOR YOUR ENVIRONMENT (WSL / HPC / etc.)
#    Everything else derives from it. Defaults to the directory this
#    file lives in, so the repo works in-place with no env var set.
BASE_DIR = os.environ.get("XG_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
# ════════════════════════════════════════════════════════════════

DATA_DIR    = os.path.join(BASE_DIR, "data")      # cached/intermediate dataframes
# Raw Wyscout JSONs. NOT flat — the real layout is data/WyScout/{events,matches}/,
# so stage 01 globs recursively from here rather than listing this dir directly.
WYSCOUT_DIR = os.path.join(DATA_DIR, "WyScout")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

for _d in (DATA_DIR, FIG_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Stage hand-off artifact paths ──────────────────────────────────
CACHE_FILE             = os.path.join(DATA_DIR, "shots_cache.csv")          # stage 01 output
FEATURES_FILE          = os.path.join(DATA_DIR, "shots_features.csv")       # stage 02 output
SPLIT_FILE             = os.path.join(DATA_DIR, "split_data.pkl")           # stage 03 output
BASELINE_MODELS_FILE   = os.path.join(MODELS_DIR, "baseline_models.pkl")    # stage 04 output
BASELINE_METRICS_FILE  = os.path.join(RESULTS_DIR, "baseline_metrics.pkl")  # stage 04 output
CTGAN_MODEL_FILE        = os.path.join(MODELS_DIR, "ctgan_model.pkl")       # stage 05 output
CTGAN_TRAIN_FILE        = os.path.join(DATA_DIR, "ctgan_train_df.csv")      # stage 05 output
MODEL_C_FILE            = os.path.join(MODELS_DIR, "model_c.pkl")           # stage 07 output
MODEL_C_METRICS_FILE    = os.path.join(RESULTS_DIR, "model_c_metrics.pkl")  # stage 07 output
ABLATION_D_FILE         = os.path.join(RESULTS_DIR, "ablation_D_mean_std.csv")   # stage 08 output
ABLATION_E_FILE         = os.path.join(RESULTS_DIR, "ablation_E_zero_shot.csv")  # stage 09 output

# ── Feature set used everywhere downstream of stage 02 ─────────────
XG_FEATURES = ["distance", "angle_deg", "is_header", "is_freekick",
               "goal_diff_clipped", "pressure_index"]

# ── Plot colors (kept identical to the original notebook) ──────────
C_ALL, C_RARE, C_LINE = "#4C72B0", "#C44E52", "#DD8452"
C_GOAL, C_SYNTH, C_B  = "#2ca02c", "#8172B3", "#DD8452"
C_C, C_E              = "#55A868", "#9467bd"

print(f"✔ Config loaded | SEED={SEED} | BASE_DIR={BASE_DIR}")
