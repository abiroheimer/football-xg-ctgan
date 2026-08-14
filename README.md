# Synthetic Augmentation for Rare-Scenario Football xG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Reproducibility: 81/81 Passed](https://img.shields.io/badge/Verification-81%2F81%20Passed-brightgreen.svg)](27_verify_paper_numbers.py)

> **Paper Title:** *Synthetic Augmentation for Rare-Scenario Football xG: Cross-Competition Cold-Start Transfer and the Instability of Single-Split Method Comparisons*  
> **Target Format:** IEEE Conference Format (6 pages)

---

## Overview

Expected goals ($xG$) models in professional football are predominantly trained on neutral game-state data, leaving late-game deficit situations (shots taken after minute 75 when trailing by $\ge 2$ goals) systematically underrepresented at **3.12% of all shot events**.

This repository provides the complete experimental pipeline, models, figures, and artifacts investigating:
1. **Conditional Tabular GAN (CTGAN)** and **SMOTE** synthetic data augmentation for rare-scenario shot modeling.
2. **Leave-One-Competition-Out (LOCO)** cross-competition rare-scenario cold-start evaluation across 7 European leagues and international tournaments.
3. A **Normal-Only Control ($N$)** isolating true synthetic information gain from simple rare-data withholding effects.
4. **Probability calibration focus** evaluating macro Expected Calibration Error (ECE), Log Loss, and Brier Score under leakage-free match-grouped splitting.

---

## Research Questions & Core Findings

### 1. Are CTGAN-vs-SMOTE conclusions stable across train/test splits?
- **Finding: No.** Under the initial shot-level split, SMOTE appeared superior. Under a match-grouped split, CTGAN appeared superior. Across 8 independently drawn match-grouped, validation-selected splits:
  - CTGAN wins 3/8 splits, SMOTE wins 5/8 splits.
  - Paired $t$-test: $p \approx 0.789$; Exact 8-split permutation test: $p \approx 0.750$.
  - **Conclusion:** Single-split method comparisons in augmentation research produce directionally unstable conclusions that are not statistically supported under repeated resampling.

### 2. Does synthetic rare-event augmentation improve cross-competition rare-scenario cold start?
- **Finding: Yes.** Under a LOCO cold-start protocol where the generator and classifier see zero real rare shots from the target competition:
  - Baseline $B_G$ Macro Calibration Error: **0.2120**
  - Normal-Only Control $N$ Macro Calibration Error: **0.1704** ($19.6\%$ improvement from withholding real rare shots)
  - Synthetic Augmentation $C_G$ (at 5% ratio): **0.1463** ($14.2\%$ further macro improvement from synthetic signal)
  - Overall $B_G \to C_G$ Improvement: **31.0%** ($95\%$ CI $[23.6\%, 39.3\%]$, $d_z = -3.562$, $t = -7.965$, $p = 0.0013$).
- Under a **fixed-weight sensitivity check** ($scale\_pos\_weight$ held constant), $C_G$ maintains a **14.6%** calibration improvement over $N$ (positive in 5/5 folds, $t = 4.804$, $p = 0.009$).

---

## Dataset

This study utilizes the open **Soccer Match Event Dataset** (Wyscout) covering 7 competitions (43,078 shots, 1,345 rare-scenario shots):
- **Source:** [Figshare Collection 4415000](https://figshare.com/collections/Soccer_match_event_dataset/4415000)
- **Reference:** Pappalardo et al. (2019), *Scientific Data*, 6, 236. [DOI: 10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7).
- **License:** CC BY 4.0.

The processed feature table (`data/shots_features.csv`) is included in this repository. To rebuild from raw Wyscout JSONs, see [data/README.md](data/README.md).

---

## Installation & Environment

### Prerequisites
- Python 3.10 to 3.12
- PyTorch (CPU or CUDA)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd football-xg-ctgan

# (Optional) Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproduction Pipeline

All experiment stages are modular numbered scripts executable in sequence:

### 1. Data Extraction & Feature Engineering
```bash
python 01_load_data.py            # Parse raw Wyscout JSON -> data/shots_cache.csv
python 02_feature_engineering.py  # Compute features -> data/shots_features.csv
```

### 2. Standard Stratified Split Experiments
```bash
python 03_train_test_split.py     # 80/20 stratified split -> data/split_data.pkl
python 04_baseline_models.py      # Baseline A & B training
python 05_ctgan_train.py          # Train CTGAN on rare training shots
python 06_synthetic_quality.py    # KS/JS distribution diagnostics & Fig 6
python 07_model_c.py              # Model C (20% ratio)
python 08_ablation_d.py           # 18-ratio CTGAN sweep (N=5 runs)
python 08b_smote_ablation_d.py    # 18-ratio SMOTENC sweep (N=5 runs)
python 09_ablation_e.py           # Zero-shot rare evaluation
python 10_results_and_figures.py  # Results & figures (Figs 7–10)
python 11_ece_reliability.py      # Expected Calibration Error (ECE)
python 12_significance_tests.py   # Paired t-tests & significance
python 13_fig3_goaldiff_context.py
python 14_fig8_smote_comparison.py
python 15_scale_pos_weight_and_pressure_parity.py
python 16_fig8_compact_for_paper.py
```

### 3. Match-Grouped Splitting & Multi-Split Resampling
```bash
python 18a_train_test_split_matchgrouped.py     # Leakage-free match-grouped split
python 18b_baseline_models_matchgrouped.py       # Baseline A/B on match-grouped
python 18c_ctgan_train_matchgrouped.py           # CTGAN on match-grouped split
python 18d_ablation_d_matchgrouped.py            # CTGAN 18-ratio sweep
python 18e_smote_ablation_d_matchgrouped.py      # SMOTE 18-ratio sweep
python 18f_scale_pos_weight_matchgrouped.py      # Class-weight sensitivity
python 18g_ece_reliability_matchgrouped.py       # Match-grouped ECE reliability
python 18h_significance_tests_matchgrouped.py    # Match-grouped significance tests
python 19a_seed_robustness_matchgrouped.py       # Multi-seed CTGAN robustness
python 19b_train_val_test_split_matchgrouped.py  # 8 match-grouped val/test splits
python 19c_ctgan_train_matchgrouped_valtest.py   # CTGAN models across 8 splits
python 19d_ratio_selection_matchgrouped_valtest.py # Validation-only ratio selection
python 19e_multiseed_valtest_matchgrouped.py     # Multi-split evaluation (Table I)
python 20_statistical_rigor.py                  # Bootstrap CIs & paired tests
python 21_final_figures.py                      # Generate paper figures
python 22_permutation_tests.py                  # Exact permutation tests
python 24_baseline_brier_multisplit.py          # Multi-split Brier baselines
```

### 4. Cross-Competition LOCO Cold-Start & Controls
```bash
python 17_ablation_e_loco.py             # LOCO cold-start evaluation (Table II, Figs 3, 4)
python 25_loco_normal_only_control.py    # Normal-only control Model N
python 26_loco_fixed_weight_check.py     # Fixed scale_pos_weight confound check
python 23_revision_figures.py            # Final 4 revision figures
```

### 5. Research Verification (81/81 Checks)
```bash
python 27_verify_paper_numbers.py        # Validates all paper numbers against artifacts
```

---

## Paper Manuscript & Figures

The finalized 6-page IEEE conference manuscript and bibliography are located in the repository root:
- `football.tex`: Complete LaTeX source file.
- `bibtex.bib`: BibTeX bibliography file.
- `football.pdf`: Compiled camera-ready manuscript.

### Manuscript Figures (`figures/`)
- `fig_loco_pipeline_diagram.png` (Fig 1: LOCO Cold-Start Study Design)
- `fig_splitdist_ctgan_smote.png` (Fig 2: 8-Split Head-to-Head CTGAN vs SMOTE Distribution)
- `fig_loco_per_fold.png` (Fig 3: Per-Competition Calibration Improvement Forest Plot)
- `fig_loco_dose_response.png` (Fig 4: LOCO Augmentation Dose-Response Curve with Holm Significance)

---

## Verification & Integrity

Run `python 27_verify_paper_numbers.py` to assert all 81 reported scientific quantities:
```
==============================================================================
PASSED: 81   FAILED: 0
==============================================================================
All manuscript numbers verified against cached artifacts.
```

---

## Citation

```bibtex
@inproceedings{football_xg_ctgan_2026,
  title={Synthetic Augmentation for Rare-Scenario Football xG: Cross-Competition Cold-Start Transfer and the Instability of Single-Split Method Comparisons},
  author={Anonymous Authors},
  booktitle={IEEE Conference Proceedings},
  year={2026}
}
```

### Dataset Citation
```bibtex
@article{pappalardo2019public,
  title={A public data set of spatio-temporal match events in soccer competitions},
  author={Pappalardo, Luca and Cintia, Paolo and Rossi, Alessio and Massucco, Emanuele and Ferragina, Paolo and Pedreschi, Dino and Giannotti, Fosca},
  journal={Scientific Data},
  volume={6},
  number={1},
  pages={236},
  year={2019},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-019-0247-7}
}
```

---

## License

- Source code is released under the [MIT License](LICENSE).
- Processed and derived datasets are governed by the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/) of the original Wyscout public release.
