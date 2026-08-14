# Data Documentation — Wyscout Soccer Match Event Dataset

## Overview

The experiments in this repository are based on the public **Soccer Match Event Dataset** collected by Wyscout and published by Pappalardo et al. (2019).

- **Official Dataset Repository:** [Figshare Collection 4415000](https://figshare.com/collections/Soccer_match_event_dataset/4415000)
- **Dataset Paper:** Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., Pedreschi, D., & Giannotti, F. (2019). *A public data set of spatio-temporal match events in soccer competitions*. Scientific Data, 6(1), 236. [DOI: 10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7)
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))

---

## Obtaining the Raw Wyscout Dataset

To reproduce the data extraction from scratch (`01_load_data.py`), download the raw JSON files from the Figshare repository:

1. Download the match event files for the seven competitions:
   - `events_England.json`
   - `events_France.json`
   - `events_Germany.json`
   - `events_Italy.json`
   - `events_Spain.json`
   - `events_European_Championship.json`
   - `events_World_Cup.json`
2. Download the corresponding match metadata files:
   - `matches_England.json`
   - `matches_France.json`
   - `matches_Germany.json`
   - `matches_Italy.json`
   - `matches_Spain.json`
   - `matches_European_Championship.json`
   - `matches_World_Cup.json`
3. (Optional) Additional metadata files (`teams.json`, `players.json`, `competitions.json`, `coaches.json`, `referees.json`, `tags2name.csv`, `eventid2name.csv`).

### Expected Raw Data Layout

Place the downloaded raw files in `data/WyScout/` as follows:

```
data/
└── WyScout/
    ├── events/
    │   ├── events_England.json
    │   ├── events_France.json
    │   ├── events_Germany.json
    │   ├── events_Italy.json
    │   ├── events_Spain.json
    │   ├── events_European_Championship.json
    │   └── events_World_Cup.json
    ├── matches/
    │   ├── matches_England.json
    │   ├── matches_France.json
    │   ├── matches_Germany.json
    │   ├── matches_Italy.json
    │   ├── matches_Spain.json
    │   ├── matches_European_Championship.json
    │   └── matches_World_Cup.json
    ├── competitions.json
    ├── teams.json
    ├── players.json
    └── ...
```

---

## Included Processed Datasets

This repository includes the derived feature datasets in `data/` to allow immediate reproduction of models, evaluations, and figures without needing to re-download or re-parse the 1.08 GB raw dataset:

| File | Description | Size | Samples |
|---|---|---|---|
| `data/shots_features.csv` | Full processed shot table with engineered features (`distance`, `angle_deg`, `is_header`, `is_freekick`, `goal_diff_clipped`, `pressure_index`) and metadata | ~6.4 MB | 43,078 shots |
| `data/ctgan_train_df.csv` | Real rare-scenario shots from the training set used to fit the standard CTGAN generator | ~56 KB | 1,076 shots |
| `data/ctgan_train_df_matchgrouped.csv` | Real rare-scenario shots under the match-grouped split used to fit the match-grouped CTGAN generator | ~56 KB | 1,076 shots |

### Attribution & Provenance
The processed CSV files are derived research artifacts generated from the Wyscout public event dataset under CC BY 4.0. They are not the unaltered original dataset.

- **License Terms:** [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
- **Original Figshare Dataset:** [https://figshare.com/collections/Soccer_match_event_dataset/4415000](https://figshare.com/collections/Soccer_match_event_dataset/4415000)