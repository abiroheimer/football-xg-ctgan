"""
01b_rebuild_and_verify_cache.py — Rebuild the shots cache from raw Wyscout
JSON (real shot_body_part / sub_event_name this time) and verify it lines
up row-for-row with the existing data/shots_cache.csv before replacing it.

Why this approach instead of merging shots_cache.csv with
shots_wyscout_v2.csv: the two files contain the same 43,078 rows but in
different order, with no shared key to merge on safely. process_match() in
01_load_data.py is fully deterministic — same JSON files, same glob order,
same chronological per-match sort — so re-running it reproduces the exact
same row order as whatever produced the existing cache. That sidesteps the
merge-key problem entirely: we just diff the two dataframes positionally.

Run standalone (from the "final codes" directory, venv active):
    python 01b_rebuild_and_verify_cache.py

Input : WyScout/events_*.json, WyScout/matches_*.json, data/shots_cache.csv
Output: data/shots_cache_backup.csv   (your original, untouched)
        data/shots_cache.csv          (overwritten ONLY if verification passes)
"""
import importlib.util
import os
import shutil
import sys

import numpy as np
import pandas as pd

from config import CACHE_FILE, DATA_DIR

# ── Load 01_load_data.py as a module (can't `import 01_load_data` — starts with a digit) ──
_loader_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_load_data.py")
_spec = importlib.util.spec_from_file_location("stage01", _loader_path)
stage01 = importlib.util.module_from_spec(_spec)
sys.modules["stage01"] = stage01
_spec.loader.exec_module(stage01)

ALIGN_COLS = ["match_id", "minute", "team_id", "goal_diff", "is_goal", "period"]

if __name__ == "__main__":
    if not os.path.exists(CACHE_FILE):
        print(f"❌ No existing cache at {CACHE_FILE} — nothing to verify against. "
              f"Just run 01_load_data.py normally instead.")
        sys.exit(1)

    backup_path = os.path.join(DATA_DIR, "shots_cache_backup.csv")
    shutil.copy2(CACHE_FILE, backup_path)
    print(f"✔ Backed up existing cache → {backup_path}")

    old = pd.read_csv(CACHE_FILE)
    print(f"  Existing cache: {len(old):,} rows")

    print("\nRebuilding from raw Wyscout JSON (this re-derives shot_body_part/"
          "sub_event_name correctly) ...\n")
    new = stage01.load_all_shots()
    print(f"\n  Freshly built : {len(new):,} rows")

    # ── Validation 1: row count ─────────────────────────────────
    if len(old) != len(new):
        print(f"\n❌ MISMATCH: row counts differ ({len(old):,} vs {len(new):,}). "
              f"Do not proceed — old cache is NOT overwritten. "
              f"Likely cause: different JSON file set, or filtering logic changed "
              f"since the old cache was generated.")
        sys.exit(1)
    print("  ✔ Row counts match")

    # ── Validation 2: positional alignment on stable columns ────
    mismatches = {}
    for col in ALIGN_COLS:
        if col not in old.columns or col not in new.columns:
            print(f"  ⚠ Column '{col}' missing from one side — skipping this check")
            continue
        eq = (old[col].astype(str).values == new[col].astype(str).values)
        n_bad = (~eq).sum()
        mismatches[col] = n_bad
        pct_ok = 100 * eq.mean()
        print(f"  {'✔' if n_bad == 0 else '❌'} {col:<12} {pct_ok:.2f}% match  ({n_bad} mismatched rows)")

    total_bad = sum(mismatches.values())
    if total_bad > 0:
        print(f"\n❌ Row order does NOT align ({total_bad} mismatched cells across "
              f"{len(ALIGN_COLS)} columns). Old cache is NOT overwritten.")
        print("   This means the fresh run used a different file order or slightly "
              "different logic than whatever produced the original cache. Don't "
              "trust a positional merge here — fall back to a real key-based merge, "
              "or just adopt the freshly rebuilt cache wholesale (it's the one with "
              "correct shot_body_part data) after double-checking its health-check "
              "numbers below.")
        sys.exit(1)

    print("\n✔ Perfect row-for-row alignment on match_id/minute/team_id/goal_diff/is_goal/period")

    # ── Show what actually changed ───────────────────────────────
    print("\nOld shot_body_part distribution:")
    print(old["shot_body_part"].value_counts().to_string())
    print("\nNew shot_body_part distribution:")
    print(new["shot_body_part"].value_counts().to_string())
    print("\nNew play_pattern distribution:")
    print(new["play_pattern"].value_counts().to_string())

    rare_old = (old["minute"] > 75) & (old["goal_diff"] <= -2)
    rare_new = (new["minute"] > 75) & (new["goal_diff"] <= -2)
    print(f"\nRare rate — old: {100*rare_old.mean():.2f}%  new: {100*rare_new.mean():.2f}%")

    # ── Carry over the old cache's other derived columns (distance/angle/xg) ──
    # since they were already health-check-verified; only body-part-related
    # columns come from the fresh run.
    merged = old.copy()
    merged["shot_body_part"] = new["shot_body_part"].values
    merged["play_pattern"] = new["play_pattern"].values
    merged["sub_event_name"] = new["sub_event_name"].values

    merged.to_csv(CACHE_FILE, index=False)
    print(f"\n✔ Verified and saved → {CACHE_FILE}")
    print(f"  (original preserved at {backup_path})")
    print("\nNext: python 02_feature_engineering.py")
