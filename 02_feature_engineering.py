"""
02_feature_engineering.py — Stage 2: derive modeling features and run the
health check from the handoff doc (point 11).

Run standalone:
    python 02_feature_engineering.py

Input : data/shots_cache.csv   (stage 01 output)
Output: data/shots_features.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import CACHE_FILE, FEATURES_FILE, SEED


def engineer_features(shots: pd.DataFrame) -> pd.DataFrame:
    shots = shots[shots["sub_event_name"] != "Penalty"].reset_index(drop=True)
    shots["distance"] = shots["distance"].clip(1, 65)
    shots["angle_deg"] = shots["angle_deg"].clip(0, 70)
    shots["goal_diff_clipped"] = shots["goal_diff"].clip(-5, 5)
    shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)
    shots["is_freekick"] = shots["sub_event_name"].str.contains("Free Kick", case=False, na=False).astype(int)
    # Paper §2: max(0, -goal_diff) * (minute-60)/30 for minute > 60, else 0 — no
    # upper clip on -goal_diff. Must match stage 01's add_features() exactly, since
    # stage 01's cache already carries this column and this recomputes it in place.
    shots["pressure_index"] = np.where(
        shots["minute"] > 60,
        np.maximum(0, -shots["goal_diff"]) * (shots["minute"] - 60) / 30,
        0.0,
    )
    shots["minute_bucket"] = pd.cut(
        shots["minute"].clip(0, 119),
        bins=[0, 15, 30, 45, 60, 75, 90, 120],
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"],
        right=True,
    ).astype(str)
    shots["is_rare"] = ((shots["minute"] > 75) & (shots["goal_diff"] <= -2)).astype(int)
    return shots


def compute_xg(df: pd.DataFrame):
    """Simple logistic-regression xG (distance + angle + angle^2), used only
    as a reference column — not one of the XGBoost model features."""
    v = df.dropna(subset=["distance", "angle_deg"]).copy()
    v["angle2"] = v["angle_deg"] ** 2
    X = v[["distance", "angle_deg", "angle2"]].values
    y = v["is_goal"].values
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=3000, C=0.5, random_state=SEED)
    clf.fit(sc.fit_transform(X), y)

    df = df.copy()
    df["xg"] = np.nan
    p = df.dropna(subset=["distance", "angle_deg"]).copy()
    p["angle2"] = p["angle_deg"] ** 2
    df.loc[p.index, "xg"] = clf.predict_proba(sc.transform(p[["distance", "angle_deg", "angle2"]].values))[:, 1]
    return df, sc, clf


def run_health_check(shots: pd.DataFrame):
    rare = shots[shots["is_rare"] == 1].copy()
    print("=" * 55)
    print("DIAGNOSTICS")
    print("=" * 55)
    print(f"  Total shots    : {len(shots):,}")
    print(f"  Rare shots     : {len(rare):,}  ({100 * len(rare) / len(shots):.2f}%)")
    print(f"  Competitions   : {shots['competition'].nunique()}")
    print(f"  Matches        : {shots['match_id'].nunique():,}")
    print(f"  Conv rate(all) : {100 * shots['is_goal'].mean():.2f}%")
    print(f"  Conv rate(rare): {100 * rare['is_goal'].mean():.2f}%")

    gd = shots.groupby("goal_diff")["is_goal"].agg(["count", "sum"])
    gd.columns = ["shots", "goals"]
    gd["conv%"] = (100 * gd["goals"] / gd["shots"]).round(1)
    print("\n  Goal differential breakdown:")
    print(gd.to_string())

    at0 = 100 * (shots["goal_diff"] == 0).mean()
    ok_score = at0 < 90
    print(f"\n  {'✔' if ok_score else '❌'} {at0:.1f}% shots at diff=0  "
          f"{'(OK)' if ok_score else '(SCORE TRACKING BROKEN — do not proceed)'}")

    ok_count = len(rare) > 500
    print(f"  {'✔' if ok_count else '⚠'} {len(rare)} rare shots  "
          f"{'(sufficient)' if ok_count else '(load more leagues)'}")
    print("=" * 55)

    if not ok_score:
        raise RuntimeError(
            f"Health check FAILED: {at0:.1f}% of shots have goal_diff==0. "
            "Score tracking is broken upstream in stage 01 — do not continue "
            "to stage 03 until this is fixed."
        )
    return rare


if __name__ == "__main__":
    print(f"Loading {CACHE_FILE} ...")
    shots = pd.read_csv(CACHE_FILE)

    shots = engineer_features(shots)
    shots, xg_sc, xg_clf = compute_xg(shots)

    rare = run_health_check(shots)

    shots.to_csv(FEATURES_FILE, index=False)
    print(f"\n✔ Rare shots: {len(rare):,} ({100 * len(rare) / len(shots):.2f}%)  "
          f"|  Rare conv: {100 * rare['is_goal'].mean():.1f}%")
    print(f"✔ Saved → {FEATURES_FILE}")
    print("Next: python 03_train_test_split.py")
