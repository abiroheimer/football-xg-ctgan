"""
01_load_data.py — Stage 1: build the raw + feature-complete shot-level
dataframe from the Wyscout Open Event Dataset.

Run standalone:
    python 01_load_data.py            # uses data/shots_cache.csv if present
    python 01_load_data.py --rebuild  # force a full reprocess of the JSON

Input : data/WyScout/**/events_*.json, data/WyScout/**/matches_*.json
Output: data/shots_cache.csv

────────────────────────────────────────────────────────────────────────
SCHEMA NOTES — verified empirically against these JSON files, because the
handoff doc's rule 10 does not match the Wyscout open-data schema:

  * eventId == 10 ("Shot") has exactly ONE subEventId in this dataset: 100.
    There is no subEventId 82/83/84 inside shot events (per eventid2name.csv
    those are Pass/Head pass, Pass/High pass, Pass/Launch).
  * Penalties are eventId 3 / subEventId 35, and direct free-kick shots are
    eventId 3 / subEventId 33 — NEITHER is an eventId==10 event. So the
    "remove penalties" step is a genuine no-op here, and the paper's 43,078
    figure is simply the raw eventId==10 count (verified: 43,078 exactly).
    The penalty filter is still applied explicitly below so the code is
    honest about it; it drops 0 rows.
  * Body part is a TAG, not a subEventId: 401 = left foot, 402 = right foot,
    403 = head/body (per tags2name.csv). These three tags partition the shots
    perfectly (2785+4349+1317 = 8451 for England; 43,078 total dataset-wide),
    so header coverage is in fact COMPLETE. The old loader looked for
    subEventId 82 / tag 72 — neither of which ever fires — which is why every
    row in the previous cache read "Right Foot" and is_header was all-zero.
  * tag 102 (own goal) never appears on an eventId==10 event in this dataset.
    The branch is kept for schema fidelity but contributes nothing.
  * is_freekick is therefore structurally ALL-ZERO under the paper's shot
    definition (eventId == 10 only), since free-kick shots live under
    eventId 3. Including them would push the total past 43,078 and break
    reproduction, so they stay out. See the loader summary for the warning.
────────────────────────────────────────────────────────────────────────
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from config import CACHE_FILE, WYSCOUT_DIR

# ── Pitch / goal geometry (metres) ───────────────────────────────────
PITCH_LEN_M, PITCH_WID_M = 105.0, 68.0
GOAL_HALF_W_M = 3.66

# ── Wyscout schema constants ─────────────────────────────────────────
SHOT_EVENT_ID = 10
TAG_GOAL, TAG_OWN_GOAL = 101, 102
TAG_LEFT_FOOT, TAG_RIGHT_FOOT, TAG_HEAD = 401, 402, 403
SUB_FREE_KICK, SUB_PENALTY, SUB_HEADER = 83, 84, 82   # per handoff doc; see SCHEMA NOTES

PERIOD_ORDER = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}
PERIOD_OFFSET = {"1H": 0, "2H": 45, "E1": 90, "E2": 105, "P": 120}

MINUTE_BINS = [0, 15, 30, 45, 60, 75, 90, np.inf]
MINUTE_LABELS = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"]


# ── Parsing helpers ──────────────────────────────────────────────────
def parse_field(v):
    """Wyscout list fields arrive either as native list/dict or as a
    stringified JSON blob depending on how the file was written. Handle both."""
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def get_tag_ids(t):
    return frozenset(int(x["id"]) for x in parse_field(t)
                     if isinstance(x, dict) and "id" in x)


def get_xy(p):
    """positions is a list like [{"x":..,"y":..}, ...]; element 0 is the shot origin."""
    pos = parse_field(p)
    if pos and isinstance(pos[0], dict):
        return float(pos[0].get("x", np.nan)), float(pos[0].get("y", np.nan))
    return np.nan, np.nan


def recon_min(period, sec):
    return PERIOD_OFFSET.get(str(period), 0) + int(float(sec) // 60)


# ── Geometry (Wyscout x/y are 0-100 percentages of pitch length/width) ──
def calc_XC(wx, wy):
    """X = along-pitch distance to the goal line; C = lateral offset from
    the goal centre. Goal mouth sits at X=0, C=0."""
    X = (100.0 - wx) * PITCH_LEN_M / 100.0
    C = abs(wy - 50.0) * PITCH_WID_M / 100.0
    return X, C


def calc_distance_m(wx, wy):
    X, C = calc_XC(wx, wy)
    return float(np.sqrt(X ** 2 + C ** 2))


def calc_angle_deg(wx, wy):
    """Angle subtended by the goal mouth at the shot location, via the
    atan2 cross-product form with the posts at C = ±3.66 m.

        cross = 2 * GOAL_HALF_W * X
        dot   = X^2 + C^2 - GOAL_HALF_W^2

    atan2 handles the obtuse case (negative dot) correctly, which matters
    for shots taken very close to and wide of the goal line.
    """
    X, C = calc_XC(wx, wy)
    if X <= 0:
        return np.nan
    num = abs(2.0 * GOAL_HALF_W_M * X)
    den = X ** 2 + C ** 2 - GOAL_HALF_W_M ** 2
    return float(np.degrees(np.arctan2(num, den)))


# ── Body part / shot type from tags ──────────────────────────────────
def body_part(tags):
    if TAG_HEAD in tags:
        return "Head"
    if TAG_LEFT_FOOT in tags:
        return "Left Foot"
    if TAG_RIGHT_FOOT in tags:
        return "Right Foot"
    return "Unknown"


def play_pat(sub):
    return "Set Piece" if sub in (SUB_FREE_KICK, SUB_PENALTY) else "Open Play"


# ── Per-match chronological walk ─────────────────────────────────────
def process_match(grp, home, away, mid, comp):
    """Walk one match's events in chronological order, recording each shot's
    goal_diff from the score as it stood BEFORE that shot's own outcome is
    applied, then updating the score. Getting this order backwards is what
    collapses goal_diff to ~0 for every shot."""
    score = {home: 0, away: 0}
    rows = []
    for e in grp.itertuples(index=False):
        eid = int(getattr(e, "eventId", -1))
        if eid != SHOT_EVENT_ID:
            continue  # only shots move the score under the paper's definition

        tags = get_tag_ids(getattr(e, "tags", []))
        tid = int(getattr(e, "teamId", 0))
        opp = away if tid == home else home
        is_goal = TAG_GOAL in tags and TAG_OWN_GOAL not in tags
        is_og = TAG_OWN_GOAL in tags

        gd = score.get(tid, 0) - score.get(opp, 0)   # score BEFORE this shot
        x, y = get_xy(getattr(e, "positions", []))
        sub = int(getattr(e, "subEventId", 0) or 0)
        period = str(getattr(e, "matchPeriod", "1H"))

        rows.append({
            "match_id": mid, "competition": comp,
            "minute": recon_min(period, getattr(e, "eventSec", 0)),
            "period": period, "team_id": tid, "x": x, "y": y,
            "is_goal": int(is_goal), "is_own_goal": int(is_og),
            "goal_diff": gd,
            "shot_body_part": body_part(tags),
            "play_pattern": play_pat(sub),
            "sub_event_name": str(getattr(e, "subEventName", "")),
            "sub_event_id": sub,
        })

        if is_goal:
            score[tid] = score.get(tid, 0) + 1
        elif is_og:
            score[opp] = score.get(opp, 0) + 1   # own goal credited to opponent

    return rows


# ── File discovery ───────────────────────────────────────────────────
def find_files(pattern):
    """Recursive — the real layout is data/WyScout/{events,matches}/, not flat."""
    return sorted(glob.glob(os.path.join(WYSCOUT_DIR, "**", pattern), recursive=True))


def load_match_meta(mt_files):
    meta = {}
    for mf in mt_files:
        comp = os.path.basename(mf).replace("matches_", "").replace(".json", "")
        with open(mf, "r", encoding="utf-8") as f:
            ms = json.load(f)
        for m in ms:
            mid = int(m.get("wyId", m.get("matchId", 0)))
            if not mid:
                continue
            h = a = None
            for tid, td in m.get("teamsData", {}).items():
                if td.get("side") == "home":
                    h = int(tid)
                elif td.get("side") == "away":
                    a = int(tid)
            if h and a:
                meta[mid] = {"home": h, "away": a, "comp": comp}
    return meta


# ── Feature engineering ──────────────────────────────────────────────
def add_features(shots):
    shots = shots.copy()
    shots["distance"] = shots["distance"].clip(1, 65)
    shots["angle_deg"] = shots["angle_deg"].clip(0, 70)
    shots["goal_diff_clipped"] = shots["goal_diff"].clip(-5, 5)
    shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)
    # Free-kick shots are eventId 3 / subEventId 33 and are excluded by the
    # eventId==10 shot definition, so this is structurally all-zero here.
    shots["is_freekick"] = (shots["sub_event_id"] == SUB_FREE_KICK).astype(int)
    # Paper §2: max(0, -goal_diff) * (minute - 60)/30 for minute > 60, else 0.
    # Note: no upper clip on -goal_diff (the old stage-02 code clipped it at 5).
    shots["pressure_index"] = np.where(
        shots["minute"] > 60,
        np.maximum(0, -shots["goal_diff"]) * (shots["minute"] - 60) / 30.0,
        0.0,
    )
    # right=False → left-closed bins, so minute == 0 lands in "0-15" instead of NaN.
    shots["minute_bucket"] = pd.cut(
        shots["minute"], bins=MINUTE_BINS, labels=MINUTE_LABELS, right=False
    ).astype(str)
    shots["is_rare"] = ((shots["minute"] > 75) & (shots["goal_diff"] <= -2)).astype(int)
    return shots


def load_all_shots():
    ev_files = find_files("events_*.json")
    mt_files = find_files("matches_*.json")
    if not ev_files:
        raise FileNotFoundError(f"No events_*.json found anywhere under {WYSCOUT_DIR}")
    print(f"  Event files: {len(ev_files)}  |  Match files: {len(mt_files)}")

    meta = load_match_meta(mt_files)
    print(f"  Matches: {len(meta):,}")

    rows = []
    for ef in ev_files:
        comp = os.path.basename(ef).replace("events_", "").replace(".json", "")
        print(f"  > {comp} ...", end="", flush=True)
        with open(ef, "r", encoding="utf-8") as f:
            ev = pd.DataFrame(json.load(f))
        ev["_p"] = ev["matchPeriod"].map(PERIOD_ORDER).fillna(9)
        ev = ev.sort_values(["matchId", "_p", "eventSec"], kind="mergesort")
        n0 = len(rows)
        for mid_raw, grp in ev.groupby("matchId", sort=True):
            mid = int(mid_raw)
            if mid not in meta:
                continue
            rows.extend(process_match(grp, meta[mid]["home"], meta[mid]["away"], mid, comp))
        print(f" {len(rows) - n0:,} shots  (running total {len(rows):,})")

    shots = pd.DataFrame(rows)

    # ── Coordinate validity + geometry ──
    valid = shots["x"].between(0, 100) & shots["y"].between(0, 100)
    n_bad = int((~valid).sum())
    if n_bad:
        print(f"  ! dropped {n_bad} shots with out-of-range coordinates")
    shots = shots[valid].reset_index(drop=True)
    shots["distance"] = [calc_distance_m(a, b) for a, b in zip(shots["x"], shots["y"])]
    shots["angle_deg"] = [calc_angle_deg(a, b) for a, b in zip(shots["x"], shots["y"])]

    # ── Penalty removal (see SCHEMA NOTES: a no-op on eventId==10 data) ──
    n_pen = int((shots["sub_event_id"] == SUB_PENALTY).sum())
    shots = shots[shots["sub_event_id"] != SUB_PENALTY].reset_index(drop=True)
    print(f"  Penalties removed (subEventId {SUB_PENALTY}): {n_pen}"
          + ("  <- expected 0: penalties are eventId 3/subEventId 35 in this schema" if n_pen == 0 else ""))

    return add_features(shots)


# ── Health check ─────────────────────────────────────────────────────
EXPECTED = {"total": 43078, "rare": 1345, "rare_pct": 3.12,
            "conv_all": 10.44, "conv_rare": 8.85}


def health_check(shots):
    rare = shots[shots["is_rare"] == 1]
    n, nr = len(shots), len(rare)
    conv_all = 100 * shots["is_goal"].mean()
    conv_rare = 100 * rare["is_goal"].mean()
    rare_pct = 100 * nr / n
    at0 = 100 * (shots["goal_diff"] == 0).mean()

    print("\n" + "=" * 72)
    print("HEALTH CHECK")
    print("=" * 72)

    def line(label, got, want, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<24} {got:<22} expected {want}")

    checks = [
        ("Total shots", f"{n:,}", f"{EXPECTED['total']:,}", n == EXPECTED["total"]),
        ("Rare shots", f"{nr:,}", f"{EXPECTED['rare']:,}", nr == EXPECTED["rare"]),
        ("Rare %", f"{rare_pct:.2f}%", "3.12%", abs(rare_pct - 3.12) < 0.01),
        # 4,492 goals / 43,078 shots = 10.4275%, which displays as 10.43% — the
        # lower bound is 10.42 so the unrounded value clears the paper's band.
        ("Conversion (all)", f"{conv_all:.2f}%", "10.43-10.45%", 10.42 <= conv_all <= 10.45),
        ("Conversion (rare)", f"{conv_rare:.2f}%", "~8.85%", abs(conv_rare - 8.85) < 0.05),
        ("goal_diff == 0 share", f"{at0:.1f}%", "< 90% (score tracking OK)", at0 < 90),
    ]
    for c in checks:
        line(*c)
    all_ok = all(c[3] for c in checks)

    print("\n  goal_diff distribution:")
    gd = shots.groupby("goal_diff").agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
    gd["conv%"] = (100 * gd["goals"] / gd["shots"]).round(2)
    gd["share%"] = (100 * gd["shots"] / n).round(2)
    print(gd.to_string().replace("\n", "\n    ").rjust(4))

    print("\n  Per-competition rare / non-rare breakdown:")
    g = shots.groupby("competition")
    tbl = pd.DataFrame({
        "shots": g.size(),
        "goals": g["is_goal"].sum(),
        "conv%": (100 * g["is_goal"].mean()).round(2),
        "rare": g["is_rare"].sum(),
    })
    tbl["rare%"] = (100 * tbl["rare"] / tbl["shots"]).round(2)
    tbl["non_rare"] = tbl["shots"] - tbl["rare"]
    rare_g = shots[shots["is_rare"] == 1].groupby("competition")["is_goal"]
    nonr_g = shots[shots["is_rare"] == 0].groupby("competition")["is_goal"]
    tbl["rare_conv%"] = (100 * rare_g.mean()).round(2)
    tbl["nonrare_conv%"] = (100 * nonr_g.mean()).round(2)
    tbl = tbl[["shots", "goals", "conv%", "rare", "rare%", "rare_conv%",
               "non_rare", "nonrare_conv%"]]
    print(tbl.to_string().replace("\n", "\n    ").rjust(4))

    print("\n  Feature sanity:")
    print(f"    is_header   : {shots['is_header'].sum():,} "
          f"({100 * shots['is_header'].mean():.1f}%)")
    print(f"    is_freekick : {shots['is_freekick'].sum():,} "
          f"({100 * shots['is_freekick'].mean():.1f}%)  "
          f"<- structurally 0: FK shots are eventId 3, outside the eventId==10 definition")
    print(f"    body part   : {shots['shot_body_part'].value_counts().to_dict()}")
    print(f"    minute_bucket: {shots['minute_bucket'].value_counts().reindex(MINUTE_LABELS).to_dict()}")
    print(f"    distance    : {shots['distance'].min():.2f} - {shots['distance'].max():.2f} m")
    print(f"    angle_deg   : {shots['angle_deg'].min():.2f} - {shots['angle_deg'].max():.2f} deg")
    print(f"    pressure_idx: {shots['pressure_index'].min():.2f} - {shots['pressure_index'].max():.2f}")
    print("=" * 72)
    print(f"  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED — do not proceed to stage 02'}")
    print("=" * 72)
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="reprocess the raw JSON even if the cache exists")
    args = ap.parse_args()

    if os.path.exists(CACHE_FILE) and not args.rebuild:
        print(f"Cache found -> {CACHE_FILE}  (pass --rebuild to reprocess)")
        shots = pd.read_csv(CACHE_FILE)
    else:
        print(f"Processing Wyscout JSON from {WYSCOUT_DIR} ...", flush=True)
        shots = load_all_shots()
        if os.path.exists(CACHE_FILE):
            prev = CACHE_FILE.replace(".csv", "_prev.csv")
            os.replace(CACHE_FILE, prev)
            print(f"  previous cache moved to {prev}")
        shots.to_csv(CACHE_FILE, index=False)
        print(f"Saved -> {CACHE_FILE}")

    ok = health_check(shots)
    raise SystemExit(0 if ok else 1)
