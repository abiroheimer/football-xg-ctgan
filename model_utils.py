"""
model_utils.py — Shared model training / evaluation / synthetic-sampling
helpers. Not a standalone stage — imported by 04, 05, 06, 07, 08, 08b, 09, 10.
"""
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import SEED, XG_FEATURES


def train_xgb(X_tr, y_tr, label="", scale_pos_weight=None):
    """scale_pos_weight: None (default) recomputes it from this training
    set's class balance, as every existing call site relies on. Pass an
    explicit value to hold it fixed instead — used by the Step 5 confound
    check (results/scale_pos_weight_sensitivity.csv) to isolate how much of
    an augmented model's improvement comes from the reweighting effect of a
    changing class balance vs. the synthetic data's information content."""
    pw = scale_pos_weight if scale_pos_weight is not None else \
        max(1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.1,
        reg_alpha=0.1, scale_pos_weight=pw, eval_metric="logloss",
        random_state=SEED, verbosity=0,
        # CPU, not GPU: GPU hist tree construction gives measurably different
        # splits on this dataset (~0.02-0.04 AUC swing, confirmed empirically),
        # and this pipeline keeps the corrected is_header feature (see loader
        # notes) rather than matching the paper's published numbers exactly,
        # so CPU determinism matters more here than GPU speed on this small
        # (34K-row) dataset.
        tree_method="hist", device="cpu",
    )
    m.fit(X_tr, y_tr, verbose=False)
    if label:
        print(f"  ✔ {label} ({len(X_tr):,} samples, pw={pw:.1f})")
    return m


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Standard equal-width-bin ECE (Guo et al. 2017): partition [0,1] into
    n_bins equal-width bins, and take the sample-weighted mean absolute gap
    between each bin's average predicted probability and its actual positive
    rate. Unlike the old aggregate |mean_xg - actual_conv| metric, this can
    catch a model that's over-confident on low-xG shots and under-confident
    on high-xG shots even when the two errors cancel in the aggregate.

    Bins with zero samples are skipped (contribute 0 to the weighted sum,
    correctly, since they have zero weight)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def reliability_bins(y_true, y_prob, n_bins=10):
    """Per-bin (mean predicted prob, actual rate, count) for a reliability
    diagram — same binning as expected_calibration_error()."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        rows.append({
            "bin_lo": bin_edges[b], "bin_hi": bin_edges[b + 1],
            "n": int(mask.sum()),
            "mean_pred": float(y_prob[mask].mean()) if mask.any() else np.nan,
            "actual_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
        })
    return rows


def evaluate(model, X, y, label=""):
    if len(y) == 0 or len(np.unique(y)) < 2:
        print(f"  ⚠ {label}: not enough samples")
        return None
    p = model.predict_proba(X)[:, 1]
    return {
        "label": label, "n": len(y), "n_goals": int(y.sum()),
        "log_loss": round(log_loss(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "auc": round(roc_auc_score(y, p), 4),
        "mean_xg": round(float(p.mean()), 4),
        "actual_conv": round(float(y.mean()), 4),
        # Old aggregate calibration metric — kept alongside ECE (not replaced)
        # so comparisons to the paper's Table 1 numbers stay traceable.
        "calib_err": round(abs(float(p.mean()) - float(y.mean())), 4),
        "ece": round(expected_calibration_error(y, p, n_bins=10), 4),
        "proba": p,
    }


def calibrate_goal_rate(s, target):
    """Flip synthetic goal labels to match the real target conversion rate."""
    s = s.copy()
    rate = s["is_goal"].mean()
    if rate > target:
        gi = s[s["is_goal"] == 1].index.tolist()
        ex = len(gi) - int(target * len(s))
        if ex > 0:
            s.loc[np.random.choice(gi, ex, replace=False), "is_goal"] = 0
    elif rate < target:
        ngi = s[s["is_goal"] == 0].index.tolist()
        sh = int(target * len(s)) - int(rate * len(s))
        if sh > 0 and len(ngi) >= sh:
            s.loc[np.random.choice(ngi, sh, replace=False), "is_goal"] = 1
    return s


def build_bucket_minute_samples(rare_df):
    """Real minute values observed in each minute_bucket, for
    compute_pressure_index()'s bootstrap draw."""
    return {b: g["minute"].values for b, g in rare_df.groupby("minute_bucket")}


def compute_pressure_index(goal_diff_clipped, minute_bucket, bucket_minute_samples):
    """Deterministic pressure_index recompute (fix for CTGAN's pressure_index
    no longer being an independently-sampled column — see 05_ctgan_train.py).

    pressure_index = max(0, -goal_diff) * (minute-60)/30 for minute>60, else 0
    is mathematically derived from goal_diff and minute, both of which CTGAN
    already conditions on / generates. Letting CTGAN sample pressure_index as
    an independent column could produce internally-inconsistent rows (e.g.
    goal_diff_clipped=-2, minute_bucket="75-90", pressure_index=0.5, which no
    real shot could ever have). Recomputing it post-generation guarantees
    every synthetic row is internally consistent, same as real rows.

    minute_bucket only carries a bin label, not an exact minute, so each row's
    minute is bootstrap-sampled from the REAL minutes observed in that bucket
    among the rare training shots — NOT collapsed to a single point estimate
    (e.g. the bucket median). An earlier version of this fix used the median,
    which collapsed pressure_index onto only ~8 distinct values (2 buckets x
    ~4 goal_diff_clipped levels) and nearly doubled its JS divergence against
    the real distribution (0.28 -> 0.64) by discarding all of the real
    within-bucket minute variance. Bootstrapping preserves that variance while
    still guaranteeing every synthetic row's minute falls within its
    generated bucket. In practice this only ever needs "75-90" and "90+":
    is_rare (and therefore CTGAN's entire training set) requires minute>75,
    so those are the only two buckets CTGAN ever sees or can generate — and
    minute>75 always clears the >60 threshold, so the "else 0" branch never
    fires for rare-shot synthetic data.
    """
    rep_minute = np.array([
        np.random.choice(bucket_minute_samples[b]) if b in bucket_minute_samples else np.nan
        for b in minute_bucket
    ])
    return np.where(
        rep_minute > 60,
        np.maximum(0, -goal_diff_clipped) * (rep_minute - 60) / 30,
        0.0,
    )


def sample_synthetic(ctgan, n, rare_df, clip=True):
    """Sample n synthetic shots, clip to domain bounds, recompute
    pressure_index deterministically, calibrate goal rate."""
    s = ctgan.sample(n)
    if clip:
        s["distance"] = s["distance"].clip(1, 65)
        s["angle_deg"] = s["angle_deg"].clip(0, 70)
        for c in ["is_goal", "is_header", "is_freekick"]:
            s[c] = s[c].round().clip(0, 1).astype(int)
        s["goal_diff_clipped"] = s["goal_diff_clipped"].round().clip(-5, 5).astype(int)
    bucket_minute_samples = build_bucket_minute_samples(rare_df)
    s["pressure_index"] = compute_pressure_index(s["goal_diff_clipped"], s["minute_bucket"], bucket_minute_samples)
    target = rare_df["is_goal"].mean()
    s = calibrate_goal_rate(s, target)
    return s


def load_ctgan(path):
    """Load a CTGAN model saved by stage 05 (handles both the native
    .save()/.load() format and the pickle fallback)."""
    from ctgan import CTGAN
    try:
        return CTGAN.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def build_aug(X_real, y_real, rare_df, ctgan, ratio):
    """Build an augmented (real + synthetic) train set at the given ratio."""
    n = max(10, int(len(rare_df) * ratio))
    s = sample_synthetic(ctgan, n, rare_df)
    Xs = s[XG_FEATURES].values
    ys = s["is_goal"].values
    return np.vstack([X_real, Xs]), np.concatenate([y_real, ys]), n


# SMOTE_CAT_COLS: judgment call (flagged per Step-3 instructions) — SMOTENC,
# not plain SMOTE, because XG_FEATURES mixes continuous columns (distance,
# angle_deg, pressure_index) with categorical/binary ones. Plain SMOTE
# linearly interpolates ALL columns, which would produce fractional,
# non-physical values for is_header (e.g. 0.4) and goal_diff_clipped (e.g.
# -1.3). SMOTENC instead assigns categorical columns by majority vote among
# nearest neighbors, which mirrors CTGAN's own discrete/continuous column
# split (is_header, is_freekick, goal_diff_clipped are CTGAN's discrete
# columns too — see CTGAN_DISC in 05_ctgan_train.py), keeping the two
# augmentation arms as comparable as the underlying methods allow.
SMOTE_CAT_COLS = ("is_header", "is_freekick", "goal_diff_clipped")


def build_smote_aug(X_real, y_real, X_rare_train, y_rare_train, ratio, seed=None):
    """Build an augmented (real + SMOTE-synthetic) train set at the given
    ratio, mirroring build_aug()'s CTGAN interface/ratio semantics exactly
    (n = max(10, int(len(rare_train) * ratio)), same target ratios as
    Ablation D) so the two augmentation arms are directly comparable.

    Unlike CTGAN, SMOTE-generated labels are exact by construction — each
    synthetic row is generated by interpolating within one class only — so
    there's no goal-rate-calibration step here (nothing to calibrate).

    Note (flagged): pressure_index has no analogous post-hoc-recompute fix
    here like the CTGAN arm's (see compute_pressure_index) — SMOTE's feature
    space has no "minute" column to recompute it from (only the already-
    derived pressure_index itself), so it's produced by ordinary continuous
    interpolation between two real neighbor rows. This is a real asymmetry
    between the two arms worth keeping in mind when comparing them directly.
    """
    n = max(10, int(len(X_rare_train) * ratio))
    target_rate = float(y_rare_train.mean())
    n_pos = max(1, round(n * target_rate))
    n_neg = n - n_pos

    n_pos_exist = int((y_rare_train == 1).sum())
    n_neg_exist = int((y_rare_train == 0).sum())
    k = max(1, min(5, n_pos_exist - 1, n_neg_exist - 1))

    cat_idx = [XG_FEATURES.index(c) for c in SMOTE_CAT_COLS]
    dist_idx, ang_idx = XG_FEATURES.index("distance"), XG_FEATURES.index("angle_deg")

    sm = SMOTENC(
        categorical_features=cat_idx,
        sampling_strategy={0: n_neg_exist + n_neg, 1: n_pos_exist + n_pos},
        k_neighbors=k,
        random_state=seed,
    )
    X_res, y_res = sm.fit_resample(X_rare_train, y_rare_train)

    n0 = len(X_rare_train)
    Xs, ys = X_res[n0:].copy(), y_res[n0:]
    Xs[:, dist_idx] = np.clip(Xs[:, dist_idx], 1, 65)
    Xs[:, ang_idx] = np.clip(Xs[:, ang_idx], 0, 70)

    return np.vstack([X_real, Xs]), np.concatenate([y_real, ys]), n
