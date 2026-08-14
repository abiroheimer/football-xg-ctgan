# Paper Text Errata — Step 3 Fixes 5, 6, 7

No LaTeX/Word source for the paper exists in this repo — only the exported
PDF (`Future works/Improving Rare-Scenario Football Expected Goals (xG)
Modeling Using CTGAN-Based Synthetic Data Augmentation.pdf`). These are
verbatim find/replace patches to apply to whatever source you edit from
(Google Doc / Word / LaTeX). Quotes below are transcribed exactly from the
PDF text extraction.

---

## Fix 5a — Abstract: ratio count ("sixteen" → "eighteen")

**Current text (Abstract):**
> "...we train CTGAN exclusively on rare-scenario shots and evaluate
> augmented XGBoost models across **sixteen** synthetic augmentation ratios
> (1% to 1,000%) using N=5 independent sampling runs per ratio..."

**Fix:** change **sixteen → eighteen**.

**Why the code is right and the text is wrong:** `RATIOS` in
`08_ablation_d.py` (and mirrored in `08b_smote_ablation_d.py`) is:
```
[0.01, 0.05, 0.10, 0.20, 0.40, 0.50, 0.80, 1.00,
 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 10.00]
```
That's 18 entries, matching Section 4.1's own list of values (see Fix 5b) —
the "sixteen" in the abstract just miscounts that same list. Per your
instruction, text was corrected to match the code, not the reverse.

---

## Fix 5b — Section 4.1: ratio count ("Sixteen" → "Eighteen")

**Current text (Section 4.1):**
> "**Sixteen** synthetic augmentation ratios were tested: 1%, 5%, 10%, 20%,
> 40%, 50%, 80%, 100%, 150%, 200%, 250%, 300%, 400%, 500%, 600%, 700%, 800%,
> and 1,000% of the 1,076 rare training shots."

**Fix:** change **Sixteen → Eighteen**. The list itself (18 values) is
already correct and needs no change — only the word "Sixteen" was wrong.

---

## Fix 5c — Section 4.2: ratio count + count/list mismatch ("Five" → "Six")

**Current text (Section 4.2):**
> "Five synthetic ratios were tested (5%, 10%, 20%, 30%, 50%, 100%) with
> N=5 runs each."

**Fix:** change **Five → Six** — the parenthetical list has always had 6
values (5%, 10%, 20%, 30%, 50%, 100%), matching `E_RATIOS` in
`09_ablation_e.py`:
```python
E_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
```

**On whether 30% was actually run:** confirmed run, not missing. The 30%
row appears in `results/ablation_E_raw_runs.csv` (5 runs) and
`results/ablation_E_zero_shot.csv` with `n_synth=322`. Both the pre-Step-3
run (Step 2) and the Step-3 re-run (after the CTGAN pressure_index fix)
include it. So this was purely a "five" vs. "six" wording slip, not a
data gap — no re-run was needed for this one.

---

## Fix 6 — AUC reliability caveat in Section 5.2 (Ablation D)

Section 4.0 already states the caveat once:
> "Metrics reported under these two protocols should not be directly
> compared numerically, particularly AUC, since the smaller 269-shot test
> slice contains only 24 positive samples and therefore exhibits
> substantially higher variance."

Per the review, this needs to be repeated at the point of use, not stated
once and assumed to travel with the reader into Section 5.2. Two places in
5.2 currently present AUC as a strong, uncaveated headline claim:

**Current text (Section 5.2):**
> "At 1,000% augmentation (10,760 synthetic shots), AUC reaches 0.7864 ±
> 0.0108, a 5.2% improvement over Baseline A."

**Suggested fix — append a caveat sentence:**
> "At 1,000% augmentation (10,760 synthetic shots), AUC reaches 0.7864 ±
> 0.0108, a 5.2% improvement over Baseline A. As noted in Section 4.0, this
> AUC estimate is computed on a 269-shot test slice with only 24 positive
> samples; the reported standard deviation likely understates the true
> sampling variance of AUC specifically, and this comparison should be
> read with that caveat, unlike the log loss and Brier comparisons above."

**Current text (Table 1 notes column):**
> "Model C (1000%) ... 0.7864 ± 0.0108 ... Best discrimination"

**Suggested fix:** append a footnote marker to "Best discrimination" (e.g.
"Best discrimination†") with a table footnote: "† AUC on the 269-shot slice
(24 positives) — treat as indicative, not a reliable point estimate; see
Section 4.0."

**Current text (Section 6.3, recommendation):**
> "For ranking or comparative analysis applications, 500% or above is
> preferable."

**Suggested fix:**
> "For ranking or comparative analysis applications, 500% or above shows
> stronger AUC — though given the 24-positive-sample caveat above, this
> recommendation rests on the more reliable log loss/Brier trend at those
> ratios more than on the AUC point estimates themselves."

**Framing suggestion (not a required text change, but recommended per the
review):** lead Section 5.2 with log loss/Brier as the headline metrics for
Ablation D specifically, and present AUC as a secondary/supporting metric
there — the reverse of Ablation E, where AUC is measured on the full
1,345-shot rare set (215+ positives) and is comparatively reliable.

---

## Fix 7 — scale_pos_weight confound (limitations note)

**Add to Section 7 (Limitations), as a new paragraph:**

> "`scale_pos_weight` is recomputed per training set in every ablation, and
> since it is set to the real:synthetic-adjusted class ratio, it changes
> systematically as the augmentation ratio increases (e.g. Baseline A's
> training set gives scale_pos_weight≈8.6, while a 1000%-augmented set
> gives a different value from oversampling rare-class-adjacent synthetic
> rows). This is a confound: some of the reported metric improvement at
> high augmentation ratios may come from this reweighting effect rather
> than from the synthetic data's information content per se, and the two
> are not isolated from each other in the current experimental design. A
> fixed-scale_pos_weight sensitivity check (holding it constant at the
> Baseline A value across all ratios) would isolate this effect but was not
> run for this paper; it is left as a natural follow-up."

**On whether to also add a fixed-scale_pos_weight sensitivity appendix
table (optional, per your instructions):** flagging this as a judgment
call rather than deciding it — a minimal version (Baseline A + 2-3
representative ratios, e.g. 80%/300%/1000%, N=5 each, fixed
scale_pos_weight=8.6) would cost roughly the same compute as one more
Ablation-D-sized run (~15-45 XGBoost fits, seconds each on this dataset —
cheap). Worth doing if you want the confound *quantified* rather than
*stated*; not done here since Fix 7 said "document only for now."

**Also worth folding into Limitations (not in the original review, but
surfaced during Step 3 implementation):** the paper's own is-header
limitation sentence ("Body-part tag coverage in the Wyscout open-data
schema is incomplete...") describes a symptom that was traced to a loader
bug during reproduction (Step 1) — `subEventId`/tag values were checked
against the wrong schema fields, defaulting every shot to "Right Foot" /
`is_header=0`. The real schema (tags 401/402/403) has complete body-part
coverage across all 43,078 shots (16.2% headers once read correctly). This
is out of scope for the 7 fixes above, but is a factual correction to that
specific limitations sentence you'll want to make at some point — happy to
draft the replacement wording on request.
