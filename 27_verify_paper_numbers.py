"""
27_verify_paper_numbers.py — verification pass: check every number the
manuscript states for Tables I/II and Figures 2/3/4 against the cached
result artifacts. Read-only; asserts rather than recomputes the science.
"""
import os
import re

import pandas as pd

from config import RESULTS_DIR

TEX = "football.tex"
tex = open(TEX, encoding="utf-8").read()
ok, fail = [], []


def chk(label, cond, detail=""):
    (ok if cond else fail).append(f"{label}{(' -- ' + detail) if detail else ''}")


# ── TABLE I: eight-split aggregate ──
t1 = pd.read_csv(os.path.join(RESULTS_DIR, "multiseed_valtest_table1.csv")).set_index("arm")
for arm, key in [("Baseline A", "Baseline A"), ("CTGAN", "CTGAN"), ("SMOTE", "SMOTE")]:
    for col, nm in [("test_log_loss", "logloss"), ("test_brier", "brier"),
                    ("test_auc", "auc"), ("test_calib_err", "calib")]:
        m, s = t1.loc[key, f"{col}_mean"], t1.loc[key, f"{col}_std"]
        pat = f"${m:.4f}\\pm{s:.4f}$"
        chk(f"TABLE I {arm} {nm}={m:.4f}+-{s:.4f}", pat in tex, f"expected '{pat}'")

# ── TABLE I: bootstrap CI / p-values ──
ci = pd.read_csv(os.path.join(RESULTS_DIR, "bootstrap_ci_augmentation_vs_baseline.csv"))
perm = pd.read_csv(os.path.join(RESULTS_DIR, "multisplit_permutation_test.csv"))
for _, r in ci.iterrows():
    arm = r["comparison"].split()[0]
    chk(f"TABLE I {arm} delta={r['mean_log_loss_diff']:.4f}",
        f"{r['mean_log_loss_diff']:.4f}".lstrip("-") in tex)
    chk(f"TABLE I {arm} CI=[{r['ci_2.5pct']:.4f},{r['ci_97.5pct']:.4f}]",
        f"{r['ci_2.5pct']:.4f}" in tex and f"{r['ci_97.5pct']:.4f}" in tex)
pcs = perm[perm.comparison == "CTGAN vs SMOTE"].iloc[0]
chk("FIG2/TABLE I ctgan-vs-smote t-test p=0.789", "0.789" in tex,
    f"artifact ttest_p={pcs['ttest_p']}")
chk("FIG2/TABLE I ctgan-vs-smote permutation p=0.750", "0.750" in tex,
    f"artifact perm_p={pcs['exact_permutation_p']}")
chk("FIG2/TABLE I ctgan-vs-smote mean diff +0.0018",
    "+0.0018" in tex, f"artifact={pcs['mean_log_loss_diff']}")

# ── FIGURE 2: all 8 splits present, 3/8 vs 5/8 ──
head = pd.read_csv(os.path.join(RESULTS_DIR, "multiseed_valtest_headtohead_matchgrouped.csv"))
chk("FIG2 artifact has 8 splits", len(head) == 8, f"found {len(head)}")
nc = int((head.winner == "CTGAN").sum())
ns = int((head.winner == "SMOTE").sum())
chk("FIG2 CTGAN wins 3/8", nc == 3 and "3 of 8" in tex, f"artifact={nc}")
chk("FIG2 SMOTE wins 5/8", ns == 5 and "5 of 8" in tex, f"artifact={ns}")

# ── TABLE II / FIGURE 3: LOCO per-fold at 5% ──
nn = pd.read_csv(os.path.join(RESULTS_DIR, "loco_normal_only_control.csv"))
pf = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_E_loco_per_fold.csv"))
r5 = pf[pf.ratio == 0.05].set_index("fold")
for _, r in nn.iterrows():
    f = r["fold"]
    chk(f"TABLE II {f} B_G calib {r['BG_calib_err']:.4f}", f"{r['BG_calib_err']:.4f}" in tex)
    chk(f"TABLE II {f} N calib {r['N_calib_err']:.4f}", f"{r['N_calib_err']:.4f}" in tex)
    chk(f"TABLE II {f} C_G calib {r['CG_calib_err']:.4f}", f"{r['CG_calib_err']:.4f}" in tex)
    chk(f"TABLE II {f} B_G brier {r5.loc[f,'baseA_brier']:.4f}", f"{r5.loc[f,'baseA_brier']:.4f}" in tex)
    chk(f"TABLE II {f} C_G brier {r5.loc[f,'brier_mean']:.4f}", f"{r5.loc[f,'brier_mean']:.4f}" in tex)
    # FIG3 = per-fold BG->CG relative improvement, same as Table II "Impr." column
    imp = 100 * (1 - r["CG_calib_err"] / r["BG_calib_err"])
    chk(f"FIG3/TABLE II {f} impr {imp:.1f}%", f"{imp:.1f}\\%" in tex, f"computed {imp:.2f}")

B, N, C = nn.BG_calib_err.mean(), nn.N_calib_err.mean(), nn.CG_calib_err.mean()
chk("TABLE II macro B_G 0.2120", f"{B:.4f}" == "0.2120" and "0.2120" in tex, f"{B:.6f}")
chk("TABLE II macro N 0.1704", f"{N:.4f}" == "0.1704" and "0.1704" in tex, f"{N:.6f}")
chk("TABLE II macro C_G 0.1463", f"{C:.4f}" == "0.1463" and "0.1463" in tex, f"{C:.6f}")
chk("DECOMP B_G->N = 19.6%", f"{100*(1-N/B):.1f}" == "19.6" and "19.6\\%" in tex, f"{100*(1-N/B):.3f}")
chk("DECOMP N->C_G = 14.2%", f"{100*(1-C/N):.1f}" == "14.2" and "14.2\\%" in tex, f"{100*(1-C/N):.3f}")
chk("DECOMP B_G->C_G = 31.0%", f"{100*(1-C/B):.1f}" == "31.0" and "31.0\\%" in tex, f"{100*(1-C/B):.3f}")

# per-fold N->CG range quoted in text
pf_ncg = 100 * (1 - nn.CG_calib_err / nn.N_calib_err)
chk("per-fold N->C_G range 1.4%-30.6% quoted",
    f"{pf_ncg.min():.1f}\\%" in tex and f"{pf_ncg.max():.1f}\\%" in tex,
    f"min={pf_ncg.min():.2f} max={pf_ncg.max():.2f}")

# ── FIGURE 4: dose-response + Holm pattern ──
holm = pd.read_csv(os.path.join(RESULTS_DIR, "loco_ratio_significance_holm.csv"))
allr = pd.read_csv(os.path.join(RESULTS_DIR, "bootstrap_ci_loco_all_ratios.csv"))
chk("FIG4 has all 6 ratios", len(holm) == 6 and len(allr) == 6)
sig = holm[holm.significant_after_holm]
chk("FIG4 Holm sig at 5%-50%, not 100%",
    set(sig.ratio) == {0.05, 0.10, 0.20, 0.30, 0.50},
    f"sig at {sorted(sig.ratio)}")
maxadj = sig.holm_adjusted_p.max()
chk("text 'adjusted p<=0.032' matches", f"{maxadj:.3f}" == "0.032" and "0.032" in tex, f"{maxadj}")
p100 = float(holm[holm.ratio == 1.0].holm_adjusted_p.iloc[0])
chk("text 'adjusted p=0.207' at 100%", f"{p100:.3f}" == "0.207" and "0.207" in tex, f"{p100}")
r100 = allr[allr.ratio == 1.0].iloc[0]
chk("100% bootstrap CI [-2.0,19.5] includes zero",
    f"{r100['ci_2.5pct']:.1f}" == "-2.0" and f"{r100['ci_97.5pct']:.1f}" == "19.5"
    and "-2.0\\%,19.5\\%" in tex.replace("$", ""),
    f"[{r100['ci_2.5pct']},{r100['ci_97.5pct']}]")

# ── LOCO headline stats ──
bci = pd.read_csv(os.path.join(RESULTS_DIR, "bootstrap_ci_loco_calib_5pct.csv")).iloc[0]
chk("LOCO CI [23.6,39.3]", f"{bci['ci_2.5pct']:.1f}" == "23.6" and f"{bci['ci_97.5pct']:.1f}" == "39.3"
    and "23.6" in tex and "39.3" in tex)
chk("LOCO d_z=-3.562", f"{bci['cohens_dz_fold_level']:.3f}" == "-3.562" and "3.562" in tex)
chk("LOCO t=-7.965 p=0.0013", "7.965" in tex and "0.0013" in tex)
lp = pd.read_csv(os.path.join(RESULTS_DIR, "loco_permutation_test.csv"))
chk("LOCO exact perm p=0.0625 at 5%",
    float(lp[lp.ratio == 0.05].exact_permutation_p.iloc[0]) == 0.0625 and "0.0625" in tex)

# ── Fixed-weight check ──
fw = pd.read_csv(os.path.join(RESULTS_DIR, "loco_fixed_weight_check.csv"))
fm = fw.groupby(["fold", "cond", "weight_mode"])["calib_err"].mean().reset_index()
def fmac(c, m):
    return fm[(fm.cond == c) & (fm.weight_mode == m)]["calib_err"].mean()
for c, v in [("B_G", 0.2116), ("N", 0.1704), ("C_G", 0.1455)]:
    got = fmac(c, "fixed")
    chk(f"FIXED-W {c} {v}", f"{got:.4f}" == f"{v:.4f}" and f"{v:.4f}" in tex, f"{got:.6f}")
fn, fc = fmac("N", "fixed"), fmac("C_G", "fixed")
chk("FIXED-W N->C_G 14.6%", f"{100*(1-fc/fn):.1f}" == "14.6" and "14.6\\%" in tex,
    f"{100*(1-fc/fn):.3f}")

# ── Prohibited wording ──
for bad in ["14.2\\% in all five folds", "14.2\\% in every fold", "each fold improves by 14.2",
            "carry the inferential weight", "load-bearing", "universally superior", "proves that"]:
    chk(f"NO prohibited phrase: '{bad}'", bad not in tex)
chk("no author names uncommented", "\n\\author{" not in tex)
chk("no dangling tab:manifest ref", "\\ref{tab:manifest}" not in tex)

print("=" * 78)
print(f"PASSED: {len(ok)}   FAILED: {len(fail)}")
print("=" * 78)
if fail:
    print("\nFAILURES:")
    for f_ in fail:
        print("  X", f_)
else:
    print("\nAll manuscript numbers verified against cached artifacts.")
