"""
KS PIPELINE — STEP 1
====================
Kennard-Stone image-level split + Mahalanobis outlier removal.

Split ratio: 80/20 (calibration/validation)
  - Matches the sample-based split ratio (4/20 samples = 20% val)
  - ~584 calibration images, ~147 validation images
  - Same physical soil samples appear on BOTH sides (this is the leakage)

Order of operations:
  1. Kennard-Stone split on all 731 images (80/20)
  2. Mahalanobis outlier removal on calibration set only
  3. Save calibration_set.csv and validation_set.csv

Reference: Kennard & Stone (1969) Technometrics 11(1):137-148
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import chi2
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "raw",
    "soil_image_features_without_commas.csv")
METADATA_CSV = os.path.join(PROJECT_ROOT, "data", "raw",
    "image_with_soc_metadata.csv")
IMAGE_RECORD = os.path.join(PROJECT_ROOT, "data", "raw",
    "IMAGe record_image No._copy.xlsx")
OUT_DIR = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step1_output_ks")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET     = "soc"
CAL_RATIO  = 0.80   # 80/20 to match sample-based split (4/20 samples = 20% val)
CONFIDENCE = 0.99   # Mahalanobis 99% threshold

EXCLUDE_COLS = ["filename", "Numeric numbers", "image_path", "image_no",
                "soil_type", "moisture", "soc", "Sample_No", "Image_No"]


# ── STEP 0: LOAD & MERGE ──────────────────────────────────────────────────────
print("="*65)
print("KS STEP 1: KENNARD-STONE SPLIT + OUTLIER REMOVAL")
print("="*65)

feat   = pd.read_csv(FEATURES_CSV)
meta   = pd.read_csv(METADATA_CSV)
merged = pd.merge(feat, meta, left_on="Numeric numbers",
                  right_on="image_no", how="inner")
merged = merged.drop_duplicates(subset=["image_no"]).reset_index(drop=True)
print(f"\nMerged: {len(merged)} images x {len(merged.columns)} cols")

# Load Excel sample mapping
import openpyxl
xl = pd.ExcelFile(IMAGE_RECORD)
def rs1():
    raw=pd.read_excel(xl,sheet_name="1-205",header=None); g=[]
    for s in [0,9,17]:
        grp=raw.iloc[1:,s:s+6].copy()
        grp.columns=["D","Sample_No","Image_No","M","ST","SOC"]
        grp=grp.dropna(subset=["Image_No"])
        grp["Sample_No"]=pd.to_numeric(grp["Sample_No"],errors="coerce")
        grp["Image_No"] =pd.to_numeric(grp["Image_No"], errors="coerce")
        grp=grp.dropna(subset=["Sample_No","Image_No"])
        g.append(grp[["Sample_No","Image_No"]])
    return pd.concat(g,ignore_index=True)
def rs(sh,suf):
    df=pd.read_excel(xl,sheet_name=sh); g=[]
    for s in suf:
        sn,ino="Sample No."+s,"Image No."+s
        if sn in df.columns and ino in df.columns:
            grp=df[[sn,ino]].copy(); grp.columns=["Sample_No","Image_No"]
            grp["Sample_No"]=pd.to_numeric(grp["Sample_No"],errors="coerce")
            grp["Image_No"] =pd.to_numeric(grp["Image_No"], errors="coerce")
            grp=grp.dropna(subset=["Sample_No","Image_No"]); g.append(grp)
    return pd.concat(g,ignore_index=True) if g else pd.DataFrame()

mapping=pd.concat([rs1(),rs("206-360",["",".1",".2"]),
                   rs("361-731",["",".1",".2",".3",".4"])],
                  ignore_index=True)
mapping["Sample_No"]=mapping["Sample_No"].astype(int)
mapping["Image_No"] =mapping["Image_No"].astype(int)
mapping=mapping.drop_duplicates(subset=["Image_No"]).reset_index(drop=True)
merged=merged.merge(mapping.rename(columns={"Image_No":"image_no"}),
                    on="image_no",how="left")
merged["Sample_No"]=pd.to_numeric(merged["Sample_No"],
                                   errors="coerce").astype("Int64")
print(f"Sample mapping: {mapping['Sample_No'].nunique()} unique samples")


# ── STEP 1: KENNARD-STONE SPLIT (image level, 80/20) ─────────────────────────
print("\n" + "="*65)
print("STEP 1: KENNARD-STONE IMAGE-LEVEL SPLIT  (80/20)")
print(f"  Ratio matches sample-based split: 4/20 samples = 20% validation")
print("="*65)

feat_cols = [c for c in merged.columns
             if c not in EXCLUDE_COLS
             and pd.api.types.is_numeric_dtype(merged[c])]

imp    = SimpleImputer(strategy="median")
scaler = StandardScaler()
X      = scaler.fit_transform(imp.fit_transform(merged[feat_cols].values))

n_total = len(merged)
n_cal   = int(np.round(n_total * CAL_RATIO))
n_val   = n_total - n_cal

print(f"\n  Total images      : {n_total}")
print(f"  Calibration (80%) : {n_cal}")
print(f"  Validation  (20%) : {n_val}")
print(f"  Running Kennard-Stone algorithm...")

# Kennard-Stone
dist_matrix = cdist(X, X, metric="euclidean")
i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
selected  = [i, j]
remaining = list(set(range(n_total)) - {i, j})

while len(selected) < n_cal:
    d2sel    = dist_matrix[np.ix_(remaining, selected)]
    min_dist = d2sel.min(axis=1)
    nxt      = remaining[min_dist.argmax()]
    selected.append(nxt)
    remaining.remove(nxt)

cal_idx = np.array(sorted(selected))
val_idx = np.array(sorted(remaining))

df_ks_cal = merged.iloc[cal_idx].reset_index(drop=True)
df_ks_val = merged.iloc[val_idx].reset_index(drop=True)

print(f"  Done: {len(df_ks_cal)} cal, {len(df_ks_val)} val")

# Check leakage
if "Sample_No" in merged.columns:
    cal_samps = set(df_ks_cal["Sample_No"].dropna().unique())
    val_samps = set(df_ks_val["Sample_No"].dropna().unique())
    both      = cal_samps & val_samps
    print(f"\n  Samples in cal only : {sorted(cal_samps - val_samps)}")
    print(f"  Samples in val only : {sorted(val_samps - cal_samps)}")
    print(f"  Samples on BOTH sides (leakage): {len(both)} — {sorted(both)}")
    print(f"\n  ⚠  {len(both)} of 20 soil samples have images on BOTH sides")
    print(f"  ⚠  This is the leakage the reviewers identified")


# ── STEP 2: MAHALANOBIS OUTLIER REMOVAL on calibration only ──────────────────
print("\n" + "="*65)
print("STEP 2: MAHALANOBIS OUTLIER REMOVAL (calibration only)")
print("  Reference: De Maesschalck et al. (2000)")
print("="*65)

numeric_df = df_ks_cal[feat_cols].copy()
mean_vec   = numeric_df.mean().values
inv_cov    = np.linalg.pinv(np.cov(numeric_df.T))
n_vars     = len(feat_cols)

print(f"  Features: {n_vars}  |  Calibration images: {len(df_ks_cal)}")

distances = numeric_df.apply(
    lambda row: mahalanobis(row, mean_vec, inv_cov), axis=1)
threshold = np.sqrt(chi2.ppf(CONFIDENCE, df=n_vars))
print(f"  Distance threshold (99%, sqrt chi2): {threshold:.4f}")

df_ks_cal["Mahalanobis_Distance"] = distances.values
df_ks_cal["Is_Outlier"]           = distances > threshold

n_out   = int(df_ks_cal["Is_Outlier"].sum())
n_clean = len(df_ks_cal) - n_out
print(f"  Outliers removed : {n_out}")
print(f"  Clean calibration: {n_clean}")

df_ks_cal_clean = df_ks_cal[~df_ks_cal["Is_Outlier"]].reset_index(drop=True)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Mahalanobis Outlier Detection — KS Calibration Set",
             fontweight="bold")
d_vals = distances.values
is_out = df_ks_cal["Is_Outlier"].values
idx = np.arange(len(d_vals))

axes[0].scatter(idx[~is_out],d_vals[~is_out],
                c="#1565C0", s=10, alpha=0.5, label="Clean")
axes[0].scatter(idx[is_out], d_vals[is_out],
                c="#C62828", s=40, label=f"Outlier (n={n_out})")
axes[0].axhline(threshold, color="orange", ls="--", lw=2,
                label=f"Threshold={threshold:.2f}")
axes[0].set_xlabel("Image index"); axes[0].set_ylabel("Mahalanobis Distance")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[1].hist(d_vals[~is_out], bins=30, color="#1565C0", alpha=0.7)
axes[1].axvline(threshold, color="orange", ls="--", lw=2)
axes[1].set_xlabel("Distance"); axes[1].set_ylabel("Count")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mahalanobis_plot.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ── SAVE ──────────────────────────────────────────────────────────────────────
df_ks_cal_clean.to_csv(
    os.path.join(OUT_DIR, "calibration_set.csv"), index=False)
df_ks_val.to_csv(
    os.path.join(OUT_DIR, "validation_set.csv"), index=False)

# Summary
lines = [
    "="*65,
    "KS STEP 1 SUMMARY",
    "="*65,
    f"Total images         : {n_total}",
    f"KS split (80/20)     : {n_cal} cal / {n_val} val",
    f"Outliers removed     : {n_out} (Mahalanobis D > {threshold:.4f})",
    f"Clean calibration    : {n_clean}",
    f"Validation           : {len(df_ks_val)} (unchanged — no outlier removal on val)",
    "",
    "LEAKAGE CHECK:",
    f"  Samples on both sides : {len(both)} of 20",
    f"  Straddling samples    : {sorted(both)}",
    "",
    "RATIO COMPARISON:",
    f"  Sample-based split : 4/20 samples = 20.0% val  ({len(df_ks_val)} val images)",
    f"  KS image split     : 80/20 = {100*n_val/n_total:.1f}% val  ({n_val} val images)",
    f"  Ratio matched      : YES — both use ~20% for validation",
    "="*65,
]
report = "\n".join(lines)
print("\n" + report)
with open(os.path.join(OUT_DIR, "step1_summary.txt"), "w",
          encoding="utf-8") as f:
    f.write(report)

print(f"\n  Saved: calibration_set.csv  ({n_clean} rows)")
print(f"  Saved: validation_set.csv   ({len(df_ks_val)} rows)")
print(f"  Saved: mahalanobis_plot.png")
print(f"  Outputs: {OUT_DIR}")
print("\nNext: run ks_step2_feature_selection.py")