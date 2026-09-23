# """
# Mahalanobis Distance Outlier Removal + Grouped Sample Split
# ============================================================
# REVISED VERSION — Reviewer Response

# Key change from original:
#   ORIGINAL : Kennard-Stone image-level split (70/30)
#              → caused data leakage (same soil on both sides)
#   REVISED  : Grouped sample-level split
#              → all images from a physical soil sample go to
#                calibration OR validation — never both

# Pipeline:
#   1. Load & merge feature CSV + metadata CSV  (unchanged)
#   2. Mahalanobis outlier removal              (unchanged)
#   3. Load image → sample mapping from Excel
#   4. Grouped sample split (replaces Step 3 Kennard-Stone)
#   5. Summary report

# Validation samples (held out entirely — all their images):
#   Sample  3  Crosby (Mulch plots)   SOC=3.34   ~21 images
#   Sample 12  Kokomo (Soybean)       SOC=1.92   ~44 images
#   Sample 13  Miami (Corn)           SOC=2.50   ~39 images
#   Sample 15  Crosby (Corn)          SOC=1.75   ~55 images
#   Sample 19  Kokomo (Soybean)       SOC=2.51   ~51 images

# Expected output after outlier removal:
#   Calibration : ~460 images  (15 samples)
#   Validation  : ~198 images  (5 samples)

# Reference: De Maesschalck et al. (2000)
# """

# import os
# import warnings
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import mahalanobis
# from scipy.stats import chi2
# from sklearn.preprocessing import StandardScaler

# warnings.filterwarnings('ignore')

# # ─────────────────────────────────────────────────────────────────────────────
# # PATHS  — update PROJECT_ROOT only, everything else updates automatically
# # ─────────────────────────────────────────────────────────────────────────────
# PROJECT_ROOT = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed'
# FEATURES_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw', 'soil_image_features_without_commas.csv')
# METADATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw', 'image_with_soc_metadata.csv')
# IMAGE_RECORD = os.path.join(PROJECT_ROOT, 'data', 'raw', 'IMAGe record_image No._copy.xlsx')
# OUT_DIR      = os.path.join(PROJECT_ROOT, 'notebooks', 'objective_1', 'output_data', 'step1_output')

# os.makedirs(OUT_DIR, exist_ok=True)

# # ─────────────────────────────────────────────────────────────────────────────
# # SETTINGS
# # ─────────────────────────────────────────────────────────────────────────────

# # Columns to exclude from Mahalanobis distance calculation
# EXCLUDE_COLS = ['filename', 'Numeric numbers', 'image_path',
#                 'image_no', 'soil_type']

# # Confidence level for chi-squared outlier threshold
# CONFIDENCE = 0.99       # 99% — De Maesschalck et al. (2000)

# # ── Grouped split configuration ───────────────────────────────────────────────
# # These 5 sample numbers and ALL their images go to validation only
# # Remaining 15 samples go entirely to calibration
# VALIDATION_SAMPLE_NOS = [3, 12, 13, 15]

# SAMPLE_INFO = {
#     3:  'Crosby (Mulch plots)',
#     12: 'Kokomo (Soybean)',
#     13: 'Miami (Corn)',
#     15: 'Crosby (Corn)',
# }


# # ═════════════════════════════════════════════════════════════════════════════
# # STEP 1 — LOAD & MERGE  (unchanged from original)
# # ═════════════════════════════════════════════════════════════════════════════
# def load_and_merge():
#     print("=" * 65)
#     print("STEP 1: Loading and merging files")
#     print("=" * 65)

#     feat   = pd.read_csv(FEATURES_CSV)
#     meta   = pd.read_csv(METADATA_CSV)
#     merged = pd.merge(feat, meta,
#                       left_on='Numeric numbers',
#                       right_on='image_no',
#                       how='inner')

#     print(f"  Features file : {feat.shape[0]} rows x {feat.shape[1]} cols")
#     print(f"  Metadata file : {meta.shape[0]} rows x {meta.shape[1]} cols")
#     print(f"  Merged master : {merged.shape[0]} rows x {merged.shape[1]} cols")

#     merged.to_csv(os.path.join(OUT_DIR, 'master_merged.csv'), index=False)
#     print(f"  Saved: master_merged.csv")
#     return merged


# # ═════════════════════════════════════════════════════════════════════════════
# # STEP 2 — MAHALANOBIS OUTLIER REMOVAL  (unchanged from original)
# # ═════════════════════════════════════════════════════════════════════════════
# def mahalanobis_outlier_removal(df):
#     print("\n" + "=" * 65)
#     print("STEP 2: Mahalanobis Distance Outlier Detection")
#     print("        (De Maesschalck et al., 2000)")
#     print("=" * 65)

#     # Select numeric feature columns — drop identifiers
#     numeric_df = df.select_dtypes(include=[np.number])
#     numeric_df = numeric_df.drop(
#         columns=[c for c in EXCLUDE_COLS if c in numeric_df.columns])

#     n_vars = numeric_df.shape[1]
#     n_obs  = numeric_df.shape[0]
#     print(f"  Variables : {n_vars}")
#     print(f"  Samples   : {n_obs}")

#     # Mean vector + covariance matrix
#     mean_vec   = numeric_df.mean().values
#     cov_matrix = np.cov(numeric_df.T)

#     # Pseudo-inverse handles non-full-rank covariance (correlated features)
#     inv_cov = np.linalg.pinv(cov_matrix)

#     print(f"\n  Computing Mahalanobis distances ...")
#     distances = numeric_df.apply(
#         lambda row: mahalanobis(row, mean_vec, inv_cov), axis=1)

#     # 99% chi-squared threshold
#     threshold   = chi2.ppf(CONFIDENCE, df=n_vars)
#     threshold_d = np.sqrt(threshold)
#     print(f"  Chi-sq threshold (99%, df={n_vars}): {threshold:.4f}")
#     print(f"  Distance threshold (sqrt)          : {threshold_d:.4f}")

#     df = df.copy()
#     df['Mahalanobis_Distance'] = distances.values
#     df['Is_Outlier']           = distances > threshold_d

#     outlier_count = int(df['Is_Outlier'].sum())
#     print(f"\n  Outliers detected : {outlier_count}")
#     print(f"  Clean samples     : {n_obs - outlier_count}")

#     # Save full results
#     df.to_csv(os.path.join(OUT_DIR, 'outlier_detection_results.csv'),
#               index=False)
#     print(f"  Saved: outlier_detection_results.csv")

#     # Save cleaned dataset
#     df_cleaned = (df[~df['Is_Outlier']]
#                   .drop(columns=['Is_Outlier'])
#                   .reset_index(drop=True))
#     df_cleaned.to_csv(os.path.join(OUT_DIR, 'soil_features_cleaned.csv'),
#                       index=False)
#     print(f"  Saved: soil_features_cleaned.csv  ({len(df_cleaned)} rows)")

#     # Log outlier image numbers
#     df_outliers = df[df['Is_Outlier']].copy()
#     print(f"\n  Outlier image numbers:")
#     print(f"    {sorted(df_outliers['Numeric numbers'].tolist())}")

#     _plot_mahalanobis(distances.values, threshold_d,
#                       df['Is_Outlier'].values, n_obs, n_vars)

#     return df_cleaned, df_outliers, distances.values, threshold_d, \
#         numeric_df.columns.tolist()


# def _plot_mahalanobis(distances, threshold, is_outlier, n, n_vars):
#     """Mahalanobis distance plot — unchanged from original."""
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#     fig.suptitle(
#         'Mahalanobis Distance Outlier Detection  (De Maesschalck et al., 2000)\n'
#         f'n={n} images  |  {n_vars} features  |  '
#         f'99% confidence threshold D={threshold:.4f}  |  '
#         f'Outliers removed: {is_outlier.sum()}',
#         fontsize=12, fontweight='bold'
#     )

#     idx = np.arange(n)

#     ax = axes[0]
#     ax.scatter(idx[~is_outlier], distances[~is_outlier],
#                color='#1565C0', s=14, alpha=0.55, label='Clean', zorder=3)
#     ax.scatter(idx[is_outlier], distances[is_outlier],
#                color='#C62828', s=55, alpha=0.95, zorder=5,
#                label=f'Outlier (n={is_outlier.sum()})')
#     ax.axhline(threshold, color='#E65100', lw=2.2, ls='--',
#                label=f'99% threshold (D={threshold:.2f})', zorder=6)
#     ax.set_xlabel('Sample Index', fontsize=11)
#     ax.set_ylabel('Mahalanobis Distance (D)', fontsize=11)
#     ax.set_title('Distance per Sample', fontsize=11, fontweight='bold')
#     ax.legend(fontsize=9)
#     ax.grid(True, ls=':', alpha=0.35)
#     ax.spines[['top', 'right']].set_visible(False)
#     ax.text(0.98, 0.97,
#             f'Outliers : {is_outlier.sum()}\nClean    : {(~is_outlier).sum()}',
#             transform=ax.transAxes, fontsize=9.5, va='top', ha='right',
#             family='monospace',
#             bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
#                       edgecolor='#AAAAAA', alpha=0.92))

#     ax2 = axes[1]
#     ax2.hist(distances[~is_outlier], bins=35, color='#1565C0',
#              alpha=0.70, edgecolor='white', lw=0.5, label='Clean')
#     if is_outlier.sum() > 0:
#         ax2.hist(distances[is_outlier], bins=8, color='#C62828',
#                  alpha=0.90, edgecolor='white', lw=0.5,
#                  label=f'Outlier (n={is_outlier.sum()})')
#     ax2.axvline(threshold, color='#E65100', lw=2.2, ls='--',
#                 label=f'99% threshold (D={threshold:.2f})')
#     ax2.set_xlabel('Mahalanobis Distance (D)', fontsize=11)
#     ax2.set_ylabel('Count', fontsize=11)
#     ax2.set_title('Distance Distribution', fontsize=11, fontweight='bold')
#     ax2.legend(fontsize=9)
#     ax2.grid(True, ls=':', alpha=0.35)
#     ax2.spines[['top', 'right']].set_visible(False)

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_DIR, 'mahalanobis_plot.png'),
#                 dpi=160, bbox_inches='tight')
#     plt.close()
#     print(f"  Saved: mahalanobis_plot.png")


# # ═════════════════════════════════════════════════════════════════════════════
# # STEP 3 — BUILD IMAGE → SAMPLE MAPPING FROM EXCEL
# # ═════════════════════════════════════════════════════════════════════════════
# def build_image_sample_mapping():
#     """
#     Parse all three sheets of the image record Excel file.
#     Returns DataFrame with columns: Image_No, Sample_No
#     covering all 731 images.
#     """
#     print("\n" + "=" * 65)
#     print("STEP 3: Building image → sample mapping from Excel")
#     print("=" * 65)

#     def _sheet_1_205():
#         raw = pd.read_excel(IMAGE_RECORD, sheet_name='1-205', header=None)
#         groups = []
#         # Three column groups at offsets 0, 9, 17
#         for start in [0, 9, 17]:
#             grp = raw.iloc[1:, start:start + 6].copy()
#             grp.columns = ['Date', 'Sample_No', 'Image_No',
#                            'Moisture', 'Soil_type', 'SOC']
#             grp = grp.dropna(subset=['Image_No'])
#             grp['Sample_No'] = pd.to_numeric(grp['Sample_No'], errors='coerce')
#             grp['Image_No']  = pd.to_numeric(grp['Image_No'],  errors='coerce')
#             grp = grp.dropna(subset=['Sample_No', 'Image_No'])
#             groups.append(grp[['Sample_No', 'Image_No']])
#         return pd.concat(groups, ignore_index=True)

#     def _sheet_other(sheet_name, suffixes):
#         df = pd.read_excel(IMAGE_RECORD, sheet_name=sheet_name)
#         groups = []
#         for suf in suffixes:
#             sn  = 'Sample No.' + suf
#             ino = 'Image No.'  + suf
#             if sn in df.columns and ino in df.columns:
#                 grp = df[[sn, ino]].copy()
#                 grp.columns = ['Sample_No', 'Image_No']
#                 grp['Sample_No'] = pd.to_numeric(grp['Sample_No'],
#                                                   errors='coerce')
#                 grp['Image_No']  = pd.to_numeric(grp['Image_No'],
#                                                   errors='coerce')
#                 grp = grp.dropna(subset=['Sample_No', 'Image_No'])
#                 groups.append(grp)
#         return pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()

#     mapping = pd.concat([
#         _sheet_1_205(),
#         _sheet_other('206-360',  ['', '.1', '.2']),
#         _sheet_other('361-731',  ['', '.1', '.2', '.3', '.4']),
#     ], ignore_index=True)

#     mapping['Sample_No'] = mapping['Sample_No'].astype(int)
#     mapping['Image_No']  = mapping['Image_No'].astype(int)
#     mapping = (mapping
#                .drop_duplicates(subset=['Image_No'])
#                .sort_values('Image_No')
#                .reset_index(drop=True))

#     print(f"  Total images in mapping : {len(mapping)}")
#     print(f"  Total unique samples    : {mapping['Sample_No'].nunique()}")

#     # Save mapping for reference
#     mapping.to_csv(os.path.join(OUT_DIR, 'image_sample_mapping.csv'),
#                    index=False)
#     print(f"  Saved: image_sample_mapping.csv")
#     return mapping


# # ═════════════════════════════════════════════════════════════════════════════
# # STEP 4 — GROUPED SAMPLE SPLIT  (replaces Kennard-Stone)
# # ═════════════════════════════════════════════════════════════════════════════
# def grouped_sample_split(df_clean, mapping):
#     """
#     Assign ALL images from each physical soil sample to either
#     calibration OR validation — never both.

#     This replaces the original Kennard-Stone image-level split
#     which caused data leakage (images of the same soil appeared
#     on both sides of the split).

#     Parameters
#     ----------
#     df_clean : cleaned DataFrame after outlier removal
#     mapping  : DataFrame with columns Image_No, Sample_No

#     Returns
#     -------
#     df_calib, df_valid
#     """
#     print("\n" + "=" * 65)
#     print("STEP 4: Grouped Sample Split  (Reviewer-Compliant)")
#     print("        Replaces: Kennard-Stone image-level split")
#     print("=" * 65)
#     print(f"\n  Validation samples held out entirely:")
#     for s in VALIDATION_SAMPLE_NOS:
#         print(f"    Sample {s:2d} — {SAMPLE_INFO.get(s,'')}")

#     # Merge sample numbers into clean dataset
#     df = df_clean.merge(
#         mapping.rename(columns={'Image_No': 'image_no'}),
#         on='image_no',
#         how='left'
#     )

#     # Check for any unmatched images
#     unmatched = df['Sample_No'].isna().sum()
#     if unmatched > 0:
#         print(f"\n  ⚠  WARNING: {unmatched} images have no sample mapping")
#         print(f"     Check that IMAGe_record_image_No__copy.xlsx covers all images")

#     df['Sample_No'] = pd.to_numeric(df['Sample_No'],
#                                      errors='coerce').astype('Int64')

#     # Split on sample number
#     val_mask = df['Sample_No'].isin(VALIDATION_SAMPLE_NOS)
#     df_valid = df[val_mask].reset_index(drop=True)
#     df_calib = df[~val_mask].reset_index(drop=True)

#     # ── Verify zero overlap ───────────────────────────────────────────────────
#     val_imgs = set(df_valid['image_no'].tolist())
#     cal_imgs = set(df_calib['image_no'].tolist())
#     overlap  = val_imgs & cal_imgs
#     if overlap:
#         raise ValueError(
#             f"❌ OVERLAP DETECTED — {len(overlap)} images appear in both sets!\n"
#             f"   Image numbers: {sorted(overlap)}")
#     print(f"\n  ✅ Zero overlap confirmed — no image appears in both sets")

#     # ── Print summary ─────────────────────────────────────────────────────────
#     print(f"\n  Calibration : {len(df_calib)} images  "
#           f"({df_calib['Sample_No'].nunique()} samples)")
#     print(f"  Validation  : {len(df_valid)} images  "
#           f"({df_valid['Sample_No'].nunique()} samples)")

#     print(f"\n  Validation breakdown (after outlier removal):")
#     for s in VALIDATION_SAMPLE_NOS:
#         n   = len(df_valid[df_valid['Sample_No'] == s])
#         soc = df_valid[df_valid['Sample_No'] == s]['soc'].iloc[0] \
#               if n > 0 else 'N/A'
#         print(f"    Sample {s:2d}  {SAMPLE_INFO.get(s,''):22s}  "
#               f"SOC={soc}  →  {n} images")

#     cal_samples = sorted(
#         df_calib['Sample_No'].dropna().unique().tolist())
#     print(f"\n  Calibration samples: {cal_samples}")

#     # ── Save outputs ──────────────────────────────────────────────────────────
#     df_calib.to_csv(os.path.join(OUT_DIR, 'calibration_set.csv'), index=False)
#     df_valid.to_csv(os.path.join(OUT_DIR, 'validation_set.csv'),  index=False)
#     print(f"\n  Saved: calibration_set.csv  ({len(df_calib)} rows)")
#     print(f"  Saved: validation_set.csv   ({len(df_valid)} rows)")

#     return df_calib, df_valid


# # ═════════════════════════════════════════════════════════════════════════════
# # STEP 5 — SUMMARY REPORT
# # ═════════════════════════════════════════════════════════════════════════════
# def write_summary(merged, df_clean, df_outliers, df_calib, df_valid,
#                   distances, threshold, n_vars):
#     outlier_ids = sorted(df_outliers['Numeric numbers'].tolist())
#     cal_samples = sorted(
#         df_calib['Sample_No'].dropna().unique().tolist())

#     lines = [
#         "=" * 65,
#         "MAHALANOBIS OUTLIER REMOVAL + GROUPED SAMPLE SPLIT",
#         "REVISED VERSION — Reviewer Response",
#         "Reference: De Maesschalck et al. (2000)",
#         "=" * 65,
#         "",
#         "-- METHOD CHANGE FROM ORIGINAL -----------------------",
#         "  ORIGINAL : Kennard-Stone image-level split (data leakage)",
#         "  REVISED  : Grouped sample-level split (reviewer compliant)",
#         "  REASON   : Images of the same physical soil sample must",
#         "             never appear on both sides of the split.",
#         "",
#         "-- DATASET -------------------------------------------",
#         f"  Total images (master merged)  : {len(merged)}",
#         f"  Feature variables             : {n_vars}",
#         "",
#         "-- MAHALANOBIS CONFIGURATION -------------------------",
#         f"  Covariance inverse : np.linalg.pinv (pseudo-inverse)",
#         f"  Confidence level   : 99%",
#         f"  Chi-sq threshold   : chi2.ppf(0.99, df={n_vars}) = "
#         f"{chi2.ppf(CONFIDENCE, df=n_vars):.4f}",
#         f"  Distance threshold : sqrt(chi2) = {threshold:.4f}",
#         f"  Outlier criterion  : D > {threshold:.4f}",
#         "",
#         "-- OUTLIER DETECTION RESULTS -------------------------",
#         f"  Outliers removed : {len(df_outliers)}",
#         f"  Clean samples    : {len(df_clean)}",
#         "",
#         f"  Outlier image numbers:",
#         f"    {outlier_ids}",
#         "",
#         "-- GROUPED SAMPLE SPLIT ------------------------------",
#         f"  Calibration set : {len(df_calib)} images  "
#         f"({df_calib['Sample_No'].nunique()} samples)",
#         f"  Validation set  : {len(df_valid)} images  "
#         f"({df_valid['Sample_No'].nunique()} samples)",
#         "",
#         "  Validation samples (held out entirely):",
#     ]
#     for s in VALIDATION_SAMPLE_NOS:
#         n   = len(df_valid[df_valid['Sample_No'] == s])
#         soc = df_valid[df_valid['Sample_No'] == s]['soc'].iloc[0] \
#               if n > 0 else 'N/A'
#         lines.append(f"    Sample {s:2d}  {SAMPLE_INFO.get(s,''):22s}  "
#                      f"SOC={soc}  {n} images")
#     lines += [
#         "",
#         f"  Calibration samples: {cal_samples}",
#         "",
#         "-- OUTPUT FILES --------------------------------------",
#         f"  master_merged.csv            - {len(merged)} rows",
#         f"  outlier_detection_results.csv- {len(merged)} rows + distance & flag",
#         f"  soil_features_cleaned.csv    - {len(df_clean)} rows (outliers removed)",
#         f"  image_sample_mapping.csv     - image → sample mapping",
#         f"  calibration_set.csv          - {len(df_calib)} rows (grouped split)",
#         f"  validation_set.csv           - {len(df_valid)} rows (grouped split)",
#         f"  mahalanobis_plot.png         - distance plot",
#         "",
#         "-- DISTANCE STATISTICS --------------------------------",
#         f"  Min    : {distances.min():.4f}",
#         f"  Max    : {distances.max():.4f}",
#         f"  Mean   : {distances.mean():.4f}",
#         f"  Median : {np.median(distances):.4f}",
#         f"  Std    : {distances.std():.4f}",
#         "=" * 65,
#     ]

#     report = "\n".join(lines)
#     print("\n" + report)
#     with open(os.path.join(OUT_DIR, 'summary_report.txt'), 'w', encoding='utf-8') as f:
#         f.write(report)
#     print(f"\n  Saved: summary_report.txt")


# # ═════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ═════════════════════════════════════════════════════════════════════════════
# def main():
#     print("\n" + "=" * 65)
#     print("SOC PIPELINE — STEP 1 (REVISED)")
#     print("Outlier Removal + Grouped Sample Split")
#     print("=" * 65)

#     # Step 1 — load and merge (unchanged)
#     merged = load_and_merge()

#     # Step 2 — Mahalanobis outlier removal (unchanged)
#     df_clean, df_outliers, distances, threshold, feature_cols = \
#         mahalanobis_outlier_removal(merged)

#     # Step 3 — build image → sample mapping from Excel (NEW)
#     mapping = build_image_sample_mapping()

#     # Step 4 — grouped sample split (replaces Kennard-Stone)
#     df_calib, df_valid = grouped_sample_split(df_clean, mapping)

#     # Step 5 — summary report (updated)
#     write_summary(merged, df_clean, df_outliers, df_calib, df_valid,
#                   distances, threshold, len(feature_cols))

#     print(f"\n✅ All outputs saved to: {OUT_DIR}/")
#     print(f"   Calibration : {len(df_calib)} images")
#     print(f"   Validation  : {len(df_valid)} images")


# if __name__ == '__main__':
#     main()

"""
STEP 1 — GROUPED PHYSICAL-SAMPLE SPLIT + CALIBRATION-ONLY OUTLIER QC
====================================================================

Reviewer-compliant workflow.

Key change from original:
    ORIGINAL:
        Image-level splitting allowed repeated images from the same
        physical soil to occur in both calibration and validation.

    REVISED:
        All images from a physical soil sample are assigned entirely
        to calibration OR validation.

Pipeline:
    1. Load and merge image features + metadata
    2. Build image -> physical Sample_No mapping
    3. Perform grouped physical-sample calibration/validation split
    4. Perform Mahalanobis outlier detection on CALIBRATION ONLY
    5. Save cleaned calibration data and untouched validation data
    6. Generate summary report

Validation physical samples:
    Sample 3
    Sample 12
    Sample 13
    Sample 15

Important:
    - SOC is NOT used for Mahalanobis outlier detection.
    - Moisture is NOT used for Mahalanobis outlier detection.
    - Sample_No and image identifiers are NOT predictors.
    - Validation observations do NOT influence outlier detection.
"""

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

FEATURES_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "soil_image_features_without_commas.csv",
)

METADATA_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "image_with_soc_metadata.csv",
)

IMAGE_RECORD = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "IMAGe record_image No._copy.xlsx",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step1_output",
)

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

CONFIDENCE = 0.99

# Four physical samples held out completely.
VALIDATION_SAMPLE_NOS = [3, 7, 13, 15]

SAMPLE_INFO = {
    3: "Crosby (Mulch plots)",
    7: "Kokomo (Soybean)",
    13: "Miami (Corn)",
    15: "Crosby (Corn)",
}

# These columns must NOT participate in Mahalanobis distance.
EXCLUDE_COLS = [
    "filename",
    "Numeric numbers",
    "image_path",
    "image_no",
    "Image_No",
    "soil_type",
    "moisture",
    "soc",
    "Sample_No",
    "Mahalanobis_Distance",
    "Is_Outlier",
    "split",
]


# =============================================================================
# STEP 1 — LOAD AND MERGE FEATURE + METADATA FILES
# =============================================================================

def load_and_merge():

    print("=" * 70)
    print("STEP 1: LOAD AND MERGE IMAGE FEATURES + METADATA")
    print("=" * 70)

    feat = pd.read_csv(FEATURES_CSV)
    meta = pd.read_csv(METADATA_CSV)

    merged = pd.merge(
        feat,
        meta,
        left_on="Numeric numbers",
        right_on="image_no",
        how="inner",
    )

    # Ensure one row per image.
    merged = (
        merged
        .drop_duplicates(subset=["image_no"])
        .reset_index(drop=True)
    )

    print(
        f"\nFeatures file : "
        f"{feat.shape[0]} rows x {feat.shape[1]} columns"
    )

    print(
        f"Metadata file : "
        f"{meta.shape[0]} rows x {meta.shape[1]} columns"
    )

    print(
        f"Merged master : "
        f"{merged.shape[0]} rows x {merged.shape[1]} columns"
    )

    merged.to_csv(
        os.path.join(
            OUT_DIR,
            "master_merged_before_sample_mapping.csv",
        ),
        index=False,
    )

    return merged


# =============================================================================
# STEP 2 — BUILD IMAGE -> PHYSICAL SAMPLE MAPPING
# =============================================================================

def build_image_sample_mapping():

    print("\n" + "=" * 70)
    print("STEP 2: BUILD IMAGE -> PHYSICAL SAMPLE MAPPING")
    print("=" * 70)

    xl = pd.ExcelFile(IMAGE_RECORD)

    # -------------------------------------------------------------------------
    # Sheet 1: images 1–205
    # -------------------------------------------------------------------------

    def read_sheet_1_205():

        raw = pd.read_excel(
            xl,
            sheet_name="1-205",
            header=None,
        )

        groups = []

        for start in [0, 9, 17]:

            grp = raw.iloc[
                1:,
                start:start + 6
            ].copy()

            grp.columns = [
                "Date",
                "Sample_No",
                "Image_No",
                "Moisture",
                "Soil_type",
                "SOC",
            ]

            grp = grp.dropna(
                subset=["Image_No"]
            )

            grp["Sample_No"] = pd.to_numeric(
                grp["Sample_No"],
                errors="coerce",
            )

            grp["Image_No"] = pd.to_numeric(
                grp["Image_No"],
                errors="coerce",
            )

            grp = grp.dropna(
                subset=[
                    "Sample_No",
                    "Image_No",
                ]
            )

            groups.append(
                grp[
                    [
                        "Sample_No",
                        "Image_No",
                    ]
                ]
            )

        return pd.concat(
            groups,
            ignore_index=True,
        )

    # -------------------------------------------------------------------------
    # Remaining sheets
    # -------------------------------------------------------------------------

    def read_other_sheet(
        sheet_name,
        suffixes,
    ):

        df = pd.read_excel(
            xl,
            sheet_name=sheet_name,
        )

        groups = []

        for suffix in suffixes:

            sample_col = (
                "Sample No." + suffix
            )

            image_col = (
                "Image No." + suffix
            )

            if (
                sample_col in df.columns
                and image_col in df.columns
            ):

                grp = df[
                    [
                        sample_col,
                        image_col,
                    ]
                ].copy()

                grp.columns = [
                    "Sample_No",
                    "Image_No",
                ]

                grp["Sample_No"] = pd.to_numeric(
                    grp["Sample_No"],
                    errors="coerce",
                )

                grp["Image_No"] = pd.to_numeric(
                    grp["Image_No"],
                    errors="coerce",
                )

                grp = grp.dropna(
                    subset=[
                        "Sample_No",
                        "Image_No",
                    ]
                )

                groups.append(grp)

        if not groups:
            return pd.DataFrame(
                columns=[
                    "Sample_No",
                    "Image_No",
                ]
            )

        return pd.concat(
            groups,
            ignore_index=True,
        )

    # -------------------------------------------------------------------------
    # Combine all image mappings
    # -------------------------------------------------------------------------

    mapping = pd.concat(
        [
            read_sheet_1_205(),

            read_other_sheet(
                "206-360",
                [
                    "",
                    ".1",
                    ".2",
                ],
            ),

            read_other_sheet(
                "361-731",
                [
                    "",
                    ".1",
                    ".2",
                    ".3",
                    ".4",
                ],
            ),
        ],
        ignore_index=True,
    )

    mapping["Sample_No"] = (
        mapping["Sample_No"]
        .astype(int)
    )

    mapping["Image_No"] = (
        mapping["Image_No"]
        .astype(int)
    )

    mapping = (
        mapping
        .drop_duplicates(
            subset=["Image_No"]
        )
        .sort_values("Image_No")
        .reset_index(drop=True)
    )

    print(
        f"\nImages mapped      : "
        f"{len(mapping)}"
    )

    print(
        f"Physical samples   : "
        f"{mapping['Sample_No'].nunique()}"
    )

    print(
        "Sample numbers     : "
        f"{sorted(mapping['Sample_No'].unique())}"
    )

    mapping.to_csv(
        os.path.join(
            OUT_DIR,
            "image_sample_mapping.csv",
        ),
        index=False,
    )

    return mapping


# =============================================================================
# STEP 3 — ATTACH PHYSICAL SAMPLE ID
# =============================================================================

def attach_sample_numbers(
    merged,
    mapping,
):

    print("\n" + "=" * 70)
    print("STEP 3: ATTACH PHYSICAL SAMPLE ID TO EACH IMAGE")
    print("=" * 70)

    df = merged.merge(
        mapping.rename(
            columns={
                "Image_No": "image_no"
            }
        ),
        on="image_no",
        how="left",
    )

    df["Sample_No"] = pd.to_numeric(
        df["Sample_No"],
        errors="coerce",
    ).astype("Int64")

    unmatched = int(
        df["Sample_No"]
        .isna()
        .sum()
    )

    if unmatched > 0:

        unmatched_ids = (
            df.loc[
                df["Sample_No"].isna(),
                "image_no",
            ]
            .tolist()
        )

        raise ValueError(
            f"{unmatched} images do not have "
            f"a Sample_No mapping.\n"
            f"Images: {unmatched_ids}"
        )

    print(
        f"\nMapped images       : "
        f"{len(df)}"
    )

    print(
        f"Physical samples    : "
        f"{df['Sample_No'].nunique()}"
    )

    df.to_csv(
        os.path.join(
            OUT_DIR,
            "master_merged.csv",
        ),
        index=False,
    )

    return df


# =============================================================================
# STEP 4 — GROUPED PHYSICAL-SAMPLE SPLIT
# =============================================================================

def grouped_sample_split(df):

    print("\n" + "=" * 70)
    print("STEP 4: GROUPED PHYSICAL-SAMPLE SPLIT")
    print("=" * 70)

    print(
        "\nPhysical samples held out "
        "entirely for external validation:"
    )

    for sample_no in VALIDATION_SAMPLE_NOS:

        print(
            f"  Sample {sample_no:2d} — "
            f"{SAMPLE_INFO.get(sample_no, '')}"
        )

    # Validation mask based on physical soil identity.
    val_mask = (
        df["Sample_No"]
        .isin(
            VALIDATION_SAMPLE_NOS
        )
    )

    df_valid = (
        df.loc[val_mask]
        .copy()
        .reset_index(drop=True)
    )

    df_calib_raw = (
        df.loc[~val_mask]
        .copy()
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Verify physical-sample independence
    # -------------------------------------------------------------------------

    cal_samples = set(
        df_calib_raw[
            "Sample_No"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    val_samples = set(
        df_valid[
            "Sample_No"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    sample_overlap = (
        cal_samples
        & val_samples
    )

    if sample_overlap:

        raise ValueError(
            "PHYSICAL-SAMPLE LEAKAGE DETECTED: "
            f"{sorted(sample_overlap)}"
        )

    # -------------------------------------------------------------------------
    # Verify image ID independence also
    # -------------------------------------------------------------------------

    cal_images = set(
        df_calib_raw[
            "image_no"
        ].tolist()
    )

    val_images = set(
        df_valid[
            "image_no"
        ].tolist()
    )

    image_overlap = (
        cal_images
        & val_images
    )

    if image_overlap:

        raise ValueError(
            "IMAGE LEAKAGE DETECTED: "
            f"{len(image_overlap)} images"
        )

    print(
        f"\nCalibration before outlier QC : "
        f"{len(df_calib_raw)} images "
        f"from {len(cal_samples)} physical samples"
    )

    print(
        f"Validation                      : "
        f"{len(df_valid)} images "
        f"from {len(val_samples)} physical samples"
    )

    print(
        f"\nCalibration samples: "
        f"{sorted(cal_samples)}"
    )

    print(
        f"Validation samples : "
        f"{sorted(val_samples)}"
    )

    print(
        "\nPhysical-sample overlap: ZERO"
    )

    print(
        "Image overlap          : ZERO"
    )

    print(
        "\nValidation breakdown:"
    )

    for sample_no in sorted(
        val_samples
    ):

        sample_rows = (
            df_valid[
                df_valid[
                    "Sample_No"
                ] == sample_no
            ]
        )

        soc_values = sorted(
            sample_rows[
                "soc"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        print(
            f"  Sample {sample_no:2d}: "
            f"{len(sample_rows):3d} images | "
            f"SOC values = {soc_values}"
        )

    return (
        df_calib_raw,
        df_valid,
    )


# =============================================================================
# STEP 5 — MAHALANOBIS OUTLIER DETECTION
#          CALIBRATION DATA ONLY
# =============================================================================

def mahalanobis_outlier_removal_calibration(
    df_calib_raw,
):

    print("\n" + "=" * 70)
    print(
        "STEP 5: MAHALANOBIS OUTLIER DETECTION "
        "— CALIBRATION ONLY"
    )
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Select ONLY genuine numerical image predictors.
    # SOC, moisture, identifiers and Sample_No are excluded.
    # -------------------------------------------------------------------------

    feature_cols = [
        column
        for column
        in df_calib_raw.columns

        if (
            column not in EXCLUDE_COLS
            and pd.api.types.is_numeric_dtype(
                df_calib_raw[column]
            )
        )
    ]

    print(
        f"\nImage predictors used : "
        f"{len(feature_cols)}"
    )

    print(
        f"Predictors             : "
        f"{feature_cols}"
    )

    # Important safety checks.
    forbidden = {
        "soc",
        "moisture",
        "Sample_No",
        "image_no",
        "Numeric numbers",
    }

    accidentally_included = (
        forbidden
        & set(feature_cols)
    )

    if accidentally_included:

        raise ValueError(
            "Forbidden columns entered "
            "Mahalanobis calculation: "
            f"{sorted(accidentally_included)}"
        )

    # -------------------------------------------------------------------------
    # Calibration predictor matrix
    # -------------------------------------------------------------------------

    X = (
        df_calib_raw[
            feature_cols
        ]
        .copy()
    )

    # Median imputation derived from CALIBRATION ONLY.
    calibration_medians = (
        X.median()
    )

    X = X.fillna(
        calibration_medians
    )

    # Check remaining missing values.
    if X.isna().any().any():

        bad_columns = (
            X.columns[
                X.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Missing values remain in "
            f"predictor columns: {bad_columns}"
        )

    # -------------------------------------------------------------------------
    # Mahalanobis mean/covariance from CALIBRATION ONLY
    # -------------------------------------------------------------------------

    X_np = X.to_numpy(
        dtype=float
    )

    mean_vec = (
        X_np.mean(
            axis=0
        )
    )

    cov_matrix = np.cov(
        X_np,
        rowvar=False,
    )

    # Pseudo-inverse accommodates highly correlated image variables.
    inv_cov = np.linalg.pinv(
        cov_matrix
    )

    distances = np.array(
        [
            mahalanobis(
                row,
                mean_vec,
                inv_cov,
            )
            for row
            in X_np
        ]
    )

    n_vars = len(
        feature_cols
    )

    chi_square_threshold = (
        chi2.ppf(
            CONFIDENCE,
            df=n_vars,
        )
    )

    distance_threshold = (
        np.sqrt(
            chi_square_threshold
        )
    )

    print(
        f"\nChi-square threshold "
        f"(99%, df={n_vars}) : "
        f"{chi_square_threshold:.4f}"
    )

    print(
        f"Mahalanobis distance threshold : "
        f"{distance_threshold:.4f}"
    )

    # -------------------------------------------------------------------------
    # Flag calibration outliers
    # -------------------------------------------------------------------------

    df_calib_flagged = (
        df_calib_raw.copy()
    )

    df_calib_flagged[
        "Mahalanobis_Distance"
    ] = distances

    df_calib_flagged[
        "Is_Outlier"
    ] = (
        distances
        > distance_threshold
    )

    df_outliers = (
        df_calib_flagged.loc[
            df_calib_flagged[
                "Is_Outlier"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    df_calib_clean = (
        df_calib_flagged.loc[
            ~df_calib_flagged[
                "Is_Outlier"
            ]
        ]
        .drop(
            columns=[
                "Is_Outlier"
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nCalibration before QC : "
        f"{len(df_calib_raw)} images"
    )

    print(
        f"Outlier images removed: "
        f"{len(df_outliers)}"
    )

    print(
        f"Clean calibration     : "
        f"{len(df_calib_clean)} images"
    )

    if len(df_outliers) > 0:

        print(
            "\nOutlier image numbers:"
        )

        print(
            sorted(
                df_outliers[
                    "image_no"
                ].tolist()
            )
        )

    # -------------------------------------------------------------------------
    # Save QC information
    # -------------------------------------------------------------------------

    df_calib_flagged.to_csv(
        os.path.join(
            OUT_DIR,
            "calibration_outlier_detection_results.csv",
        ),
        index=False,
    )

    df_outliers.to_csv(
        os.path.join(
            OUT_DIR,
            "calibration_outliers.csv",
        ),
        index=False,
    )

    _plot_mahalanobis(
        distances=distances,
        threshold=distance_threshold,
        is_outlier=(
            df_calib_flagged[
                "Is_Outlier"
            ].to_numpy()
        ),
        n=len(
            df_calib_flagged
        ),
        n_vars=n_vars,
    )

    return (
        df_calib_clean,
        df_outliers,
        distances,
        distance_threshold,
        feature_cols,
    )


# =============================================================================
# MAHALANOBIS QC FIGURE
# =============================================================================

def _plot_mahalanobis(
    distances,
    threshold,
    is_outlier,
    n,
    n_vars,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
    )

    fig.suptitle(
        "Mahalanobis Outlier Detection — "
        "Calibration Images Only\n"
        f"n={n} images | "
        f"{n_vars} image predictors | "
        f"99% threshold D={threshold:.4f} | "
        f"Outliers={is_outlier.sum()}",
        fontsize=12,
        fontweight="bold",
    )

    idx = np.arange(n)

    # -------------------------------------------------------------------------
    # Distance by image
    # -------------------------------------------------------------------------

    axes[0].scatter(
        idx[~is_outlier],
        distances[~is_outlier],
        s=14,
        alpha=0.55,
        label="Retained",
    )

    if is_outlier.sum() > 0:

        axes[0].scatter(
            idx[is_outlier],
            distances[is_outlier],
            s=50,
            alpha=0.9,
            label=(
                f"Outlier "
                f"(n={is_outlier.sum()})"
            ),
        )

    axes[0].axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"99% threshold "
            f"(D={threshold:.2f})"
        ),
    )

    axes[0].set_xlabel(
        "Calibration image index"
    )

    axes[0].set_ylabel(
        "Mahalanobis distance"
    )

    axes[0].set_title(
        "Calibration Image Distances"
    )

    axes[0].legend()
    axes[0].grid(
        True,
        linestyle=":",
        alpha=0.35,
    )

    # -------------------------------------------------------------------------
    # Distribution
    # -------------------------------------------------------------------------

    axes[1].hist(
        distances[~is_outlier],
        bins=35,
        alpha=0.70,
        label="Retained",
    )

    if is_outlier.sum() > 0:

        axes[1].hist(
            distances[is_outlier],
            bins=8,
            alpha=0.80,
            label="Outliers",
        )

    axes[1].axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"99% threshold "
            f"(D={threshold:.2f})"
        ),
    )

    axes[1].set_xlabel(
        "Mahalanobis distance"
    )

    axes[1].set_ylabel(
        "Number of images"
    )

    axes[1].set_title(
        "Distance Distribution"
    )

    axes[1].legend()
    axes[1].grid(
        True,
        linestyle=":",
        alpha=0.35,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            "mahalanobis_plot.png",
        ),
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# STEP 6 — SAVE FINAL STEP-1 DATASETS
# =============================================================================

def save_final_datasets(
    df_calib,
    df_valid,
):

    print("\n" + "=" * 70)
    print("STEP 6: SAVE FINAL STEP-1 DATASETS")
    print("=" * 70)

    calibration_path = os.path.join(
        OUT_DIR,
        "calibration_set.csv",
    )

    validation_path = os.path.join(
        OUT_DIR,
        "validation_set.csv",
    )

    df_calib.to_csv(
        calibration_path,
        index=False,
    )

    # Validation remains untouched by Mahalanobis QC.
    df_valid.to_csv(
        validation_path,
        index=False,
    )

    print(
        f"\nSaved calibration_set.csv : "
        f"{len(df_calib)} images"
    )

    print(
        f"Saved validation_set.csv  : "
        f"{len(df_valid)} images"
    )


# =============================================================================
# STEP 7 — SUMMARY REPORT
# =============================================================================

def write_summary(
    merged,
    df_calib_raw,
    df_calib_clean,
    df_valid,
    df_outliers,
    distances,
    threshold,
    feature_cols,
):

    cal_samples = sorted(
        df_calib_clean[
            "Sample_No"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    val_samples = sorted(
        df_valid[
            "Sample_No"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    sample_overlap = sorted(
        set(cal_samples)
        & set(val_samples)
    )

    lines = [
        "=" * 72,

        "GROUPED PHYSICAL-SAMPLE SPLIT "
        "+ CALIBRATION-ONLY OUTLIER QC",

        "=" * 72,

        "",

        "-- DATASET -----------------------------------------------",

        f"Total images                      : {len(merged)}",

        f"Physical soil samples             : "
        f"{merged['Sample_No'].nunique()}",

        "",

        "-- GROUPED SPLIT ------------------------------------------",

        f"Calibration before outlier QC     : "
        f"{len(df_calib_raw)} images",

        f"Calibration physical soils        : "
        f"{df_calib_raw['Sample_No'].nunique()}",

        f"Validation images                 : "
        f"{len(df_valid)}",

        f"Validation physical soils         : "
        f"{df_valid['Sample_No'].nunique()}",

        f"Validation samples                : "
        f"{val_samples}",

        f"Physical sample overlap           : "
        f"{sample_overlap}",

        "",

        "-- MAHALANOBIS QC -----------------------------------------",

        "Applied to                        : "
        "CALIBRATION ONLY",

        f"Image predictors used             : "
        f"{len(feature_cols)}",

        f"Predictor list                    : "
        f"{feature_cols}",

        "Target SOC included               : NO",

        "Moisture included                 : NO",

        "Sample_No included                : NO",

        f"Confidence level                  : "
        f"{CONFIDENCE * 100:.0f}%",

        f"Distance threshold                : "
        f"{threshold:.4f}",

        f"Calibration outliers removed      : "
        f"{len(df_outliers)}",

        f"Final clean calibration images    : "
        f"{len(df_calib_clean)}",

        "",

        "-- VALIDATION PROTECTION ----------------------------------",

        "Validation used for outlier QC    : NO",

        "Validation used for feature selection: NO",

        "All images from one physical soil remain together.",

        "",

        "-- VALIDATION BREAKDOWN -----------------------------------",
    ]

    for sample_no in val_samples:

        temp = (
            df_valid[
                df_valid[
                    "Sample_No"
                ] == sample_no
            ]
        )

        soc_values = sorted(
            temp[
                "soc"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        lines.append(
            f"Sample {sample_no:2d} : "
            f"{len(temp):3d} images | "
            f"SOC={soc_values}"
        )

    lines.extend(
        [
            "",

            "-- DISTANCE STATISTICS -----------------------------------",

            f"Minimum                           : "
            f"{distances.min():.4f}",

            f"Maximum                           : "
            f"{distances.max():.4f}",

            f"Mean                              : "
            f"{distances.mean():.4f}",

            f"Median                            : "
            f"{np.median(distances):.4f}",

            f"Standard deviation                : "
            f"{distances.std():.4f}",

            "",

            "-- FINAL OUTPUT ------------------------------------------",

            f"Calibration                       : "
            f"{len(df_calib_clean)} images "
            f"from {len(cal_samples)} soils",

            f"Validation                        : "
            f"{len(df_valid)} images "
            f"from {len(val_samples)} soils",

            f"Calibration soils                 : "
            f"{cal_samples}",

            f"Validation soils                  : "
            f"{val_samples}",

            "=" * 72,
        ]
    )

    report = "\n".join(
        lines
    )

    print(
        "\n" + report
    )

    with open(
        os.path.join(
            OUT_DIR,
            "summary_report.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n" + "=" * 72
    )

    print(
        "SOC PIPELINE — STEP 1"
    )

    print(
        "GROUPED PHYSICAL-SAMPLE SPLIT "
        "+ CALIBRATION-ONLY OUTLIER QC"
    )

    print(
        "=" * 72
    )

    # -------------------------------------------------------------------------
    # 1. Load image features and metadata
    # -------------------------------------------------------------------------

    merged = (
        load_and_merge()
    )

    # -------------------------------------------------------------------------
    # 2. Build image -> Sample_No map
    # -------------------------------------------------------------------------

    mapping = (
        build_image_sample_mapping()
    )

    # -------------------------------------------------------------------------
    # 3. Attach Sample_No before any splitting/QC
    # -------------------------------------------------------------------------

    merged = (
        attach_sample_numbers(
            merged,
            mapping,
        )
    )

    # -------------------------------------------------------------------------
    # 4. Grouped physical-sample split FIRST
    # -------------------------------------------------------------------------

    (
        df_calib_raw,
        df_valid,
    ) = grouped_sample_split(
        merged
    )

    # -------------------------------------------------------------------------
    # 5. Mahalanobis QC on calibration images ONLY
    # -------------------------------------------------------------------------

    (
        df_calib_clean,
        df_outliers,
        distances,
        threshold,
        feature_cols,
    ) = (
        mahalanobis_outlier_removal_calibration(
            df_calib_raw
        )
    )

    # -------------------------------------------------------------------------
    # 6. Final safety check after outlier removal
    # -------------------------------------------------------------------------

    final_cal_samples = set(
        df_calib_clean[
            "Sample_No"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    final_val_samples = set(
        df_valid[
            "Sample_No"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    final_overlap = (
        final_cal_samples
        & final_val_samples
    )

    if final_overlap:

        raise ValueError(
            "FINAL SAMPLE LEAKAGE DETECTED: "
            f"{sorted(final_overlap)}"
        )

    # -------------------------------------------------------------------------
    # 7. Save calibration + untouched validation datasets
    # -------------------------------------------------------------------------

    save_final_datasets(
        df_calib_clean,
        df_valid,
    )

    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------

    write_summary(
        merged=merged,
        df_calib_raw=df_calib_raw,
        df_calib_clean=df_calib_clean,
        df_valid=df_valid,
        df_outliers=df_outliers,
        distances=distances,
        threshold=threshold,
        feature_cols=feature_cols,
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "STEP 1 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nFinal calibration : "
        f"{len(df_calib_clean)} images | "
        f"{len(final_cal_samples)} physical soils"
    )

    print(
        f"Final validation  : "
        f"{len(df_valid)} images | "
        f"{len(final_val_samples)} physical soils"
    )

    print(
        f"Outliers removed  : "
        f"{len(df_outliers)} calibration images"
    )

    print(
        "Physical sample overlap: ZERO"
    )

    print(
        f"\nOutputs saved to:\n"
        f"{OUT_DIR}"
    )

    print(
        "\nNext step:"
        "\npython "
        "\"notebooks/objective_1/"
        "step2_feature_selection.py\""
    )


if __name__ == "__main__":
    main()