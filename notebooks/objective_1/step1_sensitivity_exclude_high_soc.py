#here we dropped values of sample 7 and one value of samepl2 which had soc level 7.72 in order to drop high soc values and see the effect on feature selection. aiming for better R2
"""



Mahalanobis Distance Outlier Removal + Grouped Sample Split
============================================================
REVISED VERSION — Reviewer Response

Key change from original:
  ORIGINAL : Kennard-Stone image-level split (70/30)
             → caused data leakage (same soil on both sides)
  REVISED  : Grouped sample-level split
             → all images from a physical soil sample go to
               calibration OR validation — never both

Pipeline:
  1. Load & merge feature CSV + metadata CSV  (unchanged)
  2. Mahalanobis outlier removal              (unchanged)
  3. Load image → sample mapping from Excel
  4. Grouped sample split (replaces Step 3 Kennard-Stone)
  5. Summary report

Validation samples (held out entirely — all their images):
  Sample  3  Crosby (Mulch plots)   SOC=3.34   ~21 images
  Sample 12  Kokomo (Soybean)       SOC=1.92   ~44 images
  Sample 13  Miami (Corn)           SOC=2.50   ~39 images
  Sample 15  Crosby (Corn)          SOC=1.75   ~55 images
  Sample 19  Kokomo (Soybean)       SOC=2.51   ~51 images

Expected output after outlier removal:
  Calibration : ~460 images  (15 samples)
  Validation  : ~198 images  (5 samples)

Reference: De Maesschalck et al. (2000)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — update PROJECT_ROOT only, everything else updates automatically
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed'
FEATURES_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw', 'soil_image_features_without_commas.csv')
METADATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw', 'image_with_soc_metadata.csv')
IMAGE_RECORD = os.path.join(PROJECT_ROOT, 'data', 'raw', 'IMAGe record_image No._copy.xlsx')
OUT_DIR      = os.path.join(PROJECT_ROOT, 'notebooks', 'objective_1', 'output_data', 'step1_output_sensitivity')

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Columns to exclude from Mahalanobis distance calculation
EXCLUDE_COLS = ['filename', 'Numeric numbers', 'image_path',
                'image_no', 'soil_type', 'soc', 'moisture']

# Confidence level for chi-squared outlier threshold
CONFIDENCE = 0.99       # 99% — De Maesschalck et al. (2000)

# ── Grouped split configuration ───────────────────────────────────────────────
# These 5 sample numbers and ALL their images go to validation only
# Remaining 15 samples go entirely to calibration
VALIDATION_SAMPLE_NOS = [3, 12, 13, 15]



SAMPLE_INFO = {
    3:  'Crosby (Mulch plots)',
    12: 'Kokomo (Soybean)',
    13: 'Miami (Corn)',
    15: 'Crosby (Corn)',
}
# ============================================================
# SENSITIVITY ANALYSIS — EXPLICIT IMAGE EXCLUSIONS
# ============================================================

EXCLUDED_IMAGE_NOS = [
    # Sample 2 — SOC = 7.76
    2,

    # Sample 7
    12, 13, 14,
    37, 38, 39,
    57, 58,
    77, 78, 79,
    99, 100,
    125, 126, 127,
    150, 151, 152,
    177, 178, 179,
    198, 199, 200, 201
]

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & MERGE  (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════════
def load_and_merge():
    print("=" * 65)
    print("STEP 1: Loading and merging files")
    print("=" * 65)

    feat   = pd.read_csv(FEATURES_CSV)
    meta   = pd.read_csv(METADATA_CSV)
    merged = pd.merge(feat, meta,
                      left_on='Numeric numbers',
                      right_on='image_no',
                      how='inner')

    print(f"  Features file : {feat.shape[0]} rows x {feat.shape[1]} cols")
    print(f"  Metadata file : {meta.shape[0]} rows x {meta.shape[1]} cols")
    print(f"  Merged master : {merged.shape[0]} rows x {merged.shape[1]} cols")

    merged.to_csv(os.path.join(OUT_DIR, 'master_merged.csv'), index=False)
    print(f"  Saved: master_merged.csv")
    return merged


# ═════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS — REMOVE PREDEFINED IMAGE RECORDS
# ═════════════════════════════════════════════════════════════

def remove_predefined_images(df):

    print("\n" + "=" * 65)
    print("SENSITIVITY ANALYSIS: Removing predefined image records")
    print("=" * 65)

    df = df.copy()

    # Ensure image_no is numeric
    df['image_no'] = pd.to_numeric(
        df['image_no'],
        errors='coerce'
    )

    # Find records requested for removal
    excluded_rows = df[
        df['image_no'].isin(EXCLUDED_IMAGE_NOS)
    ].copy()

    print(
        f"  Requested exclusions : "
        f"{len(EXCLUDED_IMAGE_NOS)} images"
    )

    print(
        f"  Matching rows found  : "
        f"{len(excluded_rows)}"
    )

    # Show exactly what is being removed
    show_cols = [
        c for c in [
            'image_no',
            'soc',
            'moisture',
            'soil_type'
        ]
        if c in excluded_rows.columns
    ]

    if len(excluded_rows) > 0:

        print("\n  Records being removed:")

        print(
            excluded_rows[
                show_cols
            ]
            .sort_values('image_no')
            .to_string(index=False)
        )

    # Check whether any requested image numbers are missing
    found_ids = set(
        excluded_rows['image_no']
        .dropna()
        .astype(int)
        .tolist()
    )

    missing_ids = sorted(
        set(EXCLUDED_IMAGE_NOS) - found_ids
    )

    if missing_ids:

        print(
            "\n  WARNING: These requested images "
            "were not found:"
        )

        print(f"  {missing_ids}")

    # Save audit trail BEFORE deleting them
    excluded_rows.to_csv(
        os.path.join(
            OUT_DIR,
            'explicitly_excluded_images.csv'
        ),
        index=False
    )

    n_before = len(df)

    df_filtered = df[
        ~df['image_no'].isin(
            EXCLUDED_IMAGE_NOS
        )
    ].copy()

    df_filtered = (
        df_filtered
        .reset_index(drop=True)
    )

    df_filtered.to_csv(
    os.path.join(
        OUT_DIR,
        'master_merged_filtered.csv'
    ),
    index=False
)

    print(
        f"  Saved: master_merged_filtered.csv "
        f"({len(df_filtered)} rows)"
    )

    n_removed = n_before - len(df_filtered)

    print(f"\n  Dataset before : {n_before}")
    print(f"  Explicitly removed : {n_removed}")
    print(f"  Dataset after  : {len(df_filtered)}")

    # Safety check
    remaining_excluded = df_filtered[
        'image_no'
    ].isin(EXCLUDED_IMAGE_NOS).sum()

    if remaining_excluded != 0:

        raise ValueError(
            "Excluded images still exist in filtered data."
        )

    print(
        "  Verification: all requested image records removed."
    )

    return df_filtered, excluded_rows


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — MAHALANOBIS OUTLIER REMOVAL  (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════════
def mahalanobis_outlier_removal(df):
    print("\n" + "=" * 65)
    print("STEP 2: Mahalanobis Distance Outlier Detection")
    print("        (De Maesschalck et al., 2000)")
    print("=" * 65)

    # Select numeric feature columns — drop identifiers
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.drop(
        columns=[c for c in EXCLUDE_COLS if c in numeric_df.columns])

    n_vars = numeric_df.shape[1]
    n_obs  = numeric_df.shape[0]
    print(f"  Variables : {n_vars}")
    print(f"  Samples   : {n_obs}")

    # Mean vector + covariance matrix
    mean_vec   = numeric_df.mean().values
    cov_matrix = np.cov(numeric_df.T)

    # Pseudo-inverse handles non-full-rank covariance (correlated features)
    inv_cov = np.linalg.pinv(cov_matrix)

    print(f"\n  Computing Mahalanobis distances ...")
    distances = numeric_df.apply(
        lambda row: mahalanobis(row, mean_vec, inv_cov), axis=1)

    # 99% chi-squared threshold
    threshold   = chi2.ppf(CONFIDENCE, df=n_vars)
    threshold_d = np.sqrt(threshold)
    print(f"  Chi-sq threshold (99%, df={n_vars}): {threshold:.4f}")
    print(f"  Distance threshold (sqrt)          : {threshold_d:.4f}")

    df = df.copy()
    df['Mahalanobis_Distance'] = distances.values
    df['Is_Outlier']           = distances > threshold_d

    outlier_count = int(df['Is_Outlier'].sum())
    print(f"\n  Outliers detected : {outlier_count}")
    print(f"  Clean samples     : {n_obs - outlier_count}")

    # Save full results
    df.to_csv(os.path.join(OUT_DIR, 'outlier_detection_results.csv'),
              index=False)
    print(f"  Saved: outlier_detection_results.csv")

    # Save cleaned dataset
    df_cleaned = (df[~df['Is_Outlier']]
                  .drop(columns=['Is_Outlier'])
                  .reset_index(drop=True))
    df_cleaned.to_csv(os.path.join(OUT_DIR, 'soil_features_cleaned.csv'),
                      index=False)
    print(f"  Saved: soil_features_cleaned.csv  ({len(df_cleaned)} rows)")

    # Log outlier image numbers
    df_outliers = df[df['Is_Outlier']].copy()
    print(f"\n  Outlier image numbers:")
    print(f"    {sorted(df_outliers['Numeric numbers'].tolist())}")

    _plot_mahalanobis(distances.values, threshold_d,
                      df['Is_Outlier'].values, n_obs, n_vars)

    return df_cleaned, df_outliers, distances.values, threshold_d, \
        numeric_df.columns.tolist()


def _plot_mahalanobis(distances, threshold, is_outlier, n, n_vars):
    """Mahalanobis distance plot — unchanged from original."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Mahalanobis Distance Outlier Detection  (De Maesschalck et al., 2000)\n'
        f'n={n} images  |  {n_vars} features  |  '
        f'99% confidence threshold D={threshold:.4f}  |  '
        f'Outliers removed: {is_outlier.sum()}',
        fontsize=12, fontweight='bold'
    )

    idx = np.arange(n)

    ax = axes[0]
    ax.scatter(idx[~is_outlier], distances[~is_outlier],
               color='#1565C0', s=14, alpha=0.55, label='Clean', zorder=3)
    ax.scatter(idx[is_outlier], distances[is_outlier],
               color='#C62828', s=55, alpha=0.95, zorder=5,
               label=f'Outlier (n={is_outlier.sum()})')
    ax.axhline(threshold, color='#E65100', lw=2.2, ls='--',
               label=f'99% threshold (D={threshold:.2f})', zorder=6)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Mahalanobis Distance (D)', fontsize=11)
    ax.set_title('Distance per Sample', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.98, 0.97,
            f'Outliers : {is_outlier.sum()}\nClean    : {(~is_outlier).sum()}',
            transform=ax.transAxes, fontsize=9.5, va='top', ha='right',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      edgecolor='#AAAAAA', alpha=0.92))

    ax2 = axes[1]
    ax2.hist(distances[~is_outlier], bins=35, color='#1565C0',
             alpha=0.70, edgecolor='white', lw=0.5, label='Clean')
    if is_outlier.sum() > 0:
        ax2.hist(distances[is_outlier], bins=8, color='#C62828',
                 alpha=0.90, edgecolor='white', lw=0.5,
                 label=f'Outlier (n={is_outlier.sum()})')
    ax2.axvline(threshold, color='#E65100', lw=2.2, ls='--',
                label=f'99% threshold (D={threshold:.2f})')
    ax2.set_xlabel('Mahalanobis Distance (D)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distance Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, ls=':', alpha=0.35)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mahalanobis_plot.png'),
                dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mahalanobis_plot.png")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD IMAGE → SAMPLE MAPPING FROM EXCEL
# ═════════════════════════════════════════════════════════════════════════════
def build_image_sample_mapping():
    """
    Parse all three sheets of the image record Excel file.
    Returns DataFrame with columns: Image_No, Sample_No
    covering all 731 images.
    """
    print("\n" + "=" * 65)
    print("STEP 3: Building image → sample mapping from Excel")
    print("=" * 65)

    def _sheet_1_205():
        raw = pd.read_excel(IMAGE_RECORD, sheet_name='1-205', header=None)
        groups = []
        # Three column groups at offsets 0, 9, 17
        for start in [0, 9, 17]:
            grp = raw.iloc[1:, start:start + 6].copy()
            grp.columns = ['Date', 'Sample_No', 'Image_No',
                           'Moisture', 'Soil_type', 'SOC']
            grp = grp.dropna(subset=['Image_No'])
            grp['Sample_No'] = pd.to_numeric(grp['Sample_No'], errors='coerce')
            grp['Image_No']  = pd.to_numeric(grp['Image_No'],  errors='coerce')
            grp = grp.dropna(subset=['Sample_No', 'Image_No'])
            groups.append(grp[['Sample_No', 'Image_No']])
        return pd.concat(groups, ignore_index=True)

    def _sheet_other(sheet_name, suffixes):
        df = pd.read_excel(IMAGE_RECORD, sheet_name=sheet_name)
        groups = []
        for suf in suffixes:
            sn  = 'Sample No.' + suf
            ino = 'Image No.'  + suf
            if sn in df.columns and ino in df.columns:
                grp = df[[sn, ino]].copy()
                grp.columns = ['Sample_No', 'Image_No']
                grp['Sample_No'] = pd.to_numeric(grp['Sample_No'],
                                                  errors='coerce')
                grp['Image_No']  = pd.to_numeric(grp['Image_No'],
                                                  errors='coerce')
                grp = grp.dropna(subset=['Sample_No', 'Image_No'])
                groups.append(grp)
        return pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()

    mapping = pd.concat([
        _sheet_1_205(),
        _sheet_other('206-360',  ['', '.1', '.2']),
        _sheet_other('361-731',  ['', '.1', '.2', '.3', '.4']),
    ], ignore_index=True)

    mapping['Sample_No'] = mapping['Sample_No'].astype(int)
    mapping['Image_No']  = mapping['Image_No'].astype(int)
    mapping = (mapping
               .drop_duplicates(subset=['Image_No'])
               .sort_values('Image_No')
               .reset_index(drop=True))

    print(f"  Total images in mapping : {len(mapping)}")
    print(f"  Total unique samples    : {mapping['Sample_No'].nunique()}")

    # Save mapping for reference
    mapping.to_csv(os.path.join(OUT_DIR, 'image_sample_mapping.csv'),
                   index=False)
    print(f"  Saved: image_sample_mapping.csv")
    return mapping


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — GROUPED SAMPLE SPLIT  (replaces Kennard-Stone)
# ═════════════════════════════════════════════════════════════════════════════
def grouped_sample_split(df_clean, mapping):
    """
    Assign ALL images from each physical soil sample to either
    calibration OR validation — never both.

    This replaces the original Kennard-Stone image-level split
    which caused data leakage (images of the same soil appeared
    on both sides of the split).

    Parameters
    ----------
    df_clean : cleaned DataFrame after outlier removal
    mapping  : DataFrame with columns Image_No, Sample_No

    Returns
    -------
    df_calib, df_valid
    """
    print("\n" + "=" * 65)
    print("STEP 4: Grouped Sample Split  (Reviewer-Compliant)")
    print("        Replaces: Kennard-Stone image-level split")
    print("=" * 65)
    print(f"\n  Validation samples held out entirely:")
    for s in VALIDATION_SAMPLE_NOS:
        print(f"    Sample {s:2d} — {SAMPLE_INFO.get(s,'')}")

    # Merge sample numbers into clean dataset
    df = df_clean.merge(
        mapping.rename(columns={'Image_No': 'image_no'}),
        on='image_no',
        how='left'
    )

    # Check for any unmatched images
    unmatched = df['Sample_No'].isna().sum()
    if unmatched > 0:
        print(f"\n  ⚠  WARNING: {unmatched} images have no sample mapping")
        print(f"     Check that IMAGe_record_image_No__copy.xlsx covers all images")

    df['Sample_No'] = pd.to_numeric(df['Sample_No'],
                                     errors='coerce').astype('Int64')

    # Split on sample number
    val_mask = df['Sample_No'].isin(VALIDATION_SAMPLE_NOS)
    df_valid = df[val_mask].reset_index(drop=True)
    df_calib = df[~val_mask].reset_index(drop=True)

    # ── Verify zero overlap ───────────────────────────────────────────────────
    val_imgs = set(df_valid['image_no'].tolist())
    cal_imgs = set(df_calib['image_no'].tolist())
    overlap  = val_imgs & cal_imgs
    if overlap:
        raise ValueError(
            f"❌ OVERLAP DETECTED — {len(overlap)} images appear in both sets!\n"
            f"   Image numbers: {sorted(overlap)}")
    print(f"\n  ✅ Zero overlap confirmed — no image appears in both sets")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  Calibration : {len(df_calib)} images  "
          f"({df_calib['Sample_No'].nunique()} samples)")
    print(f"  Validation  : {len(df_valid)} images  "
          f"({df_valid['Sample_No'].nunique()} samples)")

    print(f"\n  Validation breakdown (after outlier removal):")
    for s in VALIDATION_SAMPLE_NOS:
        n   = len(df_valid[df_valid['Sample_No'] == s])
        soc = df_valid[df_valid['Sample_No'] == s]['soc'].iloc[0] \
              if n > 0 else 'N/A'
        print(f"    Sample {s:2d}  {SAMPLE_INFO.get(s,''):22s}  "
              f"SOC={soc}  →  {n} images")

    cal_samples = sorted(
        df_calib['Sample_No'].dropna().unique().tolist())
    print(f"\n  Calibration samples: {cal_samples}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    df_calib.to_csv(os.path.join(OUT_DIR, 'calibration_set.csv'), index=False)
    df_valid.to_csv(os.path.join(OUT_DIR, 'validation_set.csv'),  index=False)
    print(f"\n  Saved: calibration_set.csv  ({len(df_calib)} rows)")
    print(f"  Saved: validation_set.csv   ({len(df_valid)} rows)")

    return df_calib, df_valid


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — SUMMARY REPORT
# ═════════════════════════════════════════════════════════════════════════════
def write_summary(merged, df_clean, df_outliers, df_calib, df_valid,
                  distances, threshold, n_vars):
    outlier_ids = sorted(df_outliers['Numeric numbers'].tolist())
    cal_samples = sorted(
        df_calib['Sample_No'].dropna().unique().tolist())

    lines = [
        "=" * 65,
        "MAHALANOBIS OUTLIER REMOVAL + GROUPED SAMPLE SPLIT",
        "REVISED VERSION — Reviewer Response",
        "Reference: De Maesschalck et al. (2000)",
        "=" * 65,
        "",
        "-- METHOD CHANGE FROM ORIGINAL -----------------------",
        "  ORIGINAL : Kennard-Stone image-level split (data leakage)",
        "  REVISED  : Grouped sample-level split (reviewer compliant)",
        "  REASON   : Images of the same physical soil sample must",
        "             never appear on both sides of the split.",
        "",
        "-- DATASET -------------------------------------------",
        f"  Total images (master merged)  : {len(merged)}",
        f"  Feature variables             : {n_vars}",
        "",
        "-- MAHALANOBIS CONFIGURATION -------------------------",
        f"  Covariance inverse : np.linalg.pinv (pseudo-inverse)",
        f"  Confidence level   : 99%",
        f"  Chi-sq threshold   : chi2.ppf(0.99, df={n_vars}) = "
        f"{chi2.ppf(CONFIDENCE, df=n_vars):.4f}",
        f"  Distance threshold : sqrt(chi2) = {threshold:.4f}",
        f"  Outlier criterion  : D > {threshold:.4f}",
        "",

                "",
        "-- SENSITIVITY EXCLUSIONS -----------------------------",
        f"  Explicit image exclusions : {len(EXCLUDED_IMAGE_NOS)}",
        f"  Image numbers             : {EXCLUDED_IMAGE_NOS}",

        "-- OUTLIER DETECTION RESULTS -------------------------",
        f"  Outliers removed : {len(df_outliers)}",
        f"  Clean samples    : {len(df_clean)}",
        "",
        f"  Outlier image numbers:",
        f"    {outlier_ids}",
        "",
        "-- GROUPED SAMPLE SPLIT ------------------------------",
        f"  Calibration set : {len(df_calib)} images  "
        f"({df_calib['Sample_No'].nunique()} samples)",
        f"  Validation set  : {len(df_valid)} images  "
        f"({df_valid['Sample_No'].nunique()} samples)",
        "",
        "  Validation samples (held out entirely):",
    ]
    for s in VALIDATION_SAMPLE_NOS:
        n   = len(df_valid[df_valid['Sample_No'] == s])
        soc = df_valid[df_valid['Sample_No'] == s]['soc'].iloc[0] \
              if n > 0 else 'N/A'
        lines.append(f"    Sample {s:2d}  {SAMPLE_INFO.get(s,''):22s}  "
                     f"SOC={soc}  {n} images")
    lines += [
        "",
        f"  Calibration samples: {cal_samples}",
        "",
        "-- OUTPUT FILES --------------------------------------",
        f"  master_merged.csv            - {len(merged)} rows",
        f"  outlier_detection_results.csv- {len(merged)} rows + distance & flag",
        f"  soil_features_cleaned.csv    - {len(df_clean)} rows (outliers removed)",
        f"  image_sample_mapping.csv     - image → sample mapping",
        f"  calibration_set.csv          - {len(df_calib)} rows (grouped split)",
        f"  validation_set.csv           - {len(df_valid)} rows (grouped split)",
        f"  mahalanobis_plot.png         - distance plot",
        "",
        "-- DISTANCE STATISTICS --------------------------------",
        f"  Min    : {distances.min():.4f}",
        f"  Max    : {distances.max():.4f}",
        f"  Mean   : {distances.mean():.4f}",
        f"  Median : {np.median(distances):.4f}",
        f"  Std    : {distances.std():.4f}",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  Saved: summary_report.txt")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 65)
    print("SOC PIPELINE — STEP 1 (REVISED)")
    print("Outlier Removal + Grouped Sample Split")
    print("=" * 65)
    # Step 1 — load and merge
    merged_original = load_and_merge()

    # Sensitivity analysis — explicitly remove selected images
    merged, explicitly_excluded = remove_predefined_images(
        merged_original
    )

    # Step 2 — Mahalanobis outlier removal
    # performed AFTER predefined exclusions
    df_clean, df_outliers, distances, threshold, feature_cols = \
        mahalanobis_outlier_removal(merged)

    # Step 3 — build image → sample mapping from Excel (NEW)
    mapping = build_image_sample_mapping()

    # Step 4 — grouped sample split (replaces Kennard-Stone)
    df_calib, df_valid = grouped_sample_split(df_clean, mapping)

    # Step 5 — summary report (updated)
    write_summary(merged, df_clean, df_outliers, df_calib, df_valid,
                  distances, threshold, len(feature_cols))

    print("\n" + "=" * 65)
    print("SENSITIVITY STEP 1 COMPLETE")
    print("=" * 65)

    print(
        f"Original merged images : "
        f"{len(merged_original)}"
    )

    print(
        f"Explicit exclusions    : "
        f"{len(explicitly_excluded)}"
    )

    print(
        f"Remaining before MD    : "
        f"{len(merged)}"
    )

    print(
        f"Mahalanobis outliers   : "
        f"{len(df_outliers)}"
    )

    print(
        f"Calibration images     : "
        f"{len(df_calib)}"
    )

    print(
        f"Validation images      : "
        f"{len(df_valid)}"
    )

    print(
        f"\nOutputs saved to:\n"
        f"{OUT_DIR}"
    )


if __name__ == '__main__':
    main()