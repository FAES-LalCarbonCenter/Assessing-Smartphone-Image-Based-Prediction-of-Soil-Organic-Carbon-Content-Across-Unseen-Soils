


# """
# STEP 2 — DEFENSIBLE FEATURE SELECTION
# =====================================

# Feature selection performed on CALIBRATION DATA ONLY.

# Step 0:
#     Aggregate technical image replicates by Sample_No + SOC.

# Stage 1 — Quality control
#     1. Missingness > 20% -> remove
#     2. Min-max scaled variance < 0.01 -> remove

# Stage 2 — SOC relevance
#     A feature survives if EITHER:

#     A) |Spearman rho| >= 0.20 AND screening p < 0.10

#        OR

#     B) Observed Mutual Information exceeds the 95th percentile
#        of its permutation-based null distribution.

# Stage 3 — PCA dimensionality reduction
#     PCA is fit to standardized Stage-2 survivors.
#     PC1 + PC2 are used if cumulative explained variance >= 60%.

#     Feature importance threshold:
#         contribution > 1/p on PC1 OR PC2

#     where p = number of predictors entering PCA.

# Final predictors:
#     Passed Stage 2
#     AND have above-average contribution to PC1 or PC2.

# Validation data are NEVER used for feature selection.
# """

# import os
# import warnings
# import numpy as np
# import pandas as pd

# from scipy.stats import spearmanr

# from sklearn.preprocessing import (
#     StandardScaler,
#     MinMaxScaler,
# )
# from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.decomposition import PCA

# warnings.filterwarnings("ignore")


# # ================================================================
# # PATHS
# # ================================================================

# PROJECT_ROOT = (
#     r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
#     r"\VSCode_Image_Processing_Reviewed"
# )

# CALIB_CSV = os.path.join(
#     PROJECT_ROOT,
#     "notebooks",
#     "objective_1",
#     "output_data",
#     "step1_output_sensitivity",
#     "calibration_set.csv",
# )

# VALID_CSV = os.path.join(
#     PROJECT_ROOT,
#     "notebooks",
#     "objective_1",
#     "output_data",
#     "step1_output_sensitivity",
#     "validation_set.csv",
# )

# OUT_DIR = os.path.join(
#     PROJECT_ROOT,
#     "notebooks",
#     "objective_1",
#     "output_data",
#     "step2_output_sensitivity",
# )

# os.makedirs(OUT_DIR, exist_ok=True)


# # ================================================================
# # SETTINGS
# # ================================================================

# TARGET = "soc"
# GROUP = "Sample_No"

# EXCLUDE = [
#     "filename",
#     "Numeric numbers",
#     "image_path",
#     "image_no",
#     "Image_No",
#     "soil_type",
#     "moisture",
#     "soc",
#     "Sample_No",
#     "Mahalanobis_Distance",
#     "Is_Outlier",
#     "split",
# ]

# # ------------------------------------------------
# # Stage 1
# # ------------------------------------------------

# MAX_MISSING = 0.20
# MIN_SCALED_VARIANCE = 0.01

# # ------------------------------------------------
# # Stage 2
# # ------------------------------------------------

# MIN_ABS_SPEARMAN = 0.20
# SCREENING_P = 0.10

# N_MI_PERMUTATIONS = 500
# MI_NULL_PERCENTILE = 95

# # ------------------------------------------------
# # Stage 3
# # ------------------------------------------------

# MIN_PC12_CUMULATIVE_VARIANCE = 0.60

# SEED = 42


# # ================================================================
# # STEP 0 — AGGREGATE TECHNICAL REPLICATES
# # ================================================================

# def aggregate_replicates(df, feature_cols):

#     print("\n" + "=" * 75)
#     print("STEP 0 — TECHNICAL REPLICATE AGGREGATION")
#     print("=" * 75)

#     print(f"Original image rows : {len(df)}")

#     # Median across repeated photographs belonging to same
#     # Sample_No + SOC condition.
#     agg_features = (
#         df.groupby([GROUP, TARGET], as_index=False)[feature_cols]
#         .median()
#     )

#     # Keep simple metadata if available
#     metadata = []

#     for col in ["soil_type", "moisture"]:
#         if col in df.columns:

#             temp = (
#                 df.groupby([GROUP, TARGET], as_index=False)[col]
#                 .first()
#             )

#             metadata.append(temp)

#     result = agg_features.copy()

#     for meta in metadata:
#         result = result.merge(
#             meta,
#             on=[GROUP, TARGET],
#             how="left",
#         )

#     print(
#         f"Independent soil-condition rows after aggregation: "
#         f"{len(result)}"
#     )

#     return result


# # ================================================================
# # STAGE 1 — QUALITY CONTROL
# # ================================================================

# def stage1_quality_filter(df, features):

#     print("\n" + "=" * 75)
#     print("STAGE 1 — QUALITY CONTROL")
#     print("=" * 75)

#     retained = []
#     records = []

#     for feat in features:

#         missing = df[feat].isna().mean()

#         # Missingness threshold
#         if missing > MAX_MISSING:

#             records.append({
#                 "feature": feat,
#                 "missing_fraction": missing,
#                 "scaled_variance": np.nan,
#                 "stage1_pass": False,
#                 "reason": "missingness > 20%",
#             })

#             continue

#         x = df[[feat]].copy()

#         x = x.fillna(x.median())

#         # Min-max scaling makes variance threshold comparable
#         scaler = MinMaxScaler()
#         x_scaled = scaler.fit_transform(x)

#         variance = np.var(
#             x_scaled,
#             ddof=1,
#         )

#         if variance < MIN_SCALED_VARIANCE:

#             records.append({
#                 "feature": feat,
#                 "missing_fraction": missing,
#                 "scaled_variance": variance,
#                 "stage1_pass": False,
#                 "reason": "scaled variance < 0.01",
#             })

#             continue

#         retained.append(feat)

#         records.append({
#             "feature": feat,
#             "missing_fraction": missing,
#             "scaled_variance": variance,
#             "stage1_pass": True,
#             "reason": "retained",
#         })

#     report = pd.DataFrame(records)

#     print(f"Input features : {len(features)}")
#     print(f"Retained       : {len(retained)}")
#     print(f"Removed        : {len(features) - len(retained)}")

#     report.to_csv(
#         os.path.join(
#             OUT_DIR,
#             "stage1_quality_report.csv",
#         ),
#         index=False,
#     )

#     return retained, report


# # ================================================================
# # STAGE 2A — SPEARMAN CORRELATION
# # ================================================================

# def compute_spearman(df, features):

#     results = []

#     for feat in features:

#         x = df[feat]
#         y = df[TARGET]

#         valid = (
#             x.notna()
#             & y.notna()
#         )

#         rho, p = spearmanr(
#             x[valid],
#             y[valid],
#         )

#         results.append({
#             "feature": feat,
#             "spearman_rho": rho,
#             "abs_spearman": abs(rho),
#             "spearman_p": p,
#         })

#     return pd.DataFrame(results)


# # ================================================================
# # STAGE 2B — MUTUAL INFORMATION WITH PERMUTATION THRESHOLD
# # ================================================================

# def compute_permutation_mi(df, features):

#     rng = np.random.default_rng(SEED)

#     imputer = SimpleImputer(
#         strategy="median"
#     )

#     X = imputer.fit_transform(
#         df[features]
#     )

#     y = df[TARGET].values

#     # observed MI
#     observed = mutual_info_regression(
#         X,
#         y,
#         random_state=SEED,
#         n_neighbors=min(
#             5,
#             max(2, len(y) - 1),
#         ),
#     )

#     results = []

#     print(
#         f"\nRunning {N_MI_PERMUTATIONS} "
#         f"MI permutations..."
#     )

#     for j, feat in enumerate(features):

#         null_scores = []

#         feature_x = X[:, [j]]

#         for i in range(N_MI_PERMUTATIONS):

#             y_perm = rng.permutation(y)

#             mi_perm = mutual_info_regression(
#                 feature_x,
#                 y_perm,
#                 random_state=SEED + i,
#                 n_neighbors=min(
#                     5,
#                     max(2, len(y) - 1),
#                 ),
#             )[0]

#             null_scores.append(mi_perm)

#         threshold = np.percentile(
#             null_scores,
#             MI_NULL_PERCENTILE,
#         )

#         results.append({
#             "feature": feat,
#             "mi_observed": observed[j],
#             "mi_null_95": threshold,
#             "mi_pass": observed[j] > threshold,
#         })

#     return pd.DataFrame(results)


# # ================================================================
# # STAGE 2 — COMBINE SOC ASSOCIATION TESTS
# # ================================================================

# def stage2_soc_screening(df, features):

#     print("\n" + "=" * 75)
#     print("STAGE 2 — SOC RELEVANCE SCREENING")
#     print("=" * 75)

#     corr = compute_spearman(
#         df,
#         features,
#     )

#     mi = compute_permutation_mi(
#         df,
#         features,
#     )

#     result = corr.merge(
#         mi,
#         on="feature",
#     )

#     # Correlation criterion
#     result["corr_pass"] = (
#         (result["abs_spearman"] >= MIN_ABS_SPEARMAN)
#         &
#         (result["spearman_p"] < SCREENING_P)
#     )

#     # Final Stage-2 rule:
#     # pass correlation criterion OR MI criterion

#     result["stage2_pass"] = (
#         result["corr_pass"]
#         |
#         result["mi_pass"]
#     )

#     result = result.sort_values(
#         [
#             "stage2_pass",
#             "abs_spearman",
#             "mi_observed",
#         ],
#         ascending=[
#             False,
#             False,
#             False,
#         ],
#     )

#     retained = result.loc[
#         result["stage2_pass"],
#         "feature",
#     ].tolist()

#     print(
#         f"\nEntered Stage 2 : {len(features)}"
#     )

#     print(
#         f"Passed Stage 2  : {len(retained)}"
#     )

#     print("\nStage-2 results:")
#     print(
#         result[
#             [
#                 "feature",
#                 "spearman_rho",
#                 "spearman_p",
#                 "corr_pass",
#                 "mi_observed",
#                 "mi_null_95",
#                 "mi_pass",
#                 "stage2_pass",
#             ]
#         ].to_string(index=False)
#     )

#     result.to_csv(
#         os.path.join(
#             OUT_DIR,
#             "stage2_soc_screening.csv",
#         ),
#         index=False,
#     )

#     return retained, result


# # ================================================================
# # STAGE 3 — PCA
# # ================================================================

# def stage3_pca(df, features):

#     print("\n" + "=" * 75)
#     print("STAGE 3 — PCA DIMENSIONALITY REDUCTION")
#     print("=" * 75)

#     if len(features) < 2:
#         raise ValueError(
#             "Fewer than two features entered PCA."
#         )

#     imputer = SimpleImputer(
#         strategy="median"
#     )

#     scaler = StandardScaler()

#     X = imputer.fit_transform(
#         df[features]
#     )

#     X = scaler.fit_transform(X)

#     pca = PCA()
#     pca.fit(X)

#     explained = (
#         pca.explained_variance_ratio_
#     )

#     pc1_var = explained[0]
#     pc2_var = explained[1]

#     cumulative = (
#         pc1_var + pc2_var
#     )

#     print(
#         f"PC1 variance explained : "
#         f"{pc1_var * 100:.2f}%"
#     )

#     print(
#         f"PC2 variance explained : "
#         f"{pc2_var * 100:.2f}%"
#     )

#     print(
#         f"PC1 + PC2 cumulative   : "
#         f"{cumulative * 100:.2f}%"
#     )

#     if (
#         cumulative
#         < MIN_PC12_CUMULATIVE_VARIANCE
#     ):

#         print(
#             "\nWARNING:"
#             "\nPC1 + PC2 explain less than "
#             f"{MIN_PC12_CUMULATIVE_VARIANCE*100:.0f}%."
#             "\nConsider including PC3."
#         )

#     # PCA components:
#     # rows = PCs
#     # columns = original variables

#     loading_pc1 = pca.components_[0]
#     loading_pc2 = pca.components_[1]

#     # Squared loadings within each PC
#     sq_pc1 = loading_pc1 ** 2
#     sq_pc2 = loading_pc2 ** 2

#     # Convert to proportional contribution
#     contribution_pc1 = (
#         sq_pc1 / sq_pc1.sum()
#     )

#     contribution_pc2 = (
#         sq_pc2 / sq_pc2.sum()
#     )

#     # Expected contribution if all features
#     # contributed equally
#     expected = 1 / len(features)

#     result = pd.DataFrame({
#         "feature": features,
#         "loading_PC1": loading_pc1,
#         "loading_PC2": loading_pc2,
#         "contribution_PC1": contribution_pc1,
#         "contribution_PC2": contribution_pc2,
#     })

#     result["expected_contribution"] = (
#         expected
#     )

#     result["PC1_influential"] = (
#         result["contribution_PC1"]
#         > expected
#     )

#     result["PC2_influential"] = (
#         result["contribution_PC2"]
#         > expected
#     )

#     result["PCA_pass"] = (
#         result["PC1_influential"]
#         |
#         result["PC2_influential"]
#     )

#     # Helpful combined measure for ranking,
#     # but NOT used as an additional threshold.

#     result[
#         "variance_weighted_PC12_score"
#     ] = (
#         pc1_var
#         * result["contribution_PC1"]
#         +
#         pc2_var
#         * result["contribution_PC2"]
#     )

#     result = result.sort_values(
#         "variance_weighted_PC12_score",
#         ascending=False,
#     )

#     print(
#         f"\nExpected average contribution "
#         f"= 1/{len(features)} "
#         f"= {expected*100:.2f}%"
#     )

#     print("\nPCA contribution table:")

#     display_cols = [
#         "feature",
#         "loading_PC1",
#         "loading_PC2",
#         "contribution_PC1",
#         "contribution_PC2",
#         "PC1_influential",
#         "PC2_influential",
#         "PCA_pass",
#     ]

#     print(
#         result[
#             display_cols
#         ].to_string(
#             index=False,
#             float_format=lambda x: f"{x:.4f}",
#         )
#     )

#     result.to_csv(
#         os.path.join(
#             OUT_DIR,
#             "stage3_pca_feature_contributions.csv",
#         ),
#         index=False,
#     )

#     final_features = result.loc[
#         result["PCA_pass"],
#         "feature",
#     ].tolist()

#     return (
#         final_features,
#         result,
#         pca,
#     )


# # ================================================================
# # STAGE 4 — PAIRWISE SPEARMAN REDUNDANCY PRUNING
# # ================================================================

# PAIRWISE_SPEARMAN_THRESHOLD = 0.90


# def stage4_pairwise_spearman_pruning(
#     df,
#     features,
#     stage2_report
# ):

#     print("\n" + "=" * 75)
#     print("STAGE 4 — PAIRWISE SPEARMAN REDUNDANCY PRUNING")
#     print("=" * 75)

#     print(
#         f"Redundancy threshold: "
#         f"|Spearman rho| >= {PAIRWISE_SPEARMAN_THRESHOLD:.2f}"
#     )

#     # ------------------------------------------------------------
#     # Predictor-to-predictor correlation matrix
#     # ------------------------------------------------------------

#     corr_matrix = (
#         df[features]
#         .corr(method="spearman")
#         .abs()
#     )

#     # ------------------------------------------------------------
#     # Bring in predictor-to-SOC relevance from Stage 2
#     # ------------------------------------------------------------

#     relevance = (
#         stage2_report
#         .set_index("feature")
#         [["abs_spearman", "mi_observed"]]
#         .copy()
#     )

#     # ------------------------------------------------------------
#     # Order features:
#     # strongest SOC association first,
#     # then MI as tie-breaker
#     # ------------------------------------------------------------

#     ordered_features = sorted(
#         features,
#         key=lambda f: (
#             relevance.loc[f, "abs_spearman"],
#             relevance.loc[f, "mi_observed"]
#         ),
#         reverse=True
#     )

#     retained = []
#     removed = []

#     # ------------------------------------------------------------
#     # Greedy redundancy pruning
#     # ------------------------------------------------------------

#     for feat in ordered_features:

#         redundant_with = None
#         redundancy_value = None

#         for kept in retained:

#             rho = corr_matrix.loc[feat, kept]

#             if rho >= PAIRWISE_SPEARMAN_THRESHOLD:

#                 redundant_with = kept
#                 redundancy_value = rho
#                 break

#         if redundant_with is None:

#             retained.append(feat)

#         else:

#             removed.append({
#                 "removed_feature": feat,
#                 "retained_feature": redundant_with,
#                 "pairwise_abs_spearman": redundancy_value,

#                 "removed_abs_SOC_spearman":
#                     relevance.loc[
#                         feat,
#                         "abs_spearman"
#                     ],

#                 "retained_abs_SOC_spearman":
#                     relevance.loc[
#                         redundant_with,
#                         "abs_spearman"
#                     ],

#                 "removed_MI":
#                     relevance.loc[
#                         feat,
#                         "mi_observed"
#                     ],

#                 "retained_MI":
#                     relevance.loc[
#                         redundant_with,
#                         "mi_observed"
#                     ],
#             })

#     # ------------------------------------------------------------
#     # Report
#     # ------------------------------------------------------------

#     print(f"\nInput PCA-selected features : {len(features)}")
#     print(f"Retained after pruning      : {len(retained)}")
#     print(f"Removed as redundant        : {len(features) - len(retained)}")

#     if removed:

#         removed_df = pd.DataFrame(removed)

#         print("\nRedundant predictors removed:")
#         print(
#             removed_df.to_string(
#                 index=False,
#                 float_format=lambda x: f"{x:.4f}"
#             )
#         )

#         removed_df.to_csv(
#             os.path.join(
#                 OUT_DIR,
#                 "stage4_pairwise_spearman_removed.csv"
#             ),
#             index=False
#         )

#     # Save correlation matrix too
#     corr_matrix.to_csv(
#         os.path.join(
#             OUT_DIR,
#             "stage4_pairwise_spearman_matrix.csv"
#         )
#     )

#     print("\nFinal retained features:")
#     for i, feat in enumerate(retained, 1):
#         print(f"{i:2d}. {feat}")

#     return retained, corr_matrix

# # ================================================================
# # MAIN
# # ================================================================

# def main():

#     print("\n" + "=" * 75)
#     print("DEFENSIBLE THREE-STAGE FEATURE SELECTION")
#     print("=" * 75)

#     df_cal = pd.read_csv(
#         CALIB_CSV
#     )

#     df_val = pd.read_csv(
#         VALID_CSV
#     )

#     # Original numerical image predictors
#     candidate_features = [
#         c
#         for c in df_cal.columns
#         if (
#             c not in EXCLUDE
#             and pd.api.types.is_numeric_dtype(
#                 df_cal[c]
#             )
#         )
#     ]

#     print(
#         f"\nOriginal candidate features: "
#         f"{len(candidate_features)}"
#     )

#     # ------------------------------------------------------------
#     # Aggregate calibration replicates
#     # ------------------------------------------------------------

#     df_cal_agg = aggregate_replicates(
#         df_cal,
#         candidate_features,
#     )

#     # ------------------------------------------------------------
#     # Stage 1
#     # ------------------------------------------------------------

#     stage1_features, stage1_report = (
#         stage1_quality_filter(
#             df_cal_agg,
#             candidate_features,
#         )
#     )

#     # ------------------------------------------------------------
#     # Stage 2
#     # ------------------------------------------------------------

#     stage2_features, stage2_report = (
#         stage2_soc_screening(
#             df_cal_agg,
#             stage1_features,
#         )
#     )

#     # ------------------------------------------------------------
#     # Stage 3
#     # ------------------------------------------------------------

#     pca_features, pca_report, pca_model = (
#         stage3_pca(
#             df_cal_agg,
#             stage2_features,
#         )
#     )



# # ------------------------------------------------------------
# # Stage 4 — pairwise Spearman redundancy pruning
# # ------------------------------------------------------------

#     final_features, pairwise_corr_matrix = (
#         stage4_pairwise_spearman_pruning(
#             df_cal_agg,
#            pca_features,
#             stage2_report
#         )
#     )

#     print("\n" + "=" * 75)
#     print("FINAL SELECTED FEATURES")
#     print("=" * 75)

#     for i, feat in enumerate(
#         final_features,
#         1,
#     ):
#         print(
#             f"{i:2d}. {feat}"
#         )

#     print(
#         f"\nOriginal features : "
#         f"{len(candidate_features)}"
#     )

#     print(
#         f"After Stage 1     : "
#         f"{len(stage1_features)}"
#     )

#     print(
#         f"After Stage 2     : "
#         f"{len(stage2_features)}"
#     )

#     print(
#         f"After PCA   : "
#         f"{len(pca_features)}"
#     )


#     print(
#         f"After Spearman pruning  : {len(final_features)}"
#     )
#     # ------------------------------------------------------------
#     # SAVE FEATURE LIST
#     # ------------------------------------------------------------

#     with open(
#         os.path.join(
#             OUT_DIR,
#             "selected_features_list.txt",
#         ),
#         "w",
#     ) as f:

#         for feat in final_features:
#             f.write(
#                 feat + "\n"
#             )

#     # ------------------------------------------------------------
#     # Save aggregated calibration modeling dataset
#     # ------------------------------------------------------------

#     cal_cols = (
#         [GROUP, TARGET]
#         + final_features
#     )

#     df_cal_agg[
#         cal_cols
#     ].to_csv(
#         os.path.join(
#             OUT_DIR,
#             "calibration_selected_features.csv",
#         ),
#         index=False,
#     )

#     # ------------------------------------------------------------
#     # Validation:
#     # aggregate independently,
#     # but DO NOT perform feature selection on it.
#     # ------------------------------------------------------------

#     df_val_agg = aggregate_replicates(
#         df_val,
#         candidate_features,
#     )

#     val_cols = (
#         [GROUP, TARGET]
#         + final_features
#     )

#     df_val_agg[
#         val_cols
#     ].to_csv(
#         os.path.join(
#             OUT_DIR,
#             "validation_selected_features.csv",
#         ),
#         index=False,
#     )

#     print(
#         "\nValidation feature selection: NONE."
#         "\nCalibration-derived feature list "
#         "applied unchanged."
#     )

#     print(
#         f"\nOutputs saved to:\n{OUT_DIR}"
#     )


# if __name__ == "__main__":
#     main()


"""
STEP 2 — DEFENSIBLE FEATURE SELECTION (SENSITIVITY: Sample 7 dropped)
=======================================================================
FIX: Removed aggregate_replicates() — feature selection now runs on
     ALL calibration images (~460 rows) not 15-16 sample means.

All stages unchanged:
  Stage 1: Quality control (missingness, scaled variance)
  Stage 2: Spearman + permutation MI
  Stage 3: PCA contribution
  Stage 4: Pairwise Spearman redundancy pruning
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

CALIB_CSV = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step1_output_sensitivity", "calibration_set.csv")
VALID_CSV  = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step1_output_sensitivity", "validation_set.csv")
OUT_DIR    = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step2_output_sensitivity")

os.makedirs(OUT_DIR, exist_ok=True)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
TARGET = "soc"
GROUP  = "Sample_No"

EXCLUDE = ["filename","Numeric numbers","image_path","image_no","Image_No",
           "soil_type","moisture","soc","Sample_No","Mahalanobis_Distance",
           "Is_Outlier","split"]

MAX_MISSING          = 0.20
MIN_SCALED_VARIANCE  = 0.01
MIN_ABS_SPEARMAN     = 0.20
SCREENING_P          = 0.10
N_MI_PERMUTATIONS    = 500
MI_NULL_PERCENTILE   = 95
MIN_PC12_CUMVAR      = 0.60
PAIRWISE_THRESHOLD   = 0.90
SEED = 42


# ── STAGE 1 ───────────────────────────────────────────────────────────────────
def stage1_quality_filter(df, features):
    print("\n" + "="*70)
    print("STAGE 1 — QUALITY CONTROL  (image-level calibration data)")
    print("="*70)
    retained, records = [], []
    for feat in features:
        missing = df[feat].isna().mean()
        if missing > MAX_MISSING:
            records.append({"feature":feat,"stage1_pass":False,
                             "reason":"missingness > 20%"})
            continue
        x = df[[feat]].fillna(df[feat].median())
        variance = np.var(MinMaxScaler().fit_transform(x), ddof=1)
        if variance < MIN_SCALED_VARIANCE:
            records.append({"feature":feat,"scaled_variance":variance,
                             "stage1_pass":False,"reason":"scaled variance < 0.01"})
            continue
        retained.append(feat)
        records.append({"feature":feat,"scaled_variance":variance,
                         "stage1_pass":True,"reason":"retained"})
    pd.DataFrame(records).to_csv(
        os.path.join(OUT_DIR,"stage1_quality_report.csv"), index=False)
    print(f"  Input: {len(features)}  Retained: {len(retained)}  "
          f"Removed: {len(features)-len(retained)}")
    return retained, pd.DataFrame(records)


# ── STAGE 2 ───────────────────────────────────────────────────────────────────
def compute_spearman(df, features):
    results = []
    for feat in features:
        x, y = df[feat], df[TARGET]
        valid = x.notna() & y.notna()
        rho, p = spearmanr(x[valid], y[valid])
        results.append({"feature":feat,"spearman_rho":rho,
                         "abs_spearman":abs(rho),"spearman_p":p})
    return pd.DataFrame(results)

def compute_permutation_mi(df, features):
    rng = np.random.default_rng(SEED)
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df[features])
    y   = df[TARGET].values
    n_neighbors = min(5, max(2, len(y)-1))
    observed = mutual_info_regression(X, y, random_state=SEED,
                                       n_neighbors=n_neighbors)
    results = []
    print(f"\n  Running {N_MI_PERMUTATIONS} MI permutations on "
          f"{len(y)} images...")
    for j, feat in enumerate(features):
        null_scores = [
            mutual_info_regression(X[:,[j]], rng.permutation(y),
                                    random_state=SEED+i,
                                    n_neighbors=n_neighbors)[0]
            for i in range(N_MI_PERMUTATIONS)
        ]
        threshold = np.percentile(null_scores, MI_NULL_PERCENTILE)
        results.append({"feature":feat,"mi_observed":observed[j],
                         "mi_null_95":threshold,
                         "mi_pass":observed[j]>threshold})
    return pd.DataFrame(results)

def stage2_soc_screening(df, features):
    print("\n" + "="*70)
    print("STAGE 2 — SOC RELEVANCE SCREENING  (image-level)")
    print("="*70)
    corr = compute_spearman(df, features)
    mi   = compute_permutation_mi(df, features)
    result = corr.merge(mi, on="feature")
    result["corr_pass"]   = ((result["abs_spearman"] >= MIN_ABS_SPEARMAN) &
                              (result["spearman_p"] < SCREENING_P))
    result["stage2_pass"] = result["corr_pass"] | result["mi_pass"]
    result = result.sort_values(["stage2_pass","abs_spearman","mi_observed"],
                                 ascending=[False,False,False])
    retained = result.loc[result["stage2_pass"],"feature"].tolist()
    print(f"  Entered: {len(features)}  Passed: {len(retained)}")
    print(result[["feature","spearman_rho","spearman_p","corr_pass",
                   "mi_observed","mi_null_95","mi_pass",
                   "stage2_pass"]].to_string(index=False))
    result.to_csv(os.path.join(OUT_DIR,"stage2_soc_screening.csv"), index=False)
    return retained, result


# ── STAGE 3 ───────────────────────────────────────────────────────────────────
def stage3_pca(df, features):
    print("\n" + "="*70)
    print("STAGE 3 — PCA DIMENSIONALITY REDUCTION  (image-level)")
    print("="*70)
    if len(features) < 2:
        raise ValueError("Fewer than two features entered PCA.")
    imp = SimpleImputer(strategy="median")
    X   = StandardScaler().fit_transform(imp.fit_transform(df[features]))
    pca = PCA().fit(X)
    ev  = pca.explained_variance_ratio_
    pc1_var, pc2_var = ev[0], ev[1]
    cumvar = pc1_var + pc2_var
    print(f"  PC1={pc1_var*100:.2f}%  PC2={pc2_var*100:.2f}%  "
          f"Cumulative={cumvar*100:.2f}%")
    if cumvar < MIN_PC12_CUMVAR:
        print(f"  WARNING: PC1+PC2 < {MIN_PC12_CUMVAR*100:.0f}% — consider PC3")
    lp1 = pca.components_[0]; lp2 = pca.components_[1]
    c1  = lp1**2/np.sum(lp1**2); c2 = lp2**2/np.sum(lp2**2)
    expected = 1/len(features)
    result = pd.DataFrame({"feature":features,"loading_PC1":lp1,"loading_PC2":lp2,
                            "contribution_PC1":c1,"contribution_PC2":c2,
                            "expected":expected,
                            "PC1_influential":c1>expected,
                            "PC2_influential":c2>expected})
    result["PCA_pass"] = result["PC1_influential"] | result["PC2_influential"]
    result["vw_score"] = pc1_var*c1 + pc2_var*c2
    result = result.sort_values("vw_score", ascending=False)
    result.to_csv(os.path.join(OUT_DIR,"stage3_pca_feature_contributions.csv"),
                   index=False)
    final = result.loc[result["PCA_pass"],"feature"].tolist()
    print(f"  PCA retained: {len(final)} of {len(features)}")
    return final, result, pca


# ── STAGE 4 ───────────────────────────────────────────────────────────────────
def stage4_pairwise_pruning(df, features, stage2_report):
    print("\n" + "="*70)
    print(f"STAGE 4 — PAIRWISE SPEARMAN PRUNING (threshold={PAIRWISE_THRESHOLD})")
    print("="*70)
    corr_matrix = df[features].corr(method="spearman").abs()
    relevance   = stage2_report.set_index("feature")[["abs_spearman","mi_observed"]]
    ordered     = sorted(features,
                          key=lambda f:(relevance.loc[f,"abs_spearman"],
                                         relevance.loc[f,"mi_observed"]),
                          reverse=True)
    retained, removed = [], []
    for feat in ordered:
        dup = next((k for k in retained
                    if corr_matrix.loc[feat,k] >= PAIRWISE_THRESHOLD), None)
        if dup is None:
            retained.append(feat)
        else:
            removed.append({"removed":feat,"kept":dup,
                             "rho":corr_matrix.loc[feat,dup]})
    if removed:
        pd.DataFrame(removed).to_csv(
            os.path.join(OUT_DIR,"stage4_pairwise_spearman_removed.csv"), index=False)
    corr_matrix.to_csv(os.path.join(OUT_DIR,"stage4_pairwise_spearman_matrix.csv"))
    print(f"  Input: {len(features)}  Retained: {len(retained)}  "
          f"Removed: {len(features)-len(retained)}")
    print(f"  Final features: {retained}")
    return retained, corr_matrix


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("STEP 2 — FEATURE SELECTION (SENSITIVITY: Sample 7 dropped)")
    print("  NOTE: running on IMAGE-LEVEL data — no aggregation applied")
    print("="*70)

    df_cal = pd.read_csv(CALIB_CSV)
    df_val = pd.read_csv(VALID_CSV)

    print(f"\n  Calibration: {len(df_cal)} images x {len(df_cal.columns)} cols")
    print(f"  Validation : {len(df_val)} images  (not used for selection)")

    # Candidate features — exclude identifiers
    candidate_features = [c for c in df_cal.columns
                          if c not in EXCLUDE
                          and pd.api.types.is_numeric_dtype(df_cal[c])]
    print(f"  Candidate features: {len(candidate_features)}")

    # KEY FIX: use df_cal directly — no aggregation
    # Stage 1
    stage1_feats, stage1_report = stage1_quality_filter(df_cal, candidate_features)

    # Stage 2
    stage2_feats, stage2_report = stage2_soc_screening(df_cal, stage1_feats)

    # Stage 3
    pca_feats, pca_report, pca_model = stage3_pca(df_cal, stage2_feats)

    # Stage 4
    final_features, corr_matrix = stage4_pairwise_pruning(
        df_cal, pca_feats, stage2_report)

    print("\n" + "="*70)
    print("FINAL SELECTED FEATURES")
    print("="*70)
    for i, f in enumerate(final_features, 1):
        print(f"  {i:2d}. {f}")
    print(f"\n  Original: {len(candidate_features)} → Stage1: {len(stage1_feats)} → "
          f"Stage2: {len(stage2_feats)} → PCA: {len(pca_feats)} → "
          f"Final: {len(final_features)}")

    # Save feature list
    with open(os.path.join(OUT_DIR,"selected_features_list.txt"),"w") as f:
        for feat in final_features: f.write(feat+"\n")

    # Save calibration with selected features — IMAGE LEVEL
    keep = [c for c in [GROUP,"image_no","moisture","soil_type",TARGET]
            if c in df_cal.columns] + final_features
    df_cal[keep].to_csv(
        os.path.join(OUT_DIR,"calibration_selected_features.csv"), index=False)
    print(f"\n  Saved calibration_selected_features.csv  "
          f"({len(df_cal)} rows — IMAGE LEVEL ✅)")

    # Save validation — IMAGE LEVEL, no selection applied
    val_keep = [c for c in keep if c in df_val.columns]
    df_val[val_keep].to_csv(
        os.path.join(OUT_DIR,"validation_selected_features.csv"), index=False)
    print(f"  Saved validation_selected_features.csv  "
          f"({len(df_val)} rows — IMAGE LEVEL ✅)")
    print(f"\n  Outputs: {OUT_DIR}")

if __name__ == "__main__":
    main()