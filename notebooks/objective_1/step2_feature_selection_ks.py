"""
KS PIPELINE — STEP 2
====================

Feature selection for the Kennard-Stone image-level analysis.

IMPORTANT FOR FAIR COMPARISON
-----------------------------
This script uses the SAME four-stage feature-selection logic and thresholds
as the sample-grouped Step 2 analysis. The only intended difference is the
upstream splitting strategy:

    Sample-grouped analysis -> whole physical soils held out
    KS image-level analysis -> images split by Kennard-Stone, so the same
                               physical soils may appear in both subsets

Feature selection is performed ONLY on the KS calibration images after
calibration-only Mahalanobis QC. The KS validation images are never used
for feature selection.

Stages
------
1. Quality control:
      missingness <= 20%
      scaled variance >= 0.01
2. SOC relevance screening:
      (|Spearman rho| >= 0.20 AND p < 0.10)
      OR
      observed mutual information > 95th percentile of 500-permutation null
3. PCA contribution screening:
      retain if contribution to PC1 OR PC2 > expected contribution (1/p)
      PC1+PC2 >= 60% is a diagnostic check only
4. Pairwise Spearman redundancy pruning:
      |rho| >= 0.90
      preferentially retain stronger SOC-associated feature
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


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

CALIB_CSV = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step1_output_ks",
    "calibration_set.csv",
)

VALID_CSV = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step1_output_ks",
    "validation_set.csv",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step2_output_ks",
)

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# SETTINGS — IDENTICAL TO SAMPLE-GROUPED STEP 2
# =============================================================================

TARGET = "soc"
GROUP = "Sample_No"

EXCLUDE = [
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

MAX_MISSING = 0.20
MIN_SCALED_VARIANCE = 0.01

MIN_ABS_SPEARMAN = 0.20
SCREENING_P = 0.10

N_MI_PERMUTATIONS = 500
MI_NULL_PERCENTILE = 95

MIN_PC12_CUMVAR = 0.60       # diagnostic warning only

PAIRWISE_THRESHOLD = 0.90
SEED = 42


# =============================================================================
# STAGE 1 — QUALITY CONTROL
# =============================================================================

def stage1_quality_filter(df, features):

    print("\n" + "=" * 75)
    print("STAGE 1 — QUALITY CONTROL (KS calibration images only)")
    print(f"  Input rows: {len(df)}")
    print("=" * 75)

    retained = []
    records = []

    for feat in features:

        missing = df[feat].isna().mean()

        if missing > MAX_MISSING:
            records.append({
                "feature": feat,
                "scaled_variance": np.nan,
                "stage1_pass": False,
                "reason": "missingness > 20%",
            })
            continue

        x = df[[feat]].fillna(df[feat].median())

        scaled_x = MinMaxScaler().fit_transform(x)

        variance = np.var(
            scaled_x,
            ddof=1,
        )

        if variance < MIN_SCALED_VARIANCE:
            records.append({
                "feature": feat,
                "scaled_variance": variance,
                "stage1_pass": False,
                "reason": "scaled variance < 0.01",
            })
            continue

        retained.append(feat)

        records.append({
            "feature": feat,
            "scaled_variance": variance,
            "stage1_pass": True,
            "reason": "retained",
        })

    report = pd.DataFrame(records)

    print(
        f"  Input: {len(features)}  "
        f"Retained: {len(retained)}  "
        f"Removed: {len(features) - len(retained)}"
    )

    report.to_csv(
        os.path.join(
            OUT_DIR,
            "stage1_quality_report.csv",
        ),
        index=False,
    )

    return retained, report


# =============================================================================
# STAGE 2A — SPEARMAN ASSOCIATION WITH SOC
# =============================================================================

def compute_spearman(df, features):

    results = []

    for feat in features:

        x = df[feat]
        y = df[TARGET]

        valid = x.notna() & y.notna()

        rho, p = spearmanr(
            x[valid],
            y[valid],
        )

        results.append({
            "feature": feat,
            "spearman_rho": rho,
            "abs_spearman": abs(rho),
            "spearman_p": p,
        })

    return pd.DataFrame(results)


# =============================================================================
# STAGE 2B — MUTUAL INFORMATION WITH PERMUTATION NULL
# =============================================================================

def compute_permutation_mi(df, features):

    rng = np.random.default_rng(SEED)

    imp = SimpleImputer(strategy="median")

    X = imp.fit_transform(
        df[features]
    )

    y = df[TARGET].values

    n_nb = min(
        5,
        max(2, len(y) - 1),
    )

    observed = mutual_info_regression(
        X,
        y,
        random_state=SEED,
        n_neighbors=n_nb,
    )

    results = []

    print(
        f"\n  Running {N_MI_PERMUTATIONS} MI permutations "
        f"on {len(y)} KS calibration images..."
    )

    for j, feat in enumerate(features):

        null_scores = [
            mutual_info_regression(
                X[:, [j]],
                rng.permutation(y),
                random_state=SEED + i,
                n_neighbors=n_nb,
            )[0]
            for i in range(N_MI_PERMUTATIONS)
        ]

        threshold = np.percentile(
            null_scores,
            MI_NULL_PERCENTILE,
        )

        results.append({
            "feature": feat,
            "mi_observed": observed[j],
            "mi_null_95": threshold,
            "mi_pass": observed[j] > threshold,
        })

    return pd.DataFrame(results)


# =============================================================================
# STAGE 2 — COMBINED SOC RELEVANCE SCREEN
# =============================================================================

def stage2_soc_screening(df, features):

    print("\n" + "=" * 75)
    print("STAGE 2 — SOC RELEVANCE SCREENING (KS calibration only)")
    print("=" * 75)

    corr = compute_spearman(
        df,
        features,
    )

    mi = compute_permutation_mi(
        df,
        features,
    )

    result = corr.merge(
        mi,
        on="feature",
    )

    result["corr_pass"] = (
        (result["abs_spearman"] >= MIN_ABS_SPEARMAN)
        &
        (result["spearman_p"] < SCREENING_P)
    )

    result["stage2_pass"] = (
        result["corr_pass"]
        |
        result["mi_pass"]
    )

    result = result.sort_values(
        [
            "stage2_pass",
            "abs_spearman",
            "mi_observed",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    retained = result.loc[
        result["stage2_pass"],
        "feature",
    ].tolist()

    print(
        f"  Entered: {len(features)}  "
        f"Passed: {len(retained)}"
    )

    print(
        result[
            [
                "feature",
                "spearman_rho",
                "spearman_p",
                "corr_pass",
                "mi_observed",
                "mi_null_95",
                "mi_pass",
                "stage2_pass",
            ]
        ].to_string(index=False)
    )

    result.to_csv(
        os.path.join(
            OUT_DIR,
            "stage2_soc_screening.csv",
        ),
        index=False,
    )

    return retained, result


# =============================================================================
# STAGE 3 — PCA CONTRIBUTION SCREEN
# =============================================================================

def stage3_pca(df, features):

    print("\n" + "=" * 75)
    print("STAGE 3 — PCA DIMENSIONALITY REDUCTION (KS calibration only)")
    print("=" * 75)

    if len(features) < 2:
        raise ValueError(
            "Fewer than two features entered PCA."
        )

    imp = SimpleImputer(
        strategy="median"
    )

    X = StandardScaler().fit_transform(
        imp.fit_transform(
            df[features]
        )
    )

    pca = PCA().fit(X)

    ev = pca.explained_variance_ratio_

    pc1 = ev[0]
    pc2 = ev[1]
    cumvar = pc1 + pc2

    print(
        f"  PC1={pc1 * 100:.2f}%  "
        f"PC2={pc2 * 100:.2f}%  "
        f"Cumulative={cumvar * 100:.2f}%"
    )

    if cumvar < MIN_PC12_CUMVAR:
        print(
            f"  WARNING: PC1+PC2 < "
            f"{MIN_PC12_CUMVAR * 100:.0f}%"
        )
        print(
            "  NOTE: This is diagnostic only; "
            "PC1/PC2 contribution screening still proceeds."
        )

    lp1 = pca.components_[0]
    lp2 = pca.components_[1]

    c1 = lp1 ** 2 / np.sum(lp1 ** 2)
    c2 = lp2 ** 2 / np.sum(lp2 ** 2)

    expected = 1 / len(features)

    result = pd.DataFrame({
        "feature": features,
        "loading_PC1": lp1,
        "loading_PC2": lp2,
        "contribution_PC1": c1,
        "contribution_PC2": c2,
        "expected_contribution": expected,
        "PC1_influential": c1 > expected,
        "PC2_influential": c2 > expected,
    })

    result["PCA_pass"] = (
        result["PC1_influential"]
        |
        result["PC2_influential"]
    )

    # Descriptive ranking only.
    result["vw_score"] = (
        pc1 * c1
        +
        pc2 * c2
    )

    result = result.sort_values(
        "vw_score",
        ascending=False,
    )

    result.to_csv(
        os.path.join(
            OUT_DIR,
            "stage3_pca_feature_contributions.csv",
        ),
        index=False,
    )

    retained = result.loc[
        result["PCA_pass"],
        "feature",
    ].tolist()

    print(
        f"  PCA retained: "
        f"{len(retained)} of {len(features)}"
    )

    return retained, result, pca


# =============================================================================
# STAGE 4 — PAIRWISE SPEARMAN REDUNDANCY PRUNING
# =============================================================================

def stage4_pairwise_spearman_pruning(
    df,
    features,
    stage2_report,
):

    print("\n" + "=" * 75)
    print(
        "STAGE 4 — PAIRWISE SPEARMAN PRUNING "
        f"(threshold={PAIRWISE_THRESHOLD})"
    )
    print("=" * 75)

    corr_matrix = (
        df[features]
        .corr(method="spearman")
        .abs()
    )

    relevance = (
        stage2_report
        .set_index("feature")
        [
            [
                "abs_spearman",
                "mi_observed",
            ]
        ]
    )

    # Same ordering rule as sample-grouped analysis:
    # stronger absolute SOC Spearman first,
    # mutual information as tie-breaker.
    ordered = sorted(
        features,
        key=lambda f: (
            relevance.loc[f, "abs_spearman"],
            relevance.loc[f, "mi_observed"],
        ),
        reverse=True,
    )

    retained = []
    removed = []

    for feat in ordered:

        duplicate_of = next(
            (
                kept
                for kept in retained
                if corr_matrix.loc[
                    feat,
                    kept,
                ] >= PAIRWISE_THRESHOLD
            ),
            None,
        )

        if duplicate_of is None:
            retained.append(feat)

        else:
            removed.append({
                "removed_feature": feat,
                "retained_feature": duplicate_of,
                "pairwise_abs_spearman":
                    corr_matrix.loc[
                        feat,
                        duplicate_of,
                    ],
            })

    if removed:
        pd.DataFrame(
            removed
        ).to_csv(
            os.path.join(
                OUT_DIR,
                "stage4_pairwise_spearman_removed.csv",
            ),
            index=False,
        )

    corr_matrix.to_csv(
        os.path.join(
            OUT_DIR,
            "stage4_pairwise_spearman_matrix.csv",
        )
    )

    print(
        f"  Input: {len(features)}  "
        f"Retained: {len(retained)}  "
        f"Removed: {len(features) - len(retained)}"
    )

    print("\n  Final retained features:")

    for i, feat in enumerate(
        retained,
        start=1,
    ):
        print(
            f"  {i:2d}. {feat}"
        )

    return retained, corr_matrix


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n" + "=" * 75)
    print("KS STEP 2 — DEFENSIBLE FEATURE SELECTION")
    print("  IMAGE-LEVEL KENNARD-STONE ANALYSIS")
    print("=" * 75)

    df_cal = pd.read_csv(
        CALIB_CSV
    )

    df_val = pd.read_csv(
        VALID_CSV
    )

    print(
        f"\n  Calibration: "
        f"{len(df_cal)} images x "
        f"{len(df_cal.columns)} columns"
    )

    print(
        f"  Validation : "
        f"{len(df_val)} images "
        f"(NOT used for feature selection)"
    )

    # -------------------------------------------------------------------------
    # Check the intended KS sample dependence.
    # This should NOT raise an error: overlap is expected in image-level KS.
    # -------------------------------------------------------------------------

    if (
        GROUP in df_cal.columns
        and GROUP in df_val.columns
    ):

        cal_samps = set(
            df_cal[GROUP]
            .dropna()
            .unique()
        )

        val_samps = set(
            df_val[GROUP]
            .dropna()
            .unique()
        )

        overlap = (
            cal_samps
            &
            val_samps
        )

        print(
            f"\n  Calibration physical soils: "
            f"{len(cal_samps)}"
        )

        print(
            f"  Validation physical soils : "
            f"{len(val_samps)}"
        )

        print(
            f"  Physical soils on BOTH sides: "
            f"{len(overlap)}"
        )

        print(
            f"  Overlap samples: "
            f"{sorted(overlap)}"
        )

        if overlap:
            print(
                "  EXPECTED FOR KS IMAGE-LEVEL SPLIT: "
                "sample dependence is present by design."
            )
        else:
            print(
                "  WARNING: No sample overlap detected. "
                "This is unusual for the intended KS image-level comparison."
            )

    # Candidate numerical image predictors only.
    candidate_features = [
        c
        for c in df_cal.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(
            df_cal[c]
        )
    ]

    print(
        f"\n  Candidate features: "
        f"{len(candidate_features)}"
    )

    print(
        f"  Features: "
        f"{candidate_features}"
    )

    # -------------------------------------------------------------------------
    # SAME FOUR-STAGE PROCEDURE AS SAMPLE-GROUPED ANALYSIS
    # -------------------------------------------------------------------------

    stage1_features, stage1_report = (
        stage1_quality_filter(
            df_cal,
            candidate_features,
        )
    )

    stage2_features, stage2_report = (
        stage2_soc_screening(
            df_cal,
            stage1_features,
        )
    )

    pca_features, pca_report, pca_model = (
        stage3_pca(
            df_cal,
            stage2_features,
        )
    )

    final_features, corr_matrix = (
        stage4_pairwise_spearman_pruning(
            df_cal,
            pca_features,
            stage2_report,
        )
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL SELECTED FEATURES — KS IMAGE-LEVEL ANALYSIS")
    print("=" * 75)

    for i, feat in enumerate(
        final_features,
        start=1,
    ):
        print(
            f"  {i:2d}. {feat}"
        )

    print(
        f"\n  Original: "
        f"{len(candidate_features)}"
        f" -> Stage1: {len(stage1_features)}"
        f" -> Stage2: {len(stage2_features)}"
        f" -> PCA: {len(pca_features)}"
        f" -> Final: {len(final_features)}"
    )

    # -------------------------------------------------------------------------
    # SAVE FEATURE LIST
    # -------------------------------------------------------------------------

    with open(
        os.path.join(
            OUT_DIR,
            "selected_features_list.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        for feat in final_features:
            f.write(
                feat + "\n"
            )

    # -------------------------------------------------------------------------
    # SAVE IMAGE-LEVEL CALIBRATION DATA
    # -------------------------------------------------------------------------

    metadata_keep = [
        GROUP,
        "image_no",
        "moisture",
        "soil_type",
        TARGET,
    ]

    cal_keep = [
        c
        for c in metadata_keep
        if c in df_cal.columns
    ]

    cal_keep += [
        feat
        for feat in final_features
        if feat in df_cal.columns
    ]

    df_cal[
        cal_keep
    ].to_csv(
        os.path.join(
            OUT_DIR,
            "calibration_selected_features.csv",
        ),
        index=False,
    )

    print(
        f"\n  Saved: "
        f"calibration_selected_features.csv "
        f"({len(df_cal)} rows)"
    )

    # -------------------------------------------------------------------------
    # APPLY CALIBRATION-DERIVED FEATURES TO UNTOUCHED VALIDATION DATA
    # -------------------------------------------------------------------------

    missing_val_features = [
        feat
        for feat in final_features
        if feat not in df_val.columns
    ]

    if missing_val_features:
        raise ValueError(
            "Validation set is missing selected predictors: "
            f"{missing_val_features}"
        )

    val_keep = [
        c
        for c in cal_keep
        if c in df_val.columns
    ]

    df_val[
        val_keep
    ].to_csv(
        os.path.join(
            OUT_DIR,
            "validation_selected_features.csv",
        ),
        index=False,
    )

    print(
        f"  Saved: "
        f"validation_selected_features.csv "
        f"({len(df_val)} rows)"
    )

    # -------------------------------------------------------------------------
    # SAVE COMPACT SUMMARY
    # -------------------------------------------------------------------------

    summary_lines = [
        "=" * 75,
        "KS STEP 2 FEATURE-SELECTION SUMMARY",
        "=" * 75,
        f"Calibration images                 : {len(df_cal)}",
        f"Validation images                  : {len(df_val)}",
        f"Candidate predictors               : {len(candidate_features)}",
        f"Stage 1 retained                   : {len(stage1_features)}",
        f"Stage 2 retained                   : {len(stage2_features)}",
        f"PCA retained                       : {len(pca_features)}",
        f"Final predictors                   : {len(final_features)}",
        "",
        "Final selected predictors:",
        *[
            f"  {i}. {feat}"
            for i, feat in enumerate(
                final_features,
                start=1,
            )
        ],
        "",
        "Validation used for feature selection: NO",
        "Feature-selection logic matches sample-grouped Step 2: YES",
        "=" * 75,
    ]

    with open(
        os.path.join(
            OUT_DIR,
            "step2_summary.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(summary_lines)
        )

    print(
        f"\n  Outputs: "
        f"{OUT_DIR}"
    )

    print(
        "\nNext step: run KS AutoML using the SAME "
        "RMSE + composite re-ranking logic as the sample-grouped analysis."
    )


if __name__ == "__main__":
    main()
