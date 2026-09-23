"""
OBJECTIVE 2 — LEAVE-ONE-SOIL-SERIES-OUT SOC PREDICTION
======================================================

Purpose
-------
Evaluate whether smartphone image-derived features can predict SOC in an
entire soil series excluded from model development.

This analysis is NOT a Kennard-Stone image-level analysis.

For each outer fold:
    Train = all other soil series
    Test  = one completely held-out soil series

Within the outer training data:
    1. GroupKFold by physical soil Sample_No
    2. Mahalanobis QC is performed on inner training data only
    3. Feature selection is performed on inner training data only
    4. AutoGluon models are trained using explicit grouped tuning data
    5. Models are evaluated using RMSE, MAE, R², RPD and RPIQ
    6. Model names are ranked across inner folds using the same composite
       scoring framework used in Objective 1
    7. The selected model is applied to the untouched held-out soil series
       using an ensemble of the inner-fold models

Important
---------
- Held-out soil series NEVER enters feature selection or model selection.
- Held-out test images are NEVER removed as outliers.
- Sample_No, soil_type, moisture, image_no are metadata only.
- SOC is the target and never a predictor.
- Negative R² values are retained and reported.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats import chi2, spearmanr

from sklearn.covariance import EmpiricalCovariance
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from autogluon.tabular import TabularPredictor

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "soil_image_features_without_commas.csv",
)

METADATA_FILE = os.path.join(
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
    "objective_2",
    "output_data",
    "soil_series_transfer",
)

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

TARGET_COL = "soc"
GROUP_COL = "Sample_No"
SERIES_COL = "soil_type"

SEED = 42

# Outlier QC
MAHALANOBIS_CONFIDENCE = 0.99

# Feature-selection settings
MAX_MISSING = 0.20
MIN_SCALED_VARIANCE = 0.01

MIN_ABS_SPEARMAN = 0.20
SCREENING_P = 0.10

N_MI_PERMUTATIONS = 500
MI_NULL_PERCENTILE = 95

MIN_PC12_CUMVAR = 0.60       # diagnostic only
PAIRWISE_THRESHOLD = 0.90

# AutoGluon
MAX_INNER_FOLDS = 4
FOLD_TIME_LIMIT = 600
PRESETS = "medium_quality"

# Composite model-ranking weights
WEIGHTS = {
    "RPD": 0.35,
    "RPIQ": 0.25,
    "RMSE": 0.20,
    "R2": 0.15,
    "MAE": 0.05,
}

NON_PREDICTOR_COLS = {
    "image_no",
    "Image_No",
    "Sample_No",
    "soil_type",
    "moisture",
    "soc",
}


# =============================================================================
# HELPERS
# =============================================================================

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
        sd_y = np.std(y_true, ddof=1)
    else:
        r2 = np.nan
        sd_y = np.nan

    rpd = (
        sd_y / rmse
        if np.isfinite(sd_y) and rmse > 0
        else np.nan
    )

    if len(y_true) > 0:
        q75, q25 = np.percentile(y_true, [75, 25])
        iqr_y = q75 - q25
    else:
        iqr_y = np.nan

    rpiq = (
        iqr_y / rmse
        if np.isfinite(iqr_y) and rmse > 0
        else np.nan
    )

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "RPD": rpd,
        "RPIQ": rpiq,
    }


def normalize_higher_is_better(series):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index)

    valid = np.isfinite(s)

    if valid.sum() == 0:
        return out

    lo = s[valid].min()
    hi = s[valid].max()

    if np.isclose(lo, hi):
        out.loc[valid] = 1.0
    else:
        out.loc[valid] = (s[valid] - lo) / (hi - lo)

    return out


def normalize_lower_is_better(series):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index)

    valid = np.isfinite(s)

    if valid.sum() == 0:
        return out

    lo = s[valid].min()
    hi = s[valid].max()

    if np.isclose(lo, hi):
        out.loc[valid] = 1.0
    else:
        out.loc[valid] = (hi - s[valid]) / (hi - lo)

    return out


def add_composite_score(df):
    df = df.copy()

    df["RPD_norm"] = normalize_higher_is_better(df["Mean_RPD"])
    df["RPIQ_norm"] = normalize_higher_is_better(df["Mean_RPIQ"])
    df["RMSE_norm"] = normalize_lower_is_better(df["Mean_RMSE"])
    df["R2_norm"] = normalize_higher_is_better(df["Mean_R2"])
    df["MAE_norm"] = normalize_lower_is_better(df["Mean_MAE"])

    df["Composite_Score"] = (
        WEIGHTS["RPD"] * df["RPD_norm"]
        + WEIGHTS["RPIQ"] * df["RPIQ_norm"]
        + WEIGHTS["RMSE"] * df["RMSE_norm"]
        + WEIGHTS["R2"] * df["R2_norm"]
        + WEIGHTS["MAE"] * df["MAE_norm"]
    )

    return (
        df.sort_values(
            ["Composite_Score", "Mean_RMSE"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


# =============================================================================
# LOAD + MERGE
# =============================================================================

print("\n" + "=" * 90)
print("OBJECTIVE 2 — LEAVE-ONE-SOIL-SERIES-OUT ANALYSIS")
print("=" * 90)

features = pd.read_csv(FEATURE_FILE)
metadata = pd.read_csv(METADATA_FILE)

print(f"\nFeature file : {features.shape}")
print(f"Metadata file: {metadata.shape}")


# ## =============================================================================
# MERGE FEATURE FILE WITH METADATA
# =============================================================================

FEATURE_IMAGE_COL = "Numeric numbers"
META_IMAGE_COL = "image_no"

if FEATURE_IMAGE_COL not in features.columns:
    raise ValueError(
        f"Feature file is missing '{FEATURE_IMAGE_COL}'. "
        f"Columns are: {features.columns.tolist()}"
    )

if META_IMAGE_COL not in metadata.columns:
    raise ValueError(
        f"Metadata file is missing '{META_IMAGE_COL}'. "
        f"Columns are: {metadata.columns.tolist()}"
    )

master = pd.merge(
    features,
    metadata,
    left_on=FEATURE_IMAGE_COL,
    right_on=META_IMAGE_COL,
    how="inner",
    suffixes=("", "_meta"),
)

feature_image_col = FEATURE_IMAGE_COL

print(f"Merged dataset: {master.shape}")

# =============================================================================
# BUILD IMAGE -> PHYSICAL SAMPLE MAPPING
# =============================================================================

def build_sample_mapping():

    print("\nBuilding image -> Sample_No mapping from Excel...")

    xl = pd.ExcelFile(IMAGE_RECORD)

    # -------------------------------------------------------------------------
    # Sheet 1-205
    # -------------------------------------------------------------------------
    raw = pd.read_excel(
        xl,
        sheet_name="1-205",
        header=None,
    )

    groups = []

    for start_col in [0, 9, 17]:

        grp = raw.iloc[
            1:,
            start_col:start_col + 6
        ].copy()

        grp.columns = [
            "Date",
            "Sample_No",
            "Image_No",
            "M",
            "ST",
            "SOC",
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
            subset=["Sample_No", "Image_No"]
        )

        groups.append(
            grp[["Sample_No", "Image_No"]]
        )

    first_mapping = pd.concat(
        groups,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Other sheets
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

            sample_col = "Sample No." + suffix
            image_col = "Image No." + suffix

            if (
                sample_col in df.columns
                and image_col in df.columns
            ):

                grp = df[
                    [sample_col, image_col]
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

        if groups:
            return pd.concat(
                groups,
                ignore_index=True,
            )

        return pd.DataFrame(
            columns=[
                "Sample_No",
                "Image_No",
            ]
        )

    second_mapping = read_other_sheet(
        "206-360",
        ["", ".1", ".2"],
    )

    third_mapping = read_other_sheet(
        "361-731",
        ["", ".1", ".2", ".3", ".4"],
    )

    mapping = pd.concat(
        [
            first_mapping,
            second_mapping,
            third_mapping,
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
        f"Images mapped       : {len(mapping)}"
    )

    print(
        f"Physical soils      : "
        f"{mapping['Sample_No'].nunique()}"
    )

    print(
        "Sample numbers      :",
        sorted(
            mapping["Sample_No"].unique()
        ),
    )

    return mapping


sample_mapping = build_sample_mapping()


# =============================================================================
# ATTACH Sample_No TO MASTER DATASET
# =============================================================================

master["image_no"] = pd.to_numeric(
    master["image_no"],
    errors="coerce",
)

master = master.merge(
    sample_mapping.rename(
        columns={"Image_No": "image_no"}
    ),
    on="image_no",
    how="left",
)

unmatched = master["Sample_No"].isna().sum()

if unmatched > 0:
    raise ValueError(
        f"{unmatched} images could not be mapped "
        f"to a physical Sample_No."
    )

master["Sample_No"] = (
    pd.to_numeric(
        master["Sample_No"],
        errors="coerce",
    )
    .astype(int)
)

print(
    f"\nMaster after Sample_No mapping: "
    f"{master.shape}"
)

print(
    f"Physical soils represented: "
    f"{master['Sample_No'].nunique()}"
)

if len(master) != 731:
    print(
        f"WARNING: Expected 731 merged rows, but obtained {len(master)}."
    )

print(f"Merged dataset: {master.shape}")


# =============================================================================
# VALIDATE REQUIRED COLUMNS
# =============================================================================

for col in [TARGET_COL, GROUP_COL, SERIES_COL]:
    if col not in master.columns:
        raise ValueError(
            f"Required column '{col}' not found.\n"
            f"Available columns:\n{master.columns.tolist()}"
        )

master[TARGET_COL] = pd.to_numeric(
    master[TARGET_COL],
    errors="coerce",
)

master = master.dropna(
    subset=[TARGET_COL, GROUP_COL, SERIES_COL]
).copy()


# =============================================================================
# IDENTIFY SOIL SERIES
# =============================================================================

series_counts = (
    master
    .groupby(SERIES_COL)
    .agg(
        N_Images=(TARGET_COL, "size"),
        N_Physical_Soils=(GROUP_COL, "nunique"),
        Mean_SOC=(TARGET_COL, "mean"),
        Min_SOC=(TARGET_COL, "min"),
        Max_SOC=(TARGET_COL, "max"),
    )
    .reset_index()
)

print("\nDetected soil series:")
print(series_counts.to_string(index=False))

soil_series = sorted(
    master[SERIES_COL]
    .dropna()
    .astype(str)
    .unique()
)

print(f"\nNumber of soil series detected: {len(soil_series)}")
print(f"Soil series: {soil_series}")

series_counts.to_csv(
    os.path.join(OUT_DIR, "soil_series_distribution.csv"),
    index=False,
)


# =============================================================================
# IDENTIFY NUMERIC IMAGE PREDICTORS
# =============================================================================

candidate_features = []

for col in master.columns:

    if col in NON_PREDICTOR_COLS:
        continue

    if col.endswith("_meta"):
        continue

    numeric = pd.to_numeric(
        master[col],
        errors="coerce",
    )

    if numeric.notna().sum() > 0:
        master[col] = numeric
        candidate_features.append(col)

print(
    f"\nCandidate image predictors detected: "
    f"{len(candidate_features)}"
)

print(candidate_features)


# =============================================================================
# MAHALANOBIS QC — TRAINING DATA ONLY
# =============================================================================

def mahalanobis_training_qc(train_df, features_list):

    X = train_df[features_list].copy()

    # Median imputation
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Pseudoinverse formulation is stable if covariance is near-singular
    mean_vec = np.mean(X_scaled, axis=0)
    centered = X_scaled - mean_vec

    cov = np.cov(
        X_scaled,
        rowvar=False,
    )

    inv_cov = np.linalg.pinv(cov)

    md_squared = np.einsum(
        "ij,jk,ik->i",
        centered,
        inv_cov,
        centered,
    )

    threshold_sq = chi2.ppf(
        MAHALANOBIS_CONFIDENCE,
        df=len(features_list),
    )

    keep = md_squared <= threshold_sq

    clean = train_df.loc[keep].copy()

    return clean, md_squared, threshold_sq


# =============================================================================
# FOUR-STAGE FEATURE SELECTION — TRAINING ONLY
# =============================================================================

def feature_selection(train_df, candidate_features, seed=SEED):

    y = train_df[TARGET_COL].values.astype(float)

    # -------------------------------------------------------------------------
    # STAGE 1 — missingness + scaled variance
    # -------------------------------------------------------------------------

    stage1_missing = []

    for feature in candidate_features:
        missing_fraction = train_df[feature].isna().mean()

        if missing_fraction <= MAX_MISSING:
            stage1_missing.append(feature)

    if len(stage1_missing) == 0:
        raise RuntimeError(
            "No features survived missingness screening."
        )

    X1 = train_df[stage1_missing].copy()

    imputer1 = SimpleImputer(strategy="median")
    X1_imp = imputer1.fit_transform(X1)

    mm_scaler = MinMaxScaler()
    X1_scaled = mm_scaler.fit_transform(X1_imp)

    variance_values = np.var(
        X1_scaled,
        axis=0,
        ddof=0,
    )

    stage1_features = [
        f
        for f, v in zip(stage1_missing, variance_values)
        if v >= MIN_SCALED_VARIANCE
    ]

    if len(stage1_features) == 0:
        raise RuntimeError(
            "No features survived Stage 1."
        )

    # -------------------------------------------------------------------------
    # STAGE 2 — SOC screening
    # Spearman OR permutation-based MI
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(seed)

    stage2_records = []
    stage2_features = []

    for feature in stage1_features:

        x = pd.to_numeric(
            train_df[feature],
            errors="coerce",
        )

        x_filled = x.fillna(x.median()).values

        rho, p = spearmanr(
            x_filled,
            y,
        )

        observed_mi = mutual_info_regression(
            x_filled.reshape(-1, 1),
            y,
            random_state=seed,
        )[0]

        null_mi = []

        for permutation_id in range(
            N_MI_PERMUTATIONS
        ):
            permuted_y = rng.permutation(y)

            perm_mi = mutual_info_regression(
                x_filled.reshape(-1, 1),
                permuted_y,
                random_state=seed + permutation_id + 1,
            )[0]

            null_mi.append(perm_mi)

        mi_threshold = np.percentile(
            null_mi,
            MI_NULL_PERCENTILE,
        )

        spearman_pass = (
            np.isfinite(rho)
            and np.isfinite(p)
            and abs(rho) >= MIN_ABS_SPEARMAN
            and p < SCREENING_P
        )

        mi_pass = (
            np.isfinite(observed_mi)
            and observed_mi > mi_threshold
        )

        passed = spearman_pass or mi_pass

        stage2_records.append(
            {
                "Feature": feature,
                "Spearman_rho": rho,
                "Spearman_p": p,
                "Mutual_Information": observed_mi,
                "MI_95pct_Null": mi_threshold,
                "Spearman_Pass": spearman_pass,
                "MI_Pass": mi_pass,
                "Stage2_Pass": passed,
            }
        )

        if passed:
            stage2_features.append(feature)

    if len(stage2_features) < 2:
        raise RuntimeError(
            "Fewer than two features survived Stage 2."
        )

    stage2_df = pd.DataFrame(stage2_records)

    # -------------------------------------------------------------------------
    # STAGE 3 — PCA contribution
    # -------------------------------------------------------------------------

    X3 = train_df[stage2_features].copy()

    imp3 = SimpleImputer(strategy="median")
    X3_imp = imp3.fit_transform(X3)

    scaler3 = StandardScaler()
    X3_scaled = scaler3.fit_transform(X3_imp)

    pca = PCA()
    pca.fit(X3_scaled)

    explained = pca.explained_variance_ratio_

    pc1_var = explained[0]

    pc2_var = (
        explained[1]
        if len(explained) > 1
        else 0.0
    )

    cumulative_pc12 = pc1_var + pc2_var

    loadings = pca.components_.T

    expected_contribution = (
        1.0 / len(stage2_features)
    )

    contributions = []

    stage3_features = []

    for i, feature in enumerate(stage2_features):

        pc1_contribution = (
            loadings[i, 0] ** 2
            / np.sum(loadings[:, 0] ** 2)
        )

        if loadings.shape[1] > 1:
            pc2_contribution = (
                loadings[i, 1] ** 2
                / np.sum(loadings[:, 1] ** 2)
            )
        else:
            pc2_contribution = 0.0

        retain = (
            pc1_contribution > expected_contribution
            or
            pc2_contribution > expected_contribution
        )

        contributions.append(
            {
                "Feature": feature,
                "PC1_Contribution": pc1_contribution,
                "PC2_Contribution": pc2_contribution,
                "Expected_Contribution": expected_contribution,
                "Stage3_Pass": retain,
            }
        )

        if retain:
            stage3_features.append(feature)

    if len(stage3_features) == 0:
        raise RuntimeError(
            "No features survived Stage 3 PCA."
        )

    stage3_df = pd.DataFrame(contributions)

    # -------------------------------------------------------------------------
    # STAGE 4 — redundancy pruning
    # -------------------------------------------------------------------------

    ranking_info = (
        stage2_df[
            stage2_df["Feature"].isin(stage3_features)
        ]
        .copy()
    )

    ranking_info["Abs_Spearman"] = (
        ranking_info["Spearman_rho"].abs()
    )

    ranking_info = ranking_info.sort_values(
        [
            "Abs_Spearman",
            "Mutual_Information",
        ],
        ascending=[False, False],
    )

    ordered_features = (
        ranking_info["Feature"].tolist()
    )

    selected = []

    for candidate in ordered_features:

        keep_candidate = True

        for already_selected in selected:

            rho_pair, _ = spearmanr(
                train_df[candidate],
                train_df[already_selected],
                nan_policy="omit",
            )

            if (
                np.isfinite(rho_pair)
                and abs(rho_pair)
                >= PAIRWISE_THRESHOLD
            ):
                keep_candidate = False
                break

        if keep_candidate:
            selected.append(candidate)

    if len(selected) == 0:
        raise RuntimeError(
            "No features survived Stage 4."
        )

    diagnostics = {
        "Stage1_N": len(stage1_features),
        "Stage2_N": len(stage2_features),
        "Stage3_N": len(stage3_features),
        "Stage4_N": len(selected),
        "PC1_Variance": pc1_var,
        "PC2_Variance": pc2_var,
        "PC12_Cumulative": cumulative_pc12,
    }

    return (
        selected,
        diagnostics,
        stage2_df,
        stage3_df,
    )


# =============================================================================
# OUTER LOOP — HOLD OUT ONE SOIL SERIES
# =============================================================================

all_outer_results = []
all_predictions = []
all_model_rankings = []
all_feature_records = []

for outer_index, held_series in enumerate(
    soil_series,
    start=1,
):

    print("\n" + "#" * 90)
    print(
        f"OUTER FOLD {outer_index}/{len(soil_series)}"
    )
    print(
        f"HELD-OUT SOIL SERIES: {held_series}"
    )
    print("#" * 90)

    outer_train = master[
        master[SERIES_COL].astype(str)
        != str(held_series)
    ].copy()

    outer_test = master[
        master[SERIES_COL].astype(str)
        == str(held_series)
    ].copy()

    train_soils = set(
        outer_train[GROUP_COL].unique()
    )

    test_soils = set(
        outer_test[GROUP_COL].unique()
    )

    overlap = train_soils & test_soils

    if overlap:
        raise ValueError(
            f"Physical-soil overlap between outer train/test: "
            f"{sorted(overlap)}"
        )

    print(
        f"Training images : {len(outer_train)}"
    )
    print(
        f"Training soils  : {len(train_soils)}"
    )
    print(
        f"Test images     : {len(outer_test)}"
    )
    print(
        f"Test soils      : {len(test_soils)}"
    )
    print(
        "Physical-soil overlap: ZERO"
    )

    n_inner_folds = min(
        MAX_INNER_FOLDS,
        len(train_soils),
    )

    if n_inner_folds < 2:
        raise RuntimeError(
            "Not enough physical soils for grouped "
            "internal cross-validation."
        )

    gkf = GroupKFold(
        n_splits=n_inner_folds
    )

    groups = outer_train[GROUP_COL].values

    fold_model_results = []
    fold_predictors = []

    # =========================================================================
    # INNER GROUP-AWARE CV
    # =========================================================================

    for inner_fold, (
        fit_idx,
        tune_idx,
    ) in enumerate(
        gkf.split(
            outer_train,
            outer_train[TARGET_COL],
            groups=groups,
        ),
        start=1,
    ):

        print("\n" + "-" * 90)
        print(
            f"Held series = {held_series} | "
            f"Inner fold = {inner_fold}"
        )
        print("-" * 90)

        inner_train_raw = (
            outer_train.iloc[fit_idx].copy()
        )

        inner_tune = (
            outer_train.iloc[tune_idx].copy()
        )

        inner_train_soils = set(
            inner_train_raw[GROUP_COL].unique()
        )

        inner_tune_soils = set(
            inner_tune[GROUP_COL].unique()
        )

        if (
            inner_train_soils
            & inner_tune_soils
        ):
            raise ValueError(
                "Inner fold physical-soil leakage."
            )

        # ---------------------------------------------------------------------
        # QC ONLY ON INNER TRAINING
        # ---------------------------------------------------------------------

        inner_train, md2, md_threshold = (
            mahalanobis_training_qc(
                inner_train_raw,
                candidate_features,
            )
        )

        removed = (
            len(inner_train_raw)
            - len(inner_train)
        )

        print(
            f"Inner training before QC: "
            f"{len(inner_train_raw)}"
        )

        print(
            f"Inner training after QC : "
            f"{len(inner_train)}"
        )

        print(
            f"Training outliers removed: "
            f"{removed}"
        )

        print(
            f"Inner tuning images      : "
            f"{len(inner_tune)} "
            f"(unchanged)"
        )

        # ---------------------------------------------------------------------
        # FEATURE SELECTION ONLY ON INNER TRAINING
        # ---------------------------------------------------------------------

        (
            selected_features,
            feature_diag,
            stage2_df,
            stage3_df,
        ) = feature_selection(
            inner_train,
            candidate_features,
            seed=SEED + inner_fold + outer_index * 100,
        )

        print(
            f"Selected predictors "
            f"({len(selected_features)}):"
        )
        print(selected_features)

        for feature in selected_features:
            all_feature_records.append(
                {
                    "Held_Out_Series": held_series,
                    "Inner_Fold": inner_fold,
                    "Feature": feature,
                    **feature_diag,
                }
            )

        # ---------------------------------------------------------------------
        # MODEL DATA
        # ---------------------------------------------------------------------

        train_model = inner_train[
            [TARGET_COL] + selected_features
        ].copy()

        tune_model = inner_tune[
            [TARGET_COL] + selected_features
        ].copy()

        model_dir = os.path.join(
            OUT_DIR,
            "models",
            f"held_{str(held_series).replace(' ', '_')}",
            f"inner_fold_{inner_fold}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        )

        predictor = TabularPredictor(
            label=TARGET_COL,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            path=model_dir,
        ).fit(
            train_data=train_model,
            tuning_data=tune_model,
            presets=PRESETS,
            time_limit=FOLD_TIME_LIMIT,
            verbosity=1,
        )

        y_tune = tune_model[TARGET_COL].values

        models_here = predictor.model_names()

        for model_name in models_here:

            try:
                pred = predictor.predict(
                    tune_model,
                    model=model_name,
                )

                metrics = regression_metrics(
                    y_tune,
                    pred,
                )

                fold_model_results.append(
                    {
                        "Held_Out_Series": held_series,
                        "Inner_Fold": inner_fold,
                        "Model": model_name,
                        "R2": metrics["R2"],
                        "RMSE": metrics["RMSE"],
                        "MAE": metrics["MAE"],
                        "RPD": metrics["RPD"],
                        "RPIQ": metrics["RPIQ"],
                        "N_Train_Images": len(train_model),
                        "N_Tune_Images": len(tune_model),
                        "N_Selected_Features":
                            len(selected_features),
                    }
                )

            except Exception as exc:
                print(
                    f"Warning: could not evaluate "
                    f"{model_name}: {exc}"
                )

        fold_predictors.append(
            {
                "fold": inner_fold,
                "predictor": predictor,
                "features": selected_features,
            }
        )

    # =========================================================================
    # SELECT MODEL NAME ACROSS INNER FOLDS
    # =========================================================================

    fold_results_df = pd.DataFrame(
        fold_model_results
    )

    required_fold_count = n_inner_folds

    summary = (
        fold_results_df
        .groupby("Model")
        .agg(
            N_Folds=("Inner_Fold", "nunique"),
            Mean_R2=("R2", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_MAE=("MAE", "mean"),
            Mean_RPD=("RPD", "mean"),
            Mean_RPIQ=("RPIQ", "mean"),
        )
        .reset_index()
    )

    # Require model to be available in every inner fold
    summary = summary[
        summary["N_Folds"]
        == required_fold_count
    ].copy()

    if summary.empty:
        raise RuntimeError(
            f"No model was available in all "
            f"{required_fold_count} inner folds "
            f"for held-out series {held_series}."
        )

    summary = add_composite_score(summary)

    selected_model = summary.iloc[0]["Model"]

    summary["Held_Out_Series"] = held_series

    all_model_rankings.append(summary)

    print("\nMODEL RANKING")
    print(
        summary[
            [
                "Model",
                "Mean_RMSE",
                "Mean_MAE",
                "Mean_R2",
                "Mean_RPD",
                "Mean_RPIQ",
                "Composite_Score",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSelected model: {selected_model}"
    )

    # =========================================================================
    # PREDICT COMPLETELY HELD-OUT SOIL SERIES
    # CV ENSEMBLE — NO FINAL RANDOM HOLDOUT
    # =========================================================================

    fold_test_predictions = []

    for obj in fold_predictors:

        predictor = obj["predictor"]
        selected_features = obj["features"]

        if (
            selected_model
            not in predictor.model_names()
        ):
            continue

        test_model = outer_test[
            [TARGET_COL] + selected_features
        ].copy()

        pred = predictor.predict(
            test_model,
            model=selected_model,
        )

        fold_test_predictions.append(
            np.asarray(pred, dtype=float)
        )

    if len(fold_test_predictions) != n_inner_folds:
        raise RuntimeError(
            f"Selected model {selected_model} "
            f"was not available in every inner fold."
        )

    y_pred = np.mean(
        np.vstack(fold_test_predictions),
        axis=0,
    )

    y_true = outer_test[TARGET_COL].values

    metrics = regression_metrics(
        y_true,
        y_pred,
    )

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------

    training_mean_soc = (
        outer_train[TARGET_COL].mean()
    )

    baseline_pred = np.full(
        len(outer_test),
        training_mean_soc,
    )

    baseline_metrics = regression_metrics(
        y_true,
        baseline_pred,
    )

    print("\n" + "=" * 90)
    print(
        f"HELD-OUT SERIES PERFORMANCE: "
        f"{held_series}"
    )
    print("=" * 90)

    print(
        f"R²   = {metrics['R2']:.4f}"
    )
    print(
        f"RMSE = {metrics['RMSE']:.4f}"
    )
    print(
        f"MAE  = {metrics['MAE']:.4f}"
    )
    print(
        f"RPD  = {metrics['RPD']:.4f}"
    )
    print(
        f"RPIQ = {metrics['RPIQ']:.4f}"
    )

    print(
        f"\nCalibration-mean baseline RMSE = "
        f"{baseline_metrics['RMSE']:.4f}"
    )

    print(
        f"Calibration-mean baseline MAE  = "
        f"{baseline_metrics['MAE']:.4f}"
    )

    all_outer_results.append(
        {
            "Held_Out_Series": held_series,
            "Selected_Model": selected_model,
            "N_Train_Images": len(outer_train),
            "N_Train_Soils": len(train_soils),
            "N_Test_Images": len(outer_test),
            "N_Test_Soils": len(test_soils),
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "RPD": metrics["RPD"],
            "RPIQ": metrics["RPIQ"],
            "Training_Mean_SOC": training_mean_soc,
            "Baseline_R2": baseline_metrics["R2"],
            "Baseline_RMSE": baseline_metrics["RMSE"],
            "Baseline_MAE": baseline_metrics["MAE"],
            "Baseline_RPD": baseline_metrics["RPD"],
            "Baseline_RPIQ": baseline_metrics["RPIQ"],
        }
    )

    prediction_df = outer_test[
        [
            feature_image_col,
            GROUP_COL,
            SERIES_COL,
            TARGET_COL,
        ]
    ].copy()

    prediction_df["Predicted_SOC"] = y_pred
    prediction_df["Baseline_SOC"] = (
        training_mean_soc
    )

    all_predictions.append(prediction_df)


# =============================================================================
# COMBINE ALL OUTER RESULTS
# =============================================================================

results_df = pd.DataFrame(
    all_outer_results
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

rankings_df = pd.concat(
    all_model_rankings,
    ignore_index=True,
)

features_df = pd.DataFrame(
    all_feature_records
)


# =============================================================================
# POOLED OUT-OF-SERIES PERFORMANCE
# =============================================================================

pooled_metrics = regression_metrics(
    predictions_df[TARGET_COL],
    predictions_df["Predicted_SOC"],
)

pooled_baseline_metrics = regression_metrics(
    predictions_df[TARGET_COL],
    predictions_df["Baseline_SOC"],
)

print("\n" + "=" * 90)
print(
    "POOLED LEAVE-ONE-SOIL-SERIES-OUT PERFORMANCE"
)
print("=" * 90)

print(results_df.to_string(index=False))

print("\nPooled out-of-series predictions:")

print(
    f"R²   = {pooled_metrics['R2']:.4f}"
)
print(
    f"RMSE = {pooled_metrics['RMSE']:.4f}"
)
print(
    f"MAE  = {pooled_metrics['MAE']:.4f}"
)
print(
    f"RPD  = {pooled_metrics['RPD']:.4f}"
)
print(
    f"RPIQ = {pooled_metrics['RPIQ']:.4f}"
)

print("\nPooled baseline:")

print(
    f"RMSE = "
    f"{pooled_baseline_metrics['RMSE']:.4f}"
)

print(
    f"MAE  = "
    f"{pooled_baseline_metrics['MAE']:.4f}"
)


# =============================================================================
# SOIL-CONDITION LEVEL
# =============================================================================

soil_condition = (
    predictions_df
    .groupby(
        [
            SERIES_COL,
            GROUP_COL,
            TARGET_COL,
        ],
        as_index=False,
    )
    .agg(
        Predicted_SOC=(
            "Predicted_SOC",
            "mean",
        ),
        N_Images=(
            "Predicted_SOC",
            "size",
        ),
    )
    .rename(
        columns={
            TARGET_COL: "Actual_SOC"
        }
    )
)

soil_condition_metrics = regression_metrics(
    soil_condition["Actual_SOC"],
    soil_condition["Predicted_SOC"],
)

print("\nSoil-condition-level pooled performance:")

print(
    f"R²   = "
    f"{soil_condition_metrics['R2']:.4f}"
)

print(
    f"RMSE = "
    f"{soil_condition_metrics['RMSE']:.4f}"
)

print(
    f"MAE  = "
    f"{soil_condition_metrics['MAE']:.4f}"
)

print(
    f"RPD  = "
    f"{soil_condition_metrics['RPD']:.4f}"
)

print(
    f"RPIQ = "
    f"{soil_condition_metrics['RPIQ']:.4f}"
)


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

results_df.to_csv(
    os.path.join(
        OUT_DIR,
        "soil_series_performance.csv",
    ),
    index=False,
)

predictions_df.to_csv(
    os.path.join(
        OUT_DIR,
        "soil_series_predictions.csv",
    ),
    index=False,
)

rankings_df.to_csv(
    os.path.join(
        OUT_DIR,
        "soil_series_model_rankings.csv",
    ),
    index=False,
)

features_df.to_csv(
    os.path.join(
        OUT_DIR,
        "soil_series_selected_features.csv",
    ),
    index=False,
)

soil_condition.to_csv(
    os.path.join(
        OUT_DIR,
        "soil_series_soil_condition_predictions.csv",
    ),
    index=False,
)

pooled_summary = pd.DataFrame(
    [
        {
            "Evaluation": "Image-level pooled",
            **pooled_metrics,
        },
        {
            "Evaluation": "Soil-condition pooled",
            **soil_condition_metrics,
        },
    ]
)

pooled_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "soil_series_pooled_performance.csv",
    ),
    index=False,
)

print("\n" + "=" * 90)
print("OBJECTIVE 2 SOIL-SERIES ANALYSIS COMPLETE")
print("=" * 90)

print(
    f"Outputs saved to:\n{OUT_DIR}"
)