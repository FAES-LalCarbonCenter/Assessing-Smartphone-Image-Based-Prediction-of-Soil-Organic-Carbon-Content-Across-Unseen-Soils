"""
OBJECTIVE 2 — MOISTURE EFFECT ON SOC PREDICTION
================================================

Purpose
-------
Evaluate whether SOC prediction performance changes across soil-moisture
conditions for physical soils that were completely excluded from model
calibration.

IMPORTANT DESIGN FEATURES
-------------------------
1. Uses ONLY the sample-grouped external-validation predictions from Objective 1.
2. No model is retrained.
3. No Kennard-Stone comparison is performed here.
4. Moisture is NOT a model predictor in this analysis.
5. Validation images are classified AFTER prediction into:
       Dry   : 0 <= SMC < 10
       Moist : 10 <= SMC <= 30
       Wet   : SMC > 30
6. Multiple images from the same physical soil are NOT treated as independent
   for uncertainty estimation.
7. Cluster-bootstrap confidence intervals resample physical soils (Sample_No).
8. A secondary equal-soil analysis summarizes prediction error within each
   physical soil and moisture class.
9. Variance-dependent metrics (R2, RPD, RPIQ) are reported as NaN when the
   observed SOC values have insufficient variation.

Primary interpretation
----------------------
RMSE, MAE and bias are the main moisture-comparison metrics.

R2, RPD and RPIQ are supplementary because only four independent physical
soils are represented in external validation.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

PREDICTION_FILE = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step3_output",
    "external_validation_predictions.csv",
)

# Used only to obtain the calibration mean SOC baseline.
CALIBRATION_FILE = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step2_output",
    "calibration_selected_features.csv",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_2",
    "output_data",
    "moisture_analysis",
)

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)


# =============================================================================
# SETTINGS
# =============================================================================

TARGET_COL = "soc"
PRED_COL = "Predicted_SOC"
GROUP_COL = "Sample_No"
MOISTURE_COL = "moisture"
IMAGE_COL = "image_no"

N_BOOTSTRAP = 5000
SEED = 42

MOISTURE_ORDER = [
    "Dry",
    "Moist",
    "Wet",
]


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def regression_metrics(y_true, y_pred):
    """
    Calculate SOC prediction metrics.

    R2, RPD and RPIQ require variation in observed SOC.
    They are returned as NaN when observed SOC variance is effectively zero.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n = len(y_true)

    if n == 0:
        return {
            "N": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "RPD": np.nan,
            "RPIQ": np.nan,
            "Bias": np.nan,
        }

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    bias = np.mean(
        y_pred - y_true
    )

    # -------------------------------------------------------------------------
    # Metrics requiring variation in y_true
    # -------------------------------------------------------------------------

    if n >= 2:
        variance_y = np.var(
            y_true,
            ddof=1,
        )
    else:
        variance_y = np.nan

    if (
        np.isfinite(variance_y)
        and variance_y > 1e-12
    ):

        r2 = r2_score(
            y_true,
            y_pred,
        )

        sd_y = np.sqrt(
            variance_y
        )

        rpd = (
            sd_y / rmse
            if rmse > 0
            else np.nan
        )

        q75, q25 = np.percentile(
            y_true,
            [75, 25],
        )

        iqr_y = q75 - q25

        rpiq = (
            iqr_y / rmse
            if (
                rmse > 0
                and iqr_y > 1e-12
            )
            else np.nan
        )

    else:
        r2 = np.nan
        rpd = np.nan
        rpiq = np.nan

    return {
        "N": n,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "RPD": rpd,
        "RPIQ": rpiq,
        "Bias": bias,
    }


# =============================================================================
# MOISTURE CLASSIFICATION
# =============================================================================

def classify_moisture(value):

    if pd.isna(value):
        return np.nan

    value = float(value)

    if value < 0:
        return np.nan

    if value < 10:
        return "Dry"

    if value <= 30:
        return "Moist"

    return "Wet"


# =============================================================================
# CLUSTER BOOTSTRAP
# =============================================================================

def physical_soil_cluster_bootstrap(
    subset,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):
    """
    Resample physical soils with replacement.

    Every time a physical soil is selected, all image observations belonging
    to that soil within the moisture class are included.

    This preserves within-soil dependence among repeated images.
    """

    rng = np.random.default_rng(
        seed
    )

    soils = np.array(
        sorted(
            subset[GROUP_COL]
            .dropna()
            .unique()
        )
    )

    if len(soils) < 2:
        return pd.DataFrame()

    bootstrap_records = []

    for bootstrap_id in range(
        n_bootstrap
    ):

        sampled_soils = rng.choice(
            soils,
            size=len(soils),
            replace=True,
        )

        pieces = []

        for replicate_id, soil in enumerate(
            sampled_soils
        ):

            temp = subset[
                subset[GROUP_COL] == soil
            ].copy()

            # Gives duplicated bootstrap clusters unique identities.
            temp["_bootstrap_cluster"] = (
                replicate_id
            )

            pieces.append(temp)

        boot = pd.concat(
            pieces,
            ignore_index=True,
        )

        metrics = regression_metrics(
            boot[TARGET_COL],
            boot[PRED_COL],
        )

        metrics["Bootstrap_ID"] = (
            bootstrap_id + 1
        )

        bootstrap_records.append(
            metrics
        )

    return pd.DataFrame(
        bootstrap_records
    )


def summarize_bootstrap(
    subset,
    moisture_class,
):

    observed = regression_metrics(
        subset[TARGET_COL],
        subset[PRED_COL],
    )

    boot = physical_soil_cluster_bootstrap(
        subset
    )

    summary_records = []

    for metric in [
        "R2",
        "RMSE",
        "MAE",
        "RPD",
        "RPIQ",
        "Bias",
    ]:

        if (
            not boot.empty
            and metric in boot.columns
        ):

            values = (
                boot[metric]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

        else:
            values = pd.Series(
                dtype=float
            )

        if len(values) > 0:

            ci_lower = np.percentile(
                values,
                2.5,
            )

            ci_upper = np.percentile(
                values,
                97.5,
            )

        else:

            ci_lower = np.nan
            ci_upper = np.nan

        summary_records.append(
            {
                "Moisture_Class":
                    moisture_class,
                "Metric":
                    metric,
                "Estimate":
                    observed[metric],
                "CI_Lower":
                    ci_lower,
                "CI_Upper":
                    ci_upper,
                "Valid_Bootstrap_Replicates":
                    len(values),
            }
        )

    return pd.DataFrame(
        summary_records
    )


# =============================================================================
# EXACT PAIRED SIGN-FLIP TEST
# =============================================================================

def exact_sign_flip_test(differences):
    """
    Exact paired randomization/sign-flip test.

    Used only as a supplementary analysis on physical-soil-level
    error differences.

    With four soils, there are only 2^4 = 16 possible sign patterns,
    so statistical power is necessarily very limited.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:
        return np.nan, np.nan

    observed = np.mean(
        differences
    )

    absolute_differences = np.abs(
        differences
    )

    permutation_means = []

    for signs in itertools.product(
        [-1, 1],
        repeat=len(differences),
    ):

        signs = np.asarray(
            signs
        )

        permuted = (
            signs
            * absolute_differences
        )

        permutation_means.append(
            np.mean(permuted)
        )

    permutation_means = np.asarray(
        permutation_means
    )

    p_value = np.mean(
        np.abs(permutation_means)
        >= abs(observed)
    )

    return observed, p_value


# =============================================================================
# LOAD EXTERNAL VALIDATION PREDICTIONS
# =============================================================================

print("\n" + "=" * 90)
print(
    "OBJECTIVE 2 — MOISTURE EFFECT "
    "ON SOC PREDICTION"
)
print("=" * 90)

if not os.path.exists(
    PREDICTION_FILE
):
    raise FileNotFoundError(
        f"\nPrediction file not found:\n"
        f"{PREDICTION_FILE}"
    )

df = pd.read_csv(
    PREDICTION_FILE
)

print(
    f"\nExternal-validation images: "
    f"{len(df)}"
)

print(
    f"Physical validation soils : "
    f"{df[GROUP_COL].nunique()}"
)

print(
    f"Physical soils            : "
    f"{sorted(df[GROUP_COL].unique())}"
)


# =============================================================================
# VALIDATE REQUIRED COLUMNS
# =============================================================================

required_columns = [
    GROUP_COL,
    IMAGE_COL,
    MOISTURE_COL,
    TARGET_COL,
    PRED_COL,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"\nMissing required columns: "
        f"{missing_columns}\n\n"
        f"Available columns:\n"
        f"{df.columns.tolist()}"
    )


# =============================================================================
# CLEAN NUMERIC VARIABLES
# =============================================================================

for column in [
    GROUP_COL,
    IMAGE_COL,
    MOISTURE_COL,
    TARGET_COL,
    PRED_COL,
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


before = len(df)

df = df.dropna(
    subset=[
        GROUP_COL,
        MOISTURE_COL,
        TARGET_COL,
        PRED_COL,
    ]
).copy()

after = len(df)

print(
    f"\nRows removed because of missing "
    f"essential values: {before - after}"
)

print(
    f"Usable validation images: {after}"
)


# =============================================================================
# ASSIGN MOISTURE CLASS
# =============================================================================

df["Moisture_Class"] = (
    df[MOISTURE_COL]
    .apply(classify_moisture)
)

df = df.dropna(
    subset=["Moisture_Class"]
).copy()

df["Absolute_Error"] = np.abs(
    df[PRED_COL]
    - df[TARGET_COL]
)

df["Squared_Error"] = (
    df[PRED_COL]
    - df[TARGET_COL]
) ** 2

df["Prediction_Error"] = (
    df[PRED_COL]
    - df[TARGET_COL]
)


# =============================================================================
# MOISTURE DISTRIBUTION AUDIT
# =============================================================================

print("\n" + "=" * 90)
print(
    "1. MOISTURE-CLASS DISTRIBUTION"
)
print("=" * 90)

distribution = (
    df.groupby(
        "Moisture_Class",
        as_index=False,
    )
    .agg(
        N_Images=(
            IMAGE_COL,
            "size",
        ),
        N_Physical_Soils=(
            GROUP_COL,
            "nunique",
        ),
        Mean_Moisture=(
            MOISTURE_COL,
            "mean",
        ),
        Min_Moisture=(
            MOISTURE_COL,
            "min",
        ),
        Max_Moisture=(
            MOISTURE_COL,
            "max",
        ),
        Mean_SOC=(
            TARGET_COL,
            "mean",
        ),
        Min_SOC=(
            TARGET_COL,
            "min",
        ),
        Max_SOC=(
            TARGET_COL,
            "max",
        ),
    )
)

distribution[
    "Moisture_Class"
] = pd.Categorical(
    distribution[
        "Moisture_Class"
    ],
    categories=MOISTURE_ORDER,
    ordered=True,
)

distribution = (
    distribution
    .sort_values(
        "Moisture_Class"
    )
    .reset_index(
        drop=True
    )
)

print(
    distribution.to_string(
        index=False
    )
)


# =============================================================================
# SAMPLE x MOISTURE COUNTS
# =============================================================================

print("\n" + "=" * 90)
print(
    "2. PHYSICAL-SOIL REPRESENTATION "
    "WITHIN EACH MOISTURE CLASS"
)
print("=" * 90)

sample_moisture_counts = (
    df.groupby(
        [
            GROUP_COL,
            "Moisture_Class",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

sample_moisture_counts = (
    sample_moisture_counts
    .reindex(
        columns=MOISTURE_ORDER,
        fill_value=0,
    )
)

print(
    sample_moisture_counts
)


# =============================================================================
# IMAGE-LEVEL PERFORMANCE
# =============================================================================

print("\n" + "=" * 90)
print(
    "3. IMAGE-LEVEL PERFORMANCE "
    "BY MOISTURE CLASS"
)
print("=" * 90)

image_level_records = []

for moisture_class in MOISTURE_ORDER:

    subset = df[
        df["Moisture_Class"]
        == moisture_class
    ].copy()

    if subset.empty:
        continue

    metrics = regression_metrics(
        subset[TARGET_COL],
        subset[PRED_COL],
    )

    image_level_records.append(
        {
            "Moisture_Class":
                moisture_class,
            "N_Images":
                len(subset),
            "N_Physical_Soils":
                subset[
                    GROUP_COL
                ].nunique(),
            "Mean_Observed_SOC":
                subset[
                    TARGET_COL
                ].mean(),
            "Mean_Predicted_SOC":
                subset[
                    PRED_COL
                ].mean(),
            **{
                key: value
                for key, value
                in metrics.items()
                if key != "N"
            },
        }
    )

image_performance = pd.DataFrame(
    image_level_records
)

print(
    image_performance.to_string(
        index=False
    )
)


# =============================================================================
# PHYSICAL-SOIL-LEVEL ERROR SUMMARIES
# =============================================================================
#
# This is especially important because a soil with many photographs should not
# automatically dominate the moisture comparison.
# =============================================================================

print("\n" + "=" * 90)
print(
    "4. EQUAL-SOIL ERROR SUMMARY"
)
print("=" * 90)

soil_error = (
    df.groupby(
        [
            GROUP_COL,
            "Moisture_Class",
        ],
        as_index=False,
    )
    .agg(
        N_Images=(
            IMAGE_COL,
            "size",
        ),
        Mean_Observed_SOC=(
            TARGET_COL,
            "mean",
        ),
        Mean_Predicted_SOC=(
            PRED_COL,
            "mean",
        ),
        Soil_MAE=(
            "Absolute_Error",
            "mean",
        ),
        Soil_MSE=(
            "Squared_Error",
            "mean",
        ),
        Soil_Bias=(
            "Prediction_Error",
            "mean",
        ),
    )
)

soil_error[
    "Soil_RMSE"
] = np.sqrt(
    soil_error[
        "Soil_MSE"
    ]
)

soil_error[
    "Moisture_Class"
] = pd.Categorical(
    soil_error[
        "Moisture_Class"
    ],
    categories=MOISTURE_ORDER,
    ordered=True,
)

soil_error = (
    soil_error
    .sort_values(
        [
            GROUP_COL,
            "Moisture_Class",
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    soil_error.to_string(
        index=False
    )
)


# =============================================================================
# EQUAL-WEIGHT PHYSICAL-SOIL MOISTURE SUMMARY
# =============================================================================

soil_equal_weight_summary = (
    soil_error.groupby(
        "Moisture_Class",
        observed=True,
        as_index=False,
    )
    .agg(
        N_Physical_Soils=(
            GROUP_COL,
            "nunique",
        ),
        Mean_Soil_MAE=(
            "Soil_MAE",
            "mean",
        ),
        SD_Soil_MAE=(
            "Soil_MAE",
            "std",
        ),
        Mean_Soil_RMSE=(
            "Soil_RMSE",
            "mean",
        ),
        Mean_Soil_Bias=(
            "Soil_Bias",
            "mean",
        ),
    )
)

print("\nEqual-weight soil summary:")

print(
    soil_equal_weight_summary.to_string(
        index=False
    )
)


# =============================================================================
# PHYSICAL-SOIL CLUSTER BOOTSTRAP CIs
# =============================================================================

print("\n" + "=" * 90)
print(
    "5. PHYSICAL-SOIL CLUSTER-BOOTSTRAP "
    "95% CONFIDENCE INTERVALS"
)
print("=" * 90)

bootstrap_outputs = []

for i, moisture_class in enumerate(
    MOISTURE_ORDER
):

    subset = df[
        df["Moisture_Class"]
        == moisture_class
    ].copy()

    if subset.empty:
        continue

    boot_summary = summarize_bootstrap(
        subset,
        moisture_class,
    )

    bootstrap_outputs.append(
        boot_summary
    )

bootstrap_summary_df = pd.concat(
    bootstrap_outputs,
    ignore_index=True,
)

print(
    bootstrap_summary_df.to_string(
        index=False
    )
)


# =============================================================================
# CALIBRATION-MEAN BASELINE
# =============================================================================

print("\n" + "=" * 90)
print(
    "6. CALIBRATION-MEAN BASELINE"
)
print("=" * 90)

if os.path.exists(
    CALIBRATION_FILE
):

    calibration_df = pd.read_csv(
        CALIBRATION_FILE
    )

    if TARGET_COL in calibration_df.columns:

        calibration_mean_soc = (
            pd.to_numeric(
                calibration_df[
                    TARGET_COL
                ],
                errors="coerce",
            )
            .dropna()
            .mean()
        )

        print(
            f"Calibration mean SOC = "
            f"{calibration_mean_soc:.4f}"
        )

        baseline_records = []

        for moisture_class in MOISTURE_ORDER:

            subset = df[
                df["Moisture_Class"]
                == moisture_class
            ].copy()

            if subset.empty:
                continue

            baseline_predictions = (
                np.full(
                    len(subset),
                    calibration_mean_soc,
                    dtype=float,
                )
            )

            model_metrics = regression_metrics(
                subset[TARGET_COL],
                subset[PRED_COL],
            )

            baseline_metrics = regression_metrics(
                subset[TARGET_COL],
                baseline_predictions,
            )

            baseline_records.append(
                {
                    "Moisture_Class":
                        moisture_class,

                    "Model_RMSE":
                        model_metrics["RMSE"],

                    "Baseline_RMSE":
                        baseline_metrics["RMSE"],

                    "Delta_RMSE_ModelMinusBaseline":
                        (
                            model_metrics["RMSE"]
                            - baseline_metrics["RMSE"]
                        ),

                    "Model_MAE":
                        model_metrics["MAE"],

                    "Baseline_MAE":
                        baseline_metrics["MAE"],

                    "Delta_MAE_ModelMinusBaseline":
                        (
                            model_metrics["MAE"]
                            - baseline_metrics["MAE"]
                        ),
                }
            )

        baseline_comparison = (
            pd.DataFrame(
                baseline_records
            )
        )

        print(
            baseline_comparison.to_string(
                index=False
            )
        )

    else:

        calibration_mean_soc = np.nan
        baseline_comparison = (
            pd.DataFrame()
        )

        print(
            "Calibration file does not contain "
            "'soc'. Baseline skipped."
        )

else:

    calibration_mean_soc = np.nan
    baseline_comparison = pd.DataFrame()

    print(
        "Calibration file was not found. "
        "Baseline comparison skipped."
    )


# =============================================================================
# PAIRED PHYSICAL-SOIL COMPARISONS
# =============================================================================
#
# Same four soils appear in each moisture class.
#
# We compare per-soil MAE values, not individual image errors.
#
# Exact sign-flip test is supplementary only.
# With N=4 physical soils, statistical power is extremely limited.
# =============================================================================

print("\n" + "=" * 90)
print(
    "7. PAIRED PHYSICAL-SOIL MOISTURE COMPARISONS"
)
print("=" * 90)

mae_wide = (
    soil_error.pivot(
        index=GROUP_COL,
        columns="Moisture_Class",
        values="Soil_MAE",
    )
)

mae_wide = mae_wide.reindex(
    columns=MOISTURE_ORDER
)

print("\nPer-soil MAE:")
print(mae_wide)

paired_records = []

comparisons = [
    ("Dry", "Moist"),
    ("Dry", "Wet"),
    ("Moist", "Wet"),
]

for class_a, class_b in comparisons:

    paired = (
        mae_wide[
            [class_a, class_b]
        ]
        .dropna()
    )

    differences = (
        paired[class_a]
        - paired[class_b]
    )

    mean_difference, p_value = (
        exact_sign_flip_test(
            differences.values
        )
    )

    paired_records.append(
        {
            "Comparison":
                f"{class_a} - {class_b}",

            "N_Paired_Physical_Soils":
                len(paired),

            "Mean_MAE_First":
                paired[
                    class_a
                ].mean(),

            "Mean_MAE_Second":
                paired[
                    class_b
                ].mean(),

            "Mean_Difference_FirstMinusSecond":
                mean_difference,

            "Exact_SignFlip_P":
                p_value,
        }
    )

paired_comparisons = pd.DataFrame(
    paired_records
)

print("\nSupplementary paired comparisons:")

print(
    paired_comparisons.to_string(
        index=False
    )
)

print(
    "\nNOTE: With only four independent physical soils, "
    "the paired tests have very low statistical power "
    "and should not be used as the primary basis for conclusions."
)


# =============================================================================
# SOIL-CONDITION-LEVEL PERFORMANCE
# =============================================================================
#
# Some physical samples have multiple SOC reference values.
#
# Therefore aggregation is by:
#
#     Sample_No + SOC + Moisture_Class
#
# NOT by Sample_No alone.
# =============================================================================

print("\n" + "=" * 90)
print(
    "8. SOIL-CONDITION-LEVEL PERFORMANCE"
)
print("=" * 90)

condition_rows = []
condition_metric_records = []

for moisture_class in MOISTURE_ORDER:

    subset = df[
        df["Moisture_Class"]
        == moisture_class
    ].copy()

    if subset.empty:
        continue

    condition_df = (
        subset.groupby(
            [
                GROUP_COL,
                TARGET_COL,
            ],
            as_index=False,
        )
        .agg(
            Mean_Moisture=(
                MOISTURE_COL,
                "mean",
            ),
            Predicted_SOC=(
                PRED_COL,
                "mean",
            ),
            N_Images=(
                IMAGE_COL,
                "size",
            ),
        )
    )

    condition_df[
        "Moisture_Class"
    ] = moisture_class

    condition_rows.append(
        condition_df
    )

    condition_metrics = (
        regression_metrics(
            condition_df[
                TARGET_COL
            ],
            condition_df[
                "Predicted_SOC"
            ],
        )
    )

    condition_metric_records.append(
        {
            "Moisture_Class":
                moisture_class,

            "N_Soil_Condition_Rows":
                len(condition_df),

            "N_Physical_Soils":
                condition_df[
                    GROUP_COL
                ].nunique(),

            **{
                key: value
                for key, value
                in condition_metrics.items()
                if key != "N"
            },
        }
    )

condition_predictions = pd.concat(
    condition_rows,
    ignore_index=True,
)

condition_performance = pd.DataFrame(
    condition_metric_records
)

print(
    condition_performance.to_string(
        index=False
    )
)


# =============================================================================
# FIGURE 1 — PER-SOIL MAE ACROSS MOISTURE CONDITIONS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

x_positions = np.arange(
    len(MOISTURE_ORDER)
)

for soil in sorted(
    soil_error[
        GROUP_COL
    ].unique()
):

    temp = (
        soil_error[
            soil_error[
                GROUP_COL
            ] == soil
        ]
        .set_index(
            "Moisture_Class"
        )
        .reindex(
            MOISTURE_ORDER
        )
    )

    ax.plot(
        x_positions,
        temp["Soil_MAE"].values,
        marker="o",
        label=f"Soil {int(soil)}",
    )

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    MOISTURE_ORDER
)

ax.set_xlabel(
    "Soil moisture class"
)

ax.set_ylabel(
    "Mean absolute error (% SOC)"
)

ax.set_title(
    "Prediction error across moisture conditions\n"
    "for completely held-out physical soils"
)

ax.legend(
    title="Physical soil"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "figure_moisture_per_soil_MAE.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 2 — RMSE / MAE BY MOISTURE CLASS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

x = np.arange(
    len(image_performance)
)

width = 0.35

ax.bar(
    x - width / 2,
    image_performance["RMSE"],
    width,
    label="RMSE",
)

ax.bar(
    x + width / 2,
    image_performance["MAE"],
    width,
    label="MAE",
)

ax.set_xticks(x)

ax.set_xticklabels(
    image_performance[
        "Moisture_Class"
    ]
)

ax.set_ylabel(
    "Prediction error (% SOC)"
)

ax.set_xlabel(
    "Soil moisture class"
)

ax.set_title(
    "SOC prediction error by moisture condition"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "figure_moisture_RMSE_MAE.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 3 — OBSERVED VS PREDICTED BY MOISTURE CLASS
# =============================================================================

for moisture_class in MOISTURE_ORDER:

    subset = df[
        df["Moisture_Class"]
        == moisture_class
    ].copy()

    if subset.empty:
        continue

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.scatter(
        subset[TARGET_COL],
        subset[PRED_COL],
        alpha=0.7,
    )

    global_min = min(
        subset[TARGET_COL].min(),
        subset[PRED_COL].min(),
    )

    global_max = max(
        subset[TARGET_COL].max(),
        subset[PRED_COL].max(),
    )

    ax.plot(
        [global_min, global_max],
        [global_min, global_max],
        linestyle="--",
        label="1:1 line",
    )

    # Descriptive regression line
    if (
        subset[
            TARGET_COL
        ].nunique()
        > 1
    ):

        slope, intercept = np.polyfit(
            subset[TARGET_COL],
            subset[PRED_COL],
            1,
        )

        x_line = np.linspace(
            global_min,
            global_max,
            100,
        )

        y_line = (
            slope * x_line
            + intercept
        )

        ax.plot(
            x_line,
            y_line,
            label="Regression line",
        )

    metrics = regression_metrics(
        subset[TARGET_COL],
        subset[PRED_COL],
    )

    ax.set_xlabel(
        "Observed SOC (%)"
    )

    ax.set_ylabel(
        "Predicted SOC (%)"
    )

    ax.set_title(
        f"{moisture_class} conditions\n"
        f"N={len(subset)} images, "
        f"{subset[GROUP_COL].nunique()} soils\n"
        f"RMSE={metrics['RMSE']:.2f}, "
        f"MAE={metrics['MAE']:.2f}, "
        f"R²={metrics['R2']:.2f}"
    )

    ax.legend()

    fig.tight_layout()

    safe_name = (
        moisture_class
        .lower()
        .replace(" ", "_")
    )

    fig.savefig(
        os.path.join(
            OUT_DIR,
            f"figure_observed_predicted_{safe_name}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# PUBLICATION TABLE
# =============================================================================

publication_table = (
    image_performance[
        [
            "Moisture_Class",
            "N_Images",
            "N_Physical_Soils",
            "R2",
            "RMSE",
            "MAE",
            "RPD",
            "RPIQ",
            "Bias",
        ]
    ]
    .copy()
)

for metric in [
    "R2",
    "RMSE",
    "MAE",
    "RPD",
    "RPIQ",
    "Bias",
]:

    metric_ci = (
        bootstrap_summary_df[
            bootstrap_summary_df[
                "Metric"
            ] == metric
        ][
            [
                "Moisture_Class",
                "CI_Lower",
                "CI_Upper",
            ]
        ]
        .rename(
            columns={
                "CI_Lower":
                    f"{metric}_CI_Lower",
                "CI_Upper":
                    f"{metric}_CI_Upper",
            }
        )
    )

    publication_table = (
        publication_table.merge(
            metric_ci,
            on="Moisture_Class",
            how="left",
        )
    )


# =============================================================================
# SAVE ALL OUTPUTS
# =============================================================================

distribution.to_csv(
    os.path.join(
        OUT_DIR,
        "01_moisture_distribution.csv",
    ),
    index=False,
)

sample_moisture_counts.to_csv(
    os.path.join(
        OUT_DIR,
        "02_sample_moisture_counts.csv",
    )
)

image_performance.to_csv(
    os.path.join(
        OUT_DIR,
        "03_image_level_moisture_performance.csv",
    ),
    index=False,
)

soil_error.to_csv(
    os.path.join(
        OUT_DIR,
        "04_per_soil_moisture_errors.csv",
    ),
    index=False,
)

soil_equal_weight_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "05_equal_soil_weight_summary.csv",
    ),
    index=False,
)

bootstrap_summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "06_cluster_bootstrap_confidence_intervals.csv",
    ),
    index=False,
)

paired_comparisons.to_csv(
    os.path.join(
        OUT_DIR,
        "07_paired_soil_moisture_comparisons.csv",
    ),
    index=False,
)

condition_predictions.to_csv(
    os.path.join(
        OUT_DIR,
        "08_soil_condition_predictions.csv",
    ),
    index=False,
)

condition_performance.to_csv(
    os.path.join(
        OUT_DIR,
        "09_soil_condition_moisture_performance.csv",
    ),
    index=False,
)

publication_table.to_csv(
    os.path.join(
        OUT_DIR,
        "10_PUBLICATION_TABLE_moisture_performance.csv",
    ),
    index=False,
)

df.to_csv(
    os.path.join(
        OUT_DIR,
        "11_validation_predictions_with_moisture_classes.csv",
    ),
    index=False,
)

if not baseline_comparison.empty:

    baseline_comparison.to_csv(
        os.path.join(
            OUT_DIR,
            "12_model_vs_calibration_mean_baseline.csv",
        ),
        index=False,
    )


# =============================================================================
# FINAL CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print(
    "PUBLICATION TABLE — MOISTURE PERFORMANCE"
)
print("=" * 90)

print(
    publication_table.to_string(
        index=False
    )
)

print("\n" + "=" * 90)
print(
    "OBJECTIVE 2 MOISTURE ANALYSIS COMPLETE"
)
print("=" * 90)

print(
    f"\nOutputs saved to:\n"
    f"{OUT_DIR}"
)

print(
    "\nPrimary files to review:"
)

print(
    "  01_moisture_distribution.csv"
)

print(
    "  03_image_level_moisture_performance.csv"
)

print(
    "  04_per_soil_moisture_errors.csv"
)

print(
    "  06_cluster_bootstrap_confidence_intervals.csv"
)

print(
    "  07_paired_soil_moisture_comparisons.csv"
)

print(
    "  09_soil_condition_moisture_performance.csv"
)

print(
    "  10_PUBLICATION_TABLE_moisture_performance.csv"
)

print(
    "\nFigures:"
)

print(
    "  figure_moisture_per_soil_MAE.png"
)

print(
    "  figure_moisture_RMSE_MAE.png"
)

print(
    "  figure_observed_predicted_dry.png"
)

print(
    "  figure_observed_predicted_moist.png"
)

print(
    "  figure_observed_predicted_wet.png"
)