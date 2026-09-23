"""
OBJECTIVE 3 — FIELD VS LABORATORY SOC PREDICTION
=================================================

Purpose
-------
Compare SOC prediction performance between field-acquired and
laboratory-acquired images for physical soils excluded entirely
from model calibration.

IMPORTANT
---------
1. Uses the same 141 sample-grouped external-validation predictions
   generated in Objective 1.
2. No model retraining is performed.
3. No Kennard-Stone comparison is performed here.
4. Acquisition setting is assigned from image_no using the verified
   image-number ranges used in the original project.
5. Field/Lab setting is used only for post-prediction stratification.
6. Repeated images from the same physical soil are not treated as
   independent for uncertainty estimation.
7. Physical-soil cluster bootstrap is used for 95% confidence intervals.
8. Equal-soil summaries are reported so soils with many photographs
   do not dominate the comparison.

FIELD IMAGE RANGES
------------------
1–19
206–268
361–436

LABORATORY IMAGE RANGES
-----------------------
20–205
269–360
437–731
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
    "objective_3",
    "output_data",
    "field_lab_analysis",
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
IMAGE_COL = "image_no"
SETTING_COL = "Acquisition_Setting"

SETTING_ORDER = [
    "Field",
    "Laboratory",
]

N_BOOTSTRAP = 5000
SEED = 42


# =============================================================================
# FIELD / LAB CLASSIFICATION
# =============================================================================

def classify_acquisition_setting(image_no):
    """
    Assign acquisition setting using verified image-number ranges.

    Field:
        1–19
        206–268
        361–436

    Laboratory:
        20–205
        269–360
        437–731
    """

    if pd.isna(image_no):
        return np.nan

    image_no = int(image_no)

    if (
        1 <= image_no <= 19
        or 206 <= image_no <= 268
        or 361 <= image_no <= 436
    ):
        return "Field"

    elif (
        20 <= image_no <= 205
        or 269 <= image_no <= 360
        or 437 <= image_no <= 731
    ):
        return "Laboratory"

    return "Unknown"


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def regression_metrics(y_true, y_pred):

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
# PHYSICAL-SOIL CLUSTER BOOTSTRAP
# =============================================================================

def physical_soil_cluster_bootstrap(
    subset,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):

    rng = np.random.default_rng(seed)

    soils = np.array(
        sorted(
            subset[GROUP_COL]
            .dropna()
            .unique()
        )
    )

    if len(soils) < 2:
        return pd.DataFrame()

    records = []

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

        records.append(metrics)

    return pd.DataFrame(
        records
    )


def summarize_bootstrap(
    subset,
    setting,
):

    observed = regression_metrics(
        subset[TARGET_COL],
        subset[PRED_COL],
    )

    boot = (
        physical_soil_cluster_bootstrap(
            subset
        )
    )

    output = []

    for metric in [
        "R2",
        "RMSE",
        "MAE",
        "RPD",
        "RPIQ",
        "Bias",
    ]:

        if not boot.empty:

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

        output.append(
            {
                "Acquisition_Setting":
                    setting,

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
        output
    )


# =============================================================================
# EXACT PAIRED SIGN-FLIP TEST
# =============================================================================

def exact_sign_flip_test(differences):

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

        permutation_means.append(
            np.mean(
                signs
                * absolute_differences
            )
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
    "OBJECTIVE 3 — FIELD VS LABORATORY "
    "SOC PREDICTION"
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


# =============================================================================
# VALIDATE REQUIRED COLUMNS
# =============================================================================

required_columns = [
    IMAGE_COL,
    GROUP_COL,
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
# CLEAN VARIABLES
# =============================================================================

for column in [
    IMAGE_COL,
    GROUP_COL,
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
        IMAGE_COL,
        GROUP_COL,
        TARGET_COL,
        PRED_COL,
    ]
).copy()

print(
    f"\nExternal-validation images : "
    f"{len(df)}"
)

print(
    f"Rows removed for missing essential values: "
    f"{before - len(df)}"
)

print(
    f"Physical validation soils  : "
    f"{df[GROUP_COL].nunique()}"
)

print(
    f"Validation soils           : "
    f"{sorted(df[GROUP_COL].unique())}"
)


# =============================================================================
# ASSIGN FIELD / LAB LABEL
# =============================================================================

df[SETTING_COL] = (
    df[IMAGE_COL]
    .apply(
        classify_acquisition_setting
    )
)

unknown = (
    df[
        df[SETTING_COL]
        == "Unknown"
    ]
)

if len(unknown) > 0:

    raise ValueError(
        f"\n{len(unknown)} validation images "
        f"could not be classified as Field or Laboratory.\n"
        f"Image numbers:\n"
        f"{unknown[IMAGE_COL].tolist()}"
    )


# =============================================================================
# FIELD/LAB DISTRIBUTION
# =============================================================================

print("\n" + "=" * 90)
print(
    "1. FIELD / LABORATORY DISTRIBUTION"
)
print("=" * 90)

distribution = (
    df.groupby(
        SETTING_COL,
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
    SETTING_COL
] = pd.Categorical(
    distribution[
        SETTING_COL
    ],
    categories=SETTING_ORDER,
    ordered=True,
)

distribution = (
    distribution
    .sort_values(
        SETTING_COL
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
# SOIL REPRESENTATION BY SETTING
# =============================================================================

print("\n" + "=" * 90)
print(
    "2. PHYSICAL-SOIL REPRESENTATION "
    "BY ACQUISITION SETTING"
)
print("=" * 90)

soil_setting_counts = (
    df.groupby(
        [
            GROUP_COL,
            SETTING_COL,
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

soil_setting_counts = (
    soil_setting_counts
    .reindex(
        columns=SETTING_ORDER,
        fill_value=0,
    )
)

print(
    soil_setting_counts
)


# =============================================================================
# ERROR COLUMNS
# =============================================================================

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
# IMAGE-LEVEL PERFORMANCE
# =============================================================================

print("\n" + "=" * 90)
print(
    "3. IMAGE-LEVEL PERFORMANCE "
    "BY ACQUISITION SETTING"
)
print("=" * 90)

image_records = []

for setting in SETTING_ORDER:

    subset = df[
        df[SETTING_COL]
        == setting
    ].copy()

    if subset.empty:
        continue

    metrics = regression_metrics(
        subset[TARGET_COL],
        subset[PRED_COL],
    )

    image_records.append(
        {
            "Acquisition_Setting":
                setting,

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
    image_records
)

print(
    image_performance.to_string(
        index=False
    )
)


# =============================================================================
# EQUAL-SOIL ERROR SUMMARY
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
            SETTING_COL,
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
    SETTING_COL
] = pd.Categorical(
    soil_error[
        SETTING_COL
    ],
    categories=SETTING_ORDER,
    ordered=True,
)

soil_error = (
    soil_error
    .sort_values(
        [
            GROUP_COL,
            SETTING_COL,
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


equal_soil_summary = (
    soil_error.groupby(
        SETTING_COL,
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

print(
    "\nEqual-weight soil summary:"
)

print(
    equal_soil_summary.to_string(
        index=False
    )
)


# =============================================================================
# CLUSTER-BOOTSTRAP CIs
# =============================================================================

print("\n" + "=" * 90)
print(
    "5. PHYSICAL-SOIL CLUSTER-BOOTSTRAP "
    "95% CONFIDENCE INTERVALS"
)
print("=" * 90)

bootstrap_outputs = []

for setting in SETTING_ORDER:

    subset = df[
        df[SETTING_COL]
        == setting
    ].copy()

    if subset.empty:
        continue

    bootstrap_outputs.append(
        summarize_bootstrap(
            subset,
            setting,
        )
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

baseline_comparison = pd.DataFrame()

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

        for setting in SETTING_ORDER:

            subset = df[
                df[SETTING_COL]
                == setting
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

            model_metrics = (
                regression_metrics(
                    subset[
                        TARGET_COL
                    ],
                    subset[
                        PRED_COL
                    ],
                )
            )

            baseline_metrics = (
                regression_metrics(
                    subset[
                        TARGET_COL
                    ],
                    baseline_predictions,
                )
            )

            baseline_records.append(
                {
                    "Acquisition_Setting":
                        setting,

                    "Model_RMSE":
                        model_metrics[
                            "RMSE"
                        ],

                    "Baseline_RMSE":
                        baseline_metrics[
                            "RMSE"
                        ],

                    "Delta_RMSE_ModelMinusBaseline":
                        (
                            model_metrics[
                                "RMSE"
                            ]
                            - baseline_metrics[
                                "RMSE"
                            ]
                        ),

                    "Model_MAE":
                        model_metrics[
                            "MAE"
                        ],

                    "Baseline_MAE":
                        baseline_metrics[
                            "MAE"
                        ],

                    "Delta_MAE_ModelMinusBaseline":
                        (
                            model_metrics[
                                "MAE"
                            ]
                            - baseline_metrics[
                                "MAE"
                            ]
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


# =============================================================================
# PAIRED FIELD VS LAB COMPARISON
# =============================================================================

print("\n" + "=" * 90)
print(
    "7. PAIRED PHYSICAL-SOIL FIELD VS LAB COMPARISON"
)
print("=" * 90)

mae_wide = (
    soil_error.pivot(
        index=GROUP_COL,
        columns=SETTING_COL,
        values="Soil_MAE",
    )
)

mae_wide = mae_wide.reindex(
    columns=SETTING_ORDER
)

print(
    "\nPer-soil MAE:"
)

print(
    mae_wide
)

paired = (
    mae_wide[
        [
            "Field",
            "Laboratory",
        ]
    ]
    .dropna()
)

if len(paired) > 0:

    differences = (
        paired["Field"]
        - paired["Laboratory"]
    )

    mean_difference, p_value = (
        exact_sign_flip_test(
            differences.values
        )
    )

    paired_comparison = pd.DataFrame(
        [
            {
                "Comparison":
                    "Field - Laboratory",

                "N_Paired_Physical_Soils":
                    len(paired),

                "Mean_Field_MAE":
                    paired[
                        "Field"
                    ].mean(),

                "Mean_Laboratory_MAE":
                    paired[
                        "Laboratory"
                    ].mean(),

                "Mean_Difference_FieldMinusLab":
                    mean_difference,

                "Exact_SignFlip_P":
                    p_value,
            }
        ]
    )

else:

    paired_comparison = pd.DataFrame(
        [
            {
                "Comparison":
                    "Field - Laboratory",

                "N_Paired_Physical_Soils":
                    0,

                "Mean_Field_MAE":
                    np.nan,

                "Mean_Laboratory_MAE":
                    np.nan,

                "Mean_Difference_FieldMinusLab":
                    np.nan,

                "Exact_SignFlip_P":
                    np.nan,
            }
        ]
    )

print(
    paired_comparison.to_string(
        index=False
    )
)

print(
    "\nNOTE: Statistical inference is limited by "
    "the small number of independent physical soils."
)


# =============================================================================
# SOIL-CONDITION-LEVEL PERFORMANCE
# =============================================================================

print("\n" + "=" * 90)
print(
    "8. SOIL-CONDITION-LEVEL PERFORMANCE"
)
print("=" * 90)

condition_rows = []
condition_metric_records = []

for setting in SETTING_ORDER:

    subset = df[
        df[SETTING_COL]
        == setting
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
        SETTING_COL
    ] = setting

    condition_rows.append(
        condition_df
    )

    metrics = regression_metrics(
        condition_df[
            TARGET_COL
        ],
        condition_df[
            "Predicted_SOC"
        ],
    )

    condition_metric_records.append(
        {
            "Acquisition_Setting":
                setting,

            "N_Soil_Condition_Rows":
                len(
                    condition_df
                ),

            "N_Physical_Soils":
                condition_df[
                    GROUP_COL
                ].nunique(),

            **{
                key: value
                for key, value
                in metrics.items()
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
# FIGURE 1 — PER-SOIL MAE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

x_positions = np.arange(
    len(SETTING_ORDER)
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
            SETTING_COL
        )
        .reindex(
            SETTING_ORDER
        )
    )

    ax.plot(
        x_positions,
        temp[
            "Soil_MAE"
        ].values,
        marker="o",
        label=f"Soil {int(soil)}",
    )

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    SETTING_ORDER
)

ax.set_ylabel(
    "Mean absolute error (% SOC)"
)

ax.set_xlabel(
    "Image acquisition setting"
)

ax.set_title(
    "Prediction error under field and laboratory conditions\n"
    "for completely held-out physical soils"
)

ax.legend(
    title="Physical soil"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "figure_field_lab_per_soil_MAE.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 2 — RMSE / MAE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

x = np.arange(
    len(
        image_performance
    )
)

width = 0.35

ax.bar(
    x - width / 2,
    image_performance[
        "RMSE"
    ],
    width,
    label="RMSE",
)

ax.bar(
    x + width / 2,
    image_performance[
        "MAE"
    ],
    width,
    label="MAE",
)

ax.set_xticks(x)

ax.set_xticklabels(
    image_performance[
        "Acquisition_Setting"
    ]
)

ax.set_ylabel(
    "Prediction error (% SOC)"
)

ax.set_xlabel(
    "Image acquisition setting"
)

ax.set_title(
    "SOC prediction error under field and laboratory conditions"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "figure_field_lab_RMSE_MAE.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 3 — OBSERVED VS PREDICTED
# =============================================================================

for setting in SETTING_ORDER:

    subset = df[
        df[SETTING_COL]
        == setting
    ].copy()

    if subset.empty:
        continue

    metrics = regression_metrics(
        subset[
            TARGET_COL
        ],
        subset[
            PRED_COL
        ],
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.scatter(
        subset[
            TARGET_COL
        ],
        subset[
            PRED_COL
        ],
        alpha=0.7,
    )

    global_min = min(
        subset[
            TARGET_COL
        ].min(),
        subset[
            PRED_COL
        ].min(),
    )

    global_max = max(
        subset[
            TARGET_COL
        ].max(),
        subset[
            PRED_COL
        ].max(),
    )

    ax.plot(
        [
            global_min,
            global_max,
        ],
        [
            global_min,
            global_max,
        ],
        linestyle="--",
        label="1:1 line",
    )

    if (
        subset[
            TARGET_COL
        ].nunique()
        > 1
    ):

        slope, intercept = np.polyfit(
            subset[
                TARGET_COL
            ],
            subset[
                PRED_COL
            ],
            1,
        )

        x_line = np.linspace(
            global_min,
            global_max,
            100,
        )

        ax.plot(
            x_line,
            slope * x_line + intercept,
            label="Regression line",
        )

    ax.set_xlabel(
        "Observed SOC (%)"
    )

    ax.set_ylabel(
        "Predicted SOC (%)"
    )

    ax.set_title(
        f"{setting} images\n"
        f"N={len(subset)} images, "
        f"{subset[GROUP_COL].nunique()} soils\n"
        f"RMSE={metrics['RMSE']:.2f}, "
        f"MAE={metrics['MAE']:.2f}, "
        f"R²={metrics['R2']:.2f}"
    )

    ax.legend()

    fig.tight_layout()

    safe_name = (
        setting
        .lower()
        .replace(
            " ",
            "_",
        )
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
            "Acquisition_Setting",
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
                "Acquisition_Setting",
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
            on="Acquisition_Setting",
            how="left",
        )
    )


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

distribution.to_csv(
    os.path.join(
        OUT_DIR,
        "01_field_lab_distribution.csv",
    ),
    index=False,
)

soil_setting_counts.to_csv(
    os.path.join(
        OUT_DIR,
        "02_sample_field_lab_counts.csv",
    )
)

image_performance.to_csv(
    os.path.join(
        OUT_DIR,
        "03_image_level_field_lab_performance.csv",
    ),
    index=False,
)

soil_error.to_csv(
    os.path.join(
        OUT_DIR,
        "04_per_soil_field_lab_errors.csv",
    ),
    index=False,
)

equal_soil_summary.to_csv(
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

paired_comparison.to_csv(
    os.path.join(
        OUT_DIR,
        "07_paired_field_lab_comparison.csv",
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
        "09_soil_condition_field_lab_performance.csv",
    ),
    index=False,
)

publication_table.to_csv(
    os.path.join(
        OUT_DIR,
        "10_PUBLICATION_TABLE_field_lab_performance.csv",
    ),
    index=False,
)

df.to_csv(
    os.path.join(
        OUT_DIR,
        "11_external_validation_predictions_with_setting.csv",
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
    "PUBLICATION TABLE — FIELD VS LABORATORY"
)
print("=" * 90)

print(
    publication_table.to_string(
        index=False
    )
)

print("\n" + "=" * 90)
print(
    "OBJECTIVE 3 FIELD/LAB ANALYSIS COMPLETE"
)
print("=" * 90)

print(
    f"\nOutputs saved to:\n"
    f"{OUT_DIR}"
)

print(
    "\nPrimary files:"
)

print(
    "  01_field_lab_distribution.csv"
)

print(
    "  03_image_level_field_lab_performance.csv"
)

print(
    "  04_per_soil_field_lab_errors.csv"
)

print(
    "  06_cluster_bootstrap_confidence_intervals.csv"
)

print(
    "  07_paired_field_lab_comparison.csv"
)

print(
    "  09_soil_condition_field_lab_performance.csv"
)

print(
    "  10_PUBLICATION_TABLE_field_lab_performance.csv"
)

print(
    "\nFigures:"
)

print(
    "  figure_field_lab_per_soil_MAE.png"
)

print(
    "  figure_field_lab_RMSE_MAE.png"
)

print(
    "  figure_observed_predicted_field.png"
)

print(
    "  figure_observed_predicted_laboratory.png"
)