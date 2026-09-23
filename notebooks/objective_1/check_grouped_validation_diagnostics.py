import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed'

# Calibration data used for modeling
CALIB_CSV = os.path.join(
    PROJECT_ROOT,
    'notebooks',
    'objective_1',
    'output_data',
    'step2_output_threshold',
    'calibration_selected_features.csv'
)

# Validation data with predictions produced by Step 3
PREDICTIONS_CSV = os.path.join(
    PROJECT_ROOT,
    'notebooks',
    'objective_1',
    'output_data',
    'step3_output_threshold',
    'validation_predictions.csv'
)

# Optional: original validation dataset, useful for checking groups
VALIDATION_CSV = os.path.join(
    PROJECT_ROOT,
    'notebooks',
    'objective_1',
    'output_data',
    'step2_output_threshold',
    'validation_selected_features.csv'
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    'notebooks',
    'objective_1',
    'output_data',
    'validation_diagnostics_sensitivity_threshold'
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("GROUPED VALIDATION DIAGNOSTICS")
print("=" * 80)

calib = pd.read_csv(CALIB_CSV)
pred = pd.read_csv(PREDICTIONS_CSV)
valid = pd.read_csv(VALIDATION_CSV)

print(f"\nCalibration rows : {len(calib)}")
print(f"Validation rows  : {len(valid)}")
print(f"Prediction rows  : {len(pred)}")

print("\nPrediction columns:")
print(pred.columns.tolist())


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_pred_cols = [
    'Sample_No',
    'soc',
    'Predicted_SOC'
]

missing_cols = [
    c for c in required_pred_cols
    if c not in pred.columns
]

if missing_cols:
    raise ValueError(
        f"Missing required columns in validation_predictions.csv: "
        f"{missing_cols}"
    )


# ============================================================
# 1. CHECK SAMPLE OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("1. CHECKING SAMPLE OVERLAP")
print("=" * 80)

calib_samples = set(
    pd.to_numeric(
        calib['Sample_No'],
        errors='coerce'
    ).dropna().astype(int)
)

valid_samples = set(
    pd.to_numeric(
        valid['Sample_No'],
        errors='coerce'
    ).dropna().astype(int)
)

overlap = calib_samples.intersection(valid_samples)

print(f"\nCalibration Sample_No: {sorted(calib_samples)}")
print(f"Validation Sample_No : {sorted(valid_samples)}")

if len(overlap) == 0:
    print("\nPASS: ZERO physical sample overlap.")
else:
    print("\nWARNING: SAMPLE LEAKAGE DETECTED!")
    print("Overlapping Sample_No:", sorted(overlap))


# ============================================================
# 2. CHECK SOC RANGE
# ============================================================

print("\n" + "=" * 80)
print("2. CHECKING CALIBRATION VS VALIDATION SOC RANGE")
print("=" * 80)

calib_min = calib['soc'].min()
calib_max = calib['soc'].max()

valid_min = valid['soc'].min()
valid_max = valid['soc'].max()

print(
    f"\nCalibration SOC range: "
    f"{calib_min:.3f} to {calib_max:.3f}"
)

print(
    f"Validation SOC range : "
    f"{valid_min:.3f} to {valid_max:.3f}"
)

if valid_min >= calib_min and valid_max <= calib_max:
    print(
        "\nPASS: Validation SOC values are inside "
        "the calibration SOC range."
    )
else:
    print(
        "\nWARNING: Some validation SOC values fall outside "
        "the calibration SOC range."
    )
    print(
        "The model is being asked to extrapolate "
        "for at least part of the validation set."
    )


# ============================================================
# 3. PER-SAMPLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("3. PER-SAMPLE ERROR STRUCTURE")
print("=" * 80)

pred['Residual'] = (
    pred['Predicted_SOC'] - pred['soc']
)

sample_summary = (
    pred
    .groupby('Sample_No')
    .agg(
        Actual_SOC=('soc', 'mean'),
        Mean_Predicted_SOC=('Predicted_SOC', 'mean'),
        Median_Predicted_SOC=('Predicted_SOC', 'median'),
        Prediction_SD=('Predicted_SOC', 'std'),
        Min_Prediction=('Predicted_SOC', 'min'),
        Max_Prediction=('Predicted_SOC', 'max'),
        Mean_Error=('Residual', 'mean'),
        Mean_Absolute_Error=('Residual',
                             lambda x: np.mean(np.abs(x))),
        N_Images=('Predicted_SOC', 'size')
    )
    .reset_index()
)

sample_summary['Absolute_Sample_Error'] = np.abs(
    sample_summary['Mean_Predicted_SOC']
    - sample_summary['Actual_SOC']
)

sample_summary = sample_summary.sort_values(
    'Actual_SOC'
).reset_index(drop=True)

print("\nPer-sample results:\n")

print(
    sample_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

summary_path = os.path.join(
    OUT_DIR,
    'per_sample_validation_summary.csv'
)

sample_summary.to_csv(
    summary_path,
    index=False
)

print(
    f"\nSaved: {summary_path}"
)


# ============================================================
# 4. CHECK WHETHER SAMPLE RANKING IS CORRECT
# ============================================================

print("\n" + "=" * 80)
print("4. CHECKING SOC RANKING BETWEEN SAMPLES")
print("=" * 80)

if len(sample_summary) >= 3:

    rank_rho, rank_p = spearmanr(
        sample_summary['Actual_SOC'],
        sample_summary['Mean_Predicted_SOC']
    )

    print(
        f"\nSpearman correlation between actual SOC "
        f"and sample-level predicted SOC:"
    )

    print(f"rho = {rank_rho:.4f}")
    print(f"p   = {rank_p:.4f}")

    print("\nActual SOC ranking:")

    actual_order = sample_summary.sort_values(
        'Actual_SOC'
    )[['Sample_No', 'Actual_SOC']]

    print(
        actual_order.to_string(
            index=False
        )
    )

    print("\nPredicted SOC ranking:")

    predicted_order = sample_summary.sort_values(
        'Mean_Predicted_SOC'
    )[
        ['Sample_No',
         'Mean_Predicted_SOC',
         'Actual_SOC']
    ]

    print(
        predicted_order.to_string(
            index=False
        )
    )

else:
    print(
        "\nToo few validation samples for a meaningful "
        "ranking diagnostic."
    )


# ============================================================
# 5. WITHIN-SAMPLE PREDICTION STABILITY
# ============================================================

print("\n" + "=" * 80)
print("5. CHECKING WITHIN-SAMPLE PREDICTION STABILITY")
print("=" * 80)

print(
    "\nPrediction SD tells us how much repeated images "
    "from the same physical sample disagree."
)

stability_table = sample_summary[
    [
        'Sample_No',
        'Actual_SOC',
        'Mean_Predicted_SOC',
        'Prediction_SD',
        'Min_Prediction',
        'Max_Prediction',
        'N_Images'
    ]
]

print(
    stability_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# PLOT 1:
# IMAGE-LEVEL ACTUAL VS PREDICTED,
# COLORED BY SAMPLE
# ============================================================

plt.figure(figsize=(8, 7))

samples = sorted(
    pred['Sample_No'].dropna().unique()
)

for sample in samples:

    temp = pred[
        pred['Sample_No'] == sample
    ]

    plt.scatter(
        temp['soc'],
        temp['Predicted_SOC'],
        label=f"Sample {int(sample)}",
        alpha=0.7
    )

all_values = np.concatenate([
    pred['soc'].values,
    pred['Predicted_SOC'].values
])

plot_min = np.nanmin(all_values)
plot_max = np.nanmax(all_values)

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    '--',
    linewidth=1.5,
    label='1:1 line'
)

plt.xlabel('Actual SOC')
plt.ylabel('Predicted SOC')

plt.title(
    'Image-Level Validation Predictions\n'
    'Colored by Physical Sample'
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        '01_actual_vs_predicted_by_sample.png'
    ),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# PLOT 2:
# SAMPLE-MEAN ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 7))

plt.errorbar(
    sample_summary['Actual_SOC'],
    sample_summary['Mean_Predicted_SOC'],
    yerr=sample_summary['Prediction_SD'],
    fmt='o',
    capsize=5
)

for _, row in sample_summary.iterrows():

    plt.annotate(
        f"S{int(row['Sample_No'])}",
        (
            row['Actual_SOC'],
            row['Mean_Predicted_SOC']
        ),
        xytext=(5, 5),
        textcoords='offset points'
    )

all_sample_values = np.concatenate([
    sample_summary['Actual_SOC'].values,
    sample_summary['Mean_Predicted_SOC'].values
])

plot_min = np.nanmin(all_sample_values)
plot_max = np.nanmax(all_sample_values)

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    '--'
)

plt.xlabel('Actual SOC')
plt.ylabel('Mean Predicted SOC')

plt.title(
    'Physical-Sample Level Validation\n'
    'Error bars = SD across repeated images'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        '02_sample_level_actual_vs_predicted.png'
    ),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# PLOT 3:
# RESIDUALS BY SAMPLE
# ============================================================

plt.figure(figsize=(9, 6))

sample_positions = {
    sample: i
    for i, sample in enumerate(samples)
}

for sample in samples:

    temp = pred[
        pred['Sample_No'] == sample
    ]

    x = np.full(
        len(temp),
        sample_positions[sample],
        dtype=float
    )

    # Add small horizontal jitter only for visibility
    jitter = np.random.default_rng(42).normal(
        0,
        0.05,
        size=len(temp)
    )

    plt.scatter(
        x + jitter,
        temp['Residual'],
        alpha=0.6
    )

plt.axhline(
    0,
    linestyle='--'
)

plt.xticks(
    range(len(samples)),
    [
        f"S{int(s)}"
        for s in samples
    ]
)

plt.xlabel('Physical Sample')
plt.ylabel('Residual (Predicted SOC - Actual SOC)')

plt.title(
    'Prediction Residuals by Physical Sample'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        '03_residuals_by_sample.png'
    ),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# PLOT 4:
# CALIBRATION VS VALIDATION SOC DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    calib['soc'],
    bins=15,
    alpha=0.6,
    label='Calibration'
)

plt.hist(
    valid['soc'],
    bins=15,
    alpha=0.6,
    label='Validation'
)

plt.xlabel('SOC')
plt.ylabel('Number of observations')

plt.title(
    'Calibration vs Validation SOC Distribution'
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        '04_calibration_validation_soc_distribution.png'
    ),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# SIMPLE INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC INTERPRETATION")
print("=" * 80)

mean_within_sd = sample_summary[
    'Prediction_SD'
].mean()

actual_range = (
    sample_summary['Actual_SOC'].max()
    - sample_summary['Actual_SOC'].min()
)

predicted_range = (
    sample_summary['Mean_Predicted_SOC'].max()
    - sample_summary['Mean_Predicted_SOC'].min()
)

print(
    f"\nValidation actual SOC range: "
    f"{actual_range:.4f}"
)

print(
    f"Predicted sample-mean range: "
    f"{predicted_range:.4f}"
)

print(
    f"Mean within-sample prediction SD: "
    f"{mean_within_sd:.4f}"
)

if predicted_range < actual_range:

    print(
        "\nObservation: Predicted SOC range is narrower "
        "than the actual SOC range."
    )

    print(
        "This suggests some degree of prediction-range "
        "compression / regression toward the mean."
    )

if len(sample_summary) >= 3:

    if rank_rho >= 0.8:

        print(
            "\nSample ranking: strong agreement between "
            "actual and predicted sample order."
        )

    elif rank_rho >= 0.4:

        print(
            "\nSample ranking: moderate agreement."
        )

    elif rank_rho > 0:

        print(
            "\nSample ranking: weak positive agreement."
        )

    else:

        print(
            "\nSample ranking: poor or reversed ordering "
            "between actual and predicted SOC."
        )

print(
    "\nIMPORTANT: With only a few held-out physical samples, "
    "these diagnostics should be interpreted descriptively "
    "rather than as definitive statistical tests."
)

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)

print(
    f"\nAll diagnostic outputs saved to:\n{OUT_DIR}"
)