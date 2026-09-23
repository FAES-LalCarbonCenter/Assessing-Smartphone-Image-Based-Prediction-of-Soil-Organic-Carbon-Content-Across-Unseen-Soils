
import os
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

PREDICTIONS_CSV = os.path.join(
    PROJECT_ROOT,
    "notebooks","objective_1","output_data",
    "step3_output_ks","external_validation_predictions.csv"
)

CALIBRATION_CSV = os.path.join(
    PROJECT_ROOT,
    "notebooks","objective_1","output_data",
    "step2_output_ks","calibration_selected_features.csv"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "notebooks","objective_1","output_data",
    "step4_statistical_validation_image_ks"
)

os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "soc"
PRED = "Predicted_SOC"
GROUP = "Sample_No"

N_BOOT = 5000
CI_LEVEL = 0.95
SEED = 42

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    sd_y = np.std(y_true, ddof=1) if len(y_true) > 1 else np.nan
    rpd = sd_y / rmse if rmse > 0 and np.isfinite(sd_y) else np.nan

    q75, q25 = np.percentile(y_true, [75, 25])
    rpiq = (q75 - q25) / rmse if rmse > 0 else np.nan

    bias = np.mean(y_pred - y_true)

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "RPD": rpd,
        "RPIQ": rpiq,
        "Bias": bias,
    }

def percentile_ci(values, ci=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    alpha = (1 - ci) / 2
    return (
        np.percentile(values, 100 * alpha),
        np.percentile(values, 100 * (1 - alpha)),
    )

def cluster_bootstrap(df, target_col, pred_col, group_col, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(df[group_col].dropna().unique()))

    records = []

    for b in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)

        pieces = []
        for draw_id, g in enumerate(sampled_groups):
            temp = df.loc[df[group_col] == g].copy()
            temp["_bootstrap_cluster"] = draw_id
            pieces.append(temp)

        boot = pd.concat(pieces, ignore_index=True)

        m = regression_metrics(
            boot[target_col].values,
            boot[pred_col].values,
        )
        m["Bootstrap"] = b + 1
        records.append(m)

    return pd.DataFrame(records)

def summarize_bootstrap(point_metrics, boot_df, ci=0.95):
    rows = []

    for metric_name in ["R2","RMSE","MAE","RPD","RPIQ","Bias"]:
        lower, upper = percentile_ci(boot_df[metric_name], ci)

        rows.append({
            "Metric": metric_name,
            "Estimate": point_metrics[metric_name],
            "CI_Lower": lower,
            "CI_Upper": upper,
            "Valid_Bootstrap_Replicates": int(
                np.isfinite(boot_df[metric_name]).sum()
            ),
        })

    return pd.DataFrame(rows)

print("\n" + "="*80)
print("KS STEP 4 — STATISTICAL VALIDATION")
print("="*80)

pred_df = pd.read_csv(PREDICTIONS_CSV)
cal_df = pd.read_csv(CALIBRATION_CSV)

required = [TARGET, PRED, GROUP]
missing = [c for c in required if c not in pred_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

physical_soils = sorted(pred_df[GROUP].dropna().unique())

print(f"\nValidation images: {len(pred_df)}")
print(f"Physical soils represented: {len(physical_soils)}")
print(f"Samples: {physical_soils}")
print("\nNOTE: KS validation contains new images of soils already represented in calibration.")

# 1. Image-level cluster bootstrap
print("\n" + "="*80)
print("1. IMAGE-LEVEL METRICS — CLUSTER-BOOTSTRAP 95% CIs")
print("="*80)

image_point = regression_metrics(pred_df[TARGET], pred_df[PRED])

image_boot = cluster_bootstrap(
    pred_df, TARGET, PRED, GROUP,
    n_boot=N_BOOT, seed=SEED
)

image_summary = summarize_bootstrap(
    image_point, image_boot, CI_LEVEL
)

print(image_summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

image_boot.to_csv(
    os.path.join(OUT_DIR, "image_level_cluster_bootstrap_all.csv"),
    index=False
)

image_summary.to_csv(
    os.path.join(OUT_DIR, "image_level_cluster_bootstrap_summary.csv"),
    index=False
)

# 2. Soil-condition level
print("\n" + "="*80)
print("2. SOIL-CONDITION-LEVEL METRICS — CLUSTER-BOOTSTRAP 95% CIs")
print("="*80)

condition_df = (
    pred_df
    .groupby([GROUP, TARGET], as_index=False)
    .agg(
        Predicted_SOC=(PRED, "mean"),
        N_Images=(PRED, "size"),
    )
)

condition_point = regression_metrics(
    condition_df[TARGET],
    condition_df[PRED]
)

condition_boot = cluster_bootstrap(
    condition_df, TARGET, PRED, GROUP,
    n_boot=N_BOOT, seed=SEED+1
)

condition_summary = summarize_bootstrap(
    condition_point, condition_boot, CI_LEVEL
)

print("\nSoil-condition observations:")
print(condition_df.to_string(index=False))

print("\nBootstrap summary:")
print(condition_summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

condition_df.to_csv(
    os.path.join(OUT_DIR, "soil_condition_predictions.csv"),
    index=False
)

condition_summary.to_csv(
    os.path.join(OUT_DIR, "soil_condition_cluster_bootstrap_summary.csv"),
    index=False
)

# 3. Calibration-mean baseline
print("\n" + "="*80)
print("3. MODEL VS CALIBRATION-MEAN BASELINE")
print("="*80)

calibration_mean = float(cal_df[TARGET].mean())
pred_df["Baseline_Prediction"] = calibration_mean

baseline_point = regression_metrics(
    pred_df[TARGET],
    pred_df["Baseline_Prediction"]
)

print(f"Calibration mean SOC = {calibration_mean:.4f}")
print(f"Model RMSE           = {image_point['RMSE']:.4f}")
print(f"Baseline RMSE        = {baseline_point['RMSE']:.4f}")
print(f"Model MAE            = {image_point['MAE']:.4f}")
print(f"Baseline MAE         = {baseline_point['MAE']:.4f}")

# 4. Cluster-bootstrap difference vs baseline
print("\n" + "="*80)
print("4. CLUSTER-BOOTSTRAP ERROR DIFFERENCE VS BASELINE")
print("="*80)

rng = np.random.default_rng(SEED+2)
groups = np.array(physical_soils)

difference_records = []

for b in range(N_BOOT):
    sampled_groups = rng.choice(groups, size=len(groups), replace=True)

    pieces = []
    for draw_id, g in enumerate(sampled_groups):
        temp = pred_df.loc[pred_df[GROUP] == g].copy()
        temp["_bootstrap_cluster"] = draw_id
        pieces.append(temp)

    boot = pd.concat(pieces, ignore_index=True)

    model_m = regression_metrics(boot[TARGET], boot[PRED])
    base_m = regression_metrics(boot[TARGET], boot["Baseline_Prediction"])

    difference_records.append({
        "Bootstrap": b+1,
        "Delta_RMSE_ModelMinusBaseline":
            model_m["RMSE"] - base_m["RMSE"],
        "Delta_MAE_ModelMinusBaseline":
            model_m["MAE"] - base_m["MAE"],
    })

difference_boot = pd.DataFrame(difference_records)

rmse_lower, rmse_upper = percentile_ci(
    difference_boot["Delta_RMSE_ModelMinusBaseline"], CI_LEVEL
)
mae_lower, mae_upper = percentile_ci(
    difference_boot["Delta_MAE_ModelMinusBaseline"], CI_LEVEL
)

comparison_summary = pd.DataFrame([
    {
        "Comparison": "RMSE: model - calibration mean baseline",
        "Observed_Difference": image_point["RMSE"] - baseline_point["RMSE"],
        "CI_Lower": rmse_lower,
        "CI_Upper": rmse_upper,
    },
    {
        "Comparison": "MAE: model - calibration mean baseline",
        "Observed_Difference": image_point["MAE"] - baseline_point["MAE"],
        "CI_Lower": mae_lower,
        "CI_Upper": mae_upper,
    },
])

print(comparison_summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

comparison_summary.to_csv(
    os.path.join(OUT_DIR, "baseline_comparison_summary.csv"),
    index=False
)

difference_boot.to_csv(
    os.path.join(OUT_DIR, "baseline_cluster_bootstrap_differences.csv"),
    index=False
)

# 5. Physical-soil-level paired Wilcoxon tests
print("\n" + "="*80)
print("5. PHYSICAL-SOIL-LEVEL PAIRED WILCOXON TESTS")
print("="*80)

soil_records = []

for sample_no, d in pred_df.groupby(GROUP):
    actual = d[TARGET].values
    model_pred = d[PRED].values
    baseline_pred = d["Baseline_Prediction"].values

    soil_records.append({
        GROUP: sample_no,
        "N_Images": len(d),
        "Model_MSE": mean_squared_error(actual, model_pred),
        "Baseline_MSE": mean_squared_error(actual, baseline_pred),
        "Model_MAE": mean_absolute_error(actual, model_pred),
        "Baseline_MAE": mean_absolute_error(actual, baseline_pred),
    })

soil_loss = pd.DataFrame(soil_records)

soil_loss["MSE_Difference_ModelMinusBaseline"] = (
    soil_loss["Model_MSE"] - soil_loss["Baseline_MSE"]
)

soil_loss["MAE_Difference_ModelMinusBaseline"] = (
    soil_loss["Model_MAE"] - soil_loss["Baseline_MAE"]
)

print("\nPer-soil error comparison:")
print(soil_loss.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

mse_test = wilcoxon(
    soil_loss["Model_MSE"],
    soil_loss["Baseline_MSE"],
    alternative="two-sided",
    zero_method="wilcox",
)

mae_test = wilcoxon(
    soil_loss["Model_MAE"],
    soil_loss["Baseline_MAE"],
    alternative="two-sided",
    zero_method="wilcox",
)

wilcoxon_summary = pd.DataFrame([
    {
        "Comparison": "Per-soil MSE: model vs calibration-mean baseline",
        "N_Physical_Soils": len(soil_loss),
        "Wilcoxon_Statistic": mse_test.statistic,
        "Two_Sided_P": mse_test.pvalue,
    },
    {
        "Comparison": "Per-soil MAE: model vs calibration-mean baseline",
        "N_Physical_Soils": len(soil_loss),
        "Wilcoxon_Statistic": mae_test.statistic,
        "Two_Sided_P": mae_test.pvalue,
    },
])

print("\nPaired soil-level Wilcoxon results:")
print(wilcoxon_summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}"
))

soil_loss.to_csv(
    os.path.join(OUT_DIR, "per_soil_model_vs_baseline_error.csv"),
    index=False
)

wilcoxon_summary.to_csv(
    os.path.join(OUT_DIR, "soil_level_wilcoxon_results.csv"),
    index=False
)

print("\n" + "="*80)
print("KS STEP 4 COMPLETE")
print("="*80)
print("Primary statistical validation:")
print("  1. Physical-soil cluster-bootstrap 95% confidence intervals")
print("  2. Cluster-bootstrap model-vs-baseline error differences")
print("  3. Paired Wilcoxon tests using one error summary per physical soil")
print("\nNot performed:")
print("  Image-level Friedman/Wilcoxon tests treating repeated images as independent.")
print(f"\nOutputs saved to:\n{OUT_DIR}")
print("="*80)
