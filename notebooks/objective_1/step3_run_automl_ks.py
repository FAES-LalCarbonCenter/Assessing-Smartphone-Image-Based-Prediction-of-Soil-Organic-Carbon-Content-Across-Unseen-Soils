"""
KS STEP 3 — AutoGluon model selection with grouped internal CV
             + composite re-ranking
===============================================================

This script mirrors the sample-grouped AutoML logic as closely as possible.

Workflow
--------
1. Load KS Step-2 calibration and validation datasets.
2. Use 4-fold GroupKFold by Sample_No within the KS calibration set.
3. AutoGluon trains/ranks models using RMSE.
4. Evaluate every model on the grouped tuning fold with:
   RMSE, MAE, R², RPD, RPIQ.
5. Aggregate metrics across all 4 folds.
6. Re-rank models using:
   0.35*RPD + 0.25*RPIQ + 0.20*RMSE + 0.15*R² + 0.05*MAE,
   after min-max normalization, with RMSE and MAE inverted.
7. Lock the selected model before touching KS external validation.
8. Refit AutoGluon on the full KS calibration dataset.
9. Evaluate the locked model on the untouched KS validation set.

Important:
- Physical-soil overlap between KS calibration and validation is expected.
- External validation is never used for model selection or re-ranking.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

CALIB_CSV = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step2_output_ks",
    "calibration_selected_features.csv",
)

VALID_CSV = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step2_output_ks",
    "validation_selected_features.csv",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step3_output_ks",
)
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL = "soc"
GROUP_COL = "Sample_No"

NON_PREDICTOR_COLS = [
    "image_no",
    "Image_No",
    "moisture",
    "soil_type",
    "Sample_No",
]

N_GROUP_FOLDS = 4
FOLD_TIME_LIMIT = 600
FINAL_TIME_LIMIT = 1800
PRESETS = "medium_quality"

WEIGHTS = {
    "RPD": 0.35,
    "RPIQ": 0.25,
    "RMSE": 0.20,
    "R2": 0.15,
    "MAE": 0.05,
}


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    sd_y = np.std(y_true, ddof=1) if len(y_true) > 1 else np.nan
    rpd = sd_y / rmse if rmse > 0 and np.isfinite(sd_y) else np.nan

    q75, q25 = np.percentile(y_true, [75, 25])
    rpiq = (q75 - q25) / rmse if rmse > 0 else np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "RPD": rpd,
        "RPIQ": rpiq,
    }


def normalize_higher_is_better(series):
    s = pd.to_numeric(series, errors="coerce")
    finite = np.isfinite(s)
    out = pd.Series(np.nan, index=s.index, dtype=float)

    if finite.sum() == 0:
        return out

    lo = s[finite].min()
    hi = s[finite].max()

    if np.isclose(hi, lo):
        out.loc[finite] = 1.0
    else:
        out.loc[finite] = (s[finite] - lo) / (hi - lo)

    return out


def normalize_lower_is_better(series):
    return 1.0 - normalize_higher_is_better(series)


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

    return df.sort_values(
        ["Composite_Score", "Mean_RMSE"],
        ascending=[False, True],
    ).reset_index(drop=True)


print("\n" + "=" * 80)
print("KS STEP 3 — AutoGluon + COMPOSITE RE-RANKING")
print("=" * 80)

cal_full = pd.read_csv(CALIB_CSV)
val_full = pd.read_csv(VALID_CSV)

if TARGET_COL not in cal_full.columns or TARGET_COL not in val_full.columns:
    raise ValueError(f"Target column '{TARGET_COL}' must exist in both files.")

if GROUP_COL not in cal_full.columns or GROUP_COL not in val_full.columns:
    raise ValueError(f"'{GROUP_COL}' must exist in both files.")

cal_groups = set(cal_full[GROUP_COL].dropna().unique())
val_groups = set(val_full[GROUP_COL].dropna().unique())
external_overlap = cal_groups & val_groups

print(f"\nCalibration: {len(cal_full)} images from {len(cal_groups)} physical soils")
print(f"Validation : {len(val_full)} images from {len(val_groups)} physical soils")
print(f"Physical soils on BOTH sides: {len(external_overlap)}")
print(f"Overlap samples: {sorted(external_overlap)}")

if external_overlap:
    print("EXPECTED FOR KS IMAGE-LEVEL VALIDATION: sample dependence is present.")
else:
    print("WARNING: no physical-sample overlap detected.")

cal_model = cal_full.drop(
    columns=[c for c in NON_PREDICTOR_COLS if c in cal_full.columns]
).copy()

val_model = val_full.drop(
    columns=[c for c in NON_PREDICTOR_COLS if c in val_full.columns]
).copy()

FEATURES = [c for c in cal_model.columns if c != TARGET_COL]

if not FEATURES:
    raise ValueError("No predictor columns found.")

missing_in_validation = [f for f in FEATURES if f not in val_model.columns]
if missing_in_validation:
    raise ValueError(f"Validation missing predictors: {missing_in_validation}")

cal_model = cal_model[[TARGET_COL] + FEATURES].copy()
val_model = val_model[[TARGET_COL] + FEATURES].copy()

print(f"\nPredictors ({len(FEATURES)}): {FEATURES}")


print("\n" + "=" * 80)
print("4-FOLD GROUP-AWARE INTERNAL MODEL EVALUATION")
print("=" * 80)

gkf = GroupKFold(n_splits=N_GROUP_FOLDS)
groups = cal_full[GROUP_COL].values

all_model_results = []
fold_best_rmse_models = []

for fold, (fit_idx, tune_idx) in enumerate(
    gkf.split(cal_full, cal_full[TARGET_COL], groups=groups),
    start=1,
):
    print("\n" + "-" * 80)
    print(f"GROUPED CV FOLD {fold}")
    print("-" * 80)

    fold_train_full = cal_full.iloc[fit_idx].copy()
    fold_tune_full = cal_full.iloc[tune_idx].copy()

    train_groups = set(fold_train_full[GROUP_COL].dropna().unique())
    tune_groups = set(fold_tune_full[GROUP_COL].dropna().unique())
    overlap = train_groups & tune_groups

    if overlap:
        raise ValueError(f"Internal grouped-CV leakage in fold {fold}: {sorted(overlap)}")

    print(f"Train soils ({len(train_groups)}): {sorted(train_groups)}")
    print(f"Tune soils  ({len(tune_groups)}): {sorted(tune_groups)}")
    print("Internal fold sample overlap: ZERO")

    fold_train = fold_train_full[[TARGET_COL] + FEATURES].copy()
    fold_tune = fold_tune_full[[TARGET_COL] + FEATURES].copy()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fold_path = os.path.join(
        OUT_DIR,
        "grouped_cv_models",
        f"fold_{fold}_{run_id}",
    )
    os.makedirs(os.path.dirname(fold_path), exist_ok=True)

    predictor_fold = TabularPredictor(
        label=TARGET_COL,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=fold_path,
    ).fit(
        train_data=fold_train,
        tuning_data=fold_tune,
        time_limit=FOLD_TIME_LIMIT,
        presets=PRESETS,
        verbosity=1,
    )

    fold_best_rmse_models.append({
        "Fold": fold,
        "AutoGluon_RMSE_Best_Model": predictor_fold.model_best,
    })

    y_true_fold = fold_tune[TARGET_COL].values

    for model_name in predictor_fold.model_names():
        try:
            y_pred_fold = predictor_fold.predict(
                fold_tune,
                model=model_name,
            )

            m = regression_metrics(y_true_fold, y_pred_fold)

            all_model_results.append({
                "Fold": fold,
                "Model": model_name,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "R2": m["R2"],
                "RPD": m["RPD"],
                "RPIQ": m["RPIQ"],
                "N_Train_Images": len(fold_train),
                "N_Tune_Images": len(fold_tune),
                "N_Train_Groups": len(train_groups),
                "N_Tune_Groups": len(tune_groups),
            })

        except Exception as exc:
            print(f"Could not evaluate model '{model_name}': {exc}")


cv_results = pd.DataFrame(all_model_results)

if cv_results.empty:
    raise RuntimeError("No AutoGluon models were successfully evaluated.")

cv_results.to_csv(
    os.path.join(OUT_DIR, "grouped_cv_all_models_all_folds.csv"),
    index=False,
)

pd.DataFrame(fold_best_rmse_models).to_csv(
    os.path.join(OUT_DIR, "grouped_cv_autogluon_rmse_winners.csv"),
    index=False,
)

model_summary = (
    cv_results
    .groupby("Model", as_index=False)
    .agg(
        Mean_RMSE=("RMSE", "mean"),
        SD_RMSE=("RMSE", "std"),
        Mean_MAE=("MAE", "mean"),
        SD_MAE=("MAE", "std"),
        Mean_R2=("R2", "mean"),
        SD_R2=("R2", "std"),
        Mean_RPD=("RPD", "mean"),
        SD_RPD=("RPD", "std"),
        Mean_RPIQ=("RPIQ", "mean"),
        SD_RPIQ=("RPIQ", "std"),
        Folds_Evaluated=("Fold", "nunique"),
    )
)

model_summary = model_summary[
    model_summary["Folds_Evaluated"] == N_GROUP_FOLDS
].copy()

if model_summary.empty:
    raise RuntimeError("No model was evaluated successfully in all folds.")

model_summary["RMSE_Rank"] = (
    model_summary["Mean_RMSE"]
    .rank(method="min", ascending=True)
    .astype(int)
)

model_summary = add_composite_score(model_summary)

model_summary.insert(
    0,
    "Composite_Rank",
    np.arange(1, len(model_summary) + 1),
)

model_summary.to_csv(
    os.path.join(OUT_DIR, "grouped_cv_model_composite_ranking.csv"),
    index=False,
)

print("\n" + "=" * 80)
print("KS CALIBRATION — COMPOSITE RE-RANKING")
print("=" * 80)

display_cols = [
    "Composite_Rank",
    "RMSE_Rank",
    "Model",
    "Composite_Score",
    "Mean_RMSE",
    "Mean_MAE",
    "Mean_R2",
    "Mean_RPD",
    "Mean_RPIQ",
]

print(
    model_summary[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

selected_model_name = model_summary.iloc[0]["Model"]

print("\n" + "=" * 80)
print("MODEL LOCKED BEFORE KS EXTERNAL VALIDATION")
print("=" * 80)
print(f"Selected model: {selected_model_name}")
print("KS validation was NOT used for model selection or re-ranking.")


print("\n" + "=" * 80)
print("FINAL FIT ON ALL KS CALIBRATION IMAGES")
print("=" * 80)

final_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
final_path = os.path.join(
    OUT_DIR,
    "final_full_calibration",
    f"run_{final_run_id}",
)

final_predictor = TabularPredictor(
    label=TARGET_COL,
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    path=final_path,
).fit(
    train_data=cal_model,
    time_limit=FINAL_TIME_LIMIT,
    presets=PRESETS,
    verbosity=2,
)

available_final_models = final_predictor.model_names()

if selected_model_name not in available_final_models:
    raise RuntimeError(
        "Grouped-CV-selected model was not produced in final KS run.\n"
        f"Selected model: {selected_model_name}\n"
        f"Available models: {available_final_models}"
    )

print(f"\nAutoGluon native RMSE-best model: {final_predictor.model_best}")
print(f"Model used for KS validation: {selected_model_name}")


print("\n" + "=" * 80)
print("KS EXTERNAL VALIDATION — IMAGE LEVEL")
print("=" * 80)

y_test = val_model[TARGET_COL].values
y_pred = final_predictor.predict(
    val_model,
    model=selected_model_name,
)

image_metrics = regression_metrics(y_test, y_pred)

print(f"Validation images: {len(val_model)}")
print(f"Physical soils represented: {val_full[GROUP_COL].nunique()}")
print(f"Physical soils overlapping calibration: {len(external_overlap)}")
print(f"R²   = {image_metrics['R2']:.4f}")
print(f"RMSE = {image_metrics['RMSE']:.4f}")
print(f"MAE  = {image_metrics['MAE']:.4f}")
print(f"RPD  = {image_metrics['RPD']:.4f}")
print(f"RPIQ = {image_metrics['RPIQ']:.4f}")


condition_eval = val_full.copy()
condition_eval["Predicted_SOC"] = np.asarray(y_pred)

condition_results = (
    condition_eval
    .groupby([GROUP_COL, TARGET_COL], as_index=False)
    .agg(
        Predicted_SOC=("Predicted_SOC", "mean"),
        N_Images=("Predicted_SOC", "size"),
    )
    .rename(columns={TARGET_COL: "Actual_SOC"})
)

condition_metrics = regression_metrics(
    condition_results["Actual_SOC"].values,
    condition_results["Predicted_SOC"].values,
)

print("\n" + "=" * 80)
print("KS VALIDATION — SOIL-CONDITION LEVEL")
print("=" * 80)
print(condition_results.to_string(index=False))
print(f"\nSoil-condition rows = {len(condition_results)}")
print(f"R²   = {condition_metrics['R2']:.4f}")
print(f"RMSE = {condition_metrics['RMSE']:.4f}")
print(f"MAE  = {condition_metrics['MAE']:.4f}")
print(f"RPD  = {condition_metrics['RPD']:.4f}")
print(f"RPIQ = {condition_metrics['RPIQ']:.4f}")


baseline_value = cal_model[TARGET_COL].mean()
baseline_pred = np.full(len(y_test), baseline_value, dtype=float)
baseline_metrics = regression_metrics(y_test, baseline_pred)

print("\n" + "=" * 80)
print("KS CALIBRATION-MEAN BASELINE")
print("=" * 80)
print(f"Calibration mean SOC = {baseline_value:.4f}")
print(f"Baseline R²   = {baseline_metrics['R2']:.4f}")
print(f"Baseline RMSE = {baseline_metrics['RMSE']:.4f}")
print(f"Baseline MAE  = {baseline_metrics['MAE']:.4f}")
print(f"Baseline RPD  = {baseline_metrics['RPD']:.4f}")
print(f"Baseline RPIQ = {baseline_metrics['RPIQ']:.4f}")


predictions = val_full.copy()
predictions["Predicted_SOC"] = np.asarray(y_pred)

predictions.to_csv(
    os.path.join(OUT_DIR, "external_validation_predictions.csv"),
    index=False,
)

condition_results.to_csv(
    os.path.join(
        OUT_DIR,
        "external_validation_soil_condition_results.csv",
    ),
    index=False,
)

final_results = pd.DataFrame([{
    "Validation_Design": "Kennard-Stone image-level split",
    "Selected_Model": selected_model_name,
    "Selection_Method": (
        "4-fold GroupKFold by Sample_No within KS calibration; "
        "AutoGluon trained with RMSE; models re-ranked by normalized composite"
    ),
    "Calibration_Images": len(cal_full),
    "Calibration_Physical_Soils": cal_full[GROUP_COL].nunique(),
    "Validation_Images": len(val_full),
    "Validation_Physical_Soils": val_full[GROUP_COL].nunique(),
    "Physical_Soils_Overlapping": len(external_overlap),
    "Image_R2": image_metrics["R2"],
    "Image_RMSE": image_metrics["RMSE"],
    "Image_MAE": image_metrics["MAE"],
    "Image_RPD": image_metrics["RPD"],
    "Image_RPIQ": image_metrics["RPIQ"],
    "SoilCondition_R2": condition_metrics["R2"],
    "SoilCondition_RMSE": condition_metrics["RMSE"],
    "SoilCondition_MAE": condition_metrics["MAE"],
    "SoilCondition_RPD": condition_metrics["RPD"],
    "SoilCondition_RPIQ": condition_metrics["RPIQ"],
    "Baseline_R2": baseline_metrics["R2"],
    "Baseline_RMSE": baseline_metrics["RMSE"],
    "Baseline_MAE": baseline_metrics["MAE"],
}])

final_results.to_csv(
    os.path.join(OUT_DIR, "FINAL_KS_RESULTS.csv"),
    index=False,
)

print("\n" + "=" * 80)
print("KS STEP 3 COMPLETE")
print("=" * 80)
print(f"Calibration images used: {len(cal_full)}")
print(f"Validation images used: {len(val_full)}")
print(f"Physical soils overlapping both subsets: {len(external_overlap)}")
print(f"Selected model: {selected_model_name}")
print(f"Outputs saved to: {OUT_DIR}")
