

"""
# """
# AutoML - Revised Grouped Validation Approach
# ============================================
# - AutoML trains on 489 calibration images from 15 soil samples
# - Final validation uses 169 images from 5 completely held-out soil samples
# - Image ID, moisture, soil type, and sample ID are retained only as metadata

# """
# """

# import os
# import pandas as pd
# import numpy as np
# from autogluon.tabular import TabularPredictor
# from datetime import datetime
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import GroupKFold


# # Paths
# HERE = os.path.dirname(os.path.abspath(__file__))
# CALIB_CSV = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output\calibration_selected_features.csv'
# VALID_CSV  = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output\validation_selected_features.csv'
# OUT_DIR    = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step3_output'

# os.makedirs(OUT_DIR, exist_ok=True)

# TARGET_COL = 'soc'
# GROUP = "Sample_No"
    

# print("\n" + "="*80)
# print("AUTOML - REVISION GROUPED-SAMPLE VALIDATION")
# print("="*80)
# print("  Revised grouped-sample validation")

# # Load data
# print("\nLoading data...")
# train_df = pd.read_csv(CALIB_CSV)
# test_df = pd.read_csv(VALID_CSV)
# # Keep metadata for traceability, but do NOT use it for modeling
# NON_PREDICTOR_COLS = ['image_no', 'moisture', 'soil_type', 'Sample_No']

# train_df_full = train_df.copy()
# test_df_full = test_df.copy()

# train_df = train_df.drop(
#     columns=[c for c in NON_PREDICTOR_COLS if c in train_df.columns]
# )

# test_df = test_df.drop(
#     columns=[c for c in NON_PREDICTOR_COLS if c in test_df.columns]
# )

# print("  Modeling columns:", train_df.columns.tolist())


# print(f"  Calibration (for AutoML): {train_df.shape}")
# print(f"  Validation (held out): {test_df.shape}")

# # =====================================================================
# # GROUP-AWARE INTERNAL MODEL SELECTION
# # =====================================================================

# print("\n" + "=" * 80)
# print("GROUP-AWARE INTERNAL MODEL SELECTION")
# print("=" * 80)

# FEATURES = [
#     c for c in train_df.columns
#     if c != TARGET_COL
# ]

# print(f"Selected predictors: {FEATURES}")

# # GroupKFold works on the full calibration dataset,
# # where Sample_No is still available.
# groups = train_df_full[GROUP].values

# gkf = GroupKFold(n_splits=4)

# fold_results = []
# all_model_results = []

# for fold, (fit_idx, tune_idx) in enumerate(
#     gkf.split(
#         train_df_full,
#         train_df_full[TARGET_COL],
#         groups=groups
#     ),
#     start=1
# ):

#     print("\n" + "=" * 80)
#     print(f"GROUPED CV FOLD {fold}")
#     print("=" * 80)

#     fold_train_full = train_df_full.iloc[fit_idx].copy()
#     fold_tune_full = train_df_full.iloc[tune_idx].copy()

#     train_groups = set(fold_train_full[GROUP])
#     tune_groups = set(fold_tune_full[GROUP])

#     overlap = train_groups & tune_groups

#     if overlap:
#         raise ValueError(
#             f"Group leakage in fold {fold}: {sorted(overlap)}"
#         )

#     print(
#         f"Training soils ({len(train_groups)}): "
#         f"{sorted(train_groups)}"
#     )

#     print(
#         f"Tuning soils ({len(tune_groups)}): "
#         f"{sorted(tune_groups)}"
#     )

#     print("Sample overlap: ZERO")

#     # Remove Sample_No and metadata before AutoGluon
#     fold_train = fold_train_full[
#         [TARGET_COL] + FEATURES
#     ].copy()

#     fold_tune = fold_tune_full[
#         [TARGET_COL] + FEATURES
#     ].copy()

#     run_id = datetime.now().strftime(
#         "%Y%m%d_%H%M%S_%f"
#     )

#     fold_path = os.path.join(
#         OUT_DIR,
#         f"groupcv_fold_{fold}_{run_id}"
#     )

#     predictor_fold = TabularPredictor(
#         label=TARGET_COL,
#         problem_type="regression",
#         eval_metric="root_mean_squared_error",
#         path=fold_path
#     ).fit(
#         train_data=fold_train,

#         # THIS is the important part:
#         # AutoGluon does NOT create a random tuning split.
#         tuning_data=fold_tune,

#         time_limit=600,
#         presets="medium_quality",
#         verbosity=1
#     )

#     # -------------------------------------------------------------
#     # Evaluate every AutoGluon model on this grouped tuning fold
#     # -------------------------------------------------------------

#     leaderboard_fold = predictor_fold.leaderboard(
#         fold_tune,
#         silent=True
#     )

#     y_true_fold = fold_tune[TARGET_COL].values

#     for model_name in predictor_fold.model_names():

#         try:

#             y_pred_fold = predictor_fold.predict(
#                 fold_tune,
#                 model=model_name
#             )

#             rmse_fold = np.sqrt(
#                 mean_squared_error(
#                     y_true_fold,
#                     y_pred_fold
#                 )
#             )

#             mae_fold = mean_absolute_error(
#                 y_true_fold,
#                 y_pred_fold
#             )

#             # R² can be unstable with very small folds,
#             # but save it for reporting.
#             r2_fold = r2_score(
#                 y_true_fold,
#                 y_pred_fold
#             )

#             all_model_results.append({
#                 "Fold": fold,
#                 "Model": model_name,
#                 "RMSE": rmse_fold,
#                 "MAE": mae_fold,
#                 "R2": r2_fold,
#                 "N_Train_Groups": len(train_groups),
#                 "N_Tune_Groups": len(tune_groups)
#             })

#         except Exception as e:

#             print(
#                 f"Could not evaluate "
#                 f"{model_name}: {e}"
#             )

#     best_fold_model = predictor_fold.model_best

#     fold_results.append({
#         "Fold": fold,
#         "Best_Model": best_fold_model,
#         "Train_Groups": len(train_groups),
#         "Tune_Groups": len(tune_groups)
#     })


# # =====================================================================
# # SUMMARIZE GROUPED CV
# # =====================================================================

# model_cv_df = pd.DataFrame(
#     all_model_results
# )

# model_cv_df.to_csv(
#     os.path.join(
#         OUT_DIR,
#         "grouped_cv_all_models.csv"
#     ),
#     index=False
# )

# print("\n" + "=" * 80)
# print("GROUPED CV MODEL SUMMARY")
# print("=" * 80)

# model_summary = (
#     model_cv_df
#     .groupby("Model")
#     .agg(
#         Mean_RMSE=("RMSE", "mean"),
#         SD_RMSE=("RMSE", "std"),
#         Mean_MAE=("MAE", "mean"),
#         Mean_R2=("R2", "mean"),
#         Folds_Evaluated=("Fold", "nunique")
#     )
#     .reset_index()
# )

# # Prefer models evaluated in all four folds.
# model_summary = model_summary[
#     model_summary["Folds_Evaluated"] == 4
# ].copy()

# model_summary = model_summary.sort_values(
#     "Mean_RMSE",
#     ascending=True
# )

# print(
#     model_summary.to_string(
#         index=False,
#         float_format=lambda x: f"{x:.4f}"
#     )
# )

# model_summary.to_csv(
#     os.path.join(
#         OUT_DIR,
#         "grouped_cv_model_summary.csv"
#     ),
#     index=False
# )

# selected_model_name = (
#     model_summary.iloc[0]["Model"]
# )

# print(
#     f"\n🏆 GROUPED-CV SELECTED MODEL: "
#     f"{selected_model_name}"
# )

# print(
#     f"Mean grouped CV RMSE: "
#     f"{model_summary.iloc[0]['Mean_RMSE']:.4f}"
# )

# # =====================================================================
# # FINAL MODEL FIT ON ALL CALIBRATION DATA
# # =====================================================================

# print("\n" + "=" * 80)
# print("FINAL MODEL FIT ON ALL CALIBRATION DATA")
# print("=" * 80)

# final_run_id = datetime.now().strftime(
#     "%Y%m%d_%H%M%S_%f"
# )

# final_model_path = os.path.join(
#     OUT_DIR,
#     f"final_model_{final_run_id}"
# )

# final_predictor = TabularPredictor(
#     label=TARGET_COL,
#     problem_type="regression",
#     eval_metric="root_mean_squared_error",
#     path=final_model_path
# ).fit(
#     train_data=train_df,
#     presets="medium_quality",
#     time_limit=1800,
#     verbosity=2
# )

# final_best_model = final_predictor.model_best

# print(
#     f"\nFinal full-calibration AutoGluon model: "
#     f"{final_best_model}"
# )





# # NOW test on YOUR independent validation set
# print("\n" + "="*80)
# print("FINAL INDEPENDENT VALIDATION (5 soil-condition rows)")
# print("="*80)

# y_test = test_df[TARGET_COL].values
# y_pred = final_predictor.predict(test_df)

# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# rpd = y_test.std() / rmse
# q75, q25 = np.percentile(y_test, [75, 25])
# rpiq = (q75 - q25) / rmse

# sample_eval = test_df_full.copy()
# sample_eval['Predicted_SOC'] = np.asarray(y_pred)

# sample_results = (
#     sample_eval     
#     .groupby(['Sample_No','soc'], as_index=False) 
#     .agg(
#         #actual_SOC=('soc', 'mean'),
#         Predicted_SOC=('Predicted_SOC', 'mean'),
#        # N_Images=('image_no', 'count')
#     )
#     .rename(columns={'soc': 'Actual_SOC'})
# )

# sample_r2 = r2_score(
#     sample_results['Actual_SOC'],
#     sample_results['Predicted_SOC']
# )

# sample_rmse = np.sqrt(
#     mean_squared_error(
#         sample_results['Actual_SOC'],
#         sample_results['Predicted_SOC']
#     )
# )

# sample_mae = mean_absolute_error(
#     sample_results['Actual_SOC'],
#     sample_results['Predicted_SOC']
# )

# print("\nSAMPLE-LEVEL RESULTS")
# print(sample_results.to_string(index=False))
# print(f"\nSample-level R²   = {sample_r2:.4f}")
# print(f"Sample-level RMSE = {sample_rmse:.4f}")
# print(f"Sample-level MAE  = {sample_mae:.4f}")

# print(f"\n  Best Model: {final_best_model}")
# print(f"  R²   = {r2:.4f}")
# print(f"  RMSE = {rmse:.4f}")
# print(f"  MAE  = {mae:.4f}")
# print(f"  RPD  = {rpd:.4f}")
# print(f"  RPIQ = {rpiq:.4f}")

# if rpd > 2.0:
#     capability = "EXCELLENT"
# elif rpd > 1.4:
#     capability = "GOOD"
# else:
#     capability = "MODERATE"

# print(f"\n  📊 Prediction Capability: {capability}")

# # Save results
# model_summary.to_csv(
#     os.path.join(
#         OUT_DIR,
#         'grouped_cv_model_summary.csv'
#     ),
#     index=False
# )

# model_cv_df.to_csv(
#     os.path.join(
#         OUT_DIR,
#         'grouped_cv_all_models.csv'
#     ),
#     index=False
# )

# results_df = pd.DataFrame([{
#     'Best_Model': final_best_model,
#     'Internal_Validation': 'AutoML 80/20 split on 461 samples',
#     'Independent_Validation': '197 samples (Kennard-Stone)',
#     'R²': r2,
#     'RMSE': rmse,
#     'MAE': mae,
#     'RPD': rpd,
#     'RPIQ': rpiq,
#     'Capability': capability
# }])
# results_df.to_csv(os.path.join(OUT_DIR, 'FINAL_RESULTS.csv'), index=False)

# # Save predictions
# pred_df = test_df_full.copy()
# pred_df['Predicted_SOC'] = y_pred
# pred_df.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)

# print(f"\n📁 Results saved to: {OUT_DIR}/")

# print("\n" + "="*80)
# print("✅ COMPLETE")
# print("="*80)
# print(f"Calibration rows used: {len(train_df)}")
# print(f"Validation rows used: {len(test_df)}")
# print(
#     f"Calibration physical samples: "
#     f"{train_df_full['Sample_No'].nunique()}"
# )
# print(
#     f"Validation physical samples: "
#     f"{test_df_full['Sample_No'].nunique()}"
# )
# print(f"\n  Best Model: {final_best_model}")
# print(f"  R² = {r2:.4f}")
# print(f"  RPD = {rpd:.4f} ({capability})")
# print("\n" + "="*80)
# """
# AutoML with CUSTOM COMPOSITE METRIC
# ====================================
# AutoGluon will select models based on composite score:
# 0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE

# Instead of just RMSE!
# """

# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from autogluon.tabular import TabularPredictor
# from autogluon.core.metrics import make_scorer
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # Paths
# HERE = os.path.dirname(os.path.abspath(__file__))
# # Same CALIB_CSV and VALID_CSV as above
# CALIB_CSV = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output\calibration_selected_features.csv'
# VALID_CSV  = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output\validation_selected_features.csv'
# OUT_DIR    = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step3_output\automl_COMPOSITE_METRIC'
# os.makedirs(OUT_DIR, exist_ok=True)

# TARGET_COL = 'soc'

# print("\n" + "="*80)
# print("AUTOML WITH CUSTOM COMPOSITE METRIC")
# print("="*80)
# print("  AutoGluon will select models based on:")
# print("  Composite = 0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE")
# print("  NOT just RMSE!")
# print("="*80)


# # ═══════════════════════════════════════════════════════════════════════════
# # DEFINE CUSTOM COMPOSITE METRIC
# # ═══════════════════════════════════════════════════════════════════════════

# # Weights (same as your multi-metric selection)
# WEIGHTS = {
#     'RPD':   0.35,
#     'RPIQ':  0.25,
#     'RMSE':  0.20,
#     'R²':    0.15,
#     'MAE':   0.05
# }

# print(f"\n⚖️  WEIGHTING:")
# for metric, weight in WEIGHTS.items():
#     print(f"  {metric:6s}: {weight:.0%}")


# def composite_score_metric(y_true, y_pred, sample_weight=None):
#     """
#     Custom metric: Composite score for soil science
    
#     Higher is better!
    
#     Combines:
#     - RPD (35%) - soil science standard
#     - RPIQ (25%) - robust performance
#     - RMSE (20%) - prediction error
#     - R² (15%) - variance explained
#     - MAE (5%) - average error
    
#     Returns a score where HIGHER = BETTER
#     """
    
#     # Convert to numpy arrays
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
    
#     # Calculate individual metrics
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     r2 = r2_score(y_true, y_pred)
    
#     # RPD
#     rpd = y_true.std() / rmse if rmse > 0 else 0
    
#     # RPIQ
#     q75, q25 = np.percentile(y_true, [75, 25])
#     rpiq = (q75 - q25) / rmse if rmse > 0 else 0
    
#     # Normalize each metric to 0-1 scale
#     # We need reference values - use typical ranges for SOC prediction
    
#     # RPD: 0-4 range (typical in soil science)
#     rpd_norm = np.clip(rpd / 4.0, 0, 1)
    
#     # RPIQ: 0-4 range
#     rpiq_norm = np.clip(rpiq / 4.0, 0, 1)
    
#     # RMSE: inverse (lower is better), assume 0-2 g/kg range
#     rmse_norm = 1 - np.clip(rmse / 2.0, 0, 1)
    
#     # R²: already 0-1, but can be negative
#     r2_norm = np.clip(r2, 0, 1)
    
#     # MAE: inverse (lower is better), assume 0-2 g/kg range
#     mae_norm = 1 - np.clip(mae / 2.0, 0, 1)
    
#     # Calculate composite (weighted sum)
#     composite = (
#         WEIGHTS['RPD'] * rpd_norm +
#         WEIGHTS['RPIQ'] * rpiq_norm +
#         WEIGHTS['RMSE'] * rmse_norm +
#         WEIGHTS['R²'] * r2_norm +
#         WEIGHTS['MAE'] * mae_norm
#     )
    
#     # Return composite (higher is better)
#     return composite


# # Create AutoGluon scorer from custom metric
# composite_scorer = make_scorer(
#     name='composite_score',
#     score_func=composite_score_metric,
#     optimum=1.0,          # Best possible score
#     greater_is_better=True  # Higher is better
# )

# print("\n✅ Custom composite metric defined")
# print("   AutoGluon will now optimize this instead of RMSE!")


# # ═══════════════════════════════════════════════════════════════════════════
# # LOAD DATA
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "="*80)
# print("LOADING DATA")
# print("="*80)

# train_df = pd.read_csv(CALIB_CSV)
# test_df = pd.read_csv(VALID_CSV)

# NON_PREDICTOR_COLS = ['image_no', 'moisture', 'soil_type', 'Sample_No']

# train_df_full = train_df.copy()
# test_df_full = test_df.copy()

# train_df = train_df.drop(
#     columns=[c for c in NON_PREDICTOR_COLS if c in train_df.columns]
# )

# test_df = test_df.drop(
#     columns=[c for c in NON_PREDICTOR_COLS if c in test_df.columns]
# )

# print("  Modeling columns:", train_df.columns.tolist())


# print(f"  Calibration: {train_df.shape}")
# print(f"  Validation: {test_df.shape}")


# # ═══════════════════════════════════════════════════════════════════════════
# # TRAIN AUTOGLUON WITH CUSTOM METRIC
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "="*80)
# print("TRAINING AUTOML WITH CUSTOM COMPOSITE METRIC")
# print("="*80)
# print("  Time limit: 30 minutes")
# print("  Selection criterion: COMPOSITE SCORE (not RMSE)")

# from datetime import datetime

# run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# model_path = os.path.join(
#     OUT_DIR,
#     f"models_{run_id}"
# )

# os.makedirs(model_path, exist_ok=True)

# print(f"AutoGluon model path: {model_path}")

# predictor = TabularPredictor(
#     label=TARGET_COL,
#     eval_metric=composite_scorer,  # ✅ CUSTOM METRIC!
#     path=model_path
# ).fit(
#     train_data=train_df,
#     time_limit=1800,
#     presets='medium_quality',
#     verbosity=2
# )

# print("\n✅ Training complete using COMPOSITE METRIC!")


# # ═══════════════════════════════════════════════════════════════════════════
# # RESULTS
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "="*80)
# print("AUTOML RESULTS (Selected by Composite Score)")
# print("="*80)

# leaderboard = predictor.leaderboard(train_df, silent=True)
# print("\nTOP 10 MODELS (ranked by COMPOSITE SCORE):")
# print(leaderboard[['model', 'score_val', 'pred_time_val']].head(10))

# final_best_model = predictor.model_best
# print(f"\n🏆 AutoML selected (by composite): {final_best_model}")


# # ═══════════════════════════════════════════════════════════════════════════
# # TEST ON INDEPENDENT VALIDATION SET
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "="*80)
# print("INDEPENDENT VALIDATION (197 samples)")
# print("="*80)

# y_test = test_df[TARGET_COL].values
# y_pred = predictor.predict(test_df)

# # Calculate all metrics
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# rpd = y_test.std() / rmse
# q75, q25 = np.percentile(y_test, [75, 25])
# rpiq = (q75 - q25) / rmse

# # Calculate composite score
# composite = composite_score_metric(y_test, y_pred)

# print(f"\n  Best Model: {final_best_model}")
# print(f"\n  COMPOSITE METRICS:")
# print(f"    RPD  = {rpd:.4f} (weight: {WEIGHTS['RPD']:.0%})")
# print(f"    RPIQ = {rpiq:.4f} (weight: {WEIGHTS['RPIQ']:.0%})")
# print(f"    RMSE = {rmse:.4f} (weight: {WEIGHTS['RMSE']:.0%})")
# print(f"    R²   = {r2:.4f} (weight: {WEIGHTS['R²']:.0%})")
# print(f"    MAE  = {mae:.4f} (weight: {WEIGHTS['MAE']:.0%})")
# print(f"\n  📊 COMPOSITE SCORE = {composite:.4f}")

# if rpd > 2.0:
#     capability = "EXCELLENT"
# elif rpd > 1.4:
#     capability = "GOOD"
# else:
#     capability = "MODERATE"

# print(f"\n  📈 Prediction Capability: {capability} (RPD-based)")


# # ═══════════════════════════════════════════════════════════════════════════
# # DETAILED LEADERBOARD WITH ALL METRICS
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "="*80)
# print("DETAILED LEADERBOARD (All Metrics)")
# print("="*80)

# # Get predictions for all models
# model_names = leaderboard['model'].tolist()

# detailed_results = []

# for model_name in model_names:
#     try:
#         y_pred_model = predictor.predict(test_df, model=model_name)
        
#         mae_m = mean_absolute_error(y_test, y_pred_model)
#         rmse_m = np.sqrt(mean_squared_error(y_test, y_pred_model))
#         r2_m = r2_score(y_test, y_pred_model)
#         rpd_m = y_test.std() / rmse_m
#         rpiq_m = (q75 - q25) / rmse_m
#         composite_m = composite_score_metric(y_test, y_pred_model)
        
#         detailed_results.append({
#             'Model': model_name,
#             'Composite': composite_m,
#             'RPD': rpd_m,
#             'RPIQ': rpiq_m,
#             'RMSE': rmse_m,
#             'R²': r2_m,
#             'MAE': mae_m
#         })
#     except:
#         pass

# df_detailed = pd.DataFrame(detailed_results)
# df_detailed = df_detailed.sort_values('Composite', ascending=False)
# df_detailed.insert(0, 'Rank', range(1, len(df_detailed) + 1))

# print("\n" + df_detailed.to_string(index=False))


# # ═══════════════════════════════════════════════════════════════════════════
# # SAVE RESULTS
# # ═══════════════════════════════════════════════════════════════════════════

# leaderboard.to_csv(os.path.join(OUT_DIR, 'model_leaderboard_composite.csv'), index=False)
# df_detailed.to_csv(os.path.join(OUT_DIR, 'detailed_metrics_all_models.csv'), index=False)

# results_df = pd.DataFrame([{
#     'Best_Model': final_best_model,
#     'Selection_Criterion': 'Composite Score (0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE)',
#     'Composite_Score': composite,
#     'RPD': rpd,
#     'RPIQ': rpiq,
#     'RMSE': rmse,
#     'R²': r2,
#     'MAE': mae,
#     'Capability': capability
# }])
# results_df.to_csv(os.path.join(OUT_DIR, 'FINAL_RESULTS_COMPOSITE.csv'), index=False)

# # Save predictions
# pred_df = test_df_full.copy()
# pred_df['Predicted_SOC'] = y_pred
# pred_df.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)

# print(f"\n📁 Results saved to: {OUT_DIR}/")
# print(f"  Models saved at: {model_path}")


# # ═══════════════════════════════════════════════════════════════════════════
# # FINAL SUMMARY
# # ═══════════════════════════════════════════════════════════════════════════

# print("\n" + "="*80)
# print("✅ AUTOML COMPLETE (CUSTOM COMPOSITE METRIC)")
# print("="*80)

# print(f"\n  🎯 KEY DIFFERENCE:")
# print(f"     Standard AutoML: Selects by RMSE only")
# print(f"     This AutoML: Selects by COMPOSITE SCORE")
# print(f"     (35% RPD + 25% RPIQ + 20% RMSE + 15% R² + 5% MAE)")

# print(f"\n  🏆 BEST MODEL: {final_best_model}")
# print(f"     Composite Score = {composite:.4f}")
# print(f"     RPD = {rpd:.4f} ({capability})")
# print(f"     R² = {r2:.4f}")
# print(f"     RMSE = {rmse:.4f}")

# print(f"\n  ✅ Model was selected based on ALL 5 metrics")
# print(f"     NOT just RMSE!")

# print("\n" + "="*80)



"""
STEP 3 — AutoGluon model selection with grouped CV + composite re-ranking
========================================================================

Purpose
-------
1. Use only the sample-grouped calibration set produced by Step 2.
2. Perform internal model evaluation with GroupKFold by physical soil Sample_No.
3. In every fold, AutoGluon trains/ranks models using RMSE.
4. Evaluate every available model on the grouped tuning fold using:
      RMSE, MAE, R², RPD, RPIQ
5. Aggregate each model's metrics across grouped folds.
6. Re-rank models using a normalized composite score:
      0.35*RPD + 0.25*RPIQ + 0.20*RMSE + 0.15*R² + 0.05*MAE
   where all five components are first normalized to [0, 1] across
   candidate models and RMSE/MAE are inverted so higher is always better.
7. Lock the model name selected from grouped CV.
8. Fit AutoGluon once on the complete calibration set using RMSE, and use
   the grouped-CV-selected model for the untouched external validation set.
9. Report image-level and soil-condition-level validation metrics.

IMPORTANT
---------
- The external validation set is NEVER used to select or re-rank models.
- Sample_No, moisture, soil_type, and image_no are metadata only.
- Feature selection has already been completed in Step 2.
- The same workflow can later be reused for the Kennard-Stone comparison.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold


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
    "step2_output",
    "calibration_selected_features.csv",
)

VALID_CSV = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step2_output",
    "validation_selected_features.csv",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "objective_1",
    "output_data",
    "step3_output",
)

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

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
FOLD_TIME_LIMIT = 600       # seconds per grouped-CV fold
FINAL_TIME_LIMIT = 1800     # seconds for full-calibration fit
PRESETS = "medium_quality"

# Composite weights requested for secondary re-ranking.
WEIGHTS = {
    "RPD": 0.35,
    "RPIQ": 0.25,
    "RMSE": 0.20,
    "R2": 0.15,
    "MAE": 0.05,
}


# =============================================================================
# HELPERS
# =============================================================================

def regression_metrics(y_true, y_pred):
    """Return RMSE, MAE, R², RPD, and RPIQ."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Sample standard deviation for RPD.
    sd_y = np.std(y_true, ddof=1) if len(y_true) > 1 else np.nan
    rpd = sd_y / rmse if rmse > 0 and np.isfinite(sd_y) else np.nan

    q75, q25 = np.percentile(y_true, [75, 25])
    iqr_y = q75 - q25
    rpiq = iqr_y / rmse if rmse > 0 else np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "RPD": rpd,
        "RPIQ": rpiq,
    }


def normalize_higher_is_better(series):
    """Min-max normalize to [0,1]."""
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
    """Min-max normalize to [0,1] and invert so lower raw value = higher score."""
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
        out.loc[finite] = (hi - s[finite]) / (hi - lo)

    return out


def add_composite_score(model_summary):
    """Add normalized metric components and the requested weighted score."""
    df = model_summary.copy()

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


def model_prediction(predictor, data, model_name):
    """Predict with an explicitly named AutoGluon model."""
    return predictor.predict(data, model=model_name)


# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3 — GROUPED AutoML + COMPOSITE RE-RANKING")
print("=" * 80)

cal_full = pd.read_csv(CALIB_CSV)
val_full = pd.read_csv(VALID_CSV)

if TARGET_COL not in cal_full.columns or TARGET_COL not in val_full.columns:
    raise ValueError(f"Target column '{TARGET_COL}' must exist in both datasets.")

if GROUP_COL not in cal_full.columns or GROUP_COL not in val_full.columns:
    raise ValueError(
        f"'{GROUP_COL}' must be retained in Step-2 outputs for grouped validation."
    )

cal_groups = set(cal_full[GROUP_COL].dropna().unique())
val_groups = set(val_full[GROUP_COL].dropna().unique())
external_overlap = cal_groups & val_groups

if external_overlap:
    raise ValueError(
        "External validation leakage detected. "
        f"Samples present in both calibration and validation: {sorted(external_overlap)}"
    )

print(f"\nCalibration: {len(cal_full)} images from {len(cal_groups)} physical soils")
print(f"Validation : {len(val_full)} images from {len(val_groups)} held-out soils")
print(f"Calibration soils: {sorted(cal_groups)}")
print(f"Validation soils : {sorted(val_groups)}")
print("Physical-sample overlap: ZERO")

# Model matrices: metadata removed.
cal_model = cal_full.drop(
    columns=[c for c in NON_PREDICTOR_COLS if c in cal_full.columns]
).copy()

val_model = val_full.drop(
    columns=[c for c in NON_PREDICTOR_COLS if c in val_full.columns]
).copy()

FEATURES = [c for c in cal_model.columns if c != TARGET_COL]

if not FEATURES:
    raise ValueError("No predictor columns found after metadata removal.")

missing_in_validation = [f for f in FEATURES if f not in val_model.columns]
if missing_in_validation:
    raise ValueError(
        f"Validation is missing predictors used by calibration: {missing_in_validation}"
    )

# Ensure same modeling columns and order in both datasets.
val_model = val_model[[TARGET_COL] + FEATURES].copy()
cal_model = cal_model[[TARGET_COL] + FEATURES].copy()

print(f"\nPredictors ({len(FEATURES)}): {FEATURES}")


# =============================================================================
# GROUP-AWARE INTERNAL MODEL EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("4-FOLD GROUP-AWARE INTERNAL MODEL EVALUATION")
print("AutoGluon trains/ranks each fold using RMSE.")
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
        raise ValueError(f"Group leakage in fold {fold}: {sorted(overlap)}")

    print(f"Train soils ({len(train_groups)}): {sorted(train_groups)}")
    print(f"Tune soils  ({len(tune_groups)}): {sorted(tune_groups)}")
    print("Fold sample overlap: ZERO")

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
        tuning_data=fold_tune,   # explicit grouped tuning fold
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
            y_pred_fold = model_prediction(
                predictor_fold,
                fold_tune,
                model_name,
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


# =============================================================================
# AGGREGATE GROUPED-CV METRICS
# =============================================================================

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

# Only compare models evaluated successfully in every grouped fold.
model_summary = model_summary[
    model_summary["Folds_Evaluated"] == N_GROUP_FOLDS
].copy()

if model_summary.empty:
    raise RuntimeError(
        "No model was successfully evaluated in all grouped CV folds. "
        "Increase time limits or inspect grouped_cv_all_models_all_folds.csv."
    )

# Native RMSE ranking.
model_summary["RMSE_Rank"] = (
    model_summary["Mean_RMSE"]
    .rank(method="min", ascending=True)
    .astype(int)
)

# Secondary composite re-ranking.
model_summary = add_composite_score(model_summary)
model_summary.insert(0, "Composite_Rank", np.arange(1, len(model_summary) + 1))

model_summary.to_csv(
    os.path.join(OUT_DIR, "grouped_cv_model_composite_ranking.csv"),
    index=False,
)

print("\n" + "=" * 80)
print("GROUPED-CV MODEL SUMMARY — COMPOSITE RE-RANKING")
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
print("MODEL LOCKED BEFORE EXTERNAL VALIDATION")
print("=" * 80)
print(f"Selected model from grouped-CV composite re-ranking: {selected_model_name}")
print("External validation has NOT been used for model selection or ranking.")


# =============================================================================
# FINAL AutoGluon FIT ON ALL CALIBRATION IMAGES
# =============================================================================

print("\n" + "=" * 80)
print("FINAL FIT ON ALL CALIBRATION IMAGES")
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
        "\nThe grouped-CV-selected model was not produced in the final "
        "full-calibration AutoGluon run.\n"
        f"Selected model: {selected_model_name}\n"
        f"Available final models: {available_final_models}\n"
        "Do NOT silently substitute another model. Increase the final "
        "time limit or inspect why that model failed to train."
    )

print(f"\nAutoGluon native RMSE-best model in final run: {final_predictor.model_best}")
print(f"Model used for external validation: {selected_model_name}")
print(
    "The external-validation model is the model name locked by grouped CV, "
    "not automatically replaced by the final run's native RMSE winner."
)


# =============================================================================
# EXTERNAL VALIDATION — IMAGE LEVEL
# =============================================================================

print("\n" + "=" * 80)
print("EXTERNAL VALIDATION — IMAGE LEVEL")
print("=" * 80)

y_test = val_model[TARGET_COL].values

y_pred = model_prediction(
    final_predictor,
    val_model,
    selected_model_name,
)

image_metrics = regression_metrics(y_test, y_pred)

print(f"Validation images: {len(val_model)}")
print(f"Held-out physical soils: {val_full[GROUP_COL].nunique()}")
print(f"R²   = {image_metrics['R2']:.4f}")
print(f"RMSE = {image_metrics['RMSE']:.4f}")
print(f"MAE  = {image_metrics['MAE']:.4f}")
print(f"RPD  = {image_metrics['RPD']:.4f}")
print(f"RPIQ = {image_metrics['RPIQ']:.4f}")


# =============================================================================
# EXTERNAL VALIDATION — SOIL-CONDITION LEVEL
# =============================================================================

# Some physical soils have >1 SOC reference value. Therefore repeated images
# are aggregated within Sample_No + SOC, not across distinct SOC conditions.
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
print("EXTERNAL VALIDATION — SOIL-CONDITION LEVEL")
print("=" * 80)
print(condition_results.to_string(index=False))
print(f"\nSoil-condition rows = {len(condition_results)}")
print(f"R²   = {condition_metrics['R2']:.4f}")
print(f"RMSE = {condition_metrics['RMSE']:.4f}")
print(f"MAE  = {condition_metrics['MAE']:.4f}")
print(f"RPD  = {condition_metrics['RPD']:.4f}")
print(f"RPIQ = {condition_metrics['RPIQ']:.4f}")


# =============================================================================
# SIMPLE CALIBRATION-MEAN BASELINE
# =============================================================================

baseline_value = cal_model[TARGET_COL].mean()
baseline_pred = np.full(shape=len(y_test), fill_value=baseline_value, dtype=float)
baseline_metrics = regression_metrics(y_test, baseline_pred)

print("\n" + "=" * 80)
print("CALIBRATION-MEAN BASELINE — EXTERNAL VALIDATION")
print("=" * 80)
print(f"Calibration mean SOC = {baseline_value:.4f}")
print(f"Baseline R²   = {baseline_metrics['R2']:.4f}")
print(f"Baseline RMSE = {baseline_metrics['RMSE']:.4f}")
print(f"Baseline MAE  = {baseline_metrics['MAE']:.4f}")
print(f"Baseline RPD  = {baseline_metrics['RPD']:.4f}")
print(f"Baseline RPIQ = {baseline_metrics['RPIQ']:.4f}")


# =============================================================================
# SAVE RESULTS
# =============================================================================

predictions = val_full.copy()
predictions["Predicted_SOC"] = np.asarray(y_pred)
predictions.to_csv(
    os.path.join(OUT_DIR, "external_validation_predictions.csv"),
    index=False,
)

condition_results.to_csv(
    os.path.join(OUT_DIR, "external_validation_soil_condition_results.csv"),
    index=False,
)

final_results = pd.DataFrame([{
    "Selected_Model": selected_model_name,
    "Selection_Method": (
        "4-fold GroupKFold by Sample_No; AutoGluon trained with RMSE; "
        "models secondarily re-ranked by normalized composite score"
    ),
    "Composite_Formula": (
        "0.35*RPD + 0.25*RPIQ + 0.20*RMSE + 0.15*R2 + 0.05*MAE "
        "(all components min-max normalized across grouped-CV candidate models; "
        "RMSE and MAE inverted)"
    ),
    "Calibration_Images": len(cal_full),
    "Calibration_Physical_Soils": cal_full[GROUP_COL].nunique(),
    "Validation_Images": len(val_full),
    "Validation_Physical_Soils": val_full[GROUP_COL].nunique(),
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
    os.path.join(OUT_DIR, "FINAL_GROUPED_RESULTS.csv"),
    index=False,
)

print("\n" + "=" * 80)
print("STEP 3 COMPLETE")
print("=" * 80)
print(f"Calibration images used: {len(cal_full)}")
print(f"Calibration physical soils: {cal_full[GROUP_COL].nunique()}")
print(f"Validation images used: {len(val_full)}")
print(f"Validation physical soils: {val_full[GROUP_COL].nunique()}")
print(f"Grouped-CV selected model: {selected_model_name}")
print(f"Outputs saved to: {OUT_DIR}")
print("=" * 80)
