

# # """
# # # """
# # # AutoML - Revised Grouped Validation Approach
# # # ============================================
# # # - AutoML trains on 489 calibration images from 15 soil samples
# # # - Final validation uses 169 images from 5 completely held-out soil samples
# # # - Image ID, moisture, soil type, and sample ID are retained only as metadata

# # """
# # """

# # import os
# # import pandas as pd
# # import numpy as np
# # from autogluon.tabular import TabularPredictor
# # from datetime import datetime
# # from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# # from sklearn.model_selection import GroupKFold


# # # Paths
# # HERE = os.path.dirname(os.path.abspath(__file__))
# # CALIB_CSV = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output_sensitivity\calibration_selected_features.csv'
# # VALID_CSV  = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output_sensitivity\validation_selected_features.csv'
# # OUT_DIR    = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step3_output_sensitivity\automl_grouped_validation'

# # os.makedirs(OUT_DIR, exist_ok=True)

# # TARGET_COL = 'soc'
# # GROUP = "Sample_No"
    

# # print("\n" + "="*80)
# # print("AUTOML - REVISION GROUPED-SAMPLE VALIDATION")
# # print("="*80)
# # print("  Revised grouped-sample validation")

# # # Load data
# # print("\nLoading data...")
# # train_df = pd.read_csv(CALIB_CSV)
# # test_df = pd.read_csv(VALID_CSV)
# # # Keep metadata for traceability, but do NOT use it for modeling
# # NON_PREDICTOR_COLS = ['image_no', 'moisture', 'soil_type', 'Sample_No']

# # train_df_full = train_df.copy()
# # test_df_full = test_df.copy()

# # train_df = train_df.drop(
# #     columns=[c for c in NON_PREDICTOR_COLS if c in train_df.columns]
# # )

# # test_df = test_df.drop(
# #     columns=[c for c in NON_PREDICTOR_COLS if c in test_df.columns]
# # )

# # print("  Modeling columns:", train_df.columns.tolist())


# # print(f"  Calibration (for AutoML): {train_df.shape}")
# # print(f"  Validation (held out): {test_df.shape}")

# # # =====================================================================
# # # GROUP-AWARE INTERNAL MODEL SELECTION
# # # =====================================================================

# # print("\n" + "=" * 80)
# # print("GROUP-AWARE INTERNAL MODEL SELECTION")
# # print("=" * 80)

# # FEATURES = [
# #     c for c in train_df.columns
# #     if c != TARGET_COL
# # ]

# # print(f"Selected predictors: {FEATURES}")

# # # GroupKFold works on the full calibration dataset,
# # # where Sample_No is still available.
# # groups = train_df_full[GROUP].values

# # gkf = GroupKFold(n_splits=4)

# # fold_results = []
# # all_model_results = []

# # for fold, (fit_idx, tune_idx) in enumerate(
# #     gkf.split(
# #         train_df_full,
# #         train_df_full[TARGET_COL],
# #         groups=groups
# #     ),
# #     start=1
# # ):

# #     print("\n" + "=" * 80)
# #     print(f"GROUPED CV FOLD {fold}")
# #     print("=" * 80)

# #     fold_train_full = train_df_full.iloc[fit_idx].copy()
# #     fold_tune_full = train_df_full.iloc[tune_idx].copy()

# #     train_groups = set(fold_train_full[GROUP])
# #     tune_groups = set(fold_tune_full[GROUP])

# #     overlap = train_groups & tune_groups

# #     if overlap:
# #         raise ValueError(
# #             f"Group leakage in fold {fold}: {sorted(overlap)}"
# #         )

# #     print(
# #         f"Training soils ({len(train_groups)}): "
# #         f"{sorted(train_groups)}"
# #     )

# #     print(
# #         f"Tuning soils ({len(tune_groups)}): "
# #         f"{sorted(tune_groups)}"
# #     )

# #     print("Sample overlap: ZERO")

# #     # Remove Sample_No and metadata before AutoGluon
# #     fold_train = fold_train_full[
# #         [TARGET_COL] + FEATURES
# #     ].copy()

# #     fold_tune = fold_tune_full[
# #         [TARGET_COL] + FEATURES
# #     ].copy()

# #     run_id = datetime.now().strftime(
# #         "%Y%m%d_%H%M%S_%f"
# #     )

# #     fold_path = os.path.join(
# #         OUT_DIR,
# #         f"groupcv_fold_{fold}_{run_id}"
# #     )

# #     predictor_fold = TabularPredictor(
# #         label=TARGET_COL,
# #         problem_type="regression",
# #         eval_metric="root_mean_squared_error",
# #         path=fold_path
# #     ).fit(
# #         train_data=fold_train,

# #         # THIS is the important part:
# #         # AutoGluon does NOT create a random tuning split.
# #         tuning_data=fold_tune,

# #         time_limit=600,
# #         presets="medium_quality",
# #         verbosity=1
# #     )

# #     # -------------------------------------------------------------
# #     # Evaluate every AutoGluon model on this grouped tuning fold
# #     # -------------------------------------------------------------

# #     leaderboard_fold = predictor_fold.leaderboard(
# #         fold_tune,
# #         silent=True
# #     )

# #     y_true_fold = fold_tune[TARGET_COL].values

# #     for model_name in predictor_fold.model_names():

# #         try:

# #             y_pred_fold = predictor_fold.predict(
# #                 fold_tune,
# #                 model=model_name
# #             )

# #             rmse_fold = np.sqrt(
# #                 mean_squared_error(
# #                     y_true_fold,
# #                     y_pred_fold
# #                 )
# #             )

# #             mae_fold = mean_absolute_error(
# #                 y_true_fold,
# #                 y_pred_fold
# #             )

# #             # R² can be unstable with very small folds,
# #             # but save it for reporting.
# #             r2_fold = r2_score(
# #                 y_true_fold,
# #                 y_pred_fold
# #             )

# #             all_model_results.append({
# #                 "Fold": fold,
# #                 "Model": model_name,
# #                 "RMSE": rmse_fold,
# #                 "MAE": mae_fold,
# #                 "R2": r2_fold,
# #                 "N_Train_Groups": len(train_groups),
# #                 "N_Tune_Groups": len(tune_groups)
# #             })

# #         except Exception as e:

# #             print(
# #                 f"Could not evaluate "
# #                 f"{model_name}: {e}"
# #             )

# #     best_fold_model = predictor_fold.model_best

# #     fold_results.append({
# #         "Fold": fold,
# #         "Best_Model": best_fold_model,
# #         "Train_Groups": len(train_groups),
# #         "Tune_Groups": len(tune_groups)
# #     })


# # # =====================================================================
# # # SUMMARIZE GROUPED CV
# # # =====================================================================

# # model_cv_df = pd.DataFrame(
# #     all_model_results
# # )

# # model_cv_df.to_csv(
# #     os.path.join(
# #         OUT_DIR,
# #         "grouped_cv_all_models.csv"
# #     ),
# #     index=False
# # )

# # print("\n" + "=" * 80)
# # print("GROUPED CV MODEL SUMMARY")
# # print("=" * 80)

# # model_summary = (
# #     model_cv_df
# #     .groupby("Model")
# #     .agg(
# #         Mean_RMSE=("RMSE", "mean"),
# #         SD_RMSE=("RMSE", "std"),
# #         Mean_MAE=("MAE", "mean"),
# #         Mean_R2=("R2", "mean"),
# #         Folds_Evaluated=("Fold", "nunique")
# #     )
# #     .reset_index()
# # )

# # # Prefer models evaluated in all four folds.
# # model_summary = model_summary[
# #     model_summary["Folds_Evaluated"] == 4
# # ].copy()

# # model_summary = model_summary.sort_values(
# #     "Mean_RMSE",
# #     ascending=True
# # )

# # print(
# #     model_summary.to_string(
# #         index=False,
# #         float_format=lambda x: f"{x:.4f}"
# #     )
# # )

# # model_summary.to_csv(
# #     os.path.join(
# #         OUT_DIR,
# #         "grouped_cv_model_summary.csv"
# #     ),
# #     index=False
# # )

# # selected_model_name = (
# #     model_summary.iloc[0]["Model"]
# # )

# # print(
# #     f"\n🏆 GROUPED-CV SELECTED MODEL: "
# #     f"{selected_model_name}"
# # )

# # print(
# #     f"Mean grouped CV RMSE: "
# #     f"{model_summary.iloc[0]['Mean_RMSE']:.4f}"
# # )

# # # =====================================================================
# # # FINAL MODEL FIT ON ALL CALIBRATION DATA
# # # =====================================================================

# # print("\n" + "=" * 80)
# # print("FINAL MODEL FIT ON ALL CALIBRATION DATA")
# # print("=" * 80)

# # final_run_id = datetime.now().strftime(
# #     "%Y%m%d_%H%M%S_%f"
# # )

# # final_model_path = os.path.join(
# #     OUT_DIR,
# #     f"final_model_{final_run_id}"
# # )

# # final_predictor = TabularPredictor(
# #     label=TARGET_COL,
# #     problem_type="regression",
# #     eval_metric="root_mean_squared_error",
# #     path=final_model_path
# # ).fit(
# #     train_data=train_df,
# #     presets="medium_quality",
# #     time_limit=1800,
# #     verbosity=2
# # )

# # final_best_model = final_predictor.model_best

# # print(
# #     f"\nFinal full-calibration AutoGluon model: "
# #     f"{final_best_model}"
# # )





# # # NOW test on YOUR independent validation set
# # print("\n" + "="*80)
# # print("FINAL INDEPENDENT VALIDATION (4 soil-condition rows)")
# # print("="*80)

# # y_test = test_df[TARGET_COL].values
# # y_pred = final_predictor.predict(test_df)

# # r2 = r2_score(y_test, y_pred)
# # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# # mae = mean_absolute_error(y_test, y_pred)
# # rpd = y_test.std() / rmse
# # q75, q25 = np.percentile(y_test, [75, 25])
# # rpiq = (q75 - q25) / rmse

# # sample_eval = test_df_full.copy()
# # sample_eval['Predicted_SOC'] = np.asarray(y_pred)

# # sample_results = (
# #     sample_eval
# #     .groupby(['Sample_No','soc'], as_index=False) 
# #     .agg(
# #         #actual_SOC=('soc', 'mean'),
# #         Predicted_SOC=('Predicted_SOC', 'mean'),
# #        # N_Images=('image_no', 'count')
# #     )
# #     .rename(columns={'soc': 'Actual_SOC'})
# # )

# # sample_r2 = r2_score(
# #     sample_results['Actual_SOC'],
# #     sample_results['Predicted_SOC']
# # )

# # sample_rmse = np.sqrt(
# #     mean_squared_error(
# #         sample_results['Actual_SOC'],
# #         sample_results['Predicted_SOC']
# #     )
# # )

# # sample_mae = mean_absolute_error(
# #     sample_results['Actual_SOC'],
# #     sample_results['Predicted_SOC']
# # )

# # print("\nSAMPLE-LEVEL RESULTS")
# # print(sample_results.to_string(index=False))
# # print(f"\nSample-level R²   = {sample_r2:.4f}")
# # print(f"Sample-level RMSE = {sample_rmse:.4f}")
# # print(f"Sample-level MAE  = {sample_mae:.4f}")

# # print(f"\n  Best Model: {final_best_model}")
# # print(f"  R²   = {r2:.4f}")
# # print(f"  RMSE = {rmse:.4f}")
# # print(f"  MAE  = {mae:.4f}")
# # print(f"  RPD  = {rpd:.4f}")
# # print(f"  RPIQ = {rpiq:.4f}")

# # if rpd > 2.0:
# #     capability = "EXCELLENT"
# # elif rpd > 1.4:
# #     capability = "GOOD"
# # else:
# #     capability = "MODERATE"

# # print(f"\n  📊 Prediction Capability: {capability}")

# # # Save results
# # model_summary.to_csv(
# #     os.path.join(
# #         OUT_DIR,
# #         'grouped_cv_model_summary.csv'
# #     ),
# #     index=False
# # )

# # model_cv_df.to_csv(
# #     os.path.join(
# #         OUT_DIR,
# #         'grouped_cv_all_models.csv'
# #     ),
# #     index=False
# # )
    
# # results_df = pd.DataFrame([{
# #     'Best_Model': final_best_model,
# #    'Internal_Validation': '4-fold GroupKFold by Sample_No',
# #    'Independent_Validation': '4 held-out soil-condition rows from 4 physical samples', 
# #     'R²': r2,
# #     'RMSE': rmse,
# #     'MAE': mae,
# #     'RPD': rpd,
# #     'RPIQ': rpiq,
# #     'Capability': capability
# # }])
# # results_df.to_csv(os.path.join(OUT_DIR, 'FINAL_RESULTS.csv'), index=False)

# # # Save predictions
# # pred_df = test_df_full.copy()
# # pred_df['Predicted_SOC'] = y_pred
# # pred_df.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)

# # print(f"\n📁 Results saved to: {OUT_DIR}/")

# # print("\n" + "="*80)
# # print("✅ COMPLETE")
# # print("="*80)
# # print(f"Calibration rows used: {len(train_df)}")
# # print(f"Validation rows used: {len(test_df)}")
# # print(
# #     f"Calibration physical samples: "
# #     f"{train_df_full['Sample_No'].nunique()}"
# # )
# # print(
# #     f"Validation physical samples: "
# #     f"{test_df_full['Sample_No'].nunique()}"
# # )
# # print(f"\n  Best Model: {final_best_model}")
# # print(f"  R² = {r2:.4f}")
# # print(f"  RPD = {rpd:.4f} ({capability})")
# # print("\n" + "="*80)
# # """
# # AutoML with CUSTOM COMPOSITE METRIC
# # ====================================
# # AutoGluon will select models based on composite score:
# # 0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE

# # Instead of just RMSE!
# # """

# # import os
# # import pandas as pd
# # import numpy as np
# # from datetime import datetime
# # from autogluon.tabular import TabularPredictor
# # from autogluon.core.metrics import make_scorer
# # from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # # Paths
# # HERE = os.path.dirname(os.path.abspath(__file__))
# # # Same CALIB_CSV and VALID_CSV as above
# # CALIB_CSV = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output_sensitivity\calibration_selected_features.csv'
# # VALID_CSV  = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output_sensitivity\validation_selected_features.csv'
# # OUT_DIR    = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step3_output_sensitivity\automl_COMPOSITE_METRIC'
# # os.makedirs(OUT_DIR, exist_ok=True)

# # TARGET_COL = 'soc'

# # print("\n" + "="*80)
# # print("AUTOML WITH CUSTOM COMPOSITE METRIC")
# # print("="*80)
# # print("  AutoGluon will select models based on:")
# # print("  Composite = 0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE")
# # print("  NOT just RMSE!")
# # print("="*80)


# # # ═══════════════════════════════════════════════════════════════════════════
# # # DEFINE CUSTOM COMPOSITE METRIC
# # # ═══════════════════════════════════════════════════════════════════════════

# # # Weights (same as your multi-metric selection)
# # WEIGHTS = {
# #     'RPD':   0.35,
# #     'RPIQ':  0.25,
# #     'RMSE':  0.20,
# #     'R²':    0.15,
# #     'MAE':   0.05
# # }

# # print(f"\n⚖️  WEIGHTING:")
# # for metric, weight in WEIGHTS.items():
# #     print(f"  {metric:6s}: {weight:.0%}")


# # def composite_score_metric(y_true, y_pred, sample_weight=None):
# #     """
# #     Custom metric: Composite score for soil science
    
# #     Higher is better!
    
# #     Combines:
# #     - RPD (35%) - soil science standard
# #     - RPIQ (25%) - robust performance
# #     - RMSE (20%) - prediction error
# #     - R² (15%) - variance explained
# #     - MAE (5%) - average error
    
# #     Returns a score where HIGHER = BETTER
# #     """
    
# #     # Convert to numpy arrays
# #     y_true = np.array(y_true)
# #     y_pred = np.array(y_pred)
    
# #     # Calculate individual metrics
# #     mae = mean_absolute_error(y_true, y_pred)
# #     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# #     r2 = r2_score(y_true, y_pred)
    
# #     # RPD
# #     rpd = y_true.std() / rmse if rmse > 0 else 0
    
# #     # RPIQ
# #     q75, q25 = np.percentile(y_true, [75, 25])
# #     rpiq = (q75 - q25) / rmse if rmse > 0 else 0
    
# #     # Normalize each metric to 0-1 scale
# #     # We need reference values - use typical ranges for SOC prediction
    
# #     # RPD: 0-4 range (typical in soil science)
# #     rpd_norm = np.clip(rpd / 4.0, 0, 1)
    
# #     # RPIQ: 0-4 range
# #     rpiq_norm = np.clip(rpiq / 4.0, 0, 1)
    
# #     # RMSE: inverse (lower is better), assume 0-2 g/kg range
# #     rmse_norm = 1 - np.clip(rmse / 2.0, 0, 1)
    
# #     # R²: already 0-1, but can be negative
# #     r2_norm = np.clip(r2, 0, 1)
    
# #     # MAE: inverse (lower is better), assume 0-2 g/kg range
# #     mae_norm = 1 - np.clip(mae / 2.0, 0, 1)
    
# #     # Calculate composite (weighted sum)
# #     composite = (
# #         WEIGHTS['RPD'] * rpd_norm +
# #         WEIGHTS['RPIQ'] * rpiq_norm +
# #         WEIGHTS['RMSE'] * rmse_norm +
# #         WEIGHTS['R²'] * r2_norm +
# #         WEIGHTS['MAE'] * mae_norm
# #     )
    
# #     # Return composite (higher is better)
# #     return composite


# # # Create AutoGluon scorer from custom metric
# # composite_scorer = make_scorer(
# #     name='composite_score',
# #     score_func=composite_score_metric,
# #     optimum=1.0,          # Best possible score
# #     greater_is_better=True  # Higher is better
# # )

# # print("\n✅ Custom composite metric defined")
# # print("   AutoGluon will now optimize this instead of RMSE!")


# # # ═══════════════════════════════════════════════════════════════════════════
# # # LOAD DATA
# # # ═══════════════════════════════════════════════════════════════════════════

# # print("\n" + "="*80)
# # print("LOADING DATA")
# # print("="*80)

# # train_df = pd.read_csv(CALIB_CSV)
# # test_df = pd.read_csv(VALID_CSV)

# # NON_PREDICTOR_COLS = ['image_no', 'moisture', 'soil_type', 'Sample_No']

# # train_df_full = train_df.copy()
# # test_df_full = test_df.copy()

# # train_df = train_df.drop(
# #     columns=[c for c in NON_PREDICTOR_COLS if c in train_df.columns]
# # )

# # test_df = test_df.drop(
# #     columns=[c for c in NON_PREDICTOR_COLS if c in test_df.columns]
# # )

# # print("  Modeling columns:", train_df.columns.tolist())


# # print(f"  Calibration: {train_df.shape}")
# # print(f"  Validation: {test_df.shape}")


# # # ═══════════════════════════════════════════════════════════════════════════
# # # TRAIN AUTOGLUON WITH CUSTOM METRIC
# # # ═══════════════════════════════════════════════════════════════════════════

# # print("\n" + "="*80)
# # print("TRAINING AUTOML WITH CUSTOM COMPOSITE METRIC")
# # print("="*80)
# # print("  Time limit: 30 minutes")
# # print("  Selection criterion: COMPOSITE SCORE (not RMSE)")

# # from datetime import datetime

# # run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# # model_path = os.path.join(
# #     OUT_DIR,
# #     f"models_{run_id}"
# # )

# # os.makedirs(model_path, exist_ok=True)

# # print(f"AutoGluon model path: {model_path}")

# # predictor = TabularPredictor(
# #     label=TARGET_COL,
# #     eval_metric=composite_scorer,  # ✅ CUSTOM METRIC!
# #     path=model_path
# # ).fit(
# #     train_data=train_df,
# #     time_limit=1800,
# #     presets='medium_quality',
# #     verbosity=2
# # )

# # print("\n✅ Training complete using COMPOSITE METRIC!")


# # # ═══════════════════════════════════════════════════════════════════════════
# # # RESULTS
# # # ═══════════════════════════════════════════════════════════════════════════

# # print("\n" + "="*80)
# # print("AUTOML RESULTS (Selected by Composite Score)")
# # print("="*80)

# # leaderboard = predictor.leaderboard(train_df, silent=True)
# # print("\nTOP 10 MODELS (ranked by COMPOSITE SCORE):")
# # print(leaderboard[['model', 'score_val', 'pred_time_val']].head(10))

# # final_best_model = predictor.model_best
# # print(f"\n🏆 AutoML selected (by composite): {final_best_model}")


# # # ═══════════════════════════════════════════════════════════════════════════
# # # TEST ON INDEPENDENT VALIDATION SET
# # # ═══════════════════════════════════════════════════════════════════════════

# # print("\n" + "="*80)
# # print("INDEPENDENT VALIDATION (197 samples)")
# # print("="*80)

# # y_test = test_df[TARGET_COL].values
# # y_pred = predictor.predict(test_df)

# # # Calculate all metrics
# # mae = mean_absolute_error(y_test, y_pred)
# # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# # r2 = r2_score(y_test, y_pred)
# # rpd = y_test.std() / rmse
# # q75, q25 = np.percentile(y_test, [75, 25])
# # rpiq = (q75 - q25) / rmse

# # # Calculate composite score
# # composite = composite_score_metric(y_test, y_pred)

# # print(f"\n  Best Model: {final_best_model}")
# # print(f"\n  COMPOSITE METRICS:")
# # print(f"    RPD  = {rpd:.4f} (weight: {WEIGHTS['RPD']:.0%})")
# # print(f"    RPIQ = {rpiq:.4f} (weight: {WEIGHTS['RPIQ']:.0%})")
# # print(f"    RMSE = {rmse:.4f} (weight: {WEIGHTS['RMSE']:.0%})")
# # print(f"    R²   = {r2:.4f} (weight: {WEIGHTS['R²']:.0%})")
# # print(f"    MAE  = {mae:.4f} (weight: {WEIGHTS['MAE']:.0%})")
# # print(f"\n  📊 COMPOSITE SCORE = {composite:.4f}")

# # if rpd > 2.0:
# #     capability = "EXCELLENT"
# # elif rpd > 1.4:
# #     capability = "GOOD"
# # else:
# #     capability = "MODERATE"

# # print(f"\n  📈 Prediction Capability: {capability} (RPD-based)")


# # # ═══════════════════════════════════════════════════════════════════════════
# # # DETAILED LEADERBOARD WITH ALL METRICS
# # # ═══════════════════════════════════════════════════════════════════════════

# # print("\n" + "="*80)
# # print("DETAILED LEADERBOARD (All Metrics)")
# # print("="*80)

# # # Get predictions for all models
# # model_names = leaderboard['model'].tolist()

# # detailed_results = []

# # for model_name in model_names:
# #     try:
# #         y_pred_model = predictor.predict(test_df, model=model_name)
        
# #         mae_m = mean_absolute_error(y_test, y_pred_model)
# #         rmse_m = np.sqrt(mean_squared_error(y_test, y_pred_model))
# #         r2_m = r2_score(y_test, y_pred_model)
# #         rpd_m = y_test.std() / rmse_m
# #         rpiq_m = (q75 - q25) / rmse_m
# #         composite_m = composite_score_metric(y_test, y_pred_model)
        
# #         detailed_results.append({
# #             'Model': model_name,
# #             'Composite': composite_m,
# #             'RPD': rpd_m,
# #             'RPIQ': rpiq_m,
# #             'RMSE': rmse_m,
# #             'R²': r2_m,
# #             'MAE': mae_m
# #         })
# #     except:
# #         pass

# # df_detailed = pd.DataFrame(detailed_results)
# # df_detailed = df_detailed.sort_values('Composite', ascending=False)
# # df_detailed.insert(0, 'Rank', range(1, len(df_detailed) + 1))

# # print("\n" + df_detailed.to_string(index=False))


# # # ═══════════════════════════════════════════════════════════════════════════
# # # SAVE RESULTS
# # # ═══════════════════════════════════════════════════════════════════════════

# # leaderboard.to_csv(os.path.join(OUT_DIR, 'model_leaderboard_composite.csv'), index=False)
# # df_detailed.to_csv(os.path.join(OUT_DIR, 'detailed_metrics_all_models.csv'), index=False)

# # results_df = pd.DataFrame([{
# #     'Best_Model': final_best_model,
# #     'Selection_Criterion': 'Composite Score (0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE)',
# #     'Composite_Score': composite,
# #     'RPD': rpd,
# #     'RPIQ': rpiq,
# #     'RMSE': rmse,
# #     'R²': r2,
# #     'MAE': mae,
# #     'Capability': capability
# # }])
# # results_df.to_csv(os.path.join(OUT_DIR, 'FINAL_RESULTS_COMPOSITE.csv'), index=False)

# # # Save predictions
# # pred_df = test_df_full.copy()
# # pred_df['Predicted_SOC'] = y_pred
# # pred_df.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)

# # print(f"\n📁 Results saved to: {OUT_DIR}/")
# # print(f"  Models saved at: {model_path}")


# # # ═══════════════════════════════════════════════════════════════════════════
# # # FINAL SUMMARY
# # # ═══════════════════════════════════════════════════════════════════════════

# # print("\n" + "="*80)
# # print("✅ AUTOML COMPLETE (CUSTOM COMPOSITE METRIC)")
# # print("="*80)

# # print(f"\n  🎯 KEY DIFFERENCE:")
# # print(f"     Standard AutoML: Selects by RMSE only")
# # print(f"     This AutoML: Selects by COMPOSITE SCORE")
# # print(f"     (35% RPD + 25% RPIQ + 20% RMSE + 15% R² + 5% MAE)")

# # print(f"\n  🏆 BEST MODEL: {final_best_model}")
# # print(f"     Composite Score = {composite:.4f}")
# # print(f"     RPD = {rpd:.4f} ({capability})")
# # print(f"     R² = {r2:.4f}")
# # print(f"     RMSE = {rmse:.4f}")

# # print(f"\n  ✅ Model was selected based on ALL 5 metrics")
# # print(f"     NOT just RMSE!")

# # print("\n" + "="*80)


# """
# Step 3 - AutoML Grouped Validation (SENSITIVITY: Sample 7 dropped)
# ===================================================================
# FIXES APPLIED:
#   1. Removed aggregation - trains on ALL calibration IMAGES not sample means
#   2. Fixed FileNotFoundError - shortened AutoGluon model path
#   3. GroupKFold now groups by Sample_No correctly at image level
#   4. Output folder created before AutoGluon runs
# """

# import os, shutil
# import pandas as pd
# import numpy as np
# from autogluon.tabular import TabularPredictor
# from datetime import datetime
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import GroupKFold

# # ── PATHS ─────────────────────────────────────────────────────────────────────
# PROJECT_ROOT = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed'

# CALIB_CSV = os.path.join(PROJECT_ROOT, 'notebooks', 'objective_1', 'output_data',
#                           'step2_output_sensitivity', 'calibration_selected_features.csv')
# VALID_CSV  = os.path.join(PROJECT_ROOT, 'notebooks', 'objective_1', 'output_data',
#                           'step2_output_sensitivity', 'validation_selected_features.csv')

# # SHORT path for AutoGluon — avoids Windows path length limit
# OUT_DIR    = os.path.join(PROJECT_ROOT, 'notebooks', 'objective_1', 'output_data',
#                           'step3_sens')
# AG_PATH    = os.path.join(OUT_DIR, 'ag_models')

# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(AG_PATH,  exist_ok=True)

# TARGET_COL = 'soc'
# GROUP_COL  = 'Sample_No'

# # Columns to EXCLUDE from modelling (keep for metadata only)
# NON_PREDICTOR_COLS = ['image_no', 'moisture', 'soil_type', 'Sample_No']

# print("\n" + "="*70)
# print("AUTOML - GROUPED VALIDATION (Sample 7 dropped)")
# print("="*70)

# # ── Load data ─────────────────────────────────────────────────────────────────
# print("\nLoading data...")
# train_full = pd.read_csv(CALIB_CSV)
# test_full  = pd.read_csv(VALID_CSV)

# print(f"  Calibration loaded: {len(train_full)} images x {len(train_full.columns)} cols")
# print(f"  Validation loaded : {len(test_full)} images x {len(test_full.columns)} cols")

# # KEY FIX: do NOT aggregate - use all images
# # Just drop non-predictor columns for modelling
# train_df = train_full.drop(columns=[c for c in NON_PREDICTOR_COLS if c in train_full.columns])
# test_df  = test_full.drop(columns=[c for c in NON_PREDICTOR_COLS if c in test_full.columns])

# print(f"\n  Modelling columns: {train_df.columns.tolist()}")
# print(f"  Calibration (for AutoML): {train_df.shape}  ← IMAGE LEVEL ✅")
# print(f"  Validation  (held out):   {test_df.shape}   ← IMAGE LEVEL ✅")

# # Verify calibration samples
# if GROUP_COL in train_full.columns:
#     cal_samples = sorted(train_full[GROUP_COL].dropna().unique().tolist())
#     val_samples = sorted(test_full[GROUP_COL].dropna().unique().tolist())
#     overlap = set(cal_samples) & set(val_samples)
#     print(f"\n  Calibration samples: {cal_samples}")
#     print(f"  Validation samples : {val_samples}")
#     print(f"  Overlap            : {'ZERO ✅' if not overlap else f'WARNING: {overlap}'}")

# FEATURES = [c for c in train_df.columns if c != TARGET_COL]
# print(f"\n  Predictors: {FEATURES}")

# # ── Composite metric ──────────────────────────────────────────────────────────
# WEIGHTS = {'RPD': 0.35, 'RPIQ': 0.25, 'RMSE': 0.20, 'R2': 0.15, 'MAE': 0.05}

# def composite_score_metric(y_true, y_pred):
#     y_true = np.array(y_true); y_pred = np.array(y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     if rmse == 0: return 1.0
#     rpd  = np.std(y_true) / rmse
#     q75, q25 = np.percentile(y_true, [75, 25])
#     rpiq = (q75 - q25) / rmse
#     mae  = mean_absolute_error(y_true, y_pred)
#     r2   = max(0, r2_score(y_true, y_pred))
#     rpd_n  = min(rpd  / 3.0, 1.0)
#     rpiq_n = min(rpiq / 4.0, 1.0)
#     rmse_n = max(0, 1 - rmse / (np.std(y_true) + 1e-8))
#     mae_n  = max(0, 1 - mae  / (np.std(y_true) + 1e-8))
#     return (WEIGHTS['RPD'] * rpd_n + WEIGHTS['RPIQ'] * rpiq_n +
#             WEIGHTS['RMSE'] * rmse_n + WEIGHTS['R2'] * r2 + WEIGHTS['MAE'] * mae_n)

# # ── Grouped CV for model selection ────────────────────────────────────────────
# print("\n" + "="*70)
# print("GROUPED CV MODEL SELECTION (by Sample_No)")
# print("="*70)

# if GROUP_COL in train_full.columns:
#     groups = train_full[GROUP_COL].values
#     gkf    = GroupKFold(n_splits=4)
#     fold_results = {}

#     for fold_idx, (tr_idx, tu_idx) in enumerate(gkf.split(train_df, groups=groups), 1):
#         tr_samples = sorted(set(groups[tr_idx]))
#         tu_samples = sorted(set(groups[tu_idx]))
#         print(f"\n  Fold {fold_idx}: train={tr_samples}  tune={tu_samples}")

#         X_tr = train_df.iloc[tr_idx]
#         X_tu = train_df.iloc[tu_idx]

#         # SHORT fold path to avoid Windows path length limit
#         fold_path = os.path.join(OUT_DIR, f'fold{fold_idx}')
#         if os.path.exists(fold_path):
#             shutil.rmtree(fold_path, ignore_errors=True)
#         os.makedirs(fold_path, exist_ok=True)

#         try:
#             pred_fold = TabularPredictor(
#                 label=TARGET_COL,
#                 eval_metric='rmse',
#                 path=fold_path,
#                 verbosity=0
#             ).fit(
#                 train_data=X_tr,
#                 tuning_data=X_tu,
#                 time_limit=300,
#                 presets='medium_quality',
#             )
#             y_tu_pred = pred_fold.predict(X_tu).values
#             y_tu_true = X_tu[TARGET_COL].values

#             rmse_f = np.sqrt(mean_squared_error(y_tu_true, y_tu_pred))
#             r2_f   = r2_score(y_tu_true, y_tu_pred)
#             print(f"    Fold {fold_idx} RMSE={rmse_f:.4f}  R²={r2_f:.4f}")

#             lb = pred_fold.leaderboard(X_tu, silent=True)
#             for _, row in lb.iterrows():
#                 m = row['model']
#                 fold_results.setdefault(m, []).append(-row['score_val'])

#         except Exception as e:
#             print(f"    Fold {fold_idx} error: {e}")

#     if fold_results:
#         print("\n  Grouped CV summary:")
#         model_means = {m: np.mean(v) for m,v in fold_results.items()}
#         best_cv_model = min(model_means, key=model_means.get)
#         for m, v in sorted(model_means.items(), key=lambda x: x[1]):
#             print(f"    {m:30s}: mean RMSE={v:.4f}")
#         print(f"\n  Best by grouped CV: {best_cv_model}")

# # ── Final training on all calibration images ──────────────────────────────────
# print("\n" + "="*70)
# print("FINAL MODEL — trained on ALL calibration images")
# print("="*70)

# if os.path.exists(AG_PATH):
#     shutil.rmtree(AG_PATH, ignore_errors=True)
# os.makedirs(AG_PATH, exist_ok=True)

# predictor = TabularPredictor(
#     label=TARGET_COL,
#     eval_metric='rmse',
#     path=AG_PATH,
#     verbosity=2
# ).fit(
#     train_data=train_df,
#     time_limit=1800,
#     presets='medium_quality',
# )

# # ── Validation ────────────────────────────────────────────────────────────────
# print("\n" + "="*70)
# print("EXTERNAL VALIDATION (held-out soil samples)")
# print("="*70)

# y_test = test_df[TARGET_COL].values
# y_pred = predictor.predict(test_df).values

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2   = r2_score(y_test, y_pred)
# mae  = mean_absolute_error(y_test, y_pred)
# rpd  = np.std(y_test)/rmse if rmse>0 else 0
# q75, q25 = np.percentile(y_test, [75,25])
# rpiq = (q75-q25)/rmse if rmse>0 else 0

# print(f"\n  IMAGE-LEVEL RESULTS ({len(y_test)} images):")
# print(f"    R²    = {r2:.4f}")
# print(f"    RMSE  = {rmse:.4f}")
# print(f"    RPD   = {rpd:.4f}")
# print(f"    RPIQ  = {rpiq:.4f}")
# print(f"    MAE   = {mae:.4f}")

# # Sample-level
# if GROUP_COL in test_full.columns:
#     df_r = pd.DataFrame({'Sample_No': test_full[GROUP_COL].values,
#                           'y_true': y_test, 'y_pred': y_pred})
#     samp = df_r.groupby('Sample_No').agg(
#         y_true=('y_true','mean'), y_pred=('y_pred','mean'),
#         n=('y_true','count')).reset_index()
#     from sklearn.metrics import r2_score as r2s
#     sr2   = r2s(samp['y_true'], samp['y_pred'])
#     srmse = np.sqrt(mean_squared_error(samp['y_true'], samp['y_pred']))
#     srpd  = np.std(samp['y_true'])/srmse if srmse>0 else 0
#     print(f"\n  SAMPLE-LEVEL RESULTS ({len(samp)} soil samples):")
#     print(f"    R²    = {sr2:.4f}")
#     print(f"    RMSE  = {srmse:.4f}")
#     print(f"    RPD   = {srpd:.4f}")
#     print(f"\n  Per-sample predictions:")
#     print(f"  {'Sample':8s} {'n':5s} {'Actual':10s} {'Predicted':12s} {'Error':8s}")
#     print("  " + "-"*46)
#     for _, r in samp.iterrows():
#         print(f"  {int(r['Sample_No']):8d} {int(r['n']):5d} "
#               f"{r['y_true']:10.4f} {r['y_pred']:12.4f} "
#               f"{r['y_pred']-r['y_true']:+8.4f}")

# # Save
# pred_out = test_full.copy(); pred_out['Predicted_SOC'] = y_pred
# pred_out.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)
# predictor.leaderboard(test_df, silent=True).to_csv(
#     os.path.join(OUT_DIR, 'model_leaderboard.csv'), index=False)
# print(f"\n  Outputs saved to: {OUT_DIR}")



"""
# """
# AutoML - Revised Grouped Validation Approach
# ============================================
# - AutoML trains on 489 calibration images from 15 soil samples
# - Final validation uses 169 images from 5 completely held-out soil samples
# - Image ID, moisture, soil type, and sample ID are retained only as metadata

"""
"""

import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold


# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_CSV = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output_sensitivity\calibration_selected_features.csv'
VALID_CSV  = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output_sensitivity\validation_selected_features.csv'
OUT_DIR    = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step3_output_sensitivity\automl_grouped_validation'

os.makedirs(OUT_DIR, exist_ok=True)

# Keep result files in the project folder, but save AutoGluon model artifacts
# to short local paths to avoid Windows/OneDrive path-length errors.
GROUPED_MODEL_ROOT = r'C:\\ag_models\\soc_sensitivity_grouped'
COMPOSITE_MODEL_ROOT = r'C:\\ag_models\\soc_sensitivity_composite'
os.makedirs(GROUPED_MODEL_ROOT, exist_ok=True)
os.makedirs(COMPOSITE_MODEL_ROOT, exist_ok=True)

TARGET_COL = 'soc'
GROUP = "Sample_No"
    

print("\n" + "="*80)
print("AUTOML - REVISION GROUPED-SAMPLE VALIDATION")
print("="*80)
print("  Revised grouped-sample validation")

# Load data
print("\nLoading data...")
train_df = pd.read_csv(CALIB_CSV)
test_df = pd.read_csv(VALID_CSV)
# Keep metadata for traceability, but do NOT use it for modeling
NON_PREDICTOR_COLS = ['image_no', 'moisture', 'soil_type', 'Sample_No']

train_df_full = train_df.copy()
test_df_full = test_df.copy()

train_df = train_df.drop(
    columns=[c for c in NON_PREDICTOR_COLS if c in train_df.columns]
)

test_df = test_df.drop(
    columns=[c for c in NON_PREDICTOR_COLS if c in test_df.columns]
)

print("  Modeling columns:", train_df.columns.tolist())


print(f"  Calibration (for AutoML): {train_df.shape}")
print(f"  Validation (held out): {test_df.shape}")

# =====================================================================
# GROUP-AWARE INTERNAL MODEL SELECTION
# =====================================================================

print("\n" + "=" * 80)
print("GROUP-AWARE INTERNAL MODEL SELECTION")
print("=" * 80)

FEATURES = [
    c for c in train_df.columns
    if c != TARGET_COL
]

print(f"Selected predictors: {FEATURES}")

# GroupKFold works on the full calibration dataset,
# where Sample_No is still available.
groups = train_df_full[GROUP].values

gkf = GroupKFold(n_splits=4)

fold_results = []
all_model_results = []

for fold, (fit_idx, tune_idx) in enumerate(
    gkf.split(
        train_df_full,
        train_df_full[TARGET_COL],
        groups=groups
    ),
    start=1
):

    print("\n" + "=" * 80)
    print(f"GROUPED CV FOLD {fold}")
    print("=" * 80)

    fold_train_full = train_df_full.iloc[fit_idx].copy()
    fold_tune_full = train_df_full.iloc[tune_idx].copy()

    train_groups = set(fold_train_full[GROUP])
    tune_groups = set(fold_tune_full[GROUP])

    overlap = train_groups & tune_groups

    if overlap:
        raise ValueError(
            f"Group leakage in fold {fold}: {sorted(overlap)}"
        )

    print(
        f"Training soils ({len(train_groups)}): "
        f"{sorted(train_groups)}"
    )

    print(
        f"Tuning soils ({len(tune_groups)}): "
        f"{sorted(tune_groups)}"
    )

    print("Sample overlap: ZERO")

    # Remove Sample_No and metadata before AutoGluon
    fold_train = fold_train_full[
        [TARGET_COL] + FEATURES
    ].copy()

    fold_tune = fold_tune_full[
        [TARGET_COL] + FEATURES
    ].copy()

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    fold_path = os.path.join(
        GROUPED_MODEL_ROOT,
        f"fold_{fold}_{run_id}"
    )

    predictor_fold = TabularPredictor(
        label=TARGET_COL,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=fold_path
    ).fit(
        train_data=fold_train,

        # THIS is the important part:
        # AutoGluon does NOT create a random tuning split.
        tuning_data=fold_tune,

        time_limit=600,
        presets="medium_quality",
        verbosity=1
    )

    # -------------------------------------------------------------
    # Evaluate every AutoGluon model on this grouped tuning fold
    # -------------------------------------------------------------

    leaderboard_fold = predictor_fold.leaderboard(
        fold_tune,
        silent=True
    )

    y_true_fold = fold_tune[TARGET_COL].values

    for model_name in predictor_fold.model_names():

        try:

            y_pred_fold = predictor_fold.predict(
                fold_tune,
                model=model_name
            )

            rmse_fold = np.sqrt(
                mean_squared_error(
                    y_true_fold,
                    y_pred_fold
                )
            )

            mae_fold = mean_absolute_error(
                y_true_fold,
                y_pred_fold
            )

            # R² can be unstable with very small folds,
            # but save it for reporting.
            r2_fold = r2_score(
                y_true_fold,
                y_pred_fold
            )

            all_model_results.append({
                "Fold": fold,
                "Model": model_name,
                "RMSE": rmse_fold,
                "MAE": mae_fold,
                "R2": r2_fold,
                "N_Train_Groups": len(train_groups),
                "N_Tune_Groups": len(tune_groups)
            })

        except Exception as e:

            print(
                f"Could not evaluate "
                f"{model_name}: {e}"
            )

    best_fold_model = predictor_fold.model_best

    fold_results.append({
        "Fold": fold,
        "Best_Model": best_fold_model,
        "Train_Groups": len(train_groups),
        "Tune_Groups": len(tune_groups)
    })


# =====================================================================
# SUMMARIZE GROUPED CV
# =====================================================================

model_cv_df = pd.DataFrame(
    all_model_results
)

model_cv_df.to_csv(
    os.path.join(
        OUT_DIR,
        "grouped_cv_all_models.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("GROUPED CV MODEL SUMMARY")
print("=" * 80)

model_summary = (
    model_cv_df
    .groupby("Model")
    .agg(
        Mean_RMSE=("RMSE", "mean"),
        SD_RMSE=("RMSE", "std"),
        Mean_MAE=("MAE", "mean"),
        Mean_R2=("R2", "mean"),
        Folds_Evaluated=("Fold", "nunique")
    )
    .reset_index()
)

# Prefer models evaluated in all four folds.
model_summary = model_summary[
    model_summary["Folds_Evaluated"] == 4
].copy()

model_summary = model_summary.sort_values(
    "Mean_RMSE",
    ascending=True
)

print(
    model_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

model_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "grouped_cv_model_summary.csv"
    ),
    index=False
)

selected_model_name = (
    model_summary.iloc[0]["Model"]
)

print(
    f"\n🏆 GROUPED-CV SELECTED MODEL: "
    f"{selected_model_name}"
)

print(
    f"Mean grouped CV RMSE: "
    f"{model_summary.iloc[0]['Mean_RMSE']:.4f}"
)

# =====================================================================
# FINAL MODEL FIT ON ALL CALIBRATION DATA
# =====================================================================

print("\n" + "=" * 80)
print("FINAL MODEL FIT ON ALL CALIBRATION DATA")
print("=" * 80)

final_run_id = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

final_model_path = os.path.join(
    GROUPED_MODEL_ROOT,
    f"final_{final_run_id}"
)

final_predictor = TabularPredictor(
    label=TARGET_COL,
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    path=final_model_path
).fit(
    train_data=train_df,
    presets="medium_quality",
    time_limit=1800,
    verbosity=2
)

final_best_model = final_predictor.model_best

print(
    f"\nFinal full-calibration AutoGluon model: "
    f"{final_best_model}"
)





# NOW test on YOUR independent validation set
print("\n" + "="*80)
print(f"FINAL INDEPENDENT VALIDATION ({len(test_df)} soil-condition rows)")
print("="*80)

y_test = test_df[TARGET_COL].values
y_pred = final_predictor.predict(test_df)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
rpd = y_test.std() / rmse
q75, q25 = np.percentile(y_test, [75, 25])
rpiq = (q75 - q25) / rmse

sample_eval = test_df_full.copy()
sample_eval['Predicted_SOC'] = np.asarray(y_pred)

sample_results = (
    sample_eval
    .groupby(['Sample_No','soc'], as_index=False) 
    .agg(
        #actual_SOC=('soc', 'mean'),
        Predicted_SOC=('Predicted_SOC', 'mean'),
       # N_Images=('image_no', 'count')
    )
    .rename(columns={'soc': 'Actual_SOC'})
)

sample_r2 = r2_score(
    sample_results['Actual_SOC'],
    sample_results['Predicted_SOC']
)

sample_rmse = np.sqrt(
    mean_squared_error(
        sample_results['Actual_SOC'],
        sample_results['Predicted_SOC']
    )
)

sample_mae = mean_absolute_error(
    sample_results['Actual_SOC'],
    sample_results['Predicted_SOC']
)

print("\nSAMPLE-LEVEL RESULTS")
print(sample_results.to_string(index=False))
print(f"\nSample-level R²   = {sample_r2:.4f}")
print(f"Sample-level RMSE = {sample_rmse:.4f}")
print(f"Sample-level MAE  = {sample_mae:.4f}")

print(f"\n  Best Model: {final_best_model}")
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  MAE  = {mae:.4f}")
print(f"  RPD  = {rpd:.4f}")
print(f"  RPIQ = {rpiq:.4f}")

if rpd > 2.0:
    capability = "EXCELLENT"
elif rpd > 1.4:
    capability = "GOOD"
else:
    capability = "MODERATE"

print(f"\n  📊 Prediction Capability: {capability}")

# Save results
model_summary.to_csv(
    os.path.join(
        OUT_DIR,
        'grouped_cv_model_summary.csv'
    ),
    index=False
)

model_cv_df.to_csv(
    os.path.join(
        OUT_DIR,
        'grouped_cv_all_models.csv'
    ),
    index=False
)
    
results_df = pd.DataFrame([{
    'Best_Model': final_best_model,
   'Internal_Validation': '4-fold GroupKFold by Sample_No',
   'Independent_Validation': '4 held-out soil-condition rows from 4 physical samples', 
    'R²': r2,
    'RMSE': rmse,
    'MAE': mae,
    'RPD': rpd,
    'RPIQ': rpiq,
    'Capability': capability
}])
results_df.to_csv(os.path.join(OUT_DIR, 'FINAL_RESULTS.csv'), index=False)

# Save predictions
pred_df = test_df_full.copy()
pred_df['Predicted_SOC'] = y_pred
pred_df.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)

print(f"\n📁 Results saved to: {OUT_DIR}/")

print("\n" + "="*80)
print("✅ COMPLETE")
print("="*80)
print(f"Calibration rows used: {len(train_df)}")
print(f"Validation rows used: {len(test_df)}")
print(
    f"Calibration physical samples: "
    f"{train_df_full['Sample_No'].nunique()}"
)
print(
    f"Validation physical samples: "
    f"{test_df_full['Sample_No'].nunique()}"
)
print(f"\n  Best Model: {final_best_model}")
print(f"  R² = {r2:.4f}")
print(f"  RPD = {rpd:.4f} ({capability})")
print("\n" + "="*80)
"""
AutoML with CUSTOM COMPOSITE METRIC
====================================
AutoGluon will select models based on composite score:
0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE

Instead of just RMSE!
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
# Same CALIB_CSV and VALID_CSV as above
CALIB_CSV = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output_sensitivity\calibration_selected_features.csv'
VALID_CSV  = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output_sensitivity\validation_selected_features.csv'
OUT_DIR    = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step3_output_sensitivity\automl_COMPOSITE_METRIC'
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL = 'soc'

print("\n" + "="*80)
print("AUTOML WITH CUSTOM COMPOSITE METRIC")
print("="*80)
print("  AutoGluon will select models based on:")
print("  Composite = 0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE")
print("  NOT just RMSE!")
print("="*80)


# ═══════════════════════════════════════════════════════════════════════════
# DEFINE CUSTOM COMPOSITE METRIC
# ═══════════════════════════════════════════════════════════════════════════

# Weights (same as your multi-metric selection)
WEIGHTS = {
    'RPD':   0.35,
    'RPIQ':  0.25,
    'RMSE':  0.20,
    'R²':    0.15,
    'MAE':   0.05
}

print(f"\n⚖️  WEIGHTING:")
for metric, weight in WEIGHTS.items():
    print(f"  {metric:6s}: {weight:.0%}")


def composite_score_metric(y_true, y_pred, sample_weight=None):
    """
    Custom metric: Composite score for soil science
    
    Higher is better!
    
    Combines:
    - RPD (35%) - soil science standard
    - RPIQ (25%) - robust performance
    - RMSE (20%) - prediction error
    - R² (15%) - variance explained
    - MAE (5%) - average error
    
    Returns a score where HIGHER = BETTER
    """
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate individual metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # RPD
    rpd = y_true.std() / rmse if rmse > 0 else 0
    
    # RPIQ
    q75, q25 = np.percentile(y_true, [75, 25])
    rpiq = (q75 - q25) / rmse if rmse > 0 else 0
    
    # Normalize each metric to 0-1 scale
    # We need reference values - use typical ranges for SOC prediction
    
    # RPD: 0-4 range (typical in soil science)
    rpd_norm = np.clip(rpd / 4.0, 0, 1)
    
    # RPIQ: 0-4 range
    rpiq_norm = np.clip(rpiq / 4.0, 0, 1)
    
    # RMSE: inverse (lower is better), assume 0-2 g/kg range
    rmse_norm = 1 - np.clip(rmse / 2.0, 0, 1)
    
    # R²: already 0-1, but can be negative
    r2_norm = np.clip(r2, 0, 1)
    
    # MAE: inverse (lower is better), assume 0-2 g/kg range
    mae_norm = 1 - np.clip(mae / 2.0, 0, 1)
    
    # Calculate composite (weighted sum)
    composite = (
        WEIGHTS['RPD'] * rpd_norm +
        WEIGHTS['RPIQ'] * rpiq_norm +
        WEIGHTS['RMSE'] * rmse_norm +
        WEIGHTS['R²'] * r2_norm +
        WEIGHTS['MAE'] * mae_norm
    )
    
    # Return composite (higher is better)
    return composite


# Create AutoGluon scorer from custom metric
composite_scorer = make_scorer(
    name='composite_score',
    score_func=composite_score_metric,
    optimum=1.0,          # Best possible score
    greater_is_better=True  # Higher is better
)

print("\n✅ Custom composite metric defined")
print("   AutoGluon will now optimize this instead of RMSE!")


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

train_df = pd.read_csv(CALIB_CSV)
test_df = pd.read_csv(VALID_CSV)

NON_PREDICTOR_COLS = ['image_no', 'moisture', 'soil_type', 'Sample_No']

train_df_full = train_df.copy()
test_df_full = test_df.copy()

train_df = train_df.drop(
    columns=[c for c in NON_PREDICTOR_COLS if c in train_df.columns]
)

test_df = test_df.drop(
    columns=[c for c in NON_PREDICTOR_COLS if c in test_df.columns]
)

print("  Modeling columns:", train_df.columns.tolist())


print(f"  Calibration: {train_df.shape}")
print(f"  Validation: {test_df.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN AUTOGLUON WITH CUSTOM METRIC
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TRAINING AUTOML WITH CUSTOM COMPOSITE METRIC")
print("="*80)
print("  Time limit: 30 minutes")
print("  Selection criterion: COMPOSITE SCORE (not RMSE)")

from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = os.path.join(
    COMPOSITE_MODEL_ROOT,
    f"models_{run_id}"
)

os.makedirs(model_path, exist_ok=True)

print(f"AutoGluon model path: {model_path}")

predictor = TabularPredictor(
    label=TARGET_COL,
    eval_metric=composite_scorer,  # ✅ CUSTOM METRIC!
    path=model_path
).fit(
    train_data=train_df,
    time_limit=1800,
    presets='medium_quality',
    verbosity=2
)

print("\n✅ Training complete using COMPOSITE METRIC!")


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("AUTOML RESULTS (Selected by Composite Score)")
print("="*80)

leaderboard = predictor.leaderboard(train_df, silent=True)
print("\nTOP 10 MODELS (ranked by COMPOSITE SCORE):")
print(leaderboard[['model', 'score_val', 'pred_time_val']].head(10))

final_best_model = predictor.model_best
print(f"\n🏆 AutoML selected (by composite): {final_best_model}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST ON INDEPENDENT VALIDATION SET
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print(f"INDEPENDENT VALIDATION ({len(test_df)} rows)")
print("="*80)

y_test = test_df[TARGET_COL].values
y_pred = predictor.predict(test_df)

# Calculate all metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
rpd = y_test.std() / rmse
q75, q25 = np.percentile(y_test, [75, 25])
rpiq = (q75 - q25) / rmse

# Calculate composite score
composite = composite_score_metric(y_test, y_pred)

print(f"\n  Best Model: {final_best_model}")
print(f"\n  COMPOSITE METRICS:")
print(f"    RPD  = {rpd:.4f} (weight: {WEIGHTS['RPD']:.0%})")
print(f"    RPIQ = {rpiq:.4f} (weight: {WEIGHTS['RPIQ']:.0%})")
print(f"    RMSE = {rmse:.4f} (weight: {WEIGHTS['RMSE']:.0%})")
print(f"    R²   = {r2:.4f} (weight: {WEIGHTS['R²']:.0%})")
print(f"    MAE  = {mae:.4f} (weight: {WEIGHTS['MAE']:.0%})")
print(f"\n  📊 COMPOSITE SCORE = {composite:.4f}")

if rpd > 2.0:
    capability = "EXCELLENT"
elif rpd > 1.4:
    capability = "GOOD"
else:
    capability = "MODERATE"

print(f"\n  📈 Prediction Capability: {capability} (RPD-based)")


# ═══════════════════════════════════════════════════════════════════════════
# DETAILED LEADERBOARD WITH ALL METRICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("DETAILED LEADERBOARD (All Metrics)")
print("="*80)

# Get predictions for all models
model_names = leaderboard['model'].tolist()

detailed_results = []

for model_name in model_names:
    try:
        y_pred_model = predictor.predict(test_df, model=model_name)
        
        mae_m = mean_absolute_error(y_test, y_pred_model)
        rmse_m = np.sqrt(mean_squared_error(y_test, y_pred_model))
        r2_m = r2_score(y_test, y_pred_model)
        rpd_m = y_test.std() / rmse_m
        rpiq_m = (q75 - q25) / rmse_m
        composite_m = composite_score_metric(y_test, y_pred_model)
        
        detailed_results.append({
            'Model': model_name,
            'Composite': composite_m,
            'RPD': rpd_m,
            'RPIQ': rpiq_m,
            'RMSE': rmse_m,
            'R²': r2_m,
            'MAE': mae_m
        })
    except:
        pass

df_detailed = pd.DataFrame(detailed_results)
df_detailed = df_detailed.sort_values('Composite', ascending=False)
df_detailed.insert(0, 'Rank', range(1, len(df_detailed) + 1))

print("\n" + df_detailed.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

leaderboard.to_csv(os.path.join(OUT_DIR, 'model_leaderboard_composite.csv'), index=False)
df_detailed.to_csv(os.path.join(OUT_DIR, 'detailed_metrics_all_models.csv'), index=False)

results_df = pd.DataFrame([{
    'Best_Model': final_best_model,
    'Selection_Criterion': 'Composite Score (0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE)',
    'Composite_Score': composite,
    'RPD': rpd,
    'RPIQ': rpiq,
    'RMSE': rmse,
    'R²': r2,
    'MAE': mae,
    'Capability': capability
}])
results_df.to_csv(os.path.join(OUT_DIR, 'FINAL_RESULTS_COMPOSITE.csv'), index=False)

# Save predictions
pred_df = test_df_full.copy()
pred_df['Predicted_SOC'] = y_pred
pred_df.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)

print(f"\n📁 Results saved to: {OUT_DIR}/")
print(f"  Models saved at: {model_path}")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("✅ AUTOML COMPLETE (CUSTOM COMPOSITE METRIC)")
print("="*80)

print(f"\n  🎯 KEY DIFFERENCE:")
print(f"     Standard AutoML: Selects by RMSE only")
print(f"     This AutoML: Selects by COMPOSITE SCORE")
print(f"     (35% RPD + 25% RPIQ + 20% RMSE + 15% R² + 5% MAE)")

print(f"\n  🏆 BEST MODEL: {final_best_model}")
print(f"     Composite Score = {composite:.4f}")
print(f"     RPD = {rpd:.4f} ({capability})")
print(f"     R² = {r2:.4f}")
print(f"     RMSE = {rmse:.4f}")

print(f"\n  ✅ Model was selected based on ALL 5 metrics")
print(f"     NOT just RMSE!")

print("\n" + "="*80)



