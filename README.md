# Assessing-Smartphone-Image-Based-Prediction-of-Soil-Organic-Carbon-Content-Across-Unseen-Soils

This study quantified how sample dependence influences reported prediction performance and characterized prediction errors for entirely unseen soils.

Abstract Purpose Conventional soil organic carbon (SOC) quantification relies on costly laboratory analysis, and smartphone image-based prediction has been proposed as a scalable alternative. However, reported accuracies in this literature derive almost exclusively from image-level validation, in which repeated images of the same physical soil can appear on both sides of model calibration and evaluation, leaving the generalization of such models to unseen soils untested. ****Aim **** This study quantified how sample dependence influences reported prediction performance and characterized prediction errors for entirely unseen soils. Methods A dataset of 731 smartphone images from 20 physical soil samples (three Ohio soil series; field and laboratory acquisition; multiple soil moisture stages) was modeled using an automated machine learning framework with a fixed four-stage feature-selection pipeline. Identical models and data were evaluated under image-level Kennard–Stone validation and sample-level grouped validation in which four physical soils, including the highest-SOC soil, were held out entirely. Physical-soil-level cluster bootstraps, equal-soil summaries, and a calibration-mean baseline were used for uncertainty and reference comparisons. Results Under image-level validation, the model appeared highly accurate (R² = 0.827; RMSE = 0.44% SOC), whereas the identical model collapsed for unseen soils (R² = −0.206; RMSE = 1.95% SOC) and did not outperform the calibration-mean baseline (ΔRMSE 95% CI included zero). Error characteristics for unseen soils varied descriptively with moisture (RMSE 2.06%, 1.95%, and 1.62% SOC in dry, moist, and wet classes) and between field and laboratory acquisition (RMSE 1.10 vs. 2.15% SOC), but neither factor produced reliable prediction, and the raw field–laboratory difference was reproduced by the baseline itself, indicating it reflected differing SOC distributions rather than acquisition effects. **Conclusions ** Validation design, rather than modeling complexity, governed reported accuracy: image-derived color and texture feature captured SOC-related information for new images of known soils but did not transfer to unseen soils. Image counts overstate dataset informativeness in repeated-imaging designs; accuracies from image-level splits should be interpreted as within-sample recognition rather than generalization, and progress toward field-deployable smartphone SOC estimation requires larger, independent, sample-level-validated datasets.


In notebooks

Main Pipeline (grouped sample split):

step1_outlier_removal.py — explains the grouped split logic, Mahalanobis on calibration only with 31 image features, validation samples 3/7/13/15 held out entirely
step2_feature_selection.py — all 4 stages explained with citations, 31→9 reduction documented
step3_run_automl.py — GroupKFold by Sample_No, composite re-ranking, final metrics
step4_statistical_validation_sample_grouped.py — Friedman + Wilcoxon tests

Kennard-Stone Pipeline:

step1_outlier_removal_ks.py — explains 80/20 ratio chosen to match grouped split, all 20 samples straddle the split intentionally
step2_feature_selection_ks.py — same 4-stage method, 31→8 features, 88% overlap with grouped
step3_run_automl_ks.py — same AutoML setup, R²=0.83 result shown

Sensitivity Analyses:

step1_sensitivity_exclude_high_soc.py — removes Sample 7 (SOC > 6%), explains why
step2_feature_selection_sensitivity.py
step3_run_automl_sensitivity.py
step1_outlier_removal_threshold.py — removes all SOC > 3%, explains which samples removed
step2_feature_selection_threshold.py
step3_run_automl_threshold.py
