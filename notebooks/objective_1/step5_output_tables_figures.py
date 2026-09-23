
"""
PUBLICATION TABLES + FIGURES
============================

Creates publication-ready tables and figures for the completed:

1) Sample-grouped validation analysis
2) Kennard-Stone image-level validation analysis

This script intentionally does NOT perform the moisture-class or field-vs-laboratory
analyses. Those can be added after the current sample-based and image-based results
are finalized.

OUTPUT STRUCTURE
----------------
publication_outputs/
    tables/
        Table2A_AutoGluon_models_sample_grouped.csv
        Table2B_AutoGluon_models_KS_image.csv
        Table3_descriptive_statistics.csv
        Table4A_selected_features_side_by_side.csv
        Table4B_feature_selection_stage_summary.csv
        Table5_sample_grouped_model_performance.csv
        Table6_KS_image_model_performance.csv
        Table7_validation_design_comparison.csv
        Table8A_bootstrap_confidence_intervals.csv
        Table8B_baseline_and_statistical_tests.csv
        Supplement_feature_selection_detailed.csv
        Supplement_feature_importance_sample_grouped.csv
        Supplement_feature_importance_KS_image.csv

    figures/
        Fig4_sample_grouped_observed_vs_predicted.png/.tiff
        Fig5_KS_image_observed_vs_predicted.png/.tiff
        Fig6_sample_grouped_feature_importance.png/.tiff
        Fig7_KS_image_feature_importance.png/.tiff
        Fig8A_validation_comparison_fit_metrics.png/.tiff
        Fig8B_validation_comparison_error_metrics.png/.tiff
        Fig9A_bootstrap_R2_confidence_intervals.png/.tiff
        Fig9B_bootstrap_RMSE_confidence_intervals.png/.tiff

    publication_captions.txt

NOTES
-----
- Observed-vs-predicted plots include BOTH:
      (1) 1:1 agreement line
      (2) fitted linear regression line
- Feature importance is calculated as permutation importance on the final
  validation data for the locked AutoGluon model:
      importance = increase in RMSE after shuffling the predictor
  The importance plots are descriptive post-hoc interpretation and are not used
  for feature selection or model selection.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# =============================================================================
# USER PATHS
# =============================================================================

PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OBJ_DIR = os.path.join(PROJECT_ROOT, "notebooks", "objective_1", "output_data")

# Raw source files for descriptive statistics
FEATURES_CSV = os.path.join(
    RAW_DIR,
    "soil_image_features_without_commas.csv",
)

METADATA_CSV = os.path.join(
    RAW_DIR,
    "image_with_soc_metadata.csv",
)

# -------------------------------------------------------------------------
# SAMPLE-GROUPED outputs
# -------------------------------------------------------------------------

SAMPLE_STEP2_DIR = os.path.join(OBJ_DIR, "step2_output")
SAMPLE_STEP3_DIR = os.path.join(OBJ_DIR, "step3_output")
SAMPLE_STEP4_DIR = os.path.join(
    OBJ_DIR,
    "step4_statistical_validation_sample_grouped",
)

# -------------------------------------------------------------------------
# KS IMAGE-LEVEL outputs
# -------------------------------------------------------------------------

KS_STEP2_DIR = os.path.join(OBJ_DIR, "step2_output_ks")
KS_STEP3_DIR = os.path.join(OBJ_DIR, "step3_output_ks")
KS_STEP4_DIR = os.path.join(
    OBJ_DIR,
    "step4_statistical_validation_image_ks",
)

# -------------------------------------------------------------------------
# Publication outputs
# -------------------------------------------------------------------------

PUB_DIR = os.path.join(OBJ_DIR, "step5_output_tables_figures")
TABLE_DIR = os.path.join(PUB_DIR, "tables")
FIG_DIR = os.path.join(PUB_DIR, "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

TARGET = "soc"
PRED = "Predicted_SOC"
GROUP = "Sample_No"

FIG_DPI = 600
N_IMPORTANCE_PERMUTATIONS = 30
RANDOM_SEED = 42

# The 31 numerical image-derived predictors used throughout the study.
IMAGE_FEATURES = [
    "mean_r", "mean_g", "mean_b",
    "median_r", "median_g", "median_b",
    "mean_h", "mean_s", "mean_v",
    "median_h", "median_s", "median_v",
    "mean_gray", "median_gray",
    "contrast", "energy", "homogeneity", "entropy",
    "RI", "CI", "HI", "SI",
    "lab_L", "lab_a", "lab_b",
    "luv_L", "luv_u", "luv_v",
    "xyz_X", "xyz_Y", "xyz_Z",
]

METADATA_EXCLUDE = [
    "image_no",
    "Image_No",
    "moisture",
    "soil_type",
    "Sample_No",
]


# =============================================================================
# MODEL DESCRIPTIONS FOR TABLE 2
# =============================================================================

MODEL_INFO = {
    "LightGBM": (
        "Gradient boosting",
        "Leaf-wise gradient-boosted decision trees with histogram-based splitting.",
    ),
    "LightGBMXT": (
        "Gradient boosting",
        "LightGBM extra-trees variant using increased split randomness.",
    ),
    "LightGBMLarge": (
        "Gradient boosting",
        "Larger LightGBM configuration with increased model capacity.",
    ),
    "XGBoost": (
        "Gradient boosting",
        "Regularized gradient-boosted decision trees using depth-wise growth.",
    ),
    "CatBoost": (
        "Gradient boosting",
        "Gradient boosting with ordered boosting and symmetric tree construction.",
    ),
    "RandomForestMSE": (
        "Bagged tree ensemble",
        "Bootstrap-aggregated regression trees optimized using squared-error loss.",
    ),
    "ExtraTreesMSE": (
        "Randomized tree ensemble",
        "Extremely randomized regression trees optimized using squared-error loss.",
    ),
    "NeuralNetFastAI": (
        "Neural network",
        "Feed-forward neural network implemented through the FastAI backend.",
    ),
    "NeuralNetTorch": (
        "Neural network",
        "Multi-layer feed-forward neural network implemented with PyTorch.",
    ),
    "WeightedEnsemble_L2": (
        "Stacked ensemble",
        "AutoGluon level-2 weighted ensemble combining predictions from base models.",
    ),
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def require_file(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nMissing {description}:\n{path}\n"
            "Check that the corresponding analysis step has been run."
        )
    return path


def read_csv_required(path, description):
    require_file(path, description)
    return pd.read_csv(path)


def save_figure(fig, stem):
    """Save both high-resolution PNG and TIFF."""
    png = os.path.join(FIG_DIR, f"{stem}.png")
    tif = os.path.join(FIG_DIR, f"{stem}.tiff")

    fig.savefig(
        png,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        tif,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"  Saved: {png}")
    print(f"  Saved: {tif}")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    sd_y = np.std(y_true, ddof=1)
    rpd = sd_y / rmse if rmse > 0 else np.nan

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


def read_selected_features(step2_dir):
    path = os.path.join(
        step2_dir,
        "selected_features_list.txt",
    )
    require_file(path, "selected feature list")

    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def latest_autogluon_run(step3_dir):
    """Return newest final_full_calibration/run_* directory."""
    base = os.path.join(
        step3_dir,
        "final_full_calibration",
    )

    candidates = [
        p
        for p in glob.glob(
            os.path.join(base, "run_*")
        )
        if os.path.isdir(p)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No final AutoGluon run directories found under:\n{base}"
        )

    return max(
        candidates,
        key=os.path.getmtime,
    )


def selected_model_from_ranking(step3_dir):
    path = os.path.join(
        step3_dir,
        "grouped_cv_model_composite_ranking.csv",
    )
    df = read_csv_required(
        path,
        "grouped CV model ranking",
    )

    if "Model" not in df.columns or len(df) == 0:
        raise ValueError(
            f"Model ranking file does not contain a usable Model column:\n{path}"
        )

    return str(df.iloc[0]["Model"])


# =============================================================================
# TABLE 2A / 2B — AutoGluon models used
# =============================================================================

def make_model_inventory(step3_dir, validation_design, output_name):
    ranking_path = os.path.join(
        step3_dir,
        "grouped_cv_model_composite_ranking.csv",
    )

    ranking = read_csv_required(
        ranking_path,
        f"{validation_design} model ranking",
    )

    models = ranking["Model"].astype(str).tolist()

    rows = []

    for model in models:
        family, characteristics = MODEL_INFO.get(
            model,
            ("AutoGluon model", "AutoGluon candidate regression model."),
        )

        rows.append({
            "Model": model,
            "Family": family,
            "Key characteristics": characteristics,
            "Validation framework": validation_design,
        })

    table = pd.DataFrame(rows)

    path = os.path.join(
        TABLE_DIR,
        output_name,
    )
    table.to_csv(
        path,
        index=False,
    )

    print(f"  Saved: {path}")
    return table


# =============================================================================
# TABLE 3 — Descriptive statistics
# =============================================================================

def make_descriptive_statistics():
    feat = read_csv_required(
        FEATURES_CSV,
        "raw image-feature file",
    )
    meta = read_csv_required(
        METADATA_CSV,
        "SOC metadata file",
    )

    master = pd.merge(
        feat,
        meta,
        left_on="Numeric numbers",
        right_on="image_no",
        how="inner",
    )

    master = (
        master
        .drop_duplicates(subset=["image_no"])
        .reset_index(drop=True)
    )

    wanted = [
        c
        for c in IMAGE_FEATURES + [TARGET, "moisture"]
        if c in master.columns
    ]

    rows = []

    for c in wanted:
        x = pd.to_numeric(
            master[c],
            errors="coerce",
        ).dropna()

        if len(x) == 0:
            continue

        mean = x.mean()
        sd = x.std(ddof=1)
        q25 = x.quantile(0.25)
        q75 = x.quantile(0.75)

        cv = (
            abs(sd / mean) * 100
            if not np.isclose(mean, 0)
            else np.nan
        )

        rows.append({
            "Feature": c,
            "N": len(x),
            "Mean": mean,
            "SD": sd,
            "Mean ± SD": f"{mean:.3f} ± {sd:.3f}",
            "Minimum": x.min(),
            "Maximum": x.max(),
            "IQR": q75 - q25,
            "Skewness": x.skew(),
            "CV (%)": cv,
        })

    table = pd.DataFrame(rows)

    path = os.path.join(
        TABLE_DIR,
        "Table3_descriptive_statistics.csv",
    )
    table.to_csv(
        path,
        index=False,
    )

    print(f"  Saved: {path}")
    return table


# =============================================================================
# TABLE 4 — Feature-selection results
# =============================================================================

def make_feature_selection_tables():
    sample_features = read_selected_features(
        SAMPLE_STEP2_DIR
    )
    ks_features = read_selected_features(
        KS_STEP2_DIR
    )

    max_len = max(
        len(sample_features),
        len(ks_features),
    )

    rows = []

    for i in range(max_len):
        sf = (
            sample_features[i]
            if i < len(sample_features)
            else ""
        )
        kf = (
            ks_features[i]
            if i < len(ks_features)
            else ""
        )

        rows.append({
            "Rank": i + 1,
            "Sample-grouped selected feature": sf,
            "KS image-level selected feature": kf,
            "Common to both final sets": (
                "Yes"
                if sf and sf in ks_features
                else ""
            ),
        })

    table_a = pd.DataFrame(rows)

    path_a = os.path.join(
        TABLE_DIR,
        "Table4A_selected_features_side_by_side.csv",
    )

    table_a.to_csv(
        path_a,
        index=False,
    )

    # Stage counts
    sample_stage2 = read_csv_required(
        os.path.join(
            SAMPLE_STEP2_DIR,
            "stage2_soc_screening.csv",
        ),
        "sample-grouped Stage-2 report",
    )

    sample_stage3 = read_csv_required(
        os.path.join(
            SAMPLE_STEP2_DIR,
            "stage3_pca_feature_contributions.csv",
        ),
        "sample-grouped Stage-3 report",
    )

    ks_stage2 = read_csv_required(
        os.path.join(
            KS_STEP2_DIR,
            "stage2_soc_screening.csv",
        ),
        "KS Stage-2 report",
    )

    ks_stage3 = read_csv_required(
        os.path.join(
            KS_STEP2_DIR,
            "stage3_pca_feature_contributions.csv",
        ),
        "KS Stage-3 report",
    )

    table_b = pd.DataFrame([
        {
            "Validation design": "Sample-grouped",
            "Initial predictors": 31,
            "Stage 1 retained": 31,
            "Stage 2 retained": int(sample_stage2["stage2_pass"].sum()),
            "PCA retained": int(sample_stage3["PCA_pass"].sum()),
            "Final retained": len(sample_features),
        },
        {
            "Validation design": "KS image-level",
            "Initial predictors": 31,
            "Stage 1 retained": 31,
            "Stage 2 retained": int(ks_stage2["stage2_pass"].sum()),
            "PCA retained": int(ks_stage3["PCA_pass"].sum()),
            "Final retained": len(ks_features),
        },
    ])

    path_b = os.path.join(
        TABLE_DIR,
        "Table4B_feature_selection_stage_summary.csv",
    )

    table_b.to_csv(
        path_b,
        index=False,
    )

    # Detailed supplementary comparison for selected features only
    def detailed(step2_dir, final_features, design):
        s2 = pd.read_csv(
            os.path.join(
                step2_dir,
                "stage2_soc_screening.csv",
            )
        )
        s3 = pd.read_csv(
            os.path.join(
                step2_dir,
                "stage3_pca_feature_contributions.csv",
            )
        )

        out = (
            s2.merge(
                s3[
                    [
                        "feature",
                        "contribution_PC1",
                        "contribution_PC2",
                        "PCA_pass",
                    ]
                ],
                on="feature",
                how="left",
            )
        )

        out = out[
            out["feature"].isin(final_features)
        ].copy()

        rank_map = {
            f: i + 1
            for i, f in enumerate(final_features)
        }

        out.insert(
            0,
            "Final_rank",
            out["feature"].map(rank_map),
        )

        out.insert(
            0,
            "Validation_design",
            design,
        )

        return out.sort_values(
            "Final_rank"
        )

    supplement = pd.concat(
        [
            detailed(
                SAMPLE_STEP2_DIR,
                sample_features,
                "Sample-grouped",
            ),
            detailed(
                KS_STEP2_DIR,
                ks_features,
                "KS image-level",
            ),
        ],
        ignore_index=True,
    )

    supplement_path = os.path.join(
        TABLE_DIR,
        "Supplement_feature_selection_detailed.csv",
    )

    supplement.to_csv(
        supplement_path,
        index=False,
    )

    print(f"  Saved: {path_a}")
    print(f"  Saved: {path_b}")
    print(f"  Saved: {supplement_path}")

    return table_a, table_b, supplement


# =============================================================================
# TABLE 5 / 6 — Model performance rankings
# =============================================================================

def publication_model_ranking(step3_dir, output_name):
    src = read_csv_required(
        os.path.join(
            step3_dir,
            "grouped_cv_model_composite_ranking.csv",
        ),
        "AutoGluon composite ranking",
    )

    rename = {
        "Composite_Rank": "Composite rank",
        "RMSE_Rank": "RMSE rank",
        "Model": "Model",
        "Composite_Score": "Composite score",
        "Mean_RMSE": "Mean RMSE",
        "Mean_MAE": "Mean MAE",
        "Mean_R2": "Mean R²",
        "Mean_RPD": "Mean RPD",
        "Mean_RPIQ": "Mean RPIQ",
    }

    cols = [
        c
        for c in rename.keys()
        if c in src.columns
    ]

    table = (
        src[cols]
        .rename(columns=rename)
        .copy()
    )

    numeric_cols = [
        c
        for c in table.columns
        if c not in [
            "Model",
            "Composite rank",
            "RMSE rank",
        ]
    ]

    table[numeric_cols] = (
        table[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .round(4)
    )

    path = os.path.join(
        TABLE_DIR,
        output_name,
    )

    table.to_csv(
        path,
        index=False,
    )

    print(f"  Saved: {path}")
    return table


# =============================================================================
# TABLE 7 — Direct sample-vs-KS validation comparison
# =============================================================================

def validation_summary_from_predictions(pred_path, n_overlap, design, model_name):
    df = read_csv_required(
        pred_path,
        f"{design} external validation predictions",
    )

    m = regression_metrics(
        df[TARGET],
        df[PRED],
    )

    return {
        "Validation design": design,
        "Selected model": model_name,
        "Validation images": len(df),
        "Physical soils represented": (
            df[GROUP].nunique()
            if GROUP in df.columns
            else np.nan
        ),
        "Physical soils overlapping calibration": n_overlap,
        "R²": m["R2"],
        "RMSE": m["RMSE"],
        "MAE": m["MAE"],
        "RPD": m["RPD"],
        "RPIQ": m["RPIQ"],
        "Bias": m["Bias"],
    }


def make_validation_comparison():
    sample_pred = os.path.join(
        SAMPLE_STEP3_DIR,
        "external_validation_predictions.csv",
    )

    ks_pred = os.path.join(
        KS_STEP3_DIR,
        "external_validation_predictions.csv",
    )

    sample_model = selected_model_from_ranking(
        SAMPLE_STEP3_DIR
    )

    ks_model = selected_model_from_ranking(
        KS_STEP3_DIR
    )

    rows = [
        validation_summary_from_predictions(
            sample_pred,
            n_overlap=0,
            design="Sample-grouped external validation",
            model_name=sample_model,
        ),
        validation_summary_from_predictions(
            ks_pred,
            n_overlap=20,
            design="KS image-level validation",
            model_name=ks_model,
        ),
    ]

    table = pd.DataFrame(rows)

    metric_cols = [
        "R²",
        "RMSE",
        "MAE",
        "RPD",
        "RPIQ",
        "Bias",
    ]

    table[metric_cols] = (
        table[metric_cols]
        .round(4)
    )

    path = os.path.join(
        TABLE_DIR,
        "Table7_validation_design_comparison.csv",
    )

    table.to_csv(
        path,
        index=False,
    )

    print(f"  Saved: {path}")
    return table


# =============================================================================
# TABLE 8 — Statistical validation
# =============================================================================

def make_statistical_validation_tables():
    sample_ci = read_csv_required(
        os.path.join(
            SAMPLE_STEP4_DIR,
            "image_level_cluster_bootstrap_summary.csv",
        ),
        "sample-grouped bootstrap summary",
    )

    ks_ci = read_csv_required(
        os.path.join(
            KS_STEP4_DIR,
            "image_level_cluster_bootstrap_summary.csv",
        ),
        "KS bootstrap summary",
    )

    sample_ci.insert(
        0,
        "Validation design",
        "Sample-grouped",
    )

    ks_ci.insert(
        0,
        "Validation design",
        "KS image-level",
    )

    table_a = pd.concat(
        [
            sample_ci,
            ks_ci,
        ],
        ignore_index=True,
    )

    table_a = table_a.rename(
        columns={
            "CI_Lower": "95% CI lower",
            "CI_Upper": "95% CI upper",
        }
    )

    path_a = os.path.join(
        TABLE_DIR,
        "Table8A_bootstrap_confidence_intervals.csv",
    )

    table_a.to_csv(
        path_a,
        index=False,
    )

    # Baseline comparisons
    sample_base = read_csv_required(
        os.path.join(
            SAMPLE_STEP4_DIR,
            "baseline_comparison_summary.csv",
        ),
        "sample-grouped baseline comparison",
    )

    ks_base = read_csv_required(
        os.path.join(
            KS_STEP4_DIR,
            "baseline_comparison_summary.csv",
        ),
        "KS baseline comparison",
    )

    sample_base.insert(
        0,
        "Validation design",
        "Sample-grouped",
    )

    ks_base.insert(
        0,
        "Validation design",
        "KS image-level",
    )

    baseline_combined = pd.concat(
        [
            sample_base,
            ks_base,
        ],
        ignore_index=True,
    )

    # Inferential tests are different because the independent-unit counts differ.
    inferential_rows = []

    sample_signflip_path = os.path.join(
        SAMPLE_STEP4_DIR,
        "exact_signflip_model_vs_baseline.csv",
    )

    if os.path.exists(sample_signflip_path):
        s = pd.read_csv(
            sample_signflip_path
        )

        if len(s):
            inferential_rows.append({
                "Validation design": "Sample-grouped",
                "Test": "Exact soil-level sign-flip test",
                "Error metric": "MSE",
                "Independent physical soils": 4,
                "Statistic": (
                    s.iloc[0].get(
                        "Observed_Mean_MSE_Difference",
                        np.nan,
                    )
                ),
                "Two-sided p-value": (
                    s.iloc[0].get(
                        "Exact_TwoSided_P",
                        np.nan,
                    )
                ),
            })

    ks_wilcox_path = os.path.join(
        KS_STEP4_DIR,
        "soil_level_wilcoxon_results.csv",
    )

    if os.path.exists(ks_wilcox_path):
        w = pd.read_csv(
            ks_wilcox_path
        )

        for _, row in w.iterrows():
            error_metric = (
                "MSE"
                if "MSE" in str(row.get("Comparison", ""))
                else "MAE"
            )

            inferential_rows.append({
                "Validation design": "KS image-level",
                "Test": "Paired soil-level Wilcoxon signed-rank test",
                "Error metric": error_metric,
                "Independent physical soils": row.get(
                    "N_Physical_Soils",
                    20,
                ),
                "Statistic": row.get(
                    "Wilcoxon_Statistic",
                    np.nan,
                ),
                "Two-sided p-value": row.get(
                    "Two_Sided_P",
                    np.nan,
                ),
            })

    tests = pd.DataFrame(
        inferential_rows
    )

    # Stack baseline comparisons and tests into one publication file.
    baseline_display = baseline_combined.copy()
    baseline_display.insert(
        1,
        "Section",
        "Bootstrap model-vs-baseline difference",
    )

    # Standardize columns for output.
    rows = []

    for _, row in baseline_display.iterrows():
        rows.append({
            "Validation design": row.get(
                "Validation design"
            ),
            "Analysis": row.get(
                "Comparison"
            ),
            "Estimate / statistic": row.get(
                "Observed_Difference",
                np.nan,
            ),
            "95% CI lower": row.get(
                "CI_Lower",
                np.nan,
            ),
            "95% CI upper": row.get(
                "CI_Upper",
                np.nan,
            ),
            "Two-sided p-value": np.nan,
        })

    for _, row in tests.iterrows():
        rows.append({
            "Validation design": row.get(
                "Validation design"
            ),
            "Analysis": (
                f"{row.get('Test')} "
                f"({row.get('Error metric')})"
            ),
            "Estimate / statistic": row.get(
                "Statistic",
                np.nan,
            ),
            "95% CI lower": np.nan,
            "95% CI upper": np.nan,
            "Two-sided p-value": row.get(
                "Two-sided p-value",
                np.nan,
            ),
        })

    table_b = pd.DataFrame(rows)

    path_b = os.path.join(
        TABLE_DIR,
        "Table8B_baseline_and_statistical_tests.csv",
    )

    table_b.to_csv(
        path_b,
        index=False,
    )

    print(f"  Saved: {path_a}")
    print(f"  Saved: {path_b}")

    return table_a, table_b


# =============================================================================
# FIGURES 4 / 5 — Observed vs predicted with 1:1 + regression line
# =============================================================================

def observed_vs_predicted_figure(
    pred_path,
    title,
    stem,
):
    df = read_csv_required(
        pred_path,
        f"{title} predictions",
    )

    y_true = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    ).values

    y_pred = pd.to_numeric(
        df[PRED],
        errors="coerce",
    ).values

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    m = regression_metrics(
        y_true,
        y_pred,
    )

    slope, intercept = np.polyfit(
        y_true,
        y_pred,
        1,
    )

    data_min = min(
        y_true.min(),
        y_pred.min(),
    )

    data_max = max(
        y_true.max(),
        y_pred.max(),
    )

    margin = (
        0.05 * (data_max - data_min)
        if data_max > data_min
        else 0.1
    )

    lo = data_min - margin
    hi = data_max + margin

    x_line = np.linspace(
        lo,
        hi,
        200,
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 6.0)
    )

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.70,
        s=28,
        label="Validation images",
    )

    # Reviewer-requested 1:1 line.
    ax.plot(
        x_line,
        x_line,
        linestyle="--",
        linewidth=1.7,
        label="1:1 line",
    )

    # Fitted regression line.
    ax.plot(
        x_line,
        slope * x_line + intercept,
        linewidth=1.8,
        label="Linear regression",
    )

    ax.set_xlim(
        lo,
        hi,
    )
    ax.set_ylim(
        lo,
        hi,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "Observed SOC (%)"
    )
    ax.set_ylabel(
        "Predicted SOC (%)"
    )

    ax.set_title(
        title
    )

    annotation = (
        f"N = {len(y_true)}\n"
        f"R² = {m['R2']:.3f}\n"
        f"RMSE = {m['RMSE']:.3f}\n"
        f"MAE = {m['MAE']:.3f}\n"
        f"RPD = {m['RPD']:.3f}\n"
        f"RPIQ = {m['RPIQ']:.3f}\n"
        f"Fit: y = {slope:.2f}x + {intercept:.2f}"
    )

    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    ax.legend(
        loc="lower right",
        frameon=False,
    )

    ax.grid(
        alpha=0.20
    )

    fig.tight_layout()

    save_figure(
        fig,
        stem,
    )


# =============================================================================
# FIGURES 6 / 7 — Permutation feature importance
# =============================================================================

def calculate_permutation_importance(
    step2_dir,
    step3_dir,
    validation_design,
):
    """
    Post-hoc descriptive permutation importance.

    Importance = increase in validation RMSE after shuffling one predictor.
    Positive values indicate that disrupting the feature worsens prediction.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is required for feature-importance plots. "
            "Run this script inside the same soc_env used for Step 3."
        ) from exc

    validation_path = os.path.join(
        step2_dir,
        "validation_selected_features.csv",
    )

    val_full = read_csv_required(
        validation_path,
        f"{validation_design} selected-feature validation data",
    )

    selected_features = read_selected_features(
        step2_dir
    )

    missing = [
        f
        for f in selected_features
        if f not in val_full.columns
    ]

    if missing:
        raise ValueError(
            f"{validation_design}: selected predictors missing from validation file: "
            f"{missing}"
        )

    selected_model = selected_model_from_ranking(
        step3_dir
    )

    model_path = latest_autogluon_run(
        step3_dir
    )

    print(
        f"\n  Loading {validation_design} AutoGluon model:\n"
        f"  {model_path}"
    )

    predictor = TabularPredictor.load(
        model_path
    )

    model_data = val_full[
        [TARGET] + selected_features
    ].copy()

    baseline_pred = predictor.predict(
        model_data,
        model=selected_model,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            model_data[TARGET],
            baseline_pred,
        )
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    for feature in selected_features:
        deltas = []

        for _ in range(
            N_IMPORTANCE_PERMUTATIONS
        ):
            permuted = model_data.copy()

            permuted[feature] = rng.permutation(
                permuted[feature].values
            )

            perm_pred = predictor.predict(
                permuted,
                model=selected_model,
            )

            perm_rmse = np.sqrt(
                mean_squared_error(
                    permuted[TARGET],
                    perm_pred,
                )
            )

            deltas.append(
                perm_rmse - baseline_rmse
            )

        rows.append({
            "Feature": feature,
            "Permutation importance (ΔRMSE)": np.mean(deltas),
            "SD across permutations": np.std(
                deltas,
                ddof=1,
            ),
            "Baseline validation RMSE": baseline_rmse,
            "Model": selected_model,
            "Validation design": validation_design,
        })

    importance = (
        pd.DataFrame(rows)
        .sort_values(
            "Permutation importance (ΔRMSE)",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(importance) + 1,
        ),
    )

    return importance


def feature_importance_figure(
    importance,
    title,
    stem,
):
    plot_df = importance.sort_values(
        "Permutation importance (ΔRMSE)",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.5)
    )

    ax.barh(
        plot_df["Feature"],
        plot_df["Permutation importance (ΔRMSE)"],
        xerr=plot_df["SD across permutations"],
        capsize=3,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Permutation importance (increase in RMSE)"
    )

    ax.set_ylabel(
        "Predictor"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        stem,
    )


# =============================================================================
# FIGURE 8 — Direct validation-design comparison
# =============================================================================

def validation_comparison_figures(table7):
    plot = table7.set_index(
        "Validation design"
    )

    short_names = [
        "Sample-grouped",
        "KS image-level",
    ]

    # Fit / dimensionless metrics.
    fit_metrics = [
        ("R²", "R²"),
        ("RPD", "RPD"),
        ("RPIQ", "RPIQ"),
    ]

    x = np.arange(
        len(fit_metrics)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    sample_vals = [
        plot.iloc[0][col]
        for col, _ in fit_metrics
    ]

    ks_vals = [
        plot.iloc[1][col]
        for col, _ in fit_metrics
    ]

    ax.bar(
        x - width / 2,
        sample_vals,
        width,
        label=short_names[0],
    )

    ax.bar(
        x + width / 2,
        ks_vals,
        width,
        label=short_names[1],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            label
            for _, label in fit_metrics
        ]
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_title(
        "Validation-design comparison: R², RPD, and RPIQ"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Fig8A_validation_comparison_fit_metrics",
    )

    # Error metrics.
    err_metrics = [
        ("RMSE", "RMSE"),
        ("MAE", "MAE"),
    ]

    x = np.arange(
        len(err_metrics)
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5.0)
    )

    sample_vals = [
        plot.iloc[0][col]
        for col, _ in err_metrics
    ]

    ks_vals = [
        plot.iloc[1][col]
        for col, _ in err_metrics
    ]

    ax.bar(
        x - width / 2,
        sample_vals,
        width,
        label=short_names[0],
    )

    ax.bar(
        x + width / 2,
        ks_vals,
        width,
        label=short_names[1],
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            label
            for _, label in err_metrics
        ]
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_title(
        "Validation-design comparison: prediction error"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Fig8B_validation_comparison_error_metrics",
    )


# =============================================================================
# FIGURE 9 — Bootstrap CI comparison
# =============================================================================

def bootstrap_ci_figure(
    table8a,
    metric,
    x_label,
    stem,
):
    df = table8a[
        table8a["Metric"] == metric
    ].copy()

    order = [
        "Sample-grouped",
        "KS image-level",
    ]

    df["Validation design"] = pd.Categorical(
        df["Validation design"],
        categories=order,
        ordered=True,
    )

    df = df.sort_values(
        "Validation design"
    )

    y = np.arange(
        len(df)
    )

    estimate = pd.to_numeric(
        df["Estimate"],
        errors="coerce",
    ).values

    lower = pd.to_numeric(
        df["95% CI lower"],
        errors="coerce",
    ).values

    upper = pd.to_numeric(
        df["95% CI upper"],
        errors="coerce",
    ).values

    xerr = np.vstack([
        estimate - lower,
        upper - estimate,
    ])

    fig, ax = plt.subplots(
        figsize=(7.0, 3.8)
    )

    ax.errorbar(
        estimate,
        y,
        xerr=xerr,
        fmt="o",
        capsize=5,
        linewidth=1.5,
    )

    if metric == "R2":
        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        df["Validation design"].astype(str)
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_title(
        f"Physical-soil cluster-bootstrap 95% CI for {metric}"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        stem,
    )


# =============================================================================
# CAPTIONS
# =============================================================================

def write_captions():
    captions = """
TABLE CAPTIONS
==============

Table 2A. AutoGluon candidate regression models evaluated under the sample-grouped validation framework.

Table 2B. AutoGluon candidate regression models evaluated under the Kennard-Stone image-level validation framework.

Table 3. Descriptive statistics of soil organic carbon (SOC), soil moisture content, and numerical image-derived color and texture predictors.

Table 4A. Final image-derived predictors retained by the four-stage feature-selection procedure under sample-grouped and Kennard-Stone image-level validation.

Table 4B. Number of predictors retained at each stage of the four-stage feature-selection procedure for the two validation designs.

Table 5. Group-aware internal cross-validation performance and composite ranking of AutoGluon models for the sample-grouped analysis.

Table 6. Group-aware internal cross-validation performance and composite ranking of AutoGluon models for the Kennard-Stone image-level analysis.

Table 7. External validation performance of the final AutoGluon model under sample-grouped and Kennard-Stone image-level validation.

Table 8A. Physical-soil cluster-bootstrap point estimates and 95% confidence intervals for external-validation performance metrics.

Table 8B. Model-versus-calibration-mean baseline comparisons and soil-level inferential tests.


FIGURE CAPTIONS
===============

Fig. 4. Observed versus predicted soil organic carbon (SOC) for the sample-grouped external validation set. The dashed line represents perfect 1:1 agreement between observed and predicted SOC, and the solid line represents the fitted linear regression relationship. All images from each validation soil were excluded from model development.

Fig. 5. Observed versus predicted soil organic carbon (SOC) for the Kennard-Stone image-level validation set. The dashed line represents perfect 1:1 agreement and the solid line represents the fitted linear regression relationship. Because images from all 20 physical soils occurred in both calibration and validation, this figure represents prediction of new images of soils already represented during model development.

Fig. 6. Permutation importance of the final predictors in the sample-grouped model. Importance is expressed as the increase in validation RMSE after randomly permuting each predictor; larger positive values indicate a greater contribution to prediction accuracy. Error bars represent the standard deviation across repeated permutations.

Fig. 7. Permutation importance of the final predictors in the Kennard-Stone image-level model. Importance is expressed as the increase in validation RMSE after randomly permuting each predictor; larger positive values indicate a greater contribution to prediction accuracy. Error bars represent the standard deviation across repeated permutations.

Fig. 8A. Comparison of R², RPD, and RPIQ between sample-grouped external validation and Kennard-Stone image-level validation.

Fig. 8B. Comparison of RMSE and MAE between sample-grouped external validation and Kennard-Stone image-level validation.

Fig. 9A. Physical-soil cluster-bootstrap point estimates and 95% confidence intervals for R² under the sample-grouped and Kennard-Stone image-level validation designs.

Fig. 9B. Physical-soil cluster-bootstrap point estimates and 95% confidence intervals for RMSE under the sample-grouped and Kennard-Stone image-level validation designs.
""".strip()

    path = os.path.join(
        PUB_DIR,
        "publication_captions.txt",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            captions
        )

    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 85)
    print("GENERATING PUBLICATION TABLES AND FIGURES")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # TABLE 2
    # -------------------------------------------------------------------------
    print("\nTABLE 2A / 2B — AutoGluon model inventory")

    make_model_inventory(
        SAMPLE_STEP3_DIR,
        "Sample-grouped",
        "Table2A_AutoGluon_models_sample_grouped.csv",
    )

    make_model_inventory(
        KS_STEP3_DIR,
        "KS image-level",
        "Table2B_AutoGluon_models_KS_image.csv",
    )

    # -------------------------------------------------------------------------
    # TABLE 3
    # -------------------------------------------------------------------------
    print("\nTABLE 3 — Descriptive statistics")
    make_descriptive_statistics()

    # -------------------------------------------------------------------------
    # TABLE 4
    # -------------------------------------------------------------------------
    print("\nTABLE 4 — Feature selection")
    make_feature_selection_tables()

    # -------------------------------------------------------------------------
    # TABLE 5 / 6
    # -------------------------------------------------------------------------
    print("\nTABLE 5 / 6 — AutoGluon model performance")

    publication_model_ranking(
        SAMPLE_STEP3_DIR,
        "Table5_sample_grouped_model_performance.csv",
    )

    publication_model_ranking(
        KS_STEP3_DIR,
        "Table6_KS_image_model_performance.csv",
    )

    # -------------------------------------------------------------------------
    # TABLE 7
    # -------------------------------------------------------------------------
    print("\nTABLE 7 — Validation-design comparison")

    table7 = make_validation_comparison()

    # -------------------------------------------------------------------------
    # TABLE 8
    # -------------------------------------------------------------------------
    print("\nTABLE 8 — Statistical validation")

    table8a, table8b = make_statistical_validation_tables()

    # -------------------------------------------------------------------------
    # FIGURE 4
    # -------------------------------------------------------------------------
    print("\nFIGURE 4 — Sample-grouped observed vs predicted")

    observed_vs_predicted_figure(
        os.path.join(
            SAMPLE_STEP3_DIR,
            "external_validation_predictions.csv",
        ),
        "Sample-grouped external validation",
        "Fig4_sample_grouped_observed_vs_predicted",
    )

    # -------------------------------------------------------------------------
    # FIGURE 5
    # -------------------------------------------------------------------------
    print("\nFIGURE 5 — KS image-level observed vs predicted")

    observed_vs_predicted_figure(
        os.path.join(
            KS_STEP3_DIR,
            "external_validation_predictions.csv",
        ),
        "Kennard-Stone image-level validation",
        "Fig5_KS_image_observed_vs_predicted",
    )

    # -------------------------------------------------------------------------
    # FIGURE 6 / 7
    # -------------------------------------------------------------------------
    print("\nFIGURE 6 — Sample-grouped feature importance")

    sample_importance = calculate_permutation_importance(
        SAMPLE_STEP2_DIR,
        SAMPLE_STEP3_DIR,
        "Sample-grouped",
    )

    sample_importance_path = os.path.join(
        TABLE_DIR,
        "Supplement_feature_importance_sample_grouped.csv",
    )

    sample_importance.to_csv(
        sample_importance_path,
        index=False,
    )

    print(
        f"  Saved: {sample_importance_path}"
    )

    feature_importance_figure(
        sample_importance,
        "Sample-grouped model: permutation feature importance",
        "Fig6_sample_grouped_feature_importance",
    )

    print("\nFIGURE 7 — KS image-level feature importance")

    ks_importance = calculate_permutation_importance(
        KS_STEP2_DIR,
        KS_STEP3_DIR,
        "KS image-level",
    )

    ks_importance_path = os.path.join(
        TABLE_DIR,
        "Supplement_feature_importance_KS_image.csv",
    )

    ks_importance.to_csv(
        ks_importance_path,
        index=False,
    )

    print(
        f"  Saved: {ks_importance_path}"
    )

    feature_importance_figure(
        ks_importance,
        "Kennard-Stone image-level model: permutation feature importance",
        "Fig7_KS_image_feature_importance",
    )

    # -------------------------------------------------------------------------
    # FIGURE 8
    # -------------------------------------------------------------------------
    print("\nFIGURE 8 — Validation-design comparison")
    validation_comparison_figures(
        table7
    )

    # -------------------------------------------------------------------------
    # FIGURE 9
    # -------------------------------------------------------------------------
    print("\nFIGURE 9 — Bootstrap confidence intervals")

    bootstrap_ci_figure(
        table8a,
        metric="R2",
        x_label="R²",
        stem="Fig9A_bootstrap_R2_confidence_intervals",
    )

    bootstrap_ci_figure(
        table8a,
        metric="RMSE",
        x_label="RMSE",
        stem="Fig9B_bootstrap_RMSE_confidence_intervals",
    )

    # -------------------------------------------------------------------------
    # Captions
    # -------------------------------------------------------------------------
    print("\nCAPTIONS")
    write_captions()

    print("\n" + "=" * 85)
    print("PUBLICATION OUTPUT GENERATION COMPLETE")
    print("=" * 85)
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")
    print(
        "\nNext analysis after reviewing these outputs:"
        "\n  1. moisture-stratified analysis"
        "\n  2. field-vs-laboratory acquisition analysis"
    )
    print("=" * 85)


if __name__ == "__main__":
    main()
