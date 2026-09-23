"""
STEP 3 — LEAVE-ONE-SAMPLE-OUT (LOSO) CROSS-VALIDATION
=======================================================
For each of 20 soil samples:
  - Hold out ALL images from that sample (validation)
  - Train on ALL images from the other 19 samples (calibration)
  - Predict SOC for the held-out sample
  - Record image-level and sample-level predictions

Then refit final model on ALL 731 images for downstream analyses.

Satisfies reviewer requirement:
  "all images from a given soil sample are assigned either to
   calibration or validation, but never both"

Run: python step3_loso.py
"""

import os, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

# Input — use the full feature-selected calibration file from Step 2
# This should have ALL images (not aggregated) with Sample_No column
CALIB_CSV = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step2_output", "calibration_selected_features.csv")
VALID_CSV  = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step2_output", "validation_selected_features.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step3_loso")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET    = "soc"
GROUP     = "Sample_No"
DROP_COLS = ["image_no", "moisture", "soil_type", "Sample_No"]

WEIGHTS = {"RPD":0.35,"RPIQ":0.25,"RMSE":0.20,"R2":0.15,"MAE":0.05}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rpd  = y_true.std()/rmse if rmse > 0 else 0
    q75,q25 = np.percentile(y_true,[75,25])
    rpiq = (q75-q25)/rmse if rmse > 0 else 0
    bias = float(np.mean(y_pred-y_true))
    mt,mp = np.mean(y_true),np.mean(y_pred)
    vt,vp = np.var(y_true),np.var(y_pred)
    cov   = np.mean((y_true-mt)*(y_pred-mp))
    lccc  = 2*cov/(vt+vp+(mt-mp)**2) if (vt+vp)>0 else 0
    comp  = (WEIGHTS["RPD"]*min(rpd/4,1) + WEIGHTS["RPIQ"]*min(rpiq/4,1) +
             WEIGHTS["RMSE"]*max(0,1-rmse/2) + WEIGHTS["R2"]*max(0,r2) +
             WEIGHTS["MAE"]*max(0,1-mae/2))
    return {"R2":r2,"RMSE":rmse,"MAE":mae,"RPD":rpd,"RPIQ":rpiq,
            "LCCC":lccc,"Bias":bias,"Composite":comp}

def print_metrics(m, label):
    flag = "EXCELLENT" if m["RPD"]>=2 else "GOOD" if m["RPD"]>=1.4 else "MODERATE"
    print(f"\n  {label}")
    print(f"    R²        = {m['R2']:.4f}")
    print(f"    RMSE      = {m['RMSE']:.4f} %")
    print(f"    RPD       = {m['RPD']:.4f}  ({flag})")
    print(f"    RPIQ      = {m['RPIQ']:.4f}")
    print(f"    LCCC      = {m['LCCC']:.4f}")
    print(f"    MAE       = {m['MAE']:.4f} %")
    print(f"    Bias      = {m['Bias']:.4f}")
    print(f"    Composite = {m['Composite']:.4f}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 3 — LEAVE-ONE-SAMPLE-OUT (LOSO) CROSS-VALIDATION")
print("="*65)

# Combine calibration and validation to get all images
df_cal = pd.read_csv(CALIB_CSV)
df_val = pd.read_csv(VALID_CSV)
df_all = pd.concat([df_cal, df_val], ignore_index=True)

print(f"\n  Calibration file : {len(df_cal)} images")
print(f"  Validation file  : {len(df_val)} images")
print(f"  Combined total   : {len(df_all)} images")

if len(df_all) < 100:
    raise ValueError(
        f"Only {len(df_all)} rows — Step 2 may still be aggregating. "
        "Re-run step2_fixed_final.py first.")

if GROUP not in df_all.columns:
    raise ValueError(f"Column '{GROUP}' not found. Check Step 2 output.")

samples = sorted(df_all[GROUP].dropna().unique().astype(int).tolist())
print(f"  Unique samples   : {len(samples)}  →  {samples}")

FEATURES = [c for c in df_all.columns
            if c not in DROP_COLS + [TARGET]
            and pd.api.types.is_numeric_dtype(df_all[c])]
print(f"  Predictors ({len(FEATURES)}): {FEATURES}")

# ── LOSO LOOP ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"LOSO — {len(samples)} folds (one per physical soil sample)")
print("="*65)

from autogluon.tabular import TabularPredictor

all_preds = []
fold_metrics = []

for fold_idx, held_out_sample in enumerate(samples, 1):

    val_mask = df_all[GROUP] == held_out_sample
    df_fold_val = df_all[val_mask].reset_index(drop=True)
    df_fold_cal = df_all[~val_mask].reset_index(drop=True)

    # Confirm no overlap
    overlap = (set(df_fold_cal[GROUP].unique()) &
               set(df_fold_val[GROUP].unique()))
    assert len(overlap) == 0, f"Fold {fold_idx} overlap: {overlap}"

    y_true = df_fold_val[TARGET].values
    soc_val = y_true.mean()

    # Prepare modelling dataframes
    train_ml = df_fold_cal[FEATURES + [TARGET]].copy()
    test_ml  = df_fold_val[FEATURES + [TARGET]].copy()

    print(f"\n  Fold {fold_idx:2d}/20 — held out: Sample {held_out_sample:2d} "
          f"(SOC={soc_val:.2f}%,  n={len(df_fold_val)} images)")
    print(f"           train: {len(train_ml)} images from "
          f"{df_fold_cal[GROUP].nunique()} samples")

    # SHORT path to avoid Windows 260-char limit
    fold_path = os.path.join(OUT_DIR, f"fold{fold_idx:02d}")
    if os.path.exists(fold_path):
        shutil.rmtree(fold_path, ignore_errors=True)
    os.makedirs(fold_path, exist_ok=True)

    try:
        predictor = TabularPredictor(
            label=TARGET,
            eval_metric="root_mean_squared_error",
            path=fold_path,
            verbosity=0
        ).fit(
            train_data=train_ml,
            time_limit=120,          # 2 min per fold × 20 folds = 40 min max
            presets="medium_quality",
        )

        y_pred = predictor.predict(test_ml).values
        best_model = predictor.model_best

        # Image-level predictions for this fold
        fold_df = df_fold_val[
            [c for c in [GROUP, "image_no", "moisture", "soil_type", TARGET]
             if c in df_fold_val.columns]
        ].copy()
        fold_df["Predicted_SOC"] = y_pred
        fold_df["Error"] = y_pred - y_true
        all_preds.append(fold_df)

        # Sample-level summary for this fold
        m = metrics(y_true, y_pred)
        fold_metrics.append({
            "Sample_No": held_out_sample,
            "SOC_actual": soc_val,
            "SOC_predicted": float(y_pred.mean()),
            "Error": float(y_pred.mean() - soc_val),
            "Abs_Error": abs(float(y_pred.mean() - soc_val)),
            "N_images": len(y_true),
            "Best_model": best_model,
            **{f"img_{k}": v for k,v in m.items()}
        })

        print(f"           best model: {best_model} | "
              f"pred={y_pred.mean():.3f}% | "
              f"err={y_pred.mean()-soc_val:+.3f}%")

    except Exception as e:
        print(f"           ERROR: {e}")
        fold_metrics.append({
            "Sample_No": held_out_sample,
            "SOC_actual": soc_val,
            "SOC_predicted": np.nan,
            "Error": np.nan,
            "Abs_Error": np.nan,
            "N_images": len(y_true),
            "Best_model": "ERROR",
        })

# ── COMPILE ALL PREDICTIONS ────────────────────────────────────────────────────
print("\n" + "="*65)
print("LOSO RESULTS — ALL 20 SAMPLES")
print("="*65)

df_all_preds  = pd.concat(all_preds, ignore_index=True)
df_fold_summ  = pd.DataFrame(fold_metrics)

# Image-level metrics across all 20 samples
y_true_all = df_all_preds[TARGET].values
y_pred_all = df_all_preds["Predicted_SOC"].values

m_img = metrics(y_true_all, y_pred_all)
print_metrics(m_img, f"IMAGE-LEVEL ({len(y_true_all)} images, 20 samples)")

# Sample-level metrics
df_samp = df_all_preds.groupby(GROUP).agg(
    y_true=(TARGET,"mean"),
    y_pred=("Predicted_SOC","mean"),
    n=  (TARGET,"count")
).reset_index()

m_samp = metrics(df_samp["y_true"].values, df_samp["y_pred"].values)
print_metrics(m_samp, f"SAMPLE-LEVEL ({len(df_samp)} soil samples)")

rho, p_rho = spearmanr(df_samp["y_true"], df_samp["y_pred"])
print(f"\n  Spearman rank correlation: ρ={rho:.4f}  p={p_rho:.4f}")

# Per-sample table
print(f"\n  Per-sample predictions:")
print(f"  {'Sample':8s} {'n_imgs':7s} {'Actual':10s} {'Predicted':12s} "
      f"{'Error':8s} {'Abs_Error':10s}")
print("  " + "─"*58)
for _, r in df_samp.sort_values("y_true").iterrows():
    flag = " ⚠ EXTRAPOLATION" if r["y_pred"] < 0.99 or r["y_true"] > 5.0 else ""
    print(f"  {int(r[GROUP]):8d} {int(r['n']):7d} "
          f"{r['y_true']:10.4f} {r['y_pred']:12.4f} "
          f"{r['y_pred']-r['y_true']:+8.4f} "
          f"{abs(r['y_pred']-r['y_true']):10.4f}{flag}")

# ── SAVE ─────────────────────────────────────────────────────────────────────
df_all_preds.to_csv(
    os.path.join(OUT_DIR, "loso_image_predictions.csv"), index=False)
df_samp.to_csv(
    os.path.join(OUT_DIR, "loso_sample_predictions.csv"), index=False)
df_fold_summ.to_csv(
    os.path.join(OUT_DIR, "loso_fold_summary.csv"), index=False)
pd.DataFrame([{**{"level":"image","N":len(y_true_all)}, **m_img},
              {**{"level":"sample","N":len(df_samp)},   **m_samp}]).to_csv(
    os.path.join(OUT_DIR, "LOSO_FINAL_METRICS.csv"), index=False)

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("SOC Prediction — LOSO Cross-Validation (20 soil samples)\n"
             "Each sample predicted by model trained on the other 19",
             fontweight="bold", fontsize=12)

# Image-level
ax = axes[0]
scatter = ax.scatter(y_true_all, y_pred_all,
                      c=df_all_preds[GROUP].astype(float),
                      cmap="tab20", alpha=0.4, s=15)
lo = min(y_true_all.min(), y_pred_all.min()) - 0.1
hi = max(y_true_all.max(), y_pred_all.max()) + 0.1
ax.plot([lo,hi],[lo,hi],"k--",lw=1.5,label="1:1 line")
ax.set_xlabel("Measured SOC (%)", fontsize=11)
ax.set_ylabel("Predicted SOC (%)", fontsize=11)
ax.set_title(f"Image-level (n={len(y_true_all)})\n"
             f"R²={m_img['R2']:.3f}  RMSE={m_img['RMSE']:.3f}%  "
             f"RPD={m_img['RPD']:.3f}", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label="Sample No.")

# Sample-level
ax2 = axes[1]
ax2.scatter(df_samp["y_true"], df_samp["y_pred"],
            c="steelblue", s=120, zorder=5)
for _, r in df_samp.iterrows():
    ax2.annotate(f"S{int(r[GROUP])}",
                 (r["y_true"], r["y_pred"]),
                 textcoords="offset points", xytext=(5,4), fontsize=8)
lo2 = min(df_samp["y_true"].min(), df_samp["y_pred"].min()) - 0.1
hi2 = max(df_samp["y_true"].max(), df_samp["y_pred"].max()) + 0.1
ax2.plot([lo2,hi2],[lo2,hi2],"k--",lw=1.5,label="1:1 line")
ax2.set_xlabel("Measured SOC (%)", fontsize=11)
ax2.set_ylabel("Predicted SOC (%)", fontsize=11)
ax2.set_title(f"Sample-level (n={len(df_samp)})\n"
              f"R²={m_samp['R2']:.3f}  RMSE={m_samp['RMSE']:.3f}%  "
              f"RPD={m_samp['RPD']:.3f}  ρ={rho:.3f}", fontsize=10)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loso_predicted_vs_actual.png"),
            dpi=160, bbox_inches="tight")
plt.close()

# ── FINAL MODEL ON ALL 20 SAMPLES ────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL MODEL — trained on ALL 20 samples (for downstream analyses)")
print("="*65)

final_path = os.path.join(OUT_DIR, "final_all_samples")
if os.path.exists(final_path):
    shutil.rmtree(final_path, ignore_errors=True)
os.makedirs(final_path, exist_ok=True)

final_train = df_all[FEATURES + [TARGET]].copy()
print(f"  Training on {len(final_train)} images from all 20 samples...")

final_predictor = TabularPredictor(
    label=TARGET,
    eval_metric="root_mean_squared_error",
    path=final_path,
    verbosity=2
).fit(
    train_data=final_train,
    time_limit=1800,
    presets="medium_quality",
)

print(f"\n  Final model saved: {final_path}")
print(f"  Best model: {final_predictor.model_best}")
print(f"  Use this model for moisture stratification & field/lab analyses")

# Save final predictor info
pd.DataFrame([{
    "model": final_predictor.model_best,
    "n_training_images": len(final_train),
    "n_samples": 20,
    "features": str(FEATURES),
    "use_for": "moisture_analysis, field_vs_lab, feature_importance"
}]).to_csv(os.path.join(OUT_DIR, "final_model_info.csv"), index=False)

print("\n" + "="*65)
print("LOSO COMPLETE")
print("="*65)
print(f"\n  LOSO Image-level  R²   = {m_img['R2']:.4f}")
print(f"  LOSO Image-level  RMSE = {m_img['RMSE']:.4f} %")
print(f"  LOSO Image-level  RPD  = {m_img['RPD']:.4f}")
print(f"  LOSO Sample-level R²   = {m_samp['R2']:.4f}")
print(f"  LOSO Sample-level RMSE = {m_samp['RMSE']:.4f} %")
print(f"  LOSO Sample-level RPD  = {m_samp['RPD']:.4f}")
print(f"  Spearman ρ             = {rho:.4f}  (p={p_rho:.4f})")
print(f"\n  Outputs: {OUT_DIR}")
print(f"  Key files:")
print(f"    loso_image_predictions.csv  — all image-level predictions")
print(f"    loso_sample_predictions.csv — sample-level means")
print(f"    LOSO_FINAL_METRICS.csv      — summary metrics")
print(f"    loso_predicted_vs_actual.png — publication figure")
print(f"    final_all_samples/           — model for downstream analyses")