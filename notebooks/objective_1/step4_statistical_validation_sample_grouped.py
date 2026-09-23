import os, itertools, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = (r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
                r"\VSCode_Image_Processing_Reviewed")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT,"notebooks","objective_1","output_data","step3_output","external_validation_predictions.csv")
CALIBRATION_CSV = os.path.join(PROJECT_ROOT,"notebooks","objective_1","output_data","step2_output","calibration_selected_features.csv")
OUT_DIR = os.path.join(PROJECT_ROOT,"notebooks","objective_1","output_data","step4_statistical_validation_sample_grouped")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET, PRED, GROUP = "soc", "Predicted_SOC", "Sample_No"
N_BOOT, CI_LEVEL, SEED = 5000, 0.95, 42

def metrics(y_true, y_pred):
    y_true=np.asarray(y_true,float); y_pred=np.asarray(y_pred,float)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    mae=mean_absolute_error(y_true,y_pred)
    r2=r2_score(y_true,y_pred) if len(y_true)>1 else np.nan
    sd=np.std(y_true,ddof=1) if len(y_true)>1 else np.nan
    rpd=sd/rmse if rmse>0 and np.isfinite(sd) else np.nan
    q75,q25=np.percentile(y_true,[75,25]); rpiq=(q75-q25)/rmse if rmse>0 else np.nan
    bias=np.mean(y_pred-y_true)
    return {"R2":r2,"RMSE":rmse,"MAE":mae,"RPD":rpd,"RPIQ":rpiq,"Bias":bias}

def ci(vals):
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]
    if len(vals)==0: return (np.nan,np.nan)
    a=(1-CI_LEVEL)/2
    return np.percentile(vals,[100*a,100*(1-a)])

def cluster_bootstrap(df,target_col,pred_col,group_col,seed):
    rng=np.random.default_rng(seed)
    groups=np.array(sorted(df[group_col].dropna().unique()))
    rec=[]
    for b in range(N_BOOT):
        sampled=rng.choice(groups,size=len(groups),replace=True)
        boot=pd.concat([df.loc[df[group_col]==g] for g in sampled],ignore_index=True)
        m=metrics(boot[target_col],boot[pred_col]); m["Bootstrap"]=b+1; rec.append(m)
    return pd.DataFrame(rec)

def summarize(point,boot):
    rows=[]
    for k in ["R2","RMSE","MAE","RPD","RPIQ","Bias"]:
        lo,hi=ci(boot[k])
        rows.append({"Metric":k,"Estimate":point[k],"CI_Lower":lo,"CI_Upper":hi})
    return pd.DataFrame(rows)

pred_df=pd.read_csv(PREDICTIONS_CSV)
cal_df=pd.read_csv(CALIBRATION_CSV)
req=[TARGET,PRED,GROUP]
missing=[c for c in req if c not in pred_df.columns]
if missing: raise ValueError(f"Missing columns: {missing}")

print("\n=== STEP 4: SAMPLE-GROUPED STATISTICAL VALIDATION ===")
print(f"Validation images: {len(pred_df)}")
print(f"Held-out soils: {sorted(pred_df[GROUP].unique())}")

# 1) Image-level cluster bootstrap
image_point=metrics(pred_df[TARGET],pred_df[PRED])
image_boot=cluster_bootstrap(pred_df,TARGET,PRED,GROUP,SEED)
image_summary=summarize(image_point,image_boot)
print("\nIMAGE-LEVEL CLUSTER-BOOTSTRAP 95% CIs")
print(image_summary.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
image_summary.to_csv(os.path.join(OUT_DIR,"image_level_cluster_bootstrap_summary.csv"),index=False)
image_boot.to_csv(os.path.join(OUT_DIR,"image_level_cluster_bootstrap_all.csv"),index=False)

# 2) Soil-condition level
condition_df=(pred_df.groupby([GROUP,TARGET],as_index=False)
              .agg(Predicted_SOC=(PRED,"mean"),N_Images=(PRED,"size")))
condition_point=metrics(condition_df[TARGET],condition_df[PRED])
condition_boot=cluster_bootstrap(condition_df,TARGET,PRED,GROUP,SEED+1)
condition_summary=summarize(condition_point,condition_boot)
print("\nSOIL-CONDITION-LEVEL CLUSTER-BOOTSTRAP 95% CIs")
print(condition_summary.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
condition_df.to_csv(os.path.join(OUT_DIR,"soil_condition_predictions.csv"),index=False)
condition_summary.to_csv(os.path.join(OUT_DIR,"soil_condition_cluster_bootstrap_summary.csv"),index=False)

# 3) Calibration-mean baseline comparison
baseline=float(cal_df[TARGET].mean())
pred_df["Baseline_Prediction"]=baseline
base_point=metrics(pred_df[TARGET],pred_df["Baseline_Prediction"])
rng=np.random.default_rng(SEED+2); groups=np.array(sorted(pred_df[GROUP].unique()))
diffs=[]
for b in range(N_BOOT):
    sampled=rng.choice(groups,size=len(groups),replace=True)
    boot=pd.concat([pred_df.loc[pred_df[GROUP]==g] for g in sampled],ignore_index=True)
    mm=metrics(boot[TARGET],boot[PRED]); bm=metrics(boot[TARGET],boot["Baseline_Prediction"])
    diffs.append({"Delta_RMSE":mm["RMSE"]-bm["RMSE"],"Delta_MAE":mm["MAE"]-bm["MAE"]})
diffs=pd.DataFrame(diffs)
rmse_lo,rmse_hi=ci(diffs["Delta_RMSE"]); mae_lo,mae_hi=ci(diffs["Delta_MAE"])
comparison=pd.DataFrame([
    {"Metric":"RMSE","ModelMinusBaseline":image_point["RMSE"]-base_point["RMSE"],"CI_Lower":rmse_lo,"CI_Upper":rmse_hi},
    {"Metric":"MAE","ModelMinusBaseline":image_point["MAE"]-base_point["MAE"],"CI_Lower":mae_lo,"CI_Upper":mae_hi},
])
print("\nMODEL VS CALIBRATION-MEAN BASELINE")
print(f"Calibration mean SOC: {baseline:.4f}")
print(comparison.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
comparison.to_csv(os.path.join(OUT_DIR,"baseline_comparison_summary.csv"),index=False)

# 4) Exact sign-flip test across held-out soils (supplementary)
soil_loss=[]
for g,d in pred_df.groupby(GROUP):
    model_mse=mean_squared_error(d[TARGET],d[PRED])
    base_mse=mean_squared_error(d[TARGET],d["Baseline_Prediction"])
    soil_loss.append({GROUP:g,"Model_MSE":model_mse,"Baseline_MSE":base_mse,
                      "Difference":model_mse-base_mse})
soil_loss=pd.DataFrame(soil_loss)
d=soil_loss["Difference"].values; obs=d.mean()
perm=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1], repeat=len(d))])
p=np.mean(np.abs(perm)>=abs(obs)-1e-12)
print("\nEXACT SOIL-LEVEL SIGN-FLIP TEST (supplementary)")
print(soil_loss.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print(f"Observed mean MSE difference: {obs:.4f}")
print(f"Exact two-sided p-value: {p:.4f}")
print("Note: only 4 held-out soils, so statistical power is very low.")
soil_loss.to_csv(os.path.join(OUT_DIR,"per_soil_model_vs_baseline_loss.csv"),index=False)
pd.DataFrame([{"Observed_Mean_MSE_Difference":obs,"Exact_TwoSided_P":p}]).to_csv(
    os.path.join(OUT_DIR,"exact_signflip_model_vs_baseline.csv"),index=False)

print(f"\nOutputs saved to: {OUT_DIR}")
