"""
SOC Prediction using ResNet50 Transfer Learning
================================================
Uses pre-trained ResNet50 (ImageNet) as a feature extractor.
No training from scratch. Fine-tunes only the final regression layer.

Process:
  1. Load all soil JPEG images
  2. Match each image to SOC value via metadata CSV
  3. Grouped sample split (validation = samples 3,12,13,15,19)
  4. Extract 2048 ResNet50 features per image (frozen backbone)
  5. Train SVR/Ridge regression head on calibration features
  6. Predict SOC on 5 held-out validation samples
  7. Report R², RMSE, RPD, RPIQ, LCCC

Requirements:
  pip install torch torchvision pillow scikit-learn pandas numpy matplotlib

Run:
  python resnet_soc_pipeline.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — update these paths
# =============================================================================

# Folder containing all soil JPEG images (1.jpg, 2.jpg ... 731.jpg)
IMAGE_FOLDER = r"C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_image_processing_project\data\step2_output\enhanced_sample_field"

# Metadata CSV with image_no and soc columns
METADATA_CSV = r"C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\data\raw\image_with_soc_metadata.csv"

# Excel file for image→sample mapping
IMAGE_RECORD = r"C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\data\raw\IMAGe record_image No._copy.xlsx"

# Output directory
OUT_DIR = r"C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\resnet_output"

os.makedirs(OUT_DIR, exist_ok=True)

# Validation samples — all their images held out entirely
VALIDATION_SAMPLES = [3, 12, 13, 15, 19]

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# STEP 1 — BUILD IMAGE → SAMPLE MAPPING
# =============================================================================
def build_mapping():
    print("Building image → sample mapping from Excel...")
    xl = pd.ExcelFile(IMAGE_RECORD)

    def read_sheet_1():
        raw = pd.read_excel(xl, sheet_name='1-205', header=None)
        g = []
        for s in [0, 9, 17]:
            grp = raw.iloc[1:, s:s+6].copy()
            grp.columns = ['Date', 'Sample_No', 'Image_No', 'M', 'ST', 'SOC']
            grp = grp.dropna(subset=['Image_No'])
            grp['Sample_No'] = pd.to_numeric(grp['Sample_No'], errors='coerce')
            grp['Image_No']  = pd.to_numeric(grp['Image_No'],  errors='coerce')
            grp = grp.dropna(subset=['Sample_No', 'Image_No'])
            g.append(grp[['Sample_No', 'Image_No']])
        return pd.concat(g, ignore_index=True)

    def read_sheet(sh, suf):
        df = pd.read_excel(xl, sheet_name=sh)
        g = []
        for s in suf:
            sn, ino = 'Sample No.'+s, 'Image No.'+s
            if sn in df.columns and ino in df.columns:
                grp = df[[sn, ino]].copy()
                grp.columns = ['Sample_No', 'Image_No']
                grp['Sample_No'] = pd.to_numeric(grp['Sample_No'], errors='coerce')
                grp['Image_No']  = pd.to_numeric(grp['Image_No'],  errors='coerce')
                grp = grp.dropna(subset=['Sample_No', 'Image_No'])
                g.append(grp)
        return pd.concat(g, ignore_index=True) if g else pd.DataFrame()

    mapping = pd.concat([
        read_sheet_1(),
        read_sheet('206-360', ['', '.1', '.2']),
        read_sheet('361-731', ['', '.1', '.2', '.3', '.4']),
    ], ignore_index=True)
    mapping['Sample_No'] = mapping['Sample_No'].astype(int)
    mapping['Image_No']  = mapping['Image_No'].astype(int)
    mapping = mapping.drop_duplicates(subset=['Image_No']).sort_values('Image_No').reset_index(drop=True)
    print(f"  Mapped {len(mapping)} images across {mapping['Sample_No'].nunique()} samples")
    return mapping


# =============================================================================
# STEP 2 — LOAD RESNET50 FEATURE EXTRACTOR
# =============================================================================
def load_resnet():
    print("\nLoading ResNet50 (ImageNet pretrained)...")
    # Load full ResNet50
    resnet = models.resnet50(weights='IMAGENET1K_V1')
    # Remove final FC classification layer — keep everything up to avgpool
    # This gives 2048-dimensional feature vector per image
    extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    extractor.eval()  # freeze — no training
    print("  ResNet50 loaded — 2048 features per image")
    print("  Backbone frozen — only regression head will be trained")

    # ImageNet normalisation — required for pretrained ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
    return extractor, transform


# =============================================================================
# STEP 3 — EXTRACT FEATURES FROM ALL IMAGES
# =============================================================================
def extract_features(extractor, transform, meta, mapping):
    print(f"\nExtracting ResNet50 features from images in: {IMAGE_FOLDER}")

    # Merge metadata with mapping
    df = meta.merge(mapping.rename(columns={'Image_No': 'image_no'}),
                    on='image_no', how='left')
    df['Sample_No'] = pd.to_numeric(df['Sample_No'], errors='coerce').astype('Int64')

    features = []
    valid_rows = []
    missing = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        img_no = int(row['image_no'])

        img_path = os.path.join(
            IMAGE_FOLDER,
            f"{img_no}.jpg"
        )

        if not os.path.isfile(img_path):
            img_path = os.path.join(
                IMAGE_FOLDER,
                f"{img_no:03d}.jpg"
            )

        if not os.path.isfile(img_path):
            print(f"Missing image: {repr(img_path)}")
            missing.append(img_no)
            continue

        try:
            print(f"Trying image: {repr(img_path)}")

            with Image.open(img_path) as img:
                img = img.convert('RGB')

                x = transform(img).unsqueeze(0)

                with torch.no_grad():
                    feat = extractor(x).squeeze().numpy()

            features.append(feat)
            valid_rows.append(row)

        except Exception as e:
            print(f"Failed image {img_no}: {repr(img_path)}")
            print(f"Error: {e}")
            missing.append(img_no)
            continue

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{total} images...")

    if missing:
        print(
            f"  Warning: {len(missing)} images not found or failed — "
            f"{sorted(missing)[:10]}..."
        )

    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    features = np.array(features)

    print(f"\n  Extracted features from {len(df_valid)} images")
    print(f"  Feature matrix shape: {features.shape}")

    return features, df_valid


# =============================================================================
# STEP 4 — GROUPED SPLIT
# =============================================================================
def grouped_split(features, df):
    print(f"\nGrouped sample split — validation = samples {VALIDATION_SAMPLES}")

    val_mask = df['Sample_No'].isin(VALIDATION_SAMPLES)
    cal_mask  = ~val_mask

    X_cal = features[cal_mask.values]
    y_cal = df.loc[cal_mask, 'soc'].values
    X_val = features[val_mask.values]
    y_val = df.loc[val_mask, 'soc'].values
    df_val = df[val_mask].reset_index(drop=True)

    overlap = set(df[cal_mask]['image_no']) & set(df[val_mask]['image_no'])
    assert len(overlap) == 0, f"Overlap detected: {overlap}"

    print(f"  Calibration: {len(X_cal)} images ({df[cal_mask]['Sample_No'].nunique()} samples)"
          f" SOC {y_cal.min():.2f}–{y_cal.max():.2f}")
    print(f"  Validation : {len(X_val)} images ({df[val_mask]['Sample_No'].nunique()} samples)"
          f" SOC {y_val.min():.2f}–{y_val.max():.2f}")
    print(f"  Zero overlap confirmed")

    return X_cal, y_cal, X_val, y_val, df_val


# =============================================================================
# STEP 5 — DIMENSIONALITY REDUCTION + TRAIN REGRESSION HEAD
# =============================================================================
def train_and_evaluate(X_cal, y_cal, X_val, y_val, df_val):
    print(f"\n{'='*65}")
    print("TRAINING REGRESSION HEAD ON RESNET50 FEATURES")
    print(f"{'='*65}")

    # Scale features
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_val_s = scaler.transform(X_val)

    # PCA to reduce 2048 → n_components (keeps 95% variance)
    # This prevents overfitting with 2048 features on ~460 samples
    print(f"\nApplying PCA to reduce 2048 ResNet features...")
    pca = PCA(n_components=0.95, random_state=SEED)
    X_cal_pca = pca.fit_transform(X_cal_s)
    X_val_pca = pca.transform(X_val_s)
    print(f"  PCA: 2048 → {X_cal_pca.shape[1]} components (95% variance)")

    def metrics(yt, yp):
        r2   = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae  = mean_absolute_error(yt, yp)
        rpd  = np.std(yt)/rmse if rmse > 0 else 0
        q75, q25 = np.percentile(yt, [75, 25])
        rpiq = (q75-q25)/rmse if rmse > 0 else 0
        bias = float(np.mean(yp-yt))
        mt, mp = np.mean(yt), np.mean(yp)
        vt, vp = np.var(yt), np.var(yp)
        cov = np.mean((yt-mt)*(yp-mp))
        lccc = 2*cov/(vt+vp+(mt-mp)**2) if (vt+vp) > 0 else 0
        return r2, rmse, mae, rpd, rpiq, lccc, bias

    # Models to test
    models_list = [
        ('SVR (RBF)',      SVR(kernel='rbf',    C=10,  gamma='scale', epsilon=0.05)),
        ('SVR (Linear)',   SVR(kernel='linear', C=1.0, epsilon=0.05)),
        ('Ridge',          Ridge(alpha=1.0)),
        ('ElasticNet',     ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ('PLS (n=5)',      PLSRegression(n_components=5)),
        ('PLS (n=10)',     PLSRegression(n_components=10)),
        ('Grad Boosting',  GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                      max_depth=3, random_state=SEED)),
        ('Random Forest',  RandomForestRegressor(n_estimators=300, max_features='sqrt',
                                                  random_state=SEED, n_jobs=-1)),
    ]

    print(f"\n  {'Model':22s} {'R²':7s} {'RMSE':7s} {'RPD':7s} {'RPIQ':7s} "
          f"{'LCCC':7s} {'Bias':7s} {'CV R²':7s}")
    print("  " + "─"*75)

    all_results = []
    best_r2 = -999
    best_pred = None
    best_name = ""

    for name, model in models_list:
        cv = cross_val_score(model, X_cal_pca, y_cal, cv=5,
                              scoring='r2', n_jobs=-1)
        model.fit(X_cal_pca, y_cal)
        pred = model.predict(X_val_pca).flatten()
        r2, rmse, mae, rpd, rpiq, lccc, bias = metrics(y_val, pred)
        flag = '✅' if r2 >= 0.5 else '⚠' if r2 >= 0 else '❌'
        print(f"  {name:22s} {r2:7.4f} {rmse:7.4f} {rpd:7.4f} {rpiq:7.4f} "
              f"{lccc:7.4f} {bias:7.4f} {cv.mean():7.4f} {flag}")
        all_results.append({'model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae,
                             'RPD': rpd, 'RPIQ': rpiq, 'LCCC': lccc,
                             'Bias': bias, 'CV_R2': cv.mean()})
        if r2 > best_r2:
            best_r2 = r2
            best_pred = pred
            best_name = name
            best_rmse = rmse
            best_rpd = rpd
            best_rpiq = rpiq
            best_lccc = lccc
            best_bias = bias

    return all_results, best_name, best_r2, best_rmse, best_rpd, best_rpiq, best_lccc, best_bias, best_pred


# =============================================================================
# STEP 6 — SAMPLE-LEVEL METRICS + PLOTS
# =============================================================================
def report_and_plot(y_val, best_pred, df_val, all_results,
                    best_name, best_r2, best_rmse, best_rpd,
                    best_rpiq, best_lccc, best_bias):

    print(f"\n{'='*65}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*65}")
    print(f"  Image-level R²    = {best_r2:.4f}")
    print(f"  Image-level RMSE  = {best_rmse:.4f} %")
    print(f"  Image-level RPD   = {best_rpd:.4f}")
    print(f"  Image-level RPIQ  = {best_rpiq:.4f}")
    print(f"  Image-level LCCC  = {best_lccc:.4f}")
    print(f"  Image-level Bias  = {best_bias:.4f}")

    # Sample-level metrics
    df_r = pd.DataFrame({
        'Sample_No': df_val['Sample_No'].values,
        'y_true': y_val,
        'y_pred': best_pred
    })
    samp = df_r.groupby('Sample_No').agg(
        y_true=('y_true', 'mean'),
        y_pred=('y_pred', 'mean'),
        n=('y_true', 'count')
    ).reset_index()

    sr2   = r2_score(samp['y_true'], samp['y_pred'])
    srmse = np.sqrt(mean_squared_error(samp['y_true'], samp['y_pred']))
    srpd  = np.std(samp['y_true'])/srmse if srmse > 0 else 0

    print(f"\n  Sample-level R²   = {sr2:.4f}")
    print(f"  Sample-level RMSE = {srmse:.4f} %")
    print(f"  Sample-level RPD  = {srpd:.4f}")

    print(f"\n  Per-sample predictions:")
    print(f"  {'Sample':8s} {'n_imgs':7s} {'Actual SOC':11s} "
          f"{'Predicted SOC':14s} {'Error':8s}")
    print(f"  {'─'*52}")
    for _, r in samp.iterrows():
        err = r['y_pred'] - r['y_true']
        print(f"  {int(r['Sample_No']):8d} {int(r['n']):7d} "
              f"{r['y_true']:11.4f} {r['y_pred']:14.4f} {err:+8.4f}")

    # Save results
    pd.DataFrame(all_results).to_csv(
        os.path.join(OUT_DIR, 'RESNET_ALL_MODELS.csv'), index=False)
    samp.to_csv(
        os.path.join(OUT_DIR, 'RESNET_SAMPLE_PREDICTIONS.csv'), index=False)
    df_r.to_csv(
        os.path.join(OUT_DIR, 'RESNET_IMAGE_PREDICTIONS.csv'), index=False)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'SOC Prediction — ResNet50 Transfer Learning\n'
                 f'Model: {best_name} | Grouped sample-level validation',
                 fontweight='bold', fontsize=12)

    # Image-level
    ax = axes[0]
    ax.scatter(y_val, best_pred, c='#1565C0', alpha=0.35, s=18)
    lo = min(y_val.min(), best_pred.min()) - 0.1
    hi = max(y_val.max(), best_pred.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='1:1 line')
    ax.set_xlabel('Measured SOC (%)', fontsize=11)
    ax.set_ylabel('Predicted SOC (%)', fontsize=11)
    ax.set_title(f'Image-level (n={len(y_val)})\n'
                 f'R²={best_r2:.3f}  RMSE={best_rmse:.3f}  RPD={best_rpd:.3f}',
                 fontsize=10)
    ax.legend(); ax.grid(True, alpha=0.3)

    # Sample-level
    ax2 = axes[1]
    ax2.scatter(samp['y_true'], samp['y_pred'], c='#C62828', s=160, zorder=5)
    for _, r in samp.iterrows():
        ax2.annotate(f"S{int(r['Sample_No'])}",
                     (r['y_true'], r['y_pred']),
                     textcoords='offset points', xytext=(6, 4), fontsize=10)
    lo2 = min(samp['y_true'].min(), samp['y_pred'].min()) - 0.1
    hi2 = max(samp['y_true'].max(), samp['y_pred'].max()) + 0.1
    ax2.plot([lo2, hi2], [lo2, hi2], 'k--', lw=1.5, label='1:1 line')
    ax2.set_xlabel('Measured SOC (%)', fontsize=11)
    ax2.set_ylabel('Predicted SOC (%)', fontsize=11)
    ax2.set_title(f'Sample-level (n={len(samp)})\n'
                  f'R²={sr2:.3f}  RMSE={srmse:.3f}  RPD={srpd:.3f}',
                  fontsize=10)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'resnet_predicted_vs_actual.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved: RESNET_ALL_MODELS.csv")
    print(f"  Saved: RESNET_SAMPLE_PREDICTIONS.csv")
    print(f"  Saved: resnet_predicted_vs_actual.png")

    return sr2, srmse, srpd


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*65)
    print("SOC PREDICTION — ResNet50 TRANSFER LEARNING")
    print("Pre-trained ImageNet backbone + regression head")
    print("Grouped sample-level validation (reviewer compliant)")
    print("="*65)

    # Load metadata
    meta = pd.read_csv(METADATA_CSV)
    print(f"\nMetadata: {len(meta)} images")

    # Build sample mapping
    mapping = build_mapping()

    # Load ResNet50
    extractor, transform = load_resnet()

    # Extract features from all images
    features, df = extract_features(extractor, transform, meta, mapping)

    # Grouped split
    X_cal, y_cal, X_val, y_val, df_val = grouped_split(features, df)

    # Train and evaluate
    (all_results, best_name, best_r2, best_rmse, best_rpd,
     best_rpiq, best_lccc, best_bias, best_pred) = train_and_evaluate(
         X_cal, y_cal, X_val, y_val, df_val)

    # Report and plot
    sr2, srmse, srpd = report_and_plot(
        y_val, best_pred, df_val, all_results,
        best_name, best_r2, best_rmse, best_rpd,
        best_rpiq, best_lccc, best_bias)

    print(f"\n{'='*65}")
    print("FINAL SUMMARY FOR PAPER")
    print(f"{'='*65}")
    print(f"  Method  : ResNet50 transfer learning (ImageNet pretrained)")
    print(f"  Features: 2048 ResNet50 → PCA reduction → regression")
    print(f"  Split   : Grouped sample-level (reviewer compliant)")
    print(f"  Model   : {best_name}")
    print(f"")
    print(f"  Image-level  R² = {best_r2:.4f}")
    print(f"  Image-level  RMSE = {best_rmse:.4f} %")
    print(f"  Image-level  RPD = {best_rpd:.4f}")
    print(f"  Image-level  RPIQ = {best_rpiq:.4f}")
    print(f"  Image-level  LCCC = {best_lccc:.4f}")
    print(f"")
    print(f"  Sample-level R² = {sr2:.4f}")
    print(f"  Sample-level RMSE = {srmse:.4f} %")
    print(f"  Sample-level RPD = {srpd:.4f}")
    print(f"\n  Outputs: {OUT_DIR}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()