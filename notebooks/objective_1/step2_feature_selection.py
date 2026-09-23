# # # """
# # # Multi-Criteria Feature Selection (MCFS) - CALIBRATION SET ONLY
# # # ===============================================================
# # # Proper cross-validation approach to prevent data leakage


# # # Output:  
# # #   - Features selected from calibration set ONLY
# # #   - Same features applied to validation set
# # #   - Ready for model training

# # # Pipeline (Ding et al., 2025; Viscarra Rossel & Behrens, 2010):
# # #   Stage 1: Quality Pre-filtering (calibration data only)
# # #   Stage 2: Mutual Information Scoring (calibration data only)
# # #   Stage 3: PCA-MI Hybrid Scoring (70-30, calibration data only)
# # #   Stage 4: LASSO Regularization Refinement (calibration data only)
  
# # #   Then: Apply selected features to validation set

# # # Outputs saved to ./feature_selection_calibration_output/
# # #   CALIBRATION SET:
# # #     calibration_selected_features.csv  - calibration data with selected features
# # #     feature_selection_report.txt       - complete summary
# # #     feature_selection_plots.png        - 4-panel visualization
  
# # #   VALIDATION SET:
# # #     validation_selected_features.csv   - validation data with SAME features
  
# # #   FEATURE LIST:
# # #     selected_features_list.txt         - just the feature names (for reference)

# # # Requirements: pip install pandas numpy matplotlib scipy scikit-learn
# # # Usage:        python feature_selection_calibration_only.py
# # # """

# # import os, warnings
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from scipy.stats import pearsonr
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.decomposition import PCA
# # from sklearn.impute import SimpleImputer
# # from sklearn.feature_selection import mutual_info_regression
# # from sklearn.linear_model import LassoCV
# # warnings.filterwarnings('ignore')

# # # ── PATHS ──────────────────────────────────────────────────────────────────────
# # HERE         = os.path.dirname(os.path.abspath(__file__))
# # CALIB_CSV = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step1_output\calibration_set.csv'
# # VALID_CSV = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step1_output\validation_set.csv'
# # OUT_DIR   = r'C:\Users\dharamkar.1\OneDrive - The Ohio State University\VSCode_Image_Processing_Reviewed\notebooks\objective_1\output_data\step2_output'

# # # ── CONFIGURATION ──────────────────────────────────────────────────────────────
# # # Target and identifier columns
# # TARGET_COL = 'soc'
# # ID_COL     = 'image_no'
# # EXCLUDE_COLS = ['filename', 'Numeric numbers', 'image_path', 'image_no', 
# #                 'soil_type', 'moisture', 'soc', 'Mahalanobis_Distance', 'Sample_No']

# # # Stage 1: Quality filtering thresholds
# # MIN_VARIANCE      = 0.01    # Remove features with variance < 0.01
# # MAX_MISSING_PCT   = 0.20    # Remove features with >20% missing values
# # P_VALUE_THRESHOLD = 0.10    # Keep features with p < 0.10 (marginal significance)

# # # Stage 3: Hybrid scoring weights
# # PCA_WEIGHT = 0.70           # Weight for PCA score (multivariate patterns)
# # MI_WEIGHT  = 0.30           # Weight for MI score (non-linear predictive power)

# # # Stage 3: Feature selection targets
# # N_FEATURES_TARGET = 20      # Target number of features
# # MAX_FEATURES      = 25      # Maximum features to retain

# # # Random seed
# # SEED = 42


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # STAGE 1: QUALITY PRE-FILTERING
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def stage1_quality_filter(df, candidate_features):
# #     """Stage 1: Quality-based pre-filtering (CALIBRATION SET ONLY)"""
    
# #     print("\n" + "="*80)
# #     print("STAGE 1: QUALITY PRE-FILTERING (CALIBRATION SET ONLY)")
# #     print("="*80)
# #     print(f"  Input features: {len(candidate_features)}")
    
# #     filtered_features = []
# #     removal_log = []
    
# #     for feature in candidate_features:
# #         # Check 1: Variance
# #         var = df[feature].var()
# #         if var < MIN_VARIANCE:
# #             removal_log.append({
# #                 'feature': feature,
# #                 'reason': 'low_variance',
# #                 'detail': f'variance = {var:.6f}',
# #                 'threshold': f'< {MIN_VARIANCE}'
# #             })
# #             continue
        
# #         # Check 2: Missing values
# #         missing_pct = df[feature].isna().sum() / len(df)
# #         if missing_pct > MAX_MISSING_PCT:
# #             removal_log.append({
# #                 'feature': feature,
# #                 'reason': 'missing_values',
# #                 'detail': f'{missing_pct*100:.2f}% missing',
# #                 'threshold': f'> {MAX_MISSING_PCT*100}%'
# #             })
# #             continue
        
# #         # Check 3: Statistical significance with SOC
# #         x = df[feature].dropna()
# #         y = df.loc[x.index, TARGET_COL]
        
# #         if len(x) < 10:
# #             removal_log.append({
# #                 'feature': feature,
# #                 'reason': 'insufficient_data',
# #                 'detail': f'only {len(x)} valid observations',
# #                 'threshold': '< 10 samples'
# #             })
# #             continue
        
# #         try:
# #             r, p = pearsonr(x, y)
# #             if p >= P_VALUE_THRESHOLD:
# #                 removal_log.append({
# #                     'feature': feature,
# #                     'reason': 'not_significant',
# #                     'detail': f'p = {p:.4f}, r = {r:.4f}',
# #                     'threshold': f'p ≥ {P_VALUE_THRESHOLD}'
# #                 })
# #                 continue
# #         except Exception as e:
# #             removal_log.append({
# #                 'feature': feature,
# #                 'reason': 'correlation_error',
# #                 'detail': str(e),
# #                 'threshold': 'N/A'
# #             })
# #             continue
        
# #         # Passed all checks
# #         filtered_features.append(feature)
    
# #     print(f"\n  Results:")
# #     print(f"    Passed filter: {len(filtered_features)}")
# #     print(f"    Removed:       {len(removal_log)}")
    
# #     if len(removal_log) > 0:
# #         df_removed = pd.DataFrame(removal_log)
# #         print(f"\n  Removal breakdown:")
# #         for reason, count in df_removed['reason'].value_counts().items():
# #             print(f"    {reason:20s}: {count:3d}")
        
# #         df_removed.to_csv(os.path.join(OUT_DIR, 'stage1_removal_log.csv'), 
# #                          index=False)
# #         print(f"  Saved: stage1_removal_log.csv")
    
# #     return filtered_features, removal_log


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # STAGE 2: MUTUAL INFORMATION SCORING
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def stage2_mutual_information(df, features):
# #     """Stage 2: Compute Mutual Information scores (CALIBRATION SET ONLY)"""
    
# #     print("\n" + "="*80)
# #     print("STAGE 2: MUTUAL INFORMATION SCORING (CALIBRATION SET ONLY)")
# #     print("="*80)
# #     print(f"  Computing MI for {len(features)} features...")
    
# #     X = df[features].values
# #     y = df[TARGET_COL].values
    
# #     imputer = SimpleImputer(strategy='median')
# #     X_imputed = imputer.fit_transform(X)
    
# #     mi_scores = mutual_info_regression(X_imputed, y, 
# #                                        random_state=SEED, 
# #                                        n_neighbors=5)
    
# #     mi_max = mi_scores.max()
# #     mi_scores_norm = mi_scores / mi_max if mi_max > 0 else mi_scores
    
# #     df_mi = pd.DataFrame({
# #         'feature': features,
# #         'mi_score_raw': mi_scores,
# #         'mi_score_norm': mi_scores_norm,
# #         'rank_by_mi': range(1, len(features) + 1)
# #     }).sort_values('mi_score_raw', ascending=False).reset_index(drop=True)
    
# #     df_mi['rank_by_mi'] = range(1, len(df_mi) + 1)
    
# #     print(f"\n  MI Statistics:")
# #     print(f"    Mean:   {mi_scores.mean():.4f}")
# #     print(f"    Median: {np.median(mi_scores):.4f}")
# #     print(f"    Max:    {mi_scores.max():.4f}")
# #     print(f"    Min:    {mi_scores.min():.4f}")
    
# #     print(f"\n  Top 10 features by Mutual Information:")
# #     print(f"  {'Rank':<6} {'Feature':<30} {'MI Score':<12} {'Normalized'}")
# #     print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*12}")
# #     for idx, row in df_mi.head(10).iterrows():
# #         print(f"  {int(row['rank_by_mi']):<6} {row['feature']:<30} "
# #               f"{row['mi_score_raw']:<12.4f} {row['mi_score_norm']:<12.4f}")
    
# #     df_mi.to_csv(os.path.join(OUT_DIR, 'stage2_mutual_information.csv'), 
# #                  index=False)
# #     print(f"\n  Saved: stage2_mutual_information.csv")
    
# #     return df_mi, imputer


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # STAGE 3: PCA-MI HYBRID SCORING
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def stage3_pca_scoring(df, features):
# #     """Stage 3A: Compute PCA-based importance scores (CALIBRATION SET ONLY)"""
    
# #     print("\n" + "="*80)
# #     print("STAGE 3A: PCA VARIANCE SCORING (CALIBRATION SET ONLY)")
# #     print("="*80)
    
# #     X = df[features].values
# #     imputer = SimpleImputer(strategy='median')
# #     X_imputed = imputer.fit_transform(X)
    
# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(X_imputed)
    
# #     pca = PCA(n_components=0.95, random_state=SEED)
# #     pca.fit(X_scaled)
    
# #     n_components = pca.n_components_
# #     loadings = pca.components_
# #     variance_explained = pca.explained_variance_ratio_
    
# #     print(f"  PCA retained {n_components} components")
# #     print(f"  Total variance explained: {variance_explained.sum():.4f}")
    
# #     pca_scores = np.zeros(len(features))
# #     for i in range(len(features)):
# #         pca_scores[i] = np.sum(np.abs(loadings[:, i]) * variance_explained)
    
# #     pca_max = pca_scores.max()
# #     pca_scores_norm = pca_scores / pca_max if pca_max > 0 else pca_scores
    
# #     df_pca = pd.DataFrame({
# #         'feature': features,
# #         'pca_score_raw': pca_scores,
# #         'pca_score_norm': pca_scores_norm,
# #         'rank_by_pca': range(1, len(features) + 1)
# #     }).sort_values('pca_score_raw', ascending=False).reset_index(drop=True)
    
# #     df_pca['rank_by_pca'] = range(1, len(df_pca) + 1)
    
# #     print(f"\n  Top 10 features by PCA score:")
# #     print(f"  {'Rank':<6} {'Feature':<30} {'PCA Score':<12} {'Normalized'}")
# #     print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*12}")
# #     for idx, row in df_pca.head(10).iterrows():
# #         print(f"  {int(row['rank_by_pca']):<6} {row['feature']:<30} "
# #               f"{row['pca_score_raw']:<12.4f} {row['pca_score_norm']:<12.4f}")
    
# #     df_pca.to_csv(os.path.join(OUT_DIR, 'stage3_pca_scores.csv'), index=False)
# #     print(f"\n  Saved: stage3_pca_scores.csv")
    
# #     return df_pca, pca, scaler


# # def stage3_combine_scores(df_pca, df_mi):
# #     """Stage 3B: Combine PCA and MI scores with 70-30 weighting"""
    
# #     print("\n" + "="*80)
# #     print(f"STAGE 3B: HYBRID SCORING ({int(PCA_WEIGHT*100)}-{int(MI_WEIGHT*100)} PCA-MI)")
# #     print("="*80)
    
# #     df_combined = df_pca[['feature', 'pca_score_raw', 'pca_score_norm']].merge(
# #         df_mi[['feature', 'mi_score_raw', 'mi_score_norm']], 
# #         on='feature'
# #     )
    
# #     df_combined['combined_score'] = (
# #         PCA_WEIGHT * df_combined['pca_score_norm'] +
# #         MI_WEIGHT * df_combined['mi_score_norm']
# #     )
    
# #     df_combined = df_combined.sort_values('combined_score', ascending=False)
# #     df_combined['rank'] = range(1, len(df_combined) + 1)
# #     df_combined = df_combined.reset_index(drop=True)
    
# #     print(f"\n  Weighting scheme:")
# #     print(f"    PCA (multivariate patterns):  {PCA_WEIGHT*100:.0f}%")
# #     print(f"    MI (non-linear predictive):   {MI_WEIGHT*100:.0f}%")
    
# #     print(f"\n  Top 15 features by combined score:")
# #     print(f"  {'Rank':<6} {'Feature':<25} {'Combined':<10} {'PCA':<10} {'MI':<10}")
# #     print(f"  {'-'*6} {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
# #     for idx, row in df_combined.head(15).iterrows():
# #         print(f"  {int(row['rank']):<6} {row['feature']:<25} "
# #               f"{row['combined_score']:<10.4f} {row['pca_score_norm']:<10.4f} "
# #               f"{row['mi_score_norm']:<10.4f}")
    
# #     df_combined.to_csv(os.path.join(OUT_DIR, 'stage3_combined_scores.csv'), 
# #                        index=False)
# #     print(f"\n  Saved: stage3_combined_scores.csv")
    
# #     return df_combined


# # def stage3_select_top_features(df_combined):
# #     """Stage 3C: Select top N features by combined score"""
    
# #     print("\n" + "="*80)
# #     print("STAGE 3C: TOP FEATURE SELECTION")
# #     print("="*80)
    
# #     n_select = min(N_FEATURES_TARGET, MAX_FEATURES, len(df_combined))
    
# #     top_features = df_combined.head(n_select)['feature'].tolist()
    
# #     print(f"  Target features: {N_FEATURES_TARGET}")
# #     print(f"  Bounds: {MAX_FEATURES}")
# #     print(f"  Selected: {len(top_features)} features")
    
# #     return top_features


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # STAGE 4: LASSO REGULARIZATION REFINEMENT
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def stage4_lasso_refinement(df, features, df_combined):
# #     """Stage 4: LASSO-based automatic redundancy removal (CALIBRATION SET ONLY)"""
    
# #     print("\n" + "="*80)
# #     print("STAGE 4: LASSO REGULARIZATION REFINEMENT (CALIBRATION SET ONLY)")
# #     print("="*80)
# #     print(f"  Input features: {len(features)}")
    
# #     X = df[features].values
# #     y = df[TARGET_COL].values
    
# #     imputer = SimpleImputer(strategy='median')
# #     X_imputed = imputer.fit_transform(X)
    
# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(X_imputed)
# #    #
# #    # 
# #    # 
# #    # # REPLACE grouped LASSO with standard LassoCV
# #     # Grouped CV is used for external validation (Step 3)
# #     # Standard CV is appropriate for feature selection (Step 2)

# #     print(f"\n  Running LassoCV (5-fold cross-validation)...")
# #     print(f"    Testing 100 alpha values...")

# #     alphas = np.logspace(-6, -1, 100)
# #     lasso = LassoCV(cv=5, random_state=SEED, max_iter=10000,
# #                         alphas=alphas, n_jobs=-1)
# #     lasso.fit(X_scaled, y)

# #     print(f"\n  LASSO Results:")
# #     print(f"    Optimal alpha:  {lasso.alpha_:.6f}")
# #     print(f"    Training R²:    {lasso.score(X_scaled, y):.4f}")

# #     non_zero_mask = lasso.coef_ != 0
# #     lasso_features = [f for f, keep in zip(features, non_zero_mask) if keep]
# #     lasso_coefs = lasso.coef_[non_zero_mask]

# #     n_removed = len(features) - len(lasso_features)
# #     print(f"\n    Features retained:  {len(lasso_features)}")
# #     print(f"    Features removed:   {n_removed} (coefficient shrunk to 0)")

# #     if len(lasso_features) == 0:
# #         raise ValueError("LASSO selected no features. Check alpha range.") 


# #     df_lasso = pd.DataFrame({
# #         'feature': lasso_features,
# #         'lasso_coefficient': lasso_coefs if len(lasso_coefs) > 0 else [0]*len(lasso_features),
# #         'abs_coefficient': np.abs(lasso_coefs) if len(lasso_coefs) > 0 else [0]*len(lasso_features)
# #     }).sort_values('abs_coefficient', ascending=False)

# #     print(f"\n  Top 10 features by |LASSO coefficient|:")
# #     print(f"  {'Rank':<6} {'Feature':<30} {'Coefficient':<15} {'|Coefficient|'}")
# #     print(f"  {'-'*6} {'-'*30} {'-'*15} {'-'*15}")
# #     for idx, row in df_lasso.head(10).iterrows():
# #         print(f"  {idx+1:<6} {row['feature']:<30} "
# #             f"{row['lasso_coefficient']:<15.4f} {row['abs_coefficient']:<15.4f}")

# #     df_lasso.to_csv(os.path.join(OUT_DIR, 'stage4_lasso_coefficients.csv'), 
# #                     index=False)
# #     print(f"\n  Saved: stage4_lasso_coefficients.csv")

# #     return lasso_features, lasso, df_lasso


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # APPLY TO VALIDATION SET
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def apply_to_validation(validation_csv, selected_features):
# #     """Apply selected features to validation set (NO feature selection on validation!)"""
    
# #     print("\n" + "="*80)
# #     print("APPLYING SELECTED FEATURES TO VALIDATION SET")
# #     print("="*80)
# #     print(f"  ⚠️  IMPORTANT: Validation set was NOT used for feature selection!")
# #     print(f"  ✅  Features selected from calibration set only")
    
# #     df_valid = pd.read_csv(validation_csv)
    
# #     print(f"\n  Validation set: {len(df_valid)} samples")
# #     print(f"  Extracting {len(selected_features)} selected features...")
    
# #     # Check if all selected features exist
# #     missing_features = [f for f in selected_features if f not in df_valid.columns]
# #     if missing_features:
# #         print(f"\n  ⚠️  WARNING: {len(missing_features)} features missing in validation set:")
# #         for f in missing_features:
# #             print(f"    - {f}")
# #         selected_features = [f for f in selected_features if f in df_valid.columns]
# #         print(f"  Using {len(selected_features)} available features")
    
# #     # Extract selected features + metadata
# #     cols_to_keep = [ID_COL, TARGET_COL] + selected_features
    
# #     # Add metadata if exists
# #     for col in ['moisture', 'soil_type']:
# #         if col in df_valid.columns and col not in cols_to_keep:
# #             cols_to_keep.append(col)
    
# #     df_valid_selected = df_valid[cols_to_keep].copy()
    
# #     output_path = os.path.join(OUT_DIR, 'validation_selected_features.csv')
# #     df_valid_selected.to_csv(output_path, index=False)
    
# #     print(f"\n  Saved: validation_selected_features.csv ({len(df_valid_selected)} × {len(df_valid_selected.columns)})")
    
# #     return df_valid_selected


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # VISUALIZATION
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def create_visualization(df_combined, final_features):
# #     """Create 4-panel visualization"""
    
# #     print("\n" + "="*80)
# #     print("CREATING VISUALIZATIONS")
# #     print("="*80)
    
# #     fig, axes = plt.subplots(2, 2, figsize=(18, 14))
# #     fig.suptitle(
# #         'Multi-Criteria Feature Selection (CALIBRATION SET ONLY)\n'
# #         f'PCA-MI Hybrid (70-30) + LASSO  |  '
# #         f'Final: {len(final_features)} features selected',
# #         fontsize=14, fontweight='bold', y=0.995
# #     )
    
# #     # Panel 1: Combined Scores
# #     ax = axes[0, 0]
# #     df_plot = df_combined.head(20)
# #     colors = ['#2E7D32' if f in final_features else '#BDBDBD' 
# #               for f in df_plot['feature']]
    
# #     y_pos = np.arange(len(df_plot))
# #     ax.barh(y_pos, df_plot['combined_score'], color=colors, 
# #             edgecolor='black', linewidth=1.2, alpha=0.85)
# #     ax.set_yticks(y_pos)
# #     ax.set_yticklabels(df_plot['feature'], fontsize=9)
# #     ax.set_xlabel('Combined Score (70% PCA + 30% MI)', 
# #                   fontweight='bold', fontsize=11)
# #     ax.set_title('Top 20 Features by Multi-Criteria Score\n'
# #                  '(Green = Selected by LASSO)', 
# #                  fontweight='bold', fontsize=11)
# #     ax.grid(axis='x', alpha=0.3, ls=':')
# #     ax.invert_yaxis()
# #     ax.spines[['top', 'right']].set_visible(False)
    
# #     # Panel 2: PCA vs MI Scatter
# #     ax = axes[0, 1]
# #     selected_mask = df_combined['feature'].isin(final_features)
    
# #     ax.scatter(df_combined[~selected_mask]['pca_score_norm'],
# #                df_combined[~selected_mask]['mi_score_norm'],
# #                s=70, c='#BDBDBD', alpha=0.5, edgecolors='black', 
# #                linewidth=0.5, label='Not Selected', zorder=2)
    
# #     ax.scatter(df_combined[selected_mask]['pca_score_norm'],
# #                df_combined[selected_mask]['mi_score_norm'],
# #                s=140, c='#2E7D32', alpha=0.9, edgecolors='black',
# #                linewidth=1.5, label='Selected', marker='s', zorder=3)
    
# #     for idx, row in df_combined[selected_mask].iterrows():
# #         ax.annotate(row['feature'],
# #                    (row['pca_score_norm'], row['mi_score_norm']),
# #                    fontsize=7, alpha=0.75, xytext=(4, 4),
# #                    textcoords='offset points')
    
# #     ax.set_xlabel('PCA Score (Multivariate Patterns)', 
# #                   fontweight='bold', fontsize=11)
# #     ax.set_ylabel('MI Score (Non-linear Predictive Power)', 
# #                   fontweight='bold', fontsize=11)
# #     ax.set_title('Feature Selection Space',
# #                  fontweight='bold', fontsize=11)
# #     ax.legend(fontsize=10, loc='lower right')
# #     ax.grid(alpha=0.3, ls=':')
# #     ax.spines[['top', 'right']].set_visible(False)
    
# #     # Panel 3: Score Composition
# #     ax = axes[1, 0]
# #     df_sel = df_combined[df_combined['feature'].isin(final_features)].copy()
# #     df_sel = df_sel.sort_values('combined_score', ascending=True)
    
# #     y_pos = np.arange(len(df_sel))
# #     pca_contrib = df_sel['pca_score_norm'] * PCA_WEIGHT
# #     mi_contrib = df_sel['mi_score_norm'] * MI_WEIGHT
    
# #     ax.barh(y_pos, pca_contrib, height=0.7, 
# #             label=f'PCA ({int(PCA_WEIGHT*100)}%)',
# #             color='#1565C0', edgecolor='black', linewidth=1, alpha=0.85)
# #     ax.barh(y_pos, mi_contrib, height=0.7, left=pca_contrib,
# #             label=f'MI ({int(MI_WEIGHT*100)}%)',
# #             color='#FF6F00', edgecolor='black', linewidth=1, alpha=0.85)
    
# #     ax.set_yticks(y_pos)
# #     ax.set_yticklabels(df_sel['feature'], fontsize=8)
# #     ax.set_xlabel('Score Contribution', fontweight='bold', fontsize=11)
# #     ax.set_title('Score Composition (Selected Features)',
# #                  fontweight='bold', fontsize=11)
# #     ax.legend(fontsize=9, loc='lower right')
# #     ax.grid(axis='x', alpha=0.3, ls=':')
# #     ax.spines[['top', 'right']].set_visible(False)
    
# #     # Panel 4: Score Distribution
# #     ax = axes[1, 1]
    
# #     ax.hist(df_combined['combined_score'], bins=25, alpha=0.5,
# #             color='#757575', edgecolor='black', linewidth=0.8,
# #             label='All Features', zorder=2)
    
# #     selected_scores = df_combined[selected_mask]['combined_score']
# #     ax.hist(selected_scores, bins=12, alpha=0.85,
# #             color='#2E7D32', edgecolor='black', linewidth=1.2,
# #             label=f'Selected (n={len(final_features)})', zorder=3)
    
# #     threshold = selected_scores.min()
# #     ax.axvline(threshold, color='#C62828', linestyle='--', linewidth=2.5,
# #                label=f'Threshold ({threshold:.3f})', zorder=4)
    
# #     ax.set_xlabel('Combined Score', fontweight='bold', fontsize=11)
# #     ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
# #     ax.set_title('Score Distribution',
# #                  fontweight='bold', fontsize=11)
# #     ax.legend(fontsize=9)
# #     ax.grid(alpha=0.3, ls=':')
# #     ax.spines[['top', 'right']].set_visible(False)
    
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(OUT_DIR, 'feature_selection_plots.png'),
# #                 dpi=160, bbox_inches='tight')
# #     plt.close()
# #     print(f"  Saved: feature_selection_plots.png")


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # SUMMARY REPORT
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def write_summary_report(df_calib, df_valid, candidate_features, 
# #                         filtered_features, top_features, final_features, 
# #                         df_combined, df_lasso):
# #     """Generate comprehensive summary report"""
    
# #     df_final = df_combined[df_combined['feature'].isin(final_features)].copy()
# #     df_final = df_final.merge(df_lasso[['feature', 'lasso_coefficient']], 
# #                               on='feature', how='left')
# #     df_final = df_final.sort_values('combined_score', ascending=False)
    
# #     lines = [
# #         "="*80,
# #         "MULTI-CRITERIA FEATURE SELECTION (MCFS)",
# #         "CALIBRATION SET ONLY - NO DATA LEAKAGE",
# #         "="*80,
# #         "",
# #         "── PROPER CROSS-VALIDATION METHODOLOGY ──────────────────────────────",
# #         "  ✅ Feature selection performed on CALIBRATION SET ONLY",
# #         "  ✅ Validation set was NOT used for feature selection",
# #         "  ✅ Same features applied to validation set",
# #         "  ✅ No data leakage - publication-ready approach",
# #         "",
# #         "── DATASET ─────────────────────────────────────────────────────────",
# #         f"  Calibration set:  {len(df_calib)} samples (used for feature selection)",
# #         f"  Validation set:   {len(df_valid)} samples (features applied only)",
# #         f"  Target variable:  {TARGET_COL}",
# #         f"  Candidate features: {len(candidate_features)}",
# #         "",
# #         "── STAGE 1: QUALITY PRE-FILTERING ──────────────────────────────────",
# #         f"  Thresholds:",
# #         f"    Min variance:        > {MIN_VARIANCE}",
# #         f"    Max missing:         < {MAX_MISSING_PCT*100}%",
# #         f"    Significance:        p < {P_VALUE_THRESHOLD}",
# #         "",
# #         f"  Results (calibration set):",
# #         f"    Input features:      {len(candidate_features)}",
# #         f"    Passed filter:       {len(filtered_features)}",
# #         f"    Removed:             {len(candidate_features) - len(filtered_features)}",
# #         "",
# #         "── STAGE 2: MUTUAL INFORMATION ─────────────────────────────────────",
# #         f"  Method: sklearn.feature_selection.mutual_info_regression",
# #         f"  Random state: {SEED}",
# #         f"  N neighbors: 5",
# #         f"  Data: Calibration set only ({len(df_calib)} samples)",
# #         "",
# #         "── STAGE 3: PCA-MI HYBRID SCORING ──────────────────────────────────",
# #         f"  Weighting scheme:",
# #         f"    PCA weight (multivariate):    {PCA_WEIGHT*100:.0f}%",
# #         f"    MI weight (non-linear):       {MI_WEIGHT*100:.0f}%",
# #         "",
# #         f"  Combined Score = {PCA_WEIGHT} × PCA_Score + {MI_WEIGHT} × MI_Score",
# #         "",
# #         f"  Feature selection:",
# #         f"    Target features:     {N_FEATURES_TARGET}",
# #         f"    Bounds:              {MAX_FEATURES}",
# #         f"    Selected for LASSO:  {len(top_features)}",
# #         "",
# #         "── STAGE 4: LASSO REGULARIZATION ───────────────────────────────────",
# #         f"  Method: LassoCV (5-fold cross-validation)",
# #         f"  Alpha values tested: 100",
# #         f"  Data: Calibration set only",
# #         "",
# #         f"  Results:",
# #         f"    Input to LASSO:      {len(top_features)}",
# #         f"    Final selected:      {len(final_features)}",
# #         f"    Removed by LASSO:    {len(top_features) - len(final_features)}",
# #         "",
# #         "── FINAL SELECTED FEATURES ─────────────────────────────────────────",
# #         f"  Total: {len(final_features)} features",
# #         f"  Reduction: {(1 - len(final_features)/len(candidate_features))*100:.1f}% "
# #         f"({len(candidate_features)} → {len(final_features)})",
# #         "",
# #         "  ✅ These features selected from CALIBRATION SET",
# #         "  ✅ Applied to VALIDATION SET (no leakage)",
# #         "",
# #         "  Ranked by combined score:",
# #         "",
# #         f"  {'Rank':<6} {'Feature':<25} {'Combined':<10} {'PCA':<10} {'MI':<10} {'LASSO Coef'}",
# #         f"  {'-'*6} {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*12}",
# #     ]
    
# #     for idx, row in df_final.iterrows():
# #         lines.append(
# #             f"  {int(row['rank']):<6} {row['feature']:<25} "
# #             f"{row['combined_score']:<10.4f} {row['pca_score_norm']:<10.4f} "
# #             f"{row['mi_score_norm']:<10.4f} {row['lasso_coefficient']:<12.4f}"
# #         )
    
# #     lines.extend([
# #         "",
# #         "── OUTPUT FILES ────────────────────────────────────────────────────",
# #         f"  CALIBRATION SET:",
# #         f"    calibration_selected_features.csv - calibration with selected features",
# #         f"    stage1_removal_log.csv            - removed features with reasons",
# #         f"    stage2_mutual_information.csv     - MI scores",
# #         f"    stage3_pca_scores.csv             - PCA scores",
# #         f"    stage3_combined_scores.csv        - PCA-MI hybrid scores",
# #         f"    stage4_lasso_coefficients.csv     - LASSO coefficients",
# #         "",
# #         f"  VALIDATION SET:",
# #         f"    validation_selected_features.csv  - validation with SAME features",
# #         "",
# #         f"  FEATURE LIST:",
# #         f"    selected_features_list.txt        - just feature names",
# #         "",
# #         f"  REPORTS:",
# #         f"    feature_selection_plots.png       - 4-panel visualization",
# #         f"    feature_selection_report.txt      - this report",
# #         "",
# #         "── READY FOR MODEL TRAINING ────────────────────────────────────────",
# #         "  Next steps:",
# #         "    1. Train models on calibration_selected_features.csv",
# #         "    2. Validate on validation_selected_features.csv",
# #         "    3. Both datasets have SAME {len(final_features)} features",
# #         "",
# #         "="*80,
# #     ])
    
# #     report = "\n".join(lines)
# #     print("\n" + report)
    
# #     with open(os.path.join(OUT_DIR, 'feature_selection_report.txt'), 'w', encoding='utf-8') as f:
# #         f.write(report)
# #     print(f"\n  Saved: feature_selection_report.txt")


# # # ═══════════════════════════════════════════════════════════════════════════════
# # # MAIN
# # # ═══════════════════════════════════════════════════════════════════════════════
# # def main():
# #     """Main execution pipeline"""
    
# #     print("\n" + "="*80)
# #     print("MULTI-CRITERIA FEATURE SELECTION")
# #     print("CALIBRATION SET ONLY (Proper Cross-Validation)")
# #     print("="*80)
# #     print(f"\nCalibration: {CALIB_CSV}")
# #     print(f"Validation:  {VALID_CSV}")
# #     print(f"Output:      {OUT_DIR}/")
    
# #     # Load calibration set
# #     print("\n" + "="*80)
# #     print("LOADING CALIBRATION SET")
# #     print("="*80)
# #     df_calib = pd.read_csv(CALIB_CSV)
# #     print(f"  Calibration set: {len(df_calib)} samples × {len(df_calib.columns)} columns")
    
# #     # Get candidate features
# #     candidate_features = [col for col in df_calib.columns 
# #                          if col not in EXCLUDE_COLS]
# #     candidate_features = [col for col in candidate_features 
# #                          if df_calib[col].dtype in [np.float64, np.int64]]
    
# #     print(f"  Candidate features: {len(candidate_features)}")
# #     print(f"  Target: {TARGET_COL}")
    
# #     # Run 4-stage MCFS on calibration set ONLY
# #     filtered_features, _ = stage1_quality_filter(df_calib, candidate_features)
# #     df_mi, _ = stage2_mutual_information(df_calib, filtered_features)
# #     df_pca, _, _ = stage3_pca_scoring(df_calib, filtered_features)
# #     df_combined = stage3_combine_scores(df_pca, df_mi)
# #     top_features = stage3_select_top_features(df_combined)
# #     final_features, _, df_lasso = stage4_lasso_refinement(
# #         df_calib, top_features, df_combined
# #     )
    
# #     # Save calibration set with selected features
# #     print("\n" + "="*80)
# #     print("SAVING CALIBRATION SET WITH SELECTED FEATURES")
# #     print("="*80)
    
# #     cols_to_keep = [ID_COL, TARGET_COL] + final_features
# #     for col in ['moisture', 'soil_type']:
# #         if col in df_calib.columns and col not in cols_to_keep:
# #             cols_to_keep.append(col)
    
# #     df_calib_selected = df_calib[cols_to_keep].copy()
# #     calib_output = os.path.join(OUT_DIR, 'calibration_selected_features.csv')
# #     df_calib_selected.to_csv(calib_output, index=False)
    
# #     print(f"  Calibration dataset shape: {df_calib_selected.shape}")
# #     print(f"  Saved: calibration_selected_features.csv")
    
# #     # Apply to validation set
# #     df_valid_selected = apply_to_validation(VALID_CSV, final_features)
    
# #     # Save feature list
# #     feature_list_path = os.path.join(OUT_DIR, 'selected_features_list.txt')
# #     with open(feature_list_path, 'w') as f:
# #         f.write("SELECTED FEATURES (from calibration set)\n")
# #         f.write("="*60 + "\n\n")
# #         for i, feat in enumerate(final_features, 1):
# #             f.write(f"{i:2d}. {feat}\n")
# #     print(f"\n  Saved: selected_features_list.txt")
    
# #     # Create visualizations
# #     create_visualization(df_combined, final_features)
    
# #     # Generate summary report
# #     df_valid = pd.read_csv(VALID_CSV)
# #     write_summary_report(df_calib, df_valid, candidate_features, 
# #                         filtered_features, top_features, final_features, 
# #                         df_combined, df_lasso)

# #     print("\n" + "="*80)
# #     print("✅ FEATURE SELECTION COMPLETE (NO DATA LEAKAGE)")
# #     print("="*80)
# #     print(f"\n  Selected {len(final_features)} features from {len(candidate_features)} candidates")
# #     print(f"  Reduction: {(1 - len(final_features)/len(candidate_features))*100:.1f}%")
# #     print(f"\n  ✅ Calibration: {len(df_calib_selected)} samples × {len(final_features)} features")
# #     print(f"  ✅ Validation:  {len(df_valid_selected)} samples × {len(final_features)} features")
# #     print(f"\n  All outputs saved to: {OUT_DIR}/")
# #     print("\n  🎯 READY FOR MODEL TRAINING!")
# #     print("="*80 + "\n")


# # if __name__ == '__main__':
# #     main()

# # # """
# # # STEP 2 — MULTI-CRITERIA FEATURE RANKING
# # # =======================================

# # # Purpose
# # # -------
# # # Step 2 performs feature ranking only.

# # # It DOES NOT decide the final number of retained predictors.

# # # Final feature dimensionality is selected in Step 3 using:
# # #     - nested grouped cross-validation
# # #     - grouping by physical soil Sample_No
# # #     - candidate feature counts
# # #     - one-standard-error rule

# # # Step 2 outputs:
# # #     1. Full-calibration PCA-MI ranking for reporting
# # #     2. Full candidate-feature calibration dataset
# # #     3. Full candidate-feature external-validation dataset

# # # IMPORTANT:
# # # The external validation set is NOT used for feature ranking.
# # # """

# # # import os
# # # import warnings
# # # import numpy as np
# # # import pandas as pd

# # # from scipy.stats import pearsonr
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.decomposition import PCA
# # # from sklearn.impute import SimpleImputer
# # # from sklearn.feature_selection import mutual_info_regression

# # # warnings.filterwarnings("ignore")


# # # # =============================================================================
# # # # PATHS
# # # # =============================================================================

# # # PROJECT_ROOT = (
# # #     r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
# # #     r"\VSCode_Image_Processing_Reviewed"
# # # )

# # # CALIB_CSV = os.path.join(
# # #     PROJECT_ROOT,
# # #     "notebooks",
# # #     "objective_1",
# # #     "output_data",
# # #     "step1_output",
# # #     "calibration_set.csv",
# # # )

# # # VALID_CSV = os.path.join(
# # #     PROJECT_ROOT,
# # #     "notebooks",
# # #     "objective_1",
# # #     "output_data",
# # #     "step1_output",
# # #     "validation_set.csv",
# # # )

# # # OUT_DIR = os.path.join(
# # #     PROJECT_ROOT,
# # #     "notebooks",
# # #     "objective_1",
# # #     "output_data",
# # #     "step2_output",
# # # )

# # # os.makedirs(OUT_DIR, exist_ok=True)


# # # # =============================================================================
# # # # SETTINGS
# # # # =============================================================================

# # # TARGET_COL = "soc"
# # # GROUP_COL = "Sample_No"
# # # ID_COL = "image_no"

# # # EXCLUDE_COLS = [
# # #     "filename",
# # #     "Numeric numbers",
# # #     "image_path",
# # #     "image_no",
# # #     "soil_type",
# # #     "moisture",
# # #     "soc",
# # #     "Mahalanobis_Distance",
# # #     "Sample_No",
# # # ]

# # # MIN_VARIANCE = 0.01
# # # MAX_MISSING_PCT = 0.20
# # # P_VALUE_THRESHOLD = 0.10

# # # PCA_WEIGHT = 0.70
# # # MI_WEIGHT = 0.30

# # # SEED = 42


# # # # =============================================================================
# # # # STAGE 1 — QUALITY FILTER
# # # # =============================================================================

# # # def stage1_quality_filter(df, candidate_features):

# # #     print("\n" + "=" * 80)
# # #     print("STAGE 1 — QUALITY PRE-FILTERING")
# # #     print("=" * 80)

# # #     filtered = []
# # #     removal_log = []

# # #     for feature in candidate_features:

# # #         variance = df[feature].var()

# # #         if variance < MIN_VARIANCE:
# # #             removal_log.append(
# # #                 {
# # #                     "feature": feature,
# # #                     "reason": "low_variance",
# # #                     "value": variance,
# # #                 }
# # #             )
# # #             continue

# # #         missing_pct = df[feature].isna().mean()

# # #         if missing_pct > MAX_MISSING_PCT:
# # #             removal_log.append(
# # #                 {
# # #                     "feature": feature,
# # #                     "reason": "missing_values",
# # #                     "value": missing_pct,
# # #                 }
# # #             )
# # #             continue

# # #         x = df[feature].dropna()
# # #         y = df.loc[x.index, TARGET_COL]

# # #         if len(x) < 10:
# # #             removal_log.append(
# # #                 {
# # #                     "feature": feature,
# # #                     "reason": "insufficient_data",
# # #                     "value": len(x),
# # #                 }
# # #             )
# # #             continue

# # #         try:
# # #             r, p = pearsonr(x, y)

# # #             if p >= P_VALUE_THRESHOLD:
# # #                 removal_log.append(
# # #                     {
# # #                         "feature": feature,
# # #                         "reason": "not_significant",
# # #                         "value": p,
# # #                     }
# # #                 )
# # #                 continue

# # #         except Exception:
# # #             removal_log.append(
# # #                 {
# # #                     "feature": feature,
# # #                     "reason": "correlation_error",
# # #                     "value": np.nan,
# # #                 }
# # #             )
# # #             continue

# # #         filtered.append(feature)

# # #     print(f"  Input features : {len(candidate_features)}")
# # #     print(f"  Retained       : {len(filtered)}")
# # #     print(f"  Removed        : {len(removal_log)}")

# # #     if removal_log:
# # #         pd.DataFrame(removal_log).to_csv(
# # #             os.path.join(OUT_DIR, "stage1_removal_log.csv"),
# # #             index=False,
# # #         )

# # #     return filtered


# # # # =============================================================================
# # # # STAGE 2 — MUTUAL INFORMATION
# # # # =============================================================================

# # # def stage2_mutual_information(df, features):

# # #     print("\n" + "=" * 80)
# # #     print("STAGE 2 — MUTUAL INFORMATION")
# # #     print("=" * 80)

# # #     X = df[features]
# # #     y = df[TARGET_COL].values

# # #     imputer = SimpleImputer(strategy="median")
# # #     X_imp = imputer.fit_transform(X)

# # #     mi = mutual_info_regression(
# # #         X_imp,
# # #         y,
# # #         random_state=SEED,
# # #         n_neighbors=5,
# # #     )

# # #     mi_max = np.max(mi)
# # #     mi_norm = mi / mi_max if mi_max > 0 else mi

# # #     result = pd.DataFrame(
# # #         {
# # #             "feature": features,
# # #             "mi_score_raw": mi,
# # #             "mi_score_norm": mi_norm,
# # #         }
# # #     )

# # #     result = result.sort_values(
# # #         "mi_score_raw",
# # #         ascending=False,
# # #     ).reset_index(drop=True)

# # #     result["rank_by_mi"] = np.arange(1, len(result) + 1)

# # #     print("\nTop 10 MI features:")
# # #     print(
# # #         result[
# # #             ["rank_by_mi", "feature", "mi_score_raw"]
# # #         ].head(10).to_string(index=False)
# # #     )

# # #     result.to_csv(
# # #         os.path.join(
# # #             OUT_DIR,
# # #             "stage2_mutual_information.csv",
# # #         ),
# # #         index=False,
# # #     )

# # #     return result


# # # # =============================================================================
# # # # STAGE 3A — PCA SCORING
# # # # =============================================================================

# # # def stage3_pca_scoring(df, features):

# # #     print("\n" + "=" * 80)
# # #     print("STAGE 3A — PCA SCORING")
# # #     print("=" * 80)

# # #     X = df[features]

# # #     imputer = SimpleImputer(strategy="median")
# # #     X_imp = imputer.fit_transform(X)

# # #     scaler = StandardScaler()
# # #     X_scaled = scaler.fit_transform(X_imp)

# # #     pca = PCA(
# # #         n_components=0.95,
# # #         random_state=SEED,
# # #     )

# # #     pca.fit(X_scaled)

# # #     loadings = pca.components_
# # #     variance = pca.explained_variance_ratio_

# # #     scores = np.zeros(len(features))

# # #     for i in range(len(features)):
# # #         scores[i] = np.sum(
# # #             np.abs(loadings[:, i]) * variance
# # #         )

# # #     max_score = scores.max()

# # #     scores_norm = (
# # #         scores / max_score
# # #         if max_score > 0
# # #         else scores
# # #     )

# # #     result = pd.DataFrame(
# # #         {
# # #             "feature": features,
# # #             "pca_score_raw": scores,
# # #             "pca_score_norm": scores_norm,
# # #         }
# # #     )

# # #     result = result.sort_values(
# # #         "pca_score_raw",
# # #         ascending=False,
# # #     ).reset_index(drop=True)

# # #     result["rank_by_pca"] = np.arange(
# # #         1,
# # #         len(result) + 1,
# # #     )

# # #     print(
# # #         f"  PCA components retained: {pca.n_components_}"
# # #     )
# # #     print(
# # #         f"  Variance explained: "
# # #         f"{pca.explained_variance_ratio_.sum():.4f}"
# # #     )

# # #     result.to_csv(
# # #         os.path.join(
# # #             OUT_DIR,
# # #             "stage3_pca_scores.csv",
# # #         ),
# # #         index=False,
# # #     )

# # #     return result


# # # # =============================================================================
# # # # STAGE 3B — PCA-MI HYBRID
# # # # =============================================================================

# # # def stage3_combine_scores(df_pca, df_mi):

# # #     print("\n" + "=" * 80)
# # #     print("STAGE 3B — PCA-MI HYBRID RANKING")
# # #     print("=" * 80)

# # #     combined = df_pca[
# # #         ["feature", "pca_score_raw", "pca_score_norm"]
# # #     ].merge(
# # #         df_mi[
# # #             ["feature", "mi_score_raw", "mi_score_norm"]
# # #         ],
# # #         on="feature",
# # #     )

# # #     combined["combined_score"] = (
# # #         PCA_WEIGHT * combined["pca_score_norm"]
# # #         + MI_WEIGHT * combined["mi_score_norm"]
# # #     )

# # #     combined = combined.sort_values(
# # #         "combined_score",
# # #         ascending=False,
# # #     ).reset_index(drop=True)

# # #     combined["rank"] = np.arange(
# # #         1,
# # #         len(combined) + 1,
# # #     )

# # #     print("\nTop ranked features:")
# # #     print(
# # #         combined[
# # #             [
# # #                 "rank",
# # #                 "feature",
# # #                 "combined_score",
# # #                 "pca_score_norm",
# # #                 "mi_score_norm",
# # #             ]
# # #         ]
# # #         .head(20)
# # #         .to_string(index=False)
# # #     )

# # #     combined.to_csv(
# # #         os.path.join(
# # #             OUT_DIR,
# # #             "full_ranked_features.csv",
# # #         ),
# # #         index=False,
# # #     )

# # #     return combined


# # # # =============================================================================
# # # # MAIN
# # # # =============================================================================

# # # def main():

# # #     print("\n" + "=" * 80)
# # #     print("STEP 2 — FEATURE RANKING ONLY")
# # #     print("NO FINAL FEATURE COUNT IS SELECTED HERE")
# # #     print("=" * 80)

# # #     df_calib = pd.read_csv(CALIB_CSV)
# # #     df_valid = pd.read_csv(VALID_CSV)

# # #     print(
# # #         f"\nCalibration: "
# # #         f"{len(df_calib)} images, "
# # #         f"{df_calib[GROUP_COL].nunique()} physical soils"
# # #     )

# # #     print(
# # #         f"Validation : "
# # #         f"{len(df_valid)} images, "
# # #         f"{df_valid[GROUP_COL].nunique()} physical soils"
# # #     )

# # #     overlap = (
# # #         set(df_calib[GROUP_COL])
# # #         & set(df_valid[GROUP_COL])
# # #     )

# # #     if overlap:
# # #         raise ValueError(
# # #             f"Calibration/validation group overlap: "
# # #             f"{sorted(overlap)}"
# # #         )

# # #     print("Calibration-validation overlap: ZERO")

# # #     candidate_features = [
# # #         col
# # #         for col in df_calib.columns
# # #         if col not in EXCLUDE_COLS
# # #         and pd.api.types.is_numeric_dtype(
# # #             df_calib[col]
# # #         )
# # #     ]

# # #     print(
# # #         f"\nOriginal candidate features: "
# # #         f"{len(candidate_features)}"
# # #     )

# # #     # ---------------------------------------------------------
# # #     # Overall calibration ranking for reporting
# # #     # ---------------------------------------------------------

# # #     filtered = stage1_quality_filter(
# # #         df_calib,
# # #         candidate_features,
# # #     )

# # #     df_mi = stage2_mutual_information(
# # #         df_calib,
# # #         filtered,
# # #     )

# # #     df_pca = stage3_pca_scoring(
# # #         df_calib,
# # #         filtered,
# # #     )

# # #     df_combined = stage3_combine_scores(
# # #         df_pca,
# # #         df_mi,
# # #     )

# # #     top20 = df_combined.head(20).copy()

# # #     top20.to_csv(
# # #         os.path.join(
# # #             OUT_DIR,
# # #             "ranked_features_top20.csv",
# # #         ),
# # #         index=False,
# # #     )

# # #     # ---------------------------------------------------------
# # #     # IMPORTANT:
# # #     # Save ALL original candidate features for Step 3.
# # #     #
# # #     # Step 3 will repeat filtering + ranking separately
# # #     # inside every grouped CV training fold.
# # #     # ---------------------------------------------------------

# # #     metadata_cols = [
# # #         ID_COL,
# # #         TARGET_COL,
# # #         GROUP_COL,
# # #     ]

# # #     for c in ["soil_type", "moisture"]:
# # #         if c in df_calib.columns:
# # #             metadata_cols.append(c)

# # #     save_cols = (
# # #         metadata_cols
# # #         + candidate_features
# # #     )

# # #     save_cols = list(
# # #         dict.fromkeys(save_cols)
# # #     )

# # #     calib_output = df_calib[
# # #         [c for c in save_cols if c in df_calib.columns]
# # #     ].copy()

# # #     valid_output = df_valid[
# # #         [c for c in save_cols if c in df_valid.columns]
# # #     ].copy()

# # #     calib_output.to_csv(
# # #         os.path.join(
# # #             OUT_DIR,
# # #             "calibration_all_candidates.csv",
# # #         ),
# # #         index=False,
# # #     )

# # #     valid_output.to_csv(
# # #         os.path.join(
# # #             OUT_DIR,
# # #             "validation_all_candidates.csv",
# # #         ),
# # #         index=False,
# # #     )

# # #     # save candidate names
# # #     with open(
# # #         os.path.join(
# # #             OUT_DIR,
# # #             "candidate_features.txt",
# # #         ),
# # #         "w",
# # #     ) as f:

# # #         for feat in candidate_features:
# # #             f.write(feat + "\n")

# # #     print("\n" + "=" * 80)
# # #     print("STEP 2 COMPLETE")
# # #     print("=" * 80)

# # #     print(
# # #         f"\n  Candidate features: "
# # #         f"{len(candidate_features)}"
# # #     )

# # #     print(
# # #         f"  Stage-1 survivors: "
# # #         f"{len(filtered)}"
# # #     )

# # #     print(
# # #         f"  Full-calibration ranked features: "
# # #         f"{len(df_combined)}"
# # #     )

# # #     print("\nOutputs:")

# # #     print(
# # #         "  ranked_features_top20.csv"
# # #         "        <- descriptive overall ranking"
# # #     )

# # #     print(
# # #         "  calibration_all_candidates.csv"
# # #         "   <- Step 3 calibration input"
# # #     )

# # #     print(
# # #         "  validation_all_candidates.csv"
# # #         "    <- untouched external validation"
# # #     )

# # #     print(
# # #         "\nIMPORTANT:"
# # #         "\n  The Top-20 ranking above is NOT used as a"
# # #         "\n  fixed ranking during CV."
# # #         "\n  Step 3 recomputes the feature ranking"
# # #         "\n  independently within every training fold."
# # #     )


# # # if __name__ == "__main__":
# # #     main()


# """
# STEP 2 — DEFENSIBLE FEATURE SELECTION
# =====================================

# Feature selection performed on CALIBRATION DATA ONLY.

# Step 0:
#     Aggregate technical image replicates by Sample_No + SOC.

# Stage 1 — Quality control
#     1. Missingness > 20% -> remove
#     2. Min-max scaled variance < 0.01 -> remove

# Stage 2 — SOC relevance
#     A feature survives if EITHER:

#     A) |Spearman rho| >= 0.20 AND screening p < 0.10

#        OR

#     B) Observed Mutual Information exceeds the 95th percentile
#        of its permutation-based null distribution.

# Stage 3 — PCA dimensionality reduction
#     PCA is fit to standardized Stage-2 survivors.
#     PC1 + PC2 are used if cumulative explained variance >= 60%.

#     Feature importance threshold:
#         contribution > 1/p on PC1 OR PC2

#     where p = number of predictors entering PCA.

# Final predictors:
#     Passed Stage 2
#     AND have above-average contribution to PC1 or PC2.

# Validation data are NEVER used for feature selection.
# """

# import os
# import warnings
# import numpy as np
# import pandas as pd

# from scipy.stats import spearmanr

# from sklearn.preprocessing import (
#     StandardScaler,
#     MinMaxScaler,
# )
# from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.decomposition import PCA

# warnings.filterwarnings("ignore")


# # ================================================================
# # PATHS
# # ================================================================

# PROJECT_ROOT = (
#     r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
#     r"\VSCode_Image_Processing_Reviewed"
# )

# CALIB_CSV = os.path.join(
#     PROJECT_ROOT,
#     "notebooks",
#     "objective_1",
#     "output_data",
#     "step1_output",
#     "calibration_set.csv",
# )

# VALID_CSV = os.path.join(
#     PROJECT_ROOT,
#     "notebooks",
#     "objective_1",
#     "output_data",
#     "step1_output",
#     "validation_set.csv",
# )

# OUT_DIR = os.path.join(
#     PROJECT_ROOT,
#     "notebooks",
#     "objective_1",
#     "output_data",
#     "step2_output",
# )

# os.makedirs(OUT_DIR, exist_ok=True)


# # ================================================================
# # SETTINGS
# # ================================================================

# TARGET = "soc"
# GROUP = "Sample_No"

# EXCLUDE = [
#     "filename",
#     "Numeric numbers",
#     "image_path",
#     "image_no",
#     "Image_No",
#     "soil_type",
#     "moisture",
#     "soc",
#     "Sample_No",
#     "Mahalanobis_Distance",
#     "Is_Outlier",
#     "split",
# ]

# # ------------------------------------------------
# # Stage 1
# # ------------------------------------------------

# MAX_MISSING = 0.20
# MIN_SCALED_VARIANCE = 0.01

# # ------------------------------------------------
# # Stage 2
# # ------------------------------------------------

# MIN_ABS_SPEARMAN = 0.20
# SCREENING_P = 0.10

# N_MI_PERMUTATIONS = 500
# MI_NULL_PERCENTILE = 95

# # ------------------------------------------------
# # Stage 3
# # ------------------------------------------------

# MIN_PC12_CUMULATIVE_VARIANCE = 0.60

# SEED = 42


# # ================================================================
# # STEP 0 — AGGREGATE TECHNICAL REPLICATES
# # ================================================================

# def aggregate_replicates(df, feature_cols):

#     print("\n" + "=" * 75)
#     print("STEP 0 — TECHNICAL REPLICATE AGGREGATION")
#     print("=" * 75)

#     print(f"Original image rows : {len(df)}")

#     # Median across repeated photographs belonging to same
#     # Sample_No + SOC condition.
#     agg_features = (
#         df.groupby([GROUP, TARGET], as_index=False)[feature_cols]
#         .median()
#     )

#     # Keep simple metadata if available
#     metadata = []

#     for col in ["soil_type", "moisture"]:
#         if col in df.columns:

#             temp = (
#                 df.groupby([GROUP, TARGET], as_index=False)[col]
#                 .first()
#             )

#             metadata.append(temp)

#     result = agg_features.copy()

#     for meta in metadata:
#         result = result.merge(
#             meta,
#             on=[GROUP, TARGET],
#             how="left",
#         )

#     print(
#         f"Independent soil-condition rows after aggregation: "
#         f"{len(result)}"
#     )

#     return result


# # ================================================================
# # STAGE 1 — QUALITY CONTROL
# # ================================================================

# def stage1_quality_filter(df, features):

#     print("\n" + "=" * 75)
#     print("STAGE 1 — QUALITY CONTROL")
#     print("=" * 75)

#     retained = []
#     records = []

#     for feat in features:

#         missing = df[feat].isna().mean()

#         # Missingness threshold
#         if missing > MAX_MISSING:

#             records.append({
#                 "feature": feat,
#                 "missing_fraction": missing,
#                 "scaled_variance": np.nan,
#                 "stage1_pass": False,
#                 "reason": "missingness > 20%",
#             })

#             continue

#         x = df[[feat]].copy()

#         x = x.fillna(x.median())

#         # Min-max scaling makes variance threshold comparable
#         scaler = MinMaxScaler()
#         x_scaled = scaler.fit_transform(x)

#         variance = np.var(
#             x_scaled,
#             ddof=1,
#         )

#         if variance < MIN_SCALED_VARIANCE:

#             records.append({
#                 "feature": feat,
#                 "missing_fraction": missing,
#                 "scaled_variance": variance,
#                 "stage1_pass": False,
#                 "reason": "scaled variance < 0.01",
#             })

#             continue

#         retained.append(feat)

#         records.append({
#             "feature": feat,
#             "missing_fraction": missing,
#             "scaled_variance": variance,
#             "stage1_pass": True,
#             "reason": "retained",
#         })

#     report = pd.DataFrame(records)

#     print(f"Input features : {len(features)}")
#     print(f"Retained       : {len(retained)}")
#     print(f"Removed        : {len(features) - len(retained)}")

#     report.to_csv(
#         os.path.join(
#             OUT_DIR,
#             "stage1_quality_report.csv",
#         ),
#         index=False,
#     )

#     return retained, report


# # ================================================================
# # STAGE 2A — SPEARMAN CORRELATION
# # ================================================================

# def compute_spearman(df, features):

#     results = []

#     for feat in features:

#         x = df[feat]
#         y = df[TARGET]

#         valid = (
#             x.notna()
#             & y.notna()
#         )

#         rho, p = spearmanr(
#             x[valid],
#             y[valid],
#         )

#         results.append({
#             "feature": feat,
#             "spearman_rho": rho,
#             "abs_spearman": abs(rho),
#             "spearman_p": p,
#         })

#     return pd.DataFrame(results)


# # ================================================================
# # STAGE 2B — MUTUAL INFORMATION WITH PERMUTATION THRESHOLD
# # ================================================================

# def compute_permutation_mi(df, features):

#     rng = np.random.default_rng(SEED)

#     imputer = SimpleImputer(
#         strategy="median"
#     )

#     X = imputer.fit_transform(
#         df[features]
#     )

#     y = df[TARGET].values

#     # observed MI
#     observed = mutual_info_regression(
#         X,
#         y,
#         random_state=SEED,
#         n_neighbors=min(
#             5,
#             max(2, len(y) - 1),
#         ),
#     )

#     results = []

#     print(
#         f"\nRunning {N_MI_PERMUTATIONS} "
#         f"MI permutations..."
#     )

#     for j, feat in enumerate(features):

#         null_scores = []

#         feature_x = X[:, [j]]

#         for i in range(N_MI_PERMUTATIONS):

#             y_perm = rng.permutation(y)

#             mi_perm = mutual_info_regression(
#                 feature_x,
#                 y_perm,
#                 random_state=SEED + i,
#                 n_neighbors=min(
#                     5,
#                     max(2, len(y) - 1),
#                 ),
#             )[0]

#             null_scores.append(mi_perm)

#         threshold = np.percentile(
#             null_scores,
#             MI_NULL_PERCENTILE,
#         )

#         results.append({
#             "feature": feat,
#             "mi_observed": observed[j],
#             "mi_null_95": threshold,
#             "mi_pass": observed[j] > threshold,
#         })

#     return pd.DataFrame(results)


# # ================================================================
# # STAGE 2 — COMBINE SOC ASSOCIATION TESTS
# # ================================================================

# def stage2_soc_screening(df, features):

#     print("\n" + "=" * 75)
#     print("STAGE 2 — SOC RELEVANCE SCREENING")
#     print("=" * 75)

#     corr = compute_spearman(
#         df,
#         features,
#     )

#     mi = compute_permutation_mi(
#         df,
#         features,
#     )

#     result = corr.merge(
#         mi,
#         on="feature",
#     )

#     # Correlation criterion
#     result["corr_pass"] = (
#         (result["abs_spearman"] >= MIN_ABS_SPEARMAN)
#         &
#         (result["spearman_p"] < SCREENING_P)
#     )

#     # Final Stage-2 rule:
#     # pass correlation criterion OR MI criterion

#     result["stage2_pass"] = (
#         result["corr_pass"]
#         |
#         result["mi_pass"]
#     )

#     result = result.sort_values(
#         [
#             "stage2_pass",
#             "abs_spearman",
#             "mi_observed",
#         ],
#         ascending=[
#             False,
#             False,
#             False,
#         ],
#     )

#     retained = result.loc[
#         result["stage2_pass"],
#         "feature",
#     ].tolist()

#     print(
#         f"\nEntered Stage 2 : {len(features)}"
#     )

#     print(
#         f"Passed Stage 2  : {len(retained)}"
#     )

#     print("\nStage-2 results:")
#     print(
#         result[
#             [
#                 "feature",
#                 "spearman_rho",
#                 "spearman_p",
#                 "corr_pass",
#                 "mi_observed",
#                 "mi_null_95",
#                 "mi_pass",
#                 "stage2_pass",
#             ]
#         ].to_string(index=False)
#     )

#     result.to_csv(
#         os.path.join(
#             OUT_DIR,
#             "stage2_soc_screening.csv",
#         ),
#         index=False,
#     )

#     return retained, result


# # ================================================================
# # STAGE 3 — PCA
# # ================================================================

# def stage3_pca(df, features):

#     print("\n" + "=" * 75)
#     print("STAGE 3 — PCA DIMENSIONALITY REDUCTION")
#     print("=" * 75)

#     if len(features) < 2:
#         raise ValueError(
#             "Fewer than two features entered PCA."
#         )

#     imputer = SimpleImputer(
#         strategy="median"
#     )

#     scaler = StandardScaler()

#     X = imputer.fit_transform(
#         df[features]
#     )

#     X = scaler.fit_transform(X)

#     pca = PCA()
#     pca.fit(X)

#     explained = (
#         pca.explained_variance_ratio_
#     )

#     pc1_var = explained[0]
#     pc2_var = explained[1]

#     cumulative = (
#         pc1_var + pc2_var
#     )

#     print(
#         f"PC1 variance explained : "
#         f"{pc1_var * 100:.2f}%"
#     )

#     print(
#         f"PC2 variance explained : "
#         f"{pc2_var * 100:.2f}%"
#     )

#     print(
#         f"PC1 + PC2 cumulative   : "
#         f"{cumulative * 100:.2f}%"
#     )

#     if (
#         cumulative
#         < MIN_PC12_CUMULATIVE_VARIANCE
#     ):

#         print(
#             "\nWARNING:"
#             "\nPC1 + PC2 explain less than "
#             f"{MIN_PC12_CUMULATIVE_VARIANCE*100:.0f}%."
#             "\nConsider including PC3."
#         )

#     # PCA components:
#     # rows = PCs
#     # columns = original variables

#     loading_pc1 = pca.components_[0]
#     loading_pc2 = pca.components_[1]

#     # Squared loadings within each PC
#     sq_pc1 = loading_pc1 ** 2
#     sq_pc2 = loading_pc2 ** 2

#     # Convert to proportional contribution
#     contribution_pc1 = (
#         sq_pc1 / sq_pc1.sum()
#     )

#     contribution_pc2 = (
#         sq_pc2 / sq_pc2.sum()
#     )

#     # Expected contribution if all features
#     # contributed equally
#     expected = 1 / len(features)

#     result = pd.DataFrame({
#         "feature": features,
#         "loading_PC1": loading_pc1,
#         "loading_PC2": loading_pc2,
#         "contribution_PC1": contribution_pc1,
#         "contribution_PC2": contribution_pc2,
#     })

#     result["expected_contribution"] = (
#         expected
#     )

#     result["PC1_influential"] = (
#         result["contribution_PC1"]
#         > expected
#     )

#     result["PC2_influential"] = (
#         result["contribution_PC2"]
#         > expected
#     )

#     result["PCA_pass"] = (
#         result["PC1_influential"]
#         |
#         result["PC2_influential"]
#     )

#     # Helpful combined measure for ranking,
#     # but NOT used as an additional threshold.

#     result[
#         "variance_weighted_PC12_score"
#     ] = (
#         pc1_var
#         * result["contribution_PC1"]
#         +
#         pc2_var
#         * result["contribution_PC2"]
#     )

#     result = result.sort_values(
#         "variance_weighted_PC12_score",
#         ascending=False,
#     )

#     print(
#         f"\nExpected average contribution "
#         f"= 1/{len(features)} "
#         f"= {expected*100:.2f}%"
#     )

#     print("\nPCA contribution table:")

#     display_cols = [
#         "feature",
#         "loading_PC1",
#         "loading_PC2",
#         "contribution_PC1",
#         "contribution_PC2",
#         "PC1_influential",
#         "PC2_influential",
#         "PCA_pass",
#     ]

#     print(
#         result[
#             display_cols
#         ].to_string(
#             index=False,
#             float_format=lambda x: f"{x:.4f}",
#         )
#     )

#     result.to_csv(
#         os.path.join(
#             OUT_DIR,
#             "stage3_pca_feature_contributions.csv",
#         ),
#         index=False,
#     )

#     final_features = result.loc[
#         result["PCA_pass"],
#         "feature",
#     ].tolist()

#     return (
#         final_features,
#         result,
#         pca,
#     )


# # ================================================================
# # STAGE 4 — PAIRWISE SPEARMAN REDUNDANCY PRUNING
# # ================================================================

# PAIRWISE_SPEARMAN_THRESHOLD = 0.90


# def stage4_pairwise_spearman_pruning(
#     df,
#     features,
#     stage2_report
# ):

#     print("\n" + "=" * 75)
#     print("STAGE 4 — PAIRWISE SPEARMAN REDUNDANCY PRUNING")
#     print("=" * 75)

#     print(
#         f"Redundancy threshold: "
#         f"|Spearman rho| >= {PAIRWISE_SPEARMAN_THRESHOLD:.2f}"
#     )

#     # ------------------------------------------------------------
#     # Predictor-to-predictor correlation matrix
#     # ------------------------------------------------------------

#     corr_matrix = (
#         df[features]
#         .corr(method="spearman")
#         .abs()
#     )

#     # ------------------------------------------------------------
#     # Bring in predictor-to-SOC relevance from Stage 2
#     # ------------------------------------------------------------

#     relevance = (
#         stage2_report
#         .set_index("feature")
#         [["abs_spearman", "mi_observed"]]
#         .copy()
#     )

#     # ------------------------------------------------------------
#     # Order features:
#     # strongest SOC association first,
#     # then MI as tie-breaker
#     # ------------------------------------------------------------

#     ordered_features = sorted(
#         features,
#         key=lambda f: (
#             relevance.loc[f, "abs_spearman"],
#             relevance.loc[f, "mi_observed"]
#         ),
#         reverse=True
#     )

#     retained = []
#     removed = []

#     # ------------------------------------------------------------
#     # Greedy redundancy pruning
#     # ------------------------------------------------------------

#     for feat in ordered_features:

#         redundant_with = None
#         redundancy_value = None

#         for kept in retained:

#             rho = corr_matrix.loc[feat, kept]

#             if rho >= PAIRWISE_SPEARMAN_THRESHOLD:

#                 redundant_with = kept
#                 redundancy_value = rho
#                 break

#         if redundant_with is None:

#             retained.append(feat)

#         else:

#             removed.append({
#                 "removed_feature": feat,
#                 "retained_feature": redundant_with,
#                 "pairwise_abs_spearman": redundancy_value,

#                 "removed_abs_SOC_spearman":
#                     relevance.loc[
#                         feat,
#                         "abs_spearman"
#                     ],

#                 "retained_abs_SOC_spearman":
#                     relevance.loc[
#                         redundant_with,
#                         "abs_spearman"
#                     ],

#                 "removed_MI":
#                     relevance.loc[
#                         feat,
#                         "mi_observed"
#                     ],

#                 "retained_MI":
#                     relevance.loc[
#                         redundant_with,
#                         "mi_observed"
#                     ],
#             })

#     # ------------------------------------------------------------
#     # Report
#     # ------------------------------------------------------------

#     print(f"\nInput PCA-selected features : {len(features)}")
#     print(f"Retained after pruning      : {len(retained)}")
#     print(f"Removed as redundant        : {len(features) - len(retained)}")

#     if removed:

#         removed_df = pd.DataFrame(removed)

#         print("\nRedundant predictors removed:")
#         print(
#             removed_df.to_string(
#                 index=False,
#                 float_format=lambda x: f"{x:.4f}"
#             )
#         )

#         removed_df.to_csv(
#             os.path.join(
#                 OUT_DIR,
#                 "stage4_pairwise_spearman_removed.csv"
#             ),
#             index=False
#         )

#     # Save correlation matrix too
#     corr_matrix.to_csv(
#         os.path.join(
#             OUT_DIR,
#             "stage4_pairwise_spearman_matrix.csv"
#         )
#     )

#     print("\nFinal retained features:")
#     for i, feat in enumerate(retained, 1):
#         print(f"{i:2d}. {feat}")

#     return retained, corr_matrix

# # ================================================================
# # MAIN
# # ================================================================

# def main():

#     print("\n" + "=" * 75)
#     print("DEFENSIBLE THREE-STAGE FEATURE SELECTION")
#     print("=" * 75)

#     df_cal = pd.read_csv(
#         CALIB_CSV
#     )

#     df_val = pd.read_csv(
#         VALID_CSV
#     )

#     # Original numerical image predictors
#     candidate_features = [
#         c
#         for c in df_cal.columns
#         if (
#             c not in EXCLUDE
#             and pd.api.types.is_numeric_dtype(
#                 df_cal[c]
#             )
#         )
#     ]

#     print(
#         f"\nOriginal candidate features: "
#         f"{len(candidate_features)}"
#     )

#     # ------------------------------------------------------------
#     # Aggregate calibration replicates
#     # ------------------------------------------------------------

#     df_cal_agg = aggregate_replicates(
#         df_cal,
#         candidate_features,
#     )

#     # ------------------------------------------------------------
#     # Stage 1
#     # ------------------------------------------------------------

#     stage1_features, stage1_report = (
#         stage1_quality_filter(
#             df_cal_agg,
#             candidate_features,
#         )
#     )

#     # ------------------------------------------------------------
#     # Stage 2
#     # ------------------------------------------------------------

#     stage2_features, stage2_report = (
#         stage2_soc_screening(
#             df_cal_agg,
#             stage1_features,
#         )
#     )

#     # ------------------------------------------------------------
#     # Stage 3
#     # ------------------------------------------------------------

#     pca_features, pca_report, pca_model = (
#         stage3_pca(
#             df_cal_agg,
#             stage2_features,
#         )
#     )



# # ------------------------------------------------------------
# # Stage 4 — pairwise Spearman redundancy pruning
# # ------------------------------------------------------------

#     final_features, pairwise_corr_matrix = (
#         stage4_pairwise_spearman_pruning(
#             df_cal_agg,
#            pca_features,
#             stage2_report
#         )
#     )

#     print("\n" + "=" * 75)
#     print("FINAL SELECTED FEATURES")
#     print("=" * 75)

#     for i, feat in enumerate(
#         final_features,
#         1,
#     ):
#         print(
#             f"{i:2d}. {feat}"
#         )

#     print(
#         f"\nOriginal features : "
#         f"{len(candidate_features)}"
#     )

#     print(
#         f"After Stage 1     : "
#         f"{len(stage1_features)}"
#     )

#     print(
#         f"After Stage 2     : "
#         f"{len(stage2_features)}"
#     )

#     print(
#         f"After PCA   : "
#         f"{len(pca_features)}"
#     )


#     print(
#         f"After Spearman pruning  : {len(final_features)}"
#     )
#     # ------------------------------------------------------------
#     # SAVE FEATURE LIST
#     # ------------------------------------------------------------

#     with open(
#         os.path.join(
#             OUT_DIR,
#             "selected_features_list.txt",
#         ),
#         "w",
#     ) as f:

#         for feat in final_features:
#             f.write(
#                 feat + "\n"
#             )

#     # ------------------------------------------------------------
#     # Save aggregated calibration modeling dataset
#     # ------------------------------------------------------------

#     cal_cols = (
#         [GROUP, TARGET]
#         + final_features
#     )

#     df_cal_agg[
#         cal_cols
#     ].to_csv(
#         os.path.join(
#             OUT_DIR,
#             "calibration_selected_features.csv",
#         ),
#         index=False,
#     )

#     # ------------------------------------------------------------
#     # Validation:
#     # aggregate independently,
#     # but DO NOT perform feature selection on it.
#     # ------------------------------------------------------------

#     df_val_agg = aggregate_replicates(
#         df_val,
#         candidate_features,
#     )

#     val_cols = (
#         [GROUP, TARGET]
#         + final_features
#     )

#     df_val_agg[
#         val_cols
#     ].to_csv(
#         os.path.join(
#             OUT_DIR,
#             "validation_selected_features.csv",
#         ),
#         index=False,
#     )

#     print(
#         "\nValidation feature selection: NONE."
#         "\nCalibration-derived feature list "
#         "applied unchanged."
#     )

#     print(
#         f"\nOutputs saved to:\n{OUT_DIR}"
#     )


# if __name__ == "__main__":
#     main()



"""
STEP 2 — DEFENSIBLE FEATURE SELECTION (FIXED)
==============================================
FIX: Removed aggregate_replicates() call.
     Feature selection now runs on ALL calibration IMAGES (~460 rows)
     not on 15-20 sample means.

All 4 stages unchanged — only Step 0 aggregation removed.
Validation data never used for feature selection.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = (
    r"C:\Users\dharamkar.1\OneDrive - The Ohio State University"
    r"\VSCode_Image_Processing_Reviewed"
)

CALIB_CSV = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step1_output", "calibration_set.csv")
VALID_CSV  = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step1_output", "validation_set.csv")
OUT_DIR    = os.path.join(PROJECT_ROOT, "notebooks", "objective_1",
    "output_data", "step2_output")

os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================
# SETTINGS
# ================================================================

TARGET = "soc"
GROUP  = "Sample_No"

EXCLUDE = [
    "filename", "Numeric numbers", "image_path", "image_no",
    "Image_No", "soil_type", "moisture", "soc", "Sample_No",
    "Mahalanobis_Distance", "Is_Outlier", "split",
]

MAX_MISSING             = 0.20
MIN_SCALED_VARIANCE     = 0.01
MIN_ABS_SPEARMAN        = 0.20
SCREENING_P             = 0.10
N_MI_PERMUTATIONS       = 500
MI_NULL_PERCENTILE      = 95
MIN_PC12_CUMVAR         = 0.60
PAIRWISE_THRESHOLD      = 0.90
SEED = 42


# ================================================================
# STAGE 1 — QUALITY CONTROL
# ================================================================

def stage1_quality_filter(df, features):
    print("\n" + "="*75)
    print("STAGE 1 — QUALITY CONTROL  (image-level calibration data)")
    print(f"  Input rows: {len(df)}  ← should be ~460, not 20")
    print("="*75)

    retained, records = [], []
    for feat in features:
        missing = df[feat].isna().mean()
        if missing > MAX_MISSING:
            records.append({"feature": feat, "stage1_pass": False,
                             "reason": "missingness > 20%"})
            continue
        x = df[[feat]].fillna(df[feat].median())
        variance = np.var(MinMaxScaler().fit_transform(x), ddof=1)
        if variance < MIN_SCALED_VARIANCE:
            records.append({"feature": feat, "scaled_variance": variance,
                             "stage1_pass": False,
                             "reason": "scaled variance < 0.01"})
            continue
        retained.append(feat)
        records.append({"feature": feat, "scaled_variance": variance,
                         "stage1_pass": True, "reason": "retained"})

    report = pd.DataFrame(records)
    print(f"  Input: {len(features)}  Retained: {len(retained)}  "
          f"Removed: {len(features)-len(retained)}")
    report.to_csv(os.path.join(OUT_DIR, "stage1_quality_report.csv"),
                   index=False)
    return retained, report


# ================================================================
# STAGE 2A — SPEARMAN CORRELATION
# ================================================================

def compute_spearman(df, features):
    results = []
    for feat in features:
        x, y = df[feat], df[TARGET]
        valid = x.notna() & y.notna()
        rho, p = spearmanr(x[valid], y[valid])
        results.append({"feature": feat, "spearman_rho": rho,
                         "abs_spearman": abs(rho), "spearman_p": p})
    return pd.DataFrame(results)


# ================================================================
# STAGE 2B — MUTUAL INFORMATION WITH PERMUTATION THRESHOLD
# ================================================================

def compute_permutation_mi(df, features):
    rng = np.random.default_rng(SEED)
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df[features])
    y   = df[TARGET].values
    n_nb = min(5, max(2, len(y)-1))
    observed = mutual_info_regression(X, y, random_state=SEED,
                                       n_neighbors=n_nb)
    results = []
    print(f"\n  Running {N_MI_PERMUTATIONS} MI permutations "
          f"on {len(y)} images...")
    for j, feat in enumerate(features):
        null_scores = [
            mutual_info_regression(X[:, [j]], rng.permutation(y),
                                    random_state=SEED+i,
                                    n_neighbors=n_nb)[0]
            for i in range(N_MI_PERMUTATIONS)
        ]
        threshold = np.percentile(null_scores, MI_NULL_PERCENTILE)
        results.append({"feature": feat, "mi_observed": observed[j],
                         "mi_null_95": threshold,
                         "mi_pass": observed[j] > threshold})
    return pd.DataFrame(results)


# ================================================================
# STAGE 2 — COMBINE SOC ASSOCIATION TESTS
# ================================================================

def stage2_soc_screening(df, features):
    print("\n" + "="*75)
    print("STAGE 2 — SOC RELEVANCE SCREENING  (image-level)")
    print("="*75)
    corr   = compute_spearman(df, features)
    mi     = compute_permutation_mi(df, features)
    result = corr.merge(mi, on="feature")
    result["corr_pass"]   = ((result["abs_spearman"] >= MIN_ABS_SPEARMAN) &
                              (result["spearman_p"] < SCREENING_P))
    result["stage2_pass"] = result["corr_pass"] | result["mi_pass"]
    result = result.sort_values(
        ["stage2_pass", "abs_spearman", "mi_observed"],
        ascending=[False, False, False])
    retained = result.loc[result["stage2_pass"], "feature"].tolist()
    print(f"  Entered: {len(features)}  Passed: {len(retained)}")
    print(result[["feature","spearman_rho","spearman_p","corr_pass",
                   "mi_observed","mi_null_95","mi_pass",
                   "stage2_pass"]].to_string(index=False))
    result.to_csv(os.path.join(OUT_DIR, "stage2_soc_screening.csv"),
                   index=False)
    return retained, result


# ================================================================
# STAGE 3 — PCA
# ================================================================

def stage3_pca(df, features):
    print("\n" + "="*75)
    print("STAGE 3 — PCA DIMENSIONALITY REDUCTION  (image-level)")
    print("="*75)
    if len(features) < 2:
        raise ValueError("Fewer than two features entered PCA.")
    imp    = SimpleImputer(strategy="median")
    X      = StandardScaler().fit_transform(imp.fit_transform(df[features]))
    pca    = PCA().fit(X)
    ev     = pca.explained_variance_ratio_
    pc1, pc2 = ev[0], ev[1]
    cumvar = pc1 + pc2
    print(f"  PC1={pc1*100:.2f}%  PC2={pc2*100:.2f}%  "
          f"Cumulative={cumvar*100:.2f}%")
    if cumvar < MIN_PC12_CUMVAR:
        print(f"  WARNING: PC1+PC2 < {MIN_PC12_CUMVAR*100:.0f}%")
    lp1, lp2 = pca.components_[0], pca.components_[1]
    c1 = lp1**2/np.sum(lp1**2)
    c2 = lp2**2/np.sum(lp2**2)
    expected = 1/len(features)
    result = pd.DataFrame({
        "feature": features,
        "loading_PC1": lp1, "loading_PC2": lp2,
        "contribution_PC1": c1, "contribution_PC2": c2,
        "expected_contribution": expected,
        "PC1_influential": c1 > expected,
        "PC2_influential": c2 > expected,
    })
    result["PCA_pass"] = result["PC1_influential"] | result["PC2_influential"]
    result["vw_score"] = pc1*c1 + pc2*c2
    result = result.sort_values("vw_score", ascending=False)
    result.to_csv(
        os.path.join(OUT_DIR, "stage3_pca_feature_contributions.csv"),
        index=False)
    final = result.loc[result["PCA_pass"], "feature"].tolist()
    print(f"  PCA retained: {len(final)} of {len(features)}")
    return final, result, pca


# ================================================================
# STAGE 4 — PAIRWISE SPEARMAN REDUNDANCY PRUNING
# ================================================================

def stage4_pairwise_spearman_pruning(df, features, stage2_report):
    print("\n" + "="*75)
    print(f"STAGE 4 — PAIRWISE SPEARMAN PRUNING "
          f"(threshold={PAIRWISE_THRESHOLD})")
    print("="*75)
    corr_matrix = df[features].corr(method="spearman").abs()
    relevance   = stage2_report.set_index("feature")[
        ["abs_spearman", "mi_observed"]]
    ordered = sorted(features,
                      key=lambda f: (relevance.loc[f, "abs_spearman"],
                                      relevance.loc[f, "mi_observed"]),
                      reverse=True)
    retained, removed = [], []
    for feat in ordered:
        dup = next((k for k in retained
                    if corr_matrix.loc[feat, k] >= PAIRWISE_THRESHOLD),
                   None)
        if dup is None:
            retained.append(feat)
        else:
            removed.append({"removed_feature": feat,
                             "retained_feature": dup,
                             "pairwise_abs_spearman":
                                 corr_matrix.loc[feat, dup]})
    if removed:
        pd.DataFrame(removed).to_csv(
            os.path.join(OUT_DIR,
                          "stage4_pairwise_spearman_removed.csv"),
            index=False)
    corr_matrix.to_csv(
        os.path.join(OUT_DIR, "stage4_pairwise_spearman_matrix.csv"))
    print(f"  Input: {len(features)}  Retained: {len(retained)}  "
          f"Removed: {len(features)-len(retained)}")
    print("\n  Final retained features:")
    for i, feat in enumerate(retained, 1):
        print(f"  {i:2d}. {feat}")
    return retained, corr_matrix


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*75)
    print("STEP 2 — DEFENSIBLE FEATURE SELECTION")
    print("  FIX: Running on IMAGE-LEVEL data (no aggregation)")
    print("="*75)

    df_cal = pd.read_csv(CALIB_CSV)
    df_val = pd.read_csv(VALID_CSV)

    print(f"\n  Calibration: {len(df_cal)} images x "
          f"{len(df_cal.columns)} cols")
    print(f"  Validation : {len(df_val)} images  "
          f"(not used for selection)")

    # Verify sample split
    if GROUP in df_cal.columns and GROUP in df_val.columns:
        cal_samps = set(df_cal[GROUP].dropna().unique())
        val_samps = set(df_val[GROUP].dropna().unique())
        overlap   = cal_samps & val_samps
        print(f"  Cal samples: {sorted(cal_samps)}")
        print(f"  Val samples: {sorted(val_samps)}")
        print(f"  Overlap: {'ZERO ✅' if not overlap else f'WARNING {overlap}'}")

    candidate_features = [
        c for c in df_cal.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(df_cal[c])
    ]
    print(f"\n  Candidate features: {len(candidate_features)}")

    # ── THE FIX: pass df_cal directly — NO aggregation ──────────────────
    # Original code did: df_cal_agg = aggregate_replicates(df_cal, ...)
    # That collapsed 460 image rows to 15-20 sample means.
    # We now use df_cal directly with all image rows.
    # ────────────────────────────────────────────────────────────────────

    # Stage 1
    stage1_features, stage1_report = stage1_quality_filter(
        df_cal, candidate_features)

    # Stage 2
    stage2_features, stage2_report = stage2_soc_screening(
        df_cal, stage1_features)

    # Stage 3
    pca_features, pca_report, pca_model = stage3_pca(
        df_cal, stage2_features)

    # Stage 4
    final_features, corr_matrix = stage4_pairwise_spearman_pruning(
        df_cal, pca_features, stage2_report)

    print("\n" + "="*75)
    print("FINAL SELECTED FEATURES")
    print("="*75)
    for i, f in enumerate(final_features, 1):
        print(f"  {i:2d}. {f}")
    print(f"\n  Original: {len(candidate_features)}"
          f" → Stage1: {len(stage1_features)}"
          f" → Stage2: {len(stage2_features)}"
          f" → PCA: {len(pca_features)}"
          f" → Final: {len(final_features)}")

    # Save feature list
    with open(os.path.join(OUT_DIR, "selected_features_list.txt"),
              "w") as f:
        for feat in final_features:
            f.write(feat + "\n")

    # Save calibration with selected features — IMAGE LEVEL
    cal_keep = [c for c in [GROUP, "image_no", "moisture", "soil_type",
                              TARGET] if c in df_cal.columns]
    cal_keep += [f for f in final_features if f in df_cal.columns]
    df_cal[cal_keep].to_csv(
        os.path.join(OUT_DIR, "calibration_selected_features.csv"),
        index=False)
    print(f"\n  Saved: calibration_selected_features.csv  "
          f"({len(df_cal)} rows ← IMAGE LEVEL ✅)")

    # Save validation — IMAGE LEVEL, calibration features applied
    val_keep = [c for c in cal_keep if c in df_val.columns]
    df_val[val_keep].to_csv(
        os.path.join(OUT_DIR, "validation_selected_features.csv"),
        index=False)
    print(f"  Saved: validation_selected_features.csv  "
          f"({len(df_val)} rows ← IMAGE LEVEL ✅)")
    print(f"\n  Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()