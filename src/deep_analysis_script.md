
# 🎯 Step 2.2 & 2.3: Deep Analysis Suite (Verified Fix)

## Goal
Understand **WHY** models fail. We go beyond simple mAP to categorize performance by:
1.  **Scene Density**: Does the model choke on crowded plantations?
2.  **Canopy Size**: Is the model bad at detecting young/small palms?
3.  **Ablation**: Did removing C2PSA actually hurt performance?

## Input
- `standardized_results/detailed_results_per_image.csv`

## Output
- `standardized_results/analysis_by_density.csv`
- `standardized_results/analysis_by_size.csv`
- `standardized_results/ablation_summary.csv`

```python
import pandas as pd
import numpy as np
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_CSV = 'standardized_results/detailed_results_per_image.csv'
OUTPUT_DIR = 'standardized_results'

def run_deep_analysis():
    print("🎯 STARTING DEEP ANALYSIS (Ablation & Error Modes)...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Run Step 1.2 first!")
        return
        
    df = pd.read_csv(INPUT_CSV)
    
    # ----------------------------------------------------
    # 1. PRE-PROCESSING
    # ----------------------------------------------------
    # Infer Ground Truth Count from TP + FN
    df['GT_Count'] = df['TP'] + df['FN']
    
    # Calculate Per-Row Metrics (Handle divide by zero)
    df['Precision'] = df['TP'] / (df['TP'] + df['FP'] + 1e-6)
    df['Recall'] = df['TP'] / (df['TP'] + df['FN'] + 1e-6)
    df['F1'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'] + 1e-6)
    
    # ----------------------------------------------------
    # 2. ANALYSIS BY DENSITY (Scene Complexity)
    # ----------------------------------------------------
    print("   📊 Analyzing by Density (Sparse vs Dense)...")
    # Define Categories
    # Sparse: < 10 palms
    # Medium: 10 - 30 palms
    # Dense: > 30 palms
    bins = [-1, 10, 30, 999]
    labels = ['Sparse (<10)', 'Medium (10-30)', 'Dense (>30)']
    df['Density_Group'] = pd.cut(df['GT_Count'], bins=bins, labels=labels)
    
    density_stats = df.groupby(['Model', 'Density_Group']).agg({
        'F1': ['mean', 'std'],
        'IoU': 'mean',
        'Error_Area_m2': lambda x: np.sqrt(np.mean(x**2)), # RMSE
        'filename': 'count'
    }).reset_index()
    
    # Flatten Columns
    density_stats.columns = ['Model', 'Density', 'F1_Mean', 'F1_Std', 'IoU_Mean', 'Area_RMSE', 'N_Images']
    density_path = f"{OUTPUT_DIR}/analysis_by_density.csv"
    density_stats.to_csv(density_path, index=False)
    
    # ----------------------------------------------------
    # 3. ANALYSIS BY SIZE (Canopy Maturity)
    # ----------------------------------------------------
    print("   📊 Analyzing by Canopy Size (Young vs Mature)...")
    # Use GT_Diam_m for categorization
    # Small (Young): < 5m
    # Medium: 5m - 8m
    # Large (Mature): > 8m
    bins_size = [-1, 5, 8, 999]
    labels_size = ['Small (<5m)', 'Medium (5-8m)', 'Large (>8m)']
    df['Size_Group'] = pd.cut(df['GT_Diam_m'], bins=bins_size, labels=labels_size)
    
    size_stats = df.groupby(['Model', 'Size_Group']).agg({
        'F1': 'mean',
        'IoU': 'mean',
        'Error_Diam_m': lambda x: np.sqrt(np.mean(x**2)), # RMSE
        'filename': 'count'
    }).reset_index()
    
    size_stats.columns = ['Model', 'Canopy_Size', 'F1_Mean', 'IoU_Mean', 'Diam_RMSE', 'N_Images']
    size_path = f"{OUTPUT_DIR}/analysis_by_size.csv"
    size_stats.to_csv(size_path, index=False)
    
    # ----------------------------------------------------
    # 4. ABLATION STUDY COMPARISON
    # ----------------------------------------------------
    # Check if we have YOLOv11 and YOLOv11-Ablation
    models = df['Model'].unique()
    v11_full = [m for m in models if '11' in m and 'ablation' not in m.lower()]
    v11_abl = [m for m in models if 'ablation' in m.lower()]
    
    if v11_full and v11_abl:
        print("   🔬 Generating Ablation Report...")
        m_full = v11_full[0]
        m_abl = v11_abl[0]
        
        # Build Aggregation Dictionary Dynamically to avoid KeyErrors
        agg_dict = {
            'F1': 'mean',
            'IoU': 'mean',
            'Error_Area_m2': lambda x: np.sqrt(np.mean(x**2))
        }
        # Only add mAP50 if it exists
        if 'mAP50' in df.columns:
            agg_dict['mAP50'] = 'max'

        # Filter just these two
        df_abl = df[df['Model'].isin([m_full, m_abl])].groupby('Model').agg(agg_dict).reset_index()
        
        abl_path = f"{OUTPUT_DIR}/ablation_summary.csv"
        df_abl.to_csv(abl_path, index=False)
        print(f"      ✅ Saved: {abl_path}")
    else:
        print("      ⚠️ Skipping Ablation (Models not found in CSV)")

    print(f"\n✅ DENSITY ANALYSIS: {density_path}")
    print(f"✅ SIZE ANALYSIS: {size_path}")
    
    # Preview
    print("\n--- Density Analysis Preview ---")
    print(density_stats.head(6).to_string())

if __name__ == "__main__":
    run_deep_analysis()
```
