
# 🎯 Step 1.3: Statistical Significance Testing

## Goal
Perform pairwise statistical validation to prove your model is significantly better, not just "lucky".

## Input
- `standardized_results/detailed_results_per_image.csv` (Output from Step 1.2)

## Output
- `standardized_results/statistical_tests_results.csv`: Table of p-values for all metric comparisons.

```python
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_CSV = 'standardized_results/detailed_results_per_image.csv'
OUTPUT_CSV = 'standardized_results/statistical_tests_results.csv'

def run_statistical_analysis():
    print("🎯 STARTING STATISTICAL SIGNIFICANCE TESTING (Step 1.3)...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Run Step 1.2 first!")
        return

    df = pd.read_csv(INPUT_CSV)
    models = df['Model'].unique()
    print(f"   👉 Models found: {models}")
    
    # Define Baseline for comparisons (Usually the previous SOTA or simpler model)
    # We will compare everything against 'YOLOv8l-seg' and 'Mask R-CNN' if present.
    # Or pairwise all-vs-all. Let's do Pairwise against your Best Model.
    
    # Assuming YOLOv11l-seg is your proposed best for Speed/Acc balance, 
    # and Hybrid is best for sizing. 
    # Let's run all pairwise combinations for completeness.
    
    comparisons = []
    metrics = ['IoU', 'F1_Score', 'Error_Area_m2', 'Error_Diam_m'] # We need to calc F1 per image? 
    # Note: F1 is set-based usually. For per-image stat tests, we usually use IoU 
    # or "Precision/Recall" if calculated per image.
    # detailed_comparison_results has TP/FP/FN. We can calc F1 per image.
    
    # Calculate Per-Image F1
    df['Precision_Img'] = df['TP'] / (df['TP'] + df['FP'] + 1e-6)
    df['Recall_Img'] = df['TP'] / (df['TP'] + df['FN'] + 1e-6)
    df['F1_Img'] = 2 * (df['Precision_Img'] * df['Recall_Img']) / (df['Precision_Img'] + df['Recall_Img'] + 1e-6)
    
    # Absolute Errors for stat testing (we want 0 error)
    df['Abs_Error_Area'] = df['Error_Area_m2'].abs()
    df['Abs_Error_Diam'] = df['Error_Diam_m'].abs()
    
    test_metrics = ['IoU', 'F1_Img', 'Abs_Error_Area', 'Abs_Error_Diam']
    
    # Get pairs
    model_pairs = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model_pairs.append((models[i], models[j]))
            
    stats_rows = []
    
    for m1, m2 in model_pairs:
        # Get common images (paired t-test requires same samples)
        df1 = df[df['Model'] == m1].set_index('filename')
        df2 = df[df['Model'] == m2].set_index('filename')
        
        # Align indices
        common_indices = df1.index.intersection(df2.index)
        
        if len(common_indices) < 2:
            print(f"   ⚠️ Skipping {m1} vs {m2} (insufficient overlap)")
            continue
            
        data1 = df1.loc[common_indices]
        data2 = df2.loc[common_indices]
        
        for metric in test_metrics:
            val1 = data1[metric].values
            val2 = data2[metric].values
            
            # Wilcoxon Signed-Rank Test (Non-parametric paired)
            # Use 'pratt' method to handle zeros/ties cleanly if needed, though 'wilcoxon' default works.
            # We test hypothesis: distributions are different.
            try:
                stat, p_val = wilcoxon(val1, val2)
            except ValueError:
                p_val = 1.0 # Exact match
            
            # Determine winner (Mean comparison)
            mean1 = np.mean(val1)
            mean2 = np.mean(val2)
            
            # For Error metrics, Lower is Better. For Score metrics, Higher is Better.
            is_error_metric = 'Error' in metric
            
            if is_error_metric:
                winner = m1 if mean1 < mean2 else m2
            else:
                winner = m1 if mean1 > mean2 else m2
            
            if p_val > 0.05:
                significance = "NS" # Not Significant
                winner = "Tie"
            elif p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            else:
                significance = "*"
                
            stats_rows.append({
                'Model_A': m1,
                'Model_B': m2,
                'Metric': metric,
                'p_value': round(p_val, 5),
                'Significance': significance,
                'Winner': winner,
                'Mean_A': round(mean1, 4),
                'Mean_B': round(mean2, 4)
            })

    results_df = pd.DataFrame(stats_rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ STATISTICAL RESULTS SAVED: {OUTPUT_CSV}")
    print(results_df[['Model_A', 'Model_B', 'Metric', 'p_value', 'Winner']].to_string())

if __name__ == "__main__":
    run_statistical_analysis()
```
