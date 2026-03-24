
# 🎯 Step 2.3: Visualization Script (Deep Analysis)

## Goal
Turn your CSV tables into the "Figure: Performance by Palm Characteristics" required for verification.

## Input
- `standardized_results/analysis_by_density.csv`
- `standardized_results/analysis_by_size.csv`

## Output
- `standardized_results/Figure_Performance_by_Characteristics.png`
- `standardized_results/Figure_Ablation_Impact.png`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_DENSITY = 'standardized_results/analysis_by_density.csv'
INPUT_SIZE = 'standardized_results/analysis_by_size.csv'
INPUT_ABLATION = 'standardized_results/ablation_summary.csv'
OUTPUT_DIR = 'standardized_results'

# Set IEEE Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def run_visualization():
    print("🎯 STARTING DEEP ANALYSIS VISUALIZATION...")
    
    if not os.path.exists(INPUT_DENSITY) or not os.path.exists(INPUT_SIZE):
        print("❌ Error: Input CSVs not found. Run deep_analysis_script first!")
        return

    # Load Data
    df_density = pd.read_csv(INPUT_DENSITY)
    df_size = pd.read_csv(INPUT_SIZE)
    
    # ----------------------------------------------------
    # FIGURE 1: Performance by Characteristics (2 Subplots)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Subplot A: Density (Scene Complexity)
    sns.barplot(
        data=df_density, x='Density', y='F1_Mean', hue='Model',
        ax=axes[0], palette='viridis', edgecolor='black'
    )
    axes[0].set_title('(a) Performance vs. Palm Density', fontweight='bold', pad=15)
    axes[0].set_xlabel('Palm Density Group', fontweight='bold')
    axes[0].set_ylabel('F1-Score', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower left', frameon=True)

    # Subplot B: Canopy Size (Maturity)
    sns.barplot(
        data=df_size, x='Canopy_Size', y='F1_Mean', hue='Model',
        ax=axes[1], palette='viridis', edgecolor='black'
    )
    axes[1].set_title('(b) Performance vs. Canopy Size', fontweight='bold', pad=15)
    axes[1].set_xlabel('Canopy Diameter Group', fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].get_legend().remove() # Share legend from first plot if models relate
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/Figure_Performance_by_Characteristics.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved Density/Size Figure: {save_path}")
    
    # ----------------------------------------------------
    # FIGURE 2: Ablation Study (If exists)
    # ----------------------------------------------------
    if os.path.exists(INPUT_ABLATION):
        df_abl = pd.read_csv(INPUT_ABLATION)
        # Normalize Model Names for cleaner plot if needed
        
        plt.figure(figsize=(8, 5), dpi=300)
        # Melt for Side-by-Side metrics
        id_vars = ['Model']
        value_vars = ['F1', 'IoU']
        df_melt = df_abl.melt(id_vars=id_vars, value_vars=value_vars, var_name='Metric', value_name='Score')
        
        sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette='magma', edgecolor='black')
        
        plt.title('Ablation Study: Impact of C2PSA Mechanism', fontweight='bold', pad=15)
        plt.ylim(0, 1.05)
        plt.ylabel('Score')
        plt.xlabel('')
        plt.legend(frameon=True)
        
        abl_save_path = f"{OUTPUT_DIR}/Figure_Ablation_Impact.png"
        plt.savefig(abl_save_path, bbox_inches='tight')
        print(f"✅ Saved Ablation Figure: {abl_save_path}")

if __name__ == "__main__":
    run_visualization()
```
