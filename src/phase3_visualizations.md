
# 🎯 Phase 3: Critical Visualizations (Step 3.1 & 3.2) - Fixed

## Goal
Generate high-quality IEEE figures that prove your hypotheses visually.
1.  **GSD Sensitivity**: Visual proof of "Lower GSD = Better Accuracy".
2.  **PR Curves**: Standard model comparison metric.

## Inputs
- `standardized_results/statistical_summary_by_gsd.csv`
- `standardized_results/all_models_predictions.json`
- `standardized_results/detailed_results_per_image.csv` (used for error bars if needed)

## Outputs
- `standardized_results/Figure_GSD_Sensitivity.png`
- `standardized_results/Figure_PR_Curves.png`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_SUMMARY = 'standardized_results/statistical_summary_by_gsd.csv'
INPUT_PREDS = 'standardized_results/all_models_predictions.json'
TEST_DIR = '/content/final_dataset/test_benchmark'
OUTPUT_DIR = 'standardized_results'

# Style Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

def create_ground_truth_simple(test_dir):
    from pathlib import Path
    import cv2
    
    images = []; annotations = []; ann_id = 0
    img_files = sorted(list(Path(test_dir).rglob("*.png")))
    for idx, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w, _ = img.shape
        images.append({"id": idx, "file_name": img_path.name, "height": h, "width": w})
        
        lbl_path = img_path.parent / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, 'r') as f: lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                poly_flat = parts[1:]
                poly_np = np.array(poly_flat).reshape(-1, 2)
                poly_np[:, 0] *= w; poly_np[:, 1] *= h
                poly_denorm = poly_np.flatten().tolist()
                x_min, y_min = np.min(poly_np[:, 0]), np.min(poly_np[:, 1])
                x_max, y_max = np.max(poly_np[:, 0]), np.max(poly_np[:, 1])
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                annotations.append({
                    "id": ann_id, "image_id": idx, "category_id": 1, 
                    "segmentation": [poly_denorm], "area": (x_max-x_min)*(y_max-y_min), "bbox": bbox, "iscrowd": 0
                })
                ann_id += 1
    
    # Must include info and licenses to prevent COCO init crash
    return {
        "info": {"description": "Oil Palm Test Set"},
        "licenses": [{"id": 1, "name": "Proprietary"}],
        "images": images, 
        "annotations": annotations, 
        "categories": [{"id": 1, "name": "palm"}]
    }

def plot_gsd_sensitivity():
    print("   📊 Generating GSD Sensitivity Curves...")
    if not os.path.exists(INPUT_SUMMARY):
        print(f"      ❌ Missing {INPUT_SUMMARY}")
        return

    df = pd.read_csv(INPUT_SUMMARY)
    # Extract numeric GSD
    df['GSD_Num'] = df['GSD'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    df = df.sort_values('GSD_Num')
    
    metrics = ['F1_Score', 'mAP50', 'Area_RMSE_m2']
    titles = ['(a) F1-Score vs GSD', '(b) mAP@50 vs GSD', '(c) Area RMSE vs GSD']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            sns.lineplot(data=df, x='GSD_Num', y=metric, hue='Model', marker='o', 
                         ax=axes[i], linewidth=2.5, palette='deep')
            axes[i].set_title(titles[i], fontweight='bold')
            axes[i].set_xlabel('Ground Sample Distance (m)')
            axes[i].set_ylabel(metric.replace('_', ' '))
            if i > 0: axes[i].legend_.remove()
        
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/Figure_GSD_Sensitivity.png"
    plt.savefig(save_path)
    print(f"      ✅ Saved: {save_path}")

def plot_pr_curves():
    print("   📊 Generating PR Curves...")
    if not os.path.exists(INPUT_PREDS): 
        print(f"      ❌ Missing {INPUT_PREDS}")
        return

    coco_gt_dict = create_ground_truth_simple(TEST_DIR)
    
    # Silent COCO
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
    
    with open(INPUT_PREDS, 'r') as f:
        all_preds = json.load(f)
        
    models = sorted(list(set([p['model'] for p in all_preds])))
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    for model in models:
        preds = [p for p in all_preds if p['model'] == model]
        if not preds: continue
        
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(preds)
            cocoEval = COCOeval(coco_gt, coco_dt, 'segm')
            
            # Key fix: ensure imgIds match prediction set to avoid eval mismatch errors
            pred_img_ids = sorted(list(set([p['image_id'] for p in preds])))
            cocoEval.params.imgIds = pred_img_ids
            
            cocoEval.evaluate()
            cocoEval.accumulate()
            
            # Precision: [T, R, K, A, M]
            # T=0 (IoU=0.50), K=0 (cat=palm), A=0 (all), M=2 (maxDet=100)
            # Check cocoEval.params.maxDets -> [1, 10, 100] -> Index 2 is 100
            
            precision = cocoEval.eval['precision'][0, :, 0, 0, 2]
            recall = np.linspace(0, 1, 101)
            
            valid = precision > -1
            p_val = precision[valid]
            r_val = recall[valid]
            
            ap50 = np.mean(p_val) if len(p_val) > 0 else 0
            plt.plot(r_val, p_val, label=f"{model} (AP50={ap50:.2f})", linewidth=2)
            
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curves (IoU=0.50)', fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    save_path = f"{OUTPUT_DIR}/Figure_PR_Curves.png"
    plt.savefig(save_path)
    print(f"      ✅ Saved: {save_path}")

def main():
    try:
        plot_gsd_sensitivity()
    except Exception as e:
        print(f"      ⚠️ GSD Plot Error: {e}")
        
    try:
        plot_pr_curves()
    except Exception as e:
        print(f"      ⚠️ PR Curve Error: {e}")

if __name__ == "__main__":
    main()
```
