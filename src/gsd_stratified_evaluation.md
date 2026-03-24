
# 🎯 Step 1.2: Unified GSD Analysis Script (Final)

## Output Files
1.  **`detailed_results_per_image.csv`**: Contains per-image metrics (TP, FP, FN, IoU, Area, Diameter, Errors) for every single test image. Matches your `detailed_comparison_results.csv` format and logic.
2.  **`statistical_summary_by_gsd.csv`**: Aggregates performance (mAP, F1, RMSE) by Model and GSD.

## Methodology
-   **GT Area**: Calculated via **Shoelace Formula** on Polygon labels (Precise).
-   **Pred Area**: Pixel Area * GSD².
-   **Diameter**: Equivalent Diameter = `2 * sqrt(Area / pi)`.
-   **TP Validation**: Only counted if **IoU > 0.5** (Standard COCO).

```python
import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
import cv2

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
RESULTS_JSON = 'standardized_results/all_models_predictions.json'
TEST_DIR = '/content/final_dataset/test_benchmark'
OUTPUT_DIR = 'standardized_results'

# ==========================================
# 1. HELPERS
# ==========================================
def extract_gsd(filename):
    """
    Extracts GSD from filename like '0.03m_sample.png' or '0.20m_...'.
    """
    match = re.search(r"(\d+[\.\-]\d+)m", filename)
    if match:
        val_str = match.group(1).replace('-', '.')
        try:
            return float(val_str), f"{val_str}m"
        except: pass
    return None, None

def shoelace_area(coords):
    x = np.array(coords[0::2])
    y = np.array(coords[1::2])
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_equivalent_diameter(area):
    if area <= 0: return 0.0
    return 2 * np.sqrt(area / np.pi)

def create_ground_truth_coco(test_dir):
    images = []; annotations = []; ann_id = 0
    img_files = sorted(list(Path(test_dir).rglob("*.png")))
    
    for idx, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        images.append({"id": idx, "file_name": img_path.name, "height": h, "width": w})
        
        lbl_path = img_path.parent / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, 'r') as f: lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                poly_flat = parts[1:]
                
                # Denormalize
                poly_np = np.array(poly_flat).reshape(-1, 2)
                poly_np[:, 0] *= w; poly_np[:, 1] *= h
                poly_denorm = poly_np.flatten().tolist()
                
                # Shoelace Area
                area = shoelace_area(poly_denorm)
                
                # Bbox
                x_min, y_min = np.min(poly_np[:, 0]), np.min(poly_np[:, 1])
                x_max, y_max = np.max(poly_np[:, 0]), np.max(poly_np[:, 1])
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                annotations.append({
                    "id": ann_id, "image_id": idx, "category_id": 1, 
                    "segmentation": [poly_denorm], "area": area, "bbox": bbox, "iscrowd": 0
                })
                ann_id += 1
                
    categories = [{"id": 1, "name": "oil_palm", "supercategory": "plant"}]
    coco_dict = {
        "info": {"description": "Oil Palm Test Set"},
        "licenses": [{"id": 1, "name": "Proprietary"}],
        "images": images, "annotations": annotations, "categories": categories
    }
    return coco_dict

# ==========================================
# 2. ANALYSIS ENGINE
# ==========================================
def run_unified_analysis():
    print("🎯 STARTING UNIFIED GSD ANALYSIS...")
    if not os.path.exists(RESULTS_JSON):
        print(f"❌ Error: {RESULTS_JSON} not found.")
        return
        
    with open(RESULTS_JSON, 'r') as f: all_preds = json.load(f)
    print("   👉 Generating GT COCO (Shoelace)...")
    coco_gt_dict = create_ground_truth_coco(TEST_DIR)
    
    # Silent COCO init
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    sys.stdout = old_stdout
    
    # Auto-Detect GSDs
    gsd_groups = {}
    total_imgs = 0
    for img in coco_gt_dict['images']:
        _, gsd_str = extract_gsd(img['file_name'])
        if gsd_str:
            if gsd_str not in gsd_groups: gsd_groups[gsd_str] = []
            gsd_groups[gsd_str].append(img['id'])
            total_imgs += 1
    
    print(f"   👉 Found GSDs: {list(gsd_groups.keys())} ({total_imgs} images)")
    models = sorted(list(set([p['model'] for p in all_preds])))
    
    detailed_rows = [] # Per Image
    summary_rows = []  # Per Method/GSD Group

    for gsd_name, img_ids in gsd_groups.items():
        gsd_val = float(gsd_name.replace('m',''))
        print(f"   📊 Analyzing {gsd_name}...")
        
        for model in models:
            model_preds = [p for p in all_preds if p['model'] == model and p['image_id'] in img_ids]
            
            # --- 1. GLOBAL METRICS (mAP) ---
            map50 = 0; map50_95 = 0
            if model_preds:
                sys.stdout = io.StringIO()
                try:
                    coco_dt = coco_gt.loadRes(model_preds)
                    cocoEval = COCOeval(coco_gt, coco_dt, 'segm')
                    cocoEval.params.imgIds = img_ids
                    cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
                    map50_95 = cocoEval.stats[0]
                    map50 = cocoEval.stats[1]
                except: pass
                sys.stdout = old_stdout
            
            # --- 2. PER-IMAGE METRICS (Detailed & Deployment Stats) ---
            tp_total = 0; fp_total = 0; fn_total = 0
            
            # Physical Error Accumulators for RMSE
            sq_diff_area = [] 
            
            for iid in img_ids:
                img_info = coco_gt.loadImgs(iid)[0]
                fname = img_info['file_name']
                
                # GT processing
                gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid))
                # Shoelace Area is already in 'area' field from create_ground_truth
                gt_areas_m2 = [g['area'] * (gsd_val**2) for g in gt_anns]
                gt_total_area = sum(gt_areas_m2)
                
                gt_rles = [coco_gt.annToRLE(g) for g in gt_anns]
                gt_matched = [False] * len(gt_anns)
                
                # Pred processing
                # Filter by confidence > 0.45 just for "Counting/Deployment" metrics
                curr_preds = sorted(
                    [p for p in model_preds if p['image_id'] == iid and p['score'] > 0.45],
                    key=lambda x: x['score'], reverse=True
                )
                
                pred_areas_m2 = []
                img_tp = 0; img_fp = 0
                iou_sum = 0
                
                for p in curr_preds:
                    rle_p = p['segmentation']
                    if isinstance(rle_p['counts'], str): rle_p['counts'] = rle_p['counts'].encode('utf-8')
                    p_area_px = mask_utils.area(rle_p)
                    p_area_m2 = p_area_px * (gsd_val**2)
                    pred_areas_m2.append(p_area_m2)
                    
                    # IoU Matching
                    best_iou = 0; best_gt = -1
                    for g_idx, rle_g in enumerate(gt_rles):
                        iou = mask_utils.iou([rle_p], [rle_g], [False])[0][0]
                        if iou > best_iou: best_iou = iou; best_gt = g_idx
                    
                    if best_iou > 0.5 and best_gt != -1 and not gt_matched[best_gt]:
                        img_tp += 1
                        gt_matched[best_gt] = True
                        iou_sum += best_iou
                    else:
                        img_fp += 1
                        
                img_fn = len(gt_anns) - img_tp
                pred_total_area = sum(pred_areas_m2)
                
                # Physical Metrics
                diff_area = pred_total_area - gt_total_area
                sq_diff_area.append(diff_area**2)
                
                gt_diam = get_equivalent_diameter(gt_total_area)
                pred_diam = get_equivalent_diameter(pred_total_area)
                diff_diam = pred_diam - gt_diam
                
                avg_iou = (iou_sum / img_tp) if img_tp > 0 else 0
                
                # Append DETAILED Row
                detailed_rows.append({
                    'filename': fname,
                    'gsd': gsd_name,
                    'Model': model,
                    'TP': img_tp,
                    'FP': img_fp,
                    'FN': img_fn,
                    'IoU': round(avg_iou, 4),
                    'Pred_Area_m2': round(pred_total_area, 3),
                    'GT_Area_m2': round(gt_total_area, 3),
                    'Error_Area_m2': round(diff_area, 3),
                    'Pred_Diam_m': round(pred_diam, 3),
                    'GT_Diam_m': round(gt_diam, 3),
                    'Error_Diam_m': round(diff_diam, 3)
                })
                
                # Accumulate for SUMMARY
                tp_total += img_tp
                fp_total += img_fp
                fn_total += img_fn

            # Calculate Aggregate Stats
            precision = tp_total / (tp_total + fp_total + 1e-6)
            recall = tp_total / (tp_total + fn_total + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            rmse_area = np.sqrt(np.mean(sq_diff_area)) if sq_diff_area else 0
            
            summary_rows.append({
                'Model': model,
                'GSD': gsd_name,
                'mAP50': round(map50, 3),
                'mAP50-95': round(map50_95, 3),
                'F1_Score': round(f1, 3),
                'Precision': round(precision, 3),
                'Recall': round(recall, 3),
                'Area_RMSE_m2': round(rmse_area, 2),
                'TP_Total': tp_total,
                'FP_Total': fp_total,
                'FN_Total': fn_total
            })

    # SAVE FILES
    df_det = pd.DataFrame(detailed_rows)
    df_sum = pd.DataFrame(summary_rows)
    
    # Sorting
    df_det = df_det.sort_values(['gsd', 'filename'])
    df_sum = df_sum.sort_values(['GSD', 'F1_Score'], ascending=[True, False])
    
    path_det = f"{OUTPUT_DIR}/detailed_results_per_image.csv"
    path_sum = f"{OUTPUT_DIR}/statistical_summary_by_gsd.csv"
    
    df_det.to_csv(path_det, index=False)
    df_sum.to_csv(path_sum, index=False)
    
    print(f"\n✅ DETAILED REPORT: {path_det}")
    print(f"✅ SUMMARY REPORT: {path_sum}")
    print("\nSummary Preview:")
    print(df_sum[['Model', 'GSD', 'mAP50', 'F1_Score', 'Area_RMSE_m2']].to_string())

if __name__ == "__main__":
    run_unified_analysis()
```
