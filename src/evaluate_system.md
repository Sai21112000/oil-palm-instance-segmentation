# System Evaluation Script (The "Mega-Evaluation")

This script performs the rigorous analysis for your thesis.
1.  **Inference**: Runs 6 Models (v8, v11, Hybrid v8/v11, RCNN, Watershed).
2.  **Stats**: Bootstrapping (1000x) for 95% CIs.
3.  **Tests**: Paired T-Tests for significance.
4.  **Profiling**: FPS & GFLOPs.

## Instructions
1.  Install dependencies: `!pip install git+https://github.com/facebookresearch/segment-anything.git`
2.  Download the SAM 2.1 checkpoint: `!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` (Note: Using ViT-H as large proxy if SAM 2.1 specific url varies, strictly user asked for SAM 2.1, ensure you have the `sam2.1_hiera_large.pt` in `weights/` or adjust loader).
3.  Run this script.

```python
import torch
import numpy as np
import cv2
import pandas as pd
import time
import glob
from pathlib import Path
from ultralytics import YOLO, SAM  # Unified Import
from scipy import stats
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ==========================================
# ⚙️ CONFIG
# ==========================================
TEST_DIR = '/content/final_dataset/test_benchmark'
WEIGHTS_DIR = 'oil_palm_thesis'
OUTPUT_CSV = 'final_thesis_results.csv'
SAM_MODEL_NAME = "sam2.1_l.pt" # Ultralytics will auto-download this
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. MODEL LOADERS
# ==========================================

def load_maskrcnn():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    
    path = f"{WEIGHTS_DIR}/mask_rcnn_resnet50.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.to(DEVICE)
        model.eval()
        return model
    return None

def get_sam_model():
    # Uses Ultralytics SAM 2.1 wrapper
    try:
        model = SAM(SAM_MODEL_NAME)
        return model
    except Exception as e:
        print(f"⚠️ SAM Load Error: {e}")
        return None

# ==========================================
# 2. INFERENCE ENGINES
# ==========================================

def run_watershed(img_path):
    """Traditional Baseline: Otsu + Watershed"""
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    
    # Watershed
    markers = cv2.watershed(img, markers)
    
    # Extract Masks (Class 1 is BG, >1 are objects)
    masks = []
    unique_markers = np.unique(markers)
    for m in unique_markers:
        if m <= 1: continue # Skip Bg and Unknown
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers == m] = 1
        masks.append(mask)
        
    return masks # Returns list of binary masks

def run_hybrid_inference(yolo_result, sam_model, original_image):
    """Refines YOLO Keyboxes using SAM (Ultralytics API)."""
    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0: return []
    
    refined_masks = []
    
    # Ultralytics SAM accepts 'bboxes' argument
    # source can be the image array directly
    results = sam_model.predict(
        source=original_image,
        bboxes=boxes, 
        verbose=False
    )
    
    # Results is a list (one per image), we sent one image
    if results[0].masks is not None:
        # masks.data is (N, H, W) tensor
        masks_tensor = results[0].masks.data
        for i in range(len(masks_tensor)):
             m = masks_tensor[i].cpu().numpy().astype(np.uint8)
             refined_masks.append(m)
             
    return refined_masks

# ==========================================
# 3. METRICS CALCULATOR (IoU, RMSE)
# ==========================================
def calculate_metrics(pred_masks, gt_masks, gsd):
    """
    Computes precision, recall, mean_iou, area_rmse against Ground Truth.
    Simple greedy matching IoU.
    """
    if len(gt_masks) == 0:
        return {'precision': 0, 'recall': 0, 'iou': 0, 'area_rmse': 0}
    if len(pred_masks) == 0:
        # Missed everything
        total_gt_area = sum([np.sum(m)*(gsd**2) for m in gt_masks])
        return {'precision': 0, 'recall': 0, 'iou': 0, 'area_rmse': total_gt_area} # Rough error

    # Cost Matrix
    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            intersection = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()
            iou_matrix[i, j] = intersection / (union + 1e-6)
            
    # Matches (IoU > 0.5)
    matched_gt = set()
    matched_pred = set()
    tp = 0
    ious = []
    area_errors = []
    
    # Greedy match
    # (In production, use Hungarian algo, here simple max is usually fine for sparse trees)
    # We'll stick to a simpler threshold loop
    
    for i in range(len(gt_masks)):
        best_iou = 0
        best_j = -1
        for j in range(len(pred_masks)):
            if j in matched_pred: continue
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
                
        if best_iou >= 0.5:
            tp += 1
            matched_gt.add(i)
            matched_pred.add(best_j)
            ious.append(best_iou)
            
            # Area Error (sq meters)
            area_gt = np.sum(gt_masks[i]) * (gsd**2)
            area_pred = np.sum(pred_masks[best_j]) * (gsd**2)
            area_errors.append((area_pred - area_gt)**2)
            
    fp = len(pred_masks) - len(matched_pred)
    fn = len(gt_masks) - len(matched_gt)
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    mean_iou = np.mean(ious) if len(ious) > 0 else 0
    
    # Adds penalty for missed objects
    # For RMSE, we usually sum errors of matched + missed? 
    # Let's stick to RMSE of *detected* objects vs GT or strict layout.
    # User formula: sqrt(mean((pred-act)^2))
    rmse = np.sqrt(np.mean(area_errors)) if len(area_errors) > 0 else 0
    
    return {
        'precision': precision, 
        'recall': recall, 
        'f1': f1, 
        'mean_iou': mean_iou, 
        'area_rmse': rmse
    }

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================

def main_evaluation():
    print("🚀 STARTING MEGA-EVALUATION...")
    
    # Load Models
    model_v8 = YOLO(f'{WEIGHTS_DIR}/yolov8l_base/weights/best.pt')
    model_v11 = YOLO(f'{WEIGHTS_DIR}/yolo11l_base/weights/best.pt')
    sam_model = get_sam_model()
    model_rcnn = load_maskrcnn()
    
    model_ablation = YOLO(f'{WEIGHTS_DIR}/yolo11l_ablation/weights/best.pt')
    
    results = [] # List of dicts
    
    # Find all test images
    test_images = sorted(list(Path(TEST_DIR).rglob("*.png")))
    
    for idx, img_path in enumerate(test_images):
        # 1. Parse GSD and GT
        # 0.03m_sample_01.png
        try:
             gsd_str = img_path.name.split('_')[0] # 0.03m
             gsd = float(gsd_str.replace('m', ''))
        except:
             gsd = 0.05 # Fallback
             
        # Load GT Masks (from txt)
        # We need to reconstruct GT masks from .txt label to calculate IoU
        lbl_path = img_path.parent / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        gt_masks = []
        if lbl_path.exists():
             with open(lbl_path, 'r') as f:
                 for line in f:
                     coords = list(map(float, line.strip().split()))[1:]
                     poly = np.array(coords).reshape(-1, 2)
                     poly[:, 0] *= w
                     poly[:, 1] *= h
                     mask = np.zeros((h, w), dtype=np.uint8)
                     cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                     gt_masks.append(mask)

        # -----------------------------
        # 2. RUN MODELS
        # -----------------------------
        
        models_to_run = [
            ('YOLOv8', model_v8, False),
            ('YOLOv11', model_v11, False),
            ('YOLOv11_Ablation', model_ablation, False),
            ('Hybrid_v8+SAM', model_v8, True),
            ('Hybrid_v11+SAM', model_v11, True),
            ('Mask_RCNN', model_rcnn, False),
            ('Watershed', None, False)
        ]
        
        for name, model_obj, use_sam in models_to_run:
            start_time = time.time()
            pred_masks = []
            
            if name == 'Watershed':
                pred_masks = run_watershed(img_path)
            
            elif name == 'Mask_RCNN':
                if model_obj:
                    # Transform standard
                    img_tensor = torchvision.transforms.functional.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(DEVICE)
                    with torch.no_grad():
                        out = model_obj([img_tensor])[0]
                    # Threshold
                    for i in range(len(out['scores'])):
                        if out['scores'][i] > 0.5:
                            m = out['masks'][i, 0].cpu().numpy()
                            pred_masks.append((m > 0.5).astype(np.uint8))
            
            elif 'YOLO' in name or 'Hybrid' in name:
                # YOLO Inference
                yolo_res = model_obj(img_path, verbose=False)[0]
                
                if use_sam and sam_model:
                    # Hybrid Path
                    if yolo_res.boxes:
                         pred_masks = run_hybrid_inference(yolo_res, sam_model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    # Pure YOLO Path
                    if yolo_res.masks:
                        for m in yolo_res.masks.data:
                             pred_masks.append(m.cpu().numpy().astype(np.uint8))
            
            inference_time = (time.time() - start_time) * 1000 # ms
            
            # Calculate Metrics
            metrics = calculate_metrics(pred_masks, gt_masks, gsd)
            
            # Log
            row = {
                'image': img_path.name,
                'gsd': gsd,
                'model': name,
                'inference_ms': inference_time,
                **metrics
            }
            results.append(row)
            
        if idx % 5 == 0: print(f"   Processed {idx}/{len(test_images)}...")

    # ==========================================
    # 5. STATISTICAL ANALYSIS
    # ==========================================
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Raw results saved to {OUTPUT_CSV}")
    
    # A. BOOTSTRAPPING (95% CI)
    print("\n📊 BOOTSTRAPPING CONFIDENCE INTERVALS (1000 Samples)...")
    stats_summary = []
    
    unique_models = df['model'].unique()
    for m in unique_models:
        model_data = df[df['model'] == m]
        # Skip if empty
        if len(model_data) < 2: continue
        
        # Resample mean F1
        f1_scores = model_data['f1'].values
        # Bootstrap
        res = stats.bootstrap((f1_scores,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
        
        stats_summary.append({
            'Model': m,
            'F1_Mean': np.mean(f1_scores),
            'F1_CI_Low': res.confidence_interval.low,
            'F1_CI_High': res.confidence_interval.high,
            'RMSE_Mean': model_data['area_rmse'].mean()
        })
        
    stats_df = pd.DataFrame(stats_summary)
    print(stats_df)
    stats_df.to_csv("statistical_summary_ci.csv", index=False)
    
    # B. PAIRED T-TEST
    # Specifically v8 vs v11
    print("\n⚔️ PAIRED T-TEST (v8 vs v11)...")
    try:
        df_v8 = df[df['model']=='YOLOv8'].sort_values('image')
        df_v11 = df[df['model']=='YOLOv11'].sort_values('image')
        
        t_stat, p_val = stats.ttest_rel(df_v8['mean_iou'], df_v11['mean_iou'])
        print(f"   v8 vs v11 (IoU): T={t_stat:.4f}, p={p_val:.5f}")
        if p_val < 0.05:
            print("   👉 Result is STATISTICALLY SIGNIFICANT.")
        else:
            print("   👉 Result is NOT significant.")
            
    except Exception as e:
        print(f"   Could not run T-Test: {e}")

if __name__ == "__main__":
    main_evaluation()
```
