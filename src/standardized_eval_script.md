
# 🎯 Phase 1: Standardized Inference Pipeline

## Instructions
1. Upload this script as `eval_all_models.py`
2. Run with default settings (it will auto-detect your `final_dataset`)
3. Output will be saved to `standardized_results/`

```python
import torch
import numpy as np
import cv2
import json
import time
import os
import glob
from pathlib import Path
from ultralytics import YOLO, SAM
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools import mask as mask_utils

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
TEST_DIR = '/content/final_dataset/test_benchmark'
WEIGHTS_DIR = 'oil_palm_thesis'
OUTPUT_DIR = 'standardized_results'
SAM_MODEL_NAME = "sam2.1_l.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. HELPERS: MASK TO RLE (COCO FORMAT)
# ==========================================
def binary_mask_to_rle(binary_mask):
    """Converts a binary mask to COCO RLE format"""
    # Ensure Fortran order for RLE encoding
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def gpu_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2 # MB
    return 0.0

# ==========================================
# 2. MODEL LOADERS
# ==========================================
def load_maskrcnn():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    path = f"{WEIGHTS_DIR}/mask_rcnn_resnet50.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.to(DEVICE).eval()
        return model
    return None

def get_sam_model():
    try: return SAM(SAM_MODEL_NAME)
    except: return None

# ==========================================
# 3. PIPELINE ENGINE
# ==========================================

def run_evaluation():
    print("🎯 STARTING STANDARDIZED EVALUATION PIPELINE...")
    
    # 3.1 Load Models
    print("   👉 Loading Models...")
    try:
        models = {
            'YOLOv8': YOLO(f'{WEIGHTS_DIR}/yolov8l_base/weights/best.pt'),
            'YOLOv11': YOLO(f'{WEIGHTS_DIR}/yolo11l_base/weights/best.pt'),
            'YOLOv11-Ablation': YOLO(f'{WEIGHTS_DIR}/yolo11l_ablation/weights/best.pt'),
            'Mask-RCNN': load_maskrcnn(),
            'SAM': get_sam_model() # Shared SAM instance
        }
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return

    # 3.2 Prepare Test Set
    test_images = sorted(list(Path(TEST_DIR).rglob("*.png")))
    print(f"   👉 Found {len(test_images)} test images.")
    
    # Structure for COCO Results: List of dicts
    # [{'image_id': int, 'category_id': int, 'segmentation': RLE, 'score': float}]
    coco_results = []
    
    # Image ID Mapping
    image_id_map = {img.name: i for i, img in enumerate(test_images)}
    
    # 3.3 Main Inference Loop
    for idx, img_path in enumerate(test_images):
        img_id = image_id_map[img_path.name]
        
        # Reset GPU Max Memory Tracker
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Load Image
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # DEFINE CONFIGS TO RUN
        run_configs = [
            ('YOLOv8l-seg', models['YOLOv8'], False),
            ('YOLOv11l-seg', models['YOLOv11'], False),
            ('YOLOv11-ablation', models['YOLOv11-Ablation'], False),
            ('Hybrid-v8-SAM', models['YOLOv8'], True),
            ('Hybrid-v11-SAM', models['YOLOv11'], True),
            ('Mask-RCNN', models['Mask-RCNN'], False)
        ]
        
        for model_name, model_obj, use_sam in run_configs:
            if model_obj is None: continue
            
            start_time = time.time()
            pred_masks = []
            pred_scores = []
            
            # --- INFERENCE BLOCK ---
            if model_name == 'Mask-RCNN':
                t_img = torchvision.transforms.functional.to_tensor(img_rgb).to(DEVICE)
                with torch.no_grad(): out = model_obj([t_img])[0]
                for i, s in enumerate(out['scores']):
                    if s > 0.5:
                        m = (out['masks'][i, 0].cpu().numpy() > 0.5).astype(np.uint8)
                        pred_masks.append(m)
                        pred_scores.append(float(s))
            
            else: # YOLO
                res = model_obj(img_path, verbose=False)[0]
                
                if use_sam and models['SAM'] and res.boxes:
                    # Hybrid Logic
                    boxes = res.boxes.xyxy.cpu().numpy()
                    sam_res = models['SAM'].predict(source=img_rgb, bboxes=boxes, verbose=False)
                    if sam_res[0].masks is not None:
                         for m in sam_res[0].masks.data:
                             pred_masks.append(m.cpu().numpy().astype(np.uint8))
                             pred_scores.append(1.0) # SAM doesn't give class scores easily, use 1.0
                elif res.masks:
                    # Base Logic
                    for i, m in enumerate(res.masks.data):
                        pred_masks.append(m.cpu().numpy().astype(np.uint8))
                        pred_scores.append(float(res.boxes.conf[i]))

            # --- METRICS & SAVING ---
            inference_time = (time.time() - start_time) * 1000 # ms
            peak_ram = gpu_usage()
            
            # Append to COCO Results
            for i, mask in enumerate(pred_masks):
                rle = binary_mask_to_rle(mask)
                res_entry = {
                    "image_id": img_id,
                    "image_name": img_path.name, # Extra field for debug
                    "category_id": 1, # Oil Palm
                    "segmentation": rle,
                    "score": pred_scores[i],
                    "model": model_name,
                    "inference_time_ms": round(inference_time, 2),
                    "gpu_mem_mb": round(peak_ram, 2)
                }
                coco_results.append(res_entry)
        
        if idx % 5 == 0: print(f"   Processed {idx}/{len(test_images)} images...")

    # 3.4 Save JSON
    out_path = os.path.join(OUTPUT_DIR, 'all_models_predictions.json')
    with open(out_path, 'w') as f:
        json.dump(coco_results, f)
        
    print(f"✅ Saved COCO results to: {out_path}")

if __name__ == "__main__":
    run_evaluation()
```
