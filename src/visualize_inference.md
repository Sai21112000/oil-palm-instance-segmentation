
# 2x3 Inference Visualization Script

This script generates a **comparative grid** of your 6 models for every test image.
It creates a folder `visualizations/` where you can inspect the qualitative differences.

**Grid Layout:**
| YOLOv8 | Hybrid v8+SAM | Mask R-CNN |
| :---: | :---: | :---: |
| **YOLOv11** | **Hybrid v11+SAM** | **YOLOv11-Ablation** |

## Instructions
1. Run this script in the same environment where you ran the "Mega-Evaluation".
2. Check the `visualizations/` folder for the output images.

```python
import torch
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO, SAM
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ==========================================
# ⚙️ CONFIG
# ==========================================
TEST_DIR = '/content/final_dataset/test_benchmark'
WEIGHTS_DIR = 'oil_palm_thesis'
OUTPUT_DIR = 'visualizations'
SAM_MODEL_NAME = "sam2.1_l.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. MODEL LOADERS (Identical to Eval Script)
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
    try: return SAM(SAM_MODEL_NAME)
    except: return None

# ==========================================
# 2. INFERENCE HELPERS
# ==========================================

def run_hybrid_inference(yolo_result, sam_model, original_image):
    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0: return []
    refined_masks = []
    results = sam_model.predict(source=original_image, bboxes=boxes, verbose=False)
    if results[0].masks is not None:
        masks_tensor = results[0].masks.data
        for i in range(len(masks_tensor)):
             m = masks_tensor[i].cpu().numpy().astype(np.uint8)
             refined_masks.append(m)
    return refined_masks

def overlay_mask(image, masks, color=(0, 255, 0), alpha=0.4):
    """Draws translucent masks on the image."""
    overlay = image.copy()
    for mask in masks:
        # Mask is binary 0/1, shape (H,W)
        # Color is BGR
        # Find contours for edge drawing (optional, cleaner look)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2) # Solid border
        
        # Fill
        # We can use boolean indexing to set color
        overlay[mask == 1] = color
        
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# ==========================================
# 3. MAIN VISUALIZATION LOOP
# ==========================================

def main_viz():
    print("🚀 STARTING VISUALIZATION GENERATION...")
    
    # Load Models
    print("   👉 Loading Models...")
    model_v8 = YOLO(f'{WEIGHTS_DIR}/yolov8l_base/weights/best.pt')
    model_v11 = YOLO(f'{WEIGHTS_DIR}/yolo11l_base/weights/best.pt')
    model_ablation = YOLO(f'{WEIGHTS_DIR}/yolo11l_ablation/weights/best.pt')
    sam_model = get_sam_model()
    model_rcnn = load_maskrcnn()
    
    test_images = sorted(list(Path(TEST_DIR).rglob("*.png")))
    print(f"   Found {len(test_images)} images to process.")

    for idx, img_path in enumerate(test_images):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_bgr.shape
        
        # Setup Plot Grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Sample: {img_path.name}", fontsize=16)
        
        # Define Layout: (Row, Col, Name, ModelObj, UseSAM, Color)
        # Colors: v8=Blue, v11=Green, RCNN=Red, Hybrids=Cyan/Lime, Ablation=Orange
        configs = [
            (0, 0, 'YOLOv8', model_v8, False, (0, 0, 255)),   # Blue
            (0, 1, 'Hybrid v8+SAM', model_v8, True, (0, 255, 255)), # Cyan 
            (0, 2, 'Mask R-CNN', model_rcnn, False, (255, 0, 0)),  # Red
            (1, 0, 'YOLOv11', model_v11, False, (0, 255, 0)),  # Green
            (1, 1, 'Hybrid v11+SAM', model_v11, True, (50, 205, 50)), # Lime
            (1, 2, 'YOLOv11-Ablation', model_ablation, False, (255, 165, 0)) # Orange
        ]
        
        for r, c, name, model, use_sam, color_bgr in configs:
            ax = axes[r, c]
            masks = []
            
            # RUN INFERENCE
            if name == 'Mask_RCNN':
                if model:
                    t_img = torchvision.transforms.functional.to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(DEVICE)
                    with torch.no_grad(): out = model([t_img])[0]
                    for i, s in enumerate(out['scores']):
                        if s > 0.5: masks.append((out['masks'][i, 0].cpu().numpy() > 0.5).astype(np.uint8))
            
            else: # YOLO / Hybrid
                try:
                    res = model(img_path, verbose=False)[0]
                    if use_sam and sam_model and res.boxes:
                        masks = run_hybrid_inference(res, sam_model, img_rgb)
                    elif res.masks:
                        for m in res.masks.data:
                            masks.append(m.cpu().numpy().astype(np.uint8))
                except Exception as e:
                    print(f"Error {name}: {e}")

            # RENDER
            # Create overlay on BGR, then convert to RGB for Matplotlib
            viz_img = overlay_mask(img_bgr, masks, color=color_bgr, alpha=0.5)
            viz_img_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
            
            # TEXT STATS
            count = len(masks)
            
            ax.imshow(viz_img_rgb)
            ax.set_title(f"{name}\nCount: {count}", fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"{img_path.stem}_comparison.jpg")
        plt.savefig(save_path)
        plt.close(fig)
        
        if idx % 5 == 0:
             print(f"   Generated {idx}/{len(test_images)}: {save_path}")

    print(f"✅ Visualization Complete! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main_viz()
```
