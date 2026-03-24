
# 🎯 Step 3.3: Qualitative Comparison Gallery

## Goal
Create the "Figure X: Qualitative Comparison" for the paper.
It displays side-by-side inference results to visually demonstrate:
1.  **Overlapping Palms** (Density Challenge)
2.  **Boundary/Small Palms** (Precision Challenge)
3.  **Failure Cases** (False Positives/Misses)

## Inputs
- `standardized_results/detailed_results_per_image.csv` (Used to find "interesting" images)
- `standardized_results/all_models_predictions.json` (Masks)
- Images directory (`/content/final_dataset/test_benchmark`)

## Output
- `standardized_results/Figure_Qualitative_Grid.png`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import cv2
from pathlib import Path
from pycocotools import mask as mask_utils

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_STATS = 'standardized_results/detailed_results_per_image.csv'
INPUT_PREDS = 'standardized_results/all_models_predictions.json'
IMG_DIR = '/content/final_dataset/test_benchmark'
OUTPUT_DIR = 'standardized_results'

MODELS_ORDER = [
    'GT',
    'YOLOv11l-seg',
    'YOLOv8l-seg',
    'YOLOv11-Ablation', 
    'Mask R-CNN',
    'Hybrid-v8-SAM',
    'Hybrid-v11-SAM'
]

def load_image(filename):
    # Find recursively
    for p in Path(IMG_DIR).rglob(filename):
        return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    return None

def decode_mask(rle, h, w):
    if isinstance(rle, dict): # COCO RLE
        return mask_utils.decode(rle)
    elif isinstance(rle, list): # Polygon
        # Rasterize polygon not supported easily here without COCO API or cv2
        # Assuming predictions are RLE compressed or uncompressed RLE dict
        pass
    return np.zeros((h, w), dtype=np.uint8)

def overlay_masks(image, anns, color=(0, 255, 0), alpha=0.4):
    """
    image: RGB numpy array
    anns: List of COCO format dicts (predictions or GT)
    """
    overlay = image.copy()
    h, w = image.shape[:2]
    
    # Generate borders and fills
    for ann in anns:
        # Decode mask
        if 'segmentation' not in ann: continue
        
        seg = ann['segmentation']
        # If polygon (list of lists)
        if isinstance(seg, list):
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in seg:
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        # If RLE
        else:
            if isinstance(seg['counts'], list): # Uncompressed RLE
                mask = mask_utils.decode(seg)
            else: # Compressed RLE string/bytes
                mask = mask_utils.decode(seg)
                
        # Apply color
        if mask is None: continue
        
        # Color fill
        colored_mask = np.zeros_like(image)
        colored_mask[:] = color
        
        # Blend
        # We want to blend only where mask is 1
        mask_bool = mask.astype(bool)
        
        # Simple blend: dst = src1*alpha + src2*(1-alpha)
        # But only on mask area
        roi = overlay[mask_bool]
        blended = cv2.addWeighted(roi, 1-alpha, colored_mask[mask_bool], alpha, 0)
        overlay[mask_bool] = blended
        
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2) # White border
        
    return overlay

def select_interesting_images(df):
    """
    Selects 5 unique images:
    1. High Density (Overlapping)
    2. Small/Young (Precision)
    3. Failure Case (Low F1)
    4. Best Case (High F1 for v11)
    5. Random
    """
    candidates = []
    
    # 1. High Density
    dense = df.sort_values('GT_Count', ascending=False).head(10)['filename'].tolist()
    candidates.append(dense[0] if dense else None)
    
    # 2. Small Size (by Diameter Error or just GT Diameter if available, but assuming GT_Diam isn't per row in deep analysis logic but we constructed it?)
    # detailed_results has 'GT_Diam_m' per row? Yes, hopefully.
    if 'GT_Diam_m' in df.columns:
        small = df[df['GT_Diam_m'] > 0].sort_values('GT_Diam_m').head(10)['filename'].tolist()
        for s in small:
            if s not in candidates: candidates.append(s); break
    else:
        candidates.append(df.iloc[0]['filename']) # Fallback
        
    # 3. Failure for YOLOv11 (Low IoU)
    v11_rows = df[df['Model'].str.contains('11')]
    if not v11_rows.empty:
        worst = v11_rows.sort_values('IoU').head(5)['filename'].tolist()
        for w in worst:
            if w not in candidates: candidates.append(w); break
            
    # 4. Success for YOLOv11
    if not v11_rows.empty:
        best = v11_rows.sort_values('IoU', ascending=False).head(5)['filename'].tolist()
        for b in best:
            if b not in candidates: candidates.append(b); break
            
    # 5. Random
    while len(candidates) < 5:
        rnd = df.sample(1)['filename'].values[0]
        if rnd not in candidates: candidates.append(rnd)
        
    return candidates[:5]

def get_gt_anns(filename, img_shape):
    # Quick hack: read txt file corresponding to image
    # We need to look in the IMG_DIR
    h, w = img_shape[:2]
    anns = []
    
    # Find file path
    tgt_path = None
    for p in Path(IMG_DIR).rglob(filename):
        tgt_path = p; break
        
    if tgt_path:
        lbl_path = tgt_path.parent / (tgt_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    # Normalized polygon
                    poly = np.array(parts[1:]).reshape(-1, 2)
                    poly[:, 0] *= w; poly[:, 1] *= h
                    anns.append({'segmentation': [poly.flatten().tolist()]})
    return anns

def main():
    print("🎯 STARTING QUALITATIVE VISUALIZATION (Step 3.3)...")
    
    if not os.path.exists(INPUT_STATS) or not os.path.exists(INPUT_PREDS):
        print("❌ Missing stats or preds file.")
        return
        
    df = pd.read_csv(INPUT_STATS)
    with open(INPUT_PREDS, 'r') as f: all_preds = json.load(f)
    
    # Pre-process: Calculate GT_Count and ensure GT_Diam_m
    if 'GT_Count' not in df.columns:
        df['GT_Count'] = df['TP'] + df['FN']
        
    if 'GT_Diam_m' not in df.columns:
        # If missing, just set to 0 to avoid crash in selection
        df['GT_Diam_m'] = 0.0
    
    selected_files = select_interesting_images(df)
    print(f"   🖼️ Selected Images: {selected_files}")
    
    # Grid Setup
    n_cols = len(selected_files)
    n_rows = len(MODELS_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), dpi=150)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Fixed spacing argument
    
    labels_map = {
        'GT': 'Ground Truth',
        'YOLOv11l-seg': 'YOLOv11-Seg',
        'YOLOv8l-seg': 'YOLOv8-Seg',
        'YOLOv11-Ablation': 'v11-NoPSA',
        'Mask R-CNN': 'Mask R-CNN',
        'Hybrid-v8-SAM': 'Hybrid v8-SAM',
        'Hybrid-v11-SAM': 'Hybrid v11-SAM'
    }
    
    # Color Map for Rows (Distinct colors)
    # GT=Green, v11=Blue, v8=Orange, Abl=Red, RCNN=Purple, Hybrids=Cyan/Pink
    colors = [
        (0, 255, 0),   # GT - Green
        (0, 100, 255), # v11 - Blue
        (255, 165, 0), # v8 - Orange
        (255, 0, 0),   # Abl - Red
        (128, 0, 128), # RCNN - Purple
        (0, 255, 255), # Hyb8 - Cyan
        (255, 105, 180)# Hyb11 - Pink
    ]
    
    for col_idx, fname in enumerate(selected_files):
        img_raw = load_image(fname)
        if img_raw is None: continue
        
        for row_idx, model_name in enumerate(MODELS_ORDER):
            ax = axes[row_idx, col_idx]
            
            # Get Image
            img_disp = img_raw.copy()
            
            # Get Predictions
            if model_name == 'GT':
                anns = get_gt_anns(fname, img_disp.shape)
            else:
                # Fuzzy match model name
                # JSON models might be slightly different strings
                # Let's find best match in JSON models
                json_models = list(set(p['model'] for p in all_preds))
                matched_name = next((m for m in json_models if model_name.lower() in m.lower().replace(" ", "")), None)
                # Handle specific 'ablation' distinction if needed
                if model_name == 'YOLOv11-Ablation':
                    matched_name = next((m for m in json_models if 'ablation' in m.lower()), None)
                elif model_name == 'YOLOv11l-seg': # ensure not ablation
                    matched_name = next((m for m in json_models if '11' in m and 'ablation' not in m.lower() and 'sam' not in m.lower()), None)
                
                if matched_name:
                    # Filter for this image and model
                    # Check filename match? JSON usually has image_id. 
                    # We need to map filename -> image_id from somewhere?
                    # Stats DF has filename and image_id mapping implicitly? No...
                    # But load_image found it.
                    # Problem: We don't have filename<->ID map easily unless we scan JSON info if present?
                    # Or standardize script generated 'image_id' = hash or integer?
                    # Let's try to match by filename if 'file_name' in JSON?
                    # Usually predictions format is just id...
                    
                    # Assume we can simply map by filename? 
                    # For this visualizer, let's rely on Image ID if possible.
                    # detailed_results_per_image has 'image_id' ? No, 'filename'.
                    # We need to Re-ID based on filename alphabetically sorted usually?
                    # Let's assume standard COCO sort order:
                    all_imgs_sorted = sorted([f.name for f in Path(IMG_DIR).rglob("*.png")])
                    try:
                        img_id = all_imgs_sorted.index(fname)
                        anns = [p for p in all_preds if p['model'] == matched_name and p['image_id'] == img_id]
                    except ValueError:
                        anns = []
                else:
                    anns = []

            # Overlay
            img_viz = overlay_masks(img_disp, anns, color=colors[row_idx])
            
            ax.imshow(img_viz)
            ax.set_xticks([]); ax.set_yticks([])
            
            # Row Labels (Leftmost column)
            if col_idx == 0:
                ax.set_ylabel(labels_map.get(model_name, model_name), rotation=90, size='large', fontweight='bold')
                
            # Column Titles (Top row)
            if row_idx == 0:
                ax.set_title(f"Case {col_idx+1}", fontweight='bold')
                
    save_path = f"{OUTPUT_DIR}/Figure_Qualitative_Grid.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved Qualitative Gallery: {save_path}")

if __name__ == "__main__":
    main()
```
