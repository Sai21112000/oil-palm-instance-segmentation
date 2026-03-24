# Unified GSD Tiling & YOLO Inference Script for Colab

Use the following code block in your Google Colab environment (A100 GPU recommended).

## Setup
First, ensure you have the required libraries installed in a cell:
```bash
!pip install rasterio ultralytics
```

## The Script
Copy this entire block into a python cell and run it.

```python
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import time
import argparse
import sys

# Try imports
try:
    import rasterio
    from rasterio import windows
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
except ImportError:
    print("❌ ERROR: Rasterio is not installed. Run: pip install rasterio")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO
    HAS_TORCH = True
except ImportError:
    print("⚠️ Warning: PyTorch/Ultralytics not found. GPU features/Inference disabled.")
    HAS_TORCH = False

# ==========================================
# ⚙️ USER CONFIGURATION
# ==========================================
# Update these paths to match your Google Drive structure
INPUT_PATH = "/content/drive/MyDrive/Dec31_Dataset/06.tif"
MODEL_PATH = "/content/drive/MyDrive/Dec31_Dataset/yolo11l_Seg_results/weights/best.pt"
OUTPUT_DIR = "/content/output_pipeline"

# Native GSD of the input image
NATIVE_GSD = 0.054

# Target GSDs (Meters per pixel)
TARGET_GSDS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20]
TARGET_TILE_SIZE = 640
NUM_SAMPLES = 50 
RANDOM_SEED = 42

# ==========================================
# 🧠 UNIFIED PIPELINE CLASS
# ==========================================

class UnifiedPipeline:
    def __init__(self, input_path, model_path, output_dir, native_gsd=NATIVE_GSD):
        self.input_path = Path(input_path)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.native_gsd = native_gsd
        self.device = torch.device('cuda' if torch.cuda.is_available() and HAS_TORCH else 'cpu')
        
        print(f"🚀 Initializing Unified Pipeline")
        print(f"   • Input: {self.input_path}")
        print(f"   • Device: {self.device}")
        
        # Load Model
        if HAS_TORCH and self.model_path.exists():
            print(f"   • Loading Model: {self.model_path.name}")
            self.model = YOLO(self.model_path)
        else:
            print(f"   ⚠️ Model not found or torch missing. Inference will be skipped or mocked.")
            self.model = None

    def get_random_centers(self, width, height, count, seed=42):
        """Generates random pixel centers in the image."""
        np.random.seed(seed)
        # Avoid edges (5% buffer)
        centers_x = np.random.randint(int(width * 0.05), int(width * 0.95), count)
        centers_y = np.random.randint(int(height * 0.05), int(height * 0.95), count)
        return list(zip(centers_x, centers_y))

    def extract_tile_gpu(self, src_dataset, center_x, center_y, target_gsd):
        """Extracts and resizes a tile using GPU acceleration."""
        # Calculate Scale ratio > 1.0 (Zoom out)
        downsample_ratio = target_gsd / self.native_gsd
        native_read_size = int(TARGET_TILE_SIZE * downsample_ratio)
        
        # Calculate Window
        col_off = center_x - (native_read_size // 2)
        row_off = center_y - (native_read_size // 2)
        
        # Clamp
        col_off = max(0, min(col_off, src_dataset.width - native_read_size))
        row_off = max(0, min(row_off, src_dataset.height - native_read_size))
        
        # Read Native Data (CPU)
        window = windows.Window(col_off, row_off, native_read_size, native_read_size)
        native_data = src_dataset.read(window=window) # (Components, H, W)

        if native_data.shape[1] != native_read_size or native_data.shape[2] != native_read_size:
            return None # Edge case skip

        # GPU Resize
        with torch.no_grad():
            # 1. To Tensor (C, H, W). Handle 16-bit or 8-bit.
            dtype_max = 65535.0 if src_dataset.profile['dtype'] == 'uint16' else 255.0
            tensor_img = torch.from_numpy(native_data).float().to(self.device) / dtype_max
            
            # 2. Add Batch (1, C, H, W)
            tensor_img = tensor_img.unsqueeze(0)
            
            # 3. Resize
            resized = F.interpolate(
                tensor_img,
                size=(TARGET_TILE_SIZE, TARGET_TILE_SIZE),
                mode='bilinear',
                align_corners=False
            )
            
            # 4. Back to CPU (C, H, W)
            result = (resized.squeeze(0) * 255.0).byte().cpu().numpy()
            
        return result

    def save_tile(self, img_data, output_path):
        """Saves (C,H,W) array as PNG."""
        img_chw = img_data[:3, :, :] # RGB only
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        Image.fromarray(img_hwc).save(output_path)

    def run(self):
        # 1. Setup Validation
        if not self.input_path.exists():
            print(f"❌ Input file not found: {self.input_path}")
            # Generate dummy for testing
            print("   Creating DUMMY input for demonstration...")
            self.create_dummy_input()

        # 2. Tiling Phase
        print(f"\nPhase 1: Generative Tiling ({NUM_SAMPLES} samples @ {len(TARGET_GSDS)} GSDs)")
        
        generated_files = [] # Track for inference phase

        with rasterio.open(self.input_path) as src:
            centers = self.get_random_centers(src.width, src.height, NUM_SAMPLES, RANDOM_SEED)
            
            for gsd in TARGET_GSDS:
                print(f"   🔄 Processing GSD: {gsd}m...")
                
                # Output structure: output_dir/gsd_0.03m/images/
                gsd_dir = self.output_dir / f"gsd_{gsd}m"
                images_dir = gsd_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                for i, (cx, cy) in enumerate(centers):
                    # Naming: 0.03meter_filename_id.png
                    base_name = self.input_path.stem
                    # Format: 0.03meter_06_01.png
                    filename = f"{gsd}meter_{base_name}_{i:02d}.png"
                    out_path = images_dir / filename
                    
                    try:
                        tile_data = self.extract_tile_gpu(src, cx, cy, gsd)
                        if tile_data is not None:
                            self.save_tile(tile_data, out_path)
                            generated_files.append((out_path, gsd))
                    except Exception as e:
                        print(f"      Error on sample {i}: {e}")

        # 3. Inference Phase
        print(f"\nPhase 2: YOLO Inference on {len(generated_files)} files")
        
        if self.model is None:
            print("   ⚠️ Skipping inference (No model/torch).")
            return

        # Group by GSD for organized batching structure
        for gsd in TARGET_GSDS:
            gsd_dir = self.output_dir / f"gsd_{gsd}m"
            images_dir = gsd_dir / "images"
            if not images_dir.exists(): continue
            
            # Setup Inference Outputs
            labels_dir = gsd_dir / "labels"
            vis_dir = gsd_dir / "inference_images"
            labels_dir.mkdir(exist_ok=True)
            vis_dir.mkdir(exist_ok=True)
            
            # Get images in this folder
            imgs = list(images_dir.glob("*.png"))
            if not imgs: continue
            
            print(f"   🧠 Inferencing GSD {gsd}m ({len(imgs)} images)...")
            
            # Run Batch Inference
            results = self.model.predict(
                source=imgs,
                conf=0.35, # Threshold
                imgsz=TARGET_TILE_SIZE,
                device=self.device,
                stream=False,
                verbose=False
            )
            
            for res in results:
                # Original filename: 0.03meter_06_00.png
                p = Path(res.path)
                stem = p.stem # 0.03meter_06_00
                
                # 1. Save Labels (Txt) -> 0.03meter_06_00.txt
                # Note: Ultralytics save_txt usually saves to {save_dir}/labels/{stem}.txt
                # We force the path manually:
                res.save_txt(str(labels_dir / f"{stem}.txt"), save_conf=True)
                
                # 2. Save Visualization -> 0.03meter_06_00.jpg
                res.save(str(vis_dir / f"{stem}.jpg"))

        print("\n🎉 Pipeline Complete! Output available at:", self.output_dir)

    def create_dummy_input(self):
        """Creates a random noise TIFF if input is missing."""
        data = np.random.randint(0, 255, (3, 2000, 2000), dtype='uint8')
        transform = rasterio.transform.from_origin(0, 2000, 0.01, 0.01)
        with rasterio.open(self.input_path, 'w', driver='GTiff', height=2000, width=2000, count=3, dtype='uint8', transform=transform) as dst:
            dst.write(data)

if __name__ == "__main__":
    pipeline = UnifiedPipeline(INPUT_PATH, MODEL_PATH, OUTPUT_DIR)
    pipeline.run()
```
