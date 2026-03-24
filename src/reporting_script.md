# Reporting & Analysis Script

Run this script **after** the pipeline has completed. It will scan the output directory, calculate physical dimensions (in meters) for every detected object, and generate Excel-ready CSV reports.

```python
import os
import pandas as pd
from pathlib import Path
import numpy as np
from glob import glob

# ==========================================
# ⚙️ USER CONFIGURATION
# ==========================================
# Must match the output directory from the previous step
OUTPUT_DIR = "/content/output_pipeline"
IMG_SIZE = 640  # Pixel size used for inference

# ==========================================
# 📊 REPORTING LOGIC
# ==========================================

def calculate_metrics(w_norm, h_norm, gsd_m):
    """
    Converts normalized YOLO dimensions to physical meters.
    """
    # Convert normalized (0-1) to pixels
    w_px = w_norm * IMG_SIZE
    h_px = h_norm * IMG_SIZE
    
    # Convert pixels to meters
    width_m = w_px * gsd_m
    height_m = h_px * gsd_m
    
    # Area (Rectangle approximation)
    area_m2 = width_m * height_m
    
    # Equivalent Diameter (assuming circular canopy approx)
    # Area = pi * (d/2)^2  =>  d = 2 * sqrt(Area / pi)
    diameter_m = 2 * np.sqrt(area_m2 / np.pi)
    
    return width_m, height_m, area_m2, diameter_m

def generate_reports():
    base_dir = Path(OUTPUT_DIR)
    if not base_dir.exists():
        print(f"❌ Error: Output directory not found: {base_dir}")
        return

    all_records = []
    
    # Scan for GSD folders
    gsd_folders = sorted(list(base_dir.glob("gsd_*m")))
    print(f"🔍 Found {len(gsd_folders)} GSD groups. Generating reports...")

    for folder in gsd_folders:
        folder_name = folder.name
        # Extract GSD value from folder name "gsd_0.03m"
        try:
            gsd_val = float(folder_name.replace("gsd_", "").replace("m", ""))
        except ValueError:
            print(f"   ⚠️ Skipping malformed folder: {folder_name}")
            continue

        label_dir = folder / "labels"
        if not label_dir.exists():
            print(f"   ⚠️ No labels found in {folder_name}")
            continue

        # Process all text files in this group
        group_records = []
        txt_files = list(label_dir.glob("*.txt"))
        
        print(f"   Processing {folder_name} ({len(txt_files)} files detected detections)...")

        for txt_file in txt_files:
            # Filename format: 0.03meter_06_00.txt
            img_name = txt_file.stem
            
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # YOLO Format: class x y w h [conf]
                # We expect at least 5 args. If conf is present, it's index 5.
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) > 5 else 0.0

                # Calculate Physical Metrics
                w_m, h_m, area_m, diam_m = calculate_metrics(w, h, gsd_val)

                record = {
                    'Group': folder_name,
                    'GSD (m)': gsd_val,
                    'Image Name': img_name,
                    'Object ID': i + 1,
                    'Confidence': conf,
                    'Width (m)': round(w_m, 4),
                    'Height (m)': round(h_m, 4),
                    'Area (m²)': round(area_m, 4),
                    'Diameter (m)': round(diam_m, 4)
                }
                group_records.append(record)
                all_records.append(record)

        # Save Group Summary
        if group_records:
            df_group = pd.DataFrame(group_records)
            csv_path = folder / f"inference_summary_{folder_name}.csv"
            df_group.to_csv(csv_path, index=False)
            print(f"      📄 Saved: {csv_path}")

    # Save Overall Summary
    if all_records:
        df_all = pd.DataFrame(all_records)
        
        # 1. Detailed All-in-One CSV
        master_csv = base_dir / "master_detection_log.csv"
        df_all.to_csv(master_csv, index=False)
        
        # 2. High-Level Aggregated Report
        # Pivot table: Average Diameter, Count of Trees per GSD
        summary_table = df_all.groupby('GSD (m)').agg({
            'Object ID': 'count',
            'Confidence': 'mean',
            'Diameter (m)': ['mean', 'min', 'max'],
            'Area (m²)': 'mean'
        }).reset_index()
        
        # Flatten columns for clean CSV
        summary_table.columns = ['GSD (m)', 'Total Trees', 'Avg Conf', 'Avg Dia (m)', 'Min Dia', 'Max Dia', 'Avg Area (m²)']
        
        overall_csv = base_dir / "overall_summary_report.csv"
        summary_table.to_csv(overall_csv, index=False)
        
        print("\n✅ Reporting Complete!")
        print(f"   • Master Log: {master_csv}")
        print(f"   • Executive Summary: {overall_csv}")
        print("\n📊 Executive Summary Preview:")
        print(summary_table.to_string(index=False))
    else:
        print("\n⚠️ No detections found matching the criteria. No reports generated.")

if __name__ == "__main__":
    generate_reports()
```
