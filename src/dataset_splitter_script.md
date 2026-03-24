# Dataset Splitter & Formatter Script (v2 Robust)

Run this script to strictly clean filenames and organize your dataset.

## fixes in this version:
1.  **Robust ID Cleaning**: Uses Regex to strip `0.03m_`, `0.03meter_`, `3cm_` etc. to find the true "image ID".
2.  **Strict Renaming**: Forces `0.xxm_{id}.png` format. No double prefixes.
3.  **Benchmark Integrity**: Selects 4 locations and ensures their **exact** matches are moved to test.

```python
import os
import shutil
import random
import yaml
import re
from pathlib import Path
from glob import glob

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_DIR = "/content/output_pipeline"      # Where your GSD folders are
FINAL_DATASET_DIR = "/content/final_dataset" # Where to save the YOLO dataset
TEST_LOCATIONS_COUNT = 4                    # Number of unique locations for benchmark (4-5)

def get_clean_id(filename):
    """
    Removes any GSD context prefix from the filename to get a 'clean' unique Tree-ID.
    Examples:
    '0.03meter_sample_01.png' -> 'sample_01'
    '0.03m_sample_01.png'     -> 'sample_01'
    'gsd_0.20m_img99.jpg'     -> 'img99'
    'raw_image_05.png'        -> 'raw_image_05'
    """
    stem = Path(filename).stem
    # Regex to match common prefixes: numbers + (m/meter/cm) + underscore
    # Matches: "0.03m_", "0.03meter_", "3cm_", "gsd_0.03m_"
    pattern = r"^(?:gsd_)?\d+(?:\.\d+)?(?:m|meter|cm)_(.*)$"
    
    match = re.match(pattern, stem, re.IGNORECASE)
    if match:
        return match.group(1) # Return the ID part
    return stem # Return original if no prefix found

def main():
    base_dir = Path(INPUT_DIR)
    out_dir = Path(FINAL_DATASET_DIR)
    
    if out_dir.exists():
        print(f"⚠️ Removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    
    # Create Structure
    (out_dir / "train/images").mkdir(parents=True)
    (out_dir / "train/labels").mkdir(parents=True)
    (out_dir / "val/images").mkdir(parents=True)
    (out_dir / "val/labels").mkdir(parents=True)
    (out_dir / "test_benchmark").mkdir(parents=True)

    print("🚀 Starting Robust Dataset Organization...")

    # 1. Index Clean IDs
    location_map = {}
    
    all_images = sorted(list(base_dir.rglob("*.png")) + list(base_dir.rglob("*.jpg")))
    
    for img_path in all_images:
        # Determine GSD from Folder Structure
        folder_name = img_path.parent.parent.name # e.g., 'gsd_0.03m'
        try:
            # Extract 0.03 from gsd_0.03m
            gsd_val_raw = folder_name.lower().replace("gsd_", "").replace("m", "")
            gsd_val = float(gsd_val_raw)
            # Standard Prefix: 0.03m_
            standard_prefix = f"{gsd_val:g}m" # :g removes trailing zeros if integer like
        except:
            continue
            
        clean_id = get_clean_id(img_path.name)
        
        lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        
        if clean_id not in location_map:
            location_map[clean_id] = []
            
        location_map[clean_id].append({
            'gsd': gsd_val,
            'prefix': standard_prefix,
            'clean_id': clean_id,
            'img': img_path,
            'lbl': lbl_path
        })

    unique_ids = list(location_map.keys())
    print(f"   📍 Found {len(unique_ids)} unique image locations (IDs).")

    # 2. Select Benchmark Set (FIRST)
    random.seed(42)
    # Filter for locations that appear in multiple GSDs (optional check, but good for benchmark)
    # We will just pick random for now, assuming the pipeline generated most files.
    if len(unique_ids) < TEST_LOCATIONS_COUNT:
         bench_ids = unique_ids
    else:
         bench_ids = random.sample(unique_ids, TEST_LOCATIONS_COUNT)
         
    remaining_ids = [x for x in unique_ids if x not in bench_ids]
    
    print(f"   🧪 Benchmark IDs ({len(bench_ids)}): {bench_ids}")
    
    # 3. Helper Function to Save
    def save_file(record, dest_img, dest_lbl):
        if not record['lbl'].exists(): return
        
        # Enforce Standard Name: 0.03m_CleanID.png
        final_name = f"{record['prefix']}_{record['clean_id']}.png"
        final_txt_name = f"{record['prefix']}_{record['clean_id']}.txt"
        
        shutil.copy(record['img'], dest_img / final_name)
        shutil.copy(record['lbl'], dest_lbl / final_txt_name)

    # 4. Process Benchmark
    count_benchmark = 0
    for bid in bench_ids:
        records = location_map[bid]
        for rec in records:
            # Save to test_benchmark/gsd_0.03m/
            # User wants individual directories for GSDs in Test
            target_dir = out_dir / "test_benchmark" / f"gsd_{rec['gsd']}m"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images/labels together in this viewing folder? 
            # Or split? Standard YOLO is split. 
            # User request: "craft individual directories for each GSD...". 
            # Usually for inspection, putting them together is easier.
            # But let's create 'images' and 'labels' inside if needed or just flat.
            # Let's keep flat for easy viewing as per "test the x-image across all"
            
            # Actually, to make it valid for inference, maintaining flat might be confusing.
            # Let's clean it:
            shutil.copy(rec['img'], target_dir / f"{rec['prefix']}_{rec['clean_id']}.png")
            shutil.copy(rec['lbl'], target_dir / f"{rec['prefix']}_{rec['clean_id']}.txt")
            count_benchmark += 1

    # 5. Process Train/Val
    random.shuffle(remaining_ids)
    split_idx = int(len(remaining_ids) * 0.90) # 90% of Remaining -> Train
    
    train_ids = remaining_ids[:split_idx]
    val_ids = remaining_ids[split_idx:]
    
    # Train
    for tid in train_ids:
        for rec in location_map[tid]:
            save_file(rec, out_dir/"train/images", out_dir/"train/labels")
            
    # Val
    for vid in val_ids:
        for rec in location_map[vid]:
            save_file(rec, out_dir/"val/images", out_dir/"val/labels")

    # 6. YAML
    yaml_content = {
        'path': str(out_dir),
        'train': 'train/images',
        'val': 'val/images',
        # Test pointing to nowhere specific since it's custom structure, 
        # but user can point to specific GSD folders manually for inference.
        'names': {0: 'Oil Palm'}
    }
    
    with open(out_dir / "dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)

    print("\n✅ Dataset Organized!")
    print(f"   Benchmark Pack: {count_benchmark} files (across {len(bench_ids)} locations)")
    print(f"   Train Locations: {len(train_ids)}")
    print(f"   Val Locations:   {len(val_ids)}")

if __name__ == "__main__":
    main()
```
