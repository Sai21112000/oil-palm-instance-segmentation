# Oil Palm Instance Segmentation — AIT Master's Thesis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11l--seg-brightgreen)](https://github.com/ultralytics/ultralytics)
[![Mask RCNN](https://img.shields.io/badge/Mask-RCNN-orange)](https://github.com/matterport/Mask_RCNN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AIT](https://img.shields.io/badge/AIT-Remote%20Sensing%20%26%20GIS-blue)](https://www.ait.ac.th)

> **AI Agent-Orchestrated Instance Segmentation: Deep Learning-Based Detection and Canopy Size Estimation of Oil Palm Trees Using UAV Imagery**  
> Master's Thesis | Asian Institute of Technology | Defended January 15, 2026

---

## 🌴 Overview

This repository contains the full research pipeline for detecting and measuring oil palm tree canopies from UAV (drone) imagery using state-of-the-art instance segmentation models.

**The Core Problem:** AI models trained at one drone altitude fail at others. Ground Sample Distance (GSD) — the real-world distance represented by one pixel — varies with altitude. A model trained at 0.03m GSD performs 40% worse at 0.20m GSD on the same plantation.

**The Solution:** A multi-GSD training pipeline + an AI Teacher Agent for automated annotation, tested across 8 altitude levels (0.03m–0.20m GSD).

---

## 📊 Key Results

| Model | Mean IoU | Mean F1 | Speed (ms/img) | Best GSD Level |
|-------|----------|---------|----------------|----------------|
| **Mask-RCNN** | 0.712 | **0.727** | ~340 | 0.03–0.10m |
| **YOLOv11l-seg** | **0.77** | 0.698 | **56** | 0.07–0.15m |
| Hybrid-v11-SAM 2.1 | 0.74 | 0.681 | ~600 | 0.10–0.15m |
| YOLOv8l-seg | 0.71 | 0.672 | 68 | 0.05–0.10m |
| Hybrid-v8-SAM 2.1 | 0.69 | 0.651 | ~580 | — |

> **Key Finding:** YOLOv11 standalone outperforms YOLO+SAM hybrid on mean IoU (0.77 vs 0.74) while being 10× faster.

---

## 🏗️ Architecture

```
Phase 1: Gold Standard Training
  UAV Imagery (Krabi, Thailand)
    └── Manual Annotation (500 images, 0.03m GSD)
         └── YOLOv11l-seg / YOLOv8l-seg Training

Phase 2: Teacher Agent Pipeline
  Trained Model (Teacher Agent)
    └── Generative Tiling Algorithm
         └── 8 GSD Levels (0.03m → 0.20m)
              └── Auto-Annotation (>80% time reduction)
                   └── Full Model Suite Training
                        └── Canopy Biometry (Shoelace Algorithm)
```

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/sai21112000/oil-palm-instance-segmentation.git
cd oil-palm-instance-segmentation

# Install dependencies
pip install -r requirements.txt

# Run inference on a single image
python predict.py --image path/to/uav_image.jpg --model yolov11l-seg --gsd 0.10

# Run canopy biometry estimation
python biometry.py --masks output/masks/ --gsd 0.10
```

---

## 📁 Repository Structure

```
oil-palm-instance-segmentation/
├── data/
│   ├── raw/                    # Original UAV imagery
│   ├── annotations/            # COCO-format JSON labels
│   └── tiles/                  # GSD-tiled dataset (8 levels)
├── models/
│   ├── yolov11l_seg/           # YOLOv11 weights & configs
│   ├── yolov8l_seg/            # YOLOv8 weights & configs
│   └── mask_rcnn/              # Mask-RCNN configs
├── src/
│   ├── tiling/                 # Generative Tiling Algorithm
│   ├── teacher_agent/          # Auto-annotation pipeline
│   ├── biometry/               # Shoelace canopy measurement
│   └── evaluation/             # Multi-GSD benchmark scripts
├── notebooks/
│   └── 10Jan_gsd_yolo_pipeline.ipynb  # Full training pipeline
├── results/
│   ├── metrics/                # F1, IoU, Precision, Recall per GSD
│   └── visualizations/         # Prediction overlays
├── requirements.txt
└── README.md
```

---

## 🌱 Dataset

- **Location:** Krabi Province, Southern Thailand (GPS: 8.0863° N, 98.9063° E)
- **Sensor:** DJI UAV, RGB imagery
- **Total tiles:** 5,000+ across 8 GSD levels
- **Annotation format:** COCO Instance Segmentation (polygon masks)
- **GSD range:** 0.03m, 0.05m, 0.07m, 0.10m, 0.12m, 0.15m, 0.17m, 0.20m

---

## 📐 Canopy Biometry

Canopy area is calculated using the **Shoelace (Gauss's Area) Theorem**:

```
Area = ½ |Σ(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)|
```

Applied to segmentation polygon vertices scaled by GSD → area in m².  
Error vs ground truth: **<8%** on test set.

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{vaidya2026oilpalm,
  author    = {Vaidya, Sai Teja},
  title     = {AI Agent-Orchestrated Instance Segmentation: Deep Learning-Based 
               Detection and Canopy Size Estimation of Oil Palm Trees Using UAV Imagery},
  school    = {Asian Institute of Technology},
  year      = {2026},
  address   = {Pathum Thani, Thailand},
  month     = {January}
}
```

---

## 👤 Author

**Sai Teja Vaidya**  
M.Eng. Remote Sensing & GIS, Asian Institute of Technology  
📧 sai21112000@gmail.com | 🔗 [GitHub](https://github.com/sai21112000) | [LinkedIn](https://linkedin.com/in/sai-teja-vaidya)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
