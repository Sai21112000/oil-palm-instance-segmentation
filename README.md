# oil-palm-instance-segmentation

> Training, evaluation, and comparison of 6 instance segmentation models on multi-resolution oil palm UAV imagery.

Live blog post **[B3: Six Models Enter, One Problem Wins](https://sai21112000.github.io/posts/b3-six-models-enter.html)**.

## Models Compared

| Model | Params (M) | Mean F1 | Mean IoU | FPS (A100) |
|-------|-----------|---------|----------|------------|
| Mask R-CNN (ResNet-50) | 43.9 | **0.727** | 0.751 | 42.5 |
| YOLOv11l-seg | 27.6 | 0.646 | **0.769** | 54.6 |
| YOLOv11-ablation | 27.6 | 0.667 | — | 52.8 |
| YOLOv8l-seg | 45.9 | 0.616 | 0.751 | 80.8 |
| Hybrid-v11-SAM | 251.6 | 0.680 | 0.696 | 2.4 |
| Hybrid-v8-SAM | 251.6 | 0.635 | 0.692 | 2.4 |

**Key finding:** YOLOv11l-seg is 40% lighter than YOLOv8l-seg with better accuracy on both F1 and IoU. For new deployments, there is no reason to use v8 over v11.

## What This Repo Contains

- `configs/` — Training YAML configs for all 6 models
- `notebooks/model_comparison.ipynb` — Full evaluation: metrics by GSD, PR curves, confidence intervals
- `results/` — All result CSVs: per-GSD F1/IoU tables
- `notebooks/training_yolov11.ipynb` — YOLOv11 training walkthrough on Colab A100
- `plots/` — PR curves, GSD sensitivity curves, qualitative comparison figures

---
*M.Eng. Thesis · Asian Institute of Technology, Thailand · Sai Teja Vaidya*
