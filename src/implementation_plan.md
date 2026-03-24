# Quad-Model Experiment Plan

## Goal
Implement a rigorous experimental pipeline to compare 4 model configurations:
1.  **YOLOv8l-seg** (Base)
2.  **YOLOv11l-seg** (Base)
3.  **Hybrid v8 + SAM 2.1** (YOLOv8 prompts SAM)
4.  **Hybrid v11 + SAM 2.1** (YOLOv11 prompts SAM)

## 1. Training Phase (Expanded)
We will train the following models:
1.  **YOLOv8l-seg** (Base)
2.  **YOLOv11l-seg** (Base)
3.  **Mask R-CNN (ResNet50)** (SOTA Comparison)
4.  **YOLOv11-No-C2PSA** (Ablation Study): A custom model with the attention module removed to isolate its impact.

*   **Script**: `train_experiment.py` (YOLO models), `train_mrcnn.py` (Mask R-CNN).
*   **Config**: Standardized epochs (200) and optimizer (AdamW) where possible.

## 2. Evaluation Phase (Quad-Inference)
A single comprehensive script `evaluate_system.py` will generate results for all 4 comparisons.

### A. Inference Logic
For each test image, we run **6 Systems**:
1.  **YOLOv8**
2.  **YOLOv11**
3.  **Hybrid v8 + SAM**
4.  **Hybrid v11 + SAM**
5.  **Mask R-CNN** (ResNet50 FPN)
6.  **Watershed Algorithm** (Traditional Baseline - OpenCV)

### B. Computational Profiling
*   **Metrics**:
    *   **FPS**: Average inference time per image (milliseconds).
    *   **GFLOPs**: Theoretical float operations (YOLO utils / thop).
    *   **Params**: Model size (millions).
    *   **Memory**: Peak VRAM usage.

### C. Error Analysis & Taxonomy
*   **Stratification**:
    *   **By Size**: Small vs Large Canopies (Proxy for Age/Height).
    *   **By Crowding**: Isolated Trees vs Clumped Clusters (Proxy for Occlusion).
*   **Failure Taxonomy**:
    *   Save "Worst 5" images for each model (Lowest IoU).
    *   Classify errors: "Missed Detection", "Oversized Mask", "Merged Trees".

### D. Ablation Analysis
*   Compare `YOLOv11` vs `YOLOv11-No-C2PSA`.
*   Directly quantify the gain from the attention module.

### B. Statistical Analysis (Bootstrapping & T-Tests)
*   **Bootstrapping**:
    *   For *each* of the 4 Models, resample the test set $N=1000$ times.
    *   Calculate **Mean** and **95% Confidence Interval** for: mAP50-95, F1, Recall, Area RMSE.
*   **Paired T-Tests**:
    *   Compare `v8 vs v11` and `Hybrid vs Base`.
    *   Calculate P-values for IoU significance.

### C. GSD-Stratified Reporting
*   Break down all metrics **per GSD folder** (e.g., "Performance at 0.03m" vs "Performance at 0.06m").
*   Generate `gsd_analysis.csv` showing how accuracy degrades/improves with resolution.

## 3. Deliverables
1.  `train_experiment.py`: For training the two base models.
2.  `evaluate_system.py`: The massive script handling Hybrid inference + Stats.
3.  `install_sam.sh`: Instructions to install SAM 2.1 (requires specific steps).

## User Review Required
> [!IMPORTANT]
> **SAM 2.1 checkpoints**: I assume you will provide the path to `sam2.1_hiera_large.pt` or similar in Colab.
> **Computation Time**: Running SAM inference adds significant time. The script will be optimized but expect it to be slower than pure YOLO.
