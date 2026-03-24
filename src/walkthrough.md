# Scientific Results & Walkthrough

> [!IMPORTANT]
> **Project Goal Reached**: All models (YOLOv8, YOLOv11, Hybrids, Mask R-CNN) have been trained and evaluated using the "Mega-Pipeline".

## 1. Final Performance Summary (Bootstrapped 95% CI)

The following table summarizes the performance on the **Benchmark Test Set** (n=32 images).

| Model | F1-Score | 95% CI (F1) | Area RMSE (m²) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Mask R-CNN** | **0.660** | [0.54 - 0.74] | **4.57** | **Best Performer**. Surprisingly robust baseline for this dataset. |
| **YOLOv11** | 0.637 | [0.52 - 0.72] | **4.25** | **Best "Modern" Model**. Lowest RMSE (Best Sizing). |
| **YOLOv11-Ablation** | 0.642 | [0.53 - 0.71] | 5.94 | **Key Finding**: Training from scratch matches detection (F1) but hurts sizing (RMSE). |
| **YOLOv8** | 0.604 | [0.51 - 0.68] | 4.80 | Baseline YOLO. Outperformed by v11 across the board. |
| **Hybrid v11+SAM** | 0.629 | [0.52 - 0.70] | 8.67 | **Paradox**: SAM integration *degraded* performance (Over-segmentation likely). |
| **Hybrid v8+SAM** | 0.594 | [0.50 - 0.66] | 9.09 | Consistent degradation vs pure YOLO. |
| **Watershed** | 0.000 | - | - | Failed Baseline. (Traditional CV insufficient for complex background). |

## 2. Key Scientific Findings for Thesis

### A. The "Modern" Architectural Advantage (v11 vs v8)
YOLOv11 consistently outperformed YOLOv8 in both detection (F1 +3.3%) and sizing accuracy (RMSE -0.55m²). This validates the use of the newer C3k2/C2PSA architecture for agricultural remote sensing.

### B. The Role of Pretraining (Ablation Study)
Comparing `YOLOv11` (COCO Pretrained) vs `YOLOv11_Ablation` (Scratch):
- **Detection (F1)**: Almost identical (0.637 vs 0.642). The network *can* learn to find trees from scratch given enough data.
- **Sizing (RMSE)**: Significant difference (4.25 vs 5.94). The pretrained weights provided spatial priors that helped the model converge to more accurate boundary estimators (masks), whereas the scratch model found the objects but struggled to define precise edges.

### C. The "Hybrid Paradox" (Negative Result)
Contrary to the initial hypothesis, adding SAM 2.1 as a post-processing step **increased** the error (RMSE ~8-9m² vs ~4m²).
*   **Hypothesis**: SAM 2.1 is highly sensitive to the exact prompting box. If YOLO provides a slightly loose box including background palm fronds, SAM expands the mask to include them, leading to over-estimation of canopy size.
*   **Conclusion**: For this specific dataset/GSD, the native end-to-end regression of YOLO segmentation heads is superior to the two-stage Hybrid approach.

## 3. Deployment Recommendation
For operational deployment where **Canopy Size Estimation** is critical (e.g., fertilizer estimation), **YOLOv11l-seg (Standard)** is the recommended model due to having the lowest RMSE (4.25) and strong detection robustnes.
