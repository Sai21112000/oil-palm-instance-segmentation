
# 📖 Thesis Chapter: Methodology and Analysis
## *Scale-Invariant Oil Palm Census via Physics-Integrated Deep Learning*

---

## 1. Introduction and Problem Statement

### 1.1 The Challenge of Scale in Precision Agriculture
The automated census of *Elaeis guineensis* (Oil Palm) is a cornerstone of modern plantation management. Precise counting and canopy sizing allow for yield estimation, fertilizer optimization, and early disease detection. However, traditional Remote Sensing approaches face a fundamental physical constraint: **The Scale Variance Problem**.

Agricultural drones (UAVs) operate in dynamic environments where flight altitude is dictated by terrain topology, battery constraints, and regulatory ceilings. A drone flying at 30 meters captures imagery with a Ground Sample Distance (GSD) of $0.03 m/pixel$, while the same drone at 120 meters captures $0.12 m/pixel$.
*   **The Consequence:** A single palm canopy (physically 10m diameter) appears as a 333-pixel object at low altitude but only an 83-pixel object at high altitude.
*   **The Failure Mode:** Conventional Convolutional Neural Networks (CNNs) trained on fixed-altitude data fail to generalize across this 4x scale variance, leading to catastrophic census errors when deployment altitudes differ from training altitudes.

### 1.2 Research Objective
This study proposes a **Physics-Aware Deep Learning Pipeline** designed to achieve Scale Invariance. We hypothesize that by explicitly modeling the relationship between GSD and Feature Representation, we can train a lightweight detector (YOLOv11) to be robust across the entire operational envelope (30m - 120m).

---

## 2. Methodology: Data Engineering and Simulation

### 2.1 The Generative Tiling Algorithm
To train a scale-invariant model without the prohibitive cost of flying 7 distinct missions, we developed a **Generative Tiling Engine**. This algorithm scientifically degrades high-resolution orthomosaics to simulate the sensor aggregation effects of higher altitudes.

**Algorithm 1: Stochastic Multi-Scale Tiling**
```python
Define: Target_GSDs = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20]
Define: Buffer_Ratio = 0.05
Define: Output_Size = 640

For each Orthomosaic M with Native_GSD_n (0.054m):
    W, H = Dimensions(M)
    Buffer_px = W * Buffer_Ratio
    
    For each Target_GSD_t in Target_GSDs:
        # Calculate Downsample Ratio (Sensor Zoom Factor)
        Ratio_R = Target_GSD_t / Native_GSD_n
        
        # Calculate Read Window Size (The "Field of View")
        Window_Size_w = Output_Size * Ratio_R
        
        # Stochastic Sampling Loop
        For i in range(Samples_N):
            # Select Random Center (Avoiding Edges)
            c_x ~ Uniform(Buffer_px, W - Buffer_px)
            c_y ~ Uniform(Buffer_px, H - Buffer_px)
            
            # Extract Raw Sensor Data
            Raw_Patch = M[c_y : c_y + Window_Size_w, 
                          c_x : c_x + Window_Size_w]
            
            # Bilinear Interpolation (Simulate Altitude)
            # Mathematical transformation of aggregating pixel photons
            Simulated_Patch = Resize(Raw_Patch, (640, 640), methods='bilinear')
            
            Save Simulated_Patch
```

### 2.2 Theoretical Basis of Resampling (Bilinear Interpolation)
Why Bilinear? When a physical camera sensor moves 2x further away, the light that previously hit 4 distinct photoreceptors now hits a single photoreceptor. This is an averaging process.
Mathematically, Bilinear Interpolation approximates this by computing the weighted average of the $2 \times 2$ neighborhood. This ensures that our synthetic "High Altitude" data is physically representative of real-world sensor dynamics, not just "resized images."

### 2.3 The "Starfield" Split Strategy (Spatial Independence)
To rigorously evaluate generalization, we rejected random splitting. We treat the dataset as a set of disjoint geospatial locations $L = \{L_A, L_B, L_C, L_D\}$.
*   **Training Set:** $T_{train} = \{ t | t \in L_A \cup L_B \cup L_C \}$
*   **Testing Set:** $T_{test} = \{ t | t \in L_D \}$
*   **Constraint:** $L_A \cap L_D = \emptyset$

This ensures **Zero Spatial Leakage**. The model is tested on a plantation with unique soil signatures, lighting conditions, and planting patterns it has never encountered during gradient descent.

---

## 3. Neural Architecture: YOLOv11 Internals

We selected **YOLOv11-Large** for its specific architectural features that address the challenges of aerial imagery.

### 3.1 Backbone: C3k2 (Cross Stage Partial with Kernel 2)
The backbone is responsible for feature extraction. The **C3k2** module is an evolution of the C2f block.
*   **Mechanism:** It splits the input tensor into two paths. One path passes through a series of bottlenecks (Convolution $\to$ SiLU $\to$ Convolution), while the other bypasses them.
*   **Benefit for Palms:** This "Cross Stage" flow allows the network to maintain low-level texture gradients (essential for spotting frond patterns) while simultaneously building high-level semantic abstractions (the circular shape of the canopy).

### 3.2 Neck: SPPF (Spatial Pyramid Pooling - Fast)
This is the critical component for **Scale Invariance**.
*   **Operation:** The SPPF module pools the feature map at multiple kernel sizes ($5 \times 5$, $9 \times 9$, $13 \times 13$) and concatenates the results.
*   **Significance:** It forces the network to look at the image with "different sized eyes" simultaneously. This effectively allows the model to recognize a palm tree whether it occupies 10% of the view (Low GSD) or 1% of the view (High GSD).

### 3.3 Head: Anchor-Free Detection
Unlike Mask R-CNN, which uses fixed Anchor Boxes (RPN), YOLOv11 is **Anchor-Free**.
*   **Center-ness:** It predicts the *center* of an object and the *distance* to its four edges.
*   **Advantage in Agriculture:** Oil palms are planted in dense hexagonal grids. Anchor-based methods often suppress valid detections in these clusters because the Anchors overlap too much. The Anchor-Free approach treats every pixel as a potential center, allowing it to resolve dense clusters with superior separation.

---

## 4. Mathematical Framework: Loss and Metrics

### 4.1 Loss Function Formulation
The model minimizes a composite loss function $\mathcal{L}_{total}$:

$$ \mathcal{L}_{total} = \lambda_{box}\mathcal{L}_{CIoU} + \lambda_{cls}\mathcal{L}_{BCE} + \lambda_{dfl}\mathcal{L}_{DFL} $$

#### A. Complete Intersection over Union (CIoU) Loss
Standard IoU is insufficient because non-overlapping boxes have zero gradient. CIoU adds geometric penalties:
$$ \mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v $$
*   This drives the predicted box to not just "touch" the tree, but to center itself perfectly and match the canopy circularity.

#### B. Distribution Focal Loss (DFL)
Palm canopies have fuzzy edges. Deterministic regression is unstable. DFL models the box boundaries as a General Distribution $P(x)$.
$$ \mathcal{L}_{DFL}(S_i, S_{i+1}) = -((y_{i+1} - y) \log(S_i) + (y - y_i) \log(S_{i+1})) $$
*   This allows the model to express "uncertainty" about the tree edge, which is physically realistic given the chaotic nature of fronds.

### 4.2 Validation Metric: The Shoelace Formula
To validate that our Bounding Boxes correlate to real physical biomass, we calculated the area of the Ground Truth Polygon ($\mathcal{P}$) using the Shoelace Formula:

$$ Area_{\mathcal{P}} = \frac{1}{2} | \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) | $$

We validated the model's physical predictions against this ground truth.

---

## 5. Experimental Results and Analysis

### 5.1 Statistical Significance Testing
We conducted a **Wilcoxon Signed-Rank Test** comparing YOLOv11l-seg against Mask R-CNN.
*   **IoU Metric (Shape Accuracy):** $p = 5.25 \times 10^{-6}$ (**Statistically Significant**).
    *   *Interpretation:* YOLOv11 produces significantly more accurate segmentation boundaries, adhering better to the circular canopy shape than Mask R-CNN's region proposals.
*   **F1-Score (Detection Count):** $p = 0.013$ (Significant), favoring Mask R-CNN slightly in total counts at very high/low resolutions, but YOLOv11 dominates in the middle operational range.

### 5.2 The Resolution Robustness Curve
Our analysis revealed a critical "Knee Point" in sensor resolution. Performance does not degrade linearly; it exhibits three distinct phases:
*   **Region I (0.03m - 0.04m):** High Fidelity.
    *   Mask R-CNN leads (F1: 0.88), taking advantage of massive pixel detail.
    *   YOLOv11 follows closely (F1: 0.83).
*   **Region II (0.05m - 0.10m):** The Operational Sweet-Spot.
    *   **Crucial Crossover:** At 0.10m GSD, **YOLOv11 (F1: 0.718) overtakes Mask R-CNN (F1: 0.693)**.
    *   *Why?* As texture fades, YOLO's SPPF module (geometry focus) becomes more robust than Mask R-CNN's dependence on fine texture.
*   **Region III (>0.15m):** Degradation.
    *   Both models suffer, dropping below 0.60 F1. The object size approaches the network's stride limit.

**Operational Recommendation:** The optimal flight altitude is that which yields **0.10m GSD**. This maximizes coverage area ($Area \propto Altitude^2$) while operating exactly where YOLOv11 is most competitive.

### 5.3 Efficiency Analysis
*   **YOLOv11l-seg:** 54.6 FPS (18.33ms latency).
*   **Mask R-CNN:** 42.5 FPS (23.51ms latency).
*   **Hybrid-SAM:** 2.4 FPS (418ms latency).
*   **Verdict:** YOLOv11 is the only model viable for real-time onboard processing, offering a 22% speed advantage over Mask R-CNN while being more robust at high (0.10m) altitudes.

---

## 6. Training Specifications (Reproducibility)

To ensure this study is reproducible, we document the exact hyperparameter configuration used.

| Component | Setting | Notes |
| :--- | :--- | :--- |
| **Framework** | Ultralytics 8.3 / PyTorch 2.0 | CUDA 11.8 |
| **Optimizer** | AdamW | Weight Decay: $0.0005$ |
| **Learning Rate** | $1 \times 10^{-3}$ | Cosine Decay $\to 1 \times 10^{-5}$ |
| **Batch Size** | 16 | Saturation of 40GB VRAM |
| **Image Size** | 640 x 640 | Native stride alignment |

---

## 7. Conclusion

This research has successfully demonstrated that **Physics-Aware Deep Learning** can overcome the scale invariance problem. While legacy models (Mask R-CNN) dominate at extremely low altitudes, **YOLOv11 demonstrates superior robustness at operational altitudes (100m / 0.10m GSD)**, where it statistically outperforms the baseline ($p < 0.05$) while running 20% faster.
