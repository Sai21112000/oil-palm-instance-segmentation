"""
Canopy Biometry Calculator
Estimates canopy area and diameter from segmentation masks.

Uses the Shoelace (Gauss Area) Theorem to compute polygon area,
then scales by GSD to get real-world measurements in metres.

Author: Vaidya Sai Teja
Thesis: AIT Master of Engineering, 2026
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import json
import pandas as pd


def shoelace_area(polygon: np.ndarray) -> float:
    """
    Compute polygon area using Shoelace (Gauss) theorem.
    Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|

    Args:
        polygon: Nx2 array of (x, y) polygon vertices in pixels
    Returns:
        Area in square pixels
    """
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def pixels_to_metres(area_px: float, gsd: float) -> float:
    """
    Convert pixel area to real-world area in m².

    Args:
        area_px: Area in square pixels
        gsd: Ground Sample Distance in metres/pixel
    Returns:
        Area in square metres
    """
    return area_px * (gsd ** 2)


def estimate_diameter(area_m2: float) -> float:
    """
    Estimate canopy diameter assuming circular crown.
    d = 2 * sqrt(A / pi)
    """
    return 2.0 * np.sqrt(area_m2 / np.pi)


def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    """Extract largest contour polygon from binary mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def compute_canopy_biometry(masks: List[np.ndarray], gsd: float) -> List[Dict]:
    """
    Compute canopy area and diameter for a list of instance masks.

    Args:
        masks: List of binary segmentation masks (H x W)
        gsd: Ground Sample Distance in metres/pixel
    Returns:
        List of dicts with area_px, area_m2, diameter_m
    """
    results = []
    for i, mask in enumerate(masks):
        polygon = mask_to_polygon(mask)
        if polygon is None or len(polygon) < 3:
            continue

        area_px = shoelace_area(polygon)
        area_m2 = pixels_to_metres(area_px, gsd)
        diameter_m = estimate_diameter(area_m2)

        results.append({
            "instance_id": i,
            "area_pixels": round(area_px, 2),
            "area_m2": round(area_m2, 4),
            "diameter_m": round(diameter_m, 3),
            "gsd_used": gsd
        })

    return results


def batch_process(mask_dir: str, gsd: float, output_csv: str = "canopy_results.csv"):
    """Process all mask files in a directory and export to CSV."""
    mask_path = Path(mask_dir)
    all_results = []

    for mask_file in sorted(mask_path.glob("*.png")):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Threshold to binary
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        results = compute_canopy_biometry([binary], gsd)
        for r in results:
            r["source_file"] = mask_file.name
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} canopy measurements -> {output_csv}")

    print(f"\nSummary:")
    print(f"  Mean canopy area: {df['area_m2'].mean():.4f} m²")
    print(f"  Mean diameter:    {df['diameter_m'].mean():.3f} m")
    print(f"  Total trees:      {len(df)}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Canopy Biometry from Segmentation Masks")
    parser.add_argument("--masks", required=True, help="Directory of binary mask PNG files")
    parser.add_argument("--gsd", type=float, required=True, help="GSD in metres/pixel")
    parser.add_argument("--output", default="canopy_results.csv", help="Output CSV path")
    args = parser.parse_args()

    batch_process(args.masks, args.gsd, args.output)
