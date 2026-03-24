"""
Generative Tiling Algorithm
Simulates multi-altitude UAV imagery by mathematically
resizing tiles to target GSD levels.

Author: Vaidya Sai Teja
Thesis: AIT Master of Engineering, 2026
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


GSD_LEVELS = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.17, 0.20]
BASE_GSD = 0.03  # Source imagery GSD in metres/pixel
TILE_SIZE = 640  # YOLO input size in pixels


def gsd_scale_factor(source_gsd: float, target_gsd: float) -> float:
    """
    Calculate pixel scale factor between two GSD levels.
    Downsampling = target_gsd > source_gsd (higher altitude simulation)
    """
    return source_gsd / target_gsd


def tile_image(image: np.ndarray, tile_size: int = TILE_SIZE) -> List[np.ndarray]:
    """Extract non-overlapping tiles from image."""
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
    return tiles


def simulate_gsd(tile: np.ndarray, source_gsd: float, target_gsd: float,
                 output_size: int = TILE_SIZE) -> np.ndarray:
    """
    Simulate a target GSD by downsampling and upsampling a tile.

    Method:
      1. Downsample: compress tile by scale factor
      2. Upsample: resize back to output_size

    This simulates the loss of spatial resolution at higher altitudes.
    """
    scale = gsd_scale_factor(source_gsd, target_gsd)

    if scale >= 1.0:
        # Already at or coarser resolution
        return cv2.resize(tile, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Step 1: Downsample to simulate higher altitude
    small_size = max(1, int(output_size * scale))
    downsampled = cv2.resize(tile, (small_size, small_size), interpolation=cv2.INTER_AREA)

    # Step 2: Upsample back to standard size
    simulated = cv2.resize(downsampled, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return simulated


def generate_multi_gsd_dataset(
    source_dir: str,
    output_dir: str,
    source_gsd: float = BASE_GSD,
    gsd_levels: List[float] = None
) -> dict:
    """
    Generate tiles at all target GSD levels from source imagery.

    Args:
        source_dir: Path to source UAV imagery (at source_gsd)
        output_dir: Root output directory
        source_gsd: GSD of source imagery (default: 0.03m)
        gsd_levels: List of target GSD values to simulate

    Returns:
        Dictionary with count of tiles generated per GSD level
    """
    if gsd_levels is None:
        gsd_levels = GSD_LEVELS

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    stats = {}

    for gsd in gsd_levels:
        gsd_dir = output_path / f"gsd_{gsd:.2f}m"
        gsd_dir.mkdir(parents=True, exist_ok=True)
        tile_count = 0

        for img_path in source_path.glob("*.jpg"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            tiles = tile_image(image, TILE_SIZE)

            for idx, tile in enumerate(tiles):
                simulated = simulate_gsd(tile, source_gsd, gsd)
                out_name = f"{img_path.stem}_tile{idx:04d}.jpg"
                cv2.imwrite(str(gsd_dir / out_name), simulated)
                tile_count += 1

        stats[f"gsd_{gsd:.2f}m"] = tile_count
        print(f"GSD {gsd:.2f}m: {tile_count} tiles generated -> {gsd_dir}")

    return stats


def scale_annotations(annotations: list, source_gsd: float, target_gsd: float) -> list:
    """
    Scale YOLO polygon annotations for a new GSD level.
    Normalized coordinates [0,1] are GSD-invariant — no scaling needed.
    This function is provided for absolute coordinate formats.
    """
    scale = gsd_scale_factor(source_gsd, target_gsd)
    scaled = []
    for ann in annotations:
        scaled_ann = [coord * scale for coord in ann]
        scaled.append(scaled_ann)
    return scaled


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generative Tiling Algorithm — Multi-GSD Dataset")
    parser.add_argument("--source", required=True, help="Source imagery directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--source-gsd", type=float, default=0.03, help="Source GSD in metres")
    args = parser.parse_args()

    stats = generate_multi_gsd_dataset(args.source, args.output, args.source_gsd)
    print("\nTiles generated per GSD level:")
    for k, v in stats.items():
        print(f"  {k}: {v} tiles")
