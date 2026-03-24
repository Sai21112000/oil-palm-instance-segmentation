"""
Teacher Agent Auto-Annotator
Uses a trained YOLOv11l-seg model as a Teacher Agent to 
auto-annotate new UAV imagery for further training.

Author: Vaidya Sai Teja  
Thesis: AIT Master of Engineering, 2026
"""

from ultralytics import YOLO
from pathlib import Path
import json
import numpy as np
import cv2


def run_teacher_agent(
    model_path: str,
    image_dir: str,
    output_dir: str,
    confidence: float = 0.50,
    iou_threshold: float = 0.45
) -> dict:
    """
    Run trained YOLO model as Teacher Agent to auto-annotate images.

    Output format: YOLO segmentation labels (.txt)
    Each line: class_id x1 y1 x2 y2 ... (normalized polygon coords)

    Args:
        model_path: Path to fine-tuned YOLOv11l-seg .pt weights
        image_dir:  Directory of images to annotate
        output_dir: Directory to save label .txt files
        confidence: Detection confidence threshold
        iou_threshold: NMS IoU threshold

    Returns:
        Annotation statistics
    """
    model = YOLO(model_path)
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"images_processed": 0, "total_instances": 0, "skipped": 0}

    for img_file in sorted(image_path.glob("*.jpg")):
        results = model.predict(
            source=str(img_file),
            conf=confidence,
            iou=iou_threshold,
            save=False,
            verbose=False
        )

        label_lines = []

        for result in results:
            if result.masks is None:
                stats["skipped"] += 1
                continue

            for i, mask_xy in enumerate(result.masks.xy):
                # Normalize polygon to [0, 1]
                img_h, img_w = result.orig_shape
                polygon = mask_xy.copy()
                polygon[:, 0] /= img_w  # x
                polygon[:, 1] /= img_h  # y

                coords_flat = polygon.flatten().tolist()
                coords_str = " ".join(f"{c:.6f}" for c in coords_flat)
                label_lines.append(f"0 {coords_str}")
                stats["total_instances"] += 1

        label_file = output_path / (img_file.stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines))

        stats["images_processed"] += 1

    print(f"Teacher Agent complete:")
    print(f"  Images processed: {stats['images_processed']}")
    print(f"  Instances annotated: {stats['total_instances']}")
    print(f"  Mean instances/image: {stats['total_instances']/max(1,stats['images_processed']):.1f}")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Teacher Agent Auto-Annotator")
    parser.add_argument("--model", required=True, help="Path to YOLOv11l-seg .pt weights")
    parser.add_argument("--images", required=True, help="Directory of input images")
    parser.add_argument("--output", required=True, help="Output label directory")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    args = parser.parse_args()

    run_teacher_agent(args.model, args.images, args.output, args.conf)
