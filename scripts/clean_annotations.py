#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Remove small bounding boxes from metadata.json")
    p.add_argument("--input", type=Path, required=True, help="Path to input metadata.json")
    p.add_argument("--output", type=Path, default=None, help="Path to save cleaned JSON (default: input_cleaned.json)")
    p.add_argument("--min-length", type=float, default=15.0, help="Minimum bbox side length to keep")
    return p.parse_args()

def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_name(input_path.stem + "_cleaned.json")
    min_len = args.min_length

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a list at the root of metadata.json")

    cleaned = []
    total_removed = 0
    total_boxes = 0

    for item in data:
        anns = item.get("annotations", [])
        new_anns = []
        for ann in anns:
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            total_boxes += 1
            if min(w, h) >= min_len:
                new_anns.append(ann)
            else:
                total_removed += 1
        item["annotations"] = new_anns
        cleaned.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned metadata saved to: {output_path}")
    print(f"Total boxes: {total_boxes}, Removed: {total_removed}, Kept: {total_boxes - total_removed}")

if __name__ == "__main__":
    main()
