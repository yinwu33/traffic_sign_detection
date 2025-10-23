#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create per-class image patches to visualize domain shift across countries.

- Randomly sample logs from statistic.csv (has_ts_anno==1).
- For each sampled log, load its single front image and traffic-sign annotations.
- Extract/crop bbox patches ONLY for TARGET_TS_CLASS.
- Save crops into a directory named after the traffic-sign label (e.g., "unknown_traffic_sign/").
- Keep log_id in the filename, and record country in metadata.csv.

Example:
    python make_domain_shift_gallery.py \
        --dataset_dir /home/tjhu78u/workspace/traffic_sign_detection/data \
        --version full \
        --stats_csv /home/tjhu78u/workspace/traffic_sign_detection/data/metadata/statistic.csv \
        --target_ts_class unknown_traffic_sign \
        --sample_size 1000 \
        --height 2168 --width 3848 \
        --out_dir ./domain_shift_output \
        --max_per_log 3 \
        --seed 42
"""

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from zod import ZodFrames, AnnotationProject

ZOD_DATASET_ROOT = "/home/tjhu78u/workspace/traffic_sign_detection/data"
STATS_CSV = "/home/tjhu78u/workspace/traffic_sign_detection/data/metadata/statistic.csv"
# TARGET_TS_CLASS = "Warning_GenericWarning"
# TARGET_TS_CLASS = "Priority_GiveWay" 
# TARGET_TS_CLASS = "Priority_Stop" 
TARGET_TS_CLASS = "Prohibitory_MaximumSpeedLimit50Begin" 
SAMPLE_SIZE = 1000


def load_image_path(dataset_dir: Path, log_id: str) -> Path:
    """Replicates your access pattern to the single front image."""
    img_dir = dataset_dir / "single_frames" / log_id / "camera_front_blur"
    jpgs = list(img_dir.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No jpg found for log_id={log_id} in {img_dir}")
    return jpgs[0].resolve()


def to_pixel_bbox(x1, y1, x2, y2, width, height):
    """
    Convert to integer pixel bbox.
    If boxes are already absolute, this keeps them as is (rounded).
    If they appear normalized (0..1 with x2<=1.01, y2<=1.01), multiply by W/H.
    Finally, clamp to image bounds.
    """
    # Heuristic: if coords look normalized, scale them.
    maybe_norm = (0.0 <= x1 <= 1.01) and (0.0 <= x2 <= 1.01) and (0.0 <= y1 <= 1.01) and (0.0 <= y2 <= 1.01)
    if maybe_norm:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x1, y1, x2, y2 = [int(round(v)) for v in (x1, y1, x2, y2)]

    # Ensure x1<=x2, y1<=y2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Clamp to image boundaries
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    # If bbox collapses, expand minimally where possible
    if x2 == x1 and x2 < width - 1:
        x2 += 1
    if y2 == y1 and y2 < height - 1:
        y2 += 1

    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default=ZOD_DATASET_ROOT, help="ZOD dataset root (same as your script)")
    ap.add_argument("--version", type=str, default="full", help="ZOD version (e.g., full)")
    ap.add_argument("--stats_csv", type=str, default=STATS_CSV, help="Path to statistic.csv with country info")
    ap.add_argument("--target_ts_class", type=str, default=TARGET_TS_CLASS, help="Traffic sign class to extract (folder will use this name)")
    ap.add_argument("--sample_size", type=int, default=SAMPLE_SIZE, help="Random sample size from statistics (has_ts_anno==1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--height", type=int, default=2168, help="Image height (pixels)")
    ap.add_argument("--width", type=int, default=3848, help="Image width (pixels)")
    ap.add_argument("--out_dir", type=str, default="./domain_shift_output", help="Output root directory")
    ap.add_argument("--max_per_log", type=int, default=3, help="Max patches to save per log (if multiple instances exist)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_root = Path(args.out_dir)
    out_class_dir = out_root / args.target_ts_class
    out_class_dir.mkdir(parents=True, exist_ok=True)

    # read stats and sample
    df = pd.read_csv(args.stats_csv)
    df = df[df["has_ts_anno"] == 1]
    if len(df) == 0:
        print("No rows in statistic.csv with has_ts_anno==1. Exiting.")
        return

    # Ensure deterministic sampling
    if args.sample_size < len(df):
        df_sampled = df.sample(n=args.sample_size, random_state=args.seed)
    else:
        df_sampled = df

    # Build mapping: log_id -> country
    log_to_country = {f"{int(row.log_id):06d}": row.country_cc for _, row in df_sampled.iterrows()}

    # init ZOD
    zod_frames = ZodFrames(str(dataset_dir), version=args.version)

    # Prepare metadata rows
    metadata_rows = []
    total_saved = 0
    missing_images = 0
    missing_annos = 0

    print(f"Target class: {args.target_ts_class}")
    print(f"Sampling {len(log_to_country)} logs from {args.stats_csv}")

    for log_id, country in tqdm(log_to_country.items(), total=len(log_to_country)):
        # fetch image path
        try:
            img_path = load_image_path(dataset_dir, log_id)
        except FileNotFoundError:
            missing_images += 1
            continue

        # load image once per log
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open image for log {log_id}: {e}")
            continue

        # get annotations
        try:
            frame = zod_frames[log_id]
            ts_annos = frame.get_annotation(AnnotationProject.TRAFFIC_SIGNS)
        except Exception as e:
            missing_annos += 1
            # Skip if no annotations accessible
            continue

        # filter by target class
        # ts.traffic_sign_class can be None; keep exact match
        found = 0
        for idx, ts in enumerate(ts_annos):
            ts_class = ts.traffic_sign_class if ts.traffic_sign_class is not None else "unknown"
            if ts_class != args.target_ts_class:
                continue

            x1, y1, x2, y2 = ts.bounding_box.xyxy
            bx1, by1, bx2, by2 = to_pixel_bbox(x1, y1, x2, y2, args.width, args.height)

            # extra safety clamp to actual image size (in case H/W mismatch)
            W_img, H_img = img.size
            bx1 = max(0, min(bx1, W_img - 1))
            bx2 = max(0, min(bx2, W_img - 1))
            by1 = max(0, min(by1, H_img - 1))
            by2 = max(0, min(by2, H_img - 1))
            if bx2 <= bx1 or by2 <= by1:
                continue

            crop = img.crop((bx1, by1, bx2, by2))
            out_name = f"{log_id}_{found}_{country}.jpg"
            out_path = out_class_dir / out_name
            try:
                crop.save(out_path, quality=95)
            except Exception as e:
                print(f"[WARN] Failed to save {out_path}: {e}")
                continue

            metadata_rows.append({
                "filename": out_name,
                "log_id": log_id,
                "country_cc": country,
                "bbox_x1": bx1,
                "bbox_y1": by1,
                "bbox_x2": bx2,
                "bbox_y2": by2,
                "source_image": str(img_path),
            })
            total_saved += 1
            found += 1
            if found >= args.max_per_log:
                break

    # Write metadata.csv for this class
    meta_csv_path = out_class_dir / "metadata.csv"
    if metadata_rows:
        df_meta = pd.DataFrame(metadata_rows)
        df_meta.to_csv(meta_csv_path, index=False)
    else:
        print("No patches saved; metadata.csv will not be created.")

    # Also write a small summary.json
    summary = {
        "target_ts_class": args.target_ts_class,
        "total_logs_scanned": len(log_to_country),
        "total_patches_saved": total_saved,
        "missing_images": missing_images,
        "missing_annos": missing_annos,
        "output_dir": str(out_class_dir.resolve()),
        "notes": "Each image is a cropped bbox of the target class; filename format: {log_id}_{idx}_{country}.jpg",
    }
    with open(out_class_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {total_saved} crops to: {out_class_dir}")
    if metadata_rows:
        print(f"Metadata: {meta_csv_path}")
    if missing_images or missing_annos:
        print(f"Missing images: {missing_images}, missing annos: {missing_annos}")


if __name__ == "__main__":
    main()
