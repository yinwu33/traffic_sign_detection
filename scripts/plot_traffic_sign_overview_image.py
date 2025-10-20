#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageOps, ImageDraw, ImageFont

BBox = Tuple[float, float, float, float]  # x1,y1,x2,y2

def parse_args():
    p = argparse.ArgumentParser(description="Build an MxN overview image from a single metadata.json (list of entries).")
    p.add_argument("--metadata", type=Path, required=True, help="Path to metadata.json (a LIST of entries).")
    p.add_argument("--images-root", type=Path, default=None, help="Optional root to prepend to relative file_name paths.")
    p.add_argument("--mxn", type=int, nargs=2, metavar=("M","N"), required=True, help="Grid size (rows M x cols N).")
    p.add_argument("--tile-size", type=int, nargs=2, metavar=("W","H"), default=(96,96), help="Tile size (W H).")
    p.add_argument("--output", type=Path, default=Path("traffic_sign_overview.png"), help="Output PNG path.")
    p.add_argument("--category-ids", type=int, nargs="*", default=None, help="Keep only these category_ids.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle the collected crops before placing.")
    p.add_argument("--bg", type=int, nargs=3, metavar=("R","G","B"), default=(255,255,255), help="Background color.")
    p.add_argument("--strict", action="store_true", help="Fail on malformed records; default: skip with warning.")
    return p.parse_args()

def load_json_list(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("metadata JSON must be a LIST of entries.")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON {path}: {e}")

def clip_bbox_to_img(bb: BBox, w: int, h: int) -> Optional[BBox]:
    x1, y1, x2, y2 = bb
    x1c = max(0.0, min(float(w), x1))
    y1c = max(0.0, min(float(h), y1))
    x2c = max(0.0, min(float(w), x2))
    y2c = max(0.0, min(float(h), y2))
    if x2c <= x1c or y2c <= y1c:
        return None
    return (x1c, y1c, x2c, y2c)

def safe_open_image(img_path: Path):
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[warn] Failed to open image {img_path}: {e}", file=sys.stderr)
        return None

def resolve_image_path(file_name: str, metadata_path: Path, images_root: Optional[Path]) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    if images_root is not None:
        return images_root / p
    # default: relative to JSON file directory
    return metadata_path.parent / p

def collect_crops(args) -> List[Tuple[Image.Image, str]]:
    """Collect crops until M*N is reached, assuming bbox is [x1,y1,x2,y2] and bbox_mode 0 indicates xyxy.
    Returns a list of (tile_image, label) where label is "<w> x <h>" of the original crop size.
    """
    entries = load_json_list(args.metadata)

    m, n = args.mxn
    need = m * n
    tile_w, tile_h = args.tile_size
    crops: List[Tuple[Image.Image, str]] = []

    for entry in entries:
        try:
            img_path = resolve_image_path(entry["file_name"], args.metadata, args.images_root)
            anns = entry.get("annotations", []) or []
        except Exception as e:
            if args.strict:
                raise
            print(f"[warn] Bad entry structure: {e}", file=sys.stderr)
            continue

        im = safe_open_image(img_path)
        if im is None:
            continue
        W, H = im.size

        for ann in anns:
            try:
                # Filter by category_id if requested
                cid = ann.get("category_id", None)
                if args.category_ids is not None and cid not in args.category_ids:
                    continue

                # Expect bbox to already be [x1,y1,x2,y2]; bbox_mode 0 confirms xyxy
                bbox = ann.get("bbox", None)
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    if args.strict:
                        raise ValueError("bbox must be a 4-tuple list [x1,y1,x2,y2]")
                    else:
                        continue

                x1, y1, x2, y2 = map(float, bbox)
                bb = clip_bbox_to_img((x1, y1, x2, y2), W, H)
                if bb is None:
                    continue

                xi1, yi1, xi2, yi2 = map(int, bb)
                crop = im.crop((xi1, yi1, xi2, yi2))
                orig_w, orig_h = crop.size

                # Fit with aspect ratio preserved, padded to tile
                crop_fitted = ImageOps.pad(crop, (tile_w, tile_h), method=Image.BILINEAR, color=tuple(args.bg))
                label = f"{orig_w} x {orig_h}"
                crops.append((crop_fitted, label))

                if len(crops) >= need:
                    return crops
            except Exception as e:
                if args.strict:
                    raise
                print(f"[warn] Bad annotation: {e}", file=sys.stderr)
                continue

    return crops

def assemble_grid(crops: List[Tuple[Image.Image, str]], m: int, n: int, tile_w: int, tile_h: int, bg):
    canvas = Image.new("RGB", (n * tile_w, m * tile_h), color=tuple(bg))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, item in enumerate(crops):
        r = idx // n
        c = idx % n
        if r >= m:
            break
        patch, label = item
        x0 = c * tile_w
        y0 = r * tile_h
        canvas.paste(patch, (x0, y0))
        if label:
            # Compute text size and draw centered at the bottom of the tile with a small margin
            try:
                if hasattr(draw, "textbbox"):
                    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
                else:
                    tw, th = draw.textsize(label, font=font)
            except Exception:
                # Fallback sizes if measurement fails
                tw, th = (len(label) * 6, 10)
            tx = x0 + max(0, (tile_w - tw) // 2)
            ty = y0 + max(0, tile_h - th - 2)
            draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
    return canvas

def main():
    args = parse_args()
    m, n = args.mxn
    tile_w, tile_h = args.tile_size
    need = m * n

    crops = collect_crops(args)
    if len(crops) == 0:
        print("No crops collected. Check paths and filters.", file=sys.stderr)
        sys.exit(2)

    if args.shuffle:
        random.shuffle(crops)

    # Pad or trim to exactly M*N
    if len(crops) < need:
        blanks = need - len(crops)
        blank = Image.new("RGB", (tile_w, tile_h), color=tuple(args.bg))
        crops.extend([(blank, "")] * blanks)
    elif len(crops) > need:
        crops = crops[:need]

    grid = assemble_grid(crops, m, n, tile_w, tile_h, args.bg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output, format="PNG")
    print(f"Saved collage to: {args.output.resolve()}")

if __name__ == "__main__":
    main()