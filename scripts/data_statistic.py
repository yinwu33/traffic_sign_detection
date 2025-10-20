# Create a Python script that analyzes ZOD-style bbox annotations and produces summary stats + plots.
import os, json, math, textwrap, statistics as stats
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Analyze bbox annotations from a ZOD-style JSON file.

Input format (list of images), each item like:
{
  "file_name": ".../image.jpg",
  "height": 2168,
  "width": 3848,
  "image_id": "000090",
  "annotations": [
    {
      "bbox": [x1, y1, x2, y2],
      "bbox_mode": 0,
      "category_id": 0
    },
    ...
  ]
}

What this script does:
1) Loads the JSON file.
2) Builds a per-annotation dataframe with:
   - image_id, img_w, img_h
   - x1, y1, x2, y2
   - w, h, area, aspect_ratio
   - cx, cy (pixel center)
   - cx_n, cy_n (normalized center in [0,1])
   - category_id
3) Prints summary stats and saves them to CSV.
4) Computes distribution plots:
   - Histogram of bbox widths
   - Histogram of bbox heights
   - Histogram of aspect ratios
   - 2D heatmap of bbox center density (normalized coordinates)
5) Finds "concentrated regions":
   - Based on a 2D histogram over (cx_n, cy_n) with configurable bins;
     reports the top-k densest cells as rough "hotspots".
   - Additionally, runs DBSCAN clustering on centers to propose clusters.

Usage:
    python analyze_bboxes.py --json /path/to/zod_traffic_sign_de.json --out out_dir

Optional args:
    --bins  : number of bins per axis for 2D histogram (default: 50)
    --topk  : number of densest cells to report (default: 10)
    --eps   : DBSCAN eps in normalized units (default: 0.02)
    --min_samples : DBSCAN min samples (default: 15)

Outputs (in --out directory):
    - bbox_annotations.csv          (per-annotation table)
    - summary_stats.json            (scalar statistics)
    - hist_width.png                (histogram)
    - hist_height.png               (histogram)
    - hist_aspect_ratio.png         (histogram)
    - heatmap_centers_2d.png        (2D histogram heatmap)
    - clusters_dbscan.csv           (cluster membership per annotation, -1 = noise)
    - hotspots_topk.csv             (top-k densest histogram cells with bounds and counts)
"""

import os
import json
import argparse
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


def load_annotations(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at top level.")
    return data


def build_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        img_w = int(item.get("width", 0))
        img_h = int(item.get("height", 0))
        image_id = item.get("image_id", None)

        for ann in item.get("annotations", []):
            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]

            # normalize ordering just in case
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w == 0.0 or h == 0.0:
                # skip zero-area boxes
                continue
            area = w * h

            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            # Avoid division by zero; if missing dims, set to NaN
            cx_n = cx / img_w if img_w > 0 else float("nan")
            cy_n = cy / img_h if img_h > 0 else float("nan")
            aspect = w / h if h > 0 else float("inf")

            rows.append(
                {
                    "image_id": image_id,
                    "img_w": img_w,
                    "img_h": img_h,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": w,
                    "h": h,
                    "area": area,
                    "aspect_ratio": aspect,
                    "cx": cx,
                    "cy": cy,
                    "cx_n": cx_n,
                    "cy_n": cy_n,
                    "category_id": ann.get("category_id", None),
                    "bbox_mode": ann.get("bbox_mode", None),
                    "file_name": item.get("file_name", None),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "image_id",
                "img_w",
                "img_h",
                "x1",
                "y1",
                "x2",
                "y2",
                "w",
                "h",
                "area",
                "aspect_ratio",
                "cx",
                "cy",
                "cx_n",
                "cy_n",
                "category_id",
                "bbox_mode",
                "file_name",
            ]
        )
    return pd.DataFrame(rows)


def basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    def series_stats(s: pd.Series) -> Dict[str, Any]:
        s = s.dropna()
        if len(s) == 0:
            return {}
        return {
            "count": int(s.count()),
            "min": float(s.min()),
            "p5": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        }

    out = {
        "num_images": int(df["image_id"].nunique()) if len(df) else 0,
        "num_ann": int(len(df)),
        "w_stats": series_stats(df["w"]) if "w" in df else {},
        "h_stats": series_stats(df["h"]) if "h" in df else {},
        "area_stats": series_stats(df["area"]) if "area" in df else {},
        "aspect_ratio_stats": (
            series_stats(df["aspect_ratio"]) if "aspect_ratio" in df else {}
        ),
        "center_x_norm_stats": series_stats(df["cx_n"]) if "cx_n" in df else {},
        "center_y_norm_stats": series_stats(df["cy_n"]) if "cy_n" in df else {},
        "categories": (
            df["category_id"].value_counts(dropna=False).to_dict()
            if "category_id" in df
            else {}
        ),
    }
    return out


def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def save_dataframe(df: pd.DataFrame, out_dir: str):
    out_csv = os.path.join(out_dir, "bbox_annotations.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")


def save_summary_json(summary: Dict[str, Any], out_dir: str):
    out_json = os.path.join(out_dir, "summary_stats.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {out_json}")


def plot_histogram(
    series: pd.Series, title: str, out_path: str, bins: int = 50, log: bool = False
):
    plt.figure()
    data = series.replace([np.inf, -np.inf], np.nan).dropna().values
    if len(data) == 0:
        print(f"[Warn] No data for plot: {title}")
        return
    plt.hist(data, bins=bins, log=log)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] {out_path}")


def plot_heatmap_2d(
    x: pd.Series, y: pd.Series, title: str, out_path: str, bins: int = 50
):
    plt.figure()
    xv = x.replace([np.inf, -np.inf], np.nan).dropna().values
    yv = y.replace([np.inf, -np.inf], np.nan).dropna().values
    if len(xv) == 0 or len(yv) == 0:
        print(f"[Warn] No data for heatmap: {title}")
        return
    plt.hist2d(xv, yv, bins=bins, range=[[0, 1], [0, 1]])
    plt.title(title)
    plt.xlabel("cx_n (0-1)")
    plt.ylabel("cy_n (0-1)")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] {out_path}")


def find_hotspots(df: pd.DataFrame, bins: int, topk: int, out_dir: str) -> pd.DataFrame:
    xv = df["cx_n"].replace([np.inf, -np.inf], np.nan).dropna().values
    yv = df["cy_n"].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(xv) == 0 or len(yv) == 0:
        print("[Warn] No centers for hotspot detection.")
        return pd.DataFrame(
            columns=["rank", "count", "x_low", "x_high", "y_low", "y_high"]
        )

    H, xedges, yedges = np.histogram2d(xv, yv, bins=bins, range=[[0, 1], [0, 1]])
    # Flatten
    flat = H.flatten()
    idx_sorted = np.argsort(flat)[::-1]
    records = []
    for rank, idx in enumerate(idx_sorted[:topk], start=1):
        count = int(flat[idx])
        xi = idx // bins
        yi = idx % bins
        x_low, x_high = xedges[xi], xedges[xi + 1]
        y_low, y_high = yedges[yi], yedges[yi + 1]
        records.append(
            {
                "rank": rank,
                "count": count,
                "x_low": float(x_low),
                "x_high": float(x_high),
                "y_low": float(y_low),
                "y_high": float(y_high),
            }
        )
    hotspots_df = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, "hotspots_topk.csv")
    hotspots_df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")
    return hotspots_df


def cluster_dbscan(
    df: pd.DataFrame, eps: float, min_samples: int, out_dir: str
) -> pd.DataFrame:
    centers = df[["cx_n", "cy_n"]].replace([np.inf, -np.inf], np.nan).dropna()
    if centers.empty:
        print("[Warn] No centers for DBSCAN clustering.")
        labeled = df.copy()
        labeled["cluster"] = -1
        return labeled

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(centers.values)
    labeled = df.loc[centers.index].copy()
    labeled["cluster"] = labels

    out_csv = os.path.join(out_dir, "clusters_dbscan.csv")
    labeled.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    # Print simple cluster summary
    unique, counts = np.unique(labels, return_counts=True)
    summary = {int(k): int(v) for k, v in zip(unique, counts)}
    print("[DBSCAN] cluster label -> count:", summary)
    return labeled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to ZOD-style annotations JSON")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--bins", type=int, default=50, help="Bins per axis for 2D histogram/heatmap"
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k densest 2D histogram cells to report",
    )
    ap.add_argument(
        "--eps", type=float, default=0.02, help="DBSCAN eps in normalized units (0-1)"
    )
    ap.add_argument("--min_samples", type=int, default=15, help="DBSCAN min_samples")
    ap.add_argument("--hist_bins", type=int, default=60, help="Bins for 1D histograms")
    args = ap.parse_args()

    ensure_out_dir(args.out)

    print("[Info] Loading:", args.json)
    data = load_annotations(args.json)

    print("[Info] Building dataframe ...")
    df = build_dataframe(data)
    if df.empty:
        print("[Error] No valid annotations found. Exiting.")
        return

    # Save table
    save_dataframe(df, args.out)

    # Summary stats
    print("[Info] Computing summary ...")
    summary = basic_stats(df)
    save_summary_json(summary, args.out)

    # Histograms
    print("[Info] Plotting histograms ...")
    plot_histogram(
        df["w"],
        "bbox width (px)",
        os.path.join(args.out, "hist_width.png"),
        bins=args.hist_bins,
    )
    plot_histogram(
        df["h"],
        "bbox height (px)",
        os.path.join(args.out, "hist_height.png"),
        bins=args.hist_bins,
    )
    # Aspect ratio can be skewed; log-scale y can help but we keep y linear per guidelines.
    plot_histogram(
        df["aspect_ratio"],
        "aspect ratio (w/h)",
        os.path.join(args.out, "hist_aspect_ratio.png"),
        bins=args.hist_bins,
    )

    # Heatmap of centers
    print("[Info] Plotting 2D center heatmap ...")
    plot_heatmap_2d(
        df["cx_n"],
        df["cy_n"],
        "bbox center density (normalized)",
        os.path.join(args.out, "heatmap_centers_2d.png"),
        bins=args.bins,
    )

    # Hotspots
    print("[Info] Finding hotspots ...")
    hotspots_df = find_hotspots(df, bins=args.bins, topk=args.topk, out_dir=args.out)

    # DBSCAN clustering
    print("[Info] Running DBSCAN clustering on centers ...")
    clustered = cluster_dbscan(
        df, eps=args.eps, min_samples=args.min_samples, out_dir=args.out
    )

    # Save a short textual report
    report_path = os.path.join(args.out, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ZOD BBox Analysis Report\n")
        f.write("========================\n\n")
        f.write(f"Total images: {summary['num_images']}\n")
        f.write(f"Total annotations: {summary['num_ann']}\n\n")

        def fmt_stats(name, st):
            if not st:
                return
            f.write(f"{name}:\n")
            for k in [
                "min",
                "p5",
                "p25",
                "median",
                "mean",
                "p75",
                "p95",
                "max",
                "count",
            ]:
                if k in st:
                    f.write(f"  {k:>7}: {st[k]:.4f}\n")
            f.write("\n")

        fmt_stats("Width (px)", summary.get("w_stats", {}))
        fmt_stats("Height (px)", summary.get("h_stats", {}))
        fmt_stats("Area (px^2)", summary.get("area_stats", {}))
        fmt_stats("Aspect ratio (w/h)", summary.get("aspect_ratio_stats", {}))
        fmt_stats("Center X (normalized)", summary.get("center_x_norm_stats", {}))
        fmt_stats("Center Y (normalized)", summary.get("center_y_norm_stats", {}))

        if hotspots_df is not None and not hotspots_df.empty:
            f.write("Top-k densest center bins (normalized [0,1]):\n")
            f.write(hotspots_df.to_string(index=False))
            f.write("\n\n")

        # Cluster summary
        if "cluster" in clustered.columns:
            cluster_counts = (
                clustered["cluster"].value_counts().sort_values(ascending=False)
            )
            f.write("DBSCAN cluster counts (label -> count):\n")
            f.write(cluster_counts.to_string())
            f.write("\n")

    print(f"[Saved] {report_path}")
    print("[Done] All outputs are in:", args.out)


if __name__ == "__main__":
    main()
