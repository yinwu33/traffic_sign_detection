# generate a metadata.json for detectron2 training
# e.g. zod_traffic_sign_de.json
# ! Attention: sign id is force to 0 as a binary traffic sign detection task

from pathlib import Path
import json
import random
from zod import ZodDataset, ZodFrame, ZodFrames
from zod import AnnotationProject

from detectron2.structures import BoxMode
from tqdm import tqdm

DATASET_DIR = "/home/tjhu78u/workspace/traffic_sign_detection/data"
METADATA_DIR = f"/home/tjhu78u/workspace/traffic_sign_detection/data/metadata"
VERSION = "full"

HEIGHT = 2168
WIDTH = 3848

SEED = 42

PARTIAL_SPLIT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

ts_classes = [
    "undefined",
]


def main(
    cc: str,
    min_length: float = 0,
    split: bool = False,
    val_ratio: float = 0.1,
):
    FILTER_FILE = f"{METADATA_DIR}/{cc}_ts.txt"

    zod_dataset = ZodFrames(DATASET_DIR, version=VERSION)

    # build filter list
    filter_file = Path(FILTER_FILE)

    with open(filter_file, "r") as f:
        filter_list = [line.strip() for line in f.readlines()]
    print(f"Loading {filter_file}\nKeep {len(filter_list)} driving logs")

    metadata_all = []
    for log_id in tqdm(filter_list, total=len(filter_list)):

        log = zod_dataset[log_id]
        log_dir = Path(DATASET_DIR) / "single_frames" / log_id
        img_dir = log_dir / "camera_front_blur"
        img_fpath = list(img_dir.glob("*.jpg"))[0]
        img_abs_fpath = str(img_fpath.resolve())

        ts_annos = log.get_annotation(
            AnnotationProject.TRAFFIC_SIGNS
        )  # traffic sign list

        metadata_one = {
            "file_name": img_abs_fpath,
            "height": HEIGHT,
            "width": WIDTH,
            "image_id": log_id,
            "annotations": [],
        }

        for ts in ts_annos:
            bbox = ts.bounding_box.xyxy
            # convert bbox to pixel coordinates (ints), handling normalized boxes
            x1, y1, x2, y2 = [x for x in bbox]
            bbox = [int(round(v)) for v in (x1, y1, x2, y2)]

            # filter very small traffic signs
            if min_length > 0:
                w = abs(bbox[2] - bbox[0])
                h = abs(bbox[3] - bbox[1])
                if min(w, h) < min_length:
                    continue  # skip small bbox

            bbox_mode_val = BoxMode.XYXY_ABS.value
            if ts.traffic_sign_class is None:
                ts_class = "unknown"
            else:
                ts_class = ts.traffic_sign_class
                if ts_class not in ts_classes:
                    ts_classes.append(ts_class)

            metadata_one["annotations"].append(
                {
                    "bbox": bbox,
                    "bbox_mode": bbox_mode_val,
                    # ! Attention: force id to 0
                    "category_id": 0,
                    # "category_id": ts_classes.index(ts_class),
                }
            )

        metadata_all.append(metadata_one)

    print(f"Totally {len(metadata_all)} images")

    # dump to json
    with open(f"zod_traffic_sign_{cc}.json", "w") as f:
        json.dump(metadata_all, f)

    with open(f"statistic/label_{cc}.json", "w") as f:
        json.dump(ts_classes, f, indent=2)
        
    if split:
        random.seed(SEED)
        random.shuffle(metadata_all)
        val_size = int(len(metadata_all) * val_ratio)
        train_data = metadata_all[val_size:]
        val_data = metadata_all[:val_size]

        with open(f"zod_traffic_sign_{cc}_train.json", "w") as f:
            json.dump(train_data, f)
        with open(f"zod_traffic_sign_{cc}_val.json", "w") as f:
            json.dump(val_data, f)

        print(
            f"Split dataset into train ({len(train_data)}) / val ({len(val_data)}) with ratio {val_ratio}"
        )
        
        # split train set with partial split
        for partial_split_ratio in PARTIAL_SPLIT:
            partial_size = int(len(train_data) * partial_split_ratio)
            partial_data = train_data[:partial_size]
            with open(f"zod_traffic_sign_{cc}_train_{int(partial_split_ratio*100)}.json", "w") as f:
                json.dump(partial_data, f)
            print(
                f"Create partial train set with ratio {partial_split_ratio}: {len(partial_data)} images"
            )
        


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cc",
        type=str,
        required=True,
        help="Country code for traffic signs (e.g., DE, US)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=15,  # filter out small bboxes by default
        help="Minimum bbox side length to keep",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Whether to split the dataset into train/val",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio when splitting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        cc=args.cc,
        min_length=args.min_length,
        split=args.split,
        val_ratio=args.val_ratio,
    )
