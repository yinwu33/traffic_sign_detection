# generate a metadata.json for detectron2 training


from pathlib import Path
import json

from zod import ZodDataset, ZodFrame, ZodFrames
from zod import AnnotationProject

from detectron2.structures import BoxMode
from tqdm import tqdm

DATASET_DIR = "/home/tjhu78u/workspace/traffic_sign_detection/data"
VERSION = "full"


FILTER_FILE = "/home/tjhu78u/workspace/traffic_sign_detection/data/metadata/DE_ts.txt"

HEIGHT = 2168
WIDTH = 3848

ts_classes = [
    "unknown_traffic_sign",
]


def main():
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
            # x1 *= WIDTH
            # x2 *= WIDTH
            # y1 *= HEIGHT
            # y2 *= HEIGHT
            bbox = [int(round(v)) for v in (x1, y1, x2, y2)]

            bbox_mode_val = BoxMode.XYXY_ABS.value
            if ts.traffic_sign_class is None:
                ts_class = "unknown_traffic_sign"
            else:
                ts_class = ts.traffic_sign_class
                if ts_class not in ts_classes:
                    ts_classes.append(ts_class)

            metadata_one["annotations"].append(
                {
                    "bbox": bbox,
                    "bbox_mode": bbox_mode_val,
                    "category_id": 0,
                    # "category_id": ts_classes.index(ts_class),
                }
            )

        metadata_all.append(metadata_one)

    print(f"Totally {len(metadata_all)} images")

    # dump to json
    with open("zod_traffic_sign_de.json", "w") as f:
        json.dump(metadata_all, f, indent=2, ensure_ascii=False)

    with open("label.json", "w") as f:
        json.dump(ts_classes, f, indent=2)


if __name__ == "__main__":
    main()
