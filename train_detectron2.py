# train_detectron_from_myjson.py
import json, os, random, math
import cv2
import torch
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def load_myjson(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset_dicts = []
    for i, item in enumerate(data):
        record = {
            "file_name": item["file_name"],
            "image_id": item.get("image_id", i),
            "height": item.get("height", None),
            "width": item.get("width", None),
            "annotations": [],
        }

        for ann in item.get("annotations", []):
            # 如果你的 bbox 已经是 [x1, y1, x2, y2]，且 bbox_mode=0，直接用
            bbox_mode = ann.get("bbox_mode", 0)
            if bbox_mode != 0:
                # 若存在不一致，可在此做转换；这里假设都是 XYXY_ABS
                pass

            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYXY_ABS,  # 等同于 0
                "category_id": ann["category_id"],
                # 如果将来需要分割、多边形、crowd等，可在这里扩展字段
                "iscrowd": ann.get("iscrowd", 0),
            }
            record["annotations"].append(obj)

        dataset_dicts.append(record)
    return dataset_dicts


def register_my_dataset(name, json_path, thing_classes):
    DatasetCatalog.register(name, lambda p=json_path: load_myjson(p))
    MetadataCatalog.get(name).set(
        json_file=json_path,
        evaluator_type="coco",  # 只有 bbox 时也可以用 coco evaluator 评估 AP
        thing_classes=thing_classes,
    )


def split_and_save(json_in, train_out, val_out, val_ratio=0.1, seed=42):
    random.seed(seed)
    with open(json_in, "r") as f:
        data = json.load(f)
    random.shuffle(data)
    n_val = max(1, int(len(data) * val_ratio))
    val = data[:n_val]
    train = data[n_val:]
    with open(train_out, "w") as f:
        json.dump(train, f)
    with open(val_out, "w") as f:
        json.dump(val, f)


# 继续写在 train_detectron_from_myjson.py
import os
from detectron2.engine import DefaultTrainer, DefaultPredictor, hooks
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image


class VisualizationHook(hooks.HookBase):
    """
    Dump a visualization of model predictions at the end of each epoch.
    """

    def __init__(
        self,
        cfg,
        dataset_name,
        output_dir,
        period,
        samples_per_epoch=1,
        seed=42,
    ):
        self.cfg = cfg.clone()
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(output_dir, "visualizations")
        self.period = max(1, period)
        self.samples_per_epoch = max(1, samples_per_epoch)
        self.dataset_dicts = DatasetCatalog.get(dataset_name)
        self.metadata = MetadataCatalog.get(dataset_name)
        self.rng = random.Random(seed)
        self.predictor = None

    def _get_predictor(self):
        if self.predictor is None:
            predictor_cfg = self.cfg.clone()
            predictor_cfg.MODEL.WEIGHTS = ""  # use current training weights
            self.predictor = DefaultPredictor(predictor_cfg)
        return self.predictor

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self.period != 0:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        predictor = self._get_predictor()

        epoch_idx = next_iter // self.period
        with torch.no_grad():
            predictor.model.load_state_dict(self.trainer.model.state_dict())
            for sample_idx in range(self.samples_per_epoch):
                sample = self.rng.choice(self.dataset_dicts)
                image = read_image(sample["file_name"], format="BGR")
                outputs = predictor(image)
                instances = outputs["instances"].to("cpu")

                visualizer = Visualizer(
                    image[:, :, ::-1],
                    metadata=self.metadata,
                    scale=1.0,
                )
                vis_image = visualizer.draw_instance_predictions(instances).get_image()
                file_name = f"epoch_{epoch_idx:04d}_sample_{sample_idx:02d}.jpg"
                save_path = os.path.join(self.output_dir, file_name)
                cv2.imwrite(save_path, vis_image[:, :, ::-1])


def train(
    train_json,
    val_json,
    output_dir="./output",
    thing_classes=("object",),  # 替换为你的类别列表
    ims_per_batch=32,
    base_lr=0.00025,
    max_iter=30000,
    num_workers=8,
):
    # 注册两个数据集
    register_my_dataset("my_train", train_json, list(thing_classes))
    register_my_dataset("my_val", val_json, list(thing_classes))

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ("my_val",)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # 简单起见，先不做学习率衰减
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)  # 重要：与你的类别数匹配
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    train_dataset_size = len(DatasetCatalog.get("my_train"))
    iterations_per_epoch = max(
        1, math.ceil(train_dataset_size / cfg.SOLVER.IMS_PER_BATCH)
    )

    # 开始训练
    trainer = DefaultTrainer(cfg)
    trainer.register_hooks(
        [
            VisualizationHook(
                cfg=cfg,
                dataset_name="my_val",
                output_dir=cfg.OUTPUT_DIR,
                period=iterations_per_epoch,
                samples_per_epoch=1,
            )
        ]
    )
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    # 例子：先把一个总 JSON 切分为 train/val
    total_json = "zod_traffic_sign_de.json"
    train_json = "zod_traffic_sign_de_train.json"
    val_json = "zod_traffic_sign_de_val.json"

    if not (os.path.exists(train_json) and os.path.exists(val_json)):
        split_and_save(total_json, train_json, val_json, val_ratio=0.1)

    # 类别名请自行替换（顺序需与 category_id 对应）
    thing_classes = ("unknown_traffic_sign",)  # 例如只有1类，就写一个名字

    train(
        train_json=train_json,
        val_json=val_json,
        output_dir="./output_frcnn_r50",
        thing_classes=thing_classes,
        ims_per_batch=8,
        base_lr=0.00025,
        max_iter=10000,
        num_workers=4,
    )
