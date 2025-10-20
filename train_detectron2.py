# train_detectron_from_myjson.py
import json, os, random, math
import cv2
import torch
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


OUTPUT_DIR = "./output"
BATCH_SIZE = 6
LABELS = [
    "ts",
]
NUM_EPOCHS = 20
LR = 1e-5
NUM_WORKERS = 14

MODEL_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


@lru_cache(maxsize=None)
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


def split_and_save(
    json_in,
    train_out,
    val_out,
    val_ratio=0.1,
    seed=42,
):
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

                # Also save ground truth visualization for comparison
                gt_visualizer = Visualizer(
                    image[:, :, ::-1],
                    metadata=self.metadata,
                    scale=1.0,
                )
                gt_image = gt_visualizer.draw_dataset_dict(sample).get_image()
                gt_file_name = f"epoch_{epoch_idx:04d}_sample_{sample_idx:02d}_gt.jpg"
                gt_save_path = os.path.join(self.output_dir, gt_file_name)
                cv2.imwrite(gt_save_path, gt_image[:, :, ::-1])


class EvalWriterHook(hooks.EvalHook):
    """
    Extend EvalHook to dump validation metrics to a text file every evaluation.
    """

    def __init__(self, eval_period, eval_function, output_file):
        super().__init__(eval_period, eval_function)
        self.output_file = output_file

    def _to_serializable(self, value):
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def _do_eval(self):
        results = super()._do_eval()
        if not results:
            return results

        epoch = math.ceil((self.trainer.iter + 1) / max(1, self._period))
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "iteration": self.trainer.iter + 1,
            "results": self._to_serializable(results),
        }

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return results


class TrainerWithEvalLogging(DefaultTrainer):
    """
    DefaultTrainer that writes evaluation metrics to disk at each evaluation.
    """

    def __init__(self, cfg, eval_output_file):
        self._eval_output_file = os.fspath(eval_output_file)
        super().__init__(cfg)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        for idx, hook_item in enumerate(hooks_list):
            if isinstance(hook_item, hooks.EvalHook):
                hooks_list[idx] = EvalWriterHook(
                    eval_period=hook_item._period,
                    eval_function=hook_item._func,
                    output_file=self._eval_output_file,
                )
        return hooks_list


def setup_cfg(
    model_config: str,
    output_dir: str,
    thing_classes,
    ims_per_batch=32,
    base_lr=0.00025,
    max_iter=30000,
    num_workers=8,
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    # cfg.INPUT.MIN_SIZE_TRAIN = (0,)
    # cfg.INPUT.MAX_SIZE_TRAIN = 0
    # cfg.INPUT.MIN_SIZE_TEST = 0
    # cfg.INPUT.MAX_SIZE_TEST = 0
    # 尺寸（保持高分辨率输入）
    cfg.INPUT.MIN_SIZE_TRAIN = (2000,)      # 短边≈2000
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = 4000
    cfg.INPUT.MIN_SIZE_TEST = 2000
    cfg.INPUT.MAX_SIZE_TEST = 4000
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.CROP.ENABLED = False          # 小目标不建议裁剪

    # Anchors（小目标友好）
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]

    # RPN（更多候选）
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 4000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST  = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7

    # ROI / Head
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0   # 对齐更准确
    
    cfg.DATASETS.TRAIN = ("zod_train",)
    cfg.DATASETS.TEST = ("zod_val",)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # 简单起见，先不做学习率衰减
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)  # 重要：与你的类别数匹配
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    cfg.OUTPUT_DIR = str(output_dir)
    
    print(cfg)

    return cfg


def train(
    train_json,
    val_json,
    output_dir="./output",
    thing_classes=("object",),  # 替换为你的类别列表
    ims_per_batch=32,
    base_lr=0.00025,
    max_iter=30000,
    num_workers=8,
    model_config: str = MODEL_CONFIG,
):
    # dataset registration
    register_my_dataset("zod_train", train_json, list(thing_classes))
    register_my_dataset("zod_val", val_json, list(thing_classes))

    cfg = setup_cfg(
        model_config=model_config,
        output_dir=output_dir,
        thing_classes=thing_classes,
        ims_per_batch=ims_per_batch,
        base_lr=base_lr,
        max_iter=max_iter,
        num_workers=num_workers,
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    n_train = len(DatasetCatalog.get("zod_train"))
    iterations_per_epoch = max(1, math.ceil(n_train / cfg.SOLVER.IMS_PER_BATCH))
    cfg.TEST.EVAL_PERIOD = iterations_per_epoch

    # start
    eval_metrics_path = os.path.join(cfg.OUTPUT_DIR, "val_metrics.txt")
    trainer = TrainerWithEvalLogging(cfg, eval_metrics_path)
    trainer.register_hooks(
        [
            VisualizationHook(
                cfg=cfg,
                dataset_name="zod_val",
                output_dir=cfg.OUTPUT_DIR,
                period=iterations_per_epoch,
                samples_per_epoch=1,
            )
        ]
    )
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    # split train / val
    # total_json = "zod_traffic_sign_de.json"
    # train_json = "zod_traffic_sign_de_train.json"
    # val_json = "zod_traffic_sign_de_val.json"
    
    total_json = "zod_traffic_sign_de_cleaned.json"
    train_json = "zod_traffic_sign_de_cleaned_train.json"
    val_json = "zod_traffic_sign_de_cleaned_val.json"

    if not (os.path.exists(train_json) and os.path.exists(val_json)):
        split_and_save(total_json, train_json, val_json, val_ratio=0.1)

    output_dir = Path(
        f"./{OUTPUT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_frcnn_r50"
    )

    n_train = len(load_myjson(train_json))
    n_val = len(load_myjson(val_json))

    max_iter = NUM_EPOCHS * math.ceil(n_train / BATCH_SIZE)
    print(f"Training samples: {n_train}, Validation samples: {n_val}")
    print(f"Total training iterations: {max_iter}")

    train(
        train_json=train_json,
        val_json=val_json,
        output_dir=output_dir,
        thing_classes=LABELS,
        ims_per_batch=BATCH_SIZE,
        base_lr=LR,
        max_iter=max_iter,
        num_workers=NUM_WORKERS,
    )
