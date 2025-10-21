# train_faster_rcnn.py for traffic sign detection using Detectron2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, math, argparse
from datetime import datetime
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

from tsd.hooks import VisualizationHook
from tsd.datasets.utils import dataset_split, load_dataset_json


OUTPUT_ROOT = "./output"
OUTPUT_DIR = f"{OUTPUT_ROOT}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_frcnn_r50"
BATCH_SIZE = 8  # for amp + 48 GB GPU memory
LABELS = [
    "ts",
]
NUM_EPOCHS = 20
BASE_LR = 1e-5
NUM_WORKERS = 14

DATASET_TRAIN_NAME = "zod_train"
DATASET_VAL_NAME = "zod_val"

MODEL_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def register_my_dataset(
    name: str,
    json_fpath: str,
    labels: list[str],
):
    dataset_dicts = load_dataset_json(json_fpath)  # cache it

    DatasetCatalog.register(name, lambda: dataset_dicts)
    MetadataCatalog.get(name).set(
        json_file=json_fpath,
        evaluator_type="coco",  # 只有 bbox 时也可以用 coco evaluator 评估 AP
        thing_classes=labels,
    )


class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.append(
            VisualizationHook(
                cfg=self.cfg,
                dataset_name=DATASET_TRAIN_NAME,
                output_dir=self.cfg.OUTPUT_DIR,  # TODO: use cfg directly
                period=self.cfg.SOLVER.CHECKPOINT_PERIOD,
                samples_per_epoch=1,
            )
        )
        return hooks_list


def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))

    # * INPUT
    # keep the original image size for small objects
    cfg.INPUT.MIN_SIZE_TRAIN = (0,)
    cfg.INPUT.MAX_SIZE_TRAIN = 0
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 0
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.CROP.ENABLED = False  # 小目标不建议裁剪

    # * Anchors
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 4.0]]

    # RPN: keep more proposals for small objects
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 4000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7

    # ROI / Head: more sampling and pos. ratio
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LABELS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0  # 对齐更准确
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    cfg.DATASETS.TRAIN = ("zod_train",)
    cfg.DATASETS.TEST = ("zod_val",)
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS

    # SOLVER
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE  # for all gpus
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.MAX_ITER = 27000
    cfg.SOLVER.STEPS = []  # 简单起见，先不做学习率衰减
    cfg.SOLVER.AMP.ENABLED = True  # GPU memory savings

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)
    cfg.OUTPUT_DIR = OUTPUT_DIR
    return cfg


def train(
    cfg,
    train_json,
    val_json,
    thing_classes=("object",),  # 替换为你的类别列表
    resume_from=None,
):
    # dataset registration
    register_my_dataset("zod_train", train_json, list(thing_classes))
    register_my_dataset("zod_val", val_json, list(thing_classes))

    if resume_from:
        cfg.MODEL.WEIGHTS = str(resume_from)

    n_train = len(DatasetCatalog.get("zod_train"))
    cfg.TEST.EVAL_PERIOD = max(
        1, math.ceil(n_train / cfg.SOLVER.IMS_PER_BATCH)
    )  # iter per epoch

    # start
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=bool(resume_from))
    trainer.train()


def trace(cfg, resume_from=None):
    if args.resume_from:
        cfg.MODEL.WEIGHTS = resume_from
    cfg.MODEL.DEVICE = "cpu"

    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    try:
        from detectron2.export import TracingAdapter
    except Exception:
        TracingAdapter = None

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    H, W = 1024, 1024
    dummy_img = torch.randn(3, H, W)
    # inputs = [{"image": dummy_img, "height": H, "width": W}]
    inputs = [{"image": dummy_img}]

    if TracingAdapter is not None:
        adapter = TracingAdapter(model, inputs)
        example_inputs = adapter.flattened_inputs
        ts = torch.jit.trace(adapter, example_inputs)
    else:
        ts = torch.jit.trace(model, (inputs,))

    export_path = str(Path(cfg.OUTPUT_DIR) / "model_traced.ts")
    ts.save(export_path)
    print(f"Traced model saved to: {export_path}")
    raise SystemExit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "trace"],
        help="Mode: train or trace",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Path to a Detectron2 training checkpoint (.pth) to resume from",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    total_json = "zod_traffic_sign_de_cleaned.json"
    train_json = "zod_traffic_sign_de_cleaned_train.json"
    val_json = "zod_traffic_sign_de_cleaned_val.json"

    if not (os.path.exists(train_json) and os.path.exists(val_json)):
        dataset_split(total_json, train_json, val_json, val_ratio=0.1)

    cfg = setup_cfg()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # When tracing, build model, export TorchScript for Netron and exit.
    if args.mode == "train":
        # training mode
        n_train = len(load_dataset_json(train_json))
        n_val = len(load_dataset_json(val_json))
        print(f"Training samples: {n_train}, Validation samples: {n_val}")

        max_iter = NUM_EPOCHS * math.ceil(n_train / BATCH_SIZE)
        print(f"Total Epochs: {NUM_EPOCHS}")
        print(f"Total training iterations: {max_iter}")
        cfg.SOLVER.MAX_ITER = max_iter

        train(
            cfg,
            train_json=train_json,
            val_json=val_json,
            thing_classes=LABELS,
            resume_from=args.resume_from or None,
        )
    elif args.mode == "trace":
        # trace mode for netron visualization
        trace(
            cfg,
            resume_from=args.resume_from or None,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
