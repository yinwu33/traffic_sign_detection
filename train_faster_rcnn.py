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
from tsd.datasets.utils import load_dataset_json
from tsd.eval import BinaryAPBySizeEvaluator

# for evaluation
import json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

OUTPUT_ROOT = "./output"
OUTPUT_DIR = f"{OUTPUT_ROOT}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_frcnn_r50"
BATCH_SIZE = 8  # for amp + 48 GB GPU memory
LABELS = [
    "ts",
]
NUM_EPOCHS = 20
BASE_LR = 1e-5
NUM_WORKERS = 14


DATASET_METADATA_DIR = "./data/metadata"

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
                dataset_name=cfg.DATASETS.TRAIN[0],
                output_dir=self.cfg.OUTPUT_DIR,  # TODO: use cfg directly
                period=self.cfg.SOLVER.CHECKPOINT_PERIOD,
                samples_per_epoch=1,
            )
        )
        return hooks_list


def train(cfg, resume_from=None):

    if resume_from:
        cfg.MODEL.WEIGHTS = str(resume_from)

    # start training
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


def eval(cfg, resume_from):
    """
    只使用验证集进行一次 evaluation。
    会在 OUTPUT_DIR 下生成：
      - coco_instances_results.json（预测结果）
      - eval_results.json（各项指标）
    """
    # 选择权重
    if resume_from:
        cfg.MODEL.WEIGHTS = str(resume_from)

    # 构建并加载模型
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # 构建验证集 loader + evaluator
    evaluator = BinaryAPBySizeEvaluator(
        cfg.DATASETS.VAL[0],  # TODO: should eval all val datasets
        iou_thresh=0.5,
        size_mode="area",  # 或 "max_side"
        small_thr=32,
        large_thr=96,
        class_id=0,
        ignore_images_without_bucket_gt=True,
    )
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.VAL[0])

    # 跑评估
    results = inference_on_dataset(model, val_loader, evaluator)
    print("Evaluation results:", results)

    # 另存一份指标字典
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "trace", "eval"],
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

    print(args)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.merge_from_file(args.cfg)

    for ds in cfg.DATASETS.TRAIN:
        print("Registering dataset for training:", ds)
        register_my_dataset(ds, f"{DATASET_METADATA_DIR}/{ds}.json", LABELS)

    for ds in cfg.DATASETS.TEST:
        print("Registering dataset for evaluation:", ds)
        register_my_dataset(ds, f"{DATASET_METADATA_DIR}/{ds}.json", LABELS)

    # When tracing, build model, export TorchScript for Netron and exit.
    if args.mode == "train":
        cfg.OUTPUT_DIR = OUTPUT_DIR + "_train"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        train(cfg, resume_from=args.resume_from or None)
    elif args.mode == "trace":
        cfg.OUTPUT_DIR = OUTPUT_DIR + "_trace"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # trace mode for netron visualization
        trace(
            cfg,
            resume_from=args.resume_from or None,
        )
    elif args.mode == "eval":
        cfg.OUTPUT_DIR = OUTPUT_DIR + "_eval"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        assert args.resume_from, "Evaluation requires a checkpoint to load from."
        cfg.MODEL.WEIGHTS = args.resume_from

        eval(cfg, resume_from=args.resume_from)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
