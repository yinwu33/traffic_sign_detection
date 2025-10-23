# train_retinanet_ddp.py — Detectron2 RetinaNet training with single- or multi-GPU (DDP)
# - Drop-in replacement for your original script.
# - Keeps single-GPU behavior by default; add --num-gpus > 1 for multi-GPU.

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
from datetime import datetime
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm

# ---- your project-specific imports ----
from tsd.hooks import VisualizationHook
from tsd.datasets.utils import load_dataset_json
from tsd.eval import BinaryAPBySizeEvaluator

OUTPUT_ROOT = "./output"
OUTPUT_DIR = f"{OUTPUT_ROOT}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_retina_r50"
LABELS = ["ts"]
DATASET_METADATA_DIR = "./data/metadata"
MODEL_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"


def register_my_dataset(name: str, json_fpath: str, labels: list[str]):
    # NOTE: in DDP, this runs in every process — that's OK.
    dataset_dicts = load_dataset_json(json_fpath)  # cache it
    DatasetCatalog.register(name, lambda: dataset_dicts)
    MetadataCatalog.get(name).set(
        json_file=json_fpath,
        evaluator_type="coco",
        thing_classes=labels,
    )


class MyTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks_list = super().build_hooks()
        # Only the main process adds expensive/side-effect hooks (e.g., visualization)
        if comm.is_main_process():
            hooks_list.append(
                VisualizationHook(
                    cfg=self.cfg,
                    dataset_name=self.cfg.DATASETS.TRAIN[0],
                    output_dir=self.cfg.OUTPUT_DIR,
                    period=self.cfg.SOLVER.CHECKPOINT_PERIOD,
                    samples_per_epoch=1,
                )
            )
        return hooks_list


def do_train(cfg, resume_from=None):
    if resume_from:
        cfg.MODEL.WEIGHTS = str(resume_from)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=bool(resume_from))
    trainer.train()


def do_trace(cfg, resume_from=None):
    # Only run tracing on rank-0 to avoid duplicate work
    if not comm.is_main_process():
        return

    if resume_from:
        cfg.MODEL.WEIGHTS = resume_from
    cfg.MODEL.DEVICE = "cpu"  # tracing on CPU as in your original script

    try:
        from detectron2.export import TracingAdapter
    except Exception:
        TracingAdapter = None

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    H, W = 1024, 1024
    dummy_img = torch.randn(3, H, W)
    inputs = [{"image": dummy_img}]

    if TracingAdapter is not None:
        adapter = TracingAdapter(model, inputs)
        example_inputs = adapter.flattened_inputs
        ts = torch.jit.trace(adapter, example_inputs)
    else:
        ts = torch.jit.trace(model, (inputs,))

    export_path = str(Path(cfg.OUTPUT_DIR) / "model_traced.ts")
    ts.save(export_path)
    print(f"[trace] Traced model saved to: {export_path}")


def do_eval(cfg, resume_from):
    # Evaluate only on rank-0; Detectron2 eval will handle sync if needed
    if resume_from:
        cfg.MODEL.WEIGHTS = str(resume_from)

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    evaluator = BinaryAPBySizeEvaluator(
        cfg.DATASETS.VAL[0],
        iou_thresh=0.5,
        size_mode="area",
        small_thr=32,
        large_thr=96,
        class_id=0,
        ignore_images_without_bucket_gt=True,
    )
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.VAL[0])

    results = inference_on_dataset(model, val_loader, evaluator)
    if comm.is_main_process():
        print("[eval] Evaluation results:", results)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        import json

        with open(os.path.join(cfg.OUTPUT_DIR, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)


def build_cfg(user_cfg_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.merge_from_file(user_cfg_path)
    return cfg


def register_all(cfg):
    for ds in cfg.DATASETS.TRAIN:
        if comm.is_main_process():
            print("Registering dataset for training:", ds)
        register_my_dataset(ds, f"{DATASET_METADATA_DIR}/{ds}.json", LABELS)

    for ds in cfg.DATASETS.TEST:
        if comm.is_main_process():
            print("Registering dataset for evaluation:", ds)
        register_my_dataset(ds, f"{DATASET_METADATA_DIR}/{ds}.json", LABELS)


# ----------------- argument parsing & launcher -----------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--mode", required=True, choices=["train", "trace", "eval"], help="Mode"
    )
    parser.add_argument(
        "--resume-from", type=str, default="", help="Path to a .pth checkpoint"
    )

    # DDP args — defaults keep single-GPU behavior
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs per machine")
    parser.add_argument(
        "--num-machines", type=int, default=1, help="Number of machines"
    )
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="Rank of this machine (multi-node)"
    )
    parser.add_argument(
        "--dist-url", type=str, default="auto", help="init_method for torch.distributed"
    )

    # Optional convenience: override OUTPUT_DIR
    parser.add_argument("--out", type=str, default="", help="Override OUTPUT_DIR")
    return parser.parse_args()


def main_worker(args):
    if comm.is_main_process():
        print(args)

    cfg = build_cfg(args.cfg)
    register_all(cfg)

    # OUTPUT_DIR per mode
    if args.out:
        cfg.OUTPUT_DIR = args.out
    else:
        suffix = f"_{args.mode}"
        cfg.OUTPUT_DIR = f"{OUTPUT_DIR}{suffix}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train / Trace / Eval
    if args.mode == "train":
        do_train(cfg, resume_from=args.resume_from or None)
    elif args.mode == "trace":
        do_trace(cfg, resume_from=args.resume_from or None)
    elif args.mode == "eval":
        assert args.resume_from, "Evaluation requires a checkpoint to load from."
        cfg.MODEL.WEIGHTS = args.resume_from
        do_eval(cfg, resume_from=args.resume_from)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def main():
    args = parse_args()

    # Use Detectron2's launcher to run 1 or many processes.
    # Single-GPU: --num-gpus 1 (default) → behavior identical to your old script.
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    main()
