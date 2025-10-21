import os
import random

import cv2
import torch

from detectron2.engine import DefaultTrainer, DefaultPredictor, hooks
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer


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
        self.predictor: DefaultPredictor = None

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
