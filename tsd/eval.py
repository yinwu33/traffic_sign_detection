# --- add imports ---
from typing import List, Dict, Tuple
import numpy as np
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog
from collections import defaultdict

# --- custom evaluator ---
class BinaryAPBySizeEvaluator(DatasetEvaluator):
    """
    二分类（单类别）的 AP 评估器，支持整体 AP 和按目标尺寸分桶 AP_s/AP_m/AP_l。
    - 使用 Detectron2 的 dataset_dicts（无需 COCO JSON）
    - IoU 阈值默认 0.5
    - 尺寸分桶默认使用 COCO 风格面积阈值：S < 32^2, M in [32^2, 96^2], L > 96^2
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        iou_thresh: float = 0.5,
        size_mode: str = "area",  # "area" (默认, COCO风格) 或 "max_side"
        small_thr: int = 32,
        large_thr: int = 96,
        class_id: int = 0,  # 预测类别（单类任务通常为 0）
        ignore_images_without_bucket_gt: bool = True,  # 分桶评估时只统计包含该桶GT的图片（近似COCO）
    ):
        self._dataset_name = dataset_name
        self._iou_thr = float(iou_thresh)
        self._size_mode = size_mode
        self._small_thr = int(small_thr)
        self._large_thr = int(large_thr)
        self._class_id = int(class_id)
        self._ignore_images_without_bucket_gt = bool(ignore_images_without_bucket_gt)

        self._records = []  # list of dict per image

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        x, y, w, h = xywh.T
        return np.stack([x, y, x + w, y + h], axis=1)

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a: [Na,4], b: [Nb,4], both xyxy
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

        inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
        inter_y1 = np.maximum(ay1[:, None], by1[None, :])
        inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
        inter_y2 = np.minimum(ay2[:, None], by2[None, :])

        inter_w = np.clip(inter_x2 - inter_x1, 0, None)
        inter_h = np.clip(inter_y2 - inter_y1, 0, None)
        inter = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a[:, None] + area_b[None, :] - inter
        iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
        return iou

    def _bucket_ids_for_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        # 返回 0=small, 1=medium, 2=large
        if len(boxes_xyxy) == 0:
            return np.zeros((0,), dtype=np.int64)
        w = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0, None)
        h = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0, None)
        if self._size_mode == "max_side":
            s = np.maximum(w, h)
            small_mask = s < self._small_thr
            large_mask = s > self._large_thr
        else:  # "area" (COCO style)
            area = w * h
            small_mask = area < (self._small_thr**2)
            large_mask = area > (self._large_thr**2)
        bucket = np.zeros_like(small_mask, dtype=np.int64)  # default medium (1)
        bucket[small_mask] = 0
        bucket[large_mask] = 2
        bucket[~small_mask & ~large_mask] = 1
        return bucket

    @staticmethod
    def _ap_from_scores_tpfp(scores: np.ndarray, tp: np.ndarray, num_gt: int) -> float:
        """
        VOC/COCO 常见的 AP 计算：构造PR曲线，做precision envelope，再积分。
        scores: [N], tp: [N] (0/1), num_gt: 标注目标数
        """
        if num_gt == 0 or len(scores) == 0:
            return float("nan") if num_gt == 0 else 0.0

        # 排序（分数高在前）
        order = np.argsort(-scores)
        tp = tp[order].astype(np.float32)
        fp = 1.0 - tp

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recall = cum_tp / (num_gt + 1e-12)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

        # precision envelope（从右到左取上凸包）
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # 在recall的变化点积分
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return ap

    def reset(self):
        self._records.clear()
        # 直接从 DatasetCatalog 取 GT（detectron2 的 list[dict] 结构）
        self._gt_by_image_id = {}
        dataset_dicts = DatasetCatalog.get(self._dataset_name)
        for d in dataset_dicts:
            img_id = d.get("image_id", d["file_name"])  # 兜底用 file_name 当键
            gt_xyxy = []
            for a in d.get("annotations", []):
                box = BoxMode.convert(a["bbox"], a["bbox_mode"], BoxMode.XYXY_ABS)
                gt_xyxy.append(box)
            gt_xyxy = np.asarray(gt_xyxy, np.float32)
            self._gt_by_image_id[img_id] = {
                "boxes": gt_xyxy,
                "bucket": self._bucket_ids_for_boxes(gt_xyxy),
            }

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            # 用 image_id 或 file_name 找回 GT
            img_id = inp.get("image_id", inp["file_name"])
            gt_pack = self._gt_by_image_id.get(
                img_id,
                {
                    "boxes": np.zeros((0, 4), np.float32),
                    "bucket": np.zeros((0,), np.int64),
                },
            )
            gt_xyxy = gt_pack["boxes"]
            gt_bucket = gt_pack["bucket"]

            inst = out["instances"].to("cpu")
            keep = (
                (inst.pred_classes.numpy() == self._class_id)
                if hasattr(inst, "pred_classes")
                else slice(None)
            )
            pred_boxes = inst.pred_boxes.tensor.numpy()[keep]
            scores = inst.scores.numpy()[keep]

            self._records.append(
                {
                    "gt_boxes": gt_xyxy,
                    "gt_bucket": gt_bucket,
                    "pred_boxes": pred_boxes.astype(np.float32),
                    "scores": scores.astype(np.float32),
                }
            )

    def _match_per_image(
        self, pred_boxes, scores, gt_boxes
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单张图像进行贪心匹配（IoU>=thr），返回：
          tp_flags: [Np] 0/1
          matched_gt_idx: [Np] -1 表示未匹配
        """
        Np = len(pred_boxes)
        Ng = len(gt_boxes)
        tp_flags = np.zeros((Np,), dtype=np.int8)
        matched_gt = -np.ones((Np,), dtype=np.int64)
        if Np == 0 or Ng == 0:
            return tp_flags, matched_gt

        # 按分数降序
        order = np.argsort(-scores)
        pred_boxes = pred_boxes[order]
        scores = scores[order]

        ious = self._iou_matrix(pred_boxes, gt_boxes)
        gt_taken = np.zeros((Ng,), dtype=bool)

        for i in range(Np):
            # 为当前预测选择 IoU 最高的尚未匹配 GT
            iou_row = ious[i]
            j = int(np.argmax(iou_row))
            if iou_row[j] >= self._iou_thr and not gt_taken[j]:
                tp_flags[order[i]] = 1
                matched_gt[order[i]] = j
                gt_taken[j] = True
            # else: 默认是 FP (0)

        return tp_flags, matched_gt

    def evaluate(self) -> Dict[str, float]:
        # --- 汇总整体 ---
        all_scores: List[float] = []
        all_tp: List[int] = []
        total_gt = 0

        # 分桶容器
        bucket_scores = {0: [], 1: [], 2: []}
        bucket_tp = {0: [], 1: [], 2: []}
        bucket_gt_counts = {0: 0, 1: 0, 2: 0}

        # 统计哪些图片用于每个桶（近似 COCO：只在包含该桶GT的图片上做评估）
        imgs_with_bucket = {0: [], 1: [], 2: []}
        for rec in self._records:
            for b in (0, 1, 2):
                if np.any(rec["gt_bucket"] == b):
                    imgs_with_bucket[b].append(True)
                else:
                    imgs_with_bucket[b].append(False)

        # 遍历每张图，做匹配
        for idx, rec in enumerate(self._records):
            gt_boxes = rec["gt_boxes"]
            pred_boxes = rec["pred_boxes"]
            scores = rec["scores"]
            tp_flags, matched_idx = self._match_per_image(pred_boxes, scores, gt_boxes)

            # 整体
            all_scores.extend(scores.tolist())
            all_tp.extend(tp_flags.tolist())
            total_gt += len(gt_boxes)

            # 分桶：只在“该桶存在 GT 的图片”上统计，并且只把匹配到该桶 GT 的预测作为 TP；
            # 其余未匹配或匹配到其它桶的预测，视为该桶的 FP。
            gt_bucket = rec["gt_bucket"]
            for b in (0, 1, 2):
                if self._ignore_images_without_bucket_gt and not np.any(gt_bucket == b):
                    continue  # 忽略无该桶GT的图片
                # 该图像该桶的GT数量
                bucket_gt_counts[b] += int(np.sum(gt_bucket == b))

                # 为该桶构造 tp/fp 列表：
                #  - 匹配到该桶GT的标记为 TP
                #  - 否则为 FP（包括未匹配或匹配到其他桶的）
                # 这样能得到符合直觉的 PR 曲线
                is_tp_b = np.zeros_like(tp_flags)
                for k, mj in enumerate(matched_idx):
                    if mj >= 0 and gt_bucket[mj] == b and tp_flags[k] == 1:
                        is_tp_b[k] = 1
                bucket_scores[b].extend(scores.tolist())
                bucket_tp[b].extend(is_tp_b.tolist())

        # 计算 AP
        metrics = {}
        metrics["AP"] = self._ap_from_scores_tpfp(
            np.array(all_scores, dtype=np.float32),
            np.array(all_tp, dtype=np.int8),
            total_gt,
        )

        name_map = {0: "AP_s", 1: "AP_m", 2: "AP_l"}
        for b in (0, 1, 2):
            ap_b = self._ap_from_scores_tpfp(
                np.array(bucket_scores[b], dtype=np.float32),
                np.array(bucket_tp[b], dtype=np.int8),
                bucket_gt_counts[b],
            )
            metrics[name_map[b]] = ap_b

        print(
            {
                k: (None if (isinstance(v, float) and np.isnan(v)) else float(v))
                for k, v in metrics.items()
            }
        )
        return metrics
