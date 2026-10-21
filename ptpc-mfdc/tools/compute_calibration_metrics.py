#!/usr/bin/env python3
"""Compute calibration metrics for post-hoc PTPC/PTPC-style evaluation."""

import argparse
import json
import logging
import os
import re
import sys

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defrcn.config import get_cfg, set_global_cfg

logger = logging.getLogger(__name__)


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def match_predictions_to_gt(predictions, gt_boxes, gt_classes, iou_thresh=0.5,
                            class_agnostic=False):
    results = []
    matched_gt = set()
    sorted_preds = sorted(enumerate(predictions), key=lambda x: -x[1][0])

    for _, (score, pred_cls, pred_box) in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if gt_idx in matched_gt or (not class_agnostic and gt_cls != pred_cls):
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            gt_cls = gt_classes[best_gt_idx]
            results.append(
                {
                    "score": score,
                    "pred_class": pred_cls,
                    "is_correct": int(pred_cls == gt_cls),
                    "gt_class": gt_cls,
                }
            )
        else:
            results.append(
                {
                    "score": score,
                    "pred_class": pred_cls,
                    "is_correct": 0,
                    "gt_class": None,
                }
            )

    return results


def compute_ece(scores, correct, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(scores)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (scores >= bin_boundaries[i]) & (scores <= bin_boundaries[i + 1])
        bin_size = int(mask.sum())
        if bin_size == 0:
            continue
        avg_conf = scores[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (bin_size / total) * abs(avg_conf - avg_acc)

    return float(ece)


def compute_brier(scores, correct):
    if len(scores) == 0:
        return 0.0
    return float(np.mean((scores - correct) ** 2))


def compute_base_novel_fp_rate(match_results, novel_cls_ids, base_cls_ids):
    base_gt_matched = 0
    base_as_novel = 0

    for result in match_results:
        overlap_class = result.get("overlap_gt_class", result["gt_class"])
        if overlap_class is not None and overlap_class in base_cls_ids:
            base_gt_matched += 1
            if result["pred_class"] in novel_cls_ids:
                base_as_novel += 1

    if base_gt_matched == 0:
        return 0.0
    return base_as_novel / base_gt_matched


def get_class_ids(cfg):
    from defrcn.data.builtin_meta import (
        PASCAL_VOC_ALL_CATEGORIES,
        PASCAL_VOC_BASE_CATEGORIES,
        PASCAL_VOC_NOVEL_CATEGORIES,
    )

    dsname = cfg.DATASETS.TEST[0]
    if "voc" not in dsname:
        logger.warning("Only VOC is supported for class ID derivation. Got: %s", dsname)
        return set(), set()

    match = re.search(r"novel(\d+)", dsname)
    if not match:
        match = re.search(r"all(\d+)", dsname)
    if not match:
        return set(), set()

    split = int(match.group(1))
    all_cats = PASCAL_VOC_ALL_CATEGORIES[split]
    novel_cats = PASCAL_VOC_NOVEL_CATEGORIES[split]
    base_cats = PASCAL_VOC_BASE_CATEGORIES[split]

    novel_ids = set()
    base_ids = set()
    for idx, name in enumerate(all_cats):
        if name in novel_cats:
            novel_ids.add(idx)
        if name in base_cats:
            base_ids.add(idx)
    return novel_ids, base_ids


def run_inference_and_collect(cfg, model, pcb, iou_thresh):
    from defrcn.dataloader import build_detection_test_loader

    dataset_name = cfg.DATASETS.TEST[0]
    data_loader = build_detection_test_loader(cfg, dataset_name)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    gt_by_image = {}
    for record in dataset_dicts:
        image_id = record.get("image_id", record.get("file_name"))
        boxes = []
        classes = []
        for ann in record.get("annotations", []):
            bbox = ann["bbox"]
            if ann.get("bbox_mode", BoxMode.XYXY_ABS) == BoxMode.XYWH_ABS:
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            boxes.append(bbox)
            classes.append(ann["category_id"])
        gt_by_image[image_id] = (boxes, classes)

    all_matches = []
    model.eval()

    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
            if pcb is not None:
                outputs = pcb.execute_calibration(inputs, outputs)

            for inp, out in zip(inputs, outputs):
                image_id = inp.get("image_id", inp.get("file_name"))
                instances = out["instances"].to("cpu")
                if len(instances) == 0:
                    continue

                pred_boxes = instances.pred_boxes.tensor.numpy()
                pred_scores = instances.scores.numpy()
                pred_classes = instances.pred_classes.numpy()
                predictions = list(
                    zip(
                        pred_scores.tolist(),
                        pred_classes.tolist(),
                        pred_boxes.tolist(),
                    )
                )

                gt_boxes, gt_classes = gt_by_image.get(image_id, ([], []))
                if gt_boxes:
                    matches = match_predictions_to_gt(
                        predictions,
                        gt_boxes,
                        gt_classes,
                        iou_thresh=iou_thresh,
                    )
                    confusion_matches = match_predictions_to_gt(
                        predictions, gt_boxes, gt_classes, iou_thresh=iou_thresh,
                        class_agnostic=True,
                    )
                    for match, confusion in zip(matches, confusion_matches):
                        match["overlap_gt_class"] = confusion["gt_class"]
                    all_matches.extend(matches)
                else:
                    for score, pred_cls, _ in predictions:
                        all_matches.append(
                            {
                                "score": score,
                                "pred_class": pred_cls,
                                "is_correct": 0,
                                "gt_class": None,
                            }
                        )

            if (idx + 1) % 100 == 0:
                logger.info("Processed %d / %d batches", idx + 1, len(data_loader))

    return all_matches


def build_pcb(cfg):
    if not cfg.TEST.PCB_ENABLE:
        return None

    method_name = cfg.NOVEL_METHODS.METHOD if cfg.NOVEL_METHODS.ENABLE else ""
    method_name_lower = method_name.lower() if method_name else ""
    fm_only_method = (
        method_name_lower == "ptpc" and bool(cfg.NOVEL_METHODS.PTPC.FM_ONLY)
    )

    from defrcn.evaluation.novel_methods import build_novel_method_pcb

    if fm_only_method and method_name:
        return build_novel_method_pcb(None, cfg, method_name)

    from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock

    pcb = PrototypicalCalibrationBlock(cfg)
    if method_name:
        pcb = build_novel_method_pcb(pcb, cfg, method_name)
    return pcb


def main():
    parser = argparse.ArgumentParser(description="Compute calibration metrics (ECE, Brier, FP rate)")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--n-bins", type=int, default=15, help="Number of bins for ECE")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Config overrides")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)

    logging.basicConfig(level=logging.INFO)

    from defrcn.engine import DefaultTrainer

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    pcb = build_pcb(cfg)
    novel_ids, base_ids = get_class_ids(cfg)
    logger.info("Novel class IDs: %s", sorted(novel_ids))
    logger.info("Base class IDs: %s", sorted(base_ids))

    logger.info("Running inference and collecting predictions...")
    all_matches = run_inference_and_collect(cfg, model, pcb, args.iou_thresh)
    logger.info("Collected %d prediction-GT matches", len(all_matches))

    scores = np.array([m["score"] for m in all_matches])
    correct = np.array([m["is_correct"] for m in all_matches])

    ece = compute_ece(scores, correct, n_bins=args.n_bins)
    brier = compute_brier(scores, correct)
    base_novel_fp = compute_base_novel_fp_rate(all_matches, novel_ids, base_ids)

    per_class = {}
    for cls_id in sorted(novel_ids):
        cls_mask = np.array([m["pred_class"] == cls_id for m in all_matches])
        if int(cls_mask.sum()) == 0:
            continue
        cls_scores = scores[cls_mask]
        cls_correct = correct[cls_mask]
        per_class[int(cls_id)] = {
            "ece": compute_ece(cls_scores, cls_correct, n_bins=args.n_bins),
            "brier": compute_brier(cls_scores, cls_correct),
            "n_detections": int(cls_mask.sum()),
            "precision": float(cls_correct.mean()),
        }

    bin_boundaries = np.linspace(0, 1, args.n_bins + 1)
    bin_data = []
    for i in range(args.n_bins):
        lo = float(bin_boundaries[i])
        hi = float(bin_boundaries[i + 1])
        if i == args.n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_data.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "avg_confidence": float(scores[mask].mean()),
                "avg_accuracy": float(correct[mask].mean()),
                "count": count,
            }
        )

    results = {
        "ece": ece,
        "brier": brier,
        "base_novel_fp_rate": base_novel_fp,
        "n_total_detections": len(all_matches),
        "n_correct": int(correct.sum()),
        "overall_accuracy": float(correct.mean()) if len(correct) > 0 else 0.0,
        "per_class": per_class,
        "reliability_bins": bin_data,
        "config": {
            "dataset": cfg.DATASETS.TEST[0],
            "method": cfg.NOVEL_METHODS.METHOD if cfg.NOVEL_METHODS.ENABLE else "vanilla_pcb",
            "n_bins": args.n_bins,
            "iou_thresh": args.iou_thresh,
        },
    }

    output_path = args.output or os.path.join(cfg.OUTPUT_DIR, "calibration_metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2)

    print("\n" + "=" * 60)
    print("Calibration Metrics Summary")
    print("=" * 60)
    print(f"  ECE:              {ece:.4f}")
    print(f"  Brier Score:      {brier:.4f}")
    print(f"  Base->Novel FP:   {base_novel_fp:.4f}")
    print(f"  Total detections: {len(all_matches)}")
    print(f"  Overall accuracy: {correct.mean():.4f}" if len(correct) > 0 else "  No detections")
    print(f"\n  Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
