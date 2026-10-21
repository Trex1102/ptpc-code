#!/usr/bin/env python3
"""Resumable official FSOD-VFM + paired PTPC Guard evaluation on VOC.

The baseline path calls the released FSOD-VFM prediction function without
changing its proposals, labels, graph diffusion, or top-100 selection.  The
paired Guard then changes only scores, using the exact DINOv2 mask-pooled
descriptor that FSOD-VFM already computed for each retained detection.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rescore_guard import guard_scores


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

NOVEL_CLASSES = {
    1: ["bus", "sofa", "cow", "bird", "motorbike"],
    2: ["horse", "aeroplane", "bottle", "sofa", "cow"],
    3: ["cat", "motorbike", "boat", "sofa", "sheep"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run released FSOD-VFM and a score-only paired Guard on VOC."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--query-json",
        type=Path,
        help=(
            "Optional COCO-format query annotation file. By default the full "
            "VOC2007 test annotation is used."
        ),
    )
    parser.add_argument(
        "--query-image-dir",
        type=Path,
        help=(
            "Image directory paired with --query-json. Both query overrides "
            "must be supplied together."
        ),
    )
    parser.add_argument(
        "--query-label",
        default="VOC2007 test",
        help="Human-readable query-set label recorded in provenance.",
    )
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--shot", type=int, choices=(1, 2, 3, 5, 10), default=5)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 processes every image in the selected query annotation.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guard-margin", type=float, default=0.01)
    parser.add_argument("--guard-suppression", type=float, default=0.85)
    parser.add_argument("--guard-temperature", type=float, default=0.04)
    parser.add_argument("--binary-limit", action="store_true",
                        help="Historical binary control; canonical evaluation is continuous")
    parser.add_argument("--guard-lower", type=float, default=0.05)
    parser.add_argument("--guard-upper", type=float, default=1.0)
    parser.add_argument("--base-samples-per-class", type=int, default=20)
    parser.add_argument("--min-threshold", type=float, default=0.01)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--diffusion-alpha", type=float, default=0.30)
    parser.add_argument("--diffusion-lambda", type=float, default=0.50)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def atomic_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=indent, sort_keys=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_base_support(
    voc2007_root: Path, base_classes: Sequence[str], max_per_class: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Mirror the FSCE VOC trainval traversal used by PTPCGuard.

    FSCE includes difficult instances during training.  VOC coordinates are
    converted from one-based inclusive XML coordinates to the same zero-based
    xywh convention used by the released FSOD-VFM support JSON files.
    """
    split_path = voc2007_root / "ImageSets" / "Main" / "trainval.txt"
    image_ids = [row.strip() for row in split_path.read_text().splitlines() if row.strip()]
    base_set = set(base_classes)
    support: Dict[str, List[Dict[str, Any]]] = {name: [] for name in base_classes}

    for image_id in image_ids:
        if all(len(support[name]) >= max_per_class for name in base_classes):
            break
        xml_path = voc2007_root / "Annotations" / f"{image_id}.xml"
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in base_set or len(support[name]) >= max_per_class:
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = float(bbox.findtext("xmin")) - 1.0
            ymin = float(bbox.findtext("ymin")) - 1.0
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))
            support[name].append(
                {
                    "image": f"PascalVOC/VOC2007/JPEGImages/{image_id}.jpg",
                    "bbox": [xmin, ymin, xmax - (xmin + 1.0), ymax - (ymin + 1.0)],
                    "image_id": image_id,
                    "difficult": int(obj.findtext("difficult", default="0")),
                }
            )

    counts = {name: len(rows) for name, rows in support.items()}
    incomplete = {name: count for name, count in counts.items() if count != max_per_class}
    if incomplete:
        raise RuntimeError(f"Could not collect the requested base support: {incomplete}")
    return support


def support_for_extractor(support: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        name: [{"image": row["image"], "bbox": list(row["bbox"])} for row in rows]
        for name, rows in support.items()
    }


def load_or_build_prototypes(
    *,
    cache_path: Path,
    support: Mapping[str, Sequence[Mapping[str, Any]]],
    support_util: Any,
    sam2_predictor: Any,
    feat_extractor: Any,
    image_transform: Any,
    data_dir: Path,
    device: str,
    role: str,
    model_fingerprint: str,
) -> Tuple[torch.Tensor, List[str]]:
    extractor_support = support_for_extractor(support)
    support_hash = canonical_hash(extractor_support)
    expected = {
        "schema": 1,
        "role": role,
        "support_sha256": support_hash,
        "model_fingerprint": model_fingerprint,
    }
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = cached.get("metadata", {})
        if all(metadata.get(key) == value for key, value in expected.items()):
            tensor = cached["prototype_tensor"].to(device)
            classes = list(cached["prototype_classes"])
            log(f"loaded {role} prototypes from {cache_path}")
            return tensor, classes
        log(f"ignoring stale {role} prototype cache at {cache_path}")

    log(
        f"building {role} prototypes from "
        f"{sum(len(rows) for rows in extractor_support.values())} support instances"
    )
    started = time.monotonic()
    memory_bank = support_util.extract_support_features(
        extractor_support,
        sam2_predictor,
        "DINOV2",
        feat_extractor,
        image_transform,
        str(data_dir),
        device,
    )
    tensor, classes = support_util.compute_prototype_weights(memory_bank, device)
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values in {role} prototypes")
    metadata = dict(expected)
    metadata.update(
        {
            "classes": classes,
            "shape": list(tensor.shape),
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "seconds": time.monotonic() - started,
        }
    )
    atomic_torch_save(
        cache_path,
        {
            "metadata": metadata,
            "prototype_tensor": tensor.detach().cpu(),
            "prototype_classes": classes,
        },
    )
    log(f"built {role} prototypes in {metadata['seconds']:.1f}s")
    return tensor, classes


def clean_number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric result: {result}")
    return result


def apply_guard_to_shard(
    raw_results: Sequence[MutableMapping[str, Any]],
    novel_prototypes: torch.Tensor,
    novel_classes: Sequence[str],
    base_prototypes: torch.Tensor,
    base_classes: Sequence[str],
    *,
    lower: float,
    upper: float,
    margin: float,
    suppression: float,
    temperature: float | None = None,
) -> Dict[str, Any]:
    baseline: List[Dict[str, Any]] = []
    guarded: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    novel_cpu = novel_prototypes.detach().float().cpu()
    base_cpu = base_prototypes.detach().float().cpu()

    for local_index, result in enumerate(raw_results):
        feature = torch.as_tensor(result["feat"], dtype=torch.float32).reshape(-1)
        feature = F.normalize(feature, dim=0)
        novel_sims = feature @ novel_cpu
        base_sims = feature @ base_cpu
        max_novel_value, max_novel_index = torch.max(novel_sims, dim=0)
        max_base_value, max_base_index = torch.max(base_sims, dim=0)
        max_novel = clean_number(max_novel_value.item())
        max_base = clean_number(max_base_value.item())
        gap = max_base - max_novel
        old_score = clean_number(result["score"])
        in_band = old_score > lower and old_score <= upper
        guard_fire = bool(in_band and gap > margin)
        new_score = old_score * suppression if guard_fire else old_score

        identity = {
            "image_id": int(result["image_id"]),
            "category_id": int(result["category_id"]),
            "bbox": [clean_number(item) for item in result["bbox"]],
        }
        baseline.append({**identity, "score": old_score})
        guarded.append({**identity, "score": new_score})
        diagnostics.append(
            {
                "local_index": local_index,
                **identity,
                "baseline_score": old_score,
                "guard_score": new_score,
                "max_novel": max_novel,
                "max_base": max_base,
                "base_novel_gap": gap,
                "nearest_novel": novel_classes[int(max_novel_index.item())],
                "nearest_base": base_classes[int(max_base_index.item())],
                "in_intervention_band": in_band,
                "guard_fire": guard_fire,
            }
        )
    if temperature is not None:
        native = np.asarray([r['baseline_score'] for r in diagnostics], np.float32)
        gap = np.asarray([r['max_base'] - r['max_novel'] for r in diagnostics], np.float32)
        scores = guard_scores(native, gap, dict(margin=margin, suppression=suppression,
                                               lower=lower, upper=upper), temperature)
        for row, diag, score in zip(guarded, diagnostics, scores):
            row['score'] = diag['guard_score'] = float(score)
            diag['guard_fire'] = bool(score < np.float32(diag['baseline_score']))
    return {"baseline": baseline, "guard": guarded, "diagnostics": diagnostics}


def identity_hash(predictions: Sequence[Mapping[str, Any]]) -> str:
    identities = [
        [row["image_id"], row["category_id"], [float(x) for x in row["bbox"]]]
        for row in predictions
    ]
    return canonical_hash(identities)


def subset_ground_truth(source: Path, destination: Path, image_ids: set[int]) -> Path:
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    subset = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in {"images", "annotations"}
    }
    subset["images"] = [row for row in payload["images"] if int(row["id"]) in image_ids]
    subset["annotations"] = [
        row for row in payload["annotations"] if int(row["image_id"]) in image_ids
    ]
    atomic_json(destination, subset)
    return destination


def evaluate_coco_bbox(
    gt_path: Path, predictions: Sequence[Mapping[str, Any]], target_names: Sequence[str]
) -> Dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_gt.dataset.setdefault("info", {"description": "FSOD-VFM paired evaluation"})
        coco_gt.dataset.setdefault("licenses", [])
        coco_dt = coco_gt.loadRes(list(predictions))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        cat_ids = coco_gt.getCatIds(catNms=list(target_names))
        evaluator.params.catIds = cat_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    iou_index = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.50)))
    area_index = list(evaluator.params.areaRngLbl).index("all")
    max_det_index = list(evaluator.params.maxDets).index(100)
    id_to_name = {row["id"]: row["name"] for row in coco_gt.loadCats(cat_ids)}
    per_class_ap50: Dict[str, float | None] = {}
    for category_index, category_id in enumerate(cat_ids):
        values = precision[iou_index, :, category_index, area_index, max_det_index]
        valid = values[values > -1]
        per_class_ap50[id_to_name[category_id]] = (
            float(np.mean(valid)) if valid.size else None
        )
    return {
        "stats": {
            "AP": float(evaluator.stats[0]),
            "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]),
            "APs": float(evaluator.stats[3]),
            "APm": float(evaluator.stats[4]),
            "APl": float(evaluator.stats[5]),
            "AR1": float(evaluator.stats[6]),
            "AR10": float(evaluator.stats[7]),
            "AR100": float(evaluator.stats[8]),
            "ARs": float(evaluator.stats[9]),
            "ARm": float(evaluator.stats[10]),
            "ARl": float(evaluator.stats[11]),
        },
        "per_class_AP50": per_class_ap50,
        "target_category_ids": cat_ids,
    }


def xywh_iou(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float64)
    x1, y1, width, height = (float(item) for item in box)
    x2, y2 = x1 + width, y1 + height
    bx1 = boxes[:, 0]
    by1 = boxes[:, 1]
    bx2 = bx1 + boxes[:, 2]
    by2 = by1 + boxes[:, 3]
    intersection = np.maximum(0.0, np.minimum(x2, bx2) - np.maximum(x1, bx1)) * np.maximum(
        0.0, np.minimum(y2, by2) - np.maximum(y1, by1)
    )
    union = width * height + boxes[:, 2] * boxes[:, 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_predictions(
    gt_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    target_category_ids: set[int],
    *,
    iou_threshold: float = 0.50,
    score_threshold: float = 0.0,
) -> Dict[str, Any]:
    with gt_path.open("r", encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    gt_by_key: Dict[Tuple[int, int], List[Sequence[float]]] = defaultdict(list)
    for ann in ground_truth["annotations"]:
        if int(ann["category_id"]) in target_category_ids and not int(ann.get("iscrowd", 0)):
            gt_by_key[(int(ann["image_id"]), int(ann["category_id"]))].append(ann["bbox"])
    total_gt = sum(len(rows) for rows in gt_by_key.values())

    predictions_by_key: Dict[Tuple[int, int], List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(predictions):
        category_id = int(row["category_id"])
        if category_id in target_category_ids and float(row["score"]) >= score_threshold:
            predictions_by_key[(int(row["image_id"]), category_id)].append((index, row))

    matched: Dict[int, Dict[str, Any]] = {}
    true_positives = 0
    for key, indexed_rows in predictions_by_key.items():
        gt_boxes = np.asarray(gt_by_key.get(key, []), dtype=np.float64).reshape(-1, 4)
        used = np.zeros((len(gt_boxes),), dtype=bool)
        indexed_rows.sort(key=lambda item: (-float(item[1]["score"]), item[0]))
        for index, row in indexed_rows:
            overlaps = xywh_iou(row["bbox"], gt_boxes)
            if overlaps.size:
                overlaps[used] = -1.0
                best = int(np.argmax(overlaps))
                best_iou = float(overlaps[best])
            else:
                best = -1
                best_iou = 0.0
            is_tp = bool(best >= 0 and best_iou >= iou_threshold)
            if is_tp:
                used[best] = True
                true_positives += 1
            matched[index] = {"tp": is_tp, "iou": max(0.0, best_iou)}
    return {
        "by_index": matched,
        "true_positives": true_positives,
        "ground_truths": total_gt,
        "recall": true_positives / total_gt if total_gt else 0.0,
    }


def calibration_metrics(
    predictions: Sequence[Mapping[str, Any]], matches: Mapping[int, Mapping[str, Any]], bins: int = 15
) -> Dict[str, Any]:
    rows = [
        (float(row["score"]), bool(matches[index]["tp"]), float(matches[index]["iou"]))
        for index, row in enumerate(predictions)
        if index in matches
    ]
    if not rows:
        return {"count": 0, "ECE15": None, "Brier": None, "LaECE0_15": None}
    scores = np.asarray([row[0] for row in rows], dtype=np.float64)
    labels = np.asarray([row[1] for row in rows], dtype=np.float64)
    ious = np.asarray([row[2] if row[1] else 0.0 for row in rows], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, scores, side="right") - 1, bins - 1)
    assignments = np.maximum(assignments, 0)
    ece = 0.0
    laece = 0.0
    bin_rows = []
    for bin_index in range(bins):
        selected = assignments == bin_index
        count = int(selected.sum())
        if count == 0:
            bin_rows.append({"bin": bin_index, "count": 0})
            continue
        confidence = float(scores[selected].mean())
        accuracy = float(labels[selected].mean())
        localization_precision = float(ious[selected].mean())
        weight = count / len(rows)
        ece += weight * abs(confidence - accuracy)
        laece += weight * abs(confidence - localization_precision)
        bin_rows.append(
            {
                "bin": bin_index,
                "count": count,
                "mean_confidence": confidence,
                "precision": accuracy,
                "mean_matched_iou_with_fp_zero": localization_precision,
            }
        )
    return {
        "count": len(rows),
        "ECE15": float(ece),
        "Brier": float(np.mean(np.square(scores - labels))),
        "LaECE0_15": float(laece),
        "bins": bin_rows,
    }


def attach_failure_labels(
    diagnostics: List[MutableMapping[str, Any]],
    baseline_match: Mapping[int, Mapping[str, Any]],
    guard_match: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    fired = 0
    fired_tp = 0
    fired_fp = 0
    crossing_tp = 0
    crossing_fp = 0
    nearest_base_counts: Dict[str, int] = defaultdict(int)
    event_rows = []
    for index, row in enumerate(diagnostics):
        baseline_status = baseline_match.get(index, {"tp": False, "iou": 0.0})
        guard_status = guard_match.get(index, {"tp": False, "iou": 0.0})
        row["baseline_tp_iou50"] = bool(baseline_status["tp"])
        row["baseline_match_iou"] = float(baseline_status["iou"])
        row["guard_tp_iou50"] = bool(guard_status["tp"])
        row["guard_match_iou"] = float(guard_status["iou"])
        crossed = bool(row["baseline_score"] >= 0.50 and row["guard_score"] < 0.50)
        row["crossed_score_0.50"] = crossed
        if row["guard_fire"]:
            fired += 1
            nearest_base_counts[str(row["nearest_base"])] += 1
            if baseline_status["tp"]:
                fired_tp += 1
                crossing_tp += int(crossed)
                kind = "tp_suppressed"
            else:
                fired_fp += 1
                crossing_fp += int(crossed)
                kind = "fp_suppressed"
            event_rows.append({"prediction_index": index, "event": kind, **row})

    event_rows.sort(
        key=lambda row: (
            0 if row["event"] == "tp_suppressed" else 1,
            -float(row["baseline_score"]),
            -float(row["base_novel_gap"]),
        )
    )
    return {
        "guard_fires": fired,
        "guard_fire_rate_all_predictions": fired / len(diagnostics) if diagnostics else 0.0,
        "baseline_tp_suppressed": fired_tp,
        "baseline_fp_suppressed": fired_fp,
        "tp_crossed_score_0.50": crossing_tp,
        "fp_crossed_score_0.50": crossing_fp,
        "nearest_base_on_fires": dict(sorted(nearest_base_counts.items())),
        "events": event_rows,
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    if not args.binary_limit and (not math.isfinite(args.guard_temperature) or args.guard_temperature <= 0):
        raise ValueError("--guard-temperature must be finite and positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not (0.0 < args.guard_suppression <= 1.0):
        raise ValueError("--guard-suppression must lie in (0, 1]")
    if args.guard_lower >= args.guard_upper:
        raise ValueError("--guard-lower must be below --guard-upper")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if (args.query_json is None) != (args.query_image_dir is None):
        raise ValueError(
            "--query-json and --query-image-dir must either both be supplied or both omitted"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    seed_everything(args.seed)

    novel_classes = NOVEL_CLASSES[args.split]
    base_classes = [name for name in VOC_CLASSES if name not in novel_classes]
    support_path = (
        repo_root
        / "data"
        / "Pascal"
        / "vocsplit"
        / f"split{args.split}"
        / f"{args.shot}shot_seed{args.seed}.json"
    )
    test_json = (
        args.query_json.resolve()
        if args.query_json is not None
        else repo_root
        / "data"
        / "Pascal"
        / "VOC2007Test"
        / "voc07test_coco_format.json"
    )
    test_img_dir = (
        args.query_image_dir.resolve()
        if args.query_image_dir is not None
        else repo_root / "dataset" / "PascalVOC" / "VOC2007" / "JPEGImages"
    )
    data_dir = repo_root / "dataset"
    checkpoint_dir = repo_root / "checkpoints"
    checkpoint_paths = {
        "upn": checkpoint_dir / "upn_large.pth",
        "dinov2": checkpoint_dir / "dinov2_vitl14_pretrain.pth",
        "sam2": checkpoint_dir / "sam2.1_hiera_large.pt",
    }
    required = [support_path, test_json, test_img_dir, *checkpoint_paths.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required FSOD-VFM inputs: {missing}")

    provenance = {
        "schema": 1,
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "hostname": os.uname().nodename,
        "argv": sys.argv,
        "repo_root": str(repo_root),
        "fsod_vfm_commit": git_commit(repo_root),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dinov2_xformers_disabled": os.environ.get("XFORMERS_DISABLED") is not None,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "official_source_sha256": {
            name: sha256_file(repo_root / name)
            for name in (
                "main.py",
                "metric.py",
                "support_util.py",
                "query_util.py",
                "model/dinov2.py",
                "model/sam2.py",
            )
        },
        "split": args.split,
        "shot": args.shot,
        "seed": args.seed,
        "query": {
            "label": args.query_label,
            "annotation": str(test_json),
            "image_dir": str(test_img_dir),
        },
        "novel_classes": novel_classes,
        "base_classes": base_classes,
        "guard": {
            "descriptor": "FSOD-VFM native SAM2-mask-pooled DINOv2 ViT-L/14",
            "scope": "post-diffusion confidence only; boxes and labels invariant",
            "rule": "binary_limit" if args.binary_limit else "continuous",
            "temperature": None if args.binary_limit else args.guard_temperature,
            "margin": args.guard_margin,
            "suppression_factor": args.guard_suppression,
            "score_band": [args.guard_lower, args.guard_upper],
            "base_samples_per_class": args.base_samples_per_class,
        },
        "official_inference": {
            "min_threshold": args.min_threshold,
            "diffusion_steps": args.diffusion_steps,
            "alpha": args.diffusion_alpha,
            "lambda": args.diffusion_lambda,
            "top_predictions_per_image": 100,
        },
        "checkpoint_sha256": {
            name: sha256_file(path) for name, path in checkpoint_paths.items()
        },
        "support_json_sha256": sha256_file(support_path),
        "test_json_sha256": sha256_file(test_json),
    }
    atomic_json(output_dir / "provenance.json", provenance, indent=2)
    model_fingerprint = canonical_hash(
        {
            "model": "dinov2_vitl14",
            "checkpoint": provenance["checkpoint_sha256"]["dinov2"],
            "sam2": provenance["checkpoint_sha256"]["sam2"],
            "descriptor": "sam2-mask-pooled-dinov2",
            "xformers_disabled": provenance["dinov2_xformers_disabled"],
        }
    )

    log("loading official UPN, DINOv2 ViT-L/14, and SAM2.1-L models")
    import metric
    import model.dinov2
    import model.sam2
    import query_util
    import support_util
    from chatrex.upn import UPNWrapper

    upn = UPNWrapper(str(checkpoint_paths["upn"]))
    feat_extractor, image_transform = model.dinov2.load_dinov2_model(
        args.device,
        "dinov2_vitl14",
        image_size=(630, 630),
        repo_or_dir=str(repo_root / "dinov2"),
        pretrained=str(checkpoint_paths["dinov2"]),
    )
    _, sam2_predictor, _ = model.sam2.load_sam2_components(
        model_type="large", device=args.device, points_per_side=32
    )

    with support_path.open("r", encoding="utf-8") as handle:
        novel_support = json.load(handle)
    if set(novel_support) != set(novel_classes):
        raise RuntimeError(
            f"Unexpected support classes: got {sorted(novel_support)}, expected {sorted(novel_classes)}"
        )
    novel_cache = cache_dir / f"split{args.split}_{args.shot}shot_seed{args.seed}_novel.pt"
    novel_prototypes, novel_proto_classes = load_or_build_prototypes(
        cache_path=novel_cache,
        support=novel_support,
        support_util=support_util,
        sam2_predictor=sam2_predictor,
        feat_extractor=feat_extractor,
        image_transform=image_transform,
        data_dir=data_dir,
        device=args.device,
        role="novel",
        model_fingerprint=model_fingerprint,
    )

    voc2007_root = repo_root / "dataset" / "PascalVOC" / "VOC2007"
    base_support = build_base_support(
        voc2007_root, base_classes, args.base_samples_per_class
    )
    base_manifest = cache_dir / f"split{args.split}_base_support_{args.base_samples_per_class}.json"
    atomic_json(base_manifest, base_support, indent=2)
    base_cache = cache_dir / f"split{args.split}_base_{args.base_samples_per_class}.pt"
    base_prototypes, base_proto_classes = load_or_build_prototypes(
        cache_path=base_cache,
        support=base_support,
        support_util=support_util,
        sam2_predictor=sam2_predictor,
        feat_extractor=feat_extractor,
        image_transform=image_transform,
        data_dir=data_dir,
        device=args.device,
        role="base",
        model_fingerprint=model_fingerprint,
    )

    _, coco_loader = query_util.load_voc2007_coco_json(str(test_json), str(test_img_dir))
    all_images = list(coco_loader.dataset["images"])
    if args.max_images > 0:
        all_images = all_images[: args.max_images]
    if not all_images:
        raise RuntimeError("No test images selected")
    # Deliberately exclude timestamps and host process details: the signature
    # must remain identical when a stopped run resumes in a later invocation.
    run_signature = canonical_hash(
        {
            "schema": 1,
            "fsod_vfm_commit": provenance["fsod_vfm_commit"],
            "runner_sha256": provenance["runner_sha256"],
            "official_source_sha256": provenance["official_source_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "support_json_sha256": provenance["support_json_sha256"],
            "test_json_sha256": provenance["test_json_sha256"],
            "dinov2_xformers_disabled": provenance["dinov2_xformers_disabled"],
            "split": args.split,
            "shot": args.shot,
            "seed": args.seed,
            "selected_image_ids": [int(row["id"]) for row in all_images],
            "guard": provenance["guard"],
            "official_inference": provenance["official_inference"],
        }
    )
    atomic_json(
        output_dir / "run_manifest.json",
        {
            "run_signature": run_signature,
            "image_count": len(all_images),
            "selected_image_ids_sha256": canonical_hash([int(row["id"]) for row in all_images]),
            "chunk_size": args.chunk_size,
        },
        indent=2,
    )

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    original_images = coco_loader.dataset["images"]
    completed_baseline: List[Dict[str, Any]] = []
    completed_guard: List[Dict[str, Any]] = []
    completed_diagnostics: List[Dict[str, Any]] = []
    timings = []
    run_started = time.monotonic()
    try:
        for start in range(0, len(all_images), args.chunk_size):
            stop = min(start + args.chunk_size, len(all_images))
            shard_path = shard_dir / f"shard_{start:06d}_{stop:06d}.json"
            image_chunk = all_images[start:stop]
            expected_ids = [int(row["id"]) for row in image_chunk]
            if shard_path.is_file():
                with shard_path.open("r", encoding="utf-8") as handle:
                    shard = json.load(handle)
                if (
                    shard.get("run_signature") != run_signature
                    or shard.get("image_ids") != expected_ids
                ):
                    raise RuntimeError(f"Incompatible existing shard: {shard_path}")
                log(f"resuming completed shard {start}:{stop}")
            else:
                log(f"running official FSOD-VFM shard {start}:{stop} ({stop}/{len(all_images)})")
                coco_loader.dataset["images"] = image_chunk
                shard_started = time.monotonic()
                with torch.inference_mode():
                    raw_results = metric.generate_coco_style_predictions_upn(
                        coco_loader,
                        str(test_img_dir),
                        sam2_predictor,
                        "DINOV2",
                        feat_extractor,
                        image_transform,
                        novel_prototypes,
                        novel_proto_classes,
                        upn,
                        args.diffusion_steps,
                        args.diffusion_alpha,
                        args.diffusion_lambda,
                        args.device,
                        args.min_threshold,
                    )
                paired = apply_guard_to_shard(
                    raw_results,
                    novel_prototypes,
                    novel_proto_classes,
                    base_prototypes,
                    base_proto_classes,
                    lower=args.guard_lower,
                    upper=args.guard_upper,
                    margin=args.guard_margin,
                    suppression=args.guard_suppression,
                    temperature=None if args.binary_limit else args.guard_temperature,
                )
                elapsed = time.monotonic() - shard_started
                shard = {
                    "schema": 1,
                    "run_signature": run_signature,
                    "image_range": [start, stop],
                    "image_ids": expected_ids,
                    "seconds": elapsed,
                    **paired,
                }
                atomic_json(shard_path, shard)
                log(
                    f"finished shard {start}:{stop} in {elapsed:.1f}s; "
                    f"{len(paired['baseline'])} predictions"
                )
                del raw_results, paired
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            completed_baseline.extend(shard["baseline"])
            completed_guard.extend(shard["guard"])
            completed_diagnostics.extend(shard["diagnostics"])
            timings.append(float(shard["seconds"]))
    finally:
        coco_loader.dataset["images"] = original_images

    baseline_identity = identity_hash(completed_baseline)
    guard_identity = identity_hash(completed_guard)
    if baseline_identity != guard_identity:
        raise RuntimeError("Guard changed a category, box, image, or prediction order")
    for index, row in enumerate(completed_diagnostics):
        row["prediction_index"] = index
    atomic_json(output_dir / "predictions_baseline.json", completed_baseline)
    atomic_json(output_dir / "predictions_guard.json", completed_guard)

    selected_ids = {int(row["id"]) for row in all_images}
    evaluation_gt = test_json
    if len(all_images) != len(original_images):
        evaluation_gt = subset_ground_truth(
            test_json, output_dir / "ground_truth_subset.json", selected_ids
        )
    summary: Dict[str, Any] = {
        "schema": 1,
        "run_signature": run_signature,
        "completed_at": dt.datetime.now().astimezone().isoformat(),
        "images": len(all_images),
        "predictions": len(completed_baseline),
        "identity_sha256": baseline_identity,
        "identity_invariant": True,
        "inference_seconds": sum(timings),
        "wall_seconds_this_invocation": time.monotonic() - run_started,
        "seconds_per_image": sum(timings) / len(all_images),
        "guard_fires": sum(bool(row["guard_fire"]) for row in completed_diagnostics),
    }

    if not args.skip_eval:
        log("evaluating paired baseline and Guard predictions")
        baseline_eval = evaluate_coco_bbox(evaluation_gt, completed_baseline, novel_classes)
        guard_eval = evaluate_coco_bbox(evaluation_gt, completed_guard, novel_classes)
        target_ids = set(int(value) for value in baseline_eval["target_category_ids"])
        baseline_match = match_predictions(evaluation_gt, completed_baseline, target_ids)
        guard_match = match_predictions(evaluation_gt, completed_guard, target_ids)
        baseline_match_05 = match_predictions(
            evaluation_gt, completed_baseline, target_ids, score_threshold=0.50
        )
        guard_match_05 = match_predictions(
            evaluation_gt, completed_guard, target_ids, score_threshold=0.50
        )
        failure = attach_failure_labels(
            completed_diagnostics, baseline_match["by_index"], guard_match["by_index"]
        )
        summary["evaluation"] = {
            "baseline": baseline_eval,
            "guard": guard_eval,
            "delta": {
                key: guard_eval["stats"][key] - baseline_eval["stats"][key]
                for key in ("AP", "AP50", "AP75")
            },
        }
        summary["calibration"] = {
            "baseline": calibration_metrics(
                completed_baseline, baseline_match["by_index"], bins=15
            ),
            "guard": calibration_metrics(completed_guard, guard_match["by_index"], bins=15),
        }
        summary["recall"] = {
            "baseline_score_0": baseline_match["recall"],
            "guard_score_0": guard_match["recall"],
            "baseline_score_0.5": baseline_match_05["recall"],
            "guard_score_0.5": guard_match_05["recall"],
            "ground_truths": baseline_match["ground_truths"],
        }
        summary["failure_cases"] = {key: value for key, value in failure.items() if key != "events"}
        atomic_json(output_dir / "failure_events.json", failure["events"])
        atomic_json(output_dir / "diagnostics.json", completed_diagnostics)

    atomic_json(output_dir / "summary.json", summary, indent=2)
    log(
        "complete: "
        + json.dumps(
            {
                "images": summary["images"],
                "predictions": summary["predictions"],
                "seconds_per_image": summary["seconds_per_image"],
                "guard_fires": summary["guard_fires"],
                "AP50_baseline": summary.get("evaluation", {})
                .get("baseline", {})
                .get("stats", {})
                .get("AP50"),
                "AP50_guard": summary.get("evaluation", {})
                .get("guard", {})
                .get("stats", {})
                .get("AP50"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
