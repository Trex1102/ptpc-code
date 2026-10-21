#!/usr/bin/env python3
"""Resumable official FSOD-VFM + paired PTPC Guard evaluation on COCO.

The released FSOD-VFM inference path is left unchanged.  For every retained
detection, the paired Guard changes only its confidence using the native
SAM2-mask-pooled DINOv2 descriptor; image IDs, categories, boxes, ordering,
and the number of predictions are asserted to be identical.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

import run_fsod_vfm_guard as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run released FSOD-VFM and a score-only paired Guard on COCO."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 3, 5, 10, 30), required=True)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 processes the complete 5,000-image COCO 2017 validation set.",
    )
    parser.add_argument(
        "--query-json",
        type=Path,
        help=(
            "Optional COCO-format query annotation file. By default the full "
            "COCO 2017 validation set is used. Supplied together with "
            "--query-image-dir to score a held-out validation subset instead."
        ),
    )
    parser.add_argument(
        "--query-image-dir",
        type=Path,
        help="Image directory paired with --query-json. Both are required together.",
    )
    parser.add_argument(
        "--query-label",
        default="COCO 2017 val",
        help="Human-readable query-set label recorded in provenance.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guard-margin", type=float, default=0.05)
    parser.add_argument("--guard-suppression", type=float, default=0.50)
    parser.add_argument("--guard-temperature", type=float, default=0.01)
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


def build_coco_base_support(
    train_json: Path, base_classes: Sequence[str], max_per_class: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Select a deterministic, disjoint base-class support set from COCO train2017."""
    with train_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    image_names = {int(row["id"]): row["file_name"] for row in payload["images"]}
    category_names = {int(row["id"]): row["name"] for row in payload["categories"]}
    base_set = set(base_classes)
    support: Dict[str, List[Dict[str, Any]]] = {name: [] for name in base_classes}

    # The official annotation order is stable; the annotation ID makes that
    # choice explicit and independent of JSON serialization order.
    for annotation in sorted(payload["annotations"], key=lambda row: int(row["id"])):
        name = category_names[int(annotation["category_id"])]
        if name not in base_set or len(support[name]) >= max_per_class:
            continue
        if int(annotation.get("iscrowd", 0)):
            continue
        bbox = [float(value) for value in annotation["bbox"]]
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue
        image_id = int(annotation["image_id"])
        support[name].append(
            {
                "image": f"coco/train2017/{image_names[image_id]}",
                "bbox": bbox,
                "image_id": image_id,
                "annotation_id": int(annotation["id"]),
            }
        )
        if all(len(rows) == max_per_class for rows in support.values()):
            break

    incomplete = {
        name: len(rows) for name, rows in support.items() if len(rows) != max_per_class
    }
    if incomplete:
        raise RuntimeError(f"Could not collect requested COCO base support: {incomplete}")
    return support


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    if not args.binary_limit and (not core.math.isfinite(args.guard_temperature) or args.guard_temperature <= 0):
        raise ValueError("--guard-temperature must be finite and positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not (0.0 < args.guard_suppression <= 1.0):
        raise ValueError("--guard-suppression must lie in (0, 1]")
    if args.guard_lower >= args.guard_upper:
        raise ValueError("--guard-lower must be below --guard-upper")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    core.seed_everything(args.seed)

    coco_root = repo_root / "data" / "coco"
    support_path = coco_root / f"few_shot_{args.shot}shot_seed{args.seed}.json"
    train_json = coco_root / "annotations" / "instances_train2017.json"
    if (args.query_json is None) != (args.query_image_dir is None):
        raise ValueError("--query-json and --query-image-dir must be supplied together")
    test_json = (
        args.query_json.resolve()
        if args.query_json is not None
        else coco_root / "annotations" / "instances_val2017.json"
    )
    test_img_dir = (
        args.query_image_dir.resolve()
        if args.query_image_dir is not None
        else coco_root / "val2017"
    )
    data_dir = repo_root / "data"
    checkpoint_dir = repo_root / "checkpoints"
    checkpoint_paths = {
        "upn": checkpoint_dir / "upn_large.pth",
        "dinov2": checkpoint_dir / "dinov2_vitl14_pretrain.pth",
        "sam2": checkpoint_dir / "sam2.1_hiera_large.pt",
    }
    required = [
        support_path,
        train_json,
        test_json,
        test_img_dir,
        *checkpoint_paths.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required FSOD-VFM COCO inputs: {missing}")

    with support_path.open("r", encoding="utf-8") as handle:
        novel_support = json.load(handle)
    novel_classes = list(novel_support)
    if len(novel_classes) != 20:
        raise RuntimeError(f"Expected 20 COCO novel classes, got {len(novel_classes)}")
    bad_counts = {
        name: len(rows) for name, rows in novel_support.items() if len(rows) != args.shot
    }
    if bad_counts:
        raise RuntimeError(f"Malformed {args.shot}-shot support JSON: {bad_counts}")

    with train_json.open("r", encoding="utf-8") as handle:
        train_metadata = json.load(handle)
    all_classes = [
        row["name"] for row in sorted(train_metadata["categories"], key=lambda row: int(row["id"]))
    ]
    del train_metadata
    unknown = sorted(set(novel_classes) - set(all_classes))
    if unknown:
        raise RuntimeError(f"Support classes absent from COCO categories: {unknown}")
    base_classes = [name for name in all_classes if name not in set(novel_classes)]
    if len(all_classes) != 80 or len(base_classes) != 60:
        raise RuntimeError(
            f"Expected the COCO 60/20 split, got {len(base_classes)}/{len(novel_classes)}"
        )

    provenance = {
        "schema": 1,
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "hostname": os.uname().nodename,
        "argv": sys.argv,
        "dataset": "COCO 2017 validation; 60 base / 20 novel FSOD split",
        "repo_root": str(repo_root),
        "fsod_vfm_commit": core.git_commit(repo_root),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dinov2_xformers_disabled": os.environ.get("XFORMERS_DISABLED") is not None,
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        "shared_runner_sha256": core.sha256_file(Path(core.__file__).resolve()),
        "official_source_sha256": {
            name: core.sha256_file(repo_root / name)
            for name in (
                "main.py",
                "metric.py",
                "support_util.py",
                "query_util.py",
                "model/dinov2.py",
                "model/sam2.py",
            )
        },
        "shot": args.shot,
        "seed": args.seed,
        "novel_classes": novel_classes,
        "base_classes": base_classes,
        "support_protocol": (
            "official released FSOD-VFM support JSON"
            if args.shot in (10, 30)
            else "nested prefix of the released 30-shot seed-33 support JSON"
        ),
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
            name: core.sha256_file(path) for name, path in checkpoint_paths.items()
        },
        "support_json_sha256": core.sha256_file(support_path),
        "train_json_sha256": core.sha256_file(train_json),
        "test_json_sha256": core.sha256_file(test_json),
        "query_label": args.query_label,
        "query_json": str(test_json),
        "query_image_dir": str(test_img_dir),
    }
    core.atomic_json(output_dir / "provenance.json", provenance, indent=2)
    model_fingerprint = core.canonical_hash(
        {
            "model": "dinov2_vitl14",
            "checkpoint": provenance["checkpoint_sha256"]["dinov2"],
            "sam2": provenance["checkpoint_sha256"]["sam2"],
            "descriptor": "sam2-mask-pooled-dinov2",
            "xformers_disabled": provenance["dinov2_xformers_disabled"],
        }
    )

    core.log("loading official UPN, DINOv2 ViT-L/14, and SAM2.1-L models")
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

    novel_cache = cache_dir / f"coco_{args.shot}shot_seed{args.seed}_novel.pt"
    novel_prototypes, novel_proto_classes = core.load_or_build_prototypes(
        cache_path=novel_cache,
        support=novel_support,
        support_util=support_util,
        sam2_predictor=sam2_predictor,
        feat_extractor=feat_extractor,
        image_transform=image_transform,
        data_dir=data_dir,
        device=args.device,
        role="coco_novel",
        model_fingerprint=model_fingerprint,
    )

    base_support = build_coco_base_support(
        train_json, base_classes, args.base_samples_per_class
    )
    base_support_hash = core.canonical_hash(core.support_for_extractor(base_support))
    base_manifest = cache_dir / f"coco_base_support_{args.base_samples_per_class}.json"
    core.atomic_json(base_manifest, base_support, indent=2)
    base_cache = cache_dir / f"coco_base_{args.base_samples_per_class}.pt"
    base_prototypes, base_proto_classes = core.load_or_build_prototypes(
        cache_path=base_cache,
        support=base_support,
        support_util=support_util,
        sam2_predictor=sam2_predictor,
        feat_extractor=feat_extractor,
        image_transform=image_transform,
        data_dir=data_dir,
        device=args.device,
        role="coco_base",
        model_fingerprint=model_fingerprint,
    )

    _, coco_loader = query_util.load_voc2007_coco_json(str(test_json), str(test_img_dir))
    all_images = list(coco_loader.dataset["images"])
    if args.max_images > 0:
        all_images = all_images[: args.max_images]
    if not all_images:
        raise RuntimeError("No COCO validation images selected")

    run_signature = core.canonical_hash(
        {
            "schema": 1,
            "fsod_vfm_commit": provenance["fsod_vfm_commit"],
            "runner_sha256": provenance["runner_sha256"],
            "shared_runner_sha256": provenance["shared_runner_sha256"],
            "official_source_sha256": provenance["official_source_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "support_json_sha256": provenance["support_json_sha256"],
            "train_json_sha256": provenance["train_json_sha256"],
            "test_json_sha256": provenance["test_json_sha256"],
            "base_support_sha256": base_support_hash,
            "dinov2_xformers_disabled": provenance["dinov2_xformers_disabled"],
            "shot": args.shot,
            "seed": args.seed,
            "selected_image_ids": [int(row["id"]) for row in all_images],
            "guard": provenance["guard"],
            "official_inference": provenance["official_inference"],
        }
    )
    core.atomic_json(
        output_dir / "run_manifest.json",
        {
            "run_signature": run_signature,
            "image_count": len(all_images),
            "selected_image_ids_sha256": core.canonical_hash(
                [int(row["id"]) for row in all_images]
            ),
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
    timings: List[float] = []
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
                core.log(f"resuming completed COCO shard {start}:{stop}")
            else:
                core.log(
                    f"running official FSOD-VFM COCO shard {start}:{stop} "
                    f"({stop}/{len(all_images)})"
                )
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
                paired = core.apply_guard_to_shard(
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
                core.atomic_json(shard_path, shard)
                core.log(
                    f"finished COCO shard {start}:{stop} in {elapsed:.1f}s; "
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

    baseline_identity = core.identity_hash(completed_baseline)
    guard_identity = core.identity_hash(completed_guard)
    if baseline_identity != guard_identity:
        raise RuntimeError("Guard changed a category, box, image, or prediction order")
    for index, row in enumerate(completed_diagnostics):
        row["prediction_index"] = index
    core.atomic_json(output_dir / "predictions_baseline.json", completed_baseline)
    core.atomic_json(output_dir / "predictions_guard.json", completed_guard)

    selected_ids = {int(row["id"]) for row in all_images}
    evaluation_gt = test_json
    if len(all_images) != len(original_images):
        evaluation_gt = core.subset_ground_truth(
            test_json, output_dir / "ground_truth_subset.json", selected_ids
        )
    summary: Dict[str, Any] = {
        "schema": 1,
        "run_signature": run_signature,
        "completed_at": dt.datetime.now().astimezone().isoformat(),
        "dataset": "COCO2017-val-60base-20novel",
        "shot": args.shot,
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
        core.log("evaluating paired COCO baseline and Guard predictions")
        baseline_eval = core.evaluate_coco_bbox(
            evaluation_gt, completed_baseline, novel_classes
        )
        guard_eval = core.evaluate_coco_bbox(evaluation_gt, completed_guard, novel_classes)
        target_ids = set(int(value) for value in baseline_eval["target_category_ids"])
        baseline_match = core.match_predictions(
            evaluation_gt, completed_baseline, target_ids
        )
        guard_match = core.match_predictions(evaluation_gt, completed_guard, target_ids)
        baseline_match_05 = core.match_predictions(
            evaluation_gt, completed_baseline, target_ids, score_threshold=0.50
        )
        guard_match_05 = core.match_predictions(
            evaluation_gt, completed_guard, target_ids, score_threshold=0.50
        )
        failure = core.attach_failure_labels(
            completed_diagnostics,
            baseline_match["by_index"],
            guard_match["by_index"],
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
            "baseline": core.calibration_metrics(
                completed_baseline, baseline_match["by_index"], bins=15
            ),
            "guard": core.calibration_metrics(
                completed_guard, guard_match["by_index"], bins=15
            ),
        }
        summary["recall"] = {
            "baseline_score_0": baseline_match["recall"],
            "guard_score_0": guard_match["recall"],
            "baseline_score_0.5": baseline_match_05["recall"],
            "guard_score_0.5": guard_match_05["recall"],
            "ground_truths": baseline_match["ground_truths"],
        }
        summary["failure_cases"] = {
            key: value for key, value in failure.items() if key != "events"
        }
        core.atomic_json(output_dir / "failure_events.json", failure["events"])
        core.atomic_json(output_dir / "diagnostics.json", completed_diagnostics)

    core.atomic_json(output_dir / "summary.json", summary, indent=2)
    core.log(
        "COCO complete: "
        + json.dumps(
            {
                "shot": args.shot,
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
