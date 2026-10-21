#!/usr/bin/env python3
"""Materialize typed No-Time COCO configs with a frozen score-only Guard."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from prepare_no_time_coco_support import DEFAULT_NOVEL


def build(
    template: dict,
    *,
    mode: str,
    category_num: int,
    memory_length: int,
    class_names: list[str],
    support: Path,
    log_dir: Path,
    guard_prototypes: Path | None = None,
    experiment_dir: Path | None = None,
    guard: tuple[float, float, float, float] | None = None,
) -> dict:
    config = copy.deepcopy(template)
    init = config["model"]["init_args"]
    init["test_mode"] = mode
    memory = init["model_cfg"]["memory_bank_cfg"]
    memory.update({"enable": True, "category_num": category_num, "length": memory_length})
    fill = init["dataset_cfgs"]["fill_memory"]
    fill.update(
        {
            "memory_pkl": str(support),
            "memory_length": memory_length,
            "cat_names": class_names,
        }
    )
    if mode == "test":
        if guard is None:
            raise ValueError("Guard parameters are required for inference")
        margin, suppression, lower, upper = guard
        test = init["dataset_cfgs"]["test"]
        test.update({"cat_names": DEFAULT_NOVEL, "class_split": "few_shot_classes"})
        init["model_cfg"].update(
            {
                "guard_base_prototypes_path": str(guard_prototypes),
                "guard_margin": margin,
                "guard_suppression": suppression,
                "guard_lower": lower,
                "guard_upper": upper,
                "exp_folder": str(experiment_dir),
            }
        )
    config["trainer"]["devices"] = 1
    for logger in config["trainer"].get("logger", []):
        logger.setdefault("init_args", {})["save_dir"] = str(log_dir)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--novel-support", type=Path, required=True)
    parser.add_argument("--base-support", type=Path, required=True)
    parser.add_argument("--base-prototypes", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--shot", type=int, choices=(10, 30), required=True)
    parser.add_argument("--guard-margin", type=float, default=0.01)
    parser.add_argument("--guard-suppression", type=float, default=0.50)
    parser.add_argument("--guard-lower", type=float, default=0.0)
    parser.add_argument("--guard-upper", type=float, default=0.70)
    args = parser.parse_args()
    if not 0.0 < args.guard_suppression <= 1.0:
        raise ValueError("--guard-suppression must lie in (0, 1]")
    if args.guard_lower >= args.guard_upper:
        raise ValueError("--guard-lower must be below --guard-upper")

    source = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    annotation_payload = json.loads(args.annotation.read_text(encoding="utf-8"))
    all_names = [
        str(row["name"])
        for row in sorted(annotation_payload["categories"], key=lambda row: int(row["id"]))
    ]
    base = [name for name in all_names if name not in set(DEFAULT_NOVEL)]
    if len(base) != 60:
        raise RuntimeError(f"expected 60 COCO base classes, got {len(base)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "novel_fill.yaml": build(
            source, mode="fill_memory", category_num=20,
            memory_length=args.shot, class_names=DEFAULT_NOVEL,
            support=args.novel_support, log_dir=args.output_dir / "novel_fill",
        ),
        "novel_postprocess.yaml": build(
            source, mode="postprocess_memory", category_num=20,
            memory_length=args.shot, class_names=DEFAULT_NOVEL,
            support=args.novel_support, log_dir=args.output_dir / "novel_postprocess",
        ),
        "base_fill.yaml": build(
            source, mode="fill_memory", category_num=60,
            memory_length=20, class_names=base,
            support=args.base_support, log_dir=args.output_dir / "base_fill",
        ),
        "guard_inference.yaml": build(
            source, mode="test", category_num=20,
            memory_length=args.shot, class_names=DEFAULT_NOVEL,
            support=args.novel_support, log_dir=args.output_dir / "inference",
            guard_prototypes=args.base_prototypes, experiment_dir=args.output_dir,
            guard=(args.guard_margin, args.guard_suppression, args.guard_lower, args.guard_upper),
        ),
    }
    for name, payload in definitions.items():
        (args.output_dir / name).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
