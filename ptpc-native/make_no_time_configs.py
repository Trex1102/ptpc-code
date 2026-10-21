#!/usr/bin/env python3
"""Materialize typed No-Time-to-Train configs without fragile dotted CLI merges."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


NOVEL_BY_SPLIT = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}
ALL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


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
    split: int = 1,
    novel_classes: list[str] | None = None,
    guard_margin: float = 0.01,
    guard_suppression: float = 0.70,
    guard_lower: float = 0.0,
    guard_upper: float = 0.70,
) -> dict:
    config = copy.deepcopy(template)
    init = config["model"]["init_args"]
    init["test_mode"] = mode
    memory = init["model_cfg"]["memory_bank_cfg"]
    memory["enable"] = True
    memory["category_num"] = category_num
    memory["length"] = memory_length

    fill = init["dataset_cfgs"]["fill_memory"]
    fill["memory_pkl"] = str(support)
    fill["memory_length"] = memory_length
    fill["cat_names"] = class_names

    if mode == "test":
        if novel_classes is None:
            raise ValueError("novel_classes are required for inference")
        test = init["dataset_cfgs"]["test"]
        test["cat_names"] = novel_classes
        test["class_split"] = f"pascal_voc_split_{split}"
        model_cfg = init["model_cfg"]
        model_cfg.update({
            "guard_base_prototypes_path": str(guard_prototypes),
            "guard_margin": guard_margin,
            "guard_suppression": guard_suppression,
            "guard_lower": guard_lower,
            "guard_upper": guard_upper,
            "exp_folder": str(experiment_dir),
        })

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
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--shot", type=int, choices=(1, 2, 3, 5, 10), default=5)
    parser.add_argument("--guard-margin", type=float, default=0.01)
    parser.add_argument("--guard-suppression", type=float, default=0.70)
    parser.add_argument("--guard-lower", type=float, default=0.0)
    parser.add_argument("--guard-upper", type=float, default=0.70)
    args = parser.parse_args()

    if not 0.0 < args.guard_suppression <= 1.0:
        raise ValueError("--guard-suppression must lie in (0, 1]")
    if args.guard_lower >= args.guard_upper:
        raise ValueError("--guard-lower must be below --guard-upper")
    novel = NOVEL_BY_SPLIT[args.split]
    base = [name for name in ALL_CLASSES if name not in novel]

    template = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "novel_fill.yaml": build(
            template,
            mode="fill_memory",
            category_num=5,
            memory_length=args.shot,
            class_names=novel,
            support=args.novel_support,
            log_dir=args.output_dir / "novel_fill",
        ),
        "novel_postprocess.yaml": build(
            template,
            mode="postprocess_memory",
            category_num=5,
            memory_length=args.shot,
            class_names=novel,
            support=args.novel_support,
            log_dir=args.output_dir / "novel_postprocess",
        ),
        "base_fill.yaml": build(
            template,
            mode="fill_memory",
            category_num=15,
            memory_length=20,
            class_names=base,
            support=args.base_support,
            log_dir=args.output_dir / "base_fill",
        ),
        "guard_inference.yaml": build(
            template,
            mode="test",
            category_num=5,
            memory_length=args.shot,
            class_names=novel,
            support=args.novel_support,
            log_dir=args.output_dir / "inference",
            guard_prototypes=args.base_prototypes,
            experiment_dir=args.output_dir,
            split=args.split,
            novel_classes=novel,
            guard_margin=args.guard_margin,
            guard_suppression=args.guard_suppression,
            guard_lower=args.guard_lower,
            guard_upper=args.guard_upper,
        ),
    }
    for name, payload in definitions.items():
        (args.output_dir / name).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
