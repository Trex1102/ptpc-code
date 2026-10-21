#!/usr/bin/env python3
"""Create deterministic No-Time COCO novel/base memories with its released sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

from pycocotools.coco import COCO

from no_time_to_train.dataset import few_shot_sampling as released


DEFAULT_NOVEL = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "boat", "bird", "cat", "dog", "horse", "sheep", "cow", "bottle",
    "chair", "couch", "potted plant", "dining table", "tv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample(
    annotation: Path, output: Path, names: list[str], shot: int, seed: int
) -> dict[int, list[dict]]:
    released.METAINFO["default_classes"] = tuple(names)
    random.seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    released.sample_memory_dataset(
        str(annotation),
        str(output),
        shot,
        remove_bad=True,
        dataset="coco",
        heuristics=False,
    )
    with output.open("rb") as handle:
        payload = pickle.load(handle)
    if len(payload) != len(names):
        raise RuntimeError(f"expected {len(names)} support classes, got {len(payload)}")
    if any(len(rows) != shot for rows in payload.values()):
        raise RuntimeError("support count is not exact for every class")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--novel-output", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--novel-shot", type=int, choices=(10, 30), required=True)
    parser.add_argument("--base-shot", type=int, default=20)
    parser.add_argument("--novel-classes", default=",".join(DEFAULT_NOVEL))
    args = parser.parse_args()

    novel = [name.strip() for name in args.novel_classes.split(",") if name.strip()]
    if len(novel) != 20 or len(set(novel)) != 20:
        raise ValueError("--novel-classes must contain 20 unique class names")
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    categories = sorted(annotation["categories"], key=lambda row: int(row["id"]))
    all_names = [str(row["name"]) for row in categories]
    missing = sorted(set(novel) - set(all_names))
    if missing:
        raise ValueError(f"novel classes missing from annotation: {missing}")
    base = [name for name in all_names if name not in set(novel)]
    if len(base) != 60:
        raise RuntimeError(f"expected 60 base classes, got {len(base)}")

    novel_payload = sample(
        args.annotation, args.novel_output, novel, args.novel_shot, args.seed
    )
    base_payload = sample(
        args.annotation, args.base_output, base, args.base_shot, args.seed
    )
    coco = COCO(str(args.annotation))
    names_by_id = {
        int(row["id"]): str(row["name"])
        for row in coco.loadCats(coco.getCatIds())
    }

    def serialise(payload: dict[int, list[dict]]) -> dict[str, list[dict]]:
        return {names_by_id[int(key)]: value for key, value in payload.items()}

    manifest = {
        "schema": "no-time-coco-support-v1",
        "protocol": "released No-Time random sampler; distinct train2017 images",
        "seed": args.seed,
        "annotation": str(args.annotation.resolve()),
        "annotation_sha256": sha256(args.annotation),
        "novel_shot": args.novel_shot,
        "base_shot": args.base_shot,
        "novel": serialise(novel_payload),
        "base": serialise(base_payload),
        "novel_pickle_sha256": sha256(args.novel_output),
        "base_pickle_sha256": sha256(args.base_output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
