#!/usr/bin/env python3
"""Create deterministic paired novel/base support sets with the released sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

from pycocotools.coco import COCO

from no_time_to_train.dataset import few_shot_sampling as released


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample(
    annotation: Path,
    output: Path,
    names: list[str],
    shot: int,
    seed: int,
) -> dict[int, list[dict]]:
    # The released function selects METAINFO['default_classes'] for dataset=coco.
    # Rebinding that one entry lets us reuse its exact shuffling, validation and
    # distinct-image logic for an explicit Pascal class group.
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
        raise RuntimeError(f"expected {len(names)} classes, got {len(payload)}")
    if any(len(rows) != shot for rows in payload.values()):
        raise RuntimeError(f"support count is not exactly {shot} for every class")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--novel-output", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--novel-shot", type=int, default=5)
    parser.add_argument("--base-shot", type=int, default=20)
    args = parser.parse_args()

    novel = NOVEL_BY_SPLIT[args.split]
    base = [name for name in ALL_CLASSES if name not in novel]

    novel_payload = sample(
        args.annotation, args.novel_output, novel, args.novel_shot, args.seed
    )
    base_payload = sample(
        args.annotation, args.base_output, base, args.base_shot, args.seed
    )

    coco = COCO(str(args.annotation))
    category_names = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

    def serialise(payload: dict[int, list[dict]]) -> dict[str, list[dict]]:
        return {
            category_names[int(category_id)]: rows
            for category_id, rows in payload.items()
        }

    manifest = {
        "schema": 1,
        "protocol": "released No-Time-to-Train random sampler, distinct images",
        "split": args.split,
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
