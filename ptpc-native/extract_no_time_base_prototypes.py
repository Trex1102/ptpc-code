#!/usr/bin/env python3
"""Extract the post-processed No-Time-to-Train base memory prototypes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--novel-classes", default="bird,bus,cow,motorbike,sofa"
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    candidates = [
        key for key in state
        if key.endswith("seg_model.mem_feats_ins_avg")
        or key.endswith("mem_feats_ins_avg")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"could not identify one memory tensor: {candidates}")
    memory = state[candidates[0]].float()
    if not torch.count_nonzero(memory):
        raw_keys = [key for key in state if key.endswith("seg_model.mem_feats") or key.endswith("mem_feats")]
        mask_keys = [key for key in state if key.endswith("seg_model.mem_masks") or key.endswith("mem_masks")]
        if len(raw_keys) != 1 or len(mask_keys) != 1:
            raise RuntimeError("raw memory tensors are ambiguous")
        raw = state[raw_keys[0]].float()
        masks = state[mask_keys[0]].float()
        memory = (raw * masks.unsqueeze(-1)).sum(dim=2) / masks.sum(dim=2).clamp_min(1).unsqueeze(-1)
    if memory.ndim != 3 or memory.shape[-1] != 1024:
        raise RuntimeError(f"unexpected memory shape: {tuple(memory.shape)}")
    prototypes = F.normalize(memory.mean(dim=1), p=2, dim=-1)

    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    novel = {
        name.strip() for name in args.novel_classes.split(",") if name.strip()
    }
    if not novel:
        raise ValueError("--novel-classes must not be empty")
    categories = sorted(
        (row for row in annotation["categories"] if row["name"] not in novel),
        key=lambda row: int(row["id"]),
    )
    names = [row["name"] for row in categories]
    category_ids = [int(row["id"]) for row in categories]
    if len(names) != prototypes.shape[0]:
        raise RuntimeError(
            f"class/prototype mismatch: {len(names)} vs {prototypes.shape[0]}"
        )

    payload = {
        "schema": 1,
        "prototypes": prototypes,
        "class_names": names,
        "category_ids": category_ids,
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "descriptor": "No-Time-to-Train mask-pooled DINOv2 ViT-L/14 memory mean",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
