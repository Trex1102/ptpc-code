#!/usr/bin/env python3
"""Run the Table 6 guard-selectivity VOC 5-shot sweep for PTPC."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PTPC_BUCKET_ROWS = [
    ("base_gt_overlap", "Base GT overlap"),
    ("novel_gt_overlap", "Novel GT overlap"),
    ("background_no_gt_overlap", "Background / no GT overlap"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PTPC guard-selectivity VOC 5-shot suite")
    parser.add_argument("--exp-name", default="vanilla_defrcn")
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--imagenet-pretrain-torch",
        default=".pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth",
    )
    parser.add_argument(
        "--output-root",
        default="checkpoints/voc/vanilla_defrcn/ptpc_guard_selectivity_voc5_no_cropcls",
    )
    parser.add_argument("--crop-cls-weight", type=float, default=0.0)
    parser.add_argument("--det-weight", type=float, default=0.4)
    parser.add_argument("--fm-weight", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--competitive-mode", default="softmax")
    parser.add_argument("--guard-enable", default="True")
    parser.add_argument("--guard-margin", type=float, default=0.05)
    parser.add_argument("--guard-suppression-factor", type=float, default=0.3)
    parser.add_argument("--max-base-samples-per-class", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], stdout_path: Path, env: dict[str, str]) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as handle:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT)


def build_config(split: int, shot: int, seed: int) -> Path:
    subprocess.run(
        [
            sys.executable,
            "tools/create_config.py",
            "--dataset",
            "voc",
            "--config_root",
            "configs/voc",
            "--shot",
            str(shot),
            "--seed",
            str(seed),
            "--setting",
            "fsod",
            "--split",
            str(split),
        ],
        cwd=ROOT,
        check=True,
    )
    return ROOT / f"configs/voc/defrcn_fsod_r101_novel{split}_{shot}shot_seed{seed}.yaml"


def eval_complete(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text()
    return "copypaste: AP,AP50,AP75" in text


def diag_complete(diag_path: Path) -> bool:
    if not diag_path.exists():
        return False
    try:
        payload = json.loads(diag_path.read_text())
    except json.JSONDecodeError:
        return False
    return int(payload.get("checked_total", 0)) > 0


def aggregate_worker_summaries(output_dir: Path) -> Path:
    canonical_path = output_dir / "guard_selectivity_summary.json"
    worker_paths = sorted(
        path for path in output_dir.glob("guard_selectivity_summary*.json")
        if path.name != canonical_path.name
    )
    if not worker_paths:
        if canonical_path.exists():
            return canonical_path
        raise FileNotFoundError(f"Missing worker guard summaries under {output_dir}")

    aggregate = {
        "checked_total": 0,
        "suppressed_total": 0,
        "buckets": {
            key: {"checked": 0, "suppressed": 0, "delta_sum": 0.0}
            for key, _ in PTPC_BUCKET_ROWS
        },
    }
    payloads = []

    for path in worker_paths:
        payload = json.loads(path.read_text())
        payloads.append(payload)
        aggregate["checked_total"] += int(payload["checked_total"])
        aggregate["suppressed_total"] += int(payload["suppressed_total"])
        for key, _ in PTPC_BUCKET_ROWS:
            bucket = payload["buckets"][key]
            checked = int(bucket["checked"])
            suppressed = int(bucket["suppressed"])
            aggregate["buckets"][key]["checked"] += checked
            aggregate["buckets"][key]["suppressed"] += suppressed
            aggregate["buckets"][key]["delta_sum"] += float(bucket["mean_delta_s"]) * checked

    first = payloads[0]
    canonical_payload = {
        "dataset_test": first["dataset_test"],
        "dataset_train": first["dataset_train"],
        "output_dir": first["output_dir"],
        "fm_model_name": first["fm_model_name"],
        "crop_cls_weight": first["crop_cls_weight"],
        "guard_margin": first["guard_margin"],
        "suppression_factor": first["suppression_factor"],
        "score_band": first["score_band"],
        "checked_total": aggregate["checked_total"],
        "suppressed_total": aggregate["suppressed_total"],
        "worker_files": [path.name for path in worker_paths],
        "buckets": {},
    }
    for key, _ in PTPC_BUCKET_ROWS:
        checked = aggregate["buckets"][key]["checked"]
        suppressed = aggregate["buckets"][key]["suppressed"]
        canonical_payload["buckets"][key] = {
            "checked": checked,
            "suppressed": suppressed,
            "suppression_rate": (suppressed / checked) if checked else 0.0,
            "mean_delta_s": (aggregate["buckets"][key]["delta_sum"] / checked) if checked else 0.0,
        }

    canonical_path.write_text(json.dumps(canonical_payload, indent=2))
    return canonical_path


def ptpc_opts(args: argparse.Namespace) -> list[str]:
    return [
        "TEST.PCB_ENABLE",
        "True",
        "NOVEL_METHODS.ENABLE",
        "True",
        "NOVEL_METHODS.METHOD",
        "ptpc",
        "NOVEL_METHODS.PTPC.ENABLE",
        "True",
        "NOVEL_METHODS.PTPC.FM_ONLY",
        "True",
        "NOVEL_METHODS.PTPC.FM_MODEL_NAME",
        "dinov2_vitb14",
        "NOVEL_METHODS.PTPC.FM_FEAT_DIM",
        "768",
        "NOVEL_METHODS.PTPC.IMAGE_SIZE",
        "518",
        "NOVEL_METHODS.PTPC.CROP_SIZE",
        "224",
        "NOVEL_METHODS.PTPC.PATCH_SIZE",
        "14",
        "NOVEL_METHODS.PTPC.DET_WEIGHT",
        str(args.det_weight),
        "NOVEL_METHODS.PTPC.FM_WEIGHT",
        str(args.fm_weight),
        "NOVEL_METHODS.PTPC.CROP_CLS_WEIGHT",
        str(args.crop_cls_weight),
        "NOVEL_METHODS.PTPC.TEMPERATURE",
        str(args.temperature),
        "NOVEL_METHODS.PTPC.COMPETITIVE_MODE",
        args.competitive_mode,
        "NOVEL_METHODS.PTPC_GUARD.ENABLE",
        args.guard_enable,
        "NOVEL_METHODS.PTPC_GUARD.MARGIN",
        str(args.guard_margin),
        "NOVEL_METHODS.PTPC_GUARD.SUPPRESSION_FACTOR",
        str(args.guard_suppression_factor),
        "NOVEL_METHODS.PTPC_GUARD.MAX_BASE_SAMPLES_PER_CLASS",
        str(args.max_base_samples_per_class),
        "TEST.PCB_PROTO_REASSIGN",
        "False",
    ]


def dataset_test_tuple(split: int) -> str:
    return f"(\"voc_2007_test_all{split}\",)"


def run_split(split: int, args: argparse.Namespace, output_root: Path) -> Path:
    output_dir = output_root / f"split{split}" / f"{args.shot}shot_seed{args.seed}"
    log_path = output_dir / "log.txt"
    diag_path = output_dir / "guard_selectivity_summary.json"

    if not args.force and eval_complete(log_path) and diag_complete(diag_path):
        return diag_path

    config_path = build_config(split, args.shot, args.seed)
    ckpt = ROOT / f"checkpoints/voc/{args.exp_name}/split{split}/{args.shot}shot_seed{args.seed}/model_final.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    env = os.environ.copy()
    env["PTPC_GUARD_DIAG_SUMMARY_PATH"] = str(diag_path)

    cmd = [
        sys.executable,
        "main.py",
        "--eval-only",
        "--num-gpus",
        str(args.num_gpus),
        "--config-file",
        str(config_path.relative_to(ROOT)),
        "--opts",
        "MODEL.WEIGHTS",
        str(ckpt.relative_to(ROOT)),
        "OUTPUT_DIR",
        str(output_dir.relative_to(ROOT)),
        "DATASETS.TEST",
        dataset_test_tuple(split),
        "TEST.PCB_MODELPATH",
        args.imagenet_pretrain_torch,
        *ptpc_opts(args),
    ]
    try:
        run_cmd(cmd, log_path, env)
    finally:
        try:
            config_path.unlink()
        except FileNotFoundError:
            pass

    diag_path = aggregate_worker_summaries(output_dir)
    if not diag_complete(diag_path):
        raise RuntimeError(f"Guard summary is missing or empty: {diag_path}")
    return diag_path


def aggregate_summaries(summary_paths: list[Path], output_root: Path, args: argparse.Namespace) -> None:
    aggregate = {
        "checked_total": 0,
        "suppressed_total": 0,
        "buckets": {
            key: {"checked": 0, "suppressed": 0, "delta_sum": 0.0}
            for key, _ in PTPC_BUCKET_ROWS
        },
    }

    split_payloads = []
    for path in summary_paths:
        payload = json.loads(path.read_text())
        split_payloads.append(payload)
        aggregate["checked_total"] += int(payload["checked_total"])
        aggregate["suppressed_total"] += int(payload["suppressed_total"])
        for key, _ in PTPC_BUCKET_ROWS:
            bucket = payload["buckets"][key]
            checked = int(bucket["checked"])
            suppressed = int(bucket["suppressed"])
            aggregate["buckets"][key]["checked"] += checked
            aggregate["buckets"][key]["suppressed"] += suppressed
            aggregate["buckets"][key]["delta_sum"] += float(bucket["mean_delta_s"]) * checked

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    aggregate_json = {
        "dataset_test": "voc_2007_test_all{1,2,3}",
        "exp_name": args.exp_name,
        "shot": args.shot,
        "seed": args.seed,
        "crop_cls_weight": args.crop_cls_weight,
        "checked_total": aggregate["checked_total"],
        "suppressed_total": aggregate["suppressed_total"],
        "splits": split_payloads,
        "buckets": {},
    }

    for key, _ in PTPC_BUCKET_ROWS:
        checked = aggregate["buckets"][key]["checked"]
        suppressed = aggregate["buckets"][key]["suppressed"]
        aggregate_json["buckets"][key] = {
            "checked": checked,
            "suppressed": suppressed,
            "suppression_rate": (suppressed / checked) if checked else 0.0,
            "mean_delta_s": (aggregate["buckets"][key]["delta_sum"] / checked) if checked else 0.0,
        }

    (summary_dir / "guard_selectivity_aggregate.json").write_text(
        json.dumps(aggregate_json, indent=2)
    )

    with (summary_dir / "guard_selectivity_aggregate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "checked", "suppressed", "suppression_rate", "mean_delta_s"])
        for key, label in PTPC_BUCKET_ROWS:
            bucket = aggregate_json["buckets"][key]
            writer.writerow(
                [
                    label,
                    bucket["checked"],
                    bucket["suppressed"],
                    f"{bucket['suppression_rate'] * 100.0:.4f}",
                    f"{bucket['mean_delta_s']:.6f}",
                ]
            )

    md_lines = [
        f"# Guard Selectivity Summary (crop-cls weight = {args.crop_cls_weight})",
        "",
        f"Checked detections: {aggregate_json['checked_total']}",
        "",
        "| Overlap bucket | Checked | Suppressed | Suppression rate | Mean delta s |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in PTPC_BUCKET_ROWS:
        bucket = aggregate_json["buckets"][key]
        md_lines.append(
            "| "
            f"{label} | {bucket['checked']} | {bucket['suppressed']} | "
            f"{bucket['suppression_rate'] * 100.0:.1f}% | {bucket['mean_delta_s']:.3f} |"
        )
    (summary_dir / "guard_selectivity_aggregate.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    summary_paths = [run_split(split, args, output_root) for split in args.splits]
    aggregate_summaries(summary_paths, output_root, args)


if __name__ == "__main__":
    main()
