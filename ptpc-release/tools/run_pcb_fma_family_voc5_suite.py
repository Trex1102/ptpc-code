#!/usr/bin/env python3
"""Run PCB-FMA family VOC 5-shot evaluation and calibration summaries.

This script runs PCB-FMA and PCB-FMA-Enhanced-noaug on VOC split 1/2/3 for a
given shot, computes calibration metrics for each split, and exports both
split-wise and averaged comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

METHOD_SPECS = {
    "pcb-fma": {
        "subdir": "pcb-fma",
        "opts": [
            "TEST.PCB_ENABLE", "True",
            "NOVEL_METHODS.ENABLE", "True",
            "NOVEL_METHODS.METHOD", "pcb-fma",
            "NOVEL_METHODS.PCB_FMA.ENABLE", "True",
            "NOVEL_METHODS.PCB_FMA.FM_MODEL_NAME", "dinov2_vitb14",
            "NOVEL_METHODS.PCB_FMA.FM_FEAT_DIM", "768",
            "NOVEL_METHODS.PCB_FMA.ROI_SIZE", "224",
            "NOVEL_METHODS.PCB_FMA.DET_WEIGHT", "0.4",
            "NOVEL_METHODS.PCB_FMA.FM_WEIGHT", "0.6",
            "NOVEL_METHODS.PCB_FMA.USE_ORIGINAL_PCB", "False",
            "NOVEL_METHODS.PCB_FMA.BATCH_SIZE", "32",
            "TEST.PCB_PROTO_REASSIGN", "False",
        ],
    },
    "pcb-fma-enhanced-noaug": {
        "subdir": "pcb-fma-enhanced-noaug",
        "opts": [
            "TEST.PCB_ENABLE", "True",
            "NOVEL_METHODS.ENABLE", "True",
            "NOVEL_METHODS.METHOD", "pcb-fma-enhanced",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE", "True",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME", "dinov2_vitb14",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM", "768",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.ROI_SIZE", "224",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT", "0.4",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT", "0.6",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB", "False",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.BATCH_SIZE", "32",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP", "False",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP", "False",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE", "0.1",
            "NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE", "softmax",
            "TEST.PCB_PROTO_REASSIGN", "False",
        ],
    },
}


@dataclass
class MethodSplitSummary:
    method: str
    split: str
    shot: int
    seed: int
    ap: float
    ap50: float
    ap75: float
    nap: float
    nap50: float
    nap75: float
    ece: float
    brier: float
    latency_ms_per_img: float
    compute_latency_ms_per_img: float
    throughput_img_per_s: float
    calibration_json: str
    eval_log: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCB-FMA family VOC 5-shot suite")
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
        default="checkpoints/voc/vanilla_defrcn/pcb_fma_family_voc5_suite",
        help="Root directory for method/split outputs and summary artifacts.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pcb-fma", "pcb-fma-enhanced-noaug"],
        choices=sorted(METHOD_SPECS.keys()),
    )
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--force-calibration", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], stdout_path: Path | None = None) -> None:
    if stdout_path is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def build_config(split: int, shot: int, seed: int) -> Path:
    run_cmd(
        [
            sys.executable,
            "tools/create_config.py",
            "--dataset", "voc",
            "--config_root", "configs/voc",
            "--shot", str(shot),
            "--seed", str(seed),
            "--setting", "fsod",
            "--split", str(split),
        ]
    )
    return ROOT / f"configs/voc/defrcn_fsod_r101_novel{split}_{shot}shot_seed{seed}.yaml"


def eval_complete(log_path: Path) -> bool:
    return log_path.exists() and "copypaste: AP,AP50,AP75,nAP,nAP50,nAP75" in log_path.read_text()


def calibration_complete(json_path: Path) -> bool:
    return json_path.exists()


def run_eval(
    method: str,
    split: int,
    shot: int,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[Path, Path]:
    config_path = build_config(split, shot, seed)
    ckpt = ROOT / f"checkpoints/voc/{args.exp_name}/split{split}/{shot}shot_seed{seed}/model_final.pth"
    method_dir = output_root / METHOD_SPECS[method]["subdir"]
    output_dir = method_dir / f"split{split}" / f"{shot}shot_seed{seed}"
    log_path = output_dir / "log.txt"
    cal_path = output_dir / "calibration_metrics.json"

    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    if args.force_eval or not eval_complete(log_path):
        cmd = [
            sys.executable,
            "main.py",
            "--eval-only",
            "--num-gpus", str(args.num_gpus),
            "--config-file", str(config_path.relative_to(ROOT)),
            "--opts",
            "MODEL.WEIGHTS", str(ckpt.relative_to(ROOT)),
            "OUTPUT_DIR", str(output_dir.relative_to(ROOT)),
            "TEST.PCB_MODELPATH", args.imagenet_pretrain_torch,
            *METHOD_SPECS[method]["opts"],
        ]
        run_cmd(cmd, stdout_path=log_path)

    if args.force_calibration or not calibration_complete(cal_path):
        cmd = [
            sys.executable,
            "tools/compute_calibration_metrics.py",
            "--config-file", str(config_path.relative_to(ROOT)),
            "--output", str(cal_path.relative_to(ROOT)),
            "--opts",
            "MODEL.WEIGHTS", str(ckpt.relative_to(ROOT)),
            "OUTPUT_DIR", str(output_dir.relative_to(ROOT)),
            "TEST.PCB_MODELPATH", args.imagenet_pretrain_torch,
            *METHOD_SPECS[method]["opts"],
        ]
        run_cmd(cmd)

    try:
        config_path.unlink()
    except FileNotFoundError:
        pass
    return log_path, cal_path


def extract_ap_metrics(log_text: str) -> dict[str, float]:
    if "copypaste: AP,AP50,AP75,nAP,nAP50,nAP75" not in log_text:
        raise ValueError("Could not find copypaste AP header in log.")

    value_matches = re.findall(r"copypaste: ([0-9eE+\-., ]+)", log_text)
    numeric_line = None
    for match in value_matches:
        parts = [x.strip() for x in match.split(",")]
        if len(parts) == 6 and all(parts):
            numeric_line = [float(x) for x in parts]
    if numeric_line is None:
        raise ValueError("Could not find copypaste AP values in log.")

    keys = ["ap", "ap50", "ap75", "nap", "nap50", "nap75"]
    return dict(zip(keys, numeric_line))


def extract_latency_metrics(log_text: str) -> dict[str, float]:
    wall_match = re.search(r"Total inference time: .* \(([0-9.]+) s / img per device", log_text)
    compute_match = re.search(r"Total inference pure compute time: .* \(([0-9.]+) s / img per device", log_text)
    if wall_match is None or compute_match is None:
        raise ValueError("Could not find latency lines in log.")

    wall = float(wall_match.group(1))
    compute = float(compute_match.group(1))
    return {
        "latency_ms_per_img": wall * 1000.0,
        "compute_latency_ms_per_img": compute * 1000.0,
        "throughput_img_per_s": 1.0 / wall if wall > 0 else 0.0,
    }


def build_summary(
    method: str,
    split: int,
    shot: int,
    seed: int,
    log_path: Path,
    cal_path: Path,
) -> MethodSplitSummary:
    log_text = log_path.read_text()
    ap_metrics = extract_ap_metrics(log_text)
    latency_metrics = extract_latency_metrics(log_text)
    calibration = json.loads(cal_path.read_text())

    return MethodSplitSummary(
        method=method,
        split=f"split{split}",
        shot=shot,
        seed=seed,
        ap=ap_metrics["ap"],
        ap50=ap_metrics["ap50"],
        ap75=ap_metrics["ap75"],
        nap=ap_metrics["nap"],
        nap50=ap_metrics["nap50"],
        nap75=ap_metrics["nap75"],
        ece=float(calibration["ece"]),
        brier=float(calibration["brier"]),
        latency_ms_per_img=latency_metrics["latency_ms_per_img"],
        compute_latency_ms_per_img=latency_metrics["compute_latency_ms_per_img"],
        throughput_img_per_s=latency_metrics["throughput_img_per_s"],
        calibration_json=str(cal_path.relative_to(ROOT)),
        eval_log=str(log_path.relative_to(ROOT)),
    )


def method_mean_row(rows: Iterable[MethodSplitSummary], method: str) -> dict[str, object]:
    rows = list(rows)
    numeric_fields = [
        "ap", "ap50", "ap75", "nap", "nap50", "nap75",
        "ece", "brier", "latency_ms_per_img",
        "compute_latency_ms_per_img", "throughput_img_per_s",
    ]
    out: dict[str, object] = {
        "method": method,
        "split": "mean",
        "shot": rows[0].shot,
        "seed": rows[0].seed,
    }
    for field in numeric_fields:
        out[field] = statistics.mean(getattr(row, field) for row in rows)
    out["calibration_json"] = ""
    out["eval_log"] = ""
    return out


def write_outputs(rows: list[MethodSplitSummary], output_root: Path, methods: list[str]) -> None:
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    payload = [asdict(row) for row in rows]
    mean_rows = []
    for method in methods:
        method_rows = [row for row in rows if row.method == method]
        mean_rows.append(method_mean_row(method_rows, method))
    full_payload = payload + mean_rows

    json_path = summary_dir / "pcb_fma_family_voc5_summary.json"
    json_path.write_text(json.dumps(full_payload, indent=2))

    csv_path = summary_dir / "pcb_fma_family_voc5_summary.csv"
    fieldnames = list(full_payload[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(full_payload)

    comparison_path = summary_dir / "pcb_fma_family_voc5_method_means.csv"
    comparison_fields = [
        "method", "ap", "ap50", "ece", "brier",
        "latency_ms_per_img", "compute_latency_ms_per_img", "throughput_img_per_s",
    ]
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        for row in mean_rows:
            writer.writerow({field: row[field] for field in comparison_fields})

    md_path = summary_dir / "pcb_fma_family_voc5_method_means.md"
    headers = [
        "Method", "AP", "AP50", "ECE", "Brier",
        "Latency ms/img", "Compute ms/img", "Img/s",
    ]
    lines = [
        "# PCB-FMA Family VOC 5-shot Means",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in mean_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    f'{row["ap"]:.4f}',
                    f'{row["ap50"]:.4f}',
                    f'{row["ece"]:.4f}',
                    f'{row["brier"]:.4f}',
                    f'{row["latency_ms_per_img"]:.2f}',
                    f'{row["compute_latency_ms_per_img"]:.2f}',
                    f'{row["throughput_img_per_s"]:.2f}',
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {comparison_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    rows: list[MethodSplitSummary] = []

    for method in args.methods:
        for split in args.splits:
            log_path, cal_path = run_eval(method, split, args.shot, args.seed, args, output_root)
            rows.append(build_summary(method, split, args.shot, args.seed, log_path, cal_path))

    write_outputs(rows, output_root, args.methods)


if __name__ == "__main__":
    main()
