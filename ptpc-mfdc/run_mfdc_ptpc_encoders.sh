#!/usr/bin/env bash
# Evaluate PTPC without crop-CLS fusion across frozen FM variants on VOC MFDC.

set -euo pipefail

SPLITS=${1:-"3"}
SHOTS=${2:-"3 5 10"}
SEEDS=${3:-"0"}
VARIANTS=${4:-"l14"}

show_usage() {
    echo "Usage: bash run_mfdc_ptpc_encoders.sh [splits] [shots] [seeds] [variants]"
    echo ""
    echo "Arguments:"
    echo "  splits    : VOC split ids (default: \"1 2 3\")"
    echo "  shots     : shot settings (default: \"1 2 3 5 10\")"
    echo "  seeds     : random seeds (default: \"0\")"
    echo "  variants  : FM aliases (default: \"${VARIANTS}\")"
    echo "              DINOv2 : s14 b14 l14 g14"
    echo "              DINOv1 : d1_s16 d1_b16"
    echo "              CLIP   : clip_b16 clip_l14"
    echo "              SigLIP : siglip_b siglip_l"
    echo "              MAE    : mae_b mae_l"
    echo "              Note   : CLIP ViT-B/14 is not available here; use clip_b16 or clip_l14."
    echo ""
    echo "Environment overrides:"
    echo "  PYTHON_BIN            Python executable (default: detectron2_03 env if present)"
    echo "  NUM_GPUS              detectron2 --num-gpus value (default: 1)"
    echo "  EXP_NAME              output experiment name"
    echo "  SAVE_DIR              root output directory"
    echo "  PRETRAINED_NOVEL_ROOT MFDC novel checkpoint root"
    echo "  IMAGENET_PRETRAIN_TORCH ResNet-101 checkpoint path passed through for compatibility"
    echo "  PTPC_BATCH_SIZE    PTPC feature batch size (default: 32)"
    echo "  COMPUTE_CALIBRATION_METRICS  if 1, compute ECE/Brier after each eval (default: 1)"
    echo "  CALIBRATION_BINS      number of ECE bins (default: 15)"
    echo "  CALIBRATION_IOU_THRESH IoU threshold for TP matching (default: 0.5)"
    echo "  DRY_RUN               if 1, write configs and print commands without running eval"
    echo ""
    echo "Examples:"
    echo "  bash run_mfdc_ptpc_encoders.sh"
    echo "  DRY_RUN=1 bash run_mfdc_ptpc_encoders.sh \"1 3\" \"1 5\" \"0\" \"g14 clip_l14 mae_b\""
}

if [[ "${SPLITS}" == "-h" || "${SPLITS}" == "--help" ]]; then
    show_usage
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

DEFAULT_PYTHON=/home/bio/anaconda3/envs/detectron2_03/bin/python3
PYTHON_BIN=${PYTHON_BIN:-$([ -x "${DEFAULT_PYTHON}" ] && printf "%s" "${DEFAULT_PYTHON}" || printf "%s" "python3")}
DETECTRON2_ROOT=${DETECTRON2_ROOT:-/home/bio/Tanvir_Saikat/fsod-test/detectron2}
NUM_GPUS=${NUM_GPUS:-1}
DRY_RUN=${DRY_RUN:-0}
EXP_NAME=${EXP_NAME:-mfdc_ptpc_no_cls_fm_variants}
SAVE_DIR=${SAVE_DIR:-checkpoints/voc/${EXP_NAME}}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_mfdc}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-weight/resnet101-5d3b4d8f.pth}
PTPC_BATCH_SIZE=${PTPC_BATCH_SIZE:-32}
COMPUTE_CALIBRATION_METRICS=${COMPUTE_CALIBRATION_METRICS:-0}
CALIBRATION_BINS=${CALIBRATION_BINS:-15}
CALIBRATION_IOU_THRESH=${CALIBRATION_IOU_THRESH:-0.5}
CONFIG_ROOT=configs/voc/novelMethods/generated/ptpc_no_cls
SUMMARY_DIR=${SUMMARY_DIR:-${SAVE_DIR}/summary}
SUMMARY_CSV=${SUMMARY_CSV:-${SUMMARY_DIR}/ptpc_no_cls_calibration_summary.csv}

export PYTHONPATH="${SCRIPT_DIR}:${DETECTRON2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

init_summary_csv() {
    if [ "${COMPUTE_CALIBRATION_METRICS}" != "1" ]; then
        return
    fi
    mkdir -p "${SUMMARY_DIR}"
    if [ ! -f "${SUMMARY_CSV}" ]; then
        printf 'variant,split,shot,seed,fm_model_name,fm_feat_dim,ece,brier,base_novel_fp_rate,n_total_detections,overall_accuracy,metrics_json\n' > "${SUMMARY_CSV}"
    fi
}

append_calibration_summary() {
    local variant=$1
    local split=$2
    local shot=$3
    local seed=$4
    local fm_model_name=$5
    local fm_feat_dim=$6
    local metrics_json=$7

    "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${metrics_json}" "${variant}" "${split}" "${shot}" "${seed}" "${fm_model_name}" "${fm_feat_dim}" <<'PY'
import csv
import json
import sys

summary_csv, metrics_json, variant, split_id, shot, seed, fm_model_name, fm_feat_dim = sys.argv[1:9]
with open(metrics_json, "r") as handle:
    payload = json.load(handle)

row = {
    "variant": variant,
    "split": split_id,
    "shot": int(shot),
    "seed": int(seed),
    "fm_model_name": fm_model_name,
    "fm_feat_dim": int(fm_feat_dim),
    "ece": float(payload.get("ece", 0.0)),
    "brier": float(payload.get("brier", 0.0)),
    "base_novel_fp_rate": float(payload.get("base_novel_fp_rate", 0.0)),
    "n_total_detections": int(payload.get("n_total_detections", 0)),
    "overall_accuracy": float(payload.get("overall_accuracy", 0.0)),
    "metrics_json": metrics_json,
}

fieldnames = [
    "variant",
    "split",
    "shot",
    "seed",
    "fm_model_name",
    "fm_feat_dim",
    "ece",
    "brier",
    "base_novel_fp_rate",
    "n_total_detections",
    "overall_accuracy",
    "metrics_json",
]

with open(summary_csv, "a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writerow(row)

print(
    "    calibration: "
    f"ECE={row['ece']:.4f}, "
    f"Brier={row['brier']:.4f}, "
    f"Base->Novel FP={row['base_novel_fp_rate']:.4f}, "
    f"Acc={row['overall_accuracy']:.4f}"
)
PY
}

resolve_variant() {
    local variant=$1
    case "${variant}" in
        s14|d2_s14)
            FM_MODEL_NAME="dinov2_vits14"
            FM_FEAT_DIM="384"
            FM_IMAGE_SIZE="518"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="14"
            ;;
        b14|d2_b14)
            FM_MODEL_NAME="dinov2_vitb14"
            FM_FEAT_DIM="768"
            FM_IMAGE_SIZE="518"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="14"
            ;;
        l14|d2_l14)
            FM_MODEL_NAME="dinov2_vitl14"
            FM_FEAT_DIM="1024"
            FM_IMAGE_SIZE="518"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="14"
            ;;
        g14|d2_g14)
            FM_MODEL_NAME="dinov2_vitg14"
            FM_FEAT_DIM="1536"
            FM_IMAGE_SIZE="518"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="14"
            ;;
        d1_s16|dino_s16)
            FM_MODEL_NAME="dino_vits16"
            FM_FEAT_DIM="384"
            FM_IMAGE_SIZE="518"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="16"
            ;;
        d1_b16|dino_b16)
            FM_MODEL_NAME="dino_vitb16"
            FM_FEAT_DIM="768"
            FM_IMAGE_SIZE="518"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="16"
            ;;
        clip_b14)
            echo "Unsupported alias: ${variant}" >&2
            echo "CLIP ViT-B/14 is not available in this environment. Use clip_b16 or clip_l14." >&2
            exit 1
            ;;
        clip_b16)
            FM_MODEL_NAME="timm:vit_base_patch16_clip_224"
            FM_FEAT_DIM="768"
            FM_IMAGE_SIZE="224"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="16"
            ;;
        clip_l14)
            FM_MODEL_NAME="timm:vit_large_patch14_clip_224"
            FM_FEAT_DIM="1024"
            FM_IMAGE_SIZE="224"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="14"
            ;;
        siglip_b)
            FM_MODEL_NAME="timm:vit_base_patch16_siglip_224"
            FM_FEAT_DIM="768"
            FM_IMAGE_SIZE="224"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="16"
            ;;
        siglip_l)
            FM_MODEL_NAME="timm:vit_large_patch16_siglip_256"
            FM_FEAT_DIM="1024"
            FM_IMAGE_SIZE="256"
            FM_CROP_SIZE="256"
            FM_PATCH_SIZE="16"
            ;;
        mae_b)
            FM_MODEL_NAME="timm:vit_base_patch16_224.mae"
            FM_FEAT_DIM="768"
            FM_IMAGE_SIZE="224"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="16"
            ;;
        mae_l)
            FM_MODEL_NAME="timm:vit_large_patch16_224.mae"
            FM_FEAT_DIM="1024"
            FM_IMAGE_SIZE="224"
            FM_CROP_SIZE="224"
            FM_PATCH_SIZE="16"
            ;;
        *)
            echo "Unsupported FM variant: ${variant}" >&2
            echo "Available variants: s14 b14 l14 g14 d1_s16 d1_b16 clip_b16 clip_l14 siglip_b siglip_l mae_b mae_l" >&2
            exit 1
            ;;
    esac
}

validate_split() {
    local split=$1
    case "${split}" in
        1|2|3)
            return 0
            ;;
        *)
            echo "Unsupported VOC split: ${split}" >&2
            exit 1
            ;;
    esac
}

write_ptpc_no_cls_config() {
    local config_path=$1
    local base_rel=$2
    local variant=$3
    local fm_model_name=$4
    local fm_feat_dim=$5
    local fm_image_size=$6
    local fm_crop_size=$7
    local fm_patch_size=$8

    mkdir -p "$(dirname "${config_path}")"
    cat > "${config_path}" <<EOF
_BASE_: "${base_rel}"
TEST:
  PCB_ENABLE: True
  PCB_MODELPATH: "/Path/to/ImageNet/Pre-Train/Weight"
  PCB_SKIP_PRETRAINED: True
NOVEL_METHODS:
  ENABLE: True
  METHOD: "ptpc"
  PTPC:
    ENABLE: True
    FM_MODEL_NAME: "${fm_model_name}"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: ${fm_feat_dim}
    IMAGE_SIZE: ${fm_image_size}
    CROP_SIZE: ${fm_crop_size}
    ROI_SIZE: ${fm_crop_size}
    PATCH_SIZE: ${fm_patch_size}
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    CROP_CLS_WEIGHT: 0.0
    POOLING_MODE: "center"
    BATCH_SIZE: ${PTPC_BATCH_SIZE}
    FM_ONLY: True
    TEMPERATURE: 0.1
    COMPETITIVE_MODE: "softmax"
  PTPC_GUARD:
    ENABLE: True
    FEATURE_MODE: "crop_cls"
    SMOOTH_ENABLE: False  # Published encoder ablation uses the binary limit.
    MARGIN: 0.05
    SUPPRESSION_FACTOR: 0.3
    MAX_BASE_SAMPLES_PER_CLASS: 20
# Output tag: ptpc_no_cls_${variant}
EOF
}

echo "=============================================="
echo "MFDC PTPC No-CLS FM Variant Runner"
echo "=============================================="
echo "Splits: ${SPLITS}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Variants: ${VARIANTS}"
echo "Dry run: ${DRY_RUN}"
echo "Python: ${PYTHON_BIN}"
echo "Detectron2 root: ${DETECTRON2_ROOT}"
echo "Save dir: ${SAVE_DIR}"
echo "Checkpoint root: ${PRETRAINED_NOVEL_ROOT}"
echo "Calibration metrics: ${COMPUTE_CALIBRATION_METRICS}"
echo "=============================================="

init_summary_csv

for split in ${SPLITS}; do
    validate_split "${split}"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo ""
            echo "Preparing split${split} ${shot}-shot seed${seed}"

            "${PYTHON_BIN}" tools/create_config.py \
                --dataset voc \
                --config_root configs/voc \
                --shot "${shot}" \
                --seed "${seed}" \
                --setting gfsod \
                --split "${split}"

            BASE_CONFIG_PATH="configs/voc/mfdc_gfsod_novel${split}_${shot}shot_seed${seed}.yaml"
            if [ ! -f "${BASE_CONFIG_PATH}" ]; then
                echo "Missing base config: ${BASE_CONFIG_PATH}" >&2
                exit 1
            fi

            MODEL_WEIGHT="${PRETRAINED_NOVEL_ROOT}/mfdc_gfsod_novel${split}/tfa-like/${shot}shot_seed${seed}/model_final.pth"
            if [ ! -f "${MODEL_WEIGHT}" ]; then
                echo "Missing pretrained novel checkpoint: ${MODEL_WEIGHT}" >&2
                exit 1
            fi

            for variant in ${VARIANTS}; do
                resolve_variant "${variant}"

                CONFIG_PATH="${CONFIG_ROOT}/mfdc_gfsod_novel${split}_${shot}shot_seed${seed}_ptpc_no_cls_${variant}.yaml"
                OUTPUT_DIR="${SAVE_DIR}/pretrained_eval/ptpc_no_cls_${variant}/split${split}/${shot}shot_seed${seed}"
                write_ptpc_no_cls_config \
                    "${CONFIG_PATH}" \
                    "../../../mfdc_gfsod_novel${split}_${shot}shot_seed${seed}.yaml" \
                    "${variant}" \
                    "${FM_MODEL_NAME}" \
                    "${FM_FEAT_DIM}" \
                    "${FM_IMAGE_SIZE}" \
                    "${FM_CROP_SIZE}" \
                    "${FM_PATCH_SIZE}"
                mkdir -p "${OUTPUT_DIR}"

                echo "  Variant ${variant}: ${FM_MODEL_NAME} (dim=${FM_FEAT_DIM}, image=${FM_IMAGE_SIZE}, crop=${FM_CROP_SIZE}, patch=${FM_PATCH_SIZE})"
                echo "    config: ${CONFIG_PATH}"
                echo "    output: ${OUTPUT_DIR}"

                if [ "${DRY_RUN}" = "1" ]; then
                    echo "    dry-run command:"
                    echo "      ${PYTHON_BIN} train_net.py --num-gpus ${NUM_GPUS} --eval-only --config-file ${CONFIG_PATH} --opts MODEL.WEIGHTS ${MODEL_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}"
                    if [ "${COMPUTE_CALIBRATION_METRICS}" = "1" ]; then
                        echo "      ${PYTHON_BIN} tools/compute_calibration_metrics.py --config-file ${CONFIG_PATH} --output ${OUTPUT_DIR}/calibration_metrics.json --n-bins ${CALIBRATION_BINS} --iou-thresh ${CALIBRATION_IOU_THRESH} --opts MODEL.WEIGHTS ${MODEL_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}"
                    fi
                    continue
                fi

                if [ -f "${OUTPUT_DIR}/RUN_COMPLETE" ] && \
                   [ -s "${OUTPUT_DIR}/inference/res_final.json" ]; then
                    echo "    skip: completed marker and result exist"
                else
                    "${PYTHON_BIN}" train_net.py \
                        --num-gpus "${NUM_GPUS}" \
                        --eval-only \
                        --config-file "${CONFIG_PATH}" \
                        --opts \
                            MODEL.WEIGHTS "${MODEL_WEIGHT}" \
                            OUTPUT_DIR "${OUTPUT_DIR}" \
                            TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"

                    touch "${OUTPUT_DIR}/RUN_COMPLETE"
                fi

                if [ "${COMPUTE_CALIBRATION_METRICS}" = "1" ]; then
                    METRICS_JSON="${OUTPUT_DIR}/calibration_metrics.json"
                    echo "    computing calibration metrics -> ${METRICS_JSON}"
                    "${PYTHON_BIN}" tools/compute_calibration_metrics.py \
                        --config-file "${CONFIG_PATH}" \
                        --output "${METRICS_JSON}" \
                        --n-bins "${CALIBRATION_BINS}" \
                        --iou-thresh "${CALIBRATION_IOU_THRESH}" \
                        --opts \
                            MODEL.WEIGHTS "${MODEL_WEIGHT}" \
                            OUTPUT_DIR "${OUTPUT_DIR}" \
                            TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
                    append_calibration_summary \
                        "${variant}" \
                        "${split}" \
                        "${shot}" \
                        "${seed}" \
                        "${FM_MODEL_NAME}" \
                        "${FM_FEAT_DIM}" \
                        "${METRICS_JSON}"
                fi
            done
        done
    done
done

echo ""
if [ "${COMPUTE_CALIBRATION_METRICS}" = "1" ]; then
    echo "Calibration summary CSV: ${SUMMARY_CSV}"
fi
echo "=== Requested PTPC no-CLS runs completed. ==="
