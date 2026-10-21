#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG_PATH=""
trap '[[ -z "${CONFIG_PATH}" ]] || rm -f -- "${CONFIG_PATH}"' EXIT

EXP_NAME=${1:-vanilla_defrcn}

IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
SHOTS=${SHOTS:-"1 2 3 5 10 30"}
SEEDS=${SEEDS:-"0"}
NUM_GPUS=${NUM_GPUS:-1}
if [[ "${NUM_GPUS}" != 1 ]]; then
    echo "Paper reproduction uses one process; set NUM_GPUS=1." >&2
    exit 1
fi
FORCE=${FORCE:-0}

TRAINED_ROOT=${TRAINED_ROOT:-checkpoints/coco/${EXP_NAME}/defrcn_fsod_r101_novel}
EVAL_ROOT=${EVAL_ROOT:-checkpoints/coco/${EXP_NAME}/ptpc_continuous_eval/ptpc}
QUEUE_LOG="${EVAL_ROOT}/queue.log"
mkdir -p "${EVAL_ROOT}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

PTPC_OPTS=(
    TEST.PCB_ENABLE True
    NOVEL_METHODS.ENABLE True
    NOVEL_METHODS.METHOD ptpc
    NOVEL_METHODS.PTPC.ENABLE True
    NOVEL_METHODS.PTPC.FM_ONLY True
    NOVEL_METHODS.PTPC.FM_MODEL_NAME dinov2_vitb14
    NOVEL_METHODS.PTPC.FM_FEAT_DIM 768
    NOVEL_METHODS.PTPC.IMAGE_SIZE 518
    NOVEL_METHODS.PTPC.CROP_SIZE 224
    NOVEL_METHODS.PTPC.PATCH_SIZE 14
    NOVEL_METHODS.PTPC.DET_WEIGHT 0.4
    NOVEL_METHODS.PTPC.FM_WEIGHT 0.6
    NOVEL_METHODS.PTPC.CROP_CLS_WEIGHT 0.0
    NOVEL_METHODS.PTPC.TEMPERATURE 0.1
    NOVEL_METHODS.PTPC.COMPETITIVE_MODE softmax
    NOVEL_METHODS.PTPC.SCORE_LOWER 0.05
    NOVEL_METHODS.PTPC.SCORE_UPPER 1.0
    NOVEL_METHODS.PTPC_GUARD.ENABLE True
    NOVEL_METHODS.PTPC_GUARD.MARGIN 0.05
    NOVEL_METHODS.PTPC_GUARD.SUPPRESSION_FACTOR 0.3
    NOVEL_METHODS.PTPC_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
    NOVEL_METHODS.PTPC_GUARD.SCORE_LOWER 0.05
    NOVEL_METHODS.PTPC_GUARD.SCORE_UPPER 1.0
    NOVEL_METHODS.PTPC_GUARD.NOVEL_REFERENCE max
    NOVEL_METHODS.PTPC_GUARD.BASE_CLASS_LIMIT 0
    NOVEL_METHODS.PTPC_GUARD.SMOOTH_ENABLE True
    NOVEL_METHODS.PTPC_GUARD.SMOOTH_TEMPERATURE 0.01
    TEST.PCB_PROTO_REASSIGN False
)

echo "============================================================"
echo "PTPC COCO evaluation"
echo "Started: $(date "+%F %T %z")"
echo "Experiment: ${EXP_NAME}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Output: ${EVAL_ROOT}"
echo "Method: ptpc contextual novel-prototype module with continuous crop-CLS Guard (eta=0.01)"
echo "============================================================"

for SHOT in ${SHOTS}; do
    for SEED in ${SEEDS}; do
        CKPT="${TRAINED_ROOT}/${SHOT}shot_seed${SEED}/model_final.pth"
        OUTPUT_DIR="${EVAL_ROOT}/${SHOT}shot_seed${SEED}"
        LOG_FILE="${OUTPUT_DIR}/log.txt"

        if [[ "${FORCE}" != "1" && -f "${LOG_FILE}" ]] && grep -q "copypaste: AP,AP50" "${LOG_FILE}"; then
            echo "[$(date "+%F %T %z")] SKIP completed ${SHOT}shot seed${SEED}"
            continue
        fi
        if [[ ! -f "${CKPT}" ]]; then
            echo "Missing checkpoint: ${CKPT}" >&2
            exit 1
        fi

        echo "[$(date "+%F %T %z")] START ${SHOT}shot seed${SEED}"
        CONFIG_PATH=$(mktemp "configs/coco/.ptpc_eval_XXXXXX.yaml")
        "${PYTHON_BIN}" tools/create_config.py \
            --dataset coco14 \
            --config_root configs/coco \
            --shot "${SHOT}" \
            --seed "${SEED}" \
            --setting fsod --output "${CONFIG_PATH}"

        "${PYTHON_BIN}" main.py --eval-only --num-gpus "${NUM_GPUS}" --config-file "${CONFIG_PATH}" \
            --opts MODEL.WEIGHTS "${CKPT}" OUTPUT_DIR "${OUTPUT_DIR}" \
                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}" "${PTPC_OPTS[@]}"

        rm -f -- "${CONFIG_PATH}"
        CONFIG_PATH=""
        echo "[$(date "+%F %T %z")] DONE ${SHOT}shot seed${SEED}"
    done
done

"${PYTHON_BIN}" tools/extract_results.py --res-dir "${EVAL_ROOT}" --shot-list ${SHOTS} || true

echo "============================================================"
echo "Finished: $(date "+%F %T %z")"
echo "Results saved in ${EVAL_ROOT}"
echo "============================================================"
