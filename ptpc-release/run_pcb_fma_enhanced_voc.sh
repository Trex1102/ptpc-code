#!/usr/bin/env bash
set -euo pipefail

EXP_NAME=${1:-vanilla_defrcn}
SPLIT_ID=${2:-}

if [[ -z "${SPLIT_ID}" ]]; then
    echo "Usage: SHOTS=\"1\" bash run_pcb_fma_enhanced_voc.sh <exp_name> <split_id>" >&2
    exit 1
fi

IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
SHOTS=${SHOTS:-"1"}
SEEDS=${SEEDS:-"0"}
NUM_GPUS=${NUM_GPUS:-1}
FORCE=${FORCE:-0}

TRAINED_ROOT=${TRAINED_ROOT:-checkpoints/voc/${EXP_NAME}/split${SPLIT_ID}}
EVAL_ROOT=${EVAL_ROOT:-checkpoints/voc/${EXP_NAME}/pcb_fma_enhanced_eval/split${SPLIT_ID}/pcb-fma-enhanced-noaug}
QUEUE_LOG="${EVAL_ROOT}/queue.log"
mkdir -p "${EVAL_ROOT}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

PCB_FMA_ENHANCED_OPTS=(
    TEST.PCB_ENABLE True
    NOVEL_METHODS.ENABLE True
    NOVEL_METHODS.METHOD pcb-fma-enhanced
    NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE True
    NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME dinov2_vitb14
    NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM 768
    NOVEL_METHODS.PCB_FMA_ENHANCED.ROI_SIZE 224
    NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT 0.4
    NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT 0.6
    NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB False
    NOVEL_METHODS.PCB_FMA_ENHANCED.BATCH_SIZE 32
    NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False
    NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False
    NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
    NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE softmax
    TEST.PCB_PROTO_REASSIGN False
)

echo "============================================================"
echo "PCB-FMA-Enhanced VOC evaluation"
echo "Started: $(date '+%F %T %z')"
echo "Experiment: ${EXP_NAME}"
echo "Split: ${SPLIT_ID}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Checkpoint root: ${TRAINED_ROOT}"
echo "Output: ${EVAL_ROOT}"
echo "Method: pcb-fma-enhanced BMVC-v6 competitive scorer without support aug"
echo "============================================================"

for SHOT in ${SHOTS}; do
    for SEED in ${SEEDS}; do
        CKPT="${TRAINED_ROOT}/${SHOT}shot_seed${SEED}/model_final.pth"
        OUTPUT_DIR="${EVAL_ROOT}/${SHOT}shot_seed${SEED}"
        LOG_FILE="${OUTPUT_DIR}/log.txt"

        if [[ "${FORCE}" != "1" && -f "${LOG_FILE}" ]] && grep -q "copypaste: AP,AP50" "${LOG_FILE}"; then
            echo "[$(date '+%F %T %z')] SKIP completed ${SHOT}shot seed${SEED}"
            continue
        fi
        if [[ ! -f "${CKPT}" ]]; then
            echo "Missing checkpoint: ${CKPT}" >&2
            exit 1
        fi

        echo "[$(date '+%F %T %z')] START ${SHOT}shot seed${SEED}"
        python3 tools/create_config.py \
            --dataset voc \
            --config_root configs/voc \
            --shot "${SHOT}" \
            --seed "${SEED}" \
            --setting fsod \
            --split "${SPLIT_ID}"

        CONFIG_PATH="configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml"
        python3 main.py --eval-only --num-gpus "${NUM_GPUS}" --config-file "${CONFIG_PATH}" \
            --opts MODEL.WEIGHTS "${CKPT}" OUTPUT_DIR "${OUTPUT_DIR}" \
                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}" "${PCB_FMA_ENHANCED_OPTS[@]}"

        rm -f "${CONFIG_PATH}"
        echo "[$(date '+%F %T %z')] DONE ${SHOT}shot seed${SEED}"
    done
done

python3 tools/extract_results.py --res-dir "${EVAL_ROOT}" --shot-list ${SHOTS} || true

echo "============================================================"
echo "Finished: $(date '+%F %T %z')"
echo "Results saved in ${EVAL_ROOT}"
echo "============================================================"
