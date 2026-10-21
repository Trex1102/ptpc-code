#!/usr/bin/env bash
set -euo pipefail

EXP_NAME=${1:-vanilla_defrcn}

SAVE_DIR=checkpoints/coco/${EXP_NAME}
IMAGENET_PRETRAIN=${IMAGENET_PRETRAIN:-.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}

SHOTS=${SHOTS:-"1 2 3 5 10 30"}
SEEDS=${SEEDS:-"0"}
SKIP_BASE_TRAIN=${SKIP_BASE_TRAIN:-1}
BASE_NUM_GPUS=${BASE_NUM_GPUS:-1}
NOVEL_NUM_GPUS=${NOVEL_NUM_GPUS:-1}

BASE_DIR=${SAVE_DIR}/defrcn_det_r101_base
BASE_MODEL=${BASE_DIR}/model_final.pth
BASE_WEIGHT=${BASE_DIR}/model_reset_remove.pth
NOVEL_ROOT=${SAVE_DIR}/defrcn_fsod_r101_novel

if [[ "${SKIP_BASE_TRAIN}" != "1" ]]; then
    python3 main.py --num-gpus "${BASE_NUM_GPUS}" --config-file configs/coco/defrcn_det_r101_base.yaml \
        --opts MODEL.WEIGHTS "${IMAGENET_PRETRAIN}" OUTPUT_DIR "${BASE_DIR}"
elif [[ ! -f "${BASE_MODEL}" ]]; then
    echo "Missing base checkpoint: ${BASE_MODEL}" >&2
    echo "Run once with SKIP_BASE_TRAIN=0 or place the base checkpoint there." >&2
    exit 1
fi

python3 tools/model_surgery.py --dataset coco --method remove \
    --src-path "${BASE_MODEL}" \
    --save-dir "${BASE_DIR}"

for SHOT in ${SHOTS}; do
    for SEED in ${SEEDS}; do
        python3 tools/create_config.py \
            --dataset coco14 \
            --config_root configs/coco \
            --shot "${SHOT}" \
            --seed "${SEED}" \
            --setting fsod

        CONFIG_PATH="configs/coco/defrcn_fsod_r101_novel_${SHOT}shot_seed${SEED}.yaml"
        OUTPUT_DIR="${NOVEL_ROOT}/${SHOT}shot_seed${SEED}"

        python3 main.py --num-gpus "${NOVEL_NUM_GPUS}" --config-file "${CONFIG_PATH}" \
            --opts MODEL.WEIGHTS "${BASE_WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}" \
                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"

        rm -f "${CONFIG_PATH}"
    done
done

python3 tools/extract_results.py --res-dir "${NOVEL_ROOT}" --shot-list ${SHOTS}
