#!/usr/bin/env bash
# Run MFDC novel-method evaluation / fine-tuning on VOC.
# Usage: bash run_mfdc_novel_methods.sh <split_id> [method] [shots] [seeds] [run_mode]

set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG_PATH=""
BASE_CONFIG_PATH=""
trap '[[ -z "${CONFIG_PATH}" ]] || rm -f -- "${CONFIG_PATH}"; [[ -z "${BASE_CONFIG_PATH}" ]] || rm -f -- "${BASE_CONFIG_PATH}"' EXIT

SPLIT_ID=${1:-}
METHOD=${2:-"ptpc"}
SHOTS=${3:-"1 2 3 5 10"}
SEEDS=${4:-"0"}
RUN_MODE=${5:-"infer_pretrained_novel"}
ALL_METHODS="pcb_resnet101 pcb_fma pcb_fma_noresnet pcb_fma_enhanced pcb_fma_enhanced_noaug pcb_fma_enhanced_noresnet pcb_fma_enhanced_neg pcb_fma_enhanced_neg_noaug pcb_fma_enhanced_neg_noresnet neg_proto_guard ptpc"


show_usage() {
    echo "Usage: bash run_mfdc_novel_methods.sh <split_id> [method] [shots] [seeds] [run_mode]"
    echo ""
    echo "Arguments:"
    echo "  split_id  : VOC split (1, 2, or 3)"
    echo "  method    : Method(s) to run, or 'all' (default: ptpc)"
    echo "              Available: ${ALL_METHODS}"
    echo "  shots     : Shot settings (default: \"1 2 3 5 10\")"
    echo "  seeds     : Random seeds (default: \"0\")"
    echo "  run_mode  : finetune or infer_pretrained_novel (default: infer_pretrained_novel)"
    echo ""
    echo "Examples:"
    echo "  bash run_mfdc_novel_methods.sh 1"
    echo "  bash run_mfdc_novel_methods.sh 1 pcb_fma \"1 5\" \"0\""
    echo "  bash run_mfdc_novel_methods.sh 2 all \"1 2\" \"0 1\" infer_pretrained_novel"
}

if [ -z "${SPLIT_ID}" ]; then
    show_usage
    exit 1
fi

if [[ ! " ${SPLIT_ID} " =~ ^\ [123]\ $ ]]; then
    echo "Unsupported split_id: ${SPLIT_ID}"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

EXP_NAME=${EXP_NAME:-ptpc_continuous}
SAVE_DIR=${SAVE_DIR:-checkpoints/voc/${EXP_NAME}}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-weight/resnet101-5d3b4d8f.pth}
BASE_EXP_NAME=${BASE_EXP_NAME:-vanilla_mfdc}
BASE_WEIGHT=${BASE_WEIGHT:-checkpoints/voc/${BASE_EXP_NAME}/base${SPLIT_ID}/model_reset_surgery.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/${BASE_EXP_NAME}}

case "${RUN_MODE}" in
    finetune|train)
        RUN_MODE="finetune"
        RUN_MODE_DESC="fine-tune from base weights"
        RUN_MODE_DIR="finetune"
        ;;
    infer_pretrained_novel|pretrained_novel|eval)
        RUN_MODE="infer_pretrained_novel"
        RUN_MODE_DESC="eval-only from pretrained MFDC novel checkpoints"
        RUN_MODE_DIR="pretrained_eval"
        ;;
    *)
        echo "Unknown run_mode: ${RUN_MODE}"
        echo "Available run modes: finetune, infer_pretrained_novel"
        exit 1
        ;;
esac

validate_method() {
    case "$1" in
        pcb_resnet101|pcb_fma|pcb_fma_noresnet|pcb_fma_enhanced|pcb_fma_enhanced_noaug|pcb_fma_enhanced_noresnet|pcb_fma_enhanced_neg|pcb_fma_enhanced_neg_noaug|pcb_fma_enhanced_neg_noresnet|neg_proto_guard|ptpc)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

write_method_config() {
    local config_path=$1
    local base_rel=$2
    local method=$3
    local skip_pretrained="false"

    case "${method}" in
        pcb_fma_noresnet|pcb_fma_enhanced_noresnet|pcb_fma_enhanced_neg_noresnet|ptpc)
            skip_pretrained="true"
            ;;
    esac

    mkdir -p "$(dirname "${config_path}")"
    {
        printf '_BASE_: "%s"\n' "${base_rel}"
        printf 'TEST:\n'
        printf '  PCB_ENABLE: True\n'
        printf '  PCB_MODELPATH: "/Path/to/ImageNet/Pre-Train/Weight"\n'
        if [ "${skip_pretrained}" = "true" ]; then
            printf '  PCB_SKIP_PRETRAINED: True\n'
        fi
    } > "${config_path}"

    case "${method}" in
        pcb_resnet101)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_resnet101"
EOF
            ;;
        pcb_fma)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_fma"
  PCB_FMA:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    USE_ORIGINAL_PCB: True
    ORIGINAL_PCB_WEIGHT: 0.3
    BATCH_SIZE: 32
EOF
            ;;
        pcb_fma_noresnet)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_fma_noresnet"
  PCB_FMA:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    USE_ORIGINAL_PCB: False
    ORIGINAL_PCB_WEIGHT: 0.0
    BATCH_SIZE: 32
EOF
            ;;
        pcb_fma_enhanced)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_fma_enhanced"
  PCB_FMA_ENHANCED:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    USE_ORIGINAL_PCB: True
    ORIGINAL_PCB_WEIGHT: 0.3
    BATCH_SIZE: 32
    AUG_FLIP: True
    AUG_MULTICROP: True
    AUG_MULTICROP_SCALES: [0.8, 0.9]
    AUG_MULTICROP_NUM: 2
    TEMPERATURE: 0.1
    COMPETITIVE_MODE: "softmax"
EOF
            ;;
        pcb_fma_enhanced_noaug)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_fma_enhanced_noaug"
  PCB_FMA_ENHANCED:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    USE_ORIGINAL_PCB: True
    ORIGINAL_PCB_WEIGHT: 0.3
    BATCH_SIZE: 32
    AUG_FLIP: False
    AUG_MULTICROP: False
    AUG_MULTICROP_SCALES: [0.8, 0.9]
    AUG_MULTICROP_NUM: 2
    TEMPERATURE: 0.1
    COMPETITIVE_MODE: "softmax"
EOF
            ;;
        pcb_fma_enhanced_noresnet)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_fma_enhanced_noresnet"
  PCB_FMA_ENHANCED:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    USE_ORIGINAL_PCB: False
    ORIGINAL_PCB_WEIGHT: 0.0
    BATCH_SIZE: 32
    AUG_FLIP: True
    AUG_MULTICROP: True
    AUG_MULTICROP_SCALES: [0.8, 0.9]
    AUG_MULTICROP_NUM: 2
    TEMPERATURE: 0.1
    COMPETITIVE_MODE: "softmax"
EOF
            ;;
        pcb_fma_enhanced_neg)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_fma_enhanced_neg"
  PCB_FMA_ENHANCED:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    USE_ORIGINAL_PCB: True
    ORIGINAL_PCB_WEIGHT: 0.3
    BATCH_SIZE: 32
    AUG_FLIP: True
    AUG_MULTICROP: True
    AUG_MULTICROP_SCALES: [0.8, 0.9]
    AUG_MULTICROP_NUM: 2
    TEMPERATURE: 0.1
    COMPETITIVE_MODE: "softmax"
  NEG_PROTO_GUARD:
    ENABLE: True
    MARGIN: 0.05
    SUPPRESSION_FACTOR: 0.3
    MAX_BASE_SAMPLES_PER_CLASS: 20
EOF
            ;;
        pcb_fma_enhanced_neg_noaug)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_fma_enhanced_neg_noaug"
  PCB_FMA_ENHANCED:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    USE_ORIGINAL_PCB: True
    ORIGINAL_PCB_WEIGHT: 0.3
    BATCH_SIZE: 32
    AUG_FLIP: False
    AUG_MULTICROP: False
    AUG_MULTICROP_SCALES: [0.8, 0.9]
    AUG_MULTICROP_NUM: 2
    TEMPERATURE: 0.1
    COMPETITIVE_MODE: "softmax"
  NEG_PROTO_GUARD:
    ENABLE: True
    MARGIN: 0.05
    SUPPRESSION_FACTOR: 0.3
    MAX_BASE_SAMPLES_PER_CLASS: 20
EOF
            ;;
        pcb_fma_enhanced_neg_noresnet)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "pcb_fma_enhanced_neg_noresnet"
  PCB_FMA_ENHANCED:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    USE_ORIGINAL_PCB: False
    ORIGINAL_PCB_WEIGHT: 0.0
    BATCH_SIZE: 32
    AUG_FLIP: True
    AUG_MULTICROP: True
    AUG_MULTICROP_SCALES: [0.8, 0.9]
    AUG_MULTICROP_NUM: 2
    TEMPERATURE: 0.1
    COMPETITIVE_MODE: "softmax"
  NEG_PROTO_GUARD:
    ENABLE: True
    MARGIN: 0.05
    SUPPRESSION_FACTOR: 0.3
    MAX_BASE_SAMPLES_PER_CLASS: 20
EOF
            ;;
        neg_proto_guard)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "neg_proto_guard"
  NEG_PROTO_GUARD:
    ENABLE: True
    MARGIN: 0.05
    SUPPRESSION_FACTOR: 0.3
    MAX_BASE_SAMPLES_PER_CLASS: 20
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    ROI_SIZE: 224
    BATCH_SIZE: 32
EOF
            ;;
        ptpc)
            cat >> "${config_path}" <<'EOF'
NOVEL_METHODS:
  ENABLE: True
  METHOD: "ptpc"
  PTPC:
    ENABLE: True
    FM_MODEL_NAME: "dinov2_vitb14"
    FM_MODEL_PATH: ""
    FM_FEAT_DIM: 768
    IMAGE_SIZE: 518
    CROP_SIZE: 224
    ROI_SIZE: 224
    PATCH_SIZE: 14
    DET_WEIGHT: 0.4
    FM_WEIGHT: 0.6
    CROP_CLS_WEIGHT: 0.0
    BATCH_SIZE: 32
    FM_ONLY: True
    TEMPERATURE: 0.1
    COMPETITIVE_MODE: "softmax"
    SCORE_LOWER: 0.05
    SCORE_UPPER: 1.0
  PTPC_GUARD:
    ENABLE: True
    FEATURE_MODE: "crop_cls"
    MARGIN: 0.05
    SUPPRESSION_FACTOR: 0.3
    MAX_BASE_SAMPLES_PER_CLASS: 20
    SCORE_LOWER: 0.05
    SCORE_UPPER: 1.0
    SMOOTH_ENABLE: True
    SMOOTH_TEMPERATURE: 0.04
EOF
            ;;
        *)
            echo "Unknown method: ${method}"
            exit 1
            ;;
    esac
}

if [ "${METHOD}" = "all" ]; then
    METHODS="${ALL_METHODS}"
else
    METHODS="${METHOD}"
fi

echo "=============================================="
echo "MFDC Novel Methods Runner"
echo "=============================================="
echo "Split: ${SPLIT_ID}"
echo "Methods: ${METHODS}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Run mode: ${RUN_MODE} (${RUN_MODE_DESC})"
echo "Save dir: ${SAVE_DIR}/${RUN_MODE_DIR}"
echo "Base weight: ${BASE_WEIGHT}"
echo "Pretrained novel root: ${PRETRAINED_NOVEL_ROOT}"
echo "=============================================="

for method in ${METHODS}; do
    if ! validate_method "${method}"; then
        echo "Unknown method: ${method}"
        echo "Available: ${ALL_METHODS}"
        exit 1
    fi

    METHOD_SAVE_DIR=${SAVE_DIR}/${RUN_MODE_DIR}/${method}/split${SPLIT_ID}
    mkdir -p "${METHOD_SAVE_DIR}"

    echo ""
    echo ">>> Running method: ${method}"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo ""
            echo "  Processing: ${shot}-shot, seed ${seed}"

            BASE_CONFIG_PATH=$(mktemp "configs/voc/.ptpc_base_XXXXXX.yaml")
            "${PYTHON_BIN}" tools/create_config.py --dataset voc --config_root configs/voc \
                --shot "${shot}" --seed "${seed}" --setting gfsod --split "${SPLIT_ID}" \
                --output "${BASE_CONFIG_PATH}"
            if [ ! -f "${BASE_CONFIG_PATH}" ]; then
                echo "Missing base config: ${BASE_CONFIG_PATH}"
                exit 1
            fi

            mkdir -p "configs/voc/novelMethods/generated/${method}"
            CONFIG_PATH=$(mktemp "configs/voc/novelMethods/generated/${method}/.ptpc_eval_XXXXXX.yaml")
            write_method_config "${CONFIG_PATH}" "../../../$(basename -- "${BASE_CONFIG_PATH}")" "${method}"

            OUTPUT_DIR=${METHOD_SAVE_DIR}/${shot}shot_seed${seed}

            if [ "${RUN_MODE}" = "infer_pretrained_novel" ]; then
                MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/mfdc_gfsod_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}/model_final.pth
                if [ ! -f "${MODEL_WEIGHT}" ]; then
                    echo "Missing pretrained novel checkpoint: ${MODEL_WEIGHT}"
                    exit 1
                fi
                "${PYTHON_BIN}" train_net.py --num-gpus 1 --eval-only --config-file "${CONFIG_PATH}" \
                    --opts MODEL.WEIGHTS "${MODEL_WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}" \
                           TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
            else
                if [ ! -f "${BASE_WEIGHT}" ]; then
                    echo "Missing base checkpoint: ${BASE_WEIGHT}"
                    exit 1
                fi
                "${PYTHON_BIN}" train_net.py --num-gpus 1 --config-file "${CONFIG_PATH}" \
                    --opts MODEL.WEIGHTS "${BASE_WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}" \
                           TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
            fi

            rm -f "${CONFIG_PATH}"
            rm -f "${BASE_CONFIG_PATH}"
            CONFIG_PATH=""
            BASE_CONFIG_PATH=""

            echo "  Completed: ${shot}-shot, seed ${seed}"
        done
    done
done

echo ""
echo "=== All requested runs completed. ==="
