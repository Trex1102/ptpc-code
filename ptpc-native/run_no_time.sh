#!/usr/bin/env bash
# Paired native capture followed by the frozen canonical continuous Guard.
# Usage: bash run_no_time.sh <repo> <voc|coco> <output-cell> <shot> [voc-split]
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: $0 <repo> <voc|coco> <output-cell> <shot> [voc-split]" >&2
    exit 1
fi
REPO=$(realpath -- "$1")
DATASET=$2
OUTPUT=$(realpath -m -- "$3")
SHOT=$4
SPLIT=${5:-1}
case "${DATASET}" in
    voc)
        [[ "${SPLIT}" =~ ^[123]$ && " ${SHOT} " =~ ^\ (1|2|3|5|10)\ $ ]] || exit 1
        SEED=33
        ANNOTATION=${REPO}/data/pascal_voc/annotations/voc0712_trainval_with_segm.json
        TEMPLATE=${REPO}/no_time_to_train/pl_configs/matching_pascal_voc_few_shot_template.yaml
        case "${SPLIT}" in
            1) NOVEL=bird,bus,cow,motorbike,sofa ;;
            2) NOVEL=aeroplane,bottle,cow,horse,sofa ;;
            3) NOVEL=boat,cat,motorbike,sheep,sofa ;;
        esac
        ;;
    coco)
        [[ "${SHOT}" == 10 || "${SHOT}" == 30 ]] || exit 1
        SEED=42
        ANNOTATION=${REPO}/data/coco/annotations/instances_train2017.json
        TEMPLATE=${REPO}/no_time_to_train/new_exps/coco_fewshot_10shot_Sam2L.yaml
        NOVEL='person,bicycle,car,motorcycle,airplane,bus,train,boat,bird,cat,dog,horse,sheep,cow,bottle,chair,couch,potted plant,dining table,tv'
        ;;
    *) echo "Unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac
[[ -f "${ANNOTATION}" && -f "${TEMPLATE}" ]] || {
    echo "Prepare the host's datasets and checkpoints as described in README.md." >&2
    exit 1
}
if [[ -e "${OUTPUT}" ]]; then
    echo "Use a new output-cell directory: ${OUTPUT}" >&2
    exit 1
fi
mkdir -p -- "${OUTPUT}"
cd "${REPO}"
export PYTHONHASHSEED=${SEED}
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE_PROTO=${OUTPUT}/base_prototypes.pt
SUPPORT_NOVEL=${OUTPUT}/support_novel.pkl
SUPPORT_BASE=${OUTPUT}/support_base.pkl
if [[ "${DATASET}" == voc ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_no_time_support.py" \
        --annotation "${ANNOTATION}" --novel-output "${SUPPORT_NOVEL}" \
        --base-output "${SUPPORT_BASE}" --manifest "${OUTPUT}/support_manifest.json" \
        --split "${SPLIT}" --novel-shot "${SHOT}" --base-shot 20 --seed "${SEED}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/make_no_time_configs.py" \
        --template "${TEMPLATE}" --output-dir "${OUTPUT}" \
        --novel-support "${SUPPORT_NOVEL}" --base-support "${SUPPORT_BASE}" \
        --base-prototypes "${BASE_PROTO}" --split "${SPLIT}" --shot "${SHOT}"
else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_no_time_coco_support.py" \
        --annotation "${ANNOTATION}" --novel-output "${SUPPORT_NOVEL}" \
        --base-output "${SUPPORT_BASE}" --manifest "${OUTPUT}/support_manifest.json" \
        --novel-shot "${SHOT}" --base-shot 20 --seed "${SEED}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/make_no_time_coco_configs.py" \
        --template "${TEMPLATE}" --output-dir "${OUTPUT}" \
        --novel-support "${SUPPORT_NOVEL}" --base-support "${SUPPORT_BASE}" \
        --base-prototypes "${BASE_PROTO}" --annotation "${ANNOTATION}" --shot "${SHOT}"
fi
# The template's runtime seed stays 42; the support draw seed above is separate.
"${PYTHON_BIN}" run_lightening.py test --config "${OUTPUT}/novel_fill.yaml" \
    --out_path "${OUTPUT}/novel_memory.ckpt"
"${PYTHON_BIN}" run_lightening.py test --config "${OUTPUT}/novel_postprocess.yaml" \
    --ckpt_path "${OUTPUT}/novel_memory.ckpt" --out_path "${OUTPUT}/novel_postprocessed.ckpt"
"${PYTHON_BIN}" run_lightening.py test --config "${OUTPUT}/base_fill.yaml" \
    --out_path "${OUTPUT}/base_memory.ckpt"
"${PYTHON_BIN}" "${SCRIPT_DIR}/extract_no_time_base_prototypes.py" \
    --checkpoint "${OUTPUT}/base_memory.ckpt" --annotation "${ANNOTATION}" \
    --output "${BASE_PROTO}" --novel-classes "${NOVEL}"
"${PYTHON_BIN}" run_lightening.py test --config "${OUTPUT}/guard_inference.yaml" \
    --ckpt_path "${OUTPUT}/novel_postprocessed.ckpt" \
    --export_result "${OUTPUT}/predictions_binary_limit.json" --seed "${SEED}" --n_shot "${SHOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/rescore_guard.py" --host no_time --dataset "${DATASET}" \
    --cell "${OUTPUT}" --output "${OUTPUT}/predictions_guard.json"
