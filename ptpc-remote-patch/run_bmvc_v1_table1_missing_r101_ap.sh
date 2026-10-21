#!/usr/bin/env bash
set -euo pipefail

cd /home/cse/Documents/fsod-test
source ./bmvc_v1_common.sh
bmvc_activate_env

SHOT=5
SEED=0
OUT_ROOT=checkpoints/voc/bmvc_v1/table1_missing_r101_cropcls_eval

run_case() {
  local variant="$1"
  local split="$2"
  local method="$3"
  shift 3

  local base_cfg
  local method_cfg
  local out_dir="${OUT_ROOT}/${variant}/split${split}/${SHOT}shot_seed${SEED}"

  base_cfg="$(bmvc_create_base_config "${split}" "${SHOT}" "${SEED}")"
  method_cfg="$(bmvc_create_method_config pcb_resnet101 pcb_resnet101 "${split}" "${SHOT}" "${SEED}")"

  echo "[START] $(date '+%F %T %z') ${variant} split${split} method=${method}"
  bmvc_run_eval_only "${method_cfg}" "${split}" "${SHOT}" "${SEED}" "${out_dir}" \
    NOVEL_METHODS.ENABLE True \
    NOVEL_METHODS.METHOD "${method}" \
    TEST.PCB_PROTO_REASSIGN False \
    "$@"
  echo "[DONE]  $(date '+%F %T %z') ${variant} split${split}"

  bmvc_cleanup_configs "${split}" "${SHOT}" "${SEED}" "${method_cfg}"
}

run_case comp_only 1 pcb_resnet101_competitive \
  NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
run_case comp_only 2 pcb_resnet101_competitive \
  NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
run_case comp_only 3 pcb_resnet101_competitive \
  NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1

run_case guard_only 1 pcb_resnet101_guarded \
  NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05 \
  NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
run_case guard_only 2 pcb_resnet101_guarded \
  NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05 \
  NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
run_case guard_only 3 pcb_resnet101_guarded \
  NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05 \
  NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20

run_case comp_guard 1 pcb_resnet101_competitive_guarded \
  NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05 \
  NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
run_case comp_guard 2 pcb_resnet101_competitive_guarded \
  NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05 \
  NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
run_case comp_guard 3 pcb_resnet101_competitive_guarded \
  NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05 \
  NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3 \
  NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
