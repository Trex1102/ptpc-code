#!/usr/bin/env bash
set -euo pipefail

cd /home/cse/Documents/fsod-test
source ./bmvc_v1_common.sh
bmvc_activate_env

IMAGENET=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
SHOT=5
SEED=0
OUT_ROOT=checkpoints/voc/bmvc_v1/table1_missing_r101_cropcls

run_case() {
  local variant="$1"
  local split="$2"
  local method="$3"
  shift 3

  local cfg="configs/voc/defrcn_fsod_r101_novel${split}_${SHOT}shot_seed${SEED}.yaml"
  local out_dir="${OUT_ROOT}/${variant}/split${split}/${SHOT}shot_seed${SEED}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/calibration_metrics.json" ]]; then
    echo "[SKIP] ${variant} split${split} already has calibration_metrics.json"
    return 0
  fi

  echo "[START] $(date '+%F %T %z') ${variant} split${split} method=${method}"
  python3 tools/create_config.py \
    --dataset voc \
    --config_root configs/voc \
    --shot "$SHOT" \
    --seed "$SEED" \
    --setting fsod \
    --split "$split" >/dev/null

  /usr/bin/time -p python3 tools/compute_calibration_metrics.py \
    --config-file "$cfg" \
    --output "$out_dir/calibration_metrics.json" \
    --opts \
    MODEL.WEIGHTS "checkpoints/voc/vanilla_defrcn/split${split}/${SHOT}shot_seed${SEED}/model_final.pth" \
    OUTPUT_DIR "$out_dir" \
    TEST.PCB_MODELPATH "$IMAGENET" \
    TEST.PCB_ENABLE True \
    TEST.PCB_PROTO_REASSIGN False \
    NOVEL_METHODS.ENABLE True \
    NOVEL_METHODS.METHOD "$method" \
    "$@"

  rm -f "$cfg"
  echo "[DONE]  $(date '+%F %T %z') ${variant} split${split}"
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
