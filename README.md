# Official Implementation of Prototype-Based Test-Time Prediction Calibration for Few-Shot Object Detection

**Prototype-Based Test-Time Prediction Calibration for Few-Shot Object Detection**

PTPC adjusts detection scores using novel-class prototypes and a
Base-Confusability Guard. It keeps the detector weights, boxes, labels, and
prediction order fixed.

## Code

| Directory | Contents |
| --- | --- |
| `ptpc-release/` | DeFRCN training, evaluation, and PTPC |
| `ptpc-mfdc/` | MFDC training, evaluation, and encoder ablations |
| `ptpc-native/` | Guard on FSOD-VFM and No-Time-to-Train, plus paired bootstrap evaluation |
| `ptpc-ablation-code/` | Guard-descriptor and selectivity ablations |

## Setup

The CNN experiments use Python 3.8.20, PyTorch 2.4.1+cu124,
torchvision 0.19.1+cu124, and Detectron2 v0.3. Build Detectron2 against the
installed PyTorch version, then install `requirements.txt` from the detector
directory. DeFRCN and MFDC both use the package name `defrcn`; run each from
its own directory.

Place the datasets and support splits under each detector's `datasets/`:

```text
datasets/
  VOC2007/
  VOC2012/
  vocsplit/
  coco/
  cocosplit/
```

Use the original support files and class order. DINOv2 loads through
`torch.hub`; its source and pretrained weights are downloaded on first use.
The native hosts use Python >=3.10 and separate dependencies; see
[their setup instructions](ptpc-native/README.md).

## Evaluation

### DeFRCN

From `ptpc-release/`:

```bash
for split in 1 2 3; do
  SHOTS="1 2 3 5 10" SEEDS=0 bash run_ptpc_voc.sh vanilla_defrcn "$split"
done
SHOTS="1 2 3 5 10 30" SEEDS=0 bash run_ptpc_coco.sh vanilla_defrcn
```

The runners expect `model_final.pth` at:

```text
checkpoints/voc/<experiment>/split<S>/<K>shot_seed0/model_final.pth
checkpoints/coco/<experiment>/defrcn_fsod_r101_novel/<K>shot_seed0/model_final.pth
```

Set `TRAINED_ROOT` to use another checkpoint directory and `EVAL_ROOT` to
choose an output directory. `run_voc.sh` writes training checkpoints under
`defrcn_fsod_r101_novel<S>/`; point `TRAINED_ROOT` there when evaluating them.
`PYTHON_BIN` selects the Python executable, and `FORCE=1` reruns completed
settings. Evaluation uses one GPU.

### MFDC

From `ptpc-mfdc/`:

```bash
for split in 1 2 3; do
  bash run_mfdc_novel_methods.sh "$split" ptpc "1 2 3 5 10" "0" eval
done
bash run_mfdc_novel_methods_coco.sh ptpc "1 2 3 5 10 30" "0" eval
```

`PRETRAINED_NOVEL_ROOT` defaults to `checkpoints/<voc|coco>/vanilla_mfdc`.
The runner appends:

```text
mfdc_gfsod_novel<S>/tfa-like/<K>shot_seed0/model_final.pth  # VOC
mfdc_gfsod_novel/tfa-like/<K>shot_seed0/model_final.pth     # COCO
```

Set `SAVE_DIR` for outputs and `PYTHON_BIN` for the Python executable.
Evaluation configs are generated from the shot templates.

## Settings

The evaluation scripts use DINOv2-B/14, a patch-only novel descriptor, and a
crop-CLS Guard. Config keys below are under `NOVEL_METHODS`.

| Key | Value |
| --- | --- |
| `METHOD` | `ptpc` |
| `PTPC.DET_WEIGHT`, `PTPC.FM_WEIGHT` | 0.4, 0.6 |
| `PTPC.TEMPERATURE` | 0.1 |
| `PTPC.IMAGE_SIZE`, `PTPC.CROP_SIZE` | 518, 224 |
| `PTPC_GUARD.MARGIN`, `PTPC_GUARD.SUPPRESSION_FACTOR` | 0.05, 0.3 |
| `PTPC_GUARD.MAX_BASE_SAMPLES_PER_CLASS` | 20 |
| `PTPC_GUARD.SMOOTH_ENABLE` | `True` |
| `PTPC_GUARD.SMOOTH_TEMPERATURE` | 0.04 on VOC, 0.01 on COCO |

Both modules use the score band `(0.05, 1]`. The Guard checks eligibility
after novel-score fusion. Its transition temperature is separate from the
novel softmax temperature. The VOC Guard temperature was selected on COCO;
the COCO temperature was selected on VOC Split 1.

## Baselines and ablations

DeFRCN's baseline includes its original PCB scoring. Paired evaluation should
apply PCB and PTPC to copies of the same detector outputs, with the same
checkpoint and supports.

The encoder sweep uses the binary-limit Guard:

```bash
# From ptpc-mfdc/
bash run_mfdc_ptpc_encoders.sh "1 2 3" "1 2 3 5 10" "0" \
  "s14 b14 l14 g14 clip_b16 clip_l14 siglip_b siglip_l"
```

Set `PTPC_GUARD.SMOOTH_ENABLE=False` for the binary-limit factorial,
repeated-fine-tune, and support-draw controls. For descriptor/selectivity
ablations, copy the files from the chosen `ptpc-ablation-code/` subdirectory
into a separate DeFRCN checkout. Apply `ptpc-remote-patch/defrcn/` the same way
for the competitive ResNet/Guard baselines.

Base training and fine-tuning use `run_voc.sh` and `run_coco.sh` for DeFRCN,
or `voc_train.sh` and `coco_train.sh` for MFDC.

## Metrics

`tools/compute_calibration_metrics.py` computes 15-bin ECE and Brier score
using same-class, one-to-one matching at IoU >= 0.5. The base-confusion
diagnostic uses a separate matching pass.

The [native-host instructions](ptpc-native/README.md) cover Guard-only
inference and paired image-bootstrap confidence intervals.

## Acknowledgments

We thank the authors of [DeFRCN](https://github.com/er-muyue/DeFRCN), [MFDC](https://github.com/shuangw98/MFDC), [Detectron2](https://github.com/facebookresearch/detectron2), [DINO](https://github.com/facebookresearch/dino), [DINOv2](https://github.com/facebookresearch/dinov2), [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [MAE](https://github.com/facebookresearch/mae), [timm](https://github.com/huggingface/pytorch-image-models), [FSOD-VFM](https://github.com/Intellindust-AI-Lab/FSOD-VFM), and [No-Time-to-Train](https://github.com/miquel-espinosa/no-time-to-train) for making their repositories publicly available.
