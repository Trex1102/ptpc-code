# Guard on FSOD-VFM and No-Time-to-Train

The Guard reuses each host's mask-pooled DINOv2 descriptors to compare base
and novel prototypes. It adjusts scores after detector inference.

## Settings

| Host | Target | Source | m | gamma | Score band | eta | Support seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FSOD-VFM | VOC | COCO | .01 | .85 | (.05,1] | .04 | 33 |
| FSOD-VFM | COCO | VOC Split 1 | .05 | .50 | (.05,1] | .01 | 33 |
| No Time | VOC | COCO | .01 | .70 | (0,.70] | .04 | 33 |
| No Time | COCO | VOC Split 1 | .01 | .50 | (0,.70] | .005 | 42 |

The score multiplier is `1-(1-gamma)*sigmoid((max_base-max_novel-m)/eta)`
inside the band, and 1 outside. These parameters were selected on the other
benchmark; target labels are used to build the support prototypes.
`rescore_guard.py` uses float32 arithmetic and maps paired non-finite No-Time
scores to zero. Use `--binary-limit` for the binary Guard control.

## Host code and environments

Use separate environments with Python >=3.10, torch 2.6.0+cu124,
torchvision 0.21.0+cu124, NumPy 1.26.0 and pycocotools 2.0.10. Install
`requirements.txt` for rescoring and bootstrap evaluation. Inference also
requires each host's dependencies and CUDA operators.

Check out these upstream revisions:

```bash
git clone https://github.com/Intellindust-AI-Lab/FSOD-VFM.git fsod_vfm
git -C fsod_vfm checkout 6da781bf02046fe6f70530e1d5483abd9aef6c9c
git clone https://github.com/miquel-espinosa/no-time-to-train.git no_time_to_train
git -C no_time_to_train checkout b4b6b2b2f3fb6c9fd26495f5b0e0eac36433df21
```

For FSOD-VFM, its DINOv2 checkout is
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`; SAM2 is
`2b90b9f5ceec907a1c18123530e92e794ad901a4`. Install SAM2 and compile the
UPN operator in `chatrex/upn/ops` against the chosen torch/CUDA installation.
Install opencv-python-headless 4.10.0.84, mmengine 0.10.7, hydra-core 1.3.2,
iopath 0.1.10, timm 1.0.26, huggingface-hub 0.36.2 and safetensors 0.7.0.
Follow the upstream setup for the remaining dependencies and set
`XFORMERS_DISABLED=1`.

No Time additionally uses pytorch-lightning 2.1.0,
`jsonargparse[signatures]==4.33.2`, docstring-parser 0.16, POT 0.9.5, and
tidecv 1.0.1. Install its own bundled SAM2/DINOv2 packages in that environment.
Apply the patch from this directory to the pinned **clean** host checkout:

```bash
git -C /absolute/path/no_time_to_train apply --check /absolute/path/ptpc-native/no_time_instrumentation.patch
git -C /absolute/path/no_time_to_train apply /absolute/path/ptpc-native/no_time_instrumentation.patch
```

The patch exports baseline scores and base/novel similarities, fixes bundled
DINOv2 imports, and supplies COCO metadata for pycocotools. The inference pass
writes the binary control; `rescore_guard.py` then computes continuous scores
from the same detections and similarities.

## Data and checkpoints

For FSOD-VFM, retain its support JSON files and stage:

```text
dataset/PascalVOC/VOC2007/{Annotations,JPEGImages,ImageSets}/
dataset/PascalVOC/VOC2012/{Annotations,JPEGImages,ImageSets}/
data/Pascal/vocsplit/split<S>/<K>shot_seed33.json
data/Pascal/VOC2007Test/voc07test_coco_format.json
data/coco/{train2017,val2017,annotations}/
data/coco/few_shot_<K>shot_seed33.json
checkpoints/{sam2.1_hiera_large.pt,dinov2_vitl14_pretrain.pth,upn_large.pth}
```

The VOC query annotation contains all 4,952 VOC2007 test images. COCO uses
all 5,000 val2017 images. Its published FSOD-VFM supports are 10/30 shots;
the paper's 1/2/3/5-shot extension takes a nested prefix **per class** of the
released 30-shot seed-33 JSON, preserving class/instance order:

```python
import json
from pathlib import Path
root = Path('/absolute/path/fsod_vfm/data/coco')
supports = json.loads((root / 'few_shot_30shot_seed33.json').read_text())
for shot in (1, 2, 3, 5):
    path = root / f'few_shot_{shot}shot_seed33.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({name: rows[:shot] for name, rows in supports.items()}))
```

For No Time, stage `checkpoints/sam2_hiera_large.pt` (SAM2 Hiera-L, not the
FSOD host's SAM2.1 checkpoint) and
`checkpoints/dinov2/dinov2_vitl14_pretrain.pth`. Its VOC template expects
`data/pascal_voc/VOCdevkit/{VOC2007,VOC2012,allimages}` and
`data/pascal_voc/annotations/{voc0712_trainval_with_segm,voc07_test_with_segm}.json`.
`allimages/` is a flat view of both years' JPEGImages, including VOC2007 test.
Create these two annotations with the upstream
`no_time_to_train.dataset.pascal_voc_to_coco.cvt_annotations`: years
`['2007','2012']`, split `trainval`, and year `'2007'`, split `test`.
Flatten each JSON image's `file_name` with `Path(file_name).name` to match
`allimages/`. The converter's segmentation polygons are rectangles derived
from each box. Use those, without the optional offline SAM support preprocessing.
For No-Time COCO, stage the original train2017/val2017 images and annotations
under `data/coco/`, matching the upstream COCO template.

No-Time VOC uses box-derived support masks. No-Time COCO uses the segmentation
masks in the original COCO annotations. These are different support-label
settings and should be reported separately.

## Inference

Run these commands from this directory in the FSOD-VFM environment. Set
`RUNS` to an absolute output path. Keep this directory layout for the
bootstrap manifest builder.

```bash
RUNS=/absolute/path/runs
FSOD=/absolute/path/fsod_vfm
for split in 1 2 3; do
  for shot in 1 2 3 5 10; do
    XFORMERS_DISABLED=1 PYTHONHASHSEED=33 python run_fsod_vfm_guard.py \
      --repo-root "$FSOD" --cache-dir "$RUNS/cache/fsod_vfm" \
      --output-dir "$RUNS/fsod_vfm/voc/split$split/${shot}shot_seed33" \
      --split "$split" --shot "$shot" --seed 33
  done
done
for shot in 1 2 3 5 10 30; do
  XFORMERS_DISABLED=1 PYTHONHASHSEED=33 python run_fsod_vfm_guard_coco.py \
    --repo-root "$FSOD" --cache-dir "$RUNS/cache/fsod_vfm" \
    --output-dir "$RUNS/fsod_vfm/coco/${shot}shot_seed33" --shot "$shot" --seed 33
done
```

Switch to the No-Time environment, then:

```bash
NOTIME=/absolute/path/no_time_to_train
for split in 1 2 3; do
  for shot in 1 2 3 5 10; do
    PYTHON_BIN=python bash run_no_time.sh "$NOTIME" voc \
      "$RUNS/no_time/voc/split$split/${shot}shot_seed33" "$shot" "$split"
  done
done
for shot in 10 30; do
  PYTHON_BIN=python bash run_no_time.sh "$NOTIME" coco \
    "$RUNS/no_time/coco/${shot}shot_seed42" "$shot"
done
```

No Time retains the official template's **runtime seed 42**; support seeds
are 33 for VOC and 42 for COCO. Memory files can require tens of GB, especially
the 60-class base memory; select an output volume with sufficient space.
The script builds supports, fills/postprocesses novel memory, extracts base
prototypes, runs paired inference once, then writes continuous
`predictions_guard.json`. It refuses an existing output directory.
`predictions_baseline.json` and diagnostic files remain paired in stored order.
Use `--max-images 2` with a new output directory for a small FSOD smoke test.

## Paired confidence intervals

Use NumPy 1.26.0 and pycocotools 2.0.10 on CPU. Build the manifest from the
inference outputs, then run the bootstrap:

```bash
python run_bootstrap_suite.py --prepare-manifest --manifest "$RUNS/native_manifest.json" \
  --runs-root "$RUNS" --fsod-vfm-repo "$FSOD" --no-time-repo "$NOTIME"
python run_bootstrap_suite.py --manifest "$RUNS/native_manifest.json" \
  --output-dir "$RUNS/paired_ci" --workers 6 --replicates 2000
```

The manifest builder expects all 38 host/benchmark settings and hashes each
input. A subset manifest can be used to evaluate fewer settings.

The bootstrap uses seed 20261020 and 2,000 paired image resamples, including
images with no novel GT. It uses COCOeval matching and 101-point AP: AP50 for
VOC and AP for COCO. Image draws are shared between baseline and Guard and
synchronized across shots/splits within each host/benchmark. Cell gains are
averaged within each replicate before computing the 95% percentile interval.
The intervals measure query-image uncertainty with supports, checkpoints and
parameters held fixed.
