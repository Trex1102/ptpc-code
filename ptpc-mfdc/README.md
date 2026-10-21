# MFDC + PTPC

PTPC evaluation on **Multi-Faceted Distillation of Base-Novel Commonality for
Few-Shot Object Detection** (ECCV 2022). MFDC builds on
[DeFRCN](https://github.com/er-muyue/DeFRCN).

See the [main README](../README.md) for dependencies and dataset layout.
Install this directory's `requirements.txt` after installing PyTorch and
Detectron2 v0.3.

## Evaluation

```bash
bash run_mfdc_novel_methods.sh 1 ptpc "1 2 3 5 10" "0" eval
bash run_mfdc_novel_methods_coco.sh ptpc "1 2 3 5 10 30" "0" eval
```

Run the VOC command for splits 1, 2, and 3. Set `PRETRAINED_NOVEL_ROOT` to
choose the checkpoint root, `SAVE_DIR` for outputs, and `PYTHON_BIN` for the
Python executable.

The method name is `ptpc`. Configs use `NOVEL_METHODS.PTPC` and
`NOVEL_METHODS.PTPC_GUARD`; implementation classes are in
`defrcn/evaluation/novel_methods/ptpc.py`.

The runners enable continuous Guard suppression with margin 0.05, suppression
floor 0.3, score band `(0.05, 1]`, and 20 base samples per class. The Guard
temperature is 0.04 on VOC and 0.01 on COCO. Set
`PTPC_GUARD.SMOOTH_ENABLE=False` for the binary-limit control.

## Encoder ablations

```bash
bash run_mfdc_ptpc_encoders.sh "1 2 3" "1 2 3 5 10" "0" \
  "s14 b14 l14 g14 clip_b16 clip_l14 siglip_b siglip_l"
```

This sweep uses the binary-limit Guard. `DRY_RUN=1` generates configs and
prints commands without running inference.

## Training

```bash
bash voc_train.sh vanilla_mfdc 1
bash coco_train.sh vanilla_mfdc
```

Pretrained ResNet weights belong in `weight/`; datasets and support splits
belong in `datasets/`.

## MFDC citation

```bibtex
@inproceedings{wu2022multi,
  title={Multi-faceted Distillation of Base-Novel Commonality for Few-Shot Object Detection},
  author={Wu, Shuang and Pei, Wenjie and Mei, Dianwen and Chen, Fanglin and Tian, Jiandong and Lu, Guangming},
  booktitle={European Conference on Computer Vision},
  pages={578--594},
  year={2022},
  organization={Springer}
}
```
