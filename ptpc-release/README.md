# DeFRCN + PTPC

PTPC evaluation on [DeFRCN](https://github.com/er-muyue/DeFRCN).
See the [main README](../README.md) for dependencies, data, and checkpoint paths.

## Evaluation

```bash
bash run_ptpc_voc.sh vanilla_defrcn 1
bash run_ptpc_coco.sh vanilla_defrcn
```

VOC uses shots 1, 2, 3, 5, and 10; COCO adds 30-shot. Both use support seed 0.
`SHOTS`, `SEEDS`, `TRAINED_ROOT`, `EVAL_ROOT`, and `PYTHON_BIN` can be set as
environment variables. For example:

```bash
SHOTS="1 5" TRAINED_ROOT=/path/to/split1 bash run_ptpc_voc.sh vanilla_defrcn 1
```

Results are written under:

```text
checkpoints/voc/<experiment>/ptpc_continuous_eval/split<S>/ptpc/
checkpoints/coco/<experiment>/ptpc_continuous_eval/ptpc/
```

## Implementation

`defrcn/evaluation/novel_methods/ptpc.py` contains `PTPCFeatureExtractor`,
`PTPCCalibrator`, and `PTPCGuard`. The novel module pools full-image patch
tokens, compares novel prototypes, and fuses the resulting compatibility with
the detector score. The Guard uses crop-CLS descriptors and base/novel prototypes.

The method name is `ptpc`; its config sections are `NOVEL_METHODS.PTPC` and
`NOVEL_METHODS.PTPC_GUARD`. Set `PTPC_GUARD.ENABLE=False` for novel-module-only
evaluation or `PTPC_GUARD.SMOOTH_ENABLE=False` for the binary-limit Guard.
The VOC and COCO runners enable continuous suppression with temperatures
0.04 and 0.01, respectively.

`defrcn/evaluation/calibration_layer.py` implements the PCB baseline.
`run_voc.sh` and `run_coco.sh` train and fine-tune the detector.
