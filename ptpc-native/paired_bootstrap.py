#!/usr/bin/env python3
"""Paired query-image bootstrap for the frozen canonical native-host Guard.

Matching is from the official COCOeval (also the saved native-host VOC
protocol). Cached image matches are resampled, and 101-point AP is recomputed
for each draw. Images with no novel GT remain in the sampling population.
The same seed/image ordering must be used across a host's shots and splits.
This estimates query-image uncertainty conditional on checkpoints/supports.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def reconstruct(spec):
    from rescore_guard import load_pair, guard_scores, parameters

    cell = Path(spec['cell'])
    baseline_path = cell / 'predictions_baseline.json'
    diag_path = cell / ('diagnostics.json' if spec['host'] == 'fsod_vfm'
                        else 'guard_diagnostics.json')
    hashes = {'baseline_predictions': sha256(baseline_path),
              'diagnostics': sha256(diag_path),
              'ground_truth': sha256(spec['ground_truth'])}
    for key, value in hashes.items():
        expected = spec.get('inputs_sha256', {}).get(key, value)
        if value != expected:
            raise ValueError(f'Input hash mismatch: {key}')
    baseline, native, gap, nonfinite = load_pair(spec['host'], baseline_path, diag_path)
    canonical, eta = parameters(spec['host'], spec['dataset'])
    if spec.get('parameters', canonical) != canonical or spec.get('temperature', eta) != eta:
        raise ValueError('Manifest is not the canonical cross-dataset continuous Guard')
    scores = guard_scores(native, gap, canonical, eta)
    return baseline, native, scores, hashes, nonfinite


def cache_evaluation(gt, predictions, scores, categories, thresholds):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([{**r, 'score': float(s)} for r, s in zip(predictions, scores)])
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.catIds = categories
        ev.params.iouThrs = np.asarray(thresholds)
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ['all']
        ev.params.maxDets = [100]
        ev.evaluate()
        ev.accumulate()
    n = len(ev.params.imgIds)
    arrays = []
    for c in range(len(ev.params.catIds)):
        records = ev.evalImgs[c * n:(c + 1) * n]
        lengths = np.array([len(r['dtScores']) if r else 0 for r in records])
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        gt_count = np.array([np.count_nonzero(r['gtIgnore'] == 0) if r else 0
                             for r in records])
        nonempty = [r for r in records if r is not None]
        if nonempty:
            ds = np.concatenate([r['dtScores'] for r in nonempty])
            dm = np.concatenate([r['dtMatches'] for r in nonempty], axis=1)
            di = np.concatenate([r['dtIgnore'] for r in nonempty], axis=1)
            tp = (dm > 0) & (~di)
            fp = (dm == 0) & (~di)
        else:
            ds = np.empty(0)
            tp = fp = np.zeros((len(thresholds), 0), bool)
        arrays.append((lengths, offsets, gt_count, ds, tp, fp))
    precision = ev.eval['precision']
    official = float(precision[precision > -1].mean())
    cached = bootstrap_ap(arrays, np.arange(n))
    if abs(official - cached) > 1e-12:
        raise ValueError(f'Cached accumulator mismatch: {official} != {cached}')
    return arrays, ev, official


def bootstrap_ap(arrays, picks):
    """Exact COCO accumulation with stable ties in sampled-image order."""
    class_values = []
    for lengths, offsets, gt_count, scores, tp, fp in arrays:
        positives = int(gt_count[picks].sum())
        if positives == 0:
            continue
        lens = lengths[picks]
        total = int(lens.sum())
        if total == 0:
            class_values.append(0.)
            continue
        # Expand image copies without a Python loop over detections or images.
        index = (np.repeat(offsets[picks], lens) + np.arange(total)
                 - np.repeat(np.cumsum(lens) - lens, lens))
        order = np.argsort(-scores[index], kind='mergesort')
        index = index[order]
        cumulative_tp = np.cumsum(tp[:, index], axis=1, dtype=np.float64)
        cumulative_fp = np.cumsum(fp[:, index], axis=1, dtype=np.float64)
        recall = cumulative_tp / positives
        precision = cumulative_tp / (cumulative_tp + cumulative_fp + np.spacing(1))
        precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
        sampled_precision = np.zeros((tp.shape[0], 101))
        for t in range(tp.shape[0]):
            at = np.searchsorted(recall[t], np.linspace(0, 1, 101), side='left')
            valid = at < total
            sampled_precision[t, valid] = precision[t, at[valid]]
        class_values.append(float(sampled_precision.mean()))
    if not class_values:
        raise ValueError('A bootstrap draw has no evaluable ground-truth classes')
    return float(np.mean(class_values))


def check_cached_resample(ev, arrays, picks):
    """Compare one resample with unmodified official COCOeval.accumulate."""
    n = len(ev.params.imgIds)
    sampled = copy.copy(ev)
    sampled.params = copy.deepcopy(ev.params)
    sampled.params.imgIds = list(range(n))
    sampled._paramsEval = copy.deepcopy(sampled.params)
    sampled.evalImgs = [ev.evalImgs[c * n + i]
                        for c in range(len(ev.params.catIds)) for i in picks]
    with contextlib.redirect_stdout(io.StringIO()):
        sampled.accumulate()
    pr = sampled.eval['precision']
    official = float(pr[pr > -1].mean())
    custom = bootstrap_ap(arrays, picks)
    if abs(official - custom) > 1e-12:
        raise ValueError(f'Resample accumulator mismatch: {official} != {custom}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cell-index', type=int, required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error('--replicates must be at least 2')
    spec = load(args.manifest)[args.cell_index]
    started = time.monotonic()
    baseline, native, guard, hashes, nonfinite = reconstruct(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(spec['ground_truth'])
    gt.dataset.setdefault('info', {})
    categories = sorted(gt.getCatIds(catNms=spec['target_classes']))
    if len(categories) != len(spec['target_classes']):
        raise ValueError('Missing/duplicate novel category')
    thresholds = [.5] if spec['dataset'] == 'voc' else np.linspace(.5, .95, 10)
    cache0, ev0, point0 = cache_evaluation(gt, baseline, native, categories, thresholds)
    cache1, ev1, point1 = cache_evaluation(gt, baseline, guard, categories, thresholds)
    metric = 'AP50' if spec['dataset'] == 'voc' else 'AP'
    for measured, key in [(point0, 'native'), (point1, 'soft')]:
        if key in spec and abs(measured - spec[key][metric]) > 1e-8:
            raise ValueError(f'Archived {key} {metric} mismatch: '
                             f'{measured} != {spec[key][metric]}')
    n = len(ev0.params.imgIds)
    if ev0.params.imgIds != ev1.params.imgIds:
        raise ValueError('Unpaired image populations')
    image_ids_sha256 = hashlib.sha256(json.dumps(
        [int(x) for x in ev0.params.imgIds]).encode()).hexdigest()
    rng = np.random.default_rng(args.seed)
    pilot = rng.integers(0, n, n)
    check_cached_resample(ev0, cache0, pilot)
    check_cached_resample(ev1, cache1, pilot)
    # Reset so every cell shares exactly the same replicate image draws.
    rng = np.random.default_rng(args.seed)
    deltas = []
    print(f"Validated {spec['host']} {spec['dataset']} split={spec['split']} "
          f"shot={spec['shot']}: {100 * (point1 - point0):.5f} {metric}", flush=True)
    for b in range(args.replicates):
        picks = rng.integers(0, n, n)
        deltas.append(100 * (bootstrap_ap(cache1, picks) - bootstrap_ap(cache0, picks)))
        if (b + 1) % 100 == 0:
            print(f'{b + 1}/{args.replicates}, {time.monotonic() - started:.1f}s', flush=True)
    save(args.output, {
        'schema': 'ptpc-canonical-paired-image-bootstrap-v1',
        'spec': spec, 'metric': metric, 'units': 'percentage points',
        'seed': args.seed, 'replicates': args.replicates, 'images': n,
        'image_ids_sha256': image_ids_sha256, 'input_sha256': hashes,
        'script_sha256': sha256(__file__), 'paired_nonfinite_scores': nonfinite,
        'native': 100 * point0, 'guard': 100 * point1,
        'delta': 100 * (point1 - point0),
        'percentile_95_ci': np.quantile(deltas, [.025, .975]).tolist(),
        'bootstrap_deltas': deltas,
        'point_evaluation_passed': True,
        'reference_metrics_checked': all(k in spec for k in ('native', 'soft')),
        'official_accumulator_and_resample_checks_passed': True,
        'scope': 'query-image sampling conditional on fixed checkpoints/supports/parameters',
        'elapsed_seconds': time.monotonic() - started,
    })


if __name__ == '__main__':
    main()
