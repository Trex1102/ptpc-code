#!/usr/bin/env python3
"""Canonical native-host Guard, using paired inference sufficient statistics.

This is also the input adapter for paired_bootstrap.py. No query label enters
rescoring. Native-host arithmetic deliberately follows the paper experiments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


# (margin, suppression floor, lower exclusive, upper inclusive, eta).
# Every tuple was selected on the OTHER benchmark, including eta.
FROZEN = {
    ("fsod_vfm", "voc"): (.01, .85, .05, 1., .04),
    ("fsod_vfm", "coco"): (.05, .50, .05, 1., .01),
    ("no_time", "voc"): (.01, .70, 0., .70, .04),
    ("no_time", "coco"): (.01, .50, 0., .70, .005),
}


def parameters(host, dataset):
    margin, suppression, lower, upper, eta = FROZEN[host, dataset]
    return dict(margin=margin, suppression=suppression, lower=lower, upper=upper), eta


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guard_scores(native, gap, params, temperature):
    """Float32 continuous rule; temperature=None is the binary-limit control."""
    native = np.asarray(native, dtype=np.float32)
    gap = np.asarray(gap, dtype=np.float32)
    if native.shape != gap.shape or not np.isfinite(gap).all():
        raise ValueError('Invalid paired score/evidence arrays')
    if not np.isfinite(native).all() or np.any((native < 0) | (native > 1)):
        raise ValueError('Scores must be finite probabilities')
    margin, suppression, lower, upper = (np.float32(params[k]) for k in
                                         ('margin', 'suppression', 'lower', 'upper'))
    if not (np.isfinite(margin) and 0 < suppression <= 1 and 0 <= lower < upper <= 1):
        raise ValueError('Invalid Guard parameters')
    band = (native > lower) & (native <= upper)
    if temperature is None:
        weight = (band & (gap > margin)).astype(np.float32)
    else:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Guard eta must be finite and positive')
        z = np.clip((gap - margin) / np.float32(temperature), -60., 60.)
        weight = (1 / (1 + np.exp(-z))).astype(np.float32)
        weight = np.where(band, weight, np.float32(0)).astype(np.float32)
    factor = np.float32(1) - (np.float32(1) - suppression) * weight
    return np.where(band, native * factor, native).astype(np.float32)


def load_pair(host, baseline_path, diagnostics_path):
    baseline = json.loads(Path(baseline_path).read_text())
    diagnostics = json.loads(Path(diagnostics_path).read_text())
    if host == 'fsod_vfm':
        if len(baseline) != len(diagnostics):
            raise ValueError('Prediction/diagnostic length mismatch')
        for i, (pred, diag) in enumerate(zip(baseline, diagnostics)):
            if any(pred[k] != diag[k] for k in ('image_id', 'category_id', 'bbox')):
                raise ValueError(f'Prediction identity mismatch at {i}')
        native = np.asarray([r['baseline_score'] for r in diagnostics], np.float32)
        # FSOD-VFM originally subtracts Python scalars before casting the gap.
        gap = np.asarray([float(r['max_base']) - float(r['max_novel'])
                          for r in diagnostics], np.float32)
    elif host == 'no_time':
        native_values, base_values, novel_values = [], [], []
        cursor = 0
        category_by_label = {}
        for record in diagnostics:
            n = len(record['baseline_scores'])
            keys = ('max_base', 'max_novel', 'labels', 'boxes_xyxy')
            if any(len(record[k]) != n for k in keys) or cursor + n > len(baseline):
                raise ValueError('Prediction/diagnostic length mismatch')
            for j in range(n):
                pred = baseline[cursor + j]
                if pred['image_id'] != record['image_id']:
                    raise ValueError(f'Image identity mismatch at {cursor + j}')
                label, category = int(record['labels'][j]), int(pred['category_id'])
                if category_by_label.setdefault(label, category) != category:
                    raise ValueError(f'Class identity mismatch at {cursor + j}')
                # encode_results converts class indices to dataset category IDs;
                # their mapping is host-specific. Check boxes before that mapping.
                x, y, w, h = pred['bbox']
                if not np.allclose([x, y, x + w, y + h], record['boxes_xyxy'][j],
                                   rtol=0, atol=2e-4):
                    raise ValueError(f'Box identity mismatch at {cursor + j}')
            native_values.extend(record['baseline_scores'])
            base_values.extend(record['max_base'])
            novel_values.extend(record['max_novel'])
            cursor += n
        if cursor != len(baseline):
            raise ValueError('Prediction/diagnostic length mismatch')
        mapped = [category_by_label[k] for k in sorted(category_by_label)]
        if mapped != sorted(set(mapped)):
            raise ValueError('No-Time category mapping is not ordered and one-to-one')
        native = np.asarray(native_values, np.float32)
        gap = np.asarray(base_values, np.float32) - np.asarray(novel_values, np.float32)
    else:
        raise ValueError(f'Unknown host: {host}')
    pred_scores = np.asarray([r['score'] for r in baseline], np.float64)
    if not np.array_equal(np.isfinite(pred_scores), np.isfinite(native)):
        raise ValueError('Unpaired non-finite confidence')
    finite = np.isfinite(native)
    if not np.allclose(pred_scores[finite], native[finite], rtol=0, atol=2e-6):
        raise ValueError('Baseline confidence mismatch')
    nonfinite = int((~finite).sum())
    if nonfinite and host != 'no_time':
        raise ValueError('Non-finite FSOD-VFM score')
    # Original No-Time evaluation maps exactly paired non-finite scores to zero.
    native = np.where(finite, native, np.float32(0)).astype(np.float32)
    if not np.isfinite(gap).all():
        raise ValueError('Non-finite prototype evidence')
    return baseline, native, gap, nonfinite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('fsod_vfm', 'no_time'), required=True)
    parser.add_argument('--dataset', choices=('voc', 'coco'), required=True)
    parser.add_argument('--cell', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary-limit', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_path = args.cell / 'predictions_baseline.json'
    diagnostics_path = args.cell / ('diagnostics.json' if args.host == 'fsod_vfm'
                                    else 'guard_diagnostics.json')
    rows, native, gap, nonfinite = load_pair(args.host, baseline_path, diagnostics_path)
    params, eta = parameters(args.host, args.dataset)
    scores = guard_scores(native, gap, params, None if args.binary_limit else eta)
    guarded = copy.deepcopy(rows)
    for row, score in zip(guarded, scores):
        row['score'] = float(score)
    # Deep copying all fields and replacing only score preserves even optional
    # segmentation fields, ordered identities, boxes, labels and counts.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(guarded, allow_nan=False) + '\n')
    print(json.dumps(dict(host=args.host, dataset=args.dataset, parameters=params,
                          eta=None if args.binary_limit else eta,
                          rule='binary_limit' if args.binary_limit else 'continuous',
                          predictions=len(rows), paired_nonfinite_scores=nonfinite)))


if __name__ == '__main__':
    main()
