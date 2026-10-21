#!/usr/bin/env python3
"""Run all frozen native-host cells and summarize correlated benchmark means."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--prepare-manifest', action='store_true',
                   help='Write a new 38-cell manifest from locally generated inference outputs')
    p.add_argument('--runs-root', type=Path)
    p.add_argument('--fsod-vfm-repo', type=Path)
    p.add_argument('--no-time-repo', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--summarize-only', action='store_true',
                   help='Validate and summarize already completed per-cell result files')
    args = p.parse_args()
    if args.prepare_manifest:
        if not all((args.runs_root, args.fsod_vfm_repo, args.no_time_repo)):
            p.error('Manifest preparation requires --runs-root and both host repo paths')
        if args.manifest.exists():
            raise FileExistsError(args.manifest)
        from rescore_guard import parameters, sha256
        voc_classes = {
            1: ['bus', 'sofa', 'cow', 'bird', 'motorbike'],
            2: ['horse', 'aeroplane', 'bottle', 'sofa', 'cow'],
            3: ['cat', 'motorbike', 'boat', 'sofa', 'sheep'],
        }
        coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv']
        specs = []
        for host in ('fsod_vfm', 'no_time'):
            for dataset in ('voc', 'coco'):
                splits = (1, 2, 3) if dataset == 'voc' else (None,)
                shots = (1, 2, 3, 5, 10) if dataset == 'voc' else (
                    (1, 2, 3, 5, 10, 30) if host == 'fsod_vfm' else (10, 30))
                for split in splits:
                    for shot in shots:
                        seed = 42 if (host, dataset) == ('no_time', 'coco') else 33
                        cell = args.runs_root.resolve() / host / dataset
                        if split is not None:
                            cell /= f'split{split}'
                        cell /= f'{shot}shot_seed{seed}'
                        if host == 'fsod_vfm':
                            repo = args.fsod_vfm_repo.resolve()
                            gt = repo / ('data/Pascal/VOC2007Test/voc07test_coco_format.json'
                                         if dataset == 'voc' else
                                         'data/coco/annotations/instances_val2017.json')
                            diag = cell / 'diagnostics.json'
                        else:
                            repo = args.no_time_repo.resolve()
                            gt = repo / ('data/pascal_voc/annotations/voc07_test_with_segm.json'
                                         if dataset == 'voc' else
                                         'data/coco/annotations/instances_val2017.json')
                            diag = cell / 'guard_diagnostics.json'
                        params, eta = parameters(host, dataset)
                        specs.append(dict(host=host, dataset=dataset, split=split,
                                          shot=shot, cell=str(cell), ground_truth=str(gt),
                                          target_classes=voc_classes[split] if split else coco_classes,
                                          parameters=params, temperature=eta,
                                          inputs_sha256=dict(
                                              baseline_predictions=sha256(cell / 'predictions_baseline.json'),
                                              diagnostics=sha256(diag), ground_truth=sha256(gt))))
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(specs, indent=2) + '\n')
        print(f'Wrote {len(specs)} paired cells to {args.manifest}')
        return
    if args.output_dir is None:
        p.error('--output-dir is required for evaluation')
    if args.workers < 1 or args.replicates < 2:
        p.error('--workers must be positive and --replicates must be at least 2')
    specs = json.loads(args.manifest.read_text())
    identities = [(s['host'], s['dataset'], s['split'], s['shot']) for s in specs]
    if not specs or len(set(identities)) != len(identities):
        raise ValueError('Empty or duplicate-cell manifest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('paired_bootstrap.py')

    def run(index):
        output = args.output_dir / f'cell_{index:02d}.json'
        log = args.output_dir / f'cell_{index:02d}.log'
        with log.open('w') as handle:
            subprocess.run([sys.executable, str(script), '--manifest', str(args.manifest),
                            '--cell-index', str(index), '--replicates', str(args.replicates),
                            '--output', str(output)], stdout=handle,
                           stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OPENBLAS_NUM_THREADS': '1',
                                'OMP_NUM_THREADS': '1'})
        result = json.loads(output.read_text())
        spec = specs[index]
        print(f"DONE {index:02d} {spec['host']} {spec['dataset']} "
              f"split={spec['split']} shot={spec['shot']} "
              f"delta={result['delta']:.4f} CI={result['percentile_95_ci']}", flush=True)
        return result

    if args.summarize_only:
        rows = [json.loads((args.output_dir / f'cell_{i:02d}.json').read_text())
                for i in range(len(specs))]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run, range(len(specs))))
    for spec, row in zip(specs, rows):
        if row['spec'] != spec or row['replicates'] != args.replicates:
            raise ValueError('Completed result does not match the frozen manifest/replicate count')
        if not (row['point_evaluation_passed']
                and row['official_accumulator_and_resample_checks_passed']):
            raise ValueError('Completed result did not pass evaluator validation')
    summaries = []
    for host in ['fsod_vfm', 'no_time']:
        for dataset in ['voc', 'coco']:
            selected = [r for r in rows if r['spec']['host'] == host
                        and r['spec']['dataset'] == dataset]
            if not selected:
                continue
            # Reuse the same resampled images across every correlated shot/split.
            signatures = {(r['seed'], r['replicates'], r['images'], r['image_ids_sha256'])
                          for r in selected}
            if len(signatures) != 1:
                raise ValueError('Unpaired replicate populations across shots/splits')
            means = np.mean([r['bootstrap_deltas'] for r in selected], axis=0)
            summaries.append({
                'host': host, 'dataset': dataset, 'cells': len(selected),
                'metric': selected[0]['metric'], 'units': 'percentage points',
                'delta': float(np.mean([r['delta'] for r in selected])),
                'percentile_95_ci': np.quantile(means, [.025, .975]).tolist(),
                'replicates': args.replicates, 'seed': selected[0]['seed'],
                'image_ids_sha256': selected[0]['image_ids_sha256'],
                'images': selected[0]['images'],
                'bootstrap_mean_deltas': means.tolist(),
                'scope': 'fixed checkpoints/supports/parameters; synchronized paired image resampling',
                'individual_cell_CIs_excluding_zero': sum(r['percentile_95_ci'][0] > 0
                                                         for r in selected),
            })
    result = {'method': 'paired percentile image bootstrap; same draws across shots/splits',
              'replicates': args.replicates, 'cells': len(rows), 'summaries': summaries}
    (args.output_dir / 'benchmark_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for r in summaries:
        print({k: v for k, v in r.items() if k != 'bootstrap_mean_deltas'}, flush=True)


if __name__ == '__main__':
    main()
