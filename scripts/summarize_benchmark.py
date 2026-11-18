"""Aggregate six archived US-05-03 reports without importing scientific code.

Default/--check is read-only. --write updates only the two derived summaries.
Original metrics, CSVs, thresholds and spatial artifacts are never rewritten.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ('bottle', 'hazelnut', 'transistor')
FAMILIES = ('cae', 'patchcore')
SOURCES = {
    'cae-bottle': ('kaggle-002', '9393551e7c116dabcdaa539b669bb15debbd2e05529b1169bedf92d3c9b104ef'),
    'cae-hazelnut': ('kaggle-002', '3cb839efbd128a0881a394f60b277c0662c7b8b8815b67f15d723b1bbca1d5d5'),
    'cae-transistor': ('kaggle-001', 'b46e66ff9f11f6af17ae0d453490217bca537e7a1f020348d2fa3a0060f89e19'),
    'patchcore-bottle': ('kaggle-001', '009e3c2153d6c5f8a98ede447e0bb215d4cbf4820ff6ac1e377f63df59dee56e'),
    'patchcore-hazelnut': ('kaggle-002', '575384adb3edb7b8bd40156c951625ed0c67c9a0474e2bc91367b6b3637af56a'),
    'patchcore-transistor': ('kaggle-002', '6d97d78c4bb7cf25d706c67f041782611f41a18bdf47e3fb957ef08340abd565'),
}
METRICS = ('image_auroc', 'pixel_auroc', 'precision', 'recall', 'f1', 'latency_mean_ms')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(root=ROOT):
    """Read a fixed complete cohort, verifying source files against pinned manifests."""
    rows = []
    for instance, (batch, anchor) in SOURCES.items():
        folder = root / 'experiments/benchmark' / batch / instance
        manifest_bytes = (folder / 'manifest.json').read_bytes()
        if sha(manifest_bytes) != anchor:
            raise ValueError(f'Manifest anchor mismatch: {instance}')
        manifest = json.loads(manifest_bytes)
        for filename in ('metrics.json', 'predictions.csv'):
            if sha((folder / filename).read_bytes()) != manifest['files'][filename]:
                raise ValueError(f'Original bytes changed: {instance}/{filename}')
        report = json.loads((folder / 'metrics.json').read_bytes())
        run, metrics = report['run'], report['metrics']
        if run['run_id'] != instance or run['split'] != 'test':
            raise ValueError(f'Invalid report identity: {instance}')
        row = {
            'instance': instance, 'model_family': run['model_family'], 'category': run['category'],
            'source_experiment': run['source_experiment'], 'n_images': report['n_images'],
            'n_normal': metrics['image']['n_normal'], 'n_anomalous': metrics['image']['n_anomalous'],
            **{key: metrics['image'][key] for key in ('image_auroc', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn', 'threshold')},
            'threshold_hex': run['threshold_hex'], 'pixel_auroc': metrics['pixel']['pixel_auroc'],
            **{f'latency_{key}': metrics['latency'][key] for key in ('mean_ms', 'median_ms', 'p95_ms', 'min_ms', 'max_ms')},
            'source_metrics': (folder / 'metrics.json').relative_to(root).as_posix(),
            'source_metrics_sha256': manifest['files']['metrics.json'],
            'source_manifest_sha256': anchor,
        }
        rows.append(row)
    return rows


def aggregate(rows):
    """Unweighted means of exactly three categories; a missing metric stays undefined.

    A None value is counted and makes the three-category macro None. It is never
    silently converted to zero or averaged over a smaller, changing denominator.
    """
    expected = {(family, category) for family in FAMILIES for category in CATEGORIES}
    pairs = [(r['model_family'], r['category']) for r in rows]
    if len(pairs) != 6 or set(pairs) != expected:
        raise ValueError('Exactly one report for each of the six family/category pairs is required')
    macros = []
    for family in FAMILIES:
        selected = sorted((r for r in rows if r['model_family'] == family), key=lambda r: r['category'])
        macro = {'model_family': family, 'categories': list(CATEGORIES), 'category_count': 3,
                 'image_count': sum(r['n_images'] for r in selected), 'metrics': {}}
        for key in METRICS:
            values = [r[key] for r in selected]
            defined = [v for v in values if v is not None]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in defined):
                raise ValueError(f'Invalid metric: {family}/{key}')
            macro['metrics'][key] = {
                'value': math.fsum(defined) / 3 if len(defined) == 3 else None,
                'defined_categories': len(defined), 'undefined_categories': 3 - len(defined),
            }
        macros.append(macro)
    return {
        'schema_version': '1.0', 'status': 'complete_6_of_6',
        'scientific_code_commit': '0acbf26880b755373f2e61236d009ed16f04658c',
        'threshold_registry_integrity_sha256': '41dd9d26e68840ca5d104062605d967e72a2d3907937e8ed6745743034c96b6a',
        'aggregation': 'unweighted_per_category',
        'formula': '(metric_bottle + metric_hazelnut + metric_transistor) / 3',
        'undefined_policy': 'Report counts; macro is null unless all three category metrics are defined',
        'latency_aggregation': 'Arithmetic mean of the three per-category mean_ms; not a pooled image mean or pooled percentile',
        'score_pooling_across_instances': False,
        'n_predictions': sum(r['n_images'] for r in rows),
        'n_distinct_test_images': sum(r['n_images'] for r in rows if r['model_family'] == 'cae'),
        'instances': rows, 'macro_by_family': macros,
    }


def render_outputs(rows):
    summary = aggregate(rows)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return {
        'comparison.csv': stream.getvalue().encode('utf-8'),
        'summary.json': (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write', action='store_true', help='Write only comparison.csv and summary.json')
    group.add_argument('--check', action='store_true', help='Verify derived outputs without writing (default)')
    args = parser.parse_args()
    outputs = render_outputs(read_rows())
    for name, data in outputs.items():
        path = ROOT / 'experiments/benchmark' / name
        if args.write:
            path.write_bytes(data)
        elif path.read_bytes() != data:
            raise SystemExit(f'Summary differs: {path}; inspect sources before --write')
    print('PASS: six reports; unweighted three-category macros; derived summaries match.')


if __name__ == '__main__':
    main()
