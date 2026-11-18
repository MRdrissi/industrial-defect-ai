"""Verify the final report and guard against partial cohorts or micro averaging."""
import copy
import csv
import importlib.util
import io
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('benchmark_summary', ROOT / 'scripts/summarize_benchmark.py')
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_derived_files_match_all_six_pinned_originals():
    rows = summary.read_rows(ROOT)
    assert len(rows) == 6 and sum(r['n_images'] for r in rows) == 586
    for name, data in summary.render_outputs(rows).items():
        assert (ROOT / 'experiments/benchmark' / name).read_bytes() == data
    saved = json.loads((ROOT / 'experiments/benchmark/summary.json').read_bytes())
    assert saved['n_distinct_test_images'] == 293
    for macro in saved['macro_by_family']:
        selected = [r for r in rows if r['model_family'] == macro['model_family']]
        assert {r['category'] for r in selected} == {'bottle', 'hazelnut', 'transistor'}
        for key, result in macro['metrics'].items():
            assert result['value'] == math.fsum(r[key] for r in selected) / 3
            assert result['defined_categories'] == 3 and result['undefined_categories'] == 0
    # Published table retains full precision; docs round only for readability.
    table = list(csv.DictReader(io.StringIO((ROOT / 'experiments/benchmark/comparison.csv').read_text())))
    assert [r['instance'] for r in table] == list(summary.SOURCES)
    for row, original in zip(table, rows):
        assert float(row['f1']) == original['f1']
    assert 'torch' not in summary.__dict__ and 'src' not in summary.__dict__


def test_macro_weights_categories_equally_and_averages_f1_directly():
    rows = copy.deepcopy(summary.read_rows(ROOT))
    for row in rows:
        # Deliberately imbalanced sizes make image weighting observably incorrect.
        position = summary.CATEGORIES.index(row['category'])
        row['n_images'] = (1, 10, 1000)[position]
        row['f1'] = (0.0, 0.3, 0.9)[position]
        row['precision'] = 1.0
        row['recall'] = 0.5
        row['latency_mean_ms'] = (1.0, 10.0, 100.0)[position]
    for macro in summary.aggregate(rows)['macro_by_family']:
        assert macro['metrics']['f1']['value'] == pytest.approx(0.4)
        assert macro['metrics']['f1']['value'] != pytest.approx(2 / 3)
        assert macro['metrics']['latency_mean_ms']['value'] == 37.0


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'unknown_category'])
def test_incomplete_or_ambiguous_cohort_is_rejected(case):
    rows = copy.deepcopy(summary.read_rows(ROOT))
    if case == 'missing':
        rows.pop()
    elif case == 'duplicate':
        rows[-1] = rows[0]
    else:
        rows[-1]['category'] = 'unknown'
    with pytest.raises(ValueError, match='six family/category pairs'):
        summary.aggregate(rows)


def test_undefined_metric_is_counted_and_never_silently_dropped():
    rows = copy.deepcopy(summary.read_rows(ROOT))
    rows[0]['pixel_auroc'] = None
    result = summary.aggregate(rows)['macro_by_family'][0]['metrics']['pixel_auroc']
    assert result == {'value': None, 'defined_categories': 2, 'undefined_categories': 1}


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True])
def test_nonfinite_or_boolean_metrics_are_rejected(value):
    rows = copy.deepcopy(summary.read_rows(ROOT))
    rows[0]['f1'] = value
    with pytest.raises(ValueError, match='Invalid metric'):
        summary.aggregate(rows)


def test_source_tampering_fails_before_aggregation(tmp_path):
    instance, (batch, _) = next(iter(summary.SOURCES.items()))
    relative = Path('experiments/benchmark') / batch / instance
    target = tmp_path / relative
    target.mkdir(parents=True)
    for name in ('manifest.json', 'metrics.json', 'predictions.csv'):
        (target / name).write_bytes((ROOT / relative / name).read_bytes())
    with (target / 'metrics.json').open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='Original bytes changed'):
        summary.read_rows(tmp_path)
