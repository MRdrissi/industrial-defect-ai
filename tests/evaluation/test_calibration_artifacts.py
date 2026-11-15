"""Consume original Kaggle Validation evidence; never infer or recalibrate."""
import csv
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import calibration as calibration
from src.evaluation import run_calibration as runner

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'experiments/calibration/kaggle-001'
PROVENANCE = ROOT / 'experiments/calibration/kaggle-001-provenance.json'
ARCHIVE_SHA = 'da303b28d40f3605c2f32660afbf1b52e660aba7cef6ebbc432b7a7f139c11a9'
SESSION_SHA = 'd2f24b18beb1befcab8a585181bfdf308bcd73aeb8163e70f249315d377b35c8'
CODE_COMMIT = 'c708de9ce2b682ab8cea00eabd5d3e8c7b2a0ae6'
# Execution-time anchors; not recomputed from potentially modified evidence.
EXPECTED = {
    'cae-bottle': ('b3b5b96e36cecc248c85f8a81b23b0c66769d77f870c52af777a35fd04b4b459', '0x1.4059f51999997p-6', 42),
    'cae-hazelnut': ('bc5f999100f1d6c94e1b7a3513bfe22c37148c49870e527678e267dac5487e1a', '0x1.145e0d7fffffcp-7', 78),
    'patchcore-hazelnut': ('787db120462857d7593aeeffd3ad418de07b157c0d830853e4f1197768916e76', '0x1.86b2ee0000000p+5', 78),
    'patchcore-transistor': ('8e3bbf94a7d5e28d190a1ca679f3563f136642a7251d95df495e4a8db076d99e', '0x1.47ab1f6666666p+5', 43),
}
MISSING = {'cae-transistor', 'patchcore-bottle'}


def read_json(path):
    return calibration._json(path.read_bytes())


@pytest.fixture
def no_recalibration(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Archived thresholds must be consumed without fitting or inference')
    monkeypatch.setattr(np, 'quantile', forbidden)
    monkeypatch.setattr(np, 'percentile', forbidden)
    monkeypatch.setattr(calibration, 'calibrate', forbidden)
    monkeypatch.setattr(runner, 'run_instance', forbidden)
    monkeypatch.setattr(runner, 'restore_instance', forbidden)


def load_original(instance):
    catalog = runner.read_catalog()
    item = next(i for i in catalog['instances'] if i['id'] == instance)
    threshold = calibration.load_threshold(
        EVIDENCE / instance, expected_integrity_sha256=EXPECTED[instance][0],
        manifest_path=ROOT/item['manifest'], protocol_path=ROOT/catalog['protocol'],
        expected_manifest_sha256=item['manifest_sha256'],
        expected_protocol_sha256=catalog['protocol_sha256'],
        expected_count=item['expected_validation_count'],
        model_family=item['model_family'], category=item['category'],
        expected_checkpoint_sha256=item['checkpoint_sha256'])
    return threshold, item, catalog


def test_original_inventory_and_execution_time_anchors():
    provenance = read_json(PROVENANCE)
    names = {'session.json'} | {f'{i}/{f}' for i in EXPECTED for f in
                               ('calibration.json', 'integrity.json', 'validation_scores.csv')}
    actual = {p.relative_to(EVIDENCE).as_posix() for p in EVIDENCE.rglob('*') if p.is_file()}
    assert actual == names == set(provenance['files_sha256'])
    assert len(actual) == 13
    assert provenance['archive']['sha256'] == ARCHIVE_SHA
    assert provenance['archive']['size_bytes'] == 10743
    assert provenance['kaggle']['script_version_id'] == 369724364
    assert provenance['scientific_code_commit'] == CODE_COMMIT
    assert provenance['bundle_integrity_sha256'] == {i: e[0] for i, e in EXPECTED.items()}
    for name, digest in provenance['files_sha256'].items():
        assert calibration.file_sha256(EVIDENCE/name) == digest
    assert calibration.file_sha256(EVIDENCE/'session.json') == SESSION_SHA
    session = read_json(EVIDENCE/'session.json')
    assert session['code_commit'] == CODE_COMMIT
    assert session['executed_at_utc'] == provenance['executed_at_utc']
    results = {r['instance']: r for r in session['results']}
    assert len(session['results']) == len(results) == 6
    assert set(results) == set(EXPECTED) | MISSING
    for instance in MISSING:
        assert results[instance] == {'instance': instance, 'status': 'missing_checkpoint'}
    for instance, (anchor, value_hex, count) in EXPECTED.items():
        result = results[instance]
        assert result['status'] == 'calibrated'
        assert result['integrity_sha256'] == anchor
        assert result['threshold'].hex() == result['threshold_hex'] == value_hex
        assert result['validation_images'] == count


@pytest.mark.parametrize('instance', EXPECTED)
def test_real_threshold_exact_reuse_and_validation_coverage(instance, no_recalibration):
    frozen, item, catalog = load_original(instance)
    anchor, value_hex, count = EXPECTED[instance]
    assert frozen.threshold.hex() == value_hex
    assert frozen.calibration_integrity_sha256 == anchor
    np.testing.assert_array_equal(
        frozen.predict([frozen.threshold, np.nextafter(frozen.threshold, np.inf)]), [0, 1])
    report = read_json(EVIDENCE/instance/'calibration.json')
    assert report['provenance']['git_commit'] == CODE_COMMIT
    assert report['provenance']['source_experiment'] == item['source_experiment']
    assert report['provenance']['score_definition'] == item['score_definition']
    assert report['provenance']['preprocessing'] == item['preprocessing']
    assert report['provenance']['model_config_canonical_sha256'] == runner.canonical_json_sha256(ROOT/item['model_config'])
    assert report['provenance']['catalog_sha256'] == calibration.file_sha256(runner.CATALOG)
    expected_paths = calibration.validation_paths(
        ROOT/item['manifest'], category=item['category'],
        expected_sha256=item['manifest_sha256'], expected_count=count)
    with (EVIDENCE/instance/'validation_scores.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    paths = [r['image_path'] for r in rows]
    assert len(rows) == len(set(paths)) == count
    assert set(paths) == expected_paths
    assert all(r['label'] == '0' for r in rows)
    assert all(np.isfinite(float(r['score'])) for r in rows)
    assert all('/train/good/' in path for path in paths)  # Physical train != logical Train.
    assert report['protocol_sha256'] == catalog['protocol_sha256']


def test_availability_distinguishes_calibrated_and_missing():
    availability = read_json(ROOT/'experiments/calibration/availability.json')
    assert availability['calibrated_instances'] == 4
    assert availability['kaggle_script_version_id'] == 369724364
    entries = {e['instance']: e for e in availability['instances']}
    assert len(availability['instances']) == len(entries) == 6
    assert set(entries) == set(EXPECTED) | MISSING
    for instance, (anchor, value_hex, count) in EXPECTED.items():
        entry = entries[instance]
        assert entry['status'] == 'calibrated_archived'
        assert entry['threshold'].hex() == entry['threshold_hex'] == value_hex
        assert entry['real_scores_collected'] == count
        assert entry['integrity_sha256'] == anchor
        assert ROOT/entry['evidence_directory'] == EVIDENCE/instance
    for instance in MISSING:
        assert entries[instance]['status'] == 'blocked_checkpoint_access'
        assert entries[instance]['threshold'] is None
        assert entries[instance]['real_scores_collected'] == 0
        assert not (EVIDENCE/instance).exists()


def test_four_real_thresholds_cannot_form_six_instance_registry(tmp_path, no_recalibration):
    four = [load_original(instance)[0] for instance in EXPECTED]
    target = tmp_path/'six-thresholds'
    with pytest.raises(ValueError, match='Exactly six unique instances'):
        calibration.export_threshold_registry(target, four)
    assert not target.exists()
