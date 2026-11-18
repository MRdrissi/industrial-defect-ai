"""Audit saved Kaggle scores; consume six frozen thresholds without fitting."""
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
FINAL_EVIDENCE = ROOT / 'experiments/calibration/kaggle-002'
FINAL_PROVENANCE = ROOT / 'experiments/calibration/kaggle-002-provenance.json'
FINAL_ARCHIVE_SHA = 'ec02a6554a1813a021e13fd6f66f0e8272937a676f4e17e6f174aedda42e03ab'
FINAL_SESSION_SHA = '279c756fa68e2e3c52e15c0f049d52d531006125c1551cc6290093dc16169895'
FINAL_EXPECTED = {
    'cae-transistor': ('2c2f141fa59427dbaba280cd953e5eb139bfd687e5cc02e33aa8029a9f7820b4', '0x1.c0581e0000000p-6', 43),
    'patchcore-bottle': ('1c39c5452c294170ec9020138c0bf79a70c4b971f7fb0153cf2ef1d2e36f7596', '0x1.f9ebb36666666p+4', 42),
}
ALL_EXPECTED = EXPECTED | FINAL_EXPECTED
REGISTRY = ROOT / 'experiments/calibration/six-thresholds'
REGISTRY_SHA = '41dd9d26e68840ca5d104062605d967e72a2d3907937e8ed6745743034c96b6a'


def read_json(path):
    return calibration._json(path.read_bytes())


@pytest.fixture
def no_inference_or_refitting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Archived thresholds must be consumed without fitting or inference')
    monkeypatch.setattr(calibration, 'calibrate', forbidden)
    monkeypatch.setattr(calibration, 'export_calibration', forbidden)
    monkeypatch.setattr(runner, 'export_calibration', forbidden)
    monkeypatch.setattr(runner, 'run_instance', forbidden)
    monkeypatch.setattr(runner, 'restore_instance', forbidden)
    monkeypatch.setattr(runner, 'collect_validation_scores', forbidden)


@pytest.fixture
def no_recalibration(monkeypatch, no_inference_or_refitting):
    def forbidden(*args, **kwargs):
        pytest.fail('Loading and registry assembly must not compute a quantile')
    monkeypatch.setattr(np, 'quantile', forbidden)
    monkeypatch.setattr(np, 'percentile', forbidden)


def evidence_for(instance):
    return (FINAL_EVIDENCE if instance in FINAL_EXPECTED else EVIDENCE) / instance


def load_original(instance):
    catalog = runner.read_catalog()
    item = next(i for i in catalog['instances'] if i['id'] == instance)
    threshold = calibration.load_threshold(
        evidence_for(instance), expected_integrity_sha256=ALL_EXPECTED[instance][0],
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


@pytest.mark.parametrize('instance', ALL_EXPECTED)
def test_real_threshold_exact_reuse_and_validation_coverage(instance, no_recalibration):
    frozen, item, catalog = load_original(instance)
    anchor, value_hex, count = ALL_EXPECTED[instance]
    assert frozen.threshold.hex() == value_hex
    assert frozen.calibration_integrity_sha256 == anchor
    np.testing.assert_array_equal(
        frozen.predict([frozen.threshold, np.nextafter(frozen.threshold, np.inf)]), [0, 1])
    report = read_json(evidence_for(instance)/'calibration.json')
    assert report['provenance']['git_commit'] == CODE_COMMIT
    assert report['provenance']['source_experiment'] == item['source_experiment']
    assert report['provenance']['score_definition'] == item['score_definition']
    assert report['provenance']['preprocessing'] == item['preprocessing']
    assert report['provenance']['model_config_canonical_sha256'] == runner.canonical_json_sha256(ROOT/item['model_config'])
    assert report['provenance']['catalog_sha256'] == calibration.file_sha256(runner.CATALOG)
    expected_paths = calibration.validation_paths(
        ROOT/item['manifest'], category=item['category'],
        expected_sha256=item['manifest_sha256'], expected_count=count)
    with (evidence_for(instance)/'validation_scores.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    paths = [r['image_path'] for r in rows]
    assert len(rows) == len(set(paths)) == count
    assert set(paths) == expected_paths
    assert all(r['label'] == '0' for r in rows)
    assert all(np.isfinite(float(r['score'])) for r in rows)
    assert all('/train/good/' in path for path in paths)  # Physical train != logical Train.
    with (ROOT/item['manifest']).open(encoding='utf-8', newline='') as stream:
        forbidden_paths = {r['image_path'] for r in csv.DictReader(stream) if r['split'] != 'val'}
    assert not set(paths) & forbidden_paths
    assert report['protocol_sha256'] == catalog['protocol_sha256']


def test_availability_links_all_six_to_their_original_sessions():
    availability = read_json(ROOT/'experiments/calibration/availability.json')
    assert availability['status'] == 'complete_6_of_6_calibrations_archived'
    assert availability['calibrated_instances'] == 6
    assert availability['kaggle_script_version_ids'] == [369724364, 369945024]
    assert availability['evidence_provenance'] == [
        PROVENANCE.relative_to(ROOT).as_posix(), FINAL_PROVENANCE.relative_to(ROOT).as_posix()]
    entries = {e['instance']: e for e in availability['instances']}
    assert len(availability['instances']) == len(entries) == 6
    assert set(entries) == set(ALL_EXPECTED)
    catalog = runner.read_catalog()
    items = {i['id']: i for i in catalog['instances']}
    for instance, (anchor, value_hex, count) in ALL_EXPECTED.items():
        entry = entries[instance]
        assert entry['status'] == 'calibrated_archived'
        assert entry['threshold'].hex() == entry['threshold_hex'] == value_hex
        assert entry['real_scores_collected'] == entry['expected_validation_images'] == count
        assert entry['integrity_sha256'] == anchor
        assert entry['checkpoint_sha256'] == items[instance]['checkpoint_sha256']
        assert entry['source_experiment'] == items[instance]['source_experiment']
        assert ROOT/entry['evidence_directory'] == evidence_for(instance)
        assert 'blocker' not in entry
        is_new = instance in FINAL_EXPECTED
        assert entry['kaggle_script_version_id'] == (369945024 if is_new else 369724364)
        assert ROOT/entry['evidence_provenance'] == (FINAL_PROVENANCE if is_new else PROVENANCE)
        if is_new:
            assert entry['checkpoint_local_sha256_verified'] is True
            assert entry['strict_local_restoration_verified'] is False
            assert not (EVIDENCE/instance).exists()
    registry = availability['threshold_registry']
    assert ROOT/registry['directory'] == REGISTRY
    assert registry['integrity_sha256'] == REGISTRY_SHA
    assert registry['protocol_sha256'] == catalog['protocol_sha256']
    assert registry['thresholds'] == 6
    assert registry['exact_roundtrip_verified'] is True
    assert registry['new_inference'] is False


def test_four_real_thresholds_cannot_form_six_instance_registry(tmp_path, no_recalibration):
    four = [load_original(instance)[0] for instance in EXPECTED]
    target = tmp_path/'six-thresholds'
    with pytest.raises(ValueError, match='Exactly six unique instances'):
        calibration.export_threshold_registry(target, four)
    assert not target.exists()


def test_final_two_inventory_and_execution_time_anchors():
    provenance = read_json(FINAL_PROVENANCE)
    names = {'session.json'} | {f'{i}/{f}' for i in FINAL_EXPECTED for f in
                               ('calibration.json', 'integrity.json', 'validation_scores.csv')}
    actual = {p.relative_to(FINAL_EVIDENCE).as_posix() for p in FINAL_EVIDENCE.rglob('*') if p.is_file()}
    assert actual == names == set(provenance['files_sha256'])
    assert len(actual) == 7
    assert provenance['archive']['sha256'] == FINAL_ARCHIVE_SHA
    assert provenance['archive']['size_bytes'] == 5092
    assert provenance['kaggle']['script_version_id'] == 369945024
    assert provenance['kaggle']['notebook_url'] == (
        'https://www.kaggle.com/code/marwanedrissi/'
        'industrialdefect-ai-validation-calibration-final-t?scriptVersionId=369945024')
    assert provenance['scientific_code_commit'] == CODE_COMMIT
    assert provenance['bundle_integrity_sha256'] == {i: e[0] for i, e in FINAL_EXPECTED.items()}
    for name, digest in provenance['files_sha256'].items():
        assert calibration.file_sha256(FINAL_EVIDENCE/name) == digest
    assert calibration.file_sha256(FINAL_EVIDENCE/'session.json') == provenance['session_sha256'] == FINAL_SESSION_SHA
    session = read_json(FINAL_EVIDENCE/'session.json')
    assert session['code_commit'] == CODE_COMMIT
    assert session['executed_at_utc'] == provenance['executed_at_utc']
    assert session['scope'] == 'final_two_only'
    results = {r['instance']: r for r in session['results']}
    assert len(session['results']) == len(results) == 2
    assert set(results) == set(FINAL_EXPECTED)
    for instance, (anchor, value_hex, count) in FINAL_EXPECTED.items():
        result = results[instance]
        assert result['status'] == 'calibrated'
        assert result['integrity_sha256'] == anchor
        assert result['threshold'].hex() == result['threshold_hex'] == value_hex
        assert result['validation_images'] == count
        assert calibration.file_sha256(FINAL_EVIDENCE/instance/'integrity.json') == anchor
        report = read_json(FINAL_EVIDENCE/instance/'calibration.json')
        prov = report['provenance']
        assert prov['device'] == 'cuda:0'
        assert prov['torch_version'] == '2.11.0+cu130'
        assert prov['dependencies']['torchvision'] == '0.26.0+cu130'
        if instance == 'patchcore-bottle':
            for package, version in {'anomalib': '2.7.0', 'timm': '1.0.30', 'lightning': '2.6.6'}.items():
                assert prov['dependencies'][package] == version


@pytest.mark.parametrize('instance', ALL_EXPECTED)
def test_saved_score_distribution_matches_frozen_linear_quantile(instance, no_inference_or_refitting):
    # Arithmetic audit only: no new inference, calibration call or export.
    report = read_json(evidence_for(instance)/'calibration.json')
    with (evidence_for(instance)/'validation_scores.csv').open(encoding='utf-8', newline='') as stream:
        scores = np.array([float(r['score']) for r in csv.DictReader(stream)], dtype=np.float64)
    assert len(scores) == ALL_EXPECTED[instance][2]
    assert np.isfinite(scores).all()
    assert report['policy'] == {
        'quantile': .95, 'method': 'linear', 'decision_rule': 'score > threshold',
        'dtype': 'float64', 'split': 'val', 'label': 0}
    distribution = report['distribution']
    assert set(distribution['quantiles']) == {'0.0', '0.05', '0.25', '0.5', '0.75', '0.95', '1.0'}
    for q, stored in distribution['quantiles'].items():
        assert float(np.quantile(scores, float(q), method='linear')).hex() == stored.hex()
    assert distribution['quantiles']['0.95'].hex() == report['threshold_hex'] == ALL_EXPECTED[instance][1]
    assert distribution['std_ddof'] == 0
    for name, computed in {'minimum': scores.min(), 'maximum': scores.max(), 'mean': scores.mean(),
                           'median': np.median(scores), 'std': scores.std(ddof=0)}.items():
        assert distribution[name] == pytest.approx(computed, rel=1e-15, abs=0)


def test_six_real_threshold_registry_exact_roundtrip(tmp_path, no_recalibration):
    catalog = runner.read_catalog()
    six = [load_original(instance)[0] for instance in ALL_EXPECTED]
    expected = {(t.model_family, t.category): t for t in six}
    assert {p.name for p in REGISTRY.iterdir()} == {'thresholds.json', 'integrity.json'}
    assert calibration.file_sha256(REGISTRY/'integrity.json') == REGISTRY_SHA
    loaded = calibration.load_threshold_registry(
        REGISTRY, expected_integrity_sha256=REGISTRY_SHA,
        expected_protocol_sha256=catalog['protocol_sha256'])
    assert loaded == expected
    target = tmp_path/'six-thresholds'
    assert calibration.export_threshold_registry(target, six) == REGISTRY_SHA
    for name in ('thresholds.json', 'integrity.json'):
        assert (target/name).read_bytes() == (REGISTRY/name).read_bytes()
    roundtrip = calibration.load_threshold_registry(
        target, expected_integrity_sha256=REGISTRY_SHA,
        expected_protocol_sha256=catalog['protocol_sha256'])
    assert roundtrip == loaded
    for key, t in expected.items():
        assert roundtrip[key].threshold.hex() == t.threshold.hex()
        np.testing.assert_array_equal(
            roundtrip[key].predict([t.threshold, np.nextafter(t.threshold, np.inf)]), [0, 1])
    with pytest.raises(FileExistsError):
        calibration.export_threshold_registry(target, six)


@pytest.mark.parametrize('filename', ['thresholds.json', 'integrity.json'])
def test_real_registry_rejects_modified_bytes(tmp_path, no_recalibration, filename):
    for source in REGISTRY.iterdir():
        (tmp_path/source.name).write_bytes(source.read_bytes())
    with (tmp_path/filename).open('ab') as stream:
        stream.write(b'\n')  # Even a semantically equivalent edit breaks the pinned hash.
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        calibration.load_threshold_registry(
            tmp_path, expected_integrity_sha256=REGISTRY_SHA,
            expected_protocol_sha256=runner.read_catalog()['protocol_sha256'])
