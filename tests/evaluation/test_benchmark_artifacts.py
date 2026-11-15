"""Read-only regression checks for the two original Kaggle Test exports.

No evaluation/inference/calibration module is imported. Set US0503_ARCHIVE to
the downloaded version-370020172 ZIP to repeat the spatial-artifact audit.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from zipfile import ZipFile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'experiments/benchmark/kaggle-001'
PROVENANCE = ROOT / 'experiments/benchmark/kaggle-001-provenance.json'
CODE_COMMIT = '0acbf26880b755373f2e61236d009ed16f04658c'
ARCHIVE_SHA = '3f847e7bcbda2816ebb5ef4f41cd52a6720bd758b877a4a8f6f753d9ff7fcd45'
REGISTRY_SHA = '41dd9d26e68840ca5d104062605d967e72a2d3907937e8ed6745743034c96b6a'
EXPECTED = {
    'cae-transistor': {
        'count': 100, 'normal': 60, 'anomalous': 40,
        'manifest': 'b46e66ff9f11f6af17ae0d453490217bca537e7a1f020348d2fa3a0060f89e19',
        'threshold_hex': '0x1.c0581e0000000p-6',
        'image_auroc': 0.5858333333333333, 'pixel_auroc': 0.6360072524763124,
        'precision': 1.0, 'recall': 0.15, 'f1': 0.2608695652173913,
        'tp': 6, 'tn': 60, 'fp': 0, 'fn': 34, 'mean_ms': 19.517059250003967,
    },
    'patchcore-bottle': {
        'count': 83, 'normal': 20, 'anomalous': 63,
        'manifest': '009e3c2153d6c5f8a98ede447e0bb215d4cbf4820ff6ac1e377f63df59dee56e',
        'threshold_hex': '0x1.f9ebb36666666p+4',
        'image_auroc': 1.0, 'pixel_auroc': 0.984774441492949,
        'precision': 0.984375, 'recall': 1.0, 'f1': 0.9921259842519685,
        'tp': 63, 'tn': 19, 'fp': 1, 'fn': 0, 'mean_ms': 27.72071383132851,
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8'), newline='')))


def canonical_sha(path):
    data = (json.dumps(read_json(path), sort_keys=True, indent=2,
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    return sha(data)


def test_exact_partial_inventory_and_storage_policy():
    provenance = read_json(PROVENANCE)
    names = {f'{instance}/{name}' for instance in EXPECTED
             for name in ('predictions.csv', 'metrics.json', 'manifest.json')}
    actual = {p.relative_to(EVIDENCE).as_posix() for p in EVIDENCE.rglob('*') if p.is_file()}
    assert names == actual == set(provenance['files'])
    assert provenance['status'] == 'partial_2_of_6'
    assert provenance['instances'] == list(EXPECTED)
    assert provenance['kaggle']['script_version_id'] == 370020172
    assert provenance['scientific_code_commit'] == CODE_COMMIT
    assert provenance['archive']['sha256'] == ARCHIVE_SHA
    assert provenance['archive']['size_bytes'] == 93019917
    assert provenance['verification']['npz_counts'] == {i: e['count'] for i, e in EXPECTED.items()}
    assert provenance['verification']['missing_files'] == []
    assert provenance['verification']['new_inference'] is False
    assert provenance['verification']['threshold_recalibration'] is False
    assert provenance['verification']['metrics_recomputed'] is False
    assert provenance['storage']['npz_versioned_in_git'] is False


@pytest.mark.parametrize('instance', EXPECTED)
def test_original_bytes_and_manifest_links(instance):
    folder = EVIDENCE / instance
    manifest = read_json(folder / 'manifest.json')
    assert sha((folder / 'manifest.json').read_bytes()) == EXPECTED[instance]['manifest']
    assert set(manifest['files']) == {'predictions.csv', 'metrics.json'}
    for name, digest in manifest['files'].items():
        assert sha((folder / name).read_bytes()) == digest
    provenance = read_json(PROVENANCE)
    for name in ('predictions.csv', 'metrics.json', 'manifest.json'):
        data = (folder / name).read_bytes()
        assert provenance['files'][f'{instance}/{name}'] == {
            'size_bytes': len(data), 'sha256': sha(data)}
    rows = csv_rows((folder / 'predictions.csv').read_bytes())
    assert manifest['run_id'] == instance
    assert manifest['n_images'] == len(rows) == len(manifest['artifacts']) == EXPECTED[instance]['count']
    assert len({r['image_path'] for r in rows}) == len(rows)
    for index, (row, artifact) in enumerate(zip(rows, manifest['artifacts'])):
        assert int(row['index']) == index
        assert row['artifact_file'] == artifact['file'] == f'artifacts/{index:05d}.npz'
        assert row['image_path'] == artifact['image_path']
        assert row['artifact_sha256'] == artifact['sha256']
        assert len(artifact['sha256']) == 64
        int(artifact['sha256'], 16)
        assert math.isfinite(float(row['score']))
        assert math.isfinite(float(row['latency_ms'])) and float(row['latency_ms']) > 0


@pytest.mark.parametrize('instance', EXPECTED)
def test_provenance_frozen_threshold_and_exact_test_coverage(instance):
    report = read_json(EVIDENCE / instance / 'metrics.json')
    run = report['run']
    catalog_path = ROOT / 'experiments/configs/calibration_instances.json'
    catalog = read_json(catalog_path)
    item = next(i for i in catalog['instances'] if i['id'] == instance)
    registry_dir = ROOT / 'experiments/calibration/six-thresholds'
    assert sha((registry_dir / 'integrity.json').read_bytes()) == REGISTRY_SHA
    registry = read_json(registry_dir / 'thresholds.json')
    assert sha((registry_dir / 'thresholds.json').read_bytes()) == read_json(registry_dir / 'integrity.json')['files']['thresholds.json']
    frozen = next(t for t in registry['thresholds']
                  if f"{t['model_family']}-{t['category']}" == instance)
    assert registry['decision_rule'] == 'score > threshold'
    assert run['threshold'].hex() == frozen['threshold'].hex() == run['threshold_hex'] == EXPECTED[instance]['threshold_hex']
    assert run['threshold_registry_integrity_sha256'] == REGISTRY_SHA
    assert run['calibration_integrity_sha256'] == frozen['calibration_integrity_sha256']
    for key in ('model_family', 'category', 'checkpoint_sha256', 'source_experiment',
                'score_definition', 'preprocessing', 'model_config_canonical_sha256',
                'source_registry_canonical_sha256'):
        assert run[key] == item[key]
    assert canonical_sha(ROOT / item['model_config']) == run['model_config_canonical_sha256']
    assert canonical_sha(ROOT / item['source_registry']) == run['source_registry_canonical_sha256']
    assert run['checkpoint_sha256'] == frozen['checkpoint_sha256']
    assert run['catalog_sha256'] == sha(catalog_path.read_bytes())
    assert run['protocol_sha256'] == catalog['protocol_sha256'] == sha((ROOT / catalog['protocol']).read_bytes())
    manifest_bytes = (ROOT / item['manifest']).read_bytes()
    assert run['dataset_manifest_sha256'] == item['manifest_sha256'] == sha(manifest_bytes)
    selected = [r for r in csv_rows(manifest_bytes) if r['split'] == 'test']
    rows = csv_rows((EVIDENCE / instance / 'predictions.csv').read_bytes())
    assert {r['image_path']: r['label'] for r in rows} == {r['image_path']: r['label'] for r in selected}
    assert run['test_paths_sha256'] == sha(('\n'.join(r['image_path'] for r in rows) + '\n').encode())
    assert run['split'] == 'test'
    assert run['git_commit'] == CODE_COMMIT
    assert run['hardware'] == 'Tesla T4' and run['device'] == 'cuda:0'
    assert run['batch_size'] == 1 and run['warmup_runs_per_model'] == 3
    assert run['latency_scope'] == 'model_inference_only' and run['synchronize_cuda'] is True
    assert run['target_size_hw'] == [256, 256] and run['seed'] == 42
    assert run['python_version'] == '3.13.16' and run['torch_version'] == '2.11.0+cu130'
    assert run['dependencies']['torchvision'] == '0.26.0+cu130'
    if instance == 'patchcore-bottle':
        assert {k: run['dependencies'][k] for k in ('anomalib', 'timm', 'lightning')} == {
            'anomalib': '2.7.0', 'timm': '1.0.30', 'lightning': '2.6.6'}


@pytest.mark.parametrize('instance', EXPECTED)
def test_saved_metrics_preserved_without_recomputation(instance):
    report = read_json(EVIDENCE / instance / 'metrics.json')
    expected = EXPECTED[instance]
    metrics = report['metrics']
    assert report['n_images'] == metrics['n_images'] == metrics['image']['n_images'] == expected['count']
    assert metrics['image']['n_normal'] == expected['normal']
    assert metrics['image']['n_anomalous'] == expected['anomalous']
    for key in ('image_auroc', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn'):
        assert metrics['image'][key] == expected[key]
    assert metrics['image']['threshold'].hex() == expected['threshold_hex']
    assert metrics['pixel']['pixel_auroc'] == expected['pixel_auroc']
    assert metrics['latency'] == report['latency_summary']
    assert metrics['latency']['mean_ms'] == expected['mean_ms']
    rows = csv_rows((EVIDENCE / instance / 'predictions.csv').read_bytes())
    assert sum(r['label'] == '0' for r in rows) == expected['normal']
    assert sum(r['label'] == '1' for r in rows) == expected['anomalous']


def audit_archive(archive_path):
    """Validate saved ZIP bytes and NPZ contents without extracting/executing code."""
    assert sha(archive_path.read_bytes()) == ARCHIVE_SHA
    summary = {}
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            assert not path.is_absolute() and '..' not in path.parts
            assert '\\' not in entry.filename and ':' not in entry.filename
            assert not stat.S_ISLNK(entry.external_attr >> 16)
        allowed = set()
        for instance, expected in EXPECTED.items():
            prefix = f'benchmark-us0503/{instance}/'
            folder = EVIDENCE / instance
            for name in ('predictions.csv', 'metrics.json', 'manifest.json'):
                allowed.add(prefix + name)
                assert archive.read(prefix + name) == (folder / name).read_bytes()
            manifest = read_json(folder / 'manifest.json')
            rows = csv_rows((folder / 'predictions.csv').read_bytes())
            total_bytes = 0
            for index, (artifact, row) in enumerate(zip(manifest['artifacts'], rows)):
                assert artifact['file'] == f'artifacts/{index:05d}.npz'
                name = prefix + artifact['file']
                allowed.add(name)
                data = archive.read(name)  # ZIP CRC checked by read.
                total_bytes += len(data)
                assert sha(data) == artifact['sha256'] == row['artifact_sha256']
                with np.load(io.BytesIO(data), allow_pickle=False) as spatial:
                    assert len(spatial.files) == 2 and set(spatial.files) == {'mask', 'anomaly_map'}
                    mask, amap = spatial['mask'], spatial['anomaly_map']
                    assert mask.shape == amap.shape == (256, 256)
                    assert mask.dtype == np.uint8
                    assert np.all((mask == 0) | (mask == 1))
                    assert np.issubdtype(amap.dtype, np.floating) and np.isfinite(amap).all()
                    assert bool(mask.any()) == bool(int(row['label']))
            assert len(rows) == len(manifest['artifacts']) == expected['count']
            summary[instance] = {'count': len(rows), 'size_bytes': total_bytes}
        actual = {n for n in names if n.startswith('benchmark-us0503/') and not n.endswith('/')}
        assert actual == allowed and len(actual) == 189
    return summary


@pytest.mark.skipif(not os.environ.get('US0503_ARCHIVE'), reason='External Kaggle ZIP not supplied; NPZ are deliberately not in Git')
def test_external_saved_snapshot_npz_bytes_and_contents():
    summary = audit_archive(Path(os.environ['US0503_ARCHIVE']))
    assert summary == read_json(PROVENANCE)['verification']['npz_inventory']
