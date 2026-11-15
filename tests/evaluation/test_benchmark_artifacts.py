"""Read-only regression checks for all six original Kaggle Test exports.

No evaluation/inference/calibration module is imported. Set US0503_ARCHIVE to
the version-370020172 ZIP and US0503_ARCHIVE_002 to the version-370053025 ZIP.
Spatial checks skip explicitly when their external archive is unavailable.
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
CODE_COMMIT = '0acbf26880b755373f2e61236d009ed16f04658c'
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
    'cae-bottle': {
        'count': 83, 'normal': 20, 'anomalous': 63,
        'manifest': '9393551e7c116dabcdaa539b669bb15debbd2e05529b1169bedf92d3c9b104ef',
        'threshold_hex': '0x1.4059f51999997p-6',
        'image_auroc': 0.8611111111111112, 'pixel_auroc': 0.8143878685772936,
        'precision': 1.0, 'recall': 0.5873015873015873, 'f1': 0.74,
        'tp': 37, 'tn': 20, 'fp': 0, 'fn': 26, 'mean_ms': 19.601136457828055,
    },
    'cae-hazelnut': {
        'count': 110, 'normal': 40, 'anomalous': 70,
        'manifest': '3cb839efbd128a0881a394f60b277c0662c7b8b8815b67f15d723b1bbca1d5d5',
        'threshold_hex': '0x1.145e0d7fffffcp-7',
        'image_auroc': 0.9757142857142858, 'pixel_auroc': 0.9814179040347814,
        'precision': 1.0, 'recall': 0.7428571428571429, 'f1': 0.8524590163934426,
        'tp': 52, 'tn': 40, 'fp': 0, 'fn': 18, 'mean_ms': 19.590365454546994,
    },
    'patchcore-hazelnut': {
        'count': 110, 'normal': 40, 'anomalous': 70,
        'manifest': '575384adb3edb7b8bd40156c951625ed0c67c9a0474e2bc91367b6b3637af56a',
        'threshold_hex': '0x1.86b2ee0000000p+5',
        'image_auroc': 1.0, 'pixel_auroc': 0.9864602602530212,
        'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
        'tp': 70, 'tn': 40, 'fp': 0, 'fn': 0, 'mean_ms': 28.647118600001292,
    },
    'patchcore-transistor': {
        'count': 100, 'normal': 60, 'anomalous': 40,
        'manifest': '6d97d78c4bb7cf25d706c67f041782611f41a18bdf47e3fb957ef08340abd565',
        'threshold_hex': '0x1.47ab1f6666666p+5',
        'image_auroc': 0.9908333333333335, 'pixel_auroc': 0.958983424799342,
        'precision': 0.8666666666666667, 'recall': 0.975, 'f1': 0.9176470588235294,
        'tp': 39, 'tn': 54, 'fp': 6, 'fn': 1, 'mean_ms': 28.111121619990627,
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


BATCHES = {
    'kaggle-001': {
        'instances': ['cae-transistor', 'patchcore-bottle'], 'status': 'partial_2_of_6',
        'version': 370020172, 'prefix': 'benchmark-us0503/', 'env': 'US0503_ARCHIVE',
        'sha': '3f847e7bcbda2816ebb5ef4f41cd52a6720bd758b877a4a8f6f753d9ff7fcd45',
        'size': 93019917, 'members': 439, 'selected': 189,
    },
    'kaggle-002': {
        'instances': ['cae-bottle', 'cae-hazelnut', 'patchcore-hazelnut', 'patchcore-transistor'],
        'status': 'remaining_4_of_6', 'version': 370053025, 'prefix': 'benchmark-us0503-remaining/',
        'env': 'US0503_ARCHIVE_002',
        'sha': '85d9ff82c22126c181431dda845efeea56578ca66809d53e60b5ed61da4fe879',
        'size': 153887769, 'members': 665, 'selected': 415,
    },
}


def batch_for(instance):
    return next(b for b, spec in BATCHES.items() if instance in spec['instances'])


def evidence_for(instance):
    return ROOT / 'experiments/benchmark' / batch_for(instance) / instance


def provenance_for(instance):
    return ROOT / 'experiments/benchmark' / f'{batch_for(instance)}-provenance.json'


@pytest.mark.parametrize('batch', BATCHES)
def test_exact_batch_inventory_and_storage_policy(batch):
    spec = BATCHES[batch]
    evidence = ROOT / 'experiments/benchmark' / batch
    provenance = read_json(evidence.with_name(f'{batch}-provenance.json'))
    names = {f'{instance}/{name}' for instance in spec['instances']
             for name in ('predictions.csv', 'metrics.json', 'manifest.json')}
    actual = {p.relative_to(evidence).as_posix() for p in evidence.rglob('*') if p.is_file()}
    assert names == actual == set(provenance['files'])
    assert provenance['status'] == spec['status']  # Batch 001 remains historical evidence.
    assert provenance['instances'] == spec['instances']
    assert provenance['kaggle']['script_version_id'] == spec['version']
    assert provenance['scientific_code_commit'] == CODE_COMMIT
    assert provenance['archive']['sha256'] == spec['sha']
    assert provenance['archive']['size_bytes'] == spec['size']
    assert provenance['verification']['npz_counts'] == {i: EXPECTED[i]['count'] for i in spec['instances']}
    assert provenance['verification']['missing_files'] == []
    assert provenance['verification']['new_inference'] is False
    assert provenance['verification']['threshold_recalibration'] is False
    if batch == 'kaggle-001':
        assert provenance['verification']['metrics_recomputed'] is False
    else:
        assert provenance['verification']['stored_metrics_modified'] is False
        assert provenance['verification']['pixel_auroc_recomputed'] is False
    assert provenance['storage']['npz_versioned_in_git'] is False


@pytest.mark.parametrize('instance', EXPECTED)
def test_original_bytes_and_manifest_links(instance):
    folder = evidence_for(instance)
    manifest = read_json(folder / 'manifest.json')
    assert sha((folder / 'manifest.json').read_bytes()) == EXPECTED[instance]['manifest']
    assert set(manifest['files']) == {'predictions.csv', 'metrics.json'}
    for name, digest in manifest['files'].items():
        assert sha((folder / name).read_bytes()) == digest
    provenance = read_json(provenance_for(instance))
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
    report = read_json(evidence_for(instance) / 'metrics.json')
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
    rows = csv_rows((evidence_for(instance) / 'predictions.csv').read_bytes())
    assert [(r['image_path'], r['label']) for r in rows] == [(r['image_path'], r['label']) for r in selected]
    assert not {r['image_path'] for r in rows} & {r['image_path'] for r in csv_rows(manifest_bytes) if r['split'] != 'test'}
    assert all(bool(r['mask_path']) == bool(int(r['label'])) for r in selected)
    assert run['test_paths_sha256'] == sha(('\n'.join(r['image_path'] for r in rows) + '\n').encode())
    assert run['split'] == 'test'
    assert run['git_commit'] == CODE_COMMIT
    assert run['hardware'] == 'Tesla T4' and run['device'] == 'cuda:0'
    assert run['batch_size'] == 1 and run['warmup_runs_per_model'] == 3
    assert run['latency_scope'] == 'model_inference_only' and run['synchronize_cuda'] is True
    assert run['target_size_hw'] == [256, 256] and run['seed'] == 42
    assert run['python_version'] == '3.13.16' and run['torch_version'] == '2.11.0+cu130'
    assert run['dependencies']['torchvision'] == '0.26.0+cu130'
    if instance.startswith('patchcore-'):
        assert {k: run['dependencies'][k] for k in ('anomalib', 'timm', 'lightning')} == {
            'anomalib': '2.7.0', 'timm': '1.0.30', 'lightning': '2.6.6'}


@pytest.mark.parametrize('instance', EXPECTED)
def test_saved_metrics_preserved_without_recomputation(instance):
    report = read_json(evidence_for(instance) / 'metrics.json')
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
    rows = csv_rows((evidence_for(instance) / 'predictions.csv').read_bytes())
    assert sum(r['label'] == '0' for r in rows) == expected['normal']
    assert sum(r['label'] == '1' for r in rows) == expected['anomalous']


@pytest.mark.parametrize('instance', EXPECTED)
def test_saved_scores_fixed_decisions_and_latency_consistency(instance):
    """Independent arithmetic, without inference/calibration/evaluation imports."""
    folder = evidence_for(instance)
    report = read_json(folder / 'metrics.json')
    rows = csv_rows((folder / 'predictions.csv').read_bytes())
    metrics = report['metrics']['image']
    threshold = report['run']['threshold']
    labels = [int(r['label']) for r in rows]
    scores = [float(r['score']) for r in rows]
    decisions = [score > threshold for score in scores]
    tp = sum(y == 1 and p for y, p in zip(labels, decisions))
    tn = sum(y == 0 and not p for y, p in zip(labels, decisions))
    fp = sum(y == 0 and p for y, p in zip(labels, decisions))
    fn = sum(y == 1 and not p for y, p in zip(labels, decisions))
    assert (tp, tn, fp, fn) == tuple(metrics[k] for k in ('tp', 'tn', 'fp', 'fn'))
    # Every archived instance has positives, negatives and positive predictions.
    def close(value):
        return pytest.approx(value, rel=1e-12, abs=1e-12)

    assert metrics['precision'] == close(tp / (tp + fp))
    assert metrics['recall'] == close(tp / (tp + fn))
    assert metrics['f1'] == close(2 * tp / (2 * tp + fp + fn))
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    # AUROC as pairwise ranking, with half credit for ties; no sklearn dependency.
    auc = math.fsum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    assert metrics['image_auroc'] == close(auc / (len(positives) * len(negatives)))
    latency = np.asarray([float(r['latency_ms']) for r in rows])
    expected = {'mean_ms': np.mean(latency), 'median_ms': np.median(latency),
                'p95_ms': np.percentile(latency, 95), 'min_ms': np.min(latency), 'max_ms': np.max(latency)}
    assert report['latency_summary']['unit'] == 'ms'
    for key, value in expected.items():
        assert report['latency_summary'][key] == close(value)


def audit_archive(archive_path, batch):
    """Check original bytes; read NPZ in memory, never extract or execute code."""
    spec = BATCHES[batch]
    assert archive_path.stat().st_size == spec['size']
    assert sha(archive_path.read_bytes()) == spec['sha']
    summary, mask_hashes = {}, {}
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == spec['members']
        assert len({name.casefold() for name in names}) == len(names)
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            assert not path.is_absolute() and '..' not in path.parts
            assert path.as_posix() == entry.filename.rstrip('/')
            assert '\\' not in entry.filename and ':' not in entry.filename
            assert not stat.S_ISLNK(entry.external_attr >> 16)
        assert archive.testzip() is None  # CRC of all members, including excluded files.
        allowed = set()
        for instance in spec['instances']:
            prefix = f"{spec['prefix']}{instance}/"
            folder = evidence_for(instance)
            for name in ('predictions.csv', 'metrics.json', 'manifest.json'):
                allowed.add(prefix + name)
                assert archive.read(prefix + name) == (folder / name).read_bytes()
            manifest = read_json(folder / 'manifest.json')
            assert sha((folder / 'manifest.json').read_bytes()) == EXPECTED[instance]['manifest']
            rows = csv_rows((folder / 'predictions.csv').read_bytes())
            total_bytes = 0
            mask_hashes[instance] = {}
            for index, (artifact, row) in enumerate(zip(manifest['artifacts'], rows)):
                assert artifact['file'] == row['artifact_file'] == f'artifacts/{index:05d}.npz'
                assert artifact['image_path'] == row['image_path']
                name = prefix + artifact['file']
                allowed.add(name)
                data = archive.read(name)
                total_bytes += len(data)
                assert sha(data) == artifact['sha256'] == row['artifact_sha256']
                with np.load(io.BytesIO(data), allow_pickle=False) as spatial:
                    assert len(spatial.files) == 2 and set(spatial.files) == {'mask', 'anomaly_map'}
                    mask, amap = spatial['mask'], spatial['anomaly_map']
                    assert mask.shape == amap.shape == (256, 256)
                    assert mask.dtype == np.uint8
                    assert np.all((mask == 0) | (mask == 1))
                    assert amap.dtype == np.float64 and np.isfinite(amap).all()
                    assert bool(mask.any()) == bool(int(row['label']))
                    mask_hashes[instance][row['image_path']] = sha(mask.tobytes())
                    if instance.startswith('cae-'):
                        # CAE score is the max of the same smoothed map at 256x256.
                        assert float(row['score']) == float(amap.max())
                    # PatchCore native pred_score is not defined as max(resized map).
            assert len(rows) == len(manifest['artifacts']) == EXPECTED[instance]['count']
            summary[instance] = {'count': len(rows), 'size_bytes': total_bytes}
        actual = {n for n in names if n.startswith(spec['prefix']) and not n.endswith('/')}
        assert actual == allowed and len(actual) == spec['selected']
        provenance = read_json(ROOT / 'experiments/benchmark' / f'{batch}-provenance.json')
        assert sum(archive.getinfo(n).file_size for n in actual) == provenance['archive']['selected_size_bytes']
    return summary, mask_hashes


@pytest.fixture(scope='module')
def archived_batches():
    return {batch: audit_archive(Path(os.environ[spec['env']]), batch)
            for batch, spec in BATCHES.items() if os.environ.get(spec['env'])}


@pytest.mark.parametrize('batch', BATCHES)
def test_external_saved_snapshot_npz_bytes_and_contents(batch, archived_batches):
    if batch not in archived_batches:
        pytest.skip(f"External ZIP absent: set {BATCHES[batch]['env']}; NPZ deliberately excluded from Git")
    summary, _ = archived_batches[batch]
    provenance = read_json(ROOT / 'experiments/benchmark' / f'{batch}-provenance.json')
    assert summary == provenance['verification']['npz_inventory']


def test_identical_test_images_labels_and_exported_masks_between_families(archived_batches):
    if len(archived_batches) != 2:
        pytest.skip('Both external ZIPs required for comparison of all 293 masks between families')
    masks = {i: paths for _, batch_masks in archived_batches.values() for i, paths in batch_masks.items()}
    for category in ('bottle', 'hazelnut', 'transistor'):
        cae, patchcore = f'cae-{category}', f'patchcore-{category}'
        assert masks[cae] == masks[patchcore]
        rows = [csv_rows((evidence_for(i) / 'predictions.csv').read_bytes()) for i in (cae, patchcore)]
        assert [(r['image_path'], r['label']) for r in rows[0]] == [(r['image_path'], r['label']) for r in rows[1]]
    assert sum(len(masks[f'cae-{c}']) for c in ('bottle', 'hazelnut', 'transistor')) == 293
