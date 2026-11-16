"""Test-only benchmark of frozen US-05-02 instances; no fitting or model selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data.mvtec_dataset import MVTecDataset
from src.data.transforms import MVTecTransform
from .benchmark import evaluate_predictions
from .calibration import _json, _path, _verified_bytes, load_protocol, load_threshold_registry
from .exporters import EvaluationRecord, file_sha256
from .latency import measure_inference
from .masks import prepare_ground_truth_mask, resize_anomaly_maps
from .run_calibration import CATALOG, ROOT, read_catalog, restore_instance

# External anchors reviewed in US-05-02; never derive trust from the input bundle.
CATALOG_SHA256 = 'b3aa8340bf0e5f07009a48562393aaf708ac49fc8edabd630bd820448a6239dc'
PROTOCOL_SHA256 = '6b66ee7b9f51780ceb1e3e89723bb92f4ffdac8502f73031f45052f9ccd39854'
REGISTRY_SHA256 = '41dd9d26e68840ca5d104062605d967e72a2d3907937e8ed6745743034c96b6a'
REGISTRY = ROOT/'experiments/calibration/six-thresholds'
MANIFEST_COLUMNS = ['category', 'split', 'image_path', 'label', 'defect_type', 'mask_path']


def load_benchmark_inputs(registry_dir=REGISTRY, *, expected_integrity_sha256=REGISTRY_SHA256):
    """Load six thresholds and bind every entry to the frozen scientific catalog."""
    _verified_bytes(CATALOG, CATALOG_SHA256)
    catalog = read_catalog()
    if catalog['protocol_sha256'] != PROTOCOL_SHA256:
        raise ValueError('Unexpected benchmark protocol.')
    protocol = load_protocol(ROOT/catalog['protocol'], PROTOCOL_SHA256)
    thresholds = load_threshold_registry(
        registry_dir, expected_integrity_sha256=expected_integrity_sha256,
        expected_protocol_sha256=PROTOCOL_SHA256)
    for item in catalog['instances']:
        key = (item['model_family'], item['category'])
        threshold = thresholds[key]
        if ((threshold.model_family, threshold.category) != key
                or threshold.checkpoint_sha256 != item['checkpoint_sha256']
                or threshold.manifest_sha256 != item['manifest_sha256']
                or threshold.protocol_sha256 != PROTOCOL_SHA256):
            raise ValueError(f"Threshold/catalog association mismatch: {item['id']}")
    return catalog, protocol, thresholds


def _test_rows(item):
    raw = _verified_bytes(ROOT/item['manifest'], item['manifest_sha256'])
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''))
    if reader.fieldnames != MANIFEST_COLUMNS:
        raise ValueError('Invalid Test manifest schema.')
    seen, selected = set(), []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('Malformed manifest row.')
        path = _path(row['image_path'])
        if row['category'] != item['category'] or path.split('/')[0] != item['category']:
            raise ValueError('Manifest category mismatch.')
        if path in seen:
            raise ValueError('Duplicate manifest image across splits.')
        seen.add(path)
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown logical split.')
        if row['split'] != 'test':
            continue
        if row['label'] not in ('0', '1'):
            raise ValueError('Invalid Test label.')
        if row['label'] == '0':
            if row['mask_path'] or row['defect_type'] != 'good':
                raise ValueError('Invalid normal Test sample.')
        else:
            if not row['mask_path']:
                raise ValueError('Anomalous Test image requires a mask.')
            mask_path = _path(row['mask_path'])
            if mask_path.split('/')[0] != item['category'] or row['defect_type'] in ('', 'good'):
                raise ValueError('Invalid anomalous Test sample.')
        selected.append({**row, 'label': int(row['label'])})
    if not selected:
        raise ValueError('Empty Test split.')
    return sorted(selected, key=lambda row: row['image_path'])


def _test_dataset(item, dataset_root):
    rows = _test_rows(item)
    root = Path(dataset_root).resolve()
    for row in rows:
        for relative in (row['image_path'], row['mask_path']):
            if relative:
                resolved = (root/relative).resolve()
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    raise FileNotFoundError(f'Missing/unsafe Test image or mask: {relative}')
    # The shared loader receives Test only. Physical directories never define the split.
    with tempfile.TemporaryDirectory(prefix='benchmark-test-manifest-') as temporary:
        projection = Path(temporary)/'test.csv'
        with projection.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        dataset = MVTecDataset(root, projection, item['category'], 'test',
                               MVTecTransform(size=(256, 256), profile=item['model_family']))
    if dataset.samples != rows or len(dataset) != len(rows):
        raise ValueError('Dataset differs from exact logical Test coverage.')
    return dataset, rows


def _synchronizer(device):
    device = torch.device(device)
    return (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else None


@torch.inference_mode()
def _predict_on_device(engine, family, batch):
    """Same raw score/maps as the existing APIs, without host/device copies."""
    if family == 'cae':
        reconstruction = engine.model(batch)
        maps = engine._smooth(((batch - reconstruction) ** 2).mean(dim=1))
        return maps.flatten(1).max(dim=1).values, maps
    prediction = engine.model(batch)
    maps = prediction.anomaly_map
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    return prediction.pred_score.reshape(-1), maps


def collect_test_records(dataset, expected_rows, engine, item):
    """Batch 1; three warmups; all input/output transfers outside the timed call."""
    if len(dataset) != len(expected_rows) or not expected_rows:
        raise ValueError('Dataset differs from exact logical Test coverage.')
    if engine.category != item['category'] or engine.model.training:
        raise ValueError('Restored engine identity/evaluation mode mismatch.')
    if item['model_family'] == 'cae' and (
            engine.config.score_method != 'max' or engine.config.smoothing_sigma != 4.0):
        raise ValueError('CAE score differs from the frozen calibration.')
    synchronize = _synchronizer(engine.device)
    records = []
    for index, expected in enumerate(expected_rows):
        sample = dataset[index]
        for key in ('image_path', 'category', 'label', 'defect_type'):
            if sample[key] != expected[key]:
                raise ValueError('Sample differs from the frozen Test manifest.')
        mask_value = sample.get('mask')
        if isinstance(mask_value, torch.Tensor):
            mask_value = mask_value.detach().cpu().numpy()
        mask = prepare_ground_truth_mask(mask_value, sample['label'], (256, 256))
        image = sample['image']
        if (not isinstance(image, torch.Tensor) or tuple(image.shape) != (3, 256, 256)
                or not image.is_floating_point() or not torch.isfinite(image).all()):
            raise ValueError('Expected a finite preprocessed Test image [3,256,256].')
        if item['model_family'] == 'cae' and (image.min() < 0 or image.max() > 1):
            raise ValueError('CAE input must remain in [0,1].')
        batch = image.unsqueeze(0).to(engine.device, dtype=torch.float32)
        predict = lambda: _predict_on_device(engine, item['model_family'], batch)
        if index == 0:
            for _ in range(3):
                predict()
                if synchronize is not None:
                    synchronize()
        (scores, maps), latency_ms = measure_inference(predict, synchronize=synchronize)
        if (not isinstance(scores, torch.Tensor) or not isinstance(maps, torch.Tensor)
                or tuple(scores.shape) != (1,) or maps.ndim != 3 or maps.shape[0] != 1):
            raise ValueError('Unexpected batch-1 inference output shapes.')
        # Host copies, scalar conversion and spatial alignment are outside the timer.
        score = float(scores[0].detach().cpu())
        anomaly_map = resize_anomaly_maps(maps.detach().cpu().numpy(), (256, 256))[0]
        if not np.isfinite(score):
            raise ValueError('Nonfinite Test score.')
        records.append(EvaluationRecord(sample['image_path'], sample['label'], score,
                                        mask, anomaly_map, latency_ms))
    if [r.image_path for r in records] != [r['image_path'] for r in expected_rows]:
        raise ValueError('Incomplete or unexpected Test predictions.')
    return records


def _scientific_commit():
    status = subprocess.check_output(
        ['git', 'status', '--porcelain', '--', 'src', 'experiments/configs'], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError('Commit scientific code/configuration before real benchmarking.')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def _prepare(instance, checkpoint, dataset_root, output_dir, inputs):
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    catalog, _, thresholds = inputs
    item = next((i for i in catalog['instances'] if i['id'] == instance), None)
    if item is None:
        raise ValueError(f'Unknown benchmark instance: {instance}')
    if file_sha256(checkpoint) != item['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA-256 mismatch.')
    dataset, rows = _test_dataset(item, dataset_root)
    return item, thresholds[(item['model_family'], item['category'])], dataset, rows


def _execute(prepared, *, checkpoint, output_dir, device, registry_anchor, preflight):
    item, threshold, dataset, rows = prepared
    result = {'instance': item['id'], 'test_images': len(rows),
              'threshold': threshold.threshold, 'threshold_hex': threshold.threshold.hex(),
              'checkpoint_sha256': threshold.checkpoint_sha256}
    if preflight:
        return {**result, 'status': 'inputs_verified'}
    commit = _scientific_commit()
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable.')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    engine = restore_instance(item, checkpoint, device)
    records = collect_test_records(dataset, rows, engine, item)
    packages = ['torchvision', 'Pillow']
    if item['model_family'] == 'patchcore':
        packages += ['anomalib', 'timm', 'lightning']
    config = {
        'run_id': Path(output_dir).name, 'model_family': item['model_family'],
        'category': item['category'], 'source_experiment': item['source_experiment'], 'split': 'test',
        'git_commit': commit, 'checkpoint_sha256': threshold.checkpoint_sha256,
        'dataset_manifest_sha256': threshold.manifest_sha256, 'protocol_sha256': PROTOCOL_SHA256,
        'catalog_sha256': CATALOG_SHA256, 'threshold_registry_integrity_sha256': registry_anchor,
        'calibration_integrity_sha256': threshold.calibration_integrity_sha256,
        'model_config_canonical_sha256': item['model_config_canonical_sha256'],
        'source_registry_canonical_sha256': item['source_registry_canonical_sha256'],
        'threshold': threshold.threshold, 'threshold_hex': threshold.threshold.hex(),
        'score_definition': item['score_definition'], 'preprocessing': item['preprocessing'],
        'test_paths_sha256': hashlib.sha256(
            ('\n'.join(row['image_path'] for row in rows)+'\n').encode('utf-8')).hexdigest(),
        'device': str(engine.device),
        'hardware': torch.cuda.get_device_name(engine.device) if engine.device.type == 'cuda'
                    else platform.processor() or platform.machine(),
        'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
        'numpy_version': np.__version__,
        'dependencies': {name: importlib.metadata.version(name) for name in packages},
        'seed': 42, 'batch_size': 1, 'warmup_runs_per_model': 3,
        'latency_scope': 'model_inference_only',
        'latency_detail': 'Device forward and raw score/map calculation; excludes IO, preprocessing, input/output transfers and evaluation resizing.',
        'synchronize_cuda': engine.device.type == 'cuda', 'target_size_hw': [256, 256],
    }
    report = evaluate_predictions(records, threshold=threshold.threshold,
                                   output_dir=output_dir, run_config=config)
    return {**result, 'status': 'evaluated', 'metrics': report,
            'export_manifest_sha256': file_sha256(Path(output_dir)/'manifest.json')}


def run_instance(instance, *, dataset_root, checkpoint, output_dir, device='cpu',
                 registry_dir=REGISTRY, expected_integrity_sha256=REGISTRY_SHA256, preflight=False):
    inputs = load_benchmark_inputs(registry_dir, expected_integrity_sha256=expected_integrity_sha256)
    prepared = _prepare(instance, checkpoint, dataset_root, output_dir, inputs)
    return _execute(prepared, checkpoint=checkpoint, output_dir=output_dir, device=device,
                    registry_anchor=expected_integrity_sha256, preflight=preflight)


def run_all(*, dataset_root, checkpoints, output_dir, device='cpu', registry_dir=REGISTRY,
            expected_integrity_sha256=REGISTRY_SHA256, preflight=False):
    """Preflight all six before inference; publish a separate immutable run per instance."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(root)
    inputs = load_benchmark_inputs(registry_dir, expected_integrity_sha256=expected_integrity_sha256)
    ids = [i['id'] for i in inputs[0]['instances']]
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(ids):
        raise ValueError('Exactly six checkpoint paths keyed by catalog instance are required.')
    prepared = [_prepare(i, checkpoints[i], dataset_root, root/i, inputs) for i in ids]
    return [_execute(p, checkpoint=checkpoints[i], output_dir=root/i, device=device,
                     registry_anchor=expected_integrity_sha256, preflight=preflight)
            for i, p in zip(ids, prepared)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True, help='Catalog instance id, or all')
    parser.add_argument('--dataset-root', type=Path, required=True)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--checkpoint', type=Path)
    weights.add_argument('--checkpoints-json', type=Path, help='Object mapping six instance ids to paths')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--registry-integrity-sha256', default=REGISTRY_SHA256)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    options = dict(dataset_root=args.dataset_root, output_dir=args.output, device=args.device,
                   registry_dir=args.registry, expected_integrity_sha256=args.registry_integrity_sha256,
                   preflight=args.preflight)
    if args.instance == 'all':
        if args.checkpoints_json is None:
            parser.error('--instance all requires --checkpoints-json')
        result = run_all(checkpoints=_json(args.checkpoints_json.read_bytes()), **options)
    else:
        if args.checkpoint is None:
            parser.error('A single instance requires --checkpoint')
        result = run_instance(args.instance, checkpoint=args.checkpoint, **options)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
