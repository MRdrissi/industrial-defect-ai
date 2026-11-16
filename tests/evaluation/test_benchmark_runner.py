"""Phase 1 only: generated images, fake checkpoints and CPU mock models."""
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from src.evaluation import calibration
from src.evaluation import run_calibration
from src.evaluation import run_benchmark as runner


@pytest.fixture(autouse=True)
def forbid_calibration(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Test benchmarking must never fit thresholds or run calibration')
    monkeypatch.setattr(calibration, 'calibrate', forbidden)
    monkeypatch.setattr(calibration, 'export_calibration', forbidden)
    monkeypatch.setattr(run_calibration, 'run_instance', forbidden)
    monkeypatch.setattr(run_calibration, 'collect_validation_scores', forbidden)
    monkeypatch.setattr(run_calibration, 'export_calibration', forbidden)


def write_manifest(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.MANIFEST_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


class Reconstruction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return batch * 0.5


@pytest.fixture
def synthetic_case(tmp_path, monkeypatch):
    catalog, protocol, thresholds = runner.load_benchmark_inputs()
    catalog = copy.deepcopy(catalog)
    item = catalog['instances'][0]
    rows = [
        dict(category='bottle', split='train', image_path='bottle/train/good/001.png', label='0', defect_type='good', mask_path=''),
        dict(category='bottle', split='val', image_path='bottle/train/good/002.png', label='0', defect_type='good', mask_path=''),
        # Deliberately not a physical test directory: logical split is authoritative.
        dict(category='bottle', split='test', image_path='bottle/train/good/900.png', label='0', defect_type='good', mask_path=''),
        dict(category='bottle', split='test', image_path='bottle/test/crack/900.png', label='1', defect_type='crack', mask_path='bottle/ground_truth/crack/900_mask.png'),
    ]
    manifest = tmp_path/'manifest.csv'
    write_manifest(manifest, rows)
    item['manifest'] = str(manifest)
    item['manifest_sha256'] = runner.file_sha256(manifest)
    checkpoint = tmp_path/'synthetic.pt'
    checkpoint.write_bytes(b'synthetic checkpoint; never deserialized')
    item['checkpoint_sha256'] = runner.file_sha256(checkpoint)
    key = ('cae', 'bottle')
    thresholds[key] = replace(thresholds[key], checkpoint_sha256=item['checkpoint_sha256'],
                              manifest_sha256=item['manifest_sha256'])
    dataset_root = tmp_path/'data'
    for row in rows[2:]:
        path = dataset_root/row['image_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), color=(64, 80, 96)).save(path)
        if row['mask_path']:
            path = dataset_root/row['mask_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:4, :4] = 255
            Image.fromarray(mask).save(path)
    model = Reconstruction().eval()
    engine = SimpleNamespace(model=model, device=torch.device('cpu'), category='bottle',
                             config=SimpleNamespace(score_method='max', smoothing_sigma=4.0),
                             _smooth=lambda maps: maps)
    restore = Mock(return_value=engine)
    monkeypatch.setattr(runner, 'read_catalog', lambda: catalog)
    monkeypatch.setattr(runner, 'load_threshold_registry', lambda *a, **kw: thresholds)
    monkeypatch.setattr(runner, 'restore_instance', restore)
    monkeypatch.setattr(runner, '_scientific_commit', lambda: 'a'*40)
    options = dict(dataset_root=dataset_root, checkpoint=checkpoint, output_dir=tmp_path/'output')
    return SimpleNamespace(item=item, rows=rows, manifest=manifest, thresholds=thresholds,
                           catalog=catalog, protocol=protocol, engine=engine, restore=restore,
                           options=options, root=dataset_root)


def test_load_six_archived_thresholds_without_recalculation(monkeypatch):
    files = [p for p in (runner.ROOT/'experiments/calibration').rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    monkeypatch.setattr(np, 'quantile', lambda *a, **kw: pytest.fail('No quantile during load'))
    catalog, protocol, thresholds = runner.load_benchmark_inputs()
    assert len(thresholds) == len(catalog['instances']) == 6
    assert protocol['protocol_version'] == '1.1.0'
    for item in catalog['instances']:
        t = thresholds[(item['model_family'], item['category'])]
        assert t.checkpoint_sha256 == item['checkpoint_sha256']
        assert t.manifest_sha256 == item['manifest_sha256']
    assert {p: p.read_bytes() for p in files} == before


def test_registry_requires_external_anchor():
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        runner.load_benchmark_inputs(expected_integrity_sha256='0'*64)


@pytest.mark.parametrize('field,value', [
    ('checkpoint_sha256', '0'*64), ('manifest_sha256', '0'*64),
    ('protocol_sha256', '0'*64), ('model_family', 'patchcore'), ('category', 'hazelnut')])
def test_reject_wrong_threshold_association(synthetic_case, field, value):
    case = synthetic_case
    case.thresholds[('cae', 'bottle')] = replace(case.thresholds[('cae', 'bottle')], **{field: value})
    with pytest.raises(ValueError, match='association mismatch'):
        runner.run_instance('cae-bottle', **case.options)
    case.restore.assert_not_called()


def test_checkpoint_mismatch_before_restoration(synthetic_case):
    case = synthetic_case
    case.options['checkpoint'].write_bytes(b'wrong checkpoint')
    with pytest.raises(ValueError, match='Checkpoint SHA-256 mismatch'):
        runner.run_instance('cae-bottle', **case.options)
    case.restore.assert_not_called()


def test_test_projection_coverage_and_preflight_without_inference(synthetic_case, monkeypatch):
    case = synthetic_case
    original = runner.MVTecDataset
    def test_only(root, manifest, category, split, transform):
        projected = list(csv.DictReader(Path(manifest).open(encoding='utf-8')))
        assert len(projected) == 2
        assert {r['split'] for r in projected} == {'test'}
        return original(root, manifest, category, split, transform)
    monkeypatch.setattr(runner, 'MVTecDataset', test_only)
    monkeypatch.setattr(Image, 'open', lambda *a, **kw: pytest.fail('Preflight must not decode images'))
    result = runner.run_instance('cae-bottle', preflight=True, **case.options)
    assert result['status'] == 'inputs_verified' and result['test_images'] == 2
    assert not case.options['output_dir'].exists()
    case.restore.assert_not_called()


@pytest.mark.parametrize('change', ['omit', 'duplicate', 'extra'])
def test_dataset_cannot_change_test_coverage(synthetic_case, monkeypatch, change):
    case = synthetic_case
    original = runner.MVTecDataset
    def changed(*args, **kwargs):
        dataset = original(*args, **kwargs)
        if change == 'omit':
            dataset.samples.pop()
        elif change == 'duplicate':
            dataset.samples.append(dataset.samples[0])
        else:
            dataset.samples.append({**dataset.samples[0], 'image_path': 'bottle/test/good/extra.png'})
        return dataset
    monkeypatch.setattr(runner, 'MVTecDataset', changed)
    with pytest.raises(ValueError, match='exact logical Test coverage'):
        runner.run_instance('cae-bottle', **case.options)
    case.restore.assert_not_called()


@pytest.mark.parametrize('change', ['duplicate', 'traversal', 'missing_mask', 'invalid_label'])
def test_invalid_test_manifest_rejected(synthetic_case, change):
    case = synthetic_case
    rows = copy.deepcopy(case.rows)
    if change == 'duplicate':
        rows.append(rows[-1])
    elif change == 'traversal':
        rows[-1]['image_path'] = 'bottle/../outside.png'
    elif change == 'missing_mask':
        rows[-1]['mask_path'] = ''
    else:
        rows[-1]['label'] = '2'
    write_manifest(case.manifest, rows)
    case.item['manifest_sha256'] = runner.file_sha256(case.manifest)
    with pytest.raises(ValueError):
        runner._test_dataset(case.item, case.root)


def test_altered_manifest_fingerprint_rejected(synthetic_case):
    case = synthetic_case
    with case.manifest.open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        runner.run_instance('cae-bottle', **case.options)
    case.restore.assert_not_called()


@pytest.mark.parametrize('field', ['image_path', 'mask_path'])
def test_missing_test_file_before_restoration(synthetic_case, field):
    case = synthetic_case
    (case.root/case.rows[-1][field]).unlink()  # Synthetic fixture only.
    with pytest.raises(FileNotFoundError, match='Test image or mask'):
        runner.run_instance('cae-bottle', **case.options)
    case.restore.assert_not_called()


def test_real_pipeline_receives_frozen_threshold_and_valid_records(synthetic_case, monkeypatch):
    case = synthetic_case
    original = runner.evaluate_predictions
    captured = {}
    def evaluate(records, **kwargs):
        captured.update(records=records, **kwargs)
        return original(records, **kwargs)
    monkeypatch.setattr(runner, 'evaluate_predictions', evaluate)
    opened = []
    original_open = Image.open
    def open_test_only(path, *args, **kwargs):
        relative = Path(path).relative_to(case.root).as_posix()
        assert relative in {case.rows[2]['image_path'], case.rows[3]['image_path'], case.rows[3]['mask_path']}
        opened.append(relative)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, 'open', open_test_only)
    result = runner.run_instance('cae-bottle', **case.options)
    assert result['status'] == 'evaluated' and len(opened) == 3
    records = captured['records']
    assert len(records) == 2 and all(isinstance(r, runner.EvaluationRecord) for r in records)
    assert {r.image_path for r in records} == {r['image_path'] for r in case.rows[2:]}
    assert all(r.mask.shape == r.anomaly_map.shape == (256, 256) for r in records)
    assert all(set(np.unique(r.mask)) <= {0, 1} for r in records)
    assert not next(r.mask for r in records if r.label == 0).any()
    assert next(r.mask for r in records if r.label == 1).any()
    assert captured['threshold'].hex() == case.thresholds[('cae', 'bottle')].threshold.hex()
    config = captured['run_config']
    assert config['split'] == 'test' and config['target_size_hw'] == [256, 256]
    assert config['checkpoint_sha256'] == case.item['checkpoint_sha256']
    assert config['dataset_manifest_sha256'] == case.item['manifest_sha256']
    assert config['threshold_registry_integrity_sha256'] == runner.REGISTRY_SHA256
    assert config['latency_scope'] == 'model_inference_only'
    assert config['batch_size'] == 1 and config['warmup_runs_per_model'] == 3
    output = case.options['output_dir']
    assert result['export_manifest_sha256'] == runner.file_sha256(output/'manifest.json')
    manifest = json.loads((output/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert runner.file_sha256(output/name) == digest
    for entry in manifest['artifacts']:
        assert runner.file_sha256(output/entry['file']) == entry['sha256']
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    case.restore.reset_mock()
    with pytest.raises(FileExistsError):
        runner.run_instance('cae-bottle', **case.options)
    case.restore.assert_not_called()
    assert {p: p.read_bytes() for p in before} == before


def test_latency_excludes_transfers_and_has_three_warmups(synthetic_case, monkeypatch):
    case = synthetic_case
    timing = {'inside': False}
    calls = []
    original_forward = case.engine.model.forward
    def forward(batch):
        assert tuple(batch.shape) == (1, 3, 256, 256)
        assert not torch.is_grad_enabled()
        calls.append(timing['inside'])
        return original_forward(batch)
    monkeypatch.setattr(case.engine.model, 'forward', forward)
    for name in ('to', 'cpu'):
        original = getattr(torch.Tensor, name)
        def transfer(tensor, *args, _original=original, **kwargs):
            assert not timing['inside'], 'Transfer within measured inference'
            return _original(tensor, *args, **kwargs)
        monkeypatch.setattr(torch.Tensor, name, transfer)
    original_measure = runner.measure_inference
    def measure(predict, *, synchronize):
        def timed():
            timing['inside'] = True
            try:
                return predict()
            finally:
                timing['inside'] = False
        return original_measure(timed, synchronize=synchronize)
    monkeypatch.setattr(runner, 'measure_inference', measure)
    synchronize = Mock()
    monkeypatch.setattr(runner, '_synchronizer', lambda device: synchronize)
    runner.run_instance('cae-bottle', **case.options)
    assert calls == [False, False, False, True, True]
    assert synchronize.call_count == 7  # 3 warmups + before/after each of two images.


def test_cuda_synchronization_callback_without_using_gpu(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    assert runner._synchronizer('cpu') is None
    runner._synchronizer('cuda:0')()
    synchronize.assert_called_once_with(torch.device('cuda:0'))


@pytest.mark.parametrize('family', ['cae', 'patchcore'])
def test_device_adapter_matches_existing_prediction_api(family):
    image = torch.linspace(0, 1, 3*32*32).reshape(3, 32, 32)
    if family == 'cae':
        from src.inference.cae_inference import CAEInference
        engine = CAEInference(Reconstruction(), 'bottle')
    else:
        from src.inference.patchcore_inference import PatchCoreInference
        class PatchModel(Reconstruction):
            def forward(self, batch):
                maps = batch.mean(dim=1, keepdim=True) * 50
                return SimpleNamespace(pred_score=maps.flatten(1).max(dim=1).values,
                                       anomaly_map=maps)
        engine = PatchCoreInference(PatchModel(), 'bottle')
    expected = engine.predict(image)
    scores, maps = runner._predict_on_device(engine, family, image.unsqueeze(0))
    assert float(scores[0]).hex() == expected.score.hex()
    torch.testing.assert_close(maps[0], expected.anomaly_map, rtol=0, atol=0)


@pytest.mark.parametrize('mutation', ['wrong_path', 'missing_mask', 'wrong_score_policy'])
def test_runtime_samples_and_engine_stay_bound_to_contract(synthetic_case, mutation):
    case = synthetic_case
    dataset, rows = runner._test_dataset(case.item, case.root)
    samples = [dataset[i] for i in range(len(dataset))]
    if mutation == 'wrong_path':
        samples[0]['image_path'] = 'bottle/train/good/001.png'
    elif mutation == 'missing_mask':
        next(s for s in samples if s['label'] == 1)['mask'] = None
    else:
        case.engine.config.score_method = 'mean'
    with pytest.raises(ValueError):
        runner.collect_test_records(samples, rows, case.engine, case.item)


def test_all_six_preflight_before_any_execution(tmp_path, monkeypatch):
    catalog, _, _ = runner.load_benchmark_inputs()
    ids = [item['id'] for item in catalog['instances']]
    calls = []
    def prepare(instance, checkpoint, dataset_root, output_dir, inputs):
        calls.append(('prepare', instance))
        return instance
    def execute(instance, **kwargs):
        calls.append(('execute', instance))
        assert kwargs['preflight'] is True
        return instance
    monkeypatch.setattr(runner, '_prepare', prepare)
    monkeypatch.setattr(runner, '_execute', execute)
    checkpoints = {i: tmp_path/f'{i}.pt' for i in ids}
    result = runner.run_all(dataset_root=tmp_path, checkpoints=checkpoints,
                            output_dir=tmp_path/'six', preflight=True)
    assert result == ids
    assert calls == [('prepare', i) for i in ids] + [('execute', i) for i in ids]
    assert not (tmp_path/'six').exists()
    with pytest.raises(ValueError, match='Exactly six'):
        runner.run_all(dataset_root=tmp_path, checkpoints={}, output_dir=tmp_path/'bad')


def test_all_six_preflight_failure_prevents_inference(tmp_path, monkeypatch):
    ids = [i['id'] for i in runner.read_catalog()['instances']]
    def prepare(instance, *args):
        if instance == ids[-1]:
            raise FileNotFoundError('last checkpoint missing')
        return instance
    execute = Mock()
    monkeypatch.setattr(runner, '_prepare', prepare)
    monkeypatch.setattr(runner, '_execute', execute)
    with pytest.raises(FileNotFoundError):
        runner.run_all(dataset_root=tmp_path, checkpoints={i: i for i in ids}, output_dir=tmp_path/'six')
    execute.assert_not_called()


def test_cli_targeted_preflight(synthetic_case, capsys):
    case = synthetic_case
    runner.main(['--instance', 'cae-bottle', '--dataset-root', str(case.root),
                 '--checkpoint', str(case.options['checkpoint']),
                 '--output', str(case.options['output_dir']), '--preflight'])
    result = json.loads(capsys.readouterr().out)
    assert result['instance'] == 'cae-bottle' and result['status'] == 'inputs_verified'
    case.restore.assert_not_called()
