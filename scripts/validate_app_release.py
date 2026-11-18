"""Audit all six actual engines and Streamlit interactions without mocking inference."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--data-root', type=Path, required=True)
args = parser.parse_args()
if not __debug__:
    raise SystemExit('Release validation requires assertions enabled; do not use python -O.')
root = args.root.resolve()
sys.path.insert(0, str(root))
out = args.output
out.mkdir(parents=True, exist_ok=True)

import torch
from PIL import Image
from app.image_utils import decode_image, image_to_tensor
from app.overlay import make_overlay
from app.report import build_report
from src.inference.model_registry import load_registry
from src.inference.model_loader import load_real_engine, clear_model_cache
from src.inference.inference_contract import InferenceRequest, run_inference

torch.set_num_threads(4)
registry = load_registry(root / 'configs/real_model_registry.json')
catalog = json.loads((root / 'experiments/configs/calibration_instances.json').read_text())
catalog = {i['id']: i for i in catalog['instances']}
source_images = json.loads((root / 'docs/data/us-05-04/source-images.json').read_text())['files']
evidence = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                python=sys.version, torch_threads=torch.get_num_threads(), artifacts=[], predictions=[], ui=[], errors=[])
evidence['dependencies'] = {x: importlib.metadata.version(x) for x in
                            ['torch', 'torchvision', 'anomalib', 'timm', 'streamlit', 'numpy', 'pillow']}
evidence['source_files_sha256'] = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in ('src', 'app') for path in (root / folder).rglob('*.py')
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save():
    (out / 'real-validation.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')

threshold_path = root / 'experiments/calibration/six-thresholds/thresholds.json'
threshold_hash = sha(threshold_path.read_bytes())
evidence['threshold_registry_sha256'] = threshold_hash

for key, model in registry.items():
    instance = f'{key[1]}-{key[0]}'
    try:
        assert model['backend'] == model['method'] != 'fake'
        artifact = root / model['artifact_path']
        actual = sha(artifact.read_bytes())
        assert actual == model['sha256'] == catalog[instance]['checkpoint_sha256']
        assert threshold_hash == model['threshold_source']['registry_sha256']
        assert float(model['threshold']).hex() == model['threshold_source']['threshold_hex']
        record = dict(instance=instance, path=model['artifact_path'], sha256=actual,
                      bytes=artifact.stat().st_size, threshold=model['threshold'])
        engine = load_real_engine(registry, *key)
        if model['method'] == 'patchcore':
            bank = engine.model.memory_bank.detach().cpu().contiguous()
            record['memory_bank_shape'] = list(bank.shape)
            record['memory_bank_sha256'] = sha(bank.numpy().tobytes())
            assert record['memory_bank_shape'] == catalog[instance]['memory_bank_shape']
            if 'memory_bank_sha256' in catalog[instance]:
                assert record['memory_bank_sha256'] == catalog[instance]['memory_bank_sha256']
        evidence['artifacts'].append(record)
        benchmark = next((root / 'experiments/benchmark').glob(f'kaggle-*/{instance}/predictions.csv'))
        with benchmark.open(newline='', encoding='utf-8') as stream:
            archived = {r['image_path']: r for r in csv.DictReader(stream)}
        for rel, provenance in source_images.items():
            if not rel.startswith(key[0] + '/test/'):
                continue
            content = (args.data_root / rel).read_bytes()
            assert sha(content) == provenance['sha256'], rel
            image = decode_image(content)
            result = run_inference(InferenceRequest(*key, image_to_tensor(image)), registry)
            reference = float(archived[rel]['score'])
            assert not result.simulated
            assert tuple(result.anomaly_map.shape) == (256, 256)
            assert torch.isfinite(result.anomaly_map).all()
            assert math.isclose(result.score, reference, rel_tol=1e-4, abs_tol=1e-5), (rel, result.score, reference)
            assert result.decision == ('suspect' if reference > model['threshold'] else 'normal')
            item = result.to_dict() | dict(instance=instance, image_path=rel, image_sha256=sha(content),
                                          archived_score=reference, score_abs_error=abs(result.score-reference))
            stem = f"{instance}-{Path(rel).parent.name}-{Path(rel).stem}"
            torch.save(result.anomaly_map, out / f'{stem}-map.pt')
            heatmap, overlay, scale = make_overlay(image, result.anomaly_map, result.threshold)
            Image.fromarray(overlay).save(out / f'{stem}-overlay.png')
            Image.fromarray(heatmap).save(out / f'{stem}-heatmap.png')
            report = build_report(result, model, content, Path(rel).name, image.size)
            (out / f'{stem}-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            item['map_sha256'] = sha(result.anomaly_map.contiguous().numpy().tobytes())
            evidence['predictions'].append(item)
            print(f'PASS {instance} {rel} score={result.score:.10g} decision={result.decision}', flush=True)
    except Exception:
        evidence['errors'].append(dict(stage='real_inference', instance=instance, traceback=traceback.format_exc()))
        print(evidence['errors'][-1]['traceback'], flush=True)
    finally:
        clear_model_cache()
        save()

from streamlit.testing.v1 import AppTest
for category in ('bottle', 'hazelnut', 'transistor'):
    rel = next(r for r in source_images if r.startswith(category + '/test/'))
    content = (args.data_root / rel).read_bytes()
    for method in ('cae', 'patchcore', 'both'):
        try:
            at = AppTest.from_file(str(root / 'app/streamlit_app.py'), default_timeout=120).run()
            assert not at.exception and not at.error
            at.selectbox(key='category_selector').set_value(category).run()
            if method == 'both':
                at.checkbox(key='compare_mode').set_value(True).run()
            else:
                at.selectbox(key='method_selector').set_value(method).run()
            at.file_uploader[0].upload(Path(rel).name, content, 'image/png').run()
            at.button(key='analyze_image').click().run(timeout=120)
            assert not at.exception and not at.error, ([str(e.value) for e in at.exception], [e.value for e in at.error])
            state = at.session_state['last_comparison' if method == 'both' else 'last_inspection']
            report = state['report']
            if method != 'both':
                assert not state['result'].simulated
                assert report['model']['checkpoint_sha256'] == registry[(category, method, state['result'].version)]['sha256']
                assert report['image']['sha256'] == sha(content)
            else:
                assert set(report['reports']) == {'cae', 'patchcore'}
                for compared_method, compared_report in report['reports'].items():
                    assert not compared_report['model']['simulated']
                    assert compared_report['image']['sha256'] == sha(content)
            assert len(at.metric) == (8 if method == 'both' else 4)
            assert at.button(key='analyze_image').disabled
            assert len(at.download_button) == 1
            at._page_hash = at.get('page_link')[0].proto.page_script_hash
            at.run()
            assert not at.exception and not at.error
            assert at.session_state['last_comparison' if method == 'both' else 'last_inspection']['report'] == report
            at._page_hash = at.get('page_link')[0].proto.page_script_hash
            at.run()
            assert not at.exception and not at.error
            assert len(at.metric) == (8 if method == 'both' else 4)
            evidence['ui'].append(dict(category=category, method=method, status='passed', metrics=[m.value for m in at.metric]))
            (out / f'ui-{category}-{method}-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'PASS Streamlit {category} {method}', flush=True)
        except Exception:
            evidence['errors'].append(dict(stage='streamlit', category=category, method=method, traceback=traceback.format_exc()))
            print(evidence['errors'][-1]['traceback'], flush=True)
        finally:
            clear_model_cache()
            save()

expected_predictions = 2 * sum('/test/' in rel for rel in source_images)
evidence['expected_predictions'] = expected_predictions
evidence['passed'] = (not evidence['errors'] and len(evidence['artifacts']) == 6
                      and len(evidence['ui']) == 9 and len(evidence['predictions']) == expected_predictions)
save()
sys.exit(0 if evidence['passed'] else 1)
