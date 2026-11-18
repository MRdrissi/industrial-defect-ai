import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.restore_app_artifacts import restore


def setup(root, data=b'approved checkpoint'):
    (root / 'configs').mkdir()
    model = dict(category='bottle', method='cae', version='exp-002',
                 artifact_path='artifacts/cae/bottle/EXP-002/cae_model.pt',
                 sha256=hashlib.sha256(data).hexdigest())
    (root / 'configs/real_model_registry.json').write_text(json.dumps({'models': [model]}))
    return root / model['artifact_path']


def test_zip_member_is_matched_by_hash_not_extracted_path(tmp_path):
    target = setup(tmp_path)
    archive = tmp_path / 'input.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('../../untrusted.pth', b'approved checkpoint')
        output.writestr('cae_model.pt', b'wrong checkpoint')
    assert restore(tmp_path, [archive])[0]['status'] == 'ready'
    assert not target.exists()
    assert restore(tmp_path, [archive], apply=True)[0]['status'] == 'restored_verified'
    assert target.read_bytes() == b'approved checkpoint'
    assert restore(tmp_path, [], apply=True)[0]['status'] == 'already_verified'


def test_existing_bytes_are_never_overwritten(tmp_path):
    target = setup(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'original data')
    source = tmp_path / 'weights.pt'
    source.write_bytes(b'approved checkpoint')
    with pytest.raises(ValueError, match='overwrite'):
        restore(tmp_path, [source], apply=True)
    assert target.read_bytes() == b'original data'


def test_wrong_checkpoint_fails_without_creating_destination(tmp_path):
    target = setup(tmp_path)
    source = tmp_path / 'weights.pt'
    source.write_bytes(b'wrong checkpoint')
    with pytest.raises(FileNotFoundError, match='matching source'):
        restore(tmp_path, [source], apply=True)
    assert not target.exists()


def test_destination_must_be_inside_artifacts(tmp_path):
    setup(tmp_path)
    config = tmp_path / 'configs/real_model_registry.json'
    value = json.loads(config.read_text())
    value['models'][0]['artifact_path'] = '../outside.pt'
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='inside artifacts'):
        restore(tmp_path, [], apply=True)
