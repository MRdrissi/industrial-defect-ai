"""Runner tests use tiny generated images and synthetic checkpoint tensors only."""
import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from src.evaluation import run_calibration as runner


def test_dataset_reads_only_logical_validation(tmp_path, monkeypatch):
    item=next(i for i in runner.read_catalog()["instances"] if i["id"]=="cae-bottle")
    paths=runner.validation_paths(runner.ROOT/item["manifest"],category="bottle",
             expected_sha256=item["manifest_sha256"],expected_count=42)
    for relative in paths:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True)
        Image.new("RGB",(8,8),color=(64,64,64)).save(p)
    opened=[]
    original=Image.open
    def guarded(path,*a,**kw):
        relative=Path(path).relative_to(tmp_path).as_posix()
        assert relative in paths
        opened.append(relative)
        return original(path,*a,**kw)
    monkeypatch.setattr(Image,"open",guarded)
    import src.data.mvtec_dataset as dataset_module
    original_dataset = dataset_module.MVTecDataset
    def validation_only_dataset(root, manifest, *args):
        text = Path(manifest).read_text(encoding="utf-8")
        assert ",test," not in text and ",train," not in text
        assert text.count(",val,") == 42
        return original_dataset(root, manifest, *args)
    monkeypatch.setattr(dataset_module, "MVTecDataset", validation_only_dataset)
    ds=runner._validation_dataset(item,tmp_path)
    engine=Mock()
    engine.predict.return_value=types.SimpleNamespace(score=.25)
    records=runner.collect_validation_scores(ds,engine)
    assert len(records)==len(opened)==42
    assert {s.image_path for s in records}==set(paths)
    assert all(s.label==0 for s in records)
    assert engine.predict.call_count==42


def test_missing_validation_files_fail_before_inference(tmp_path):
    item=runner.read_catalog()["instances"][0]
    with pytest.raises(FileNotFoundError,match="Validation"):
        runner._validation_dataset(item,tmp_path)


def test_checkpoint_wrong_hash_rejected_before_deserialization(tmp_path,monkeypatch):
    item=runner.read_catalog()["instances"][0]
    p=tmp_path/"weights.pt";p.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(torch,"load",lambda *a,**kw:pytest.fail("must not unpickle"))
    with pytest.raises(ValueError,match="SHA-256"):
        runner.restore_instance(item,p)


def test_cae_score_definition_frozen_before_loading(tmp_path,monkeypatch):
    import src.inference.cae_inference as cae
    item=copy.deepcopy(runner.read_catalog()["instances"][0])
    p=tmp_path/"weights.pt";p.write_bytes(b"synthetic")
    item["checkpoint_sha256"]=runner.file_sha256(p)
    called=[]
    def load(category,config_path,checkpoint,device,config):
        called.append(config)
        return "engine"
    monkeypatch.setattr(cae.CAEInference,"from_checkpoint",load)
    assert runner.restore_instance(item,p)=="engine"
    assert called[0].score_method=="max" and called[0].smoothing_sigma==4.0


@pytest.fixture
def hazelnut_checkpoint(tmp_path,monkeypatch):
    item=copy.deepcopy(next(i for i in runner.read_catalog()["instances"] if i["id"]=="patchcore-hazelnut"))
    class SyntheticModel(torch.nn.Module):
        def __init__(self,backbone,layers,pre_trained,num_neighbors):
            super().__init__()
            assert backbone=="wide_resnet50_2" and layers==["layer2","layer3"]
            assert pre_trained is False and num_neighbors==1
            self.feature_extractor=torch.nn.Linear(1,1,bias=False)
            self.register_buffer("memory_bank",torch.zeros(3205,1536))
    namespace="anomalib.models.image.patchcore.torch_model"
    for name in ["anomalib","anomalib.models","anomalib.models.image","anomalib.models.image.patchcore",namespace]:
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    sys.modules[namespace].PatchcoreModel=SyntheticModel
    original=runner.importlib.metadata.version
    monkeypatch.setattr(runner.importlib.metadata,"version",lambda n:"2.7.0" if n=="anomalib" else original(n))
    model=SyntheticModel("wide_resnet50_2",["layer2","layer3"],False,1)
    item["memory_bank_sha256"]=hashlib.sha256(model.memory_bank.numpy().tobytes()).hexdigest()
    config={"category":"hazelnut","experiment_id":"EXP-002","pooling_corrected":True,
            "backbone":"wide_resnet50_2","layers":["layer2","layer3"],"num_neighbors":1,
            "memory_bank_shape":[3205,1536],"image_size":[256,256],
            "memory_bank_sha256":item["memory_bank_sha256"]}
    saved={"config":config,"state_dict":model.state_dict()}
    p=tmp_path/"synthetic.pt"
    return item,saved,p


def test_corrected_hazelnut_schema_strict_restore(hazelnut_checkpoint):
    item,saved,p=hazelnut_checkpoint
    torch.save(saved,p);item["checkpoint_sha256"]=runner.file_sha256(p)
    before=p.read_bytes()
    engine=runner.restore_instance(item,p)
    assert tuple(engine.model.memory_bank.shape)==(3205,1536)
    assert engine.model.training is False
    assert all(not v.requires_grad for v in engine.model.feature_extractor.parameters())
    assert p.read_bytes()==before


@pytest.mark.parametrize("mutation",["exp001","pooling","missing_weight","bank"])
def test_hazelnut_restoration_rejects_bad_state(hazelnut_checkpoint,mutation):
    item,saved,p=hazelnut_checkpoint
    if mutation=="exp001": saved["config"]["experiment_id"]="EXP-001"
    elif mutation=="pooling": saved["config"]["pooling_corrected"]=False
    elif mutation=="missing_weight": saved["state_dict"].pop("feature_extractor.weight")
    else: saved["state_dict"]["memory_bank"][0,0]=1.
    torch.save(saved,p);item["checkpoint_sha256"]=runner.file_sha256(p)
    with pytest.raises((ValueError,RuntimeError)):
        runner.restore_instance(item,p)


def test_patchcore_version_mismatch_rejected(tmp_path,monkeypatch):
    item=copy.deepcopy(next(i for i in runner.read_catalog()["instances"] if i["id"]=="patchcore-bottle"))
    p=tmp_path/"weights.pt";p.write_bytes(b"synthetic")
    item["checkpoint_sha256"]=runner.file_sha256(p)
    monkeypatch.setattr(runner.importlib.metadata,"version",lambda n:"wrong-version")
    with pytest.raises(ValueError,match="2.7.0"):
        runner.restore_instance(item,p)
