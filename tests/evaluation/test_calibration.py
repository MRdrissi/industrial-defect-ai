"""Synthetic anti-leakage, exact-reuse and artifact-integrity regressions."""
import copy
import csv
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import calibration as c

ROOT = Path(__file__).resolve().parents[2]


def write_manifest(path, category="bottle", count=4):
    rows = [[category, "val", f"{category}/train/good/{i:03d}.png", 0, "good", ""]
            for i in range(count)]
    rows += [[category, "train", f"{category}/train/good/999.png", 0, "good", ""],
             [category, "test", f"{category}/test/crack/000.png", 1, "crack", "mask.png"]]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["category", "split", "image_path", "label", "defect_type", "mask_path"])
        w.writerows(rows)
    return path


@pytest.fixture
def case(tmp_path):
    manifest = write_manifest(tmp_path/"manifest.csv")
    protocol = tmp_path/"protocol.json"
    protocol.write_bytes((ROOT/"experiments/configs/evaluation_protocol.json").read_bytes())
    provenance = {
        "run_id": "synthetic-test-only", "model_family": "cae", "category": "bottle",
        "source_experiment": "synthetic/EXP-TEST", "git_commit": "a"*40,
        "checkpoint_sha256": "b"*64, "model_config_sha256": "c"*64,
        "device": "cpu", "hardware": "synthetic", "python_version": "synthetic",
        "torch_version": "synthetic", "numpy_version": np.__version__, "split": "val",
        "score_definition": copy.deepcopy(c.SCORE_DEFINITIONS["cae"]),
        "preprocessing": {"profile": "cae", "size": [256, 256]},
    }
    return {
        "samples": [c.CalibrationSample(f"bottle/train/good/{i:03d}.png", 0, float(i))
                    for i in range(4)],
        "kwargs": {"manifest_path": manifest, "protocol_path": protocol,
                   "expected_manifest_sha256": c.file_sha256(manifest),
                   "expected_protocol_sha256": c.file_sha256(protocol),
                   "expected_count": 4, "provenance": provenance},
    }


def fit(case):
    return c.calibrate(case["samples"], **case["kwargs"])[0]


def loader_kwargs(case, digest):
    kw = {k: v for k, v in case["kwargs"].items() if k != "provenance"}
    return {**kw, "expected_integrity_sha256": digest, "model_family": "cae",
            "category": "bottle", "expected_checkpoint_sha256": "b"*64}


def test_reference_linear_quantile_and_statistics(case):
    report = fit(case)
    assert report["threshold"] == pytest.approx(2.85)
    assert report["threshold"] != np.quantile([0., 1., 2., 3.], .95, method="nearest")
    assert report["distribution"]["mean"] == 1.5
    assert report["distribution"]["median"] == 1.5
    assert report["distribution"]["std"] == pytest.approx(np.sqrt(1.25))
    assert report["n_images"] == 4
    assert report["threshold_hex"] == report["threshold"].hex()


@pytest.mark.parametrize("bad", [None, True, "1", 1+2j, np.nan, np.inf, -np.inf, [], {}])
def test_invalid_scores(case, bad):
    case["samples"][0] = replace(case["samples"][0], score=bad)
    with pytest.raises(ValueError):
        fit(case)


@pytest.mark.parametrize("label", [1, -1, 0.0, False, "0", None])
def test_only_integer_normal_labels(case, label):
    case["samples"][0] = replace(case["samples"][0], label=label)
    with pytest.raises(ValueError):
        fit(case)


@pytest.mark.parametrize("path", [
    "bottle/test/crack/000.png", "bottle/train/good/999.png", "bottle/train/good/missing.png",
    "../bottle/train/good/000.png", "/bottle/train/good/000.png", "bottle//train/good/000.png",
    "bottle\\train\\good\\000.png", "hazelnut/train/good/000.png", "C:/data/image.png", "",
])
def test_test_train_unknown_and_noncanonical_paths_rejected(case, path):
    case["samples"][0] = replace(case["samples"][0], image_path=path)
    with pytest.raises(ValueError):
        fit(case)


@pytest.mark.parametrize("mode", ["empty", "missing", "duplicate", "wrong_count", "one"])
def test_full_unique_validation_required(case, mode):
    if mode == "empty":
        case["samples"] = []
    elif mode == "missing":
        case["samples"].pop()
    elif mode == "duplicate":
        case["samples"][1] = case["samples"][0]
    elif mode == "wrong_count":
        case["kwargs"]["expected_count"] = 5
    else:
        case["kwargs"]["expected_count"] = 1
    with pytest.raises(ValueError):
        fit(case)


@pytest.mark.parametrize("values,expected", [
    ([-9., -4., -2., -1.], -1.15), ([7., 7., 7., 7.], 7.),
    ([1., 1.+1e-9, 1.+2e-9, 1.+3e-9], 1.+2.85e-9),
])
def test_unbounded_scores_ties_and_float64(case, values, expected):
    case["samples"] = [replace(s, score=v) for s,v in zip(case["samples"],values)]
    assert fit(case)["threshold"] == pytest.approx(expected, abs=1e-14, rel=0)


def test_permutation_invariant_report_and_artifact_bytes(case, tmp_path):
    first = fit(case)
    d1 = c.export_calibration(tmp_path/"one", case["samples"], **case["kwargs"])
    case["samples"].reverse()
    assert fit(case) == first
    d2 = c.export_calibration(tmp_path/"two", case["samples"], **case["kwargs"])
    assert d1 == d2


@pytest.mark.parametrize("section,key,value", [
    ("calibration","quantile",.90), ("calibration","quantile_method","nearest"),
    ("dataset","calibration_split","test"), ("dataset","test_used_for_calibration",True),
    ("calibration","test_labels_used_for_calibration",True),
    ("calibration","thresholds_frozen_before_test",False),
    ("prediction_contract","decision_rule","score >= threshold"),
    ("prediction_contract","score_normalization_on_test",True),
    ("prediction_contract","cae_image_score","mean"),
    ("prediction_contract","patchcore_image_score","normalized"),
])
def test_changed_policy_rejected_even_with_new_hash(case, section, key, value):
    p=case["kwargs"]["protocol_path"]; data=json.loads(p.read_text())
    data[section][key]=value; p.write_bytes(c._bytes(data))
    case["kwargs"]["expected_protocol_sha256"]=c.file_sha256(p)
    with pytest.raises(ValueError):
        fit(case)


@pytest.mark.parametrize("key,value", [
    ("split","test"), ("model_family","other"), ("category","other"),
    ("checkpoint_sha256","wrong"), ("git_commit","a2efa10"),
    ("score_definition",{"method":"mean"}), ("preprocessing",{"profile":"patchcore","size":[256,256]}),
])
def test_contradictory_provenance(case,key,value):
    case["kwargs"]["provenance"][key]=value
    with pytest.raises(ValueError):
        fit(case)


def test_missing_provenance(case):
    del case["kwargs"]["provenance"]["checkpoint_sha256"]
    with pytest.raises(ValueError):
        fit(case)


@pytest.mark.parametrize("file_key", ["manifest_path","protocol_path"])
def test_changed_source_bytes_rejected(case,file_key):
    p=case["kwargs"][file_key];p.write_bytes(p.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="SHA-256"):
        fit(case)


@pytest.mark.parametrize("mutation",["duplicate","anomaly","split","mask"])
def test_invalid_manifest_with_matching_hash(case,mutation):
    p=case["kwargs"]["manifest_path"]
    lines=p.read_text().splitlines()
    if mutation=="duplicate": lines.append(lines[1].replace(",val,",",test,"))
    elif mutation=="anomaly": lines[1]=lines[1].replace(",0,good,",",1,crack,")
    elif mutation=="split": lines[1]=lines[1].replace(",val,",",other,")
    else: lines[1]+="mask.png"
    p.write_text("\n".join(lines)+"\n",encoding="utf-8")
    case["kwargs"]["expected_manifest_sha256"]=c.file_sha256(p)
    with pytest.raises(ValueError):
        fit(case)


def test_test_labels_are_not_used(case):
    # Membership is read for all rows, but Test labels never enter calibration.
    p=case["kwargs"]["manifest_path"]
    p.write_text(p.read_text().replace(",1,crack,mask.png",",UNREAD,crack,mask.png"),encoding="utf-8")
    case["kwargs"]["expected_manifest_sha256"]=c.file_sha256(p)
    assert fit(case)["threshold"] == pytest.approx(2.85)


def test_round_trip_exact_threshold_without_recalculation(case,tmp_path,monkeypatch):
    out=tmp_path/"run";digest=c.export_calibration(out,case["samples"],**case["kwargs"])
    def forbidden(*a,**kw): raise AssertionError("No recalibration during loading")
    monkeypatch.setattr(np,"quantile",forbidden)
    frozen=c.load_threshold(out,**loader_kwargs(case,digest))
    assert frozen.threshold.hex()==json.loads((out/"calibration.json").read_text())["threshold_hex"]
    np.testing.assert_array_equal(frozen.predict([frozen.threshold,np.nextafter(frozen.threshold,np.inf)]),[0,1])
    assert (out/"validation_scores.csv").read_text().count("train/good")==4


@pytest.mark.parametrize("filename",["calibration.json","validation_scores.csv","integrity.json"])
def test_modified_bundle_rejected(case,tmp_path,filename):
    out=tmp_path/"run";digest=c.export_calibration(out,case["samples"],**case["kwargs"])
    p=out/filename;p.write_bytes(p.read_bytes()+b" ")
    with pytest.raises(ValueError,match="SHA-256"):
        c.load_threshold(out,**loader_kwargs(case,digest))


def test_attacker_rehashing_bundle_cannot_replace_external_anchor(case,tmp_path):
    out=tmp_path/"run";digest=c.export_calibration(out,case["samples"],**case["kwargs"])
    p=out/"calibration.json";data=json.loads(p.read_text());data["threshold"]=999.;p.write_bytes(c._bytes(data))
    m=out/"integrity.json";meta=json.loads(m.read_text());meta["files"]["calibration.json"]=c.file_sha256(p);m.write_bytes(c._bytes(meta))
    with pytest.raises(ValueError,match="SHA-256"):
        c.load_threshold(out,**loader_kwargs(case,digest))


@pytest.mark.parametrize("key,value",[("category","hazelnut"),("expected_checkpoint_sha256","d"*64),("model_family","patchcore")])
def test_wrong_consumer_identity_rejected(case,tmp_path,key,value):
    out=tmp_path/"run";digest=c.export_calibration(out,case["samples"],**case["kwargs"])
    kwargs=loader_kwargs(case,digest);kwargs[key]=value
    with pytest.raises(ValueError):
        c.load_threshold(out,**kwargs)


def test_existing_export_preserved(case,tmp_path):
    out=tmp_path/"run";digest=c.export_calibration(out,case["samples"],**case["kwargs"])
    with pytest.raises(FileExistsError):
        c.export_calibration(out,case["samples"],**case["kwargs"])
    assert c.file_sha256(out/"integrity.json")==digest


def test_write_failure_no_incomplete_run(case,tmp_path,monkeypatch):
    original=Path.write_bytes
    def failing(path,raw):
        if path.name=="validation_scores.csv": raise OSError("disk")
        return original(path,raw)
    monkeypatch.setattr(Path,"write_bytes",failing)
    out=tmp_path/"run"
    with pytest.raises(OSError):
        c.export_calibration(out,case["samples"],**case["kwargs"])
    assert not out.exists() and not (tmp_path/".run.lock").exists()
    assert not list(tmp_path.glob(".calibration-staging-*"))


def test_hazelnut_exp001_blocked(case):
    p=case["kwargs"]["provenance"]
    p.update(model_family="patchcore",category="hazelnut",source_experiment="patchcore-hazelnut/EXP-001",
             score_definition=c.SCORE_DEFINITIONS["patchcore"],
             preprocessing={"profile":"patchcore","size":[256,256],"mean":[.485,.456,.406],"std":[.229,.224,.225]})
    with pytest.raises(ValueError,match="EXP-002"): fit(case)


def test_six_threshold_registry_consumer_no_recalibration(case,tmp_path,monkeypatch):
    out=tmp_path/"run";digest=c.export_calibration(out,case["samples"],**case["kwargs"])
    frozen=c.load_threshold(out,**loader_kwargs(case,digest))
    # Synthetic registry objects, not real six-model calibrations.
    six=[replace(frozen,model_family=f,category=k) for f in c.FAMILIES for k in c.CATEGORIES]
    root=tmp_path/"registry";anchor=c.export_threshold_registry(root,six)
    monkeypatch.setattr(np,"quantile",lambda *a,**kw:pytest.fail("implicit recalibration"))
    loaded=c.load_threshold_registry(root,expected_integrity_sha256=anchor,
                                     expected_protocol_sha256=frozen.protocol_sha256)
    assert len(loaded)==6
    assert all(t.threshold.hex()==frozen.threshold.hex() for t in loaded.values())
    with pytest.raises(ValueError): c.export_threshold_registry(tmp_path/"incomplete",six[:-1])
    with pytest.raises(ValueError): c.export_threshold_registry(tmp_path/"duplicate",[six[0]]*6)
    with pytest.raises(FileExistsError): c.export_threshold_registry(root,six)


def test_repository_catalog_manifest_counts():
    from src.evaluation.run_calibration import read_catalog
    catalog=read_catalog()
    assert len(catalog["instances"])==6
    for item in catalog["instances"]:
        paths=c.validation_paths(ROOT/item["manifest"],category=item["category"],
             expected_sha256=item["manifest_sha256"],expected_count=item["expected_validation_count"])
        assert len(paths)=={"bottle":42,"hazelnut":78,"transistor":43}[item["category"]]
        assert all("/train/good/" in p for p in paths)
    hz=next(i for i in catalog["instances"] if i["id"]=="patchcore-hazelnut")
    assert hz["source_experiment"]=="patchcore-hazelnut/EXP-002"

def test_frozen_threshold_rejects_invalid_values():
    with pytest.raises(ValueError):
        c.FrozenThreshold("cae","bottle",float("nan"),"b"*64,"c"*64,"d"*64,"e"*64)


def test_registry_refuses_different_splits_for_same_category(tmp_path):
    six=[c.FrozenThreshold(f,k,2.5,"b"*64,"c"*64,"d"*64,"e"*64)
         for f in c.FAMILIES for k in c.CATEGORIES]
    six[-1]=replace(six[-1],manifest_sha256="f"*64)
    with pytest.raises(ValueError,match="same category manifest"):
        c.export_threshold_registry(tmp_path/"registry",six)
