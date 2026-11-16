"""Validation-only threshold calibration; no model or test inference."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from .exporters import _export_lock, file_sha256

FAMILIES = ("cae", "patchcore")
CATEGORIES = ("bottle", "hazelnut", "transistor")
SCORE_DEFINITIONS = {
    "cae": {"method": "max_gaussian_smoothed_channel_mean_squared_reconstruction_error",
            "smoothing_sigma": 4.0, "kernel_size": 33},
    "patchcore": {"method": "native_pred_score"},
}


def _hash(value, length=64):
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError("Invalid SHA fingerprint.")
    return value


def _real(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("Expected a real numeric score.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Scores must be finite.")
    return value


def _path(value):
    if (not isinstance(value, str) or not value or ":" in value or "\\" in value
            or any(ord(c) < 32 for c in value)):
        raise ValueError("Invalid relative image path.")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or p.as_posix() != value or value == ".":
        raise ValueError("Invalid relative image path.")
    return value


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"Duplicate JSON key: {k}")
        result[k] = v
    return result


def _json(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UTF-8 BOM is forbidden.")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _bytes(document):
    return (json.dumps(document, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, indent=2) + "\n").encode("utf-8")


def _verified_bytes(path, expected):
    _hash(expected)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return raw


def load_protocol(path, expected_sha256):
    """Validate the frozen policy against an externally supplied fingerprint."""
    p = _json(_verified_bytes(path, expected_sha256))
    expected_dataset = {"calibration_split": "val", "evaluation_split": "test",
                        "validation_contains_only_normal_images": True,
                        "test_used_for_calibration": False}
    expected_calibration = {
        "method": "normal_validation_quantile", "quantile": 0.95,
        "quantile_method": "linear", "calibrate_per_model_and_category": True,
        "number_of_thresholds": 6, "thresholds_frozen_before_test": True,
        "test_labels_used_for_calibration": False,
    }
    for section, expected in (("dataset", expected_dataset), ("calibration", expected_calibration)):
        for key, value in expected.items():
            actual = p[section].get(key)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"Forbidden calibration protocol: {section}.{key}")
    prediction = p["prediction_contract"]
    if (prediction.get("decision_rule") != "score > threshold"
            or prediction.get("higher_score_means_more_anomalous") is not True
            or prediction.get("score_normalization_on_test") is not False
            or type(prediction.get("normal_label")) is not int
            or type(prediction.get("anomaly_label")) is not int
            or prediction.get("normal_label") != 0 or prediction.get("anomaly_label") != 1
            or prediction.get("cae_image_score") != SCORE_DEFINITIONS["cae"]["method"]
            or prediction.get("patchcore_image_score") != SCORE_DEFINITIONS["patchcore"]["method"]
            or prediction.get("score_definitions") != SCORE_DEFINITIONS):
        raise ValueError("Inconsistent score definition or decision rule.")
    if (p["spatial_evaluation"]["target_size_hw"] != [256, 256]
            or set(p["instances"]["model_families"]) != set(FAMILIES)
            or set(p["instances"]["categories"]) != set(CATEGORIES)
            or p["instances"]["expected_count"] != 6):
        raise ValueError("Inconsistent instances or preprocessing resolution.")
    return p


def validation_paths(manifest_path, *, category, expected_sha256, expected_count):
    """Use logical splits, never physical folder names; inspect labels only in val."""
    if category not in CATEGORIES or type(expected_count) is not int or expected_count < 2:
        raise ValueError("Invalid category or insufficient validation count.")
    raw = _verified_bytes(manifest_path, expected_sha256)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if reader.fieldnames != ["category", "split", "image_path", "label", "defect_type", "mask_path"]:
        raise ValueError("Invalid manifest schema.")
    seen, val = set(), set()
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed manifest row.")
        path = _path(row["image_path"])
        if row["category"] != category or path.split("/")[0] != category:
            raise ValueError("Manifest category mismatch.")
        if path in seen:
            raise ValueError("Duplicate manifest path across splits.")
        seen.add(path)
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown logical split.")
        if row["split"] == "val":
            if row["label"] != "0" or row["defect_type"] != "good" or row["mask_path"]:
                raise ValueError("Calibration requires normal Validation only.")
            val.add(path)
    if len(val) != expected_count:
        raise ValueError("Validation manifest count mismatch.")
    return frozenset(val)


@dataclass(frozen=True)
class CalibrationSample:
    image_path: str
    label: int
    score: float


def _samples(samples, expected_paths):
    samples = list(samples)
    seen, result = set(), []
    for sample in samples:
        if not isinstance(sample, CalibrationSample):
            raise TypeError("Expected CalibrationSample.")
        path = _path(sample.image_path)
        if isinstance(sample.label, (bool, np.bool_)) or not isinstance(sample.label, (int, np.integer)) or sample.label != 0:
            raise ValueError("Only normal label 0 is accepted.")
        if path not in expected_paths:
            raise ValueError("Path is outside logical Validation (Train/Test forbidden).")
        if path in seen:
            raise ValueError("Duplicate calibration path.")
        seen.add(path)
        result.append(CalibrationSample(path, 0, _real(sample.score)))
    if seen != set(expected_paths):
        raise ValueError("Complete Validation coverage required; missing images.")
    return sorted(result, key=lambda x: x.image_path)


def _provenance(provenance):
    required = ("run_id", "model_family", "category", "source_experiment", "git_commit",
                "checkpoint_sha256", "model_config_sha256", "device", "hardware",
                "python_version", "torch_version", "numpy_version")
    if not isinstance(provenance, dict):
        raise ValueError("Provenance must be a dictionary.")
    for key in required:
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"Missing provenance: {key}")
    family, category = provenance["model_family"], provenance["category"]
    if family not in FAMILIES or category not in CATEGORIES:
        raise ValueError("Unknown model/category.")
    _hash(provenance["git_commit"], 40)
    _hash(provenance["checkpoint_sha256"])
    _hash(provenance["model_config_sha256"])
    if provenance.get("split") != "val":
        raise ValueError("Calibration split must be val.")
    if provenance.get("score_definition") != SCORE_DEFINITIONS[family]:
        raise ValueError("Score definition differs from the frozen protocol.")
    expected_transform = {"profile": family, "size": [256, 256]}
    if family == "patchcore":
        expected_transform.update(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if provenance.get("preprocessing") != expected_transform:
        raise ValueError("Preprocessing differs from the frozen protocol.")
    # Defensive block for the explicitly superseded scientific experiment.
    if family == "patchcore" and category == "hazelnut":
        if (provenance["source_experiment"] != "patchcore-hazelnut/EXP-002"
                or provenance["checkpoint_sha256"] !=
                "7b4c3784434d69724e8e7fb74d587e4c8a059faa24819dd556c224d03e580a93"):
            raise ValueError("Only corrected Hazelnut EXP-002 is accepted.")
    # Also reject non-JSON / nonfinite nested provenance.
    _json(_bytes(provenance))


def calibrate(samples, *, manifest_path, protocol_path, expected_manifest_sha256,
              expected_protocol_sha256, expected_count, provenance):
    """Compute the predeclared q=.95 linear quantile on complete normal Validation."""
    load_protocol(protocol_path, expected_protocol_sha256)
    _provenance(provenance)
    paths = validation_paths(manifest_path, category=provenance["category"],
                             expected_sha256=expected_manifest_sha256, expected_count=expected_count)
    ordered = _samples(samples, paths)
    scores = np.array([s.score for s in ordered], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            quantiles = {str(q): float(np.quantile(scores, q, method="linear"))
                         for q in (0., .05, .25, .5, .75, .95, 1.)}
            stats = {"minimum": float(scores.min()), "maximum": float(scores.max()),
                     "mean": float(scores.mean()), "median": quantiles["0.5"],
                     "std": float(scores.std(ddof=0)), "std_ddof": 0, "quantiles": quantiles}
        except FloatingPointError as exc:
            raise ValueError("Distribution arithmetic exceeds float64 range.") from exc
    threshold = quantiles["0.95"]
    report = {
        "schema_version": "1.0", "status": "calibrated", "n_images": len(ordered),
        "threshold": threshold, "threshold_hex": threshold.hex(),
        "policy": {"quantile": .95, "method": "linear", "decision_rule": "score > threshold",
                   "dtype": "float64", "split": "val", "label": 0},
        "distribution": stats, "provenance": dict(provenance),
        "protocol_sha256": expected_protocol_sha256, "manifest_sha256": expected_manifest_sha256,
        "validation_paths_sha256": hashlib.sha256(
            ("\n".join(s.image_path for s in ordered)+"\n").encode("utf-8")).hexdigest(),
    }
    _bytes(report)
    return report, ordered


def _publish(output_dir, files):
    """Exclusive publication, with hashes anchored by the returned integrity digest."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _export_lock(target), tempfile.TemporaryDirectory(dir=target.parent,
                                                         prefix=".calibration-staging-") as temp:
        staging = Path(temp)
        for name, raw in files.items():
            (staging/name).write_bytes(raw)
        integrity = {"schema_version": "1.0", "files": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
        (staging/"integrity.json").write_bytes(_bytes(integrity))
        digest = file_sha256(staging/"integrity.json")
        if target.exists():
            raise FileExistsError(target)
        staging.rename(target)
    return digest


def export_calibration(output_dir, samples, **kwargs):
    """Validate and compute before writing; return integrity.json SHA-256."""
    report, ordered = calibrate(samples, **kwargs)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["image_path", "label", "score"])
    writer.writerows((s.image_path, s.label, repr(s.score)) for s in ordered)
    return _publish(output_dir, {"calibration.json": _bytes(report),
                                 "validation_scores.csv": stream.getvalue().encode("utf-8")})


@dataclass(frozen=True)
class FrozenThreshold:
    model_family: str
    category: str
    threshold: float
    checkpoint_sha256: str
    protocol_sha256: str
    manifest_sha256: str
    calibration_integrity_sha256: str

    def __post_init__(self):
        if self.model_family not in FAMILIES or self.category not in CATEGORIES:
            raise ValueError("Unknown frozen-threshold instance.")
        object.__setattr__(self, "threshold", _real(self.threshold))
        for name in ("checkpoint_sha256", "protocol_sha256", "manifest_sha256",
                     "calibration_integrity_sha256"):
            _hash(getattr(self, name))

    def predict(self, scores):
        """Use the stored binary64 threshold; never recalibrate."""
        values = np.asarray(scores)
        if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuf":
            raise ValueError("Expected a nonempty real score vector.")
        values = values.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Scores must be finite.")
        return (values > self.threshold).astype(np.uint8)


def _read_bundle(directory, expected_integrity_sha256, filenames):
    directory = Path(directory)
    integrity = _json(_verified_bytes(directory/"integrity.json", expected_integrity_sha256))
    if integrity.get("schema_version") != "1.0" or set(integrity.get("files", {})) != set(filenames):
        raise ValueError("Invalid integrity manifest.")
    return {name: _verified_bytes(directory/name, integrity["files"][name]) for name in filenames}


def load_threshold(directory, *, expected_integrity_sha256, manifest_path, protocol_path,
                   expected_manifest_sha256, expected_protocol_sha256, expected_count,
                   model_family, category, expected_checkpoint_sha256):
    """Verify an externally anchored bundle and load its exact threshold, without fitting."""
    load_protocol(protocol_path, expected_protocol_sha256)
    files = _read_bundle(directory, expected_integrity_sha256,
                         ("calibration.json", "validation_scores.csv"))
    report = _json(files["calibration.json"])
    prov = report["provenance"]
    _provenance(prov)
    if (report.get("schema_version") != "1.0" or report.get("status") != "calibrated"
            or prov["model_family"] != model_family or prov["category"] != category
            or prov["checkpoint_sha256"] != _hash(expected_checkpoint_sha256)
            or report["protocol_sha256"] != expected_protocol_sha256
            or report["manifest_sha256"] != expected_manifest_sha256
            or report["n_images"] != expected_count):
        raise ValueError("Threshold identity/provenance mismatch.")
    paths = validation_paths(manifest_path, category=category,
                             expected_sha256=expected_manifest_sha256, expected_count=expected_count)
    reader = csv.DictReader(io.StringIO(files["validation_scores.csv"].decode("utf-8"), newline=""))
    if reader.fieldnames != ["image_path", "label", "score"]:
        raise ValueError("Invalid score CSV schema.")
    samples = []
    for row in reader:
        if set(row) != {"image_path", "label", "score"} or row["label"] != "0":
            raise ValueError("Invalid score CSV row.")
        samples.append(CalibrationSample(row["image_path"], 0, float(row["score"])))
    ordered = _samples(samples, paths)
    digest = hashlib.sha256(("\n".join(s.image_path for s in ordered)+"\n").encode()).hexdigest()
    if digest != report["validation_paths_sha256"]:
        raise ValueError("Validation path fingerprint mismatch.")
    if report["policy"] != {"quantile": .95, "method": "linear", "decision_rule": "score > threshold",
                             "dtype": "float64", "split": "val", "label": 0}:
        raise ValueError("Threshold policy mismatch.")
    threshold = _real(report["threshold"])
    if threshold.hex() != report["threshold_hex"]:
        raise ValueError("Threshold representation mismatch.")
    return FrozenThreshold(model_family, category, threshold, expected_checkpoint_sha256,
                           expected_protocol_sha256, expected_manifest_sha256,
                           expected_integrity_sha256)


def export_threshold_registry(output_dir, thresholds):
    """Publish six already-loaded thresholds. Never compute a quantile."""
    items = list(thresholds)
    if any(not isinstance(t, FrozenThreshold) for t in items):
        raise TypeError("Expected verified FrozenThreshold objects.")
    keys = [(t.model_family, t.category) for t in items]
    if len(keys) != 6 or set(keys) != {(f, c) for f in FAMILIES for c in CATEGORIES}:
        raise ValueError("Exactly six unique instances are required.")
    if len({t.protocol_sha256 for t in items}) != 1:
        raise ValueError("All thresholds must use the same protocol.")
    for category in CATEGORIES:
        if len({t.manifest_sha256 for t in items if t.category == category}) != 1:
            raise ValueError("Both families must share the same category manifest.")
    rows = []
    for t in sorted(items, key=lambda x: (x.model_family, x.category)):
        for name in ("checkpoint_sha256", "protocol_sha256", "manifest_sha256",
                     "calibration_integrity_sha256"):
            _hash(getattr(t, name))
        rows.append({**asdict(t), "threshold": _real(t.threshold), "threshold_hex": t.threshold.hex()})
    return _publish(output_dir, {"thresholds.json": _bytes({
        "schema_version": "1.0", "decision_rule": "score > threshold", "thresholds": rows})})


def load_threshold_registry(directory, *, expected_integrity_sha256, expected_protocol_sha256):
    """Load the six frozen thresholds without access to calibration or Test scores."""
    _hash(expected_protocol_sha256)
    raw = _read_bundle(directory, expected_integrity_sha256, ("thresholds.json",))
    report = _json(raw["thresholds.json"])
    if report.get("schema_version") != "1.0" or report.get("decision_rule") != "score > threshold":
        raise ValueError("Invalid threshold registry.")
    result = {}
    for entry in report["thresholds"]:
        entry = dict(entry)
        value = _real(entry["threshold"])
        if value.hex() != entry.pop("threshold_hex"):
            raise ValueError("Threshold precision mismatch.")
        if entry["protocol_sha256"] != expected_protocol_sha256:
            raise ValueError("Protocol mismatch.")
        for name in ("checkpoint_sha256", "protocol_sha256", "manifest_sha256",
                     "calibration_integrity_sha256"):
            _hash(entry[name])
        entry["threshold"] = value
        t = FrozenThreshold(**entry)
        key = (t.model_family, t.category)
        if key in result:
            raise ValueError("Duplicate threshold.")
        result[key] = t
    if set(result) != {(f, c) for f in FAMILIES for c in CATEGORIES}:
        raise ValueError("Incomplete six-instance registry.")
    return result
