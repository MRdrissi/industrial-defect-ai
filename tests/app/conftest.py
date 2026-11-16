"""Interface testée sans charger les checkpoints ni déclencher d'inférence réelle."""

from pathlib import Path

import pytest
import torch
from streamlit.testing.v1 import AppTest

from src.inference.inference_contract import InferenceResult


@pytest.fixture
def inference_app(monkeypatch):
    calls = []

    def infer(request, registry):
        calls.append(request)
        threshold = 0.2 if request.method == "cae" else 40.0
        score = 0.1 if request.method == "cae" else 42.0
        return InferenceResult(
            category=request.category, method=request.method, version=request.version,
            score=score, threshold=threshold,
            anomaly_map=torch.arange(64, dtype=torch.float32).reshape(8, 8),
            decision="normal" if score < threshold else "suspect",
            latency_ms=12.5, simulated=False,
        )

    monkeypatch.setattr("src.inference.inference_contract.run_inference", infer)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[2] / "app/streamlit_app.py"), default_timeout=20,
    ).run()
    assert not app.exception
    return app, calls
