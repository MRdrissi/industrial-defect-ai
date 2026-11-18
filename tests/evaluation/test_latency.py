import math
import pytest

from src.evaluation.latency import (
    measure_inference,
    benchmark_latency,
)


def test_measure_inference_returns_prediction():
    result, elapsed = measure_inference(lambda: 42)

    assert result == 42
    assert elapsed >= 0
    assert math.isfinite(elapsed)


def test_measure_inference_synchronizes():
    events = []

    def sync():
        events.append("sync")

    def predict():
        events.append("predict")
        return "ok"

    result, _ = measure_inference(
        predict,
        synchronize=sync
    )

    assert result == "ok"
    assert events == ["sync", "predict", "sync"]


def test_benchmark_latency_statistics():
    report = benchmark_latency(
        lambda: sum(range(1000)),
        warmup=2,
        repetitions=5
    )

    assert report["unit"] == "ms"
    assert report["warmup_runs"] == 2
    assert report["measured_runs"] == 5

    values = report["latencies_ms"]

    assert len(values) == 5
    assert all(math.isfinite(v) and v >= 0 for v in values)

    assert report["min_ms"] <= report["median_ms"]
    assert report["median_ms"] <= report["max_ms"]
    assert report["min_ms"] <= report["p95_ms"]
    assert report["p95_ms"] <= report["max_ms"]


def test_warmup_excluded_from_measurements():
    calls = []

    def predict():
        calls.append(1)

    report = benchmark_latency(
        predict,
        warmup=3,
        repetitions=4
    )

    assert len(calls) == 7
    assert len(report["latencies_ms"]) == 4


def test_gpu_synchronization_callback():
    calls = []

    def sync():
        calls.append("sync")

    report = benchmark_latency(
        lambda: None,
        warmup=2,
        repetitions=3,
        synchronize=sync
    )

    # One sync per warm-up, two per measurement.
    assert len(calls) == 2 + 2 * 3
    assert report["measured_runs"] == 3


@pytest.mark.parametrize(
    "warmup,repetitions",
    [
        (-1, 5),
        (0, 0),
        (2, -3),
        (1.5, 5),
        (False, 5),
        (0, True),
    ]
)
def test_invalid_benchmark_parameters(warmup, repetitions):
    with pytest.raises(ValueError):
        benchmark_latency(
            lambda: None,
            warmup=warmup,
            repetitions=repetitions
        )


def test_non_callable_prediction_rejected():
    with pytest.raises(TypeError):
        measure_inference(123)


def test_invalid_synchronizer_rejected():
    with pytest.raises(TypeError):
        measure_inference(
            lambda: None,
            synchronize=123
        )


def test_measured_interval_includes_completion_not_initial_sync(monkeypatch):
    from src.evaluation import latency
    events = []
    ticks = iter([10., 10.125])
    def clock():
        events.append("clock")
        return next(ticks)
    def sync():
        events.append("sync")
    def predict():
        events.append("predict")
        return "result"
    monkeypatch.setattr(latency.time, "perf_counter", clock)
    result, elapsed = measure_inference(predict, synchronize=sync)
    assert result == "result"
    assert elapsed == 125.
    assert events == ["sync", "clock", "predict", "sync", "clock"]
