"""Reproducible inference latency measurements."""

import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def measure_inference(
        predict: Callable[[], T],
        *,
        synchronize: Callable[[], None] | None = None,
) -> tuple[T, float]:
    """Execute one prediction and measure latency in milliseconds.

    The caller defines the measured scope through `predict`.
    For CUDA, supply a synchronization callback.
    """
    if not callable(predict):
        raise TypeError("predict must be callable.")

    if synchronize is not None and not callable(synchronize):
        raise TypeError("synchronize must be callable.")

    # Ensure previous device operations have completed.
    if synchronize is not None:
        synchronize()

    start = time.perf_counter()

    result = predict()

    # CUDA operations may be asynchronous.
    if synchronize is not None:
        synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return result, float(elapsed_ms)


def benchmark_latency(
        predict: Callable[[], object],
        *,
        warmup: int = 3,
        repetitions: int = 10,
        synchronize: Callable[[], None] | None = None,
) -> dict:
    """Measure repeated inference latency.

    Warm-up executions are excluded from statistics.
    The function does not include data loading unless
    explicitly included by the caller.
    """
    if not callable(predict):
        raise TypeError("predict must be callable.")

    if synchronize is not None and not callable(synchronize):
        raise TypeError("synchronize must be callable.")

    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer.")

    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")

    # Warm-up without recording latency.
    for _ in range(warmup):
        predict()

        if synchronize is not None:
            synchronize()

    latencies = []

    for _ in range(repetitions):
        _, latency_ms = measure_inference(
            predict,
            synchronize=synchronize,
        )
        latencies.append(latency_ms)

    values = np.asarray(latencies, dtype=np.float64)

    return {
        "unit": "ms",
        "warmup_runs": warmup,
        "measured_runs": repetitions,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "latencies_ms": values.tolist(),
    }
