"""Benchmark helpers for GPUBoost."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so timing includes actual CUDA execution."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_run(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, float]:
    """Return average, median, min latency in milliseconds."""
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "latency_ms_mean": float(np.mean(samples_ms)),
        "latency_ms_median": float(np.median(samples_ms)),
        "latency_ms_min": float(np.min(samples_ms)),
    }


def matrix_multiplication_benchmark(
    size: int,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, float | int | str]:
    """Benchmark A @ B for two size x size FP32 matrices."""
    generator = torch.Generator(device="cpu").manual_seed(42)
    a_cpu = torch.randn((size, size), generator=generator, dtype=torch.float32)
    b_cpu = torch.randn((size, size), generator=generator, dtype=torch.float32)

    if device.type == "cuda":
        a = a_cpu.to(device)
        b = b_cpu.to(device)
    else:
        a, b = a_cpu, b_cpu

    # One untimed result for correctness/reference and operation setup.
    result = a @ b
    synchronize(device)
    reference = a_cpu @ b_cpu
    if device.type == "cuda":
        reference = reference.to(device)
    max_abs_error = float((result - reference).abs().max().item())

    timings = timed_run(lambda: a @ b, device, warmup, repeats)
    # A square matrix multiplication is approximately 2*N^3 floating-point ops.
    gflops = (2.0 * (size ** 3)) / (timings["latency_ms_mean"] / 1000.0) / 1e9

    return {
        "workload": "matrix_multiplication",
        "device": str(device),
        "size": size,
        **timings,
        "throughput_gflops": gflops,
        "max_abs_error": max_abs_error,
    }


def mlp_inference_benchmark(
    batch_size: int,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, float | int | str]:
    """Benchmark a randomly initialized MLP inference batch."""
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    ).eval().to(device)
    x = torch.randn(batch_size, 128, dtype=torch.float32, device=device)

    with torch.inference_mode():
        timings = timed_run(lambda: model(x), device, warmup, repeats)

    return {
        "workload": "mlp_inference",
        "device": str(device),
        "size": batch_size,
        **timings,
        "throughput_samples_per_sec": batch_size / (timings["latency_ms_mean"] / 1000.0),
        "throughput_gflops": float("nan"),
        "max_abs_error": float("nan"),
    }
