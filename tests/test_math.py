"""Small correctness tests for benchmark helpers."""
import torch

from src.benchmark import matrix_multiplication_benchmark


def test_matrix_benchmark_returns_expected_fields():
    row = matrix_multiplication_benchmark(
        size=8, device=torch.device("cpu"), warmup=0, repeats=1
    )
    assert row["workload"] == "matrix_multiplication"
    assert row["device"] == "cpu"
    assert row["size"] == 8
    assert row["latency_ms_mean"] >= 0
    assert row["max_abs_error"] < 1e-4
