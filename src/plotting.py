"""CSV plotting helpers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_plots(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    matrix = df[df["workload"] == "matrix_multiplication"].copy()
    if matrix.empty:
        print("No matrix multiplication rows available; skipping plots.")
        return

    # Plot mean latency by device and size.
    pivot = matrix.pivot_table(index="size", columns="device", values="latency_ms_mean", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Matrix Multiplication: Mean Latency")
    ax.set_xlabel("Matrix dimension N (N x N)")
    ax.set_ylabel("Latency (ms; lower is better)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_latency.png", dpi=160)
    plt.close()

    # Compute speedup only where both CPU and CUDA measurements exist.
    cpu = matrix[matrix["device"] == "cpu"].set_index("size")["latency_ms_mean"]
    gpu = matrix[matrix["device"] == "cuda"].set_index("size")["latency_ms_mean"]
    common = cpu.index.intersection(gpu.index)
    if len(common):
        speedup = cpu.loc[common] / gpu.loc[common]
        ax = speedup.plot(kind="bar", figsize=(8, 4))
        ax.set_title("Observed GPU Speedup (CPU latency / GPU latency)")
        ax.set_xlabel("Matrix dimension N (N x N)")
        ax.set_ylabel("Speedup (x; higher is faster)")
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_speedup.png", dpi=160)
        plt.close()
    else:
        print("CUDA results absent; speedup plot skipped.")
