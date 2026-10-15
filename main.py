"""GPUBoost command-line entry point."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import torch

from src.benchmark import matrix_multiplication_benchmark, mlp_inference_benchmark
from src.plotting import create_plots

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def get_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CPU and CUDA GPU performance.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512],
                        help="Matrix dimensions; use moderate sizes on low-memory machines.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repetitions per benchmark.")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up iterations.")
    parser.add_argument("--include-mlp", action="store_true", help="Also benchmark a small MLP.")
    args = parser.parse_args()

    if any(s <= 0 for s in args.sizes):
        parser.error("All sizes must be positive integers.")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be >= 1 and --warmup must be >= 0.")

    RESULTS.mkdir(exist_ok=True)
    devices = get_devices()

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": __import__("numpy").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version_reported_by_torch": torch.version.cuda,
        "devices": [str(d) for d in devices],
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sizes": args.sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    print("=" * 64)
    print("GPUBoost — CPU vs GPU Performance Analyzer")
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA GPU detected; running CPU baseline only.")
    print("=" * 64)

    rows: list[dict] = []
    for size in args.sizes:
        for device in devices:
            print(f"Benchmarking matrix multiplication | size={size} | device={device} ...")
            try:
                row = matrix_multiplication_benchmark(size, device, args.warmup, args.repeats)
                rows.append(row)
                print(f"  mean={row['latency_ms_mean']:.3f} ms | "
                      f"throughput={row['throughput_gflops']:.3f} GFLOP/s | "
                      f"max error={row['max_abs_error']:.6g}")
            except (RuntimeError, MemoryError) as exc:
                print(f"  SKIPPED: {exc}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if args.include_mlp:
        for batch_size in args.sizes:
            for device in devices:
                print(f"Benchmarking MLP inference | batch={batch_size} | device={device} ...")
                try:
                    rows.append(mlp_inference_benchmark(batch_size, device, args.warmup, args.repeats))
                except (RuntimeError, MemoryError) as exc:
                    print(f"  SKIPPED: {exc}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    if not rows:
        print("No benchmark completed. Try smaller sizes or check your PyTorch installation.")
        return 1

    csv_path = RESULTS / "benchmark_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    create_plots(csv_path, RESULTS)

    print("\nFinished.")
    print(f"CSV results: {csv_path}")
    print(f"Environment: {RESULTS / 'environment.json'}")
    print(f"Plots folder: {RESULTS}")
    print("Use your own measured results in reports and resumes; results vary by hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
