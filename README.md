# GPUBoost — CPU vs GPU Performance Analyzer

A beginner-friendly benchmarking project for NVIDIA-oriented internship applications. It compares matrix multiplication and a small neural-network inference workload across CPU and (when available) CUDA GPU execution.

## Why this project?

NVIDIA's Hardware Spring Intern description mentions GPU-accelerated workloads, parallel algorithms, benchmarking, profiling, and performance analysis in the Compute Developer Technology and Compute Architecture teams. GPUBoost is a learning project aligned with those themes; it does **not** imply NVIDIA affiliation or guarantee selection.

## Features

- CPU matrix multiplication baseline using PyTorch
- CUDA GPU matrix multiplication benchmark when a compatible GPU is available
- Optional NumPy CPU baseline
- Warm-up and repeated measurements
- Latency, throughput (operations/second), and observed speedup
- Simple MLP inference benchmark on CPU/GPU
- CSV output and plots
- Environment report to make results reproducible
- Graceful CPU-only fallback

## Requirements

- Python 3.10+
- CPU-only: `pip install -r requirements.txt`
- For GPU support, install the PyTorch build compatible with your NVIDIA driver/CUDA setup using the official PyTorch installation selector: https://pytorch.org/get-started/locally/
- A CUDA-capable NVIDIA GPU is optional. Without one, the CPU benchmark still works.

## Setup (Windows PowerShell)

```powershell
cd GPUBoost
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## Run

```powershell
python main.py --sizes 128 256 512 --repeats 5 --warmup 2
```

Run a larger benchmark if your machine has enough RAM and time:

```powershell
python main.py --sizes 256 512 1024 2048 --repeats 5 --warmup 2
```

Optional neural-network inference benchmark:

```powershell
python main.py --sizes 128 256 512 --repeats 5 --warmup 2 --include-mlp
```

Results are written to `results/`:
- `benchmark_results.csv`
- `benchmark_latency.png`
- `benchmark_speedup.png`
- `environment.json`

**Important:** Run on an idle machine when possible. Results depend on hardware, power mode, software versions, memory, and GPU availability. Do not put invented speedup numbers on your resume; use only measurements from your own run.

## Project structure

```text
GPUBoost/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── benchmark.py
│   └── plotting.py
├── tests/
│   └── test_math.py
└── results/
```

## What the metrics mean

- **Latency (ms):** Average time for one operation or one batch.
- **Throughput (GFLOP/s):** Estimated floating-point operations per second for square matrix multiplication. This is a rough benchmark metric and excludes some overheads.
- **Speedup:** CPU latency divided by GPU latency for the same matrix size. It is reported only if both measurements exist. Small tasks can be slower on GPU due to launch and data-transfer overhead.
- **Correctness:** CPU and GPU outputs are compared within a numerical tolerance.

## Limitations

- This starter version uses PyTorch operations, which call optimized backend kernels; it is not yet a custom CUDA kernel.
- CPU and GPU hardware/software are not identical, so results are environment-specific.
- The MLP benchmark is a small synthetic workload, not a trained model evaluation.
- GPU timing synchronizes CUDA to avoid under-reporting asynchronous execution time.

## Next improvements

1. Add a custom CUDA C++ kernel for vector addition or matrix multiplication.
2. Profile memory transfer vs kernel execution separately.
3. Compare FP32 and mixed precision where supported.
4. Add unit tests and a short screen-recorded demo.
