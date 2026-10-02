# GPU Tiling Deep Dive — GEMM Optimization in Julia

High-performance General Matrix-Matrix Multiplication (GEMM) on NVIDIA GPU using CUDA.jl.  
Benchmarks multiple shared-memory tiling strategies in **Float32** and compares against cuBLAS.

---

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4.3 GB VRAM) |
| CPU | AMD Ryzen 7 4800H (8 cores / 16 threads) |
| RAM | 15.4 GB DDR5 |
| Language | Julia 1.12 + CUDA.jl |

---

## Files

| File | Description |
|------|-------------|
| `gpu_tiling.jl` | Main tiling benchmark — compares Shmem 8×8, 16×16, 32×32, Rect 16×32, Register 2×2, and cuBLAS across matrix sizes N=2048 to N=8192 |
| `tiling_sweep.jl` | Comprehensive sweep — all tile sizes × all matrix sizes with detailed results table |
| `tiling_sweep_fast.jl` | Fast version of the sweep — reduced samples for quicker execution, includes compilation progress indicators |

---

## Tiling Strategies Compared

| Strategy | Description | Threads/block |
|----------|-------------|--------------|
| Shmem 8×8 | Square shared-memory tile | 64 |
| Shmem 16×16 | Square shared-memory tile | 256 |
| Shmem 32×32 | Square shared-memory tile (max) | 1024 |
| Rect 16×32 | Rectangular tile — more columns than rows | 512 |
| Register 2×2 | Each thread computes 2×2=4 elements using register reuse | 256 |
| cuBLAS | NVIDIA optimized library — reference ceiling | auto |

---

## Key Results (Float32)

### Tile Size Sweep (N=4096, shared memory only)

| Tile size | Time (ms) | GFLOPS |
|-----------|-----------|--------|
| 8×8 | ~430 | ~320 |
| **16×16** | **~229** | **~600** |
| 32×32 | ~276 | ~497 |

### Strategy Comparison (N=4096)

| Strategy | GFLOPS | Time (ms) | % of cuBLAS |
|----------|--------|-----------|-------------|
| Shmem 8×8 | ~320 | ~430 | 6.5% |
| Shmem 16×16 | ~600 | ~229 | 12.2% |
| Shmem 32×32 | ~497 | ~276 | 10.1% |
| Rect 16×32 | ~560 | ~249 | 11.4% |
| **Register 2×2** | **~1182** | **~116** | **24.1%** |
| **cuBLAS** | **~4890** | **~28** | **100%** |

### Strategy Comparison (N=8192)

| Strategy | GFLOPS | Time (ms) | % of cuBLAS |
|----------|--------|-----------|-------------|
| Shmem 8×8 | ~298 | ~3774 | 6.7% |
| Shmem 16×16 | ~571 | ~1973 | 12.8% |
| Shmem 32×32 | ~469 | ~2384 | 10.6% |
| **Register 2×2** | **~1090** | **~1010** | **24.0%** |
| **cuBLAS** | **~4390** | **~250** | **100%** |

> **Optimal tile:** 16×16 (256 threads/block) gives best occupancy on RTX 3050  
> **Best custom kernel:** Register 2×2 — each thread computes 4 output elements, reducing shared memory reads per result from 2 to 1

---

## Why Tiling Works

Without tiling, every thread reads directly from slow global memory (VRAM) — ~600 cycle latency.  
With shared-memory tiling, all threads in a block collaborate to load a tile once into fast on-chip shared memory (~30 cycles), then compute from there.

```
Global memory (4.3 GB, ~600 cycles latency)
       ↓  load tile once (all 256 threads cooperate)
Shared memory (48 KB per SM, ~30 cycles latency)
       ↓  compute (each thread independently)
Registers (0 cycles — inside the thread)
```

Register tiling adds a second optimization level: each thread accumulates multiple output elements in registers, reusing the values already loaded from shared memory instead of re-reading them.

### Register tile efficiency

| Register tile | Elements/thread | Shmem reads/result | Expected |
|--------------|-----------------|-------------------|---------|
| 1×1 | 1 | 2 reads | baseline |
| 2×2 | 4 | 1 read | ~2× faster |
| 2×4 | 8 | 0.75 reads | ~2.5× faster |
| 4×2 | 8 | 0.75 reads | ~2.5× faster |
| 4×4 | 16 | 0.5 reads | ~3× OR slower if registers spill |

---

## How to Run

### Prerequisites

```julia
using Pkg
Pkg.add(["CUDA", "BenchmarkTools", "Printf"])
```

Verify GPU detection:
```julia
using CUDA
CUDA.versioninfo()
```

### Run benchmarks

```bash
# Main tiling comparison (strategies × matrix sizes)
julia phase4\gpu_tiling.jl

# Full tile size sweep (large matrices, faster)
julia phase4\tiling_sweep_fast.jl
```

---

## Matrix Sizes

| Script | Sizes | Reason |
|--------|-------|--------|
| `gpu_tiling.jl` | N = 2048, 4096, 6144, 8192 | Safe within 4.3 GB VRAM |
| `tiling_sweep_fast.jl` | N = 8192, 10240, 12288, 16384 | Large enough to be compute-bound |

VRAM usage: N=8192 → 0.81 GB · N=12288 → 1.81 GB · N=16384 → 3.22 GB

---

## Hardware Constraints

- **Max threads per block:** 1024 → max square tile = **32×32**
- **Shared memory per block:** 48 KB → limits tile size
- **VRAM:** 4.3 GB → limits max matrix size in Float32
- **GPU FP32 peak:** ~9 TFLOPS (RTX 3050 Laptop)

---

## Context

This is Phase 4 of a broader 7-phase GEMM optimization project:

| Phase | Description | Best result |
|-------|-------------|-------------|
| 1 | Naive CPU baseline (triple loop) | ~4 GFLOPS (FP64) |
| 2 | CPU multi-threading (12 threads + OpenBLAS) | ~154 GFLOPS |
| 3 | Naive GPU kernel (global memory, 1 thread/element) | ~455 GFLOPS (FP32) |
| **4** | **GPU tiling ← this repo** | **~1182 GFLOPS (Register 2×2)** |
| 5 | Async data transfer (CUDA streams) | 13% gain at N=12288 |
| 6 | Low precision (FP32 → FP16 Tensor Cores via cuBLAS) | ~13,648 GFLOPS |
| 7 | Roofline analysis + final graphs | — |

**Total project speedup: 3,412× from naive CPU to FP16 Tensor Cores.**

FP16 via cuBLAS reaches 13,648 GFLOPS — **1.5× above the FP32 hardware roofline** (9 TFLOPS), confirming Tensor Core activation.

---

## Author

Anouar Braham — HPC Research Project, 2025  
Hardware: RTX 3050 Laptop GPU + Ryzen 7 4800H · Julia 1.12 + CUDA.jl
