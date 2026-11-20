# american-option-pricing-hpc
Comparative implementation and performance analysis of American option pricing using CRR binomial trees and Longstaff–Schwartz Monte Carlo across sequential CPU, OpenMP, and CUDA architectures.

## Overview

American options can be exercised at any time before expiry. This early-exercise feature makes their valuation more challenging than that of European options, for which closed-form pricing formulas exist.

This project implements two numerical methods for pricing American put options and investigates how their computational workloads can be parallelized across multicore CPUs and GPUs.

## Methods

### 1. Cox-Ross-Rubinstein (CRR) Binomial Tree

The CRR model represents stock-price movements using a recombining binomial tree, where the stock price can move up or down at each time step.

Option values are computed using backward induction. At each node, the option value is the maximum of:

- **Continuation value:** The discounted expected value of holding the option.
- **Exercise value:** The payoff obtained by exercising the option immediately.

This comparison at every node allows the model to account for early exercise.

### 2. Longstaff-Schwartz Monte Carlo (LSM)

The Longstaff-Schwartz method uses Monte Carlo simulation to generate multiple possible stock-price paths.

Working backward through the exercise dates, it estimates the continuation value using least-squares regression on in-the-money paths. This estimate is compared with the immediate exercise payoff to determine whether the option should be exercised or held.

## Implementations

| Implementation | Description |
|---|---|
| Sequential C++ | Single-threaded baseline and correctness reference |
| OpenMP | Shared-memory parallelism across CPU threads |
| CUDA | GPU parallelism using CUDA kernels |

Each pricing method has three implementations, allowing their computational performance and parallelization strategies to be compared.

The implementations produce matching CRR prices to floating-point precision and matching LSM prices for the same simulated paths, subject to floating-point differences.

## Project Structure

```text
american-option-pricing-hpc/
├── README.md
├── requirements.txt
└── src/
    ├── crr_cpu.cpp
    ├── crr_omp.cpp
    ├── crr_cuda.cu
    ├── lsm_cpu.cpp
    ├── lsm_omp.cpp
    ├── lsm_cuda.cu
└── results/
    ├── benchmark tables
    └── performance plots
```

## Experimental Setup

The primary benchmark configuration uses the following financial parameters:

| Parameter | Value |
|---|---:|
| Initial stock price (S₀) | 100 |
| Strike price (K) | 100 |
| Risk-free interest rate (r) | 5% |
| Volatility (σ) | 20% |
| Time to maturity (T) | 1 year |

**Computational workloads**

| Method | Workload |
|---|---|
| CRR | 1,000, 2,000, 5,000, 10,000 and 20,000 tree steps |
| LSM | 10,000, 50,000, 100,000 and 500,000 simulated paths |
| LSM exercise dates | 50 |

**Hardware and benchmarking**

- GPU: NVIDIA Tesla T4 (Compute Capability 7.5)
- CPU: Google Colab runtime with 2 CPU cores
- CUDA block size: 256 threads
- CUDA benchmark repetitions: 5
- CUDA timings: Median execution time across repetitions

## Results

### 1. CRR: Sequential CPU vs CUDA

| Tree Steps | CPU (ms) | CUDA (ms) | Speedup |
|---:|---:|---:|---:|
| 1,000 | 19.113 | 17.548 | 1.09× |
| 2,000 | 70.849 | 32.103 | 2.21× |
| 5,000 | 348.839 | 53.545 | 6.51× |
| 10,000 | 1,195.197 | 106.762 | 11.19× |
| 20,000 | 4,951.448 | 286.583 | **17.28×** |

### 2. LSM: Sequential CPU vs CUDA

| Simulated Paths | CPU (ms) | CUDA (ms) | Speedup |
|---:|---:|---:|---:|
| 10,000 | 20.894 | 7.728 | 2.70× |
| 50,000 | 131.456 | 31.654 | 4.15× |
| 100,000 | 212.567 | 42.811 | 4.97× |
| 500,000 | 1,413.360 | 201.288 | **7.02×** |

### 3. OpenMP Performance

OpenMP was evaluated using different CPU thread counts.

- **CRR:** Best observed speedup was 1.84× using two threads at 5,000 tree steps.
- **LSM:** Best observed speedup was 1.32× using two threads at 500,000 simulated paths.

Performance varied with workload size, with parallelization overhead affecting smaller workloads.

### Performance Analysis

The CUDA implementations demonstrate increasing performance benefits as the computational workload grows. CRR benefits from parallel evaluation of independent tree nodes, while LSM benefits from parallel processing of simulated paths.

OpenMP improvements were more modest and workload-dependent, partly due to the limited number of available CPU cores and parallelization overhead.

### Performance Plots

Benchmark tables and performance plots are available in the [`results/`](results/) directory.

## Validation

- Compare American put prices against European put prices calculated using Black-Scholes.
- Examine price convergence as the number of binomial tree steps or Monte Carlo paths increases.
- Compare sequential, OpenMP, and CUDA implementations for numerical consistency.
- Account for Monte Carlo sampling variability when evaluating LSM estimates.

## Build and Run

**Sequential CPU:**
```bash
g++ -O3 -std=c++17 crr_cpu.cpp -o crr_cpu
./crr_cpu
```

**OpenMP:**
```bash
g++ -O3 -std=c++17 -fopenmp crr_omp.cpp -o crr_omp
./crr_omp
```

**CUDA (NVIDIA Tesla T4):**
```bash
nvcc -O3 -std=c++17 -arch=sm_75 crr_cuda.cu -o crr_cuda
./crr_cuda
```

The same build approach applies to the corresponding LSM source files.

## Technologies

C++17, OpenMP, NVIDIA CUDA, Python.
