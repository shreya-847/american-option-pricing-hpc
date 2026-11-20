# Observed benchmark results supplied for this project

These tables preserve the user's reported Colab results. They are included as historical observations, not guaranteed reproducible measurements. Timings are machine- and runtime-dependent.

## CRR: sequential CPU vs CUDA on Tesla T4

| Tree steps (N) | CPU price | CUDA price | CPU (ms) | GPU total (ms) | GPU event (ms) | Speedup | Absolute error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 6.089595 | 6.089595 | 19.113 | 17.548 | 17.263 | 1.09× | 0.00e+00 |
| 2,000 | 6.089990 | 6.089990 | 70.849 | 32.103 | 31.814 | 2.21× | 0.00e+00 |
| 5,000 | 6.090219 | 6.090219 | 348.839 | 53.545 | 53.259 | 6.51× | 2.66e-15 |
| 10,000 | 6.090295 | 6.090295 | 1,195.197 | 106.762 | 106.447 | 11.19× | 8.88e-16 |
| 20,000 | 6.090333 | 6.090333 | 4,951.448 | 286.583 | 286.260 | 17.28× | 2.66e-15 |

GPU: NVIDIA Tesla T4, compute capability 7.5. Block size: 256. Five repetitions. Speedup uses the reported GPU Total time, not GPU Event time.

## CRR: sequential CPU vs OpenMP CPU

| N | Serial (ms) | 1 thread (ms) | 2 threads (ms) | 4 threads (ms) | 8 threads (ms) |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.181 | 0.179 | 0.175 | 2.497 | 5.954 |
| 500 | 3.553 | 3.902 | 2.989 | 13.722 | 31.414 |
| 1,000 | 13.467 | 14.334 | 11.668 | 38.852 | 70.743 |
| 2,000 | 55.615 | 55.024 | 45.946 | 91.825 | 164.722 |
| 5,000 | 543.176 | 352.060 | 295.258 | 415.648 | 639.890 |
| 10,000 | 1,418.753 | 1,615.820 | 1,195.863 | 1,509.874 | 1,883.444 |
| 20,000 | 5,999.383 | 5,751.220 | 4,921.571 | 6,118.391 | 7,331.937 |

The Colab runtime reported two available OpenMP threads, while 4- and 8-thread settings were also tested. Those larger settings were oversubscribed and generally slower.

## LSM: sequential CPU vs OpenMP CPU

Time steps: 50.

| Paths | Serial price | Serial (ms) | OpenMP 1 thread (ms) | Speedup | OpenMP 2 threads (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 6.077179 | 13.309 | 15.048 | 0.88× | 10.714 | 1.24× |
| 50,000 | 6.032697 | 83.282 | 94.412 | 0.88× | 71.815 | 1.16× |
| 100,000 | 6.053391 | 182.176 | 261.001 | 0.70× | 266.039 | 0.68× |
| 500,000 | 6.054775 | 1,171.957 | 1,331.716 | 0.88× | 886.506 | 1.32× |

Sequential and OpenMP prices agreed to floating-point precision. OpenMP was not consistently faster; at 100,000 paths, the reported 2-thread run was slower than the serial baseline.

## LSM: sequential CPU vs CUDA on Tesla T4

Time steps: 50. Five repetitions. Same host-generated path matrix supplied to CPU and GPU valuation.

| Paths | CPU price | CUDA price | CPU (ms) | GPU total (ms) | Speedup | Absolute error |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 6.077179 | 6.077179 | 20.894 | 7.728 | 2.70× | 2.576e-14 |
| 50,000 | 6.032697 | 6.032697 | 131.456 | 31.654 | 4.15× | 2.931e-14 |
| 100,000 | 6.053391 | 6.053391 | 212.567 | 42.811 | 4.97× | 2.043e-14 |
| 500,000 | 6.054775 | 6.054775 | 1,413.360 | 201.288 | 7.02× | 1.332e-14 |

## Interpreting the records

- The maximum reported CUDA speedups in these runs were 17.28× for CRR at 20,000 tree steps and 7.02× for LSM at 500,000 paths.
- These are separate benchmark sessions and should not be used to rank CRR against LSM directly.
- The LSM prices vary with the simulated sample size. The price is a Monte Carlo estimate and does not, by itself, establish statistical accuracy.
- CPU and GPU price agreement validates implementation consistency for the same inputs. It is not independent validation against the exact American option value.
