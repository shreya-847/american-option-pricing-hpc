#include "config.hpp"
#include "lsm.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

namespace {

constexpr int BLOCK_SIZE = 256;

__global__ void initialize_kernel(const double* paths, double* cashflow,
                                  int* exercise_time, int path_count,
                                  int steps, double strike) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < path_count) {
        const double terminal_stock = paths[static_cast<std::size_t>(i) * (steps + 1) + steps];
        cashflow[i] = fmax(strike - terminal_stock, 0.0);
        exercise_time[i] = steps;
    }
}

// stats = [count, sum(S), sum(S^2), sum(S^3), sum(S^4),
//          sum(y), sum(S*y), sum(S^2*y), reserved]
__global__ void regression_statistics_kernel(
    const double* paths, const double* cashflow, const int* exercise_time,
    double* stats, int path_count, int steps, int time_step,
    double discount, double strike) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= path_count) return;

    const double stock = paths[static_cast<std::size_t>(i) * (steps + 1) + time_step];
    const double immediate_exercise = fmax(strike - stock, 0.0);
    if (immediate_exercise <= 0.0) return;

    const double x = stock;
    const double x2 = x * x;
    const double y = cashflow[i] * pow(discount, exercise_time[i] - time_step);

    // Global double-precision atomic additions are supported on compute
    // capability 6.0 and later (including the Tesla T4, CC 7.5).
    atomicAdd(&stats[0], 1.0);
    atomicAdd(&stats[1], x);
    atomicAdd(&stats[2], x2);
    atomicAdd(&stats[3], x2 * x);
    atomicAdd(&stats[4], x2 * x2);
    atomicAdd(&stats[5], y);
    atomicAdd(&stats[6], x * y);
    atomicAdd(&stats[7], x2 * y);
}

__global__ void solve_regression_kernel(const double* stats,
                                        double* coeff, int* valid) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double matrix[3][4] = {
        {stats[0], stats[1], stats[2], stats[5]},
        {stats[1], stats[2], stats[3], stats[6]},
        {stats[2], stats[3], stats[4], stats[7]}
    };

    if (stats[0] < 3.0) {
        *valid = 0;
        return;
    }

    // Gaussian elimination with partial pivoting.
    for (int i = 0; i < 3; ++i) {
        int pivot = i;
        for (int j = i + 1; j < 3; ++j) {
            if (fabs(matrix[j][i]) > fabs(matrix[pivot][i])) pivot = j;
        }
        if (fabs(matrix[pivot][i]) < 1e-12) {
            *valid = 0;
            return;
        }
        for (int j = 0; j < 4; ++j) {
            const double temp = matrix[i][j];
            matrix[i][j] = matrix[pivot][j];
            matrix[pivot][j] = temp;
        }
        const double divisor = matrix[i][i];
        for (int j = i; j < 4; ++j) matrix[i][j] /= divisor;
        for (int j = 0; j < 3; ++j) {
            if (j == i) continue;
            const double factor = matrix[j][i];
            for (int k = i; k < 4; ++k)
                matrix[j][k] -= factor * matrix[i][k];
        }
    }

    coeff[0] = matrix[0][3];
    coeff[1] = matrix[1][3];
    coeff[2] = matrix[2][3];
    *valid = 1;
}

__global__ void exercise_decision_kernel(
    const double* paths, double* cashflow, int* exercise_time,
    const double* coeff, const int* valid,
    int path_count, int steps, int time_step, double strike) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= path_count || *valid == 0) return;

    const double stock = paths[static_cast<std::size_t>(i) * (steps + 1) + time_step];
    const double immediate_exercise = fmax(strike - stock, 0.0);
    if (immediate_exercise <= 0.0) return;

    const double continuation = coeff[0] + coeff[1] * stock +
                                coeff[2] * stock * stock;
    if (immediate_exercise > continuation) {
        cashflow[i] = immediate_exercise;
        exercise_time[i] = time_step;
    }
}

__global__ void final_value_kernel(const double* cashflow,
                                   const int* exercise_time,
                                   double* total, int path_count,
                                   double discount) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < path_count) {
        const double value = cashflow[i] * pow(discount, exercise_time[i]);
        atomicAdd(total, value);
    }
}

double lsm_price_cuda(const std::vector<double>& host_paths,
                      int path_count, int steps, double& total_ms) {
    double* d_paths = nullptr;
    double* d_cashflow = nullptr;
    int* d_exercise_time = nullptr;
    double* d_stats = nullptr;
    double* d_coeff = nullptr;
    int* d_valid = nullptr;
    double* d_total = nullptr;

    const std::size_t path_bytes =
        static_cast<std::size_t>(path_count) * (steps + 1) * sizeof(double);
    const std::size_t cashflow_bytes = static_cast<std::size_t>(path_count) * sizeof(double);
    const std::size_t time_bytes = static_cast<std::size_t>(path_count) * sizeof(int);

    const auto start = aophpc::Clock::now();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_paths), path_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cashflow), cashflow_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_exercise_time), time_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_stats), 9 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_coeff), 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_valid), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_total), sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_paths, host_paths.data(), path_bytes, cudaMemcpyHostToDevice));

    const int blocks = (path_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const double dt = aophpc::MATURITY / steps;
    const double discount = std::exp(-aophpc::RISK_FREE_RATE * dt);

    initialize_kernel<<<blocks, BLOCK_SIZE>>>(
        d_paths, d_cashflow, d_exercise_time, path_count, steps, aophpc::STRIKE);
    CUDA_CHECK(cudaGetLastError());

    // Backward induction. A kernel boundary orders the regression, solve,
    // and exercise-decision phases for each time step in the default stream.
    for (int t = steps - 1; t >= 1; --t) {
        CUDA_CHECK(cudaMemset(d_stats, 0, 9 * sizeof(double)));

        regression_statistics_kernel<<<blocks, BLOCK_SIZE>>>(
            d_paths, d_cashflow, d_exercise_time, d_stats,
            path_count, steps, t, discount, aophpc::STRIKE);
        CUDA_CHECK(cudaGetLastError());

        solve_regression_kernel<<<1, 1>>>(d_stats, d_coeff, d_valid);
        CUDA_CHECK(cudaGetLastError());

        exercise_decision_kernel<<<blocks, BLOCK_SIZE>>>(
            d_paths, d_cashflow, d_exercise_time, d_coeff, d_valid,
            path_count, steps, t, aophpc::STRIKE);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(double)));
    final_value_kernel<<<blocks, BLOCK_SIZE>>>(
        d_cashflow, d_exercise_time, d_total, path_count, discount);
    CUDA_CHECK(cudaGetLastError());

    double total = 0.0;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    const double price = total / path_count;

    CUDA_CHECK(cudaFree(d_paths));
    CUDA_CHECK(cudaFree(d_cashflow));
    CUDA_CHECK(cudaFree(d_exercise_time));
    CUDA_CHECK(cudaFree(d_stats));
    CUDA_CHECK(cudaFree(d_coeff));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_total));

    const auto stop = aophpc::Clock::now();
    total_ms = aophpc::elapsed_ms(start, stop);
    return price;
}

} // namespace

int main() {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    constexpr int steps = 50;
    const int path_counts[] = {10000, 50000, 100000, 500000};

    std::printf("American Put Pricing using LSM Monte Carlo\n");
    std::printf("Sequential CPU vs CUDA GPU\n\n");
    std::printf("GPU: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    std::printf("Time steps: %d\n", steps);
    std::printf("Threads per CUDA block: %d\n", BLOCK_SIZE);
    std::printf("Benchmark repetitions: %d\n\n", aophpc::BENCHMARK_REPETITIONS);
    std::printf("%10s %14s %12s %12s %12s %12s\n",
                "Paths", "CPU Price", "CUDA Price", "CPU(ms)", "GPU Total", "Speedup");

    for (int path_count : path_counts) {
        const auto paths = aophpc::generate_lsm_paths(path_count, steps);

        std::vector<double> cpu_times;
        double cpu_price = 0.0;
        for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
            const auto start = aophpc::Clock::now();
            cpu_price = aophpc::lsm_price_cpu(paths, path_count, steps);
            const auto stop = aophpc::Clock::now();
            cpu_times.push_back(aophpc::elapsed_ms(start, stop));
        }
        const double cpu_ms = aophpc::median(cpu_times);

        // Warm-up is excluded from the reported GPU median.
        double warmup_ms = 0.0;
        (void)lsm_price_cuda(paths, path_count, steps, warmup_ms);

        std::vector<double> gpu_times;
        double gpu_price = 0.0;
        for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
            double elapsed = 0.0;
            gpu_price = lsm_price_cuda(paths, path_count, steps, elapsed);
            gpu_times.push_back(elapsed);
        }
        const double gpu_ms = aophpc::median(gpu_times);
        const double error = std::abs(cpu_price - gpu_price);
        const double speedup = cpu_ms / gpu_ms;

        std::printf("%10d %14.6f %12.6f %12.3f %12.3f %11.2fx\n",
                    path_count, cpu_price, gpu_price, cpu_ms, gpu_ms, speedup);
        std::printf("  Price error: %.3e | %s\n\n",
                    error, error < 1e-8 ? "CORRECTNESS PASSED" : "CHECK REQUIRED");
    }
    return 0;
}
