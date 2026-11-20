#include "config.hpp"
#include "crr.hpp"
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

__global__ void initialize_terminal_kernel(double* values, int steps,
                                           double spot, double strike,
                                           double log_u) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j <= steps) {
        const double stock = spot * exp((2.0 * j - steps) * log_u);
        values[j] = fmax(strike - stock, 0.0);
    }
}

__global__ void backward_step_kernel(const double* input, double* output,
                                     int level, double spot, double strike,
                                     double log_u, double p, double discount) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j <= level) {
        const double stock = spot * exp((2.0 * j - level) * log_u);
        const double continuation = discount *
            (p * input[j + 1] + (1.0 - p) * input[j]);
        const double exercise = fmax(strike - stock, 0.0);
        output[j] = fmax(exercise, continuation);
    }
}

double crr_price_cuda(int steps, double& total_ms) {
    double* d_a = nullptr;
    double* d_b = nullptr;
    const std::size_t bytes = (static_cast<std::size_t>(steps) + 1) * sizeof(double);

    const auto start = aophpc::Clock::now();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));

    const double dt = aophpc::MATURITY / steps;
    const double log_u = aophpc::VOLATILITY * std::sqrt(dt);
    const double u = std::exp(log_u);
    const double d = 1.0 / u;
    const double p = (std::exp(aophpc::RISK_FREE_RATE * dt) - d) / (u - d);
    const double discount = std::exp(-aophpc::RISK_FREE_RATE * dt);

    constexpr int block_size = 256;
    const int blocks = (steps + 1 + block_size - 1) / block_size;

    initialize_terminal_kernel<<<blocks, block_size>>>(
        d_a, steps, aophpc::SPOT, aophpc::STRIKE, log_u);
    CUDA_CHECK(cudaGetLastError());

    double* input = d_a;
    double* output = d_b;

    // Each time layer is a separate kernel launch. The launch boundary
    // provides the global synchronization needed between dependent layers.
    for (int level = steps - 1; level >= 0; --level) {
        const int level_blocks = (level + 1 + block_size - 1) / block_size;
        backward_step_kernel<<<level_blocks, block_size>>>(
            input, output, level, aophpc::SPOT, aophpc::STRIKE,
            log_u, p, discount);
        CUDA_CHECK(cudaGetLastError());
        double* temp = input;
        input = output;
        output = temp;
    }

    double price = 0.0;
    CUDA_CHECK(cudaMemcpy(&price, input, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

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

    const int step_counts[] = {1000, 2000, 5000, 10000, 20000};

    std::printf("American Put Pricing using CRR Binomial Tree\n");
    std::printf("Sequential CPU vs CUDA GPU\n\n");
    std::printf("GPU: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    std::printf("Threads per CUDA block: 256\n");
    std::printf("Benchmark repetitions: %d\n\n", aophpc::BENCHMARK_REPETITIONS);
    std::printf("%8s %14s %12s %12s %12s %12s %12s\n",
                "N", "CPU Price", "CUDA Price", "CPU(ms)",
                "GPU Total", "Speedup", "Price Error");

    for (int steps : step_counts) {
        std::vector<double> cpu_times;
        double cpu_price = 0.0;
        for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
            const auto start = aophpc::Clock::now();
            cpu_price = aophpc::crr_price_cpu(steps);
            const auto stop = aophpc::Clock::now();
            cpu_times.push_back(aophpc::elapsed_ms(start, stop));
        }
        const double cpu_ms = aophpc::median(cpu_times);

        double warmup_ms = 0.0;
        (void)crr_price_cuda(steps, warmup_ms);

        std::vector<double> gpu_times;
        double gpu_price = 0.0;
        for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
            double gpu_ms = 0.0;
            gpu_price = crr_price_cuda(steps, gpu_ms);
            gpu_times.push_back(gpu_ms);
        }
        const double gpu_ms = aophpc::median(gpu_times);
        const double error = std::abs(cpu_price - gpu_price);
        const double speedup = cpu_ms / gpu_ms;

        std::printf("%8d %14.6f %12.6f %12.3f %12.3f %11.2fx %12.3e\n",
                    steps, cpu_price, gpu_price, cpu_ms, gpu_ms, speedup, error);
        std::printf("  Correctness: %s\n\n", error < 1e-8 ? "PASSED" : "CHECK REQUIRED");
    }
    return 0;
}
