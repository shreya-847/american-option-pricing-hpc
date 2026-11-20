#include "config.hpp"
#include "crr.hpp"
#include <omp.h>
#include <cstdio>
#include <vector>

namespace {

double crr_price_omp(int steps, int threads) {
    const double dt = aophpc::MATURITY / steps;
    const double log_u = aophpc::VOLATILITY * std::sqrt(dt);
    const double u = std::exp(log_u);
    const double d = 1.0 / u;
    const double p = (std::exp(aophpc::RISK_FREE_RATE * dt) - d) / (u - d);
    const double discount = std::exp(-aophpc::RISK_FREE_RATE * dt);

    std::vector<double> a(static_cast<std::size_t>(steps) + 1);
    std::vector<double> b(static_cast<std::size_t>(steps) + 1);

    for (int j = 0; j <= steps; ++j) {
        const double stock = aophpc::SPOT * std::exp((2.0 * j - steps) * log_u);
        a[j] = std::max(aophpc::STRIKE - stock, 0.0);
    }

    double* in = a.data();
    double* out = b.data();

    // A single parallel region avoids creating a new team at every tree level.
    // The worksharing-loop and single-region barriers make each pointer swap safe.
    #pragma omp parallel num_threads(threads) shared(in, out)
    {
        for (int level = steps - 1; level >= 0; --level) {
            #pragma omp for schedule(static)
            for (int j = 0; j <= level; ++j) {
                const double stock = aophpc::SPOT *
                    std::exp((2.0 * j - level) * log_u);
                const double continuation = discount *
                    (p * in[j + 1] + (1.0 - p) * in[j]);
                const double exercise = std::max(aophpc::STRIKE - stock, 0.0);
                out[j] = std::max(exercise, continuation);
            }
            #pragma omp single
            {
                double* temp = in;
                in = out;
                out = temp;
            }
        }
    }
    return in[0];
}

} // namespace

int main() {
    const int step_counts[] = {100, 500, 1000, 2000, 5000, 10000, 20000};
    const int requested_threads[] = {1, 2, 4, 8};
    const int max_threads = omp_get_max_threads();

    std::printf("American Put Pricing using CRR Binomial Tree\n");
    std::printf("Sequential CPU vs OpenMP CPU\n\n");
    std::printf("Maximum OpenMP threads available: %d\n", max_threads);
    std::printf("Benchmark repetitions: %d\n\n", aophpc::BENCHMARK_REPETITIONS);
    std::printf("%8s %10s %14s %12s %12s %12s\n",
                "N", "Threads", "Price", "Time(ms)", "Speedup", "Price Error");

    for (int steps : step_counts) {
        std::vector<double> serial_times;
        double serial_price = 0.0;
        for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
            const auto start = aophpc::Clock::now();
            serial_price = aophpc::crr_price_cpu(steps);
            const auto stop = aophpc::Clock::now();
            serial_times.push_back(aophpc::elapsed_ms(start, stop));
        }
        const double serial_ms = aophpc::median(serial_times);
        std::printf("%8d %10s %14.6f %12.3f %11.2fx %12s\n",
                    steps, "Serial", serial_price, serial_ms, 1.0, "-");

        for (int requested : requested_threads) {
            if (requested > max_threads) continue;
            std::vector<double> timings;
            double parallel_price = 0.0;
            parallel_price = crr_price_omp(steps, requested); // warm-up
            for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
                const auto start = aophpc::Clock::now();
                parallel_price = crr_price_omp(steps, requested);
                const auto stop = aophpc::Clock::now();
                timings.push_back(aophpc::elapsed_ms(start, stop));
            }
            const double parallel_ms = aophpc::median(timings);
            const double speedup = serial_ms / parallel_ms;
            const double error = std::abs(serial_price - parallel_price);
            std::printf("%8d %10d %14.6f %12.3f %11.2fx %12.3e\n",
                        steps, requested, parallel_price, parallel_ms, speedup, error);
        }
        std::printf("\n");
    }
    return 0;
}
