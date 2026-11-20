#include "config.hpp"
#include "crr.hpp"
#include <cstdio>
#include <vector>

int main() {
    const int step_counts[] = {100, 500, 1000, 2000, 5000, 10000, 20000};

    std::printf("American Put Pricing using CRR Binomial Tree\n");
    std::printf("Sequential CPU\n\n");
    std::printf("Benchmark repetitions: %d\n\n", aophpc::BENCHMARK_REPETITIONS);
    std::printf("%8s %14s %12s\n", "N", "Price", "Median(ms)");

    for (int steps : step_counts) {
        std::vector<double> timings;
        double price = 0.0;
        for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
            const auto start = aophpc::Clock::now();
            price = aophpc::crr_price_cpu(steps);
            const auto stop = aophpc::Clock::now();
            timings.push_back(aophpc::elapsed_ms(start, stop));
        }
        std::printf("%8d %14.6f %12.3f\n", steps, price, aophpc::median(timings));
    }
    return 0;
}
