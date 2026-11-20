#include "config.hpp"
#include "lsm.hpp"
#include <cstdio>
#include <vector>

int main() {
    constexpr int steps = 50;
    const int path_counts[] = {10000, 50000, 100000, 500000};

    std::printf("American Put Pricing using LSM Monte Carlo\n");
    std::printf("Sequential CPU\n\n");
    std::printf("Time steps: %d\n", steps);
    std::printf("Benchmark repetitions: %d\n\n", aophpc::BENCHMARK_REPETITIONS);
    std::printf("%10s %14s %12s\n", "Paths", "Price", "Median(ms)");

    for (int paths_count : path_counts) {
        const auto paths = aophpc::generate_lsm_paths(paths_count, steps);
        std::vector<double> timings;
        double price = 0.0;

        for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
            const auto start = aophpc::Clock::now();
            price = aophpc::lsm_price_cpu(paths, paths_count, steps);
            const auto stop = aophpc::Clock::now();
            timings.push_back(aophpc::elapsed_ms(start, stop));
        }

        std::printf("%10d %14.6f %12.3f\n",
                    paths_count, price, aophpc::median(timings));
    }
    return 0;
}
