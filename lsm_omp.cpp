#include "config.hpp"
#include "lsm.hpp"
#include <omp.h>
#include <cstdio>
#include <vector>

namespace {

double lsm_price_omp(const std::vector<double>& paths,
                     int path_count, int steps, int threads) {
    const double dt = aophpc::MATURITY / steps;
    const double discount = std::exp(-aophpc::RISK_FREE_RATE * dt);
    std::vector<double> cashflow(path_count);
    std::vector<int> exercise_time(path_count, steps);
    omp_set_num_threads(threads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < path_count; ++i) {
        const double terminal_stock = paths[static_cast<std::size_t>(i) * (steps + 1) + steps];
        cashflow[i] = std::max(aophpc::STRIKE - terminal_stock, 0.0);
    }

    for (int t = steps - 1; t >= 1; --t) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
        double y0 = 0.0, y1 = 0.0, y2 = 0.0;
        int count = 0;

        #pragma omp parallel for reduction(+:s0,s1,s2,s3,s4,y0,y1,y2,count) schedule(static)
        for (int i = 0; i < path_count; ++i) {
            const double stock = paths[static_cast<std::size_t>(i) * (steps + 1) + t];
            const double exercise = std::max(aophpc::STRIKE - stock, 0.0);
            if (exercise <= 0.0) continue;

            const double x = stock;
            const double x2 = x * x;
            const double discounted_future = cashflow[i] *
                std::pow(discount, exercise_time[i] - t);

            s0 += 1.0;
            s1 += x;
            s2 += x2;
            s3 += x2 * x;
            s4 += x2 * x2;
            y0 += discounted_future;
            y1 += x * discounted_future;
            y2 += x2 * discounted_future;
            ++count;
        }

        if (count < 3) continue;

        double matrix[3][4] = {
            {s0, s1, s2, y0},
            {s1, s2, s3, y1},
            {s2, s3, s4, y2}
        };
        double coeff[3];
        if (!aophpc::solve_3x3(matrix, coeff)) continue;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < path_count; ++i) {
            const double stock = paths[static_cast<std::size_t>(i) * (steps + 1) + t];
            const double exercise = std::max(aophpc::STRIKE - stock, 0.0);
            if (exercise <= 0.0) continue;

            const double continuation = coeff[0] + coeff[1] * stock +
                                        coeff[2] * stock * stock;
            if (exercise > continuation) {
                cashflow[i] = exercise;
                exercise_time[i] = t;
            }
        }
    }

    double total = 0.0;
    #pragma omp parallel for reduction(+:total) schedule(static)
    for (int i = 0; i < path_count; ++i)
        total += cashflow[i] * std::pow(discount, exercise_time[i]);

    return total / path_count;
}

} // namespace

int main() {
    constexpr int steps = 50;
    const int path_counts[] = {10000, 50000, 100000, 500000};
    const int requested_threads[] = {1, 2, 4, 8};
    const int max_threads = omp_get_max_threads();

    std::printf("American Put Pricing using LSM Monte Carlo\n");
    std::printf("Sequential CPU vs OpenMP CPU\n\n");
    std::printf("Time steps: %d\n", steps);
    std::printf("Maximum OpenMP threads available: %d\n", max_threads);
    std::printf("Benchmark repetitions: %d\n\n", aophpc::BENCHMARK_REPETITIONS);
    std::printf("%10s %10s %14s %12s %12s %12s\n",
                "Paths", "Threads", "Price", "Time(ms)", "Speedup", "Price Error");

    for (int path_count : path_counts) {
        const auto paths = aophpc::generate_lsm_paths(path_count, steps);
        std::vector<double> serial_times;
        double serial_price = 0.0;

        for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
            const auto start = aophpc::Clock::now();
            serial_price = aophpc::lsm_price_cpu(paths, path_count, steps);
            const auto stop = aophpc::Clock::now();
            serial_times.push_back(aophpc::elapsed_ms(start, stop));
        }
        const double serial_ms = aophpc::median(serial_times);

        std::printf("%10d %10s %14.6f %12.3f %11.2fx %12s\n",
                    path_count, "Serial", serial_price, serial_ms, 1.0, "-");

        for (int requested : requested_threads) {
            if (requested > max_threads) continue;
            double parallel_price = lsm_price_omp(paths, path_count, steps, requested); // warm-up
            std::vector<double> timings;

            for (int rep = 0; rep < aophpc::BENCHMARK_REPETITIONS; ++rep) {
                const auto start = aophpc::Clock::now();
                parallel_price = lsm_price_omp(paths, path_count, steps, requested);
                const auto stop = aophpc::Clock::now();
                timings.push_back(aophpc::elapsed_ms(start, stop));
            }

            const double parallel_ms = aophpc::median(timings);
            const double speedup = serial_ms / parallel_ms;
            const double error = std::abs(serial_price - parallel_price);
            std::printf("%10d %10d %14.6f %12.3f %11.2fx %12.3e\n",
                        path_count, requested, parallel_price,
                        parallel_ms, speedup, error);
        }
        std::printf("\n");
    }
    return 0;
}
