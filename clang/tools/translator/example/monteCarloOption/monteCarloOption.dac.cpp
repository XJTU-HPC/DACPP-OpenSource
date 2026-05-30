#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;

const int NX = 512;
const int NY = 512;
const int PATHS_PER_OPTION = 4096;
const int INNER_REPEATS = 2;

const double S0 = 100.0;
const double STRIKE = 100.0;
const double RATE = 0.05;
const double VOLATILITY = 0.2;
const double MATURITY = 1.0;

unsigned int lcgNext(unsigned int state) {
    return state * 1664525u + 1013904223u;
}

double uniform01(unsigned int value) {
    return ((double)(value & 0x00FFFFFFu) + 1.0) / 16777217.0;
}

shell dacpp::list MONTE_CARLO_OPTION(const dacpp::Matrix<int>& seeds,
                                     dacpp::Matrix<double>& prices WRITE) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{seeds[idx1][idx2], prices[idx1][idx2]};
    return dataList;
}

calc void monteCarloOption(int* seed,
                           double* price) {
    unsigned int state = (unsigned int)seed[0] + 1u;
    double payoff_sum = 0.0;

    for (int repeat = 0; repeat < INNER_REPEATS; repeat++) {
        for (int path = 0; path < PATHS_PER_OPTION; path++) {
            state = lcgNext(state);
            double u1 = uniform01(state);
            state = lcgNext(state);
            double u2 = uniform01(state);

            double radius = std::sqrt(-2.0 * std::log(u1));
            double angle = 6.2831853071795864769 * u2;
            double normal = radius * std::cos(angle);

            double drift = (RATE - 0.5 * VOLATILITY * VOLATILITY) * MATURITY;
            double diffusion = VOLATILITY * std::sqrt(MATURITY) * normal;
            double terminal = S0 * std::exp(drift + diffusion);
            double payoff = terminal - STRIKE;
            if (payoff < 0.0) {
                payoff = 0.0;
            }

            payoff_sum += payoff;
        }
    }

    price[0] = std::exp(-RATE * MATURITY) * payoff_sum / (PATHS_PER_OPTION * INNER_REPEATS);
}

int main() {
    std::vector<int> seeds(NX * NY, 0);
    std::vector<double> prices(NX * NY, 0.0);

    for (int i = 0; i < NY; i++) {
        for (int j = 0; j < NX; j++) {
            seeds[i * NX + j] = i * NX + j;
        }
    }

    dacpp::Matrix<int> seeds_tensor({NY, NX}, seeds);
    dacpp::Matrix<double> prices_tensor({NY, NX}, prices);

    MONTE_CARLO_OPTION(seeds_tensor, prices_tensor) <-> monteCarloOption;

    std::cout << prices_tensor[NY / 2][NX / 2] << std::endl;

    return 0;
}