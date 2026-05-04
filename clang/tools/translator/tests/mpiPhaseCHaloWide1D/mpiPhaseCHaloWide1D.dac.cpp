#include <iomanip>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 96;
const int TIME_STEPS = 7;

calc void wide_step(dacpp::Vector<double>& state, double* next) {
    next[0] = 0.05 * state[0] + 0.15 * state[1] + 0.6 * state[2] +
              0.15 * state[3] + 0.05 * state[4];
}

shell dacpp::list wideShell(dacpp::Vector<double>& state READ,
                            dacpp::Vector<double>& next WRITE) {
    dacpp::split S1(5, 2);
    dacpp::index idx1;
    binding(S1, idx1);
    dacpp::list dataList{state[S1], next[idx1]};
    return dataList;
}

int main() {
    std::vector<double> state_data(WIDTH, 0.0);
    std::vector<double> next_data(WIDTH, 0.0);

    for (int i = 0; i < WIDTH; ++i) {
        state_data[i] = static_cast<double>((i % 17) + (i / 11));
    }

    dacpp::Vector<double> state_tensor(state_data);
    dacpp::Vector<double> next_tensor(next_data);
    dacpp::Vector<double> state = state_tensor[{0, WIDTH - 1}];
    dacpp::Vector<double> next = next_tensor[{1, WIDTH - 1}];

    for (int t = 0; t < TIME_STEPS; ++t) {
        wideShell(state, next) <-> wide_step;

        for (int i = 1; i <= WIDTH - 2; ++i) {
            state[i] = next[i - 1];
        }
    }

    std::cout << std::fixed << std::setprecision(4)
              << state[8] << " " << state[44] << std::endl;
    return 0;
}
