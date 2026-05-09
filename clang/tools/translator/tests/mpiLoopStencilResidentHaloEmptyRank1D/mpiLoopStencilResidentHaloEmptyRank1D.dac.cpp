#include <iomanip>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int WIDTH = 5;
const int STEPS = 4;

shell dacpp::list tinyHaloShell(dacpp::Vector<double>& state READ_WRITE,
                                dacpp::Vector<double>& next READ_WRITE) {
    dacpp::index idx;
    dacpp::split sp(3, 1);
    binding(idx, sp);
    dacpp::list dataList{state[sp], next[idx]};
    return dataList;
}

calc void tinyHaloStep(dacpp::Vector<double>& state, double* next) {
    next[0] = 0.25 * state[0] + 0.5 * state[1] + 0.25 * state[2];
}

int main() {
    std::vector<double> state_data(WIDTH, 0.0);
    std::vector<double> next_data(WIDTH - 2, 0.0);
    for (int i = 0; i < WIDTH; ++i) {
        state_data[i] = static_cast<double>(i + 1);
    }

    dacpp::Vector<double> state(state_data);
    dacpp::Vector<double> next(next_data);

    for (int step = 0; step < STEPS; ++step) {
        tinyHaloShell(state, next) <-> tinyHaloStep;
        for (int i = 0; i <= WIDTH - 3; ++i) {
            state[i + 1] = next[i];
        }
    }

    std::cout << std::fixed << std::setprecision(6) << state[2]
              << std::endl;
    return 0;
}
