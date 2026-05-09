#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int WIDTH = 12;
const int STEPS = 3;

shell dacpp::list rightBoundaryShell(
    dacpp::Vector<double>& state READ_WRITE,
    dacpp::Vector<double>& next READ_WRITE) {
    dacpp::index idx;
    dacpp::split sp(3, 1);
    binding(idx, sp);
    dacpp::list dataList{state[sp], next[idx]};
    return dataList;
}

calc void rightBoundaryStep(dacpp::Vector<double>& state, double* next) {
    next[0] = 0.2 * state[0] + 0.6 * state[1] + 0.2 * state[2];
}

int main() {
    std::vector<double> state_data(WIDTH, 0.0);
    std::vector<double> next_data(WIDTH - 2, 0.0);
    for (int i = 0; i < WIDTH; ++i) {
        state_data[i] = static_cast<double>((i % 4) + i);
    }

    dacpp::Vector<double> state(state_data);
    dacpp::Vector<double> next(next_data);

    for (int step = 0; step < STEPS; ++step) {
        rightBoundaryShell(state, next) <-> rightBoundaryStep;
        for (int i = 0; i <= WIDTH - 3; ++i) {
            state[i + 1] = next[i];
        }
        for (int i = 0; i < 1; ++i) {
            state[WIDTH - 1] = next[WIDTH - 3];
        }
    }

    std::cout << state[WIDTH - 1] << std::endl;
    return 0;
}
