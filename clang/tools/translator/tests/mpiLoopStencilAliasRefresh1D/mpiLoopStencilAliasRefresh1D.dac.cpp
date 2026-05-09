#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int WIDTH = 10;
const int STEPS = 3;

shell dacpp::list aliasShell(const dacpp::Vector<double>& state,
                             dacpp::Vector<double>& next,
                             const dacpp::Vector<double>& gain) {
    dacpp::index idx;
    dacpp::split sp(3, 1);
    binding(idx, sp);
    dacpp::list dataList{state[sp], next[idx], gain[{}]};
    return dataList;
}

calc void aliasStep(dacpp::Vector<double>& state,
                    double* next,
                    double* gain) {
    next[0] =
        gain[0] * state[0] + (1.0 - 2.0 * gain[0]) * state[1] +
        gain[0] * state[2];
}

int main() {
    std::vector<double> state_data(WIDTH, 0.0);
    std::vector<double> gain_data{0.25};
    for (int i = 0; i < WIDTH; ++i) {
        state_data[i] = static_cast<double>(i + 1);
    }

    dacpp::Vector<double> state_tensor(state_data);
    dacpp::Vector<double> gain(gain_data);
    dacpp::Vector<double> state = state_tensor[{0, WIDTH - 1}];

    for (int step = 0; step < STEPS; ++step) {
        aliasShell(state, state, gain) <-> aliasStep;
    }

    state.print();
    return 0;
}
