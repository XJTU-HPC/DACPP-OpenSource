#include <iomanip>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int ROWS = 10;
const int COLS = 12;
const int TIME_STEPS = 8;

calc void heat2d_step(dacpp::Matrix<double>& state, double* next) {
    next[0] = 0.125 * state[0][1] + 0.125 * state[2][1] +
              0.125 * state[1][0] + 0.125 * state[1][2] +
              0.5 * state[1][1];
}

shell dacpp::list heat2dShell(dacpp::Matrix<double>& state READ,
                              dacpp::Matrix<double>& next WRITE) {
    dacpp::split SR(3, 1), SC(3, 1);
    dacpp::index IR, IC;
    binding(SR, IR);
    binding(SC, IC);
    dacpp::list dataList{state[SR][SC], next[IR][IC]};
    return dataList;
}

int main() {
    std::vector<double> state_data(ROWS * COLS, 0.0);
    std::vector<double> next_data((ROWS - 2) * (COLS - 2), 0.0);

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            state_data[i * COLS + j] = static_cast<double>((i + 1) * 3 + (j % 7));
        }
    }

    dacpp::Matrix<double> state_tensor({ROWS, COLS}, state_data);
    dacpp::Matrix<double> next_tensor({ROWS - 2, COLS - 2}, next_data);
    dacpp::Matrix<double> state = state_tensor;
    dacpp::Matrix<double> next = next_tensor;

    for (int t = 0; t < TIME_STEPS; ++t) {
        heat2dShell(state, next) <-> heat2d_step;

        for (int i = 1; i <= ROWS - 2; ++i) {
            for (int j = 1; j <= COLS - 2; ++j) {
                state[i][j] = next[i - 1][j - 1];
            }
        }
    }

    std::cout << std::fixed << std::setprecision(4) << state[4][5] << std::endl;
    return 0;
}
