#include <iomanip>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int NX = 8;
const int NY = 8;
const int STEPS = 1;

shell dacpp::list strideShell(dacpp::Matrix<double>& state READ_WRITE,
                              dacpp::Matrix<double>& next READ_WRITE) {
    dacpp::split sr(3, 1), sc(3, 1);
    dacpp::index ir, ic;
    binding(sr, ir);
    binding(sc, ic);
    dacpp::list dataList{state[sr][sc], next[ir][ic]};
    return dataList;
}

calc void stride_step(dacpp::Matrix<double>& state, double* next) {
    next[0] = 0.2 * state[0][1] + 0.2 * state[2][1] +
              0.2 * state[1][0] + 0.2 * state[1][2] +
              0.2 * state[1][1];
}

int main() {
    std::vector<double> state_data(NX * NY, 0.0);
    std::vector<double> next_data(NX * NY, 0.0);

    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            state_data[i * NY + j] = static_cast<double>(i * 100 + j);
        }
    }

    dacpp::Matrix<double> state_tensor({NX, NY}, state_data);
    dacpp::Matrix<double> next_tensor({NX, NY}, next_data);
    dacpp::Matrix<double> state = state_tensor;
    dacpp::Matrix<double> next = next_tensor[{1, NX - 1}][{1, NY - 1}];

    for (int t = 0; t < STEPS; ++t) {
        strideShell(state, next) <-> stride_step;

        for (int i = 1; i <= NX - 2; ++i) {
            for (int j = 1; j <= NY - 2; ++j) {
                state[i][j] = next[i - 1][j - 1];
            }
        }

        for (int i = 0; i < NX - 1; i += 2) {
            state[i][0] = state[i][1];
            state[i][NY - 1] = state[i][NY - 2];
        }
    }

    std::cout << std::fixed << std::setprecision(4) << state[2][0]
              << std::endl;
    return 0;
}
