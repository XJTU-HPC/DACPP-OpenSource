#include <iomanip>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int NX = 8;
const int NY = 10;
const int STEPS = 1;

shell dacpp::list spatialStencilShell(dacpp::Matrix<double>& state READ_WRITE,
                                      dacpp::Matrix<double>& next READ_WRITE) {
    dacpp::split rows(3, 1), cols(3, 1);
    dacpp::index row, col;
    binding(rows, row);
    binding(cols, col);
    dacpp::list dataList{state[rows][cols], next[row][col]};
    return dataList;
}

calc void spatialStencilStep(dacpp::Matrix<double>& state, double* next) {
    next[0] = 0.5 * state[1][1] +
              0.125 * (state[0][1] + state[2][1] +
                       state[1][0] + state[1][2]);
}

int main() {
    std::vector<double> stateData(NX * NY, 0.0);
    std::vector<double> nextData(NX * NY, 0.0);

    for (int row = 0; row < NX; ++row) {
        for (int col = 0; col < NY; ++col) {
            stateData[row * NY + col] = static_cast<double>(row * 10 + col);
        }
    }

    dacpp::Matrix<double> state({NX, NY}, stateData);
    dacpp::Matrix<double> nextTensor({NX, NY}, nextData);
    dacpp::Matrix<double> next = nextTensor[{1, NX - 1}][{1, NY - 1}];

    for (int step = 1; step <= STEPS; ++step) {
        spatialStencilShell(state, next) <-> spatialStencilStep;

        for (int row = 1; row <= NX - 2; ++row) {
            for (int col = 1; col <= NY - 2; ++col) {
                state[row][col] = next[row - 1][col - 1];
            }
        }
        for (int col = 0; col <= NY - 1; ++col) {
            state[0][col] = state[1][col];
            state[NX - 1][col] = state[NX - 2][col];
        }
        for (int row = 0; row <= NX - 1; ++row) {
            state[row][0] = state[row][1];
            state[row][NY - 1] = state[row][NY - 2];
        }
    }

    std::cout << std::fixed << std::setprecision(4) << state[0][0] << " "
              << state[4][5] << std::endl;
    return 0;
}
