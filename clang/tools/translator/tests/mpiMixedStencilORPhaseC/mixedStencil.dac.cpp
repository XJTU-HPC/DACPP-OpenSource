#include <cmath>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

using namespace std;

namespace dacpp {
typedef std::vector<std::any> list;
}

const int NX = 8;
const int NY = 8;
const double LX = 10.0;
const double LY = 10.0;
const double DX = LX / (NX - 1);
const double DY = LY / (NY - 1);
const double dx = DX;
const double dy = DY;

const double ALPHA = 0.01;
const int STENCIL_STEPS = 6;
const double STENCIL_DT_STABILITY =
    (DX * DX * DY * DY) / (2.0 * ALPHA * (DX * DX + DY * DY));
const double STENCIL_DT = 0.4 * STENCIL_DT_STABILITY;

shell dacpp::list stencilShell(dacpp::Matrix<double>& matIn READ_WRITE,
                               dacpp::Matrix<double>& matOut READ_WRITE) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matIn[sp1][sp2], matOut[idx1][idx2]};
    return dataList;
}

calc void stencil(dacpp::Matrix<double>& mat, double* out) {
    out[0] = mat[1][1] +
             ALPHA * STENCIL_DT *
                 (((mat[2][1] - 2.0 * mat[1][1] + mat[0][1]) / (DX * DX)) +
                  ((mat[1][2] - 2.0 * mat[1][1] + mat[1][0]) / (DY * DY)));
}

const int WIDTH = 64;
const int PHASE_C_STEPS = 8;

shell dacpp::list clampShell(dacpp::Vector<double>& state READ,
                             dacpp::Vector<double>& next READ_WRITE) {
    dacpp::split S1(2, 1);
    dacpp::index idx1;
    binding(S1, idx1);
    dacpp::list dataList{state[S1], next[idx1]};
    return dataList;
}

calc void clamp_step(dacpp::Vector<double>& state, double* next) {
    next[0] = 0.5 * state[0] + 0.5 * state[1] - 3.0;
    next[0] = std::max(0.0, next[0]);
}

void runStencilPhase() {
    vector<double> stencilInit(NX * NY, 0.0);
    vector<double> stencilNext(NX * NY, 0.0);

    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            double x = i * DX;
            double y = j * DY;
            stencilInit[i * NY + j] =
                std::exp(-((x - LX / 2.0) * (x - LX / 2.0) +
                           (y - LY / 2.0) * (y - LY / 2.0)) /
                         2.0);
        }
    }

    dacpp::Matrix<double> matIn({NX, NY}, stencilInit);
    dacpp::Matrix<double> nextTensor({NX, NY}, stencilNext);
    dacpp::Matrix<double> matOut = nextTensor[{1, NX - 1}][{1, NY - 1}];

    for (int step = 0; step < STENCIL_STEPS; ++step) {
        stencilShell(matIn, matOut) <-> stencil;

        for (int row = 1; row <= NX - 2; ++row) {
            for (int col = 1; col <= NY - 2; ++col) {
                matIn[row][col] = matOut[row - 1][col - 1];
            }
        }

        for (int col = 0; col <= NY - 1; ++col) {
            matIn[0][col] = matIn[1][col];
            matIn[NX - 1][col] = matIn[NX - 2][col];
        }
        for (int row = 0; row <= NX - 1; ++row) {
            matIn[row][0] = matIn[row][1];
            matIn[row][NY - 1] = matIn[row][NY - 2];
        }
    }

    matIn[0].print();
}

void runPhaseCPhase() {
    vector<double> state_data(WIDTH, 0.0);
    vector<double> next_data(WIDTH, 0.0);

    for (int i = 0; i < WIDTH; ++i) {
        state_data[i] = static_cast<double>((i % 9) + 1);
    }

    dacpp::Vector<double> state_tensor(state_data);
    dacpp::Vector<double> next_tensor(next_data);
    dacpp::Vector<double> state = state_tensor[{0, WIDTH - 1}];
    dacpp::Vector<double> next = next_tensor[{1, WIDTH - 1}];

    for (int step = 0; step < PHASE_C_STEPS; ++step) {
        clampShell(state, next) <-> clamp_step;

        for (int i = 1; i <= WIDTH - 2; ++i) {
            state[i] = next[i - 1];
        }
    }

    std::cout << state[5] << std::endl;
}

int main() {
    runStencilPhase();
    runPhaseCPhase();
    return 0;
}
