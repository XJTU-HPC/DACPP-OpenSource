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

const double WAVE_C = 1.0;
const double c = WAVE_C;
const int WAVE_STEPS = 6;
const double WAVE_DT = 0.5 * std::fmin(DX, DY) / WAVE_C;

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

shell dacpp::list waveEqShell(dacpp::Matrix<double>& matCur WRITE,
                              dacpp::Matrix<double>& matPrev READ_WRITE,
                              dacpp::Matrix<double>& matNext WRITE) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matCur[sp1][sp2], matPrev[idx1][idx2],
                         matNext[idx1][idx2]};
    return dataList;
}

calc void waveEq(dacpp::Matrix<double>& cur, double* prev, double* next) {
    double u_xx = (cur[2][1] - 2.0 * cur[1][1] + cur[0][1]) / (DX * DX);
    double u_yy = (cur[1][2] - 2.0 * cur[1][1] + cur[1][0]) / (DY * DY);
    next[0] =
        2.0 * cur[1][1] - prev[0] + (WAVE_C * WAVE_C) * WAVE_DT * WAVE_DT *
                                          (u_xx + u_yy);
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

void runWavePhase() {
    vector<double> wavePrevVec(NX * NY, 0.0);
    vector<double> waveCurVec(NX * NY, 0.0);
    vector<double> waveNextVec(NX * NY, 0.0);

    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            double x = i * DX;
            double y = j * DY;
            wavePrevVec[i * NY + j] =
                std::exp(-((x - LX / 2.0) * (x - LX / 2.0) +
                           (y - LY / 2.0) * (y - LY / 2.0)) /
                         0.5);
        }
    }

    dacpp::Matrix<double> matCur({NX, NY}, waveCurVec);
    dacpp::Matrix<double> prevTensor({NX, NY}, wavePrevVec);
    dacpp::Matrix<double> nextTensor({NX, NY}, waveNextVec);
    dacpp::Matrix<double> matPrev = prevTensor[{1, NX - 1}][{1, NY - 1}];
    dacpp::Matrix<double> matNext = nextTensor[{1, NX - 1}][{1, NY - 1}];

    for (int step = 0; step < WAVE_STEPS; ++step) {
        waveEqShell(matCur, matPrev, matNext) <-> waveEq;

        for (int row = 1; row <= NX - 2; ++row) {
            for (int col = 1; col <= NY - 2; ++col) {
                matPrev[row - 1][col - 1] = matCur[row][col];
            }
        }

        for (int row = 1; row <= NX - 2; ++row) {
            for (int col = 1; col <= NY - 2; ++col) {
                matCur[row][col] = matNext[row - 1][col - 1];
            }
        }

        for (int row = 0; row <= NX - 1; ++row) {
            matCur[row][0] = 0;
            matCur[row][NY - 1] = 0;
        }
        for (int col = 0; col <= NY - 1; ++col) {
            matCur[0][col] = 0;
            matCur[NX - 1][col] = 0;
        }
    }

    matCur.print();
}

int main() {
    runStencilPhase();
    runWavePhase();
    return 0;
}
