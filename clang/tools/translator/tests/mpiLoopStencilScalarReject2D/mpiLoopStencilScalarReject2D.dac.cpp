#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int NX = 8;
const int NY = 8;
const int STEPS = 2;

shell dacpp::list scalarStencilShell(
    dacpp::Matrix<double>& matIn READ_WRITE,
    dacpp::Matrix<double>& matOut READ_WRITE,
    const dacpp::Vector<double>& gain) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matIn[sp1][sp2], matOut[idx1][idx2], gain[{}]};
    return dataList;
}

calc void scalarStencil(dacpp::Matrix<double>& mat,
                        double* out,
                        double* gain) {
    out[0] = gain[0] * mat[1][1] +
             0.125 * (mat[0][1] + mat[2][1] + mat[1][0] + mat[1][2]);
}

int main() {
    std::vector<double> state_data(NX * NY, 1.0);
    std::vector<double> next_data(NX * NY, 0.0);
    std::vector<double> gain_data{0.5};

    dacpp::Matrix<double> matIn({NX, NY}, state_data);
    dacpp::Matrix<double> nextTensor({NX, NY}, next_data);
    dacpp::Matrix<double> matOut = nextTensor[{1, NX - 1}][{1, NY - 1}];
    dacpp::Vector<double> gain(gain_data);

    for (int step = 0; step < STEPS; ++step) {
        scalarStencilShell(matIn, matOut, gain) <-> scalarStencil;

        for (int i = 1; i <= NX - 2; ++i) {
            for (int j = 1; j <= NY - 2; ++j) {
                matIn[i][j] = matOut[i - 1][j - 1];
            }
        }
        for (int j = 0; j <= NY - 1; ++j) {
            matIn[0][j] = matIn[1][j];
            matIn[NX - 1][j] = matIn[NX - 2][j];
        }
        for (int i = 0; i < NX - 1; ++i) {
            matIn[i][0] = matIn[i][1];
            matIn[i][NY - 1] = matIn[i][NY - 2];
        }
    }

    std::cout << matIn[0][0] << std::endl;
    return 0;
}
