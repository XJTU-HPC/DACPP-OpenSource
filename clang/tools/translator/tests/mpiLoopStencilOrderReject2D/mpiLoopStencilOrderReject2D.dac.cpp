#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int NX = 8;
const int NY = 8;
const int STEPS = 2;

shell dacpp::list orderRejectShell(dacpp::Matrix<double>& matCur WRITE,
                                   dacpp::Matrix<double>& matPrev READ_WRITE,
                                   dacpp::Matrix<double>& matNext WRITE) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matCur[sp1][sp2], matPrev[idx1][idx2], matNext[idx1][idx2]};
    return dataList;
}

calc void orderRejectStep(dacpp::Matrix<double>& cur, double* prev, double* next) {
    next[0] = 2.0 * cur[1][1] - prev[0] +
              0.125 * (cur[0][1] + cur[2][1] + cur[1][0] + cur[1][2]);
}

int main() {
    std::vector<double> cur_data(NX * NY, 1.0);
    std::vector<double> prev_data(NX * NY, 0.5);
    std::vector<double> next_data(NX * NY, 0.0);

    dacpp::Matrix<double> matCur({NX, NY}, cur_data);
    dacpp::Matrix<double> prevTensor({NX, NY}, prev_data);
    dacpp::Matrix<double> nextTensor({NX, NY}, next_data);
    dacpp::Matrix<double> matPrev = prevTensor[{1, NX - 1}][{1, NY - 1}];
    dacpp::Matrix<double> matNext = nextTensor[{1, NX - 1}][{1, NY - 1}];

    for (int step = 0; step < STEPS; ++step) {
        orderRejectShell(matCur, matPrev, matNext) <-> orderRejectStep;
        for (int i = 1; i <= NX - 2; ++i) {
            for (int j = 1; j <= NY - 2; ++j) {
                matCur[i][j] = matNext[i - 1][j - 1];
            }
        }
        for (int i = 1; i <= NX - 2; ++i) {
            for (int j = 1; j <= NY - 2; ++j) {
                matPrev[i - 1][j - 1] = matCur[i][j];
            }
        }
        for (int i = 0; i <= NX - 1; ++i) {
            matCur[i][NY - 1] = 0;
            matCur[i][0] = 0;
        }
        for (int j = 0; j <= NY - 1; ++j) {
            matCur[NX - 1][j] = 0;
            matCur[0][j] = 0;
        }
    }

    return 0;
}
