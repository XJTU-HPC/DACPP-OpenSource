#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int ROWS = 256;
const int COLS = 256;
const int TIME_STEPS = 10;
const int UPDATE_ROWS = ((ROWS - 3) / 2) + 1;

shell dacpp::list denseCoverShell(dacpp::Matrix<int>& state READ_WRITE,
                                  dacpp::Matrix<int>& updates WRITE) {
    dacpp::split row_band(3, 2);
    dacpp::index row_idx;
    binding(row_band, row_idx);
    dacpp::list dataList{state[row_band][{}], updates[row_idx][{}]};
    return dataList;
}

calc void denseCoverStep(dacpp::Matrix<int>& state,
                         dacpp::Vector<int>& update) {
    for (int col = 0; col < COLS; ++col) {
        update[col] = state[0][col] + state[1][col] - state[2][col];
    }
}

int main() {
    std::vector<int> state_data(ROWS * COLS, 0);
    std::vector<int> updates_data(UPDATE_ROWS * COLS, 0);

    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
            state_data[row * COLS + col] = row * 100 + col;
        }
    }

    dacpp::Matrix<int> state({ROWS, COLS}, state_data);
    dacpp::Matrix<int> updates({UPDATE_ROWS, COLS}, updates_data);

    for (int time_step = 0; time_step < TIME_STEPS; ++time_step) {
        denseCoverShell(state, updates) <-> denseCoverStep;

        for (int row = 0; row <= UPDATE_ROWS - 1; ++row) {
            for (int col = 0; col <= COLS - 1; ++col) {
                state[1 + row * 2][col] = updates[row][col] + row;
            }
        }

        for (int col = 0; col <= COLS - 1; ++col) {
            state[ROWS - 1][col] =
                state[ROWS - 2][col] + state[1][col] - state[0][col];
        }
    }

    state.print();
    return 0;
}
