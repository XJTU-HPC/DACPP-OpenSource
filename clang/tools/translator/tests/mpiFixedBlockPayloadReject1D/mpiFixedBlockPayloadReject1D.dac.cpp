#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int ROWS = 6;
const int COLS = 4;

shell dacpp::list PAYLOAD(const dacpp::Matrix<int>& input,
                          dacpp::Matrix<int>& output) {
    dacpp::split S1(2, 2);
    dacpp::list dataList{input[{S1}][{}], output[{S1}][{}]};
    return dataList;
}

calc void copyPayload(dacpp::Matrix<int>& input,
                      dacpp::Matrix<int>& output) {
    for (int col = 0; col < COLS; ++col) {
        output[0][col] = input[0][col] + input[1][col];
        output[1][col] = input[1][col];
    }
}

int main() {
    std::vector<int> input_data(ROWS * COLS, 0);
    std::vector<int> output_data(ROWS * COLS, 0);
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
            input_data[row * COLS + col] = row * 10 + col;
        }
    }
    dacpp::Matrix<int> input({ROWS, COLS}, input_data);
    dacpp::Matrix<int> output({ROWS, COLS}, output_data);

    PAYLOAD(input, output) <-> copyPayload;

    output.print();
    return 0;
}
