#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

shell dacpp::list accumulate2DShell(const dacpp::Matrix<int>& input READ,
                                    dacpp::Matrix<int>& output WRITE) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{input[idx1][idx2], output[idx1][idx2]};
    return dataList;
}

calc void accumulate2D(int* input, int* output) {
    output[0] += input[0] + 3;
}

int main() {
    std::vector<int> input{
        1, 2, 3,
        4, 5, 6
    };
    std::vector<int> output{
        10, 20, 30,
        40, 50, 60
    };

    dacpp::Matrix<int> input_tensor({2, 3}, input);
    dacpp::Matrix<int> output_tensor({2, 3}, output);

    accumulate2DShell(input_tensor, output_tensor) <-> accumulate2D;

    output_tensor.print();
    return 0;
}
