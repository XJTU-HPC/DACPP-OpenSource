#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 8;

shell dacpp::list accumulate1DShell(const dacpp::Vector<int>& input READ,
                                    dacpp::Vector<int>& output WRITE) {
    dacpp::index i;
    dacpp::list dataList{input[i], output[i]};
    return dataList;
}

calc void accumulate1D(int* input, int* output) {
    output[0] += input[0] * 2;
}

int main() {
    std::vector<int> input{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> output{10, 20, 30, 40, 50, 60, 70, 80};

    dacpp::Vector<int> input_tensor(input);
    dacpp::Vector<int> output_tensor(output);

    accumulate1DShell(input_tensor, output_tensor) <-> accumulate1D;

    output_tensor.print();
    return 0;
}
