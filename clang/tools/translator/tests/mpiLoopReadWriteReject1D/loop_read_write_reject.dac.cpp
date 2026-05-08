#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 8;

shell dacpp::list accumulateLoopShell(const dacpp::Vector<int>& input READ,
                                      dacpp::Vector<int>& output READ_WRITE) {
    dacpp::index i;
    dacpp::list dataList{input[i], output[i]};
    return dataList;
}

calc void accumulateLoop(int* input, int* output) {
    output[0] += input[0] * 2;
}

int main() {
    std::vector<int> input{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> output{10, 20, 30, 40, 50, 60, 70, 80};

    dacpp::Vector<int> input_tensor(input);
    dacpp::Vector<int> output_tensor(output);

    for (int step = 0; step < 2; ++step) {
        accumulateLoopShell(input_tensor, output_tensor) <-> accumulateLoop;
    }

    output_tensor.print();
    return 0;
}
