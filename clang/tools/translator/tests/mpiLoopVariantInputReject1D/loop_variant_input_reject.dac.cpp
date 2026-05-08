#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 8;

shell dacpp::list copyScaleShell(const dacpp::Vector<int>& input,
                                 dacpp::Vector<int>& output) {
    dacpp::index i;
    dacpp::list dataList{input[i], output[i]};
    return dataList;
}

calc void copyScale(int* input, int* output) {
    output[0] = input[0] * 3;
}

int main() {
    std::vector<int> input{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> output(N, 0);
    dacpp::Vector<int> input_tensor(input);
    dacpp::Vector<int> output_tensor(output);

    for (int step = 0; step < 2; ++step) {
        copyScaleShell(input_tensor, output_tensor) <-> copyScale;
        input_tensor[0] = step + 2;
    }

    output_tensor.print();
    return 0;
}
