#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 8;

shell dacpp::list genShell(const dacpp::Vector<float>& input READ,
                           dacpp::Vector<float>& output WRITE) {
    dacpp::index i;
    dacpp::list dataList{input[i], output[i]};
    return dataList;
}

calc void gen(float* input, float* output) {
    output[0] = input[0] + 1.0f;
}

int main() {
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> output(N, 0.0f);

    dacpp::Vector<float> input_tensor(input);
    dacpp::Vector<float> output_tensor(output);

    genShell(input_tensor, output_tensor) <-> gen;

    std::cout << output_tensor[3] << std::endl;
    return 0;
}
