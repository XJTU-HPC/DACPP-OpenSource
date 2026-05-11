#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 8;

shell dacpp::list COPY_BLOCK(const dacpp::Vector<int>& input,
                             dacpp::Vector<int>& output) {
    dacpp::split S1(2, 2);
    dacpp::list dataList{input[{S1}], output[{S1}]};
    return dataList;
}

calc void copy_block(int* input, int* output) {
    output[0] = input[0] + 1;
    output[1] = input[1] + 1;
}

int sumTensor(dacpp::Vector<int>& values) {
    int total = 0;
    for (int i = 0; i < N; ++i) {
        total += values[i];
    }
    return total;
}

int main() {
    std::vector<int> input_data{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> output_data(N, 0);

    dacpp::Vector<int> input(input_data);
    dacpp::Vector<int> output(output_data);

    COPY_BLOCK(input, output) <-> copy_block;

    int total = sumTensor(output);
    std::cout << total << std::endl;
    return 0;
}
