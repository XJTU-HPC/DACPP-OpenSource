#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 9;

shell dacpp::list OVERLAP(const dacpp::Vector<int>& input,
                          dacpp::Vector<int>& output) {
    dacpp::split S1(3, 2);
    dacpp::list dataList{input[{S1}], output[{S1}]};
    return dataList;
}

calc void copy3(int* input, int* output) {
    output[0] = input[0];
    output[1] = input[1];
    output[2] = input[2];
}

int main() {
    std::vector<int> input_data(N, 0);
    std::vector<int> output_data(N, 0);
    for (int i = 0; i < N; ++i) {
        input_data[i] = i + 1;
    }
    dacpp::Vector<int> input(input_data);
    dacpp::Vector<int> output(output_data);

    OVERLAP(input, output) <-> copy3;

    output.print();
    return 0;
}
