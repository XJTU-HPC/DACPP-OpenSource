#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 8;

shell dacpp::list aliasLoopShell(const dacpp::Vector<int>& input,
                                 dacpp::Vector<int>& output) {
    dacpp::index i;
    dacpp::list dataList{input[i], output[i]};
    return dataList;
}

calc void aliasLoop(int* input, int* output) {
    output[0] = input[0];
}

int main() {
    std::vector<int> values{1, 2, 3, 4, 5, 6, 7, 8};
    dacpp::Vector<int> values_tensor(values);

    for (int step = 0; step < 2; ++step) {
        aliasLoopShell(values_tensor, values_tensor) <-> aliasLoop;
    }

    values_tensor.print();
    return 0;
}
