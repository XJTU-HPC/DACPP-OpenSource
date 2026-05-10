#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 16;

shell dacpp::list ODDEVEN(const dacpp::Vector<int> & array, dacpp::Vector<int> & array_out) {
    dacpp::split S1(2, 2);
    dacpp::list dataList{array[{S1}], array_out[{S1}]};
    return dataList;
}

calc void oddeven(int* array, int* array_out) {
    if (array[0] > array[1]) {
        array_out[0] = array[1];
        array_out[1] = array[0];
    } else {
        array_out[0] = array[0];
        array_out[1] = array[1];
    }
}

void run(std::vector<int>& array) {
    dacpp::Tensor<int, 1> array_tensor(array);
    std::vector<int> array_out(N);
    dacpp::Tensor<int, 1> array_out_tensor(array_out);

    for (int phase = 0; phase < 2; phase++) {
        ODDEVEN(array_tensor, array_out_tensor) <-> oddeven;

        // Slice offset 2 (not the proven phase shift of 1) must reject the
        // phase-exchange match and fall back to the standalone P5 wrapper.
        dacpp::Tensor<int, 1> array2_tensor = array_out_tensor[{2, N-1}];
        std::vector<int> array_out2(N - 3, 0);
        dacpp::Tensor<int, 1> array_out2_tensor(array_out2);

        ODDEVEN(array2_tensor, array_out2_tensor) <-> oddeven;

        for (int i = 1; i < N - 1; i++) {
            array_tensor[i] = array_out2_tensor[i - 1];
        }
        array_tensor[0] = array_out_tensor[0];
        array_tensor[N - 1] = array_out_tensor[N - 1];
    }
    array_tensor.print();
}

int main() {
    std::vector<int> array(N);
    for (int i = 0; i < N; i++) {
        array[i] = N - i;
    }
    run(array);
    return 0;
}
