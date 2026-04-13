#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int N = 8;

shell dacpp::list VADD(const dacpp::Vector<float>& lhs,
                       const dacpp::Vector<float>& rhs,
                       dacpp::Vector<float>& out) {
    dacpp::index i;
    dacpp::list dataList{lhs[i], rhs[i], out[i]};
    return dataList;
}

calc void vadd(float* lhs, float* rhs, float* out) {
    out[0] = lhs[0] + rhs[0];
}

shell dacpp::list VSHIFT(const dacpp::Vector<float>& in,
                         const dacpp::Vector<float>& bias,
                         dacpp::Vector<float>& out) {
    dacpp::index i;
    dacpp::list dataList{in[i], bias[{}], out[i]};
    return dataList;
}

calc void vshift(float* in, float* bias, float* out) {
    out[0] = in[0] + bias[0];
}

int main() {
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> c{100, 200, 300, 400, 500, 600, 700, 800};
    std::vector<float> bias{0.5f};
    std::vector<float> tmp(N, 0.0f);
    std::vector<float> shifted(N, 0.0f);
    std::vector<float> out(N, 0.0f);

    dacpp::Vector<float> a_tensor(a);
    dacpp::Vector<float> b_tensor(b);
    dacpp::Vector<float> c_tensor(c);
    dacpp::Vector<float> bias_tensor(bias);
    dacpp::Vector<float> tmp_tensor(tmp);
    dacpp::Vector<float> shifted_tensor(shifted);
    dacpp::Vector<float> out_tensor(out);

    VADD(a_tensor, b_tensor, tmp_tensor) <-> vadd;
    VSHIFT(tmp_tensor, bias_tensor, shifted_tensor) <-> vshift;
    VADD(shifted_tensor, c_tensor, out_tensor) <-> vadd;

    std::vector<float> host_out;
    out_tensor.tensor2Array(host_out);

    for (float value : host_out) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
