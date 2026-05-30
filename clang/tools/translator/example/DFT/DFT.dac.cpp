#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
using Complex = std::complex<double>;
const int N = 8;



shell dacpp::list DFT(const dacpp::Vector<std::complex<double>>& input,
                        dacpp::Vector<std::complex<double>>& output,
                    const dacpp::Vector<int>& vec) {
    dacpp::index i;


    dacpp::list dataList{input[{}], output[i], vec[i]};
    return dataList;
}

calc void dft(std::complex<double>* input,
                std::complex<double>* output,
                int* vec) {
    Complex sum(0, 0);
    for (int n = 0; n < N; ++n) {
        double angle = -2.0 * M_PI * vec[0] * n / N;
        Complex W_n(std::cos(angle), std::sin(angle));
        sum += input[n] * W_n;
    }
    output[0] = sum;
}


void dftfunc(const vector<std::complex<double>>& input, vector<std::complex<double>>& output) {
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);


    for (int i = 0; i < N; ++i) {
        vec[i] = i;
    }
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);


    DFT(input_tensor, output_tensor, vec_tensor) <-> dft;
    output_tensor.print();
}

int main() {


    vector<std::complex<double>> input(N);


    for (int i = 0; i < N; ++i) {
        input[i] = Complex(i, 0);
    }








    vector<Complex> output(N);
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);


    for (int i = 0; i < N; ++i) {
        vec[i] = i;
    }
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);


    DFT(input_tensor, output_tensor, vec_tensor) <-> dft;
    output_tensor.print();







    return 0;
}