#include <iostream>
#include <vector>

int main() {
    constexpr int N = 8;

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> c{100, 200, 300, 400, 500, 600, 700, 800};
    std::vector<float> bias{0.5f};
    std::vector<float> tmp(N, 0.0f);
    std::vector<float> shifted(N, 0.0f);
    std::vector<float> out(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        tmp[i] = a[i] + b[i];
    }

    for (int i = 0; i < N; ++i) {
        shifted[i] = tmp[i] + bias[0];
    }

    for (int i = 0; i < N; ++i) {
        out[i] = shifted[i] + c[i];
    }

    for (float value : out) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
