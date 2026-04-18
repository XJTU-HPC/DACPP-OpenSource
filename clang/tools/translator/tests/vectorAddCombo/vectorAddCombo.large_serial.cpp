#include <iostream>
#include <vector>

int main() {
    constexpr int N = 4096;

    std::vector<float> a(N), b(N), c(N);
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
        c[i] = static_cast<float>(1000 + i);
    }
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

    for (int i = 0; i < 16; ++i) {
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
