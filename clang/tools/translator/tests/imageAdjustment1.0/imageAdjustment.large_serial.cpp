#include <algorithm>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

struct Pixel {
    int r, g, b;

    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
        return os;
    }
};

int main() {
    const int width = 64;
    const int height = 64;

    std::vector<Pixel> image(height * width, {100, 100, 100});
    std::vector<Pixel> image2(height * width, {0, 0, 0});
    std::vector<Pixel> image3(height * width, {0, 0, 0});

    std::cout << "Original Image:" << std::endl;

    for (int idx = 0; idx < height * width; ++idx) {
        image2[idx].r = std::min(255, image[idx].r + 50);
    }

    std::cout << "\nImage After Color Adjustment:" << std::endl;

    for (int idx = 0; idx < height * width; ++idx) {
        image3[idx].r = std::min(255, image2[idx].r + 30);
        image3[idx].g = std::min(255, image2[idx].g + 30);
        image3[idx].b = std::min(255, image2[idx].b + 30);
    }

    std::cout << "\nImage After Brightness Enhancement:" << std::endl;
    dacpp::Matrix<Pixel> image_tensor3({height, width}, image3);
    image_tensor3.print();

    return 0;
}
