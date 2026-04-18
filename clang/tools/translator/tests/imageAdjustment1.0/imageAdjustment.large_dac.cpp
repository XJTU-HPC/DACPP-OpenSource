#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

struct Pixel {
    int r, g, b;
    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
        return os;
    }
};

shell dacpp::list imageAdjustment(const dacpp::Matrix<Pixel>& image_tensor,
                                  dacpp::Matrix<Pixel>& image_tensor2) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{image_tensor[idx1][idx2], image_tensor2[idx1][idx2]};
    return dataList;
}

calc void image_1(Pixel* image_tensor, Pixel* image_tensor2) {
    image_tensor2[0].r = std::min(255, image_tensor[0].r + 50);
}

calc void image_2(Pixel* image_tensor2, Pixel* image_tensor3) {
    int value = 30;
    image_tensor3[0].r = std::min(255, image_tensor2[0].r + value);
    image_tensor3[0].g = std::min(255, image_tensor2[0].g + value);
    image_tensor3[0].b = std::min(255, image_tensor2[0].b + value);
}

int main() {
    const int width = 64;
    const int height = 64;
    std::vector<Pixel> image(height * width, {100, 100, 100});
    std::vector<Pixel> image2(height * width, {100, 100, 100});

    std::cout << "Original Image:" << std::endl;

    dacpp::Matrix<Pixel> image_tensor({height, width}, image);
    dacpp::Matrix<Pixel> image_tensor2({height, width}, image2);

    imageAdjustment(image_tensor, image_tensor2) <-> image_1;
    std::cout << "\nImage After Color Adjustment:" << std::endl;

    std::vector<Pixel> image3(height * width, {0, 0, 0});
    dacpp::Tensor<Pixel, 2> image_tensor3({height, width}, image3);

    imageAdjustment(image_tensor2, image_tensor3) <-> image_2;
    std::cout << "\nImage After Brightness Enhancement:" << std::endl;
    image_tensor3.print();

    return 0;
}
