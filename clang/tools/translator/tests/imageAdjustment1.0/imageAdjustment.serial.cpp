#include <iostream>
#include <vector>

// 定义图像类型，假设每个像素由 RGB 三个值组成
struct Pixel {
    int r, g, b;
};

// 色彩调整操作：增加红色分量
void adjust_color(std::vector<std::vector<Pixel>>& image) {
    for (auto& row : image) {
        for (auto& pixel : row) {
            pixel.r = std::min(255, pixel.r + 50);  // 增加红色分量，并确保不超过255
        }
    }
}

// 亮度增强操作：增加每个像素的 RGB 分量
void enhance_brightness(std::vector<std::vector<Pixel>>& image, int value) {
    for (auto& row : image) {
        for (auto& pixel : row) {
            pixel.r = std::min(255, pixel.r + value);  // 增加红色分量，限制在255以内
            pixel.g = std::min(255, pixel.g + value);  // 增加绿色分量
            pixel.b = std::min(255, pixel.b + value);  // 增加蓝色分量
        }
    }
}

// 打印图像的前几个像素，作为调试
void print_image(const std::vector<std::vector<Pixel>>& image, int num_rows = 5, int num_cols = 5) {
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            std::cout << "(" << image[i][j].r << "," << image[i][j].g << "," << image[i][j].b << ") ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // 初始化一个简单的图像（10x10），所有像素值初始化为(100, 100, 100)
    int width = 10, height = 10;
    std::vector<std::vector<Pixel>> image(height, std::vector<Pixel>(width, {100, 100, 100}));

    // 打印初始图像
    std::cout << "Original Image:" << std::endl;
    print_image(image);

    // 执行色彩调整操作
    adjust_color(image);
    std::cout << "\nImage After Color Adjustment:" << std::endl;
    print_image(image);

    // 执行亮度增强操作
    enhance_brightness(image, 30);
    std::cout << "\nImage After Brightness Enhancement:" << std::endl;
    print_image(image);

    return 0;
}
