#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

// 网格参数
const int NX = 512;           // x方向网格数量
const int NY = 512;           // y方向网格数量
const float Lx = 10.0f;       // x方向长度
const float Ly = 10.0f;       // y方向长度
const float alpha = 0.01f;    // 热扩散系数
const int TIME_STEPS = 1000;  // 时间步数

// 保存结果到文件
void save_to_file(const vector<float>& data, int nx, int ny, const string& filename) {
    ofstream file(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            file << data[i * ny + j] << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    // 空间步长
    float dx = Lx / (NX - 1);
    float dy = Ly / (NY - 1);

    // 稳定性条件
    float dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
    float delta_t = 0.4f * dt_stability; // 稳定的时间步长

    cout << "Grid size: " << NX << "x" << NY << "\n";
    cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // 初始化温度场
    vector<float> u_prev(NX * NY, 0.0f); // 前一步
    vector<float> u_curr(NX * NY, 0.0f); // 当前步

    // 初始条件：中心高斯分布
    int cx = NX / 2;
    int cy = NY / 2;
    float sigma = 1.0f;
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            float x = i * dx;
            float y = j * dy;
            u_curr[i * NY + j] = exp(-((x - Lx / 2.0f) * (x - Lx / 2.0f) + 
                                       (y - Ly / 2.0f) * (y - Ly / 2.0f)) / (2.0f * sigma * sigma));
        }
    }

    // 主迭代循环
    for (int t = 0; t < TIME_STEPS; ++t) {
        vector<float> u_next(NX * NY, 0.0f);

        // 更新内部点
        for (int i = 1; i < NX - 1; ++i) {
            for (int j = 1; j < NY - 1; ++j) {
                float u_xx = (u_curr[(i + 1) * NY + j] - 2.0f * u_curr[i * NY + j] + 
                              u_curr[(i - 1) * NY + j]) / (dx * dx);
                float u_yy = (u_curr[i * NY + (j + 1)] - 2.0f * u_curr[i * NY + j] + 
                              u_curr[i * NY + (j - 1)]) / (dy * dy);

                u_next[i * NY + j] = u_curr[i * NY + j] + alpha * delta_t * (u_xx + u_yy);
            }
        }

        // 处理边界条件（绝热边界：导数为零）
        for (int j = 0; j < NY; ++j) {
            u_next[0 * NY + j] = u_next[1 * NY + j];              // 顶部边界
            u_next[(NX - 1) * NY + j] = u_next[(NX - 2) * NY + j]; // 底部边界
        }
        for (int i = 0; i < NX; ++i) {
            u_next[i * NY + 0] = u_next[i * NY + 1];              // 左边界
            u_next[i * NY + (NY - 1)] = u_next[i * NY + (NY - 2)]; // 右边界
        }

        // 更新当前温度场
        u_curr = u_next;

        // 可选：每隔一定步数输出进度
        if (t % 100 == 0) {
            cout << "Step " << t << " completed.\n";
        }
    }

    // 保存最终结果
    save_to_file(u_curr, NX, NY, "final_temperature_serial.txt");

    cout << "Final temperature at center: " << u_curr[(NX / 2) * NY + (NY / 2)] << "\n";

    return 0;
}
