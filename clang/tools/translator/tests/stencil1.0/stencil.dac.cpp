#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
// #define DACPP_TRANSLATE_MODE 1

using namespace std;
namespace dacpp {
    typedef std::vector<std::any> list;
}

// 网格参数
const int NX = 32;           // x方向网格数量
const int NY = 32;           // y方向网格数量
const double Lx = 10.0f;       // x方向长度
const double Ly = 10.0f;       // y方向长度
const double alpha = 0.01f;    // 热扩散系数
const int TIME_STEPS = 100;  // 时间步数
// 空间步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// 稳定性条件
const double dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
const double delta_t = 0.4f * dt_stability; // 选择一个更严格的时间步长以确保稳定性

shell dacpp::list stencilShell([[clang::annotate("read_write")]] dacpp::Matrix<double>& matIn, 
                                [[clang::annotate("read_write")]] dacpp::Matrix<double>& matOut) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matIn[sp1][sp2], matOut[idx1][idx2]};
    return dataList;
}

calc void stencil(dacpp::Matrix<double>& mat, 
                    double* out) {
    out[0] = mat[1][1] + alpha *delta_t * (((mat[2][1] - 2.0f * mat[1][1] + mat[0][1]) / (dx * dx))+ ((mat[1][2] - 2.0f * mat[1][1] + mat[1][0]) / (dy * dy)));                
}

int main() {


    //cout << "Grid size: " << NX << "x" << NY << "\n";
    //cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // 初始化温度场
    vector<double> u_prev(NX * NY, 0.0f); // 前一步（在热传导方程中，只有当前步和上一步）
    vector<double> u_curr(NX * NY, 0.0f);  // 当前步
    vector<double> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如，中心有一个高斯分布的热源
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            // 高斯分布
            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }

    //std::vector<int> shape = {32, 32};
    dacpp::Matrix<double> matIn({NX, NY}, u_curr);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matOut = u_next_tensor[{1,NX-1}][{1,NY-1}];
    for(int i=0;i<TIME_STEPS;i++) {
        stencilShell(matIn, matOut) <-> stencil;

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                matIn[i][j]=matOut[i-1][j-1];
            }
        }

        // 处理边界条件（绝热边界：导数为零）
        for (int j = 0; j <= NY-1; ++j) {
            //double* data = new double[1];
            matIn[0][j]=matIn[1][j];              // 顶部边界
            matIn[NX - 1][j]=matIn[NX-2][j];
             // 底部边界
        }
        for (int i = 0; i < NX-1; ++i) {
            //double* data = new double[1];
            matIn[i][0]=matIn[i][1];              // 顶部边界
            matIn[i][NY-1]=matIn[i][NY-2];
        }
        
    }
    matIn[0].print();


    // 输出最终结果的某些值作为示例
    //cout << "Final temperature at center: " << vec2D[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
