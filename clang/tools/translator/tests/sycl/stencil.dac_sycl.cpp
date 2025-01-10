#include <iostream>
#include <vector>
#include <cmath>
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"


using namespace std;
using dacpp::Tensor;
namespace dacpp {
    typedef std::vector<std::any> list;
}

// 网格参数
const int NX = 32;           // x方向网格数量
const int NY = 32;           // y方向网格数量
const float Lx = 10.0f;       // x方向长度
const float Ly = 10.0f;       // y方向长度
const float alpha = 0.01f;    // 热扩散系数
const int TIME_STEPS = 1000;  // 时间步数





#include <sycl/sycl.hpp>
#include "/data/qinian/ice/dacpp/clang/tools/translator/dpcppLib/include/DataReconstructor.old.h"

using namespace sycl;

void stencil(float* mat, float* out) 
{
    out[0] = mat[1 * 32 + 1] + (mat[0 * 32 + 1] + mat[2 * 32 + 1] + mat[1 * 32 + 0] + mat[1 * 32 + 2] - mat[1 * 32 + 1]);
}


// 生成函数调用
void stencilShell(const dacpp::Tensor<float> & matIn, Tensor<float> & matOut) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    
    // 规则分区算子初始化
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.SetSplitSize(30);
    // 规则分区算子初始化
    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.SetSplitSize(30);
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.SetSplitSize(30);
    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.SetSplitSize(30);
    // 数据重组
    
    // 数据重组
    DataReconstructor<float> matIn_tool;
    float* r_matIn=(float*)malloc(sizeof(float)*8100);
    
    // 数据算子组初始化
    Dac_Ops matIn_ops;

    sp1.setDimId(0);
    sp1.setSplitLength(270);
    matIn_ops.push_back(sp1);
    sp2.setDimId(1);
    sp2.setSplitLength(9);
    matIn_ops.push_back(sp2);
    
    matIn_tool.init(matIn,matIn_ops);
    matIn_tool.Reconstruct(r_matIn);
    // 数据重组
    DataReconstructor<float> matOut_tool;
    float* r_matOut=(float*)malloc(sizeof(float)*900);
    
    // 数据算子组初始化
    Dac_Ops matOut_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(30);
    matOut_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(1);
    matOut_ops.push_back(idx2);
    matOut_tool.init(matOut,matOut_ops);
    matOut_tool.Reconstruct(r_matOut);
    // 设备内存分配
    
    // 设备内存分配
    float *d_matIn=malloc_device<float>(8100,q);
    // 设备内存分配
    float *d_matOut=malloc_device<float>(900,q);
    // 数据移动
    
    // 数据移动
    q.memcpy(d_matIn,r_matIn,8100*sizeof(float)).wait();   
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 900);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto sp1=(item_id/30+(0)+30)%30;
            const auto idx1=(item_id/30+( + 1)+30)%30;
            const auto sp2=(item_id+(0)+30)%30;
            const auto idx2=(item_id+( + 1)+30)%30;
            // 嵌入计算
			
            stencil(d_matIn+(sp1*270+sp2*9),d_matOut+(idx1*30+idx2*1));
        });
    }).wait();
    
    
    // 归约
    
    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(r_matOut, d_matOut, 900*sizeof(float)).wait();
    matOut = matOut_tool.UpdateData(r_matOut);
    // 内存释放
    
    sycl::free(d_matIn, q);
    sycl::free(d_matOut, q);
}

int main() {
    // 空间步长
    float dx = Lx / (NX - 1);
    float dy = Ly / (NY - 1);

    // 稳定性条件
    float dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
    float delta_t = 0.4f * dt_stability; // 选择一个更严格的时间步长以确保稳定性

    cout << "Grid size: " << NX << "x" << NY << "\n";
    cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // 初始化温度场
    vector<float> u_prev(NX * NY, 0.0f); // 前一步（在热传导方程中，只有当前步和上一步）
    vector<float> u_curr(NX * NY, 0.0f);  // 当前步
    vector<float> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如，中心有一个高斯分布的热源
    int cx = NX / 2;
    int cy = NY / 2;
    float sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            float x = i * dx;
            float y = j * dy;
            // 高斯分布
            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }

    for (int i = 0; i < NX; ++i) {  // 遍历每一行
        for (int j = 0; j < NY; ++j) {  // 遍历每一列
            //std::cout << u_curr[i * NY + j] << " ";  // 打印当前元素
        }
        //std::cout << std::endl;  // 每行结束后换行
    }

    std::vector<int> shape = {32, 32};
    Tensor<float> u_curr_tensor(u_curr, shape);

    for(int i=0;i<TIME_STEPS;i++) {
        std::vector<float> middle_points(NX * NY, 0.0f);
        for (int i = 1; i <= 30; i++) {
            std::vector<float> row;
            for (int j = 1; j <= 30; j++) {  
                middle_points.push_back(static_cast<float>(u_next[i*NY+j]));  
            }
            
        }
        std::vector<int> shape2 = {30, 30};
        Tensor<float> middle_tensor(middle_points, shape2);

        stencilShell(u_curr_tensor, middle_tensor);

        for (int i = 1; i <= 30; i++) {
            for(int j = 1; j <=30; j++){
                u_curr_tensor[i][j] = middle_tensor[i-1][j-1];
            }
        }

        // 处理边界条件（绝热边界：导数为零）
        for (int j = 0; j < NY; ++j) {
            u_curr_tensor[0 * NY + j] = u_curr_tensor[1 * NY + j];              // 顶部边界
            u_curr_tensor[(NX - 1) * NY + j] = u_curr_tensor[(NX - 2) * NY + j]; // 底部边界
        }
        for (int i = 0; i < NX; ++i) {
            u_curr_tensor[i * NY + 0] = u_curr_tensor[i * NY + 1];              // 左边界
            u_curr_tensor[i * NY + (NY - 1)] = u_curr_tensor[i * NY + (NY - 2)]; // 右边界
        }
        //u_curr_tensor.print();
    }
    //u_curr_tensor.print();

    float* tmp1 = new float[1024];
    u_curr_tensor.tensor2Array(tmp1);
    for (int i = 0; i < 1; i++) {
        for(int j = 0; j <NY; j++){
            std::cout<<tmp1[(i)*NX+(j)]<<" ";
        }
        std::cout<<"\n";
    }
    // float* data = new float[32 * 32];
    // u_curr_tensor.tensor2Array(data);

    // std::vector<std::vector<int>> vec2D;
    // vec2D.resize(32, std::vector<int>(32));

    // // 将一维数组的数据填充到二维数组中
    // for (int i = 0; i < 32; ++i) {
    //     for (int j = 0; j < 32; ++j) {
    //         vec2D[i][j] = data[i * 32 + j];
    //     }
    // }
    // for (const auto& row : vec2D) {  // 遍历二维vector的每一行
    //     for (int val : row) {  // 遍历每一行中的元素
    //         //std::cout << val << " ";  // 打印元素
    //     }
    //     std::cout << std::endl;  // 每行结束后换行
    // }




    // 输出最终结果的某些值作为示例
    //cout << "Final temperature at center: " << vec2D[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}


