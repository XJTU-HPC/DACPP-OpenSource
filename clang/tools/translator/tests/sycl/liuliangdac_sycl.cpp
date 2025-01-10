#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"

using dacpp::Tensor;
namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 100;       // 路段长度
const double TIME_STEPS = 200;  // 时间步数
const double DELTA_T = 0.01; // 时间步长
const double DELTA_X = 1.0;  // 空间步长

// 流量函数，考虑密度对流量的影响
double q(double rho) {
    double V_max = 30; // 最大速度
    double rho_max = 50; // 最大密度
    return rho * V_max * (1 - rho / rho_max);
}

// 初始化密度，使用随机的分布
void initializeDensity(std::vector<double>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40; // 高密度区
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20; // 中密度区
        } else {
            rho[i] = 10; // 低密度区
        }
    }
}

// 计算交通流量




#include <sycl/sycl.hpp>
#include "/data/qinian/ice/dacpp/clang/tools/translator/dpcppLib/include/DataReconstructor.old.h"

using namespace sycl;

void lwr(double* rho, double* new_rho) 
{
    new_rho[0] = rho[0] - (DELTA_T / DELTA_X) * (q(rho[1]) - q(rho[0]));
    new_rho[0] = std::max(0., new_rho[0]);
}


// 生成函数调用
void LWR_shell(const dacpp::Tensor<double> & rho, dacpp::Tensor<double> & new_rho) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.SetSplitSize(98);
    // 规则分区算子初始化
    RegularSlice S1 = RegularSlice("S1", 2, 1);
    S1.SetSplitSize(98);
    // 数据重组
    
    // 数据重组
    DataReconstructor<double> rho_tool;
    double* r_rho=(double*)malloc(sizeof(double)*196);
    
    // 数据算子组初始化
    Dac_Ops rho_ops;
    
    rho_tool.init(rho,rho_ops);
    rho_tool.Reconstruct(r_rho);
    // 数据重组
    DataReconstructor<double> new_rho_tool;
    double* r_new_rho=(double*)malloc(sizeof(double)*98);
    
    // 数据算子组初始化
    Dac_Ops new_rho_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(1);
    new_rho_ops.push_back(idx1);
    new_rho_tool.init(new_rho,new_rho_ops);
    new_rho_tool.Reconstruct(r_new_rho);
    // 设备内存分配
    
    // 设备内存分配
    double *d_rho=malloc_device<double>(196,q);
    // 设备内存分配
    double *d_new_rho=malloc_device<double>(98,q);
    // 数据移动
    
    // 数据移动
    q.memcpy(d_rho,r_rho,196*sizeof(double)).wait();   
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 98);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx1=(item_id+(0)+98)%98;
            const auto S1=(item_id+(0)+98)%98;
            // 嵌入计算
			
            lwr(d_rho+(S1*2),d_new_rho+(idx1*1));
        });
    }).wait();
    
    
    // 归约
    
    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(r_new_rho, d_new_rho, 98*sizeof(double)).wait();
    new_rho = new_rho_tool.UpdateData(r_new_rho);
    // 内存释放
    
    sycl::free(d_rho, q);
    sycl::free(d_new_rho, q);
}

int main() {
    // 创建 Tensor 类型对象
    std::vector<double> rho(WIDTH, 0.0);
    std::vector<double> new_rho(WIDTH, 0.0);
    //std::vector<int> shape3 = {1, 100};
    //Tensor<int> new_rho(middle_points, shape3);

    initializeDensity(rho);
    for (int x = 0; x < WIDTH; ++x) {
        std::cout <<  static_cast<int>(rho[x]) <<",";
    }

    for (int t = 0; t < TIME_STEPS; ++t) {

        // 使用 LWR 算法
        std::vector<double> middle_points_out;
        for (int i = 1; i <= 98; i++) {
          middle_points_out.push_back(static_cast<double>(new_rho[i]));
        }
        std::vector<int> shape = {98,1};
        Tensor<double> middle_out_tensor(middle_points_out, shape);

        std::vector<double> middle_points_in;
        for (int i = 1; i <= 99; i++) {
          middle_points_in.push_back(static_cast<double>(rho[i]));
        }
        std::vector<int> shape2 = {99,1};
        Tensor<double> middle_in_tensor(middle_points_in, shape2);
        LWR_shell(middle_in_tensor, middle_out_tensor);

        double* r_1 = new double[98];
        middle_out_tensor.tensor2Array(r_1);
        std::vector<double> r(r_1, r_1 + 98);

        for (int i = 1; i <= 98; i++) {
            rho[i] = r[i-1];
        }
        
        // 处理边界条件
        rho[0] = rho[1]; // 左边界无车流
        rho[WIDTH - 1] = rho[WIDTH - 2]; // 右边界无车流

    }
    for (int x = 0; x < WIDTH; ++x) {
        std::cout <<  static_cast<int>(rho[x]) <<",";
    }
    //rho_tensor.print();

    // 释放动态分配的内存

    return 0;
}


