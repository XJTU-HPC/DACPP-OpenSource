#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// 网格参数

const int NX = 8;    // x方向网格数量
const int NY = 8;    // y方向网格数量
const double Lx = 10.0f; // x方向长度
const double Ly = 10.0f; // y方向长度
const double c = 1.0f;   // 波速
const int TIME_STEPS = 10; // 时间步数
// 网格步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// CFL条件
const double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void waveEq(const double* cur,const double* prev,double* next,int cur_0,int cur_1,int prev_0,int prev_1,int next_0,int next_1,int cur_0_shape,int cur_1_shape,int prev_0_shape,int prev_1_shape,int next_0_shape,int next_1_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_cur_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_prev_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_next_acc) {
    double dt = 0.5F * std::fmin(dx, dy) / c;
    double u_xx = (cur[(2 + cur_0) * cur_1_shape + (1 + cur_1)] - 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] + cur[(0 + cur_0) * cur_1_shape + (1 + cur_1)]) / (dx * dx);
    double u_yy = (cur[(1 + cur_0) * cur_1_shape + (2 + cur_1)] - 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] + cur[(1 + cur_0) * cur_1_shape + (0 + cur_1)]) / (dy * dy);
    next[(0 + next_0) * next_1_shape + (0 + next_1)] = 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] - prev[(0 + prev_0) * prev_1_shape + (0 + prev_1)] + (c * c) * dt * dt * (u_xx + u_yy);
}


// 生成函数调用
void waveEqShell_waveEq(dacpp::Matrix<double> & matCur, dacpp::Matrix<double> & matPrev, dacpp::Matrix<double> & matNext) { 
    using namespace sycl;
    // 设备选择
    auto selector = default_selector_v;
    sycl::queue dacpp_q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_matCur;
    info_matCur.dim = matCur.getDim();
    int info_matCur_Shape[2] = {0};
    for(int i = 0; i < info_matCur.dim; i++)
    {
        info_matCur.dimLength.push_back(matCur.getShape(i));
        info_matCur_Shape[i] = matCur.getShape(i);
    }
	
    // 数据信息初始化
    DataInfo info_matPrev;
    info_matPrev.dim = matPrev.getDim();
    int info_matPrev_Shape[2] = {0};
    for(int i = 0; i < info_matPrev.dim; i++)
    {
        info_matPrev.dimLength.push_back(matPrev.getShape(i));
        info_matPrev_Shape[i] = matPrev.getShape(i);
    }
	
    // 数据信息初始化
    DataInfo info_matNext;
    info_matNext.dim = matNext.getDim();
    int info_matNext_Shape[2] = {0};
    for(int i = 0; i < info_matNext.dim; i++)
    {
        info_matNext.dimLength.push_back(matNext.getShape(i));
        info_matNext_Shape[i] = matNext.getShape(i);
    }
	
    // 规则分区算子初始化
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.setDimId(0);
    sp1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp1,info_matCur));

    // 规则分区算子初始化
    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.setDimId(1);
    sp2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp2,info_matCur));

    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matPrev));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matPrev));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops matCur_Ops;
    
    sp1.setDimId(0);
    matCur_Ops.push_back(sp1);

    sp2.setDimId(1);
    matCur_Ops.push_back(sp2);


    // 算子组初始化
    Dac_Ops matPrev_Ops;
    
    idx1.setDimId(0);
    matPrev_Ops.push_back(idx1);

    idx2.setDimId(1);
    matPrev_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops matNext_Ops;
    
    idx1.setDimId(0);
    matNext_Ops.push_back(idx1);

    idx2.setDimId(1);
    matNext_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    sp1.setDimId(0);
    In_Ops.push_back(sp1);

    sp2.setDimId(1);
    In_Ops.push_back(sp2);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


	
	
	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    // 设备内存分配
    
    // 数据关联计算
    
	
    // 数据移动
    double* h_matCur = (double*)malloc(matCur.getSize()*sizeof(double));
    matCur.tensor2Array(h_matCur);
	buffer<double, 1> r_matCur(h_matCur, range<1>(matCur.getSize()));

    // 数据移动
    double* h_matPrev = (double*)malloc(matPrev.getSize()*sizeof(double));
    matPrev.tensor2Array(h_matPrev);
	buffer<double, 1> r_matPrev(h_matPrev, range<1>(matPrev.getSize()));

    // 数据移动
    double* h_matNext = (double*)malloc(matNext.getSize()*sizeof(double));

    // 数据重组
    
    
    // 数据算子组初始化
    Dac_Ops matCur_ops;
    
    sp1.setDimId(0);
    matCur_ops.push_back(sp1);
    sp2.setDimId(1);
    matCur_ops.push_back(sp2);


	std::vector<int> info_partition_matCur=para_gene_tool.init_partition_data_shape(info_matCur,matCur_ops);
    sycl::buffer<int> info_partition_matCur_buffer(info_partition_matCur.data(), sycl::range<1>(info_partition_matCur.size()));

    // 数据重组
    
    
    // 数据算子组初始化
    Dac_Ops matPrev_ops;
    
    idx1.setDimId(0);
    matPrev_ops.push_back(idx1);
    idx2.setDimId(1);
    matPrev_ops.push_back(idx2);


	std::vector<int> info_partition_matPrev=para_gene_tool.init_partition_data_shape(info_matPrev,matPrev_ops);
    sycl::buffer<int> info_partition_matPrev_buffer(info_partition_matPrev.data(), sycl::range<1>(info_partition_matPrev.size()));

    // 数据重组
    
    // 数据算子组初始化
    Dac_Ops matNext_ops;
    
    idx1.setDimId(0);
    matNext_ops.push_back(idx1);
    idx2.setDimId(1);
    matNext_ops.push_back(idx2);


    auto r_matNext = std::make_unique<sycl::buffer<double, 1>>(h_matNext,sycl::range<1>(matNext.getSize()));
    r_matNext->set_final_data(h_matNext);

	std::vector<int> info_partition_matNext=para_gene_tool.init_partition_data_shape(info_matNext,matNext_ops);
    sycl::buffer<int> info_partition_matNext_buffer(info_partition_matNext.data(), sycl::range<1>(info_partition_matNext.size()));
    
	
	
    sycl::device device = dacpp_q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];

	// 二维划分（可测试三维拓展）
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);

    // 固定 local 为 16*16,但受设备上限约束
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);

    // 对齐 global 到 local 的整数倍（防止越界）
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);

    //队列提交命令组
    dacpp_q.submit([&](handler &h) {
    
        accessor<double, 1, access::mode::read> acc_matCur(r_matCur, h);
        r_matCur.set_final_data(nullptr);
        
        accessor<double, 1, access::mode::read> acc_matPrev(r_matPrev, h);
        r_matPrev.set_final_data(nullptr);
        
        accessor<double, 1, sycl::access::mode::discard_write> acc_matNext(*r_matNext, h);
    
        auto info_partition_matCur_accessor = info_partition_matCur_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matPrev_accessor = info_partition_matPrev_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matNext_accessor = info_partition_matNext_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id+(0))%sp2.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
			// 获得划分数据单元左上角（第一个元素）的位置
			
			const auto matCur_0 = sp1_ * sp1.stride;
			const auto matCur_1 = sp2_ * sp2.stride;
			const auto matPrev_0 = idx1_ * idx1.stride;
			const auto matPrev_1 = idx2_ * idx2.stride;
			const auto matNext_0 = idx1_ * idx1.stride;
			const auto matNext_1 = idx2_ * idx2.stride;
            // 获得accessor指针
            
            auto* d_matCur = acc_matCur.get_multi_ptr<access::decorated::no>().get();
            auto* d_matPrev = acc_matPrev.get_multi_ptr<access::decorated::no>().get();
            auto* d_matNext = acc_matNext.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            waveEq(d_matCur,d_matPrev,d_matNext,matCur_0,matCur_1,matPrev_0,matPrev_1,matNext_0,matNext_1,info_matCur_Shape[0],info_matCur_Shape[1],info_matPrev_Shape[0],info_matPrev_Shape[1],info_matNext_Shape[0],info_matNext_Shape[1],info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);
        });
    }).wait();
    

	
    //结果返回语句改为析构语句
    r_matNext.reset();
    matNext.array2Tensor(h_matNext);

	

}

int main() {
    // 初始化波场
    vector<double> u_prev(NX * NY, 0.0f); // 前一步
    vector<double> u_curr(NX * NY, 0.0f);  // 当前步
    vector<double> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如一个高斯脉冲
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
        }
    }

    // for(int i = 0; i < 1; i++){
    //     for(int j = 0; j < NY; j++){
    //         //std::cout << u_prev[i * NX + j] << " " ;
    //     }
    //     //std::cout << std::endl;
    // }

    dacpp::Matrix<double> matCur({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matPrev = u_prev_tensor[{1,NX-1}][{1,NY-1}];
    dacpp::Matrix<double> matNext = u_next_tensor[{1,NX-1}][{1,NY-1}];
    
    for(int i = 0;i < TIME_STEPS; i++) {
        waveEqShell_waveEq(matCur, matPrev, matNext);
        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                matPrev[i-1][j-1]=matCur[i][j];
            }
        }

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                matCur[i][j]=matNext[i-1][j-1];
            }
        }
        // 处理边界条件（绝热边界：导数为零）
        for (int i = 0; i <= NX-1; ++i) {       
            matCur[i][NY-1]=0;
            matCur[i][0]=0;
        }
        for (int j = 0; j <= NY-1; ++j) {
            matCur[NX - 1][j]=0;
            matCur[0][j]=0;
             // 底部边界
        }
    }
    //
    matCur.print(); 
    return 0;
}

// #include <iostream>
// #include <vector>
// #include <cmath>
// #include "ReconTensor.h"

// using namespace std;

// namespace dacpp {
//     typedef std::vector<std::any> list;
// }
// // 网格参数
// const int NX = 8;    // x方向网格数量
// const int NY = 8;    // y方向网格数量
// const double Lx = 10.0f; // x方向长度
// const double Ly = 10.0f; // y方向长度
// const double c = 1.0f;   // 波速
// const int TIME_STEPS = 1000; // 时间步数
// // 网格步长
// const double dx = Lx / (NX - 1);
// const double dy = Ly / (NY - 1);

// // CFL条件
// const double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件

// shell dacpp::list waveEqShell(const dacpp::Matrix<double>& matCur, 
//                                 const dacpp::Matrix<double>& matPrev, 
//                                 dacpp::Matrix<double>& matNext) {
//     dacpp::split sp1(3, 1), sp2(3, 1);
//     dacpp::index idx1, idx2;
//     binding(sp1, idx1);
//     binding(sp2, idx2);
//     dacpp::list dataList{matCur[sp1][sp2], matPrev[idx1][idx2], matNext[idx1][idx2]};
//     return dataList;
// }

// calc void waveEq(dacpp::Matrix<double>& cur, double* prev, double* next) {
//     double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件
//     double u_xx = (cur[2][1] - 2.0f * cur[1][1] + cur[0][1])/ (dx * dx);
//     double u_yy = (cur[1][2] - 2.0f * cur[1][1] + cur[1][0])/ (dy * dy);
//     next[0]=2.0f*cur[1][1]-prev[0]+(c * c)*dt*dt*(u_xx+u_yy);
// }

// int main() {
//     // 初始化波场
//     vector<double> u_prev(NX * NY, 0.0f); // 前一步
//     vector<double> u_curr(NX * NY, 0.0f);  // 当前步
//     vector<double> u_next(NX * NY, 0.0f);  // 当前步

//     // 初始条件：例如一个高斯脉冲
//     int cx = NX / 2;
//     int cy = NY / 2;
//     double sigma = 0.5f;
//     for(int i = 0; i < NX; ++i) {
//         for(int j = 0; j < NY; ++j) {
//             double x = i * dx;
//             double y = j * dy;
//             u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
//         }
//     }

//     // for(int i = 0; i < 1; i++){
//     //     for(int j = 0; j < NY; j++){
//     //         std::cout << u_prev[i * NX + j] << " " ;
//     //     }
//     //     std::cout << std::endl;
//     // }

//     dacpp::Matrix<double> matCur({NX, NY}, u_curr);
//     dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
//     dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
//     dacpp::Matrix<double> matPrev = u_prev_tensor[{1,7}][{1,7}];
//     dacpp::Matrix<double> matNext = u_next_tensor[{1,7}][{1,7}];
//         // 使用 dac_for 包裹计算表达式和指针操作代码块
//     dacpp::dac_for(TIME_STEPS, [&](int STEPS) {
//         for(int i = 0; i < STEPS; i++){
//             waveEqShell(matCur, matPrev, matNext) <-> waveEq;
//             dacpp::swap(matPrev, matCur[{1,7}][{1,7}]);
//             dacpp::swap(matCur[{1,7}][{1,7}], matNext);
//         }
//     });
//     for (int i = 1; i <= NX-2; i++) {
//         for(int j = 1; j <=NY-2; j++){
//             matPrev[i-1][j-1]=matCur[i][j];
//         }
//     }
//     for (int i = 1; i <= NX-2; i++) {
//         for(int j = 1; j <=NY-2; j++){
//             matCur[i][j]=matNext[i-1][j-1];
//         }
//     }
//     // 处理边界条件（绝热边界：导数为零）
//     for (int i = 0; i < NX; ++i) {       
//         matCur[i][NY-1]=0;
//         matCur[i][0]=0;
//     }
//     for (int j = 0; j < NY; ++j) {
//         matCur[NX - 1][j]=0;
//         matCur[0][j]=0;
//             // 底部边界
//     }    
//    matCur.print(); 
//     return 0;
// }

