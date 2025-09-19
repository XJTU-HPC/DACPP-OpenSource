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
const int TIME_STEPS = 1000; // 时间步数
// 网格步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// CFL条件
const double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void waveEq(double* cur,double* prev,double* next,sycl::accessor<int, 1, sycl::access::mode::read_write> info_cur_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_prev_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_next_acc) 
{
    double dt = 0.5F * std::fmin(dx, dy) / c;
    double u_xx = (cur[2*info_cur_acc[1]+1] - 2.F * cur[1*info_cur_acc[1]+1] + cur[0*info_cur_acc[1]+1]) / (dx * dx);
    double u_yy = (cur[1*info_cur_acc[1]+2] - 2.F * cur[1*info_cur_acc[1]+1] + cur[1*info_cur_acc[1]+0]) / (dy * dy);
    next[0] = 2.F * cur[1*info_cur_acc[1]+1] - prev[0] + (c * c) * dt * dt * (u_xx + u_yy);
}


// 生成函数调用
void waveEqShell_waveEq(const dacpp::Matrix<double> & matCur, const dacpp::Matrix<double> & matPrev, dacpp::Matrix<double> & matNext) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_matCur;
    info_matCur.dim = matCur.getDim();
    for(int i = 0; i < info_matCur.dim; i++) info_matCur.dimLength.push_back(matCur.getShape(i));
	
    // 数据信息初始化
    DataInfo info_matPrev;
    info_matPrev.dim = matPrev.getDim();
    for(int i = 0; i < info_matPrev.dim; i++) info_matPrev.dimLength.push_back(matPrev.getShape(i));
	
    // 数据信息初始化
    DataInfo info_matNext;
    info_matNext.dim = matNext.getDim();
    for(int i = 0; i < info_matNext.dim; i++) info_matNext.dimLength.push_back(matNext.getShape(i));
	
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


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);

    idx2.setDimId(1);
    Reduction_Ops.push_back(idx2);


	
    //生成设备内存分配大小
    int matCur_Size = para_gene_tool.init_device_memory_size(info_matCur,matCur_Ops);

    //生成设备内存分配大小
    int matPrev_Size = para_gene_tool.init_device_memory_size(info_matPrev,matPrev_Ops);

    //生成设备内存分配大小
    int matNext_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_matNext);

    //生成设备内存分配大小
    int matNextReduction_Size = para_gene_tool.init_device_memory_size(info_matNext,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matCur_Ops,matCur_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matPrev_Ops,matPrev_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,matNext_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(matCur_Ops);

    ops_s.push_back(matPrev_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[3][2] = {0};
    para_gene_tool.init_split_length_martix(3,2,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // Buffer设备内存分配
    buffer <double> b_matCur{matCur_Size};

    // Buffer设备内存分配
    buffer <double> b_matPrev{matPrev_Size};

    // Buffer设备内存分配
    buffer <double> b_matNext{matNext_Size};

    // 规约Buffer设备内存分配
    std::vector<sycl::buffer<double, 1>> b_reduction_matNext(matNextReduction_Size, buffer<double, 1>{1});
    for(int i = 0; i < matNextReduction_Size; i++){
        host_accessor temp_accessor{b_reduction_matNext[i]};
        temp_accessor[0] = 0;
    }
    // 数据关联计算
    
	
    // 数据移动
    double* h_matCur = (double*)malloc(matCur.getSize()*sizeof(double));
    matCur.tensor2Array(h_matCur);
    {
        host_accessor temp_accessor{b_matCur};
        for(int i = 0; i < matCur_Size; i++){
            temp_accessor[i] = h_matCur[i];
        }
    }

    // 数据移动
    double* h_matPrev = (double*)malloc(matPrev.getSize()*sizeof(double));
    matPrev.tensor2Array(h_matPrev);
    {
        host_accessor temp_accessor{b_matPrev};
        for(int i = 0; i < matPrev_Size; i++){
            temp_accessor[i] = h_matPrev[i];
        }
    }

    // 数据移动
    double* h_matNext = (double*)malloc(matNext.getSize()*sizeof(double));

    // 数据重组
    DataReconstructor<double> matCur_tool;
    
    // 数据算子组初始化
    Dac_Ops matCur_ops;
    
    sp1.setDimId(0);
    matCur_ops.push_back(sp1);
    sp2.setDimId(1);
    matCur_ops.push_back(sp2);

    matCur_tool.init(info_matCur,matCur_ops);
    buffer<double> r_matCur{matCur_Size};
    matCur_tool.Reconstruct(r_matCur,b_matCur,q);
	std::vector<int> info_partition_matCur=para_gene_tool.init_partition_data_shape(info_matCur,matCur_ops);
    sycl::buffer<int> info_partition_matCur_buffer(info_partition_matCur.data(), sycl::range<1>(info_partition_matCur.size()));

    // 数据重组
    DataReconstructor<double> matPrev_tool;
    
    // 数据算子组初始化
    Dac_Ops matPrev_ops;
    
    idx1.setDimId(0);
    matPrev_ops.push_back(idx1);
    idx2.setDimId(1);
    matPrev_ops.push_back(idx2);

    matPrev_tool.init(info_matPrev,matPrev_ops);
    buffer<double> r_matPrev{matPrev_Size};
    matPrev_tool.Reconstruct(r_matPrev,b_matPrev,q);
	std::vector<int> info_partition_matPrev=para_gene_tool.init_partition_data_shape(info_matPrev,matPrev_ops);
    sycl::buffer<int> info_partition_matPrev_buffer(info_partition_matPrev.data(), sycl::range<1>(info_partition_matPrev.size()));

    // 数据重组
    DataReconstructor<double> matNext_tool;
    
    // 数据算子组初始化
    Dac_Ops matNext_ops;
    
    idx1.setDimId(0);
    matNext_ops.push_back(idx1);
    idx2.setDimId(1);
    matNext_ops.push_back(idx2);

    matNext_tool.init(info_matNext,matNext_ops);
    buffer<double> r_matNext{matNext_Size};
    matNext_tool.Reconstruct(r_matNext,b_matNext,q);
	std::vector<int> info_partition_matNext=para_gene_tool.init_partition_data_shape(info_matNext,matNext_ops);
    sycl::buffer<int> info_partition_matNext_buffer(info_partition_matNext.data(), sycl::range<1>(info_partition_matNext.size()));
    
	
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

    //队列提交命令组
    q.submit([&](handler &h) {
    
        accessor acc_matCur{r_matCur, h};
        accessor acc_matPrev{r_matPrev, h};
        accessor acc_matNext{r_matNext, h};
    
        auto info_partition_matCur_accessor = info_partition_matCur_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_matPrev_accessor = info_partition_matPrev_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_matNext_accessor = info_partition_matNext_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id+(0))%sp2.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 获得accessor指针
            
            auto* d_matCur = acc_matCur.get_multi_ptr<access::decorated::no>().get();
            auto* d_matPrev = acc_matPrev.get_multi_ptr<access::decorated::no>().get();
            auto* d_matNext = acc_matNext.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            waveEq(d_matCur+(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),d_matPrev+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),d_matNext+(sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]),info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<matNextReduction_Size;i++) {
            q.submit([&](handler &h) {
                accessor d_matNext{r_matNext, h};
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(b_reduction_matNext[i], h, 
                sycl::plus<>(),property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_matNext[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_matNext};
            for(int i = 0; i < matNextReduction_Size; i++){
                host_accessor temp_accessor{b_reduction_matNext[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }


    //结果返回
    matNext_tool.UpdateData(r_matNext,b_matNext,q);
    {
        host_accessor temp_accessor{b_matNext};
        for(int i = 0; i < matNext_Size; i++){
            h_matNext[i] = temp_accessor[i];
        }
    }
    matNext.array2Tensor(h_matNext);

	

    // 内存释放
    
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

    dacpp::Matrix<double> u_curr_tensor({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> u_prev_middle_tensor = u_prev_tensor[{1,NX-1}][{1,NY-1}];
    for(int i = 0;i < TIME_STEPS; i++) {
        dacpp::Matrix<double> u_next_middle_tensor = u_next_tensor[{1,NX-1}][{1,NY-1}];
        waveEqShell_waveEq(u_curr_tensor, u_prev_middle_tensor, u_next_middle_tensor);
        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                u_prev_middle_tensor[i-1][j-1]=u_curr_tensor[i][j];
            }
        }

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                u_curr_tensor[i][j]=u_next_middle_tensor[i-1][j-1];
            }
        }
        // 处理边界条件（绝热边界：导数为零）
        for (int i = 0; i < NX; ++i) {       
            u_curr_tensor[i][NY-1]=0;
            u_curr_tensor[i][0]=0;
        }
        for (int j = 0; j < NY; ++j) {
            u_curr_tensor[NX - 1][j]=0;
            u_curr_tensor[0][j]=0;
             // 底部边界
        }

        
    }
    u_curr_tensor.print(); 
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

