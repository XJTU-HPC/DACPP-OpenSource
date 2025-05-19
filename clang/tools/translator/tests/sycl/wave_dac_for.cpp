#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
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
#include "DataReconstructor.new.h"
#include "ParameterGeneration.h"
#include "utils.h"

using namespace sycl;

void waveEq(double* cur,
            double* prev,
            double* next,
            int cur_index,
            int prev_index,
            int next_index,
            VirtualMapParams cur_params,
            VirtualMapParams prev_params,
            VirtualMapParams next_params,
            sycl::accessor<int, 1, sycl::access::mode::read_write> info_cur_acc,
            sycl::accessor<int, 1, sycl::access::mode::read_write> info_prev_acc,
            sycl::accessor<int, 1, sycl::access::mode::read_write> info_next_acc) 
{
    double dt = 0.5F * std::fmin(dx, dy) / c;
    double u_xx = (virtual_to_physical(cur, cur_index+2*info_cur_acc[1]+1, cur_params) - 2.F * virtual_to_physical(cur, cur_index+1*info_cur_acc[1]+1, cur_params) + virtual_to_physical(cur, cur_index+0*info_cur_acc[1]+1, cur_params)) / (dx * dx);
    double u_yy = (virtual_to_physical(cur, cur_index+1*info_cur_acc[1]+2, cur_params) - 2.F * virtual_to_physical(cur, cur_index+1*info_cur_acc[1]+1, cur_params) + virtual_to_physical(cur, cur_index+1*info_cur_acc[1]+0, cur_params)) / (dy * dy);
    virtual_to_physical(next, next_index+0, next_params) = 2.F * virtual_to_physical(cur, cur_index+1*info_cur_acc[1]+1, cur_params) - virtual_to_physical(prev, prev_index+0, prev_params) + (c * c) * dt * dt * (u_xx + u_yy);
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
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_matNext,Reduction_Ops);

	
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
    
    // 设备内存分配
    double *d_matCur=malloc_device<double>(matCur.getSize(),q);
    // 设备内存分配
    double *d_matPrev=malloc_device<double>(matPrev.getSize(),q);
    // 设备内存分配
    double *d_matNext=malloc_device<double>(matNext.getSize(),q);
    // 归约设备内存分配
    double *reduction_matNext = malloc_device<double>(Reduction_Size,q);

    // 数据移动
    double* h_matCur = (double*)malloc(matCur.getSize()*sizeof(double));
    matCur.tensor2Array(h_matCur);
    q.memcpy(d_matCur,h_matCur,matCur.getSize()*sizeof(double)).wait();
    // 数据移动
    double* h_matPrev=(double*)malloc(matPrev.getSize()*sizeof(double));
    matPrev.tensor2Array(h_matPrev);
    q.memcpy(d_matPrev,h_matPrev,matPrev.getSize()*sizeof(double)).wait();
    double* h_matNext=(double*)malloc(matNext.getSize()*sizeof(double));
    matPrev.tensor2Array(h_matNext);
    q.memset(d_matNext, 0, matNext.getSize()*sizeof(double)).wait();

    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<double> matCur_tool;
    
    // 数据算子组初始化
    Dac_Ops matCur_ops;
    
    sp1.setDimId(0);
    sp1.setSplitLength(8);
    matCur_ops.push_back(sp1);
    sp2.setDimId(1);
    sp2.setSplitLength(8);
    matCur_ops.push_back(sp2);
    
    matCur_tool.init(info_matCur,matCur_ops);    

	std::vector<int> info_partition_matCur=para_gene_tool.init_partition_data_shape(info_matCur,matCur_ops);
    sycl::buffer<int> info_partition_matCur_buffer(info_partition_matCur.data(), sycl::range<1>(info_partition_matCur.size()));
    
    // 数据重组
    DataReconstructor<double> matPrev_tool;
    
    // 数据算子组初始化
    Dac_Ops matPrev_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    matPrev_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    matPrev_ops.push_back(idx2);

    matPrev_tool.init(info_matPrev,matPrev_ops);

	std::vector<int> info_partition_matPrev=para_gene_tool.init_partition_data_shape(info_matPrev,matPrev_ops);
    sycl::buffer<int> info_partition_matPrev_buffer(info_partition_matPrev.data(), sycl::range<1>(info_partition_matPrev.size()));
    // 数据重组
    DataReconstructor<double> matNext_tool;
    
    // 数据算子组初始化
    Dac_Ops matNext_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    matNext_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    matNext_ops.push_back(idx2);

    matNext_tool.init(info_matNext,matNext_ops);

    double *r_matNext=malloc_device<double>(matNext_Size,q);

	std::vector<int> info_partition_matNext=para_gene_tool.init_partition_data_shape(info_matNext,matNext_ops);
    sycl::buffer<int> info_partition_matNext_buffer(info_partition_matNext.data(), sycl::range<1>(info_partition_matNext.size()));
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
    
    std::vector<int> start_offset(info_matCur.dim);
    for(int i=0;i<info_matCur.dim;i++) {
        start_offset[i] = 1;
    }
    
    for(int step = 0; step < TIME_STEPS; step++) {
        //队列提交命令组
        q.submit([&](handler &h) {
            // 访问器初始化
            auto info_partition_matCur_accessor = info_partition_matCur_buffer.get_access<sycl::access::mode::read_write>(h);
            auto info_partition_matPrev_accessor = info_partition_matPrev_buffer.get_access<sycl::access::mode::read_write>(h);
            auto info_partition_matNext_accessor = info_partition_matNext_buffer.get_access<sycl::access::mode::read_write>(h);
            VirtualMapParams matCur_params {
                .block_size = matCur_tool.block_size,
                .dim_num = matCur_tool.dim_num,
                .start = matCur_tool.start_buffer.get_access<sycl::access::mode::read_write>(h),
                .data_shape = matCur_tool.data_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .data_stride = matCur_tool.data_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .view_shape = matCur_tool.view_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .view_stride = matCur_tool.view_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_shape = matCur_tool.block_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_stride = matCur_tool.block_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_move = matCur_tool.block_move_buffer.get_access<sycl::access::mode::read_write>(h),
                .grid_shape = matCur_tool.grid_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .grid_stride = matCur_tool.grid_stride_buffer.get_access<sycl::access::mode::read_write>(h),
            };
            VirtualMapParams matPrev_params {
                .block_size = matPrev_tool.block_size,
                .dim_num = matPrev_tool.dim_num,
                .start = matPrev_tool.start_buffer.get_access<sycl::access::mode::read_write>(h),
                .data_shape = matPrev_tool.data_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .data_stride = matPrev_tool.data_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .view_shape = matCur_tool.view_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .view_stride = matCur_tool.view_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_shape = matPrev_tool.block_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_stride = matPrev_tool.block_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_move = matPrev_tool.block_move_buffer.get_access<sycl::access::mode::read_write>(h),
                .grid_shape = matPrev_tool.grid_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .grid_stride = matPrev_tool.grid_stride_buffer.get_access<sycl::access::mode::read_write>(h),
            };
            VirtualMapParams matNext_params {
                .block_size = matNext_tool.block_size,
                .dim_num = matNext_tool.dim_num,
                .start = matNext_tool.start_buffer.get_access<sycl::access::mode::read_write>(h),
                .data_shape = matNext_tool.data_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .data_stride = matNext_tool.data_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .view_shape = matCur_tool.view_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .view_stride = matCur_tool.view_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_shape = matNext_tool.block_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_stride = matNext_tool.block_stride_buffer.get_access<sycl::access::mode::read_write>(h),
                .block_move = matNext_tool.block_move_buffer.get_access<sycl::access::mode::read_write>(h),
                .grid_shape = matNext_tool.grid_shape_buffer.get_access<sycl::access::mode::read_write>(h),
                .grid_stride = matNext_tool.grid_stride_buffer.get_access<sycl::access::mode::read_write>(h),
            };
            h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                // 索引初始化
                
                const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
                const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
                const auto sp2_=(item_id+(0))%sp2.split_size;
                const auto idx2_=(item_id+(0))%idx2.split_size;
                // 嵌入计算
                
                waveEq(d_matCur, d_matPrev, d_matNext, (sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]), (idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]), (sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]), matCur_params, matPrev_params, matNext_params, info_partition_matCur_accessor, info_partition_matPrev_accessor, info_partition_matNext_accessor);
            });
        }).wait();

        if(step < TIME_STEPS - 1)
        {
            swap(d_matPrev, d_matCur);
            swap(d_matCur, d_matNext);
            swap(matPrev_tool.data_stride_buffer, matCur_tool.data_stride_buffer);
            swap(matCur_tool.data_stride_buffer, matNext_tool.data_stride_buffer);
            swap(matPrev_tool.data_shape_buffer, matCur_tool.data_shape_buffer);
            swap(matCur_tool.data_shape_buffer, matNext_tool.data_shape_buffer);
            
            swap(matPrev_tool.start, matCur_tool.start);
            swap(matCur_tool.start, matNext_tool.start);

            matPrev_tool.add_start(start_offset);
            matCur_tool.sub_start(start_offset);

            matNext_tool.init(info_matNext,matNext_ops);
            sycl::free(d_matNext, q);
            d_matNext=malloc_device<double>(matNext.getSize(),q);
            q.memset(d_matNext, 0, matNext.getSize()*sizeof(double)).wait();
        }

    }

    q.memcpy(h_matNext, d_matNext, matNext_Size*sizeof(double)).wait();
    
    matNext.array2Tensor(h_matNext);

    // 内存释放
    
    sycl::free(d_matCur, q);
    sycl::free(d_matPrev, q);
    sycl::free(d_matNext, q);

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

    for(int i = 0; i < 1; i++){
        for(int j = 0; j < NY; j++){
            std::cout << u_prev[i * NX + j] << " " ;
        }
        std::cout << std::endl;
    }

    dacpp::Matrix<double> u_curr_tensor({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> u_prev_middle_tensor = u_prev_tensor[{1,7}][{1,7}];
    dacpp::Matrix<double> u_next_middle_tensor = u_next_tensor[{1,7}][{1,7}];

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
   u_curr_tensor.print(); 
    return 0;
}
