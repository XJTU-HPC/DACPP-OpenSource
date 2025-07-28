#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"

using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// 网格参数
const int NX = 8000;    // x方向网格数量
const int NY = 8000;    // y方向网格数量
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

void waveEq(double* cur,double* prev,double* next,int cur_index,int prev_index,int next_index, VirtualMapParams cur_params, VirtualMapParams prev_params, VirtualMapParams next_params,sycl::accessor<int, 1, sycl::access::mode::read_write> info_cur_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_prev_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_next_acc) 
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
    queue q(selector,{property::queue::enable_profiling()});
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化

    double total_ParameterGeneration = 0, total_memcpy = 0,total_DataReconstruct = 0;

    auto ParameterGeneration_start = std::chrono::high_resolution_clock::now();
    
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

    std::cout << "matCur_Size: " << matCur_Size << std::endl;
std::cout << "matPrev_Size: " << matPrev_Size << std::endl;
std::cout << "matNext_Size: " << matNext_Size << std::endl;
std::cout << "matNextReduction_Size: " << matNextReduction_Size << std::endl;

	
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
    
    // 归约设备内存分配
    double *reduction_matNext = malloc_device<double>(matNextReduction_Size,q);
    // 数据关联计算
    
	    
	
    // 设备内存分配
    double *d_matCur=malloc_device<double>(matCur_Size,q);
    // 设备内存分配
    double *d_matPrev=malloc_device<double>(matPrev_Size,q);
    // 设备内存分配
    double *d_matNext=malloc_device<double>(matNext_Size,q);

    auto ParameterGeneration_end = std::chrono::high_resolution_clock::now();
    total_ParameterGeneration += std::chrono::duration<double>(ParameterGeneration_end - ParameterGeneration_start).count();
    std::cout << "ParameterGenerationTime: " << total_ParameterGeneration << "s" << std::endl;

    auto memcpy_start = std::chrono::high_resolution_clock::now();
    // 数据移动
	double* h_matCur = (double*)malloc(matCur_Size*sizeof(double));
	matCur.tensor2Array(h_matCur);
    q.memcpy(d_matCur,h_matCur,matCur_Size*sizeof(double)).wait();

    // 数据移动
	double* h_matPrev = (double*)malloc(matPrev_Size*sizeof(double));
	matPrev.tensor2Array(h_matPrev);
    q.memcpy(d_matPrev,h_matPrev,matPrev_Size*sizeof(double)).wait();

    // 数据移动
	double* h_matNext = (double*)malloc(matNext_Size*sizeof(double));
	// matNext.tensor2Array(h_matNext);
    q.memset(d_matNext, 0, matNext_Size*sizeof(double)).wait();

    auto memcpy_end = std::chrono::high_resolution_clock::now();
    total_memcpy += std::chrono::duration<double>(memcpy_end - memcpy_start).count();
    std::cout << "MemcpyTime: " << total_memcpy << "s" << std::endl;

    auto DataReconstruct_start = std::chrono::high_resolution_clock::now();
    // 数据重组
    DataReconstructor<double> matCur_tool;
    
    // 数据算子组初始化
    Dac_Ops matCur_ops;
    
    sp1.setDimId(0);
    matCur_ops.push_back(sp1);
    sp2.setDimId(1);
    matCur_ops.push_back(sp2);

    matCur_tool.init(info_matCur,matCur_ops);
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
	std::vector<int> info_partition_matNext=para_gene_tool.init_partition_data_shape(info_matNext,matNext_ops);
    sycl::buffer<int> info_partition_matNext_buffer(info_partition_matNext.data(), sycl::range<1>(info_partition_matNext.size()));
    
    auto DataReconstruct_end = std::chrono::high_resolution_clock::now();
    total_DataReconstruct += std::chrono::duration<double>(DataReconstruct_end - DataReconstruct_start).count();
    std::cout << "DataReconstructTime: " << total_DataReconstruct << "s" << std::endl;
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    //int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    int work_group_size = (Item_Size + 256 - 1) / 256;
    std::cout << "work_group_size: " << work_group_size << std::endl;
    //sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> local(1, 1, 256);
    //std::cout << "local: " << std::min(Item_Size, max_global_size) << std::endl;
    sycl::range<3> global(1, 1, (Item_Size <= 256) ? 256 : work_group_size * 256);
    size_t global1 = (Item_Size <= 256) ? 256 : work_group_size * 256;
    std::cout << "global: " << global1 << std::endl;
	
        std::vector<int> start_offset(info_matCur.dim);
		for(int i = 0;i < info_matCur.dim;i ++)
		{
			start_offset[i] = 1;
		}

    for(int step = 0;step < TIME_STEPS;step ++)
    {
        event e;
        double total_kernel = 0;
        //队列提交命令组
		e = q.submit([&](handler &h) {
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
			.view_shape = matPrev_tool.view_shape_buffer.get_access<sycl::access::mode::read_write>(h),
			.view_stride = matPrev_tool.view_stride_buffer.get_access<sycl::access::mode::read_write>(h),
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
			.view_shape = matNext_tool.view_shape_buffer.get_access<sycl::access::mode::read_write>(h),
			.view_stride = matNext_tool.view_stride_buffer.get_access<sycl::access::mode::read_write>(h),
			.block_shape = matNext_tool.block_shape_buffer.get_access<sycl::access::mode::read_write>(h),
			.block_stride = matNext_tool.block_stride_buffer.get_access<sycl::access::mode::read_write>(h),
			.block_move = matNext_tool.block_move_buffer.get_access<sycl::access::mode::read_write>(h),
			.grid_shape = matNext_tool.grid_shape_buffer.get_access<sycl::access::mode::read_write>(h),
			.grid_stride = matNext_tool.grid_stride_buffer.get_access<sycl::access::mode::read_write>(h),
		};

			h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
				const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
				if(item_id >= Item_Size)
					return;
				// 索引初始化
				
            const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id+(0))%sp2.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
				// 嵌入计算
				
            waveEq(d_matCur,d_matPrev,d_matNext,(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),(sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]),matCur_params,matPrev_params,matNext_params,info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);
			});
		});
        e.wait();
        auto start_kernel = e.get_profiling_info<info::event_profiling::command_start>();
        auto end_kernel = e.get_profiling_info<info::event_profiling::command_end>();
        total_kernel += (end_kernel - start_kernel) / 1e9;
        std::cout << "total_kernel: " << total_kernel << "s" << std::endl;
		if(step < TIME_STEPS - 1)
		{
			
	swap(d_matPrev,d_matCur);
	swap(matPrev_tool.data_stride_buffer, matCur_tool.data_stride_buffer);
	swap(matPrev_tool.data_shape_buffer, matCur_tool.data_shape_buffer);

	swap(d_matCur,d_matNext);
	swap(matCur_tool.data_stride_buffer, matNext_tool.data_stride_buffer);
	swap(matCur_tool.data_shape_buffer, matNext_tool.data_shape_buffer);

	swap(matPrev_tool.start, matCur_tool.start);

	swap(matCur_tool.start, matNext_tool.start);

			
	
	
	matCur_tool.sub_start(start_offset);

	
	matPrev_tool.add_start(start_offset);


			
	matNext_tool.init(info_matNext,matNext_ops);
	sycl::free(d_matNext,q);
	d_matNext = malloc_device<double>(matNext.getSize(),q);
	q.memset(d_matNext,0,matNext.getSize() * sizeof(double)).wait();

		}
    }

	
    // 归约
    // if(Reduction_Split_Size > 1)
    // {
    //     for(int i=0;i<matNextReduction_Size;i++) {
    //         q.submit([&](handler &h) {
    // 	        h.parallel_for(
    //             range<1>(Reduction_Split_Size),
    //             reduction(reduction_matNext+i, 
    //             sycl::plus<>(),
    //             property::reduction::initialize_to_identity()),
    //             [=](id<1> idx,auto &reducer) {
    //                 reducer.combine(d_matNext[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
    //  	        });
    //      }).wait();
    //     }
    //     q.memcpy(d_matNext,reduction_matNext, matNextReduction_Size*sizeof(double)).wait();
    // }


    // 归并结果返回
	q.memcpy(h_matNext,d_matNext, matNext_Size*sizeof(double)).wait();
	matNext.array2Tensor(h_matNext);

	

    // 内存释放
    
    sycl::free(d_matCur, q);
    sycl::free(d_matPrev, q);
    sycl::free(d_matNext, q);
}

int main() {
    double total_program = 0;
    auto program_start = std::chrono::high_resolution_clock::now();
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
    //         std::cout << u_prev[i * NX + j] << " " ;
    //     }
    //     std::cout << std::endl;
    // }

    dacpp::Matrix<double> matCur({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matPrev = u_prev_tensor[{1,NX - 1}][{1,NY - 1}];
    dacpp::Matrix<double> matNext = u_next_tensor[{1,NX - 1}][{1,NY - 1}];
        // 使用 dac_for 包裹计算表达式和指针操作代码块
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
    for (int i = 0; i < NX; ++i) {       
        matCur[i][NY-1]=0;
        matCur[i][0]=0;
    }
    for (int j = 0; j < NY; ++j) {
        matCur[NX - 1][j]=0;
        matCur[0][j]=0;
            // 底部边界
    }    
   matCur.print(); 
   auto program_end = std::chrono::high_resolution_clock::now();
    total_program += std::chrono::duration<double>(program_end - program_start).count();
    std::cout << "total_program: " << total_program << "s" << std::endl;
    return 0;
}

