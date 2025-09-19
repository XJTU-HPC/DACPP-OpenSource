#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const double dt = 0.1;       // 时间步长
const double T = 5.0;       // 总时间
const size_t numIsotopes = 10000; // 设定大量同位素（例如，10000个）




// 计算每种同位素在时间 t 的数量
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void decay(double* N0s,double* lambdas,double* local_A,double* t,sycl::accessor<int, 1, sycl::access::mode::read_write> info_N0s_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_lambdas_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_local_A_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_t_acc) 
{
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);
}




// 生成函数调用
void DECAY_decay(const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    auto start_time = std::chrono::high_resolution_clock::now();
    double total_kernel_time = 0.0;//内核执行时间
    double total_recon_time = 0.0; //数据重组时间
    double total_memcpy = 0.0; //数据移动时间
    double total_parameter_time = 0.0;  //参数生成时间
	//设备初始化计时
	auto device_init_start=std::chrono::high_resolution_clock::now();

    // 设备选择
    auto selector = default_selector_v;
    queue q(selector,sycl::property::queue::enable_profiling{});
	auto device_init_end=std::chrono::high_resolution_clock::now();
	double deviceInit_time = std::chrono::duration_cast<std::chrono::microseconds>(
    device_init_end - device_init_start).count() * 1e-3; //设备初始化时间

    
	//声明参数生成工具
	auto parameter_start = std::chrono::high_resolution_clock::now();
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_N0s;
    info_N0s.dim = N0s.getDim();
    for(int i = 0; i < info_N0s.dim; i++) info_N0s.dimLength.push_back(N0s.getShape(i));
	
    // 数据信息初始化
    DataInfo info_lambdas;
    info_lambdas.dim = lambdas.getDim();
    for(int i = 0; i < info_lambdas.dim; i++) info_lambdas.dimLength.push_back(lambdas.getShape(i));
	
    // 数据信息初始化
    DataInfo info_local_A;
    info_local_A.dim = local_A.getDim();
    for(int i = 0; i < info_local_A.dim; i++) info_local_A.dimLength.push_back(local_A.getShape(i));
	
    // 数据信息初始化
    DataInfo info_t;
    info_t.dim = t.getDim();
    for(int i = 0; i < info_t.dim; i++) info_t.dimLength.push_back(t.getShape(i));
	
    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_N0s));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops N0s_Ops;
    
    i.setDimId(0);
    N0s_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops lambdas_Ops;
    
    i.setDimId(0);
    lambdas_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops local_A_Ops;
    
    i.setDimId(0);
    local_A_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops t_Ops;
    

    // 算子组初始化
    Dac_Ops In_Ops;
    
    i.setDimId(0);
    In_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    i.setDimId(0);
    Out_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    i.setDimId(0);
    Reduction_Ops.push_back(i);


	
    //生成设备内存分配大小
    int N0s_Size = para_gene_tool.init_device_memory_size(info_N0s,N0s_Ops);

    //生成设备内存分配大小
    int lambdas_Size = para_gene_tool.init_device_memory_size(info_lambdas,lambdas_Ops);

    //生成设备内存分配大小
    int local_A_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_local_A);

    //生成设备内存分配大小
    int local_AReduction_Size = para_gene_tool.init_device_memory_size(info_local_A,Reduction_Ops);

    //生成设备内存分配大小
    int t_Size = para_gene_tool.init_device_memory_size(info_t,t_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(N0s_Ops,N0s_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(lambdas_Ops,lambdas_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,local_A_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(t_Ops,t_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(N0s_Ops);

    ops_s.push_back(lambdas_Ops);

    ops_s.push_back(In_Ops);

    ops_s.push_back(t_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[4][1] = {0};
    para_gene_tool.init_split_length_martix(4,1,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
	auto parameter_end = std::chrono::high_resolution_clock::now();
    total_parameter_time = std::chrono::duration_cast<std::chrono::microseconds>(parameter_end - parameter_start).count() * 1e-3;
	auto start_memory = std::chrono::high_resolution_clock::now();
    
    // 归约设备内存分配
    double *reduction_local_A = malloc_device<double>(local_AReduction_Size,q);
    // 数据关联计算
    
	  
	auto memcpy_end = std::chrono::high_resolution_clock::now();
    total_memcpy = std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;
	std::chrono::high_resolution_clock::time_point recon_start = std::chrono::high_resolution_clock::now();
	
    // 设备内存分配
    double *d_N0s=malloc_device<double>(N0s_Size,q);
    // 设备内存分配
    double *d_lambdas=malloc_device<double>(lambdas_Size,q);
    // 设备内存分配
    double *d_local_A=malloc_device<double>(local_A_Size,q);
    // 设备内存分配
    double *d_t=malloc_device<double>(t_Size,q);
    // 数据移动
	double* h_N0s = (double*)malloc(N0s_Size*sizeof(double));
	N0s.tensor2Array(h_N0s);
    q.memcpy(d_N0s,h_N0s,N0s_Size*sizeof(double)).wait();

    // 数据移动
	double* h_lambdas = (double*)malloc(lambdas_Size*sizeof(double));
	lambdas.tensor2Array(h_lambdas);
    q.memcpy(d_lambdas,h_lambdas,lambdas_Size*sizeof(double)).wait();

    // 数据移动
	double* h_local_A = (double*)malloc(local_A_Size*sizeof(double));
	// local_A.tensor2Array(h_local_A);
    q.memset(d_local_A, 0, local_A_Size*sizeof(double)).wait();
    // 数据移动
	double* h_t = (double*)malloc(t_Size*sizeof(double));
	t.tensor2Array(h_t);
    q.memcpy(d_t,h_t,t_Size*sizeof(double)).wait();

    // 数据重组
    DataReconstructor<double> N0s_tool;
    
    // 数据算子组初始化
    Dac_Ops N0s_ops;
    
    i.setDimId(0);
    N0s_ops.push_back(i);

    N0s_tool.init(info_N0s,N0s_ops);
	start_memory = std::chrono::high_resolution_clock::now();
	double *r_N0s=malloc_device<double>(N0s_Size,q);
	memcpy_end = std::chrono::high_resolution_clock::now();
	total_memcpy += std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;
    N0s_tool.Reconstruct(r_N0s,d_N0s,q);
	std::vector<int> info_partition_N0s=para_gene_tool.init_partition_data_shape(info_N0s,N0s_ops);
    sycl::buffer<int> info_partition_N0s_buffer(info_partition_N0s.data(), sycl::range<1>(info_partition_N0s.size()));

    // 数据重组
    DataReconstructor<double> lambdas_tool;
    
    // 数据算子组初始化
    Dac_Ops lambdas_ops;
    
    i.setDimId(0);
    lambdas_ops.push_back(i);

    lambdas_tool.init(info_lambdas,lambdas_ops);
	start_memory = std::chrono::high_resolution_clock::now();
	double *r_lambdas=malloc_device<double>(lambdas_Size,q);
	memcpy_end = std::chrono::high_resolution_clock::now();
	total_memcpy += std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;
    lambdas_tool.Reconstruct(r_lambdas,d_lambdas,q);
	std::vector<int> info_partition_lambdas=para_gene_tool.init_partition_data_shape(info_lambdas,lambdas_ops);
    sycl::buffer<int> info_partition_lambdas_buffer(info_partition_lambdas.data(), sycl::range<1>(info_partition_lambdas.size()));

    // 数据重组
    DataReconstructor<double> local_A_tool;
    
    // 数据算子组初始化
    Dac_Ops local_A_ops;
    
    i.setDimId(0);
    local_A_ops.push_back(i);

    local_A_tool.init(info_local_A,local_A_ops);
	start_memory = std::chrono::high_resolution_clock::now();
	double *r_local_A=malloc_device<double>(local_A_Size,q);
	memcpy_end = std::chrono::high_resolution_clock::now();
	total_memcpy += std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;
    local_A_tool.Reconstruct(r_local_A,d_local_A,q);
	std::vector<int> info_partition_local_A=para_gene_tool.init_partition_data_shape(info_local_A,local_A_ops);
    sycl::buffer<int> info_partition_local_A_buffer(info_partition_local_A.data(), sycl::range<1>(info_partition_local_A.size()));

    // 数据重组
    DataReconstructor<double> t_tool;
    
    // 数据算子组初始化
    Dac_Ops t_ops;
    

    t_tool.init(info_t,t_ops);
	start_memory = std::chrono::high_resolution_clock::now();
	double *r_t=malloc_device<double>(t_Size,q);
	memcpy_end = std::chrono::high_resolution_clock::now();
	total_memcpy += std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;
    t_tool.Reconstruct(r_t,d_t,q);
	std::vector<int> info_partition_t=para_gene_tool.init_partition_data_shape(info_t,t_ops);
    sycl::buffer<int> info_partition_t_buffer(info_partition_t.data(), sycl::range<1>(info_partition_t.size()));

	std::chrono::high_resolution_clock::time_point recon_end = std::chrono::high_resolution_clock::now();
    total_recon_time = std::chrono::duration_cast<std::chrono::microseconds>(recon_end - recon_start).count() * 1e-3;
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
    //队列提交命令组
    sycl::event e = q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_N0s_accessor = info_partition_N0s_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_lambdas_accessor = info_partition_lambdas_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_local_A_accessor = info_partition_local_A_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_t_accessor = info_partition_t_buffer.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 嵌入计算
			
            decay(r_N0s+(i_*SplitLength[0][0]),r_lambdas+(i_*SplitLength[1][0]),r_local_A+(i_*SplitLength[2][0]),r_t,info_partition_N0s_accessor,info_partition_lambdas_accessor,info_partition_local_A_accessor,info_partition_t_accessor);
        });
    });
	e.wait();
	auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    total_kernel_time = (end - start) * 1e-6; // 纳秒转毫秒
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<local_AReduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_local_A+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(r_local_A[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(r_local_A,reduction_local_A, local_AReduction_Size*sizeof(double)).wait();
    }


    // 归并结果返回
    local_A_tool.UpdateData(r_local_A,d_local_A,q);
	q.memcpy(h_local_A,d_local_A, local_A_Size*sizeof(double)).wait();
	local_A.array2Tensor(h_local_A);

	start_memory = std::chrono::high_resolution_clock::now();
	
	memcpy_end = std::chrono::high_resolution_clock::now();
    total_memcpy += std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;

    // 内存释放
    
    sycl::free(d_N0s, q);
    sycl::free(d_lambdas, q);
    sycl::free(d_local_A, q);
    sycl::free(d_t, q);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = end_time - start_time;
    std::cout << "设备初始化时间: " << deviceInit_time<< " ms" << std::endl;
    std::cout << "参数生成时间: " << total_parameter_time << " ms" << std::endl;
    std::cout << "数据重组时间: " << total_recon_time << " ms" << std::endl;
    std::cout << "数据移动时间: " << total_memcpy << " ms" << std::endl;
    std::cout << "内核执行时间: " << total_kernel_time << " ms" << std::endl;
    std::cout << "总执行时间: " << total_duration.count() << " ms" << std::endl;
} 

void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size(); // 同位素的数量
    std::vector<double> A(T/dt*numIsotopes, 0.0);  // 存储每个同位素在不同时间点的数量
    std::vector<double> time;  // 时间序列
    std::vector<double> t;
    t.push_back(static_cast<double>(0));

    // 串行计算每个同位素的衰变过程
    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({static_cast<int>(T/dt), static_cast<int>(numIsotopes)}, A);
    

    while(t_tensor[0] <= T){  
        DECAY_decay(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    A_tensor[1].print();
}

int main() {
    

    // 随机生成衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01*i;  // lambda 范围 [0.01, 0.2]
    }


    //size_t numOutputSteps = 10; // 输出的时间步数量

    calculateDecay(lambdas, N0s, dt, T);

    return 0;
}
