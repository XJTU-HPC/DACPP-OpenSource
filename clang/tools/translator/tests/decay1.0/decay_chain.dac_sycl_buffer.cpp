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
const size_t numIsotopes = 10; // 设定大量同位素（例如，10000个）




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
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
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
    
    // Buffer设备内存分配
    buffer <double> b_N0s{N0s_Size};

    // Buffer设备内存分配
    buffer <double> b_lambdas{lambdas_Size};

    // Buffer设备内存分配
    buffer <double> b_local_A{local_A_Size};

    // 规约Buffer设备内存分配
    std::vector<sycl::buffer<double, 1>> b_reduction_local_A(local_AReduction_Size, buffer<double, 1>{1});
    for(int i = 0; i < local_AReduction_Size; i++){
        host_accessor temp_accessor{b_reduction_local_A[i]};
        temp_accessor[0] = 0;
    }
    // Buffer设备内存分配
    buffer <double> b_t{t_Size};

    // 数据关联计算
    
	
    // 数据移动
    double* h_N0s = (double*)malloc(N0s.getSize()*sizeof(double));
    N0s.tensor2Array(h_N0s);
    {
        host_accessor temp_accessor{b_N0s};
        for(int i = 0; i < N0s_Size; i++){
            temp_accessor[i] = h_N0s[i];
        }
    }

    // 数据移动
    double* h_lambdas = (double*)malloc(lambdas.getSize()*sizeof(double));
    lambdas.tensor2Array(h_lambdas);
    {
        host_accessor temp_accessor{b_lambdas};
        for(int i = 0; i < lambdas_Size; i++){
            temp_accessor[i] = h_lambdas[i];
        }
    }

    // 数据移动
    double* h_local_A = (double*)malloc(local_A.getSize()*sizeof(double));

    // 数据移动
    double* h_t = (double*)malloc(t.getSize()*sizeof(double));
    t.tensor2Array(h_t);
    {
        host_accessor temp_accessor{b_t};
        for(int i = 0; i < t_Size; i++){
            temp_accessor[i] = h_t[i];
        }
    }

    // 数据重组
    DataReconstructor<double> N0s_tool;
    
    // 数据算子组初始化
    Dac_Ops N0s_ops;
    
    i.setDimId(0);
    N0s_ops.push_back(i);

    N0s_tool.init(info_N0s,N0s_ops);
    buffer<double> r_N0s{N0s_Size};
    N0s_tool.Reconstruct(r_N0s,b_N0s,q);
	std::vector<int> info_partition_N0s=para_gene_tool.init_partition_data_shape(info_N0s,N0s_ops);
    sycl::buffer<int> info_partition_N0s_buffer(info_partition_N0s.data(), sycl::range<1>(info_partition_N0s.size()));

    // 数据重组
    DataReconstructor<double> lambdas_tool;
    
    // 数据算子组初始化
    Dac_Ops lambdas_ops;
    
    i.setDimId(0);
    lambdas_ops.push_back(i);

    lambdas_tool.init(info_lambdas,lambdas_ops);
    buffer<double> r_lambdas{lambdas_Size};
    lambdas_tool.Reconstruct(r_lambdas,b_lambdas,q);
	std::vector<int> info_partition_lambdas=para_gene_tool.init_partition_data_shape(info_lambdas,lambdas_ops);
    sycl::buffer<int> info_partition_lambdas_buffer(info_partition_lambdas.data(), sycl::range<1>(info_partition_lambdas.size()));

    // 数据重组
    DataReconstructor<double> local_A_tool;
    
    // 数据算子组初始化
    Dac_Ops local_A_ops;
    
    i.setDimId(0);
    local_A_ops.push_back(i);

    local_A_tool.init(info_local_A,local_A_ops);
    buffer<double> r_local_A{local_A_Size};
    local_A_tool.Reconstruct(r_local_A,b_local_A,q);
	std::vector<int> info_partition_local_A=para_gene_tool.init_partition_data_shape(info_local_A,local_A_ops);
    sycl::buffer<int> info_partition_local_A_buffer(info_partition_local_A.data(), sycl::range<1>(info_partition_local_A.size()));

    // 数据重组
    DataReconstructor<double> t_tool;
    
    // 数据算子组初始化
    Dac_Ops t_ops;
    

    t_tool.init(info_t,t_ops);
    buffer<double> r_t{t_Size};
    t_tool.Reconstruct(r_t,b_t,q);
	std::vector<int> info_partition_t=para_gene_tool.init_partition_data_shape(info_t,t_ops);
    sycl::buffer<int> info_partition_t_buffer(info_partition_t.data(), sycl::range<1>(info_partition_t.size()));
    
	
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

    //队列提交命令组
    q.submit([&](handler &h) {
    
        accessor acc_N0s{r_N0s, h};
        accessor acc_lambdas{r_lambdas, h};
        accessor acc_local_A{r_local_A, h};
        accessor acc_t{r_t, h};
    
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
            // 获得accessor指针
            
            auto* d_N0s = acc_N0s.get_multi_ptr<access::decorated::no>().get();
            auto* d_lambdas = acc_lambdas.get_multi_ptr<access::decorated::no>().get();
            auto* d_local_A = acc_local_A.get_multi_ptr<access::decorated::no>().get();
            auto* d_t = acc_t.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            decay(d_N0s+(i_*SplitLength[0][0]),d_lambdas+(i_*SplitLength[1][0]),d_local_A+(i_*SplitLength[2][0]),d_t,info_partition_N0s_accessor,info_partition_lambdas_accessor,info_partition_local_A_accessor,info_partition_t_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<local_AReduction_Size;i++) {
            q.submit([&](handler &h) {
                accessor d_local_A{r_local_A, h};
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(b_reduction_local_A[i], h, 
                sycl::plus<>(),property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_local_A[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_local_A};
            for(int i = 0; i < local_AReduction_Size; i++){
                host_accessor temp_accessor{b_reduction_local_A[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }


    //结果返回
    local_A_tool.UpdateData(r_local_A,b_local_A,q);
    {
        host_accessor temp_accessor{b_local_A};
        for(int i = 0; i < local_A_Size; i++){
            h_local_A[i] = temp_accessor[i];
        }
    }
    local_A.array2Tensor(h_local_A);

	

    // 内存释放
    
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
