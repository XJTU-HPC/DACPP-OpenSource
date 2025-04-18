#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}
void DECAY(const dacpp::Tensor<double, 1> & N0s, const dacpp::Tensor<double, 1> & lambdas, dacpp::Tensor<double, 1> & local_A, const dacpp::Tensor<double, 1> & t);




// 计算每种同位素在时间 t 的数量
void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size(); // 同位素的数量
    std::vector<double> A(50*10, 0.0);  // 存储每个同位素在不同时间点的数量
    std::vector<double> time;  // 时间序列
    std::vector<double> t;
    t.push_back(static_cast<double>(0));

    // 串行计算每个同位素的衰变过程
    std::vector<double> local_A(10, 0.0);
    dacpp::Tensor<double, 1> local_A_tensor(local_A);
    dacpp::Tensor<double, 1> N0s_tensor(N0s);
    dacpp::Tensor<double, 1> lambdas_tensor(lambdas);
    dacpp::Tensor<double, 1> t_tensor(t);
    dacpp::Tensor<double, 2> A_tensor({50, 10}, A);
    

    while(t_tensor[0] < T){        
        DECAY(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    A_tensor.print();
}

#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void decay(double* N0s, double* lambdas, double* local_A, double* t) 
{
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);
}


// 生成函数调用
void DECAY(const dacpp::Tensor<double, 1> & N0s, const dacpp::Tensor<double, 1> & lambdas, dacpp::Tensor<double, 1> & local_A, const dacpp::Tensor<double, 1> & t) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    //ParameterGeneration<int,2> para_gene_tool;
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
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_local_A,Reduction_Ops);

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
    
    // 设备内存分配
    double *d_N0s=malloc_device<double>(N0s_Size,q);
    // 设备内存分配
    double *d_lambdas=malloc_device<double>(lambdas_Size,q);
    // 设备内存分配
    double *d_local_A=malloc_device<double>(local_A_Size,q);
    // 归约设备内存分配
    double *reduction_local_A = malloc_device<double>(Reduction_Size,q);
    // 设备内存分配
    double *d_t=malloc_device<double>(t_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<double> N0s_tool;
    double* r_N0s=(double*)malloc(sizeof(double)*N0s_Size);
    
    // 数据算子组初始化
    Dac_Ops N0s_ops;
    
    i.setDimId(0);
    i.setSplitLength(8);
    N0s_ops.push_back(i);
    N0s_tool.init(info_N0s,N0s_ops);
    N0s_tool.Reconstruct(r_N0s,N0s);
    // 数据重组
    DataReconstructor<double> lambdas_tool;
    double* r_lambdas=(double*)malloc(sizeof(double)*lambdas_Size);
    
    // 数据算子组初始化
    Dac_Ops lambdas_ops;
    
    i.setDimId(0);
    i.setSplitLength(8);
    lambdas_ops.push_back(i);
    lambdas_tool.init(info_lambdas,lambdas_ops);
    lambdas_tool.Reconstruct(r_lambdas,lambdas);
    // 数据重组
    DataReconstructor<double> local_A_tool;
    double* r_local_A=(double*)malloc(sizeof(double)*local_A_Size);
    
    // 数据算子组初始化
    Dac_Ops local_A_ops;
    
    i.setDimId(0);
    i.setSplitLength(8);
    local_A_ops.push_back(i);
    local_A_tool.init(info_local_A,local_A_ops);
    local_A_tool.Reconstruct(r_local_A,local_A);
    // 数据重组
    DataReconstructor<double> t_tool;
    double* r_t=(double*)malloc(sizeof(double)*t_Size);
    
    // 数据算子组初始化
    Dac_Ops t_ops;
    
    t_tool.init(info_t,t_ops);
    t_tool.Reconstruct(r_t,t);
    
    // 数据移动
    q.memcpy(d_N0s,r_N0s,N0s_Size*sizeof(double)).wait();
    // 数据移动
    q.memcpy(d_lambdas,r_lambdas,lambdas_Size*sizeof(double)).wait();
    // 数据移动
    q.memcpy(d_t,r_t,t_Size*sizeof(double)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 嵌入计算
			
            decay(d_N0s+(i_*SplitLength[0][0]),d_lambdas+(i_*SplitLength[1][0]),d_local_A+(i_*SplitLength[2][0]),d_t);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_local_A+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_local_A[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_local_A,reduction_local_A, Reduction_Size*sizeof(double)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_local_A, d_local_A, local_A_Size*sizeof(double)).wait();
    local_A_tool.UpdateData(r_local_A,local_A);

    // 内存释放
    
    sycl::free(d_N0s, q);
    sycl::free(d_lambdas, q);
    sycl::free(d_local_A, q);
    sycl::free(d_t, q);
}

int main() {
    size_t numIsotopes = 10; // 设定大量同位素（例如，10000个）

    // 随机生成衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01*i;  // lambda 范围 [0.01, 0.2]
    }

    double dt = 0.1;       // 时间步长
    double T = 5.0;       // 总时间
    //size_t numOutputSteps = 10; // 输出的时间步数量

    calculateDecay(lambdas, N0s, dt, T);

    return 0;
}
