#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

// 参数设置
const double A = 1.0;  // 吸引力系数
const double D = 0.1;  // 扩散系数
const double dx = 0.1; // 空间步长
const double dt = 0.01; // 时间步长
const int N = 100;     // 空间网格点数
const int T = 1000;    // 时间步数


// 初始化用户偏好分布
void initialize(std::vector<double>& p) {
    for (int i = 0; i < N; ++i) {
        // 假设初始偏好为高斯分布
        double x = i * dx;
        p[i] = std::exp(-std::pow(x - 5.0, 2) / 2.0); // 初始偏好分布中心在x=5
    }
}

// 归一化函数
void normalize(dacpp::Tensor<double, 1>& p) {
    double sum = 0.0;
    for (int i = 0;i < N; i++) {
        sum += p[i];
    }
    for (int i = 0;i < N; i++) {
        p[i] /= sum; // 归一化    
    }
}







// 数值求解Fokker-Planck方程
#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void mdp(double* p,double* new_p,sycl::accessor<int, 1, sycl::access::mode::read_write> info_p_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_new_p_acc) 
{
    new_p[0] = p[1] + dt * (D * (p[2] - 2 * p[1] + p[0]) / (dx * dx) + (-A) * (p[2] - p[0]) / (2 * dx));
}


// 生成函数调用
void mdp_shell_mdp(const dacpp::Tensor<double, 1> & p, dacpp::Tensor<double, 1> & new_p) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_p;
    info_p.dim = p.getDim();
    for(int i = 0; i < info_p.dim; i++) info_p.dimLength.push_back(p.getShape(i));
    // 数据信息初始化
    DataInfo info_new_p;
    info_new_p.dim = new_p.getDim();
    for(int i = 0; i < info_new_p.dim; i++) info_new_p.dimLength.push_back(new_p.getShape(i));
    // 规则分区算子初始化
    RegularSlice sp = RegularSlice("sp", 3, 1);
    sp.setDimId(0);
    sp.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp,info_p));

    // 降维算子初始化
    Index idx = Index("idx");
    idx.setDimId(0);
    idx.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx,info_new_p));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops p_Ops;
    
    sp.setDimId(0);
    p_Ops.push_back(sp);


    // 算子组初始化
    Dac_Ops new_p_Ops;
    
    idx.setDimId(0);
    new_p_Ops.push_back(idx);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    sp.setDimId(0);
    In_Ops.push_back(sp);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx.setDimId(0);
    Out_Ops.push_back(idx);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx.setDimId(0);
    Reduction_Ops.push_back(idx);


	
    //生成设备内存分配大小
    int p_Size = para_gene_tool.init_device_memory_size(info_p,p_Ops);

    //生成设备内存分配大小
    int new_p_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_new_p);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_new_p,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(p_Ops,p_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,new_p_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(p_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[2][1] = {0};
    para_gene_tool.init_split_length_martix(2,1,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // 设备内存分配
    double *d_p=malloc_device<double>(p_Size,q);
    // 设备内存分配
    double *d_new_p=malloc_device<double>(new_p_Size,q);
    // 归约设备内存分配
    double *reduction_new_p = malloc_device<double>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<double> p_tool;
    double* r_p=(double*)malloc(sizeof(double)*p_Size);
    
    // 数据算子组初始化
    Dac_Ops p_ops;
    
    sp.setDimId(0);
    sp.setSplitLength(8);
    p_ops.push_back(sp);
    p_tool.init(info_p,p_ops);
    p_tool.Reconstruct(r_p,p);
	std::vector<int> info_partition_p=para_gene_tool.init_partition_data_shape(info_p,p_ops);
    sycl::buffer<int> info_partition_p_buffer(info_partition_p.data(), sycl::range<1>(info_partition_p.size()));
    // 数据重组
    DataReconstructor<double> new_p_tool;
    double* r_new_p=(double*)malloc(sizeof(double)*new_p_Size);
    
    // 数据算子组初始化
    Dac_Ops new_p_ops;
    
    idx.setDimId(0);
    idx.setSplitLength(8);
    new_p_ops.push_back(idx);
    new_p_tool.init(info_new_p,new_p_ops);
    new_p_tool.Reconstruct(r_new_p,new_p);
	std::vector<int> info_partition_new_p=para_gene_tool.init_partition_data_shape(info_new_p,new_p_ops);
    sycl::buffer<int> info_partition_new_p_buffer(info_partition_new_p.data(), sycl::range<1>(info_partition_new_p.size()));
    
    // 数据移动
    q.memcpy(d_p,r_p,p_Size*sizeof(double)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_p_accessor = info_partition_p_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_new_p_accessor = info_partition_new_p_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx_=(item_id/sp.split_size+(0))%idx.split_size;
            const auto sp_=(item_id+(0))%sp.split_size;
            // 嵌入计算
			
            mdp(d_p+(sp_*SplitLength[0][0]),d_new_p+(sp_*SplitLength[1][0]),info_partition_p_accessor,info_partition_new_p_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_new_p+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_new_p[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_new_p,reduction_new_p, Reduction_Size*sizeof(double)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_new_p, d_new_p, new_p_Size*sizeof(double)).wait();
    new_p_tool.UpdateData(r_new_p,new_p);

    // 内存释放
    
    sycl::free(d_p, q);
    sycl::free(d_new_p, q);
}

void solveFokkerPlanck(std::vector<double>& p) {
    std::vector<double> new_p(N-2, 0.0); // 存储下一时间步的分布
    dacpp::Tensor<double, 1> p_tensor(p);
    dacpp::Tensor<double, 1> new_p_tensor(new_p);
        

    for (int t = 0; t < T; ++t) {
        // 对内部点进行更新，边界条件暂不考虑


        mdp_shell_mdp(p_tensor, new_p_tensor); 
        
        // 更新分布
        for(int i = 0; i < N-2; i++){
            p_tensor[i+1] = new_p_tensor[i];
        }
        normalize(p_tensor); // 归一化分布
    }
    p_tensor.print();

}

int main() {

    std::vector<double> p(N, 0.0); // 存储用户偏好分布

    // 初始化偏好分布
    initialize(p);

    // 数值求解Fokker-Planck方程
    solveFokkerPlanck(p);

    return 0;
}
