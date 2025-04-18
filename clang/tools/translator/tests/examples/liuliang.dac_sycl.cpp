#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 100;       // 路段长度
const double TIME_STEPS = 200;  // 时间步数
const double DELTA_T = 0.01; // 时间步长
const double DELTA_X = 1.0;  // 空间步长

// 流量函数，考虑密度对流量的影响
double q1(double rho) {
    const double V_max = 30; // 最大速度
    const double rho_max = 50; // 最大密度
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
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void lwr(double* rho, double* new_rho) 
{
    new_rho[0] = rho[1] - (DELTA_T / DELTA_X) * (q1(rho[1]) - q1(rho[0]));
    new_rho[0] = std::max(0.0, new_rho[0]);
}


// 生成函数调用
void LWR_shell(const dacpp::Tensor<double, 1> & rho, dacpp::Tensor<double, 1> & new_rho) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_rho;
    info_rho.dim = rho.getDim();
    for(int i = 0; i < info_rho.dim; i++) info_rho.dimLength.push_back(rho.getShape(i));
    // 数据信息初始化
    DataInfo info_new_rho;
    info_new_rho.dim = new_rho.getDim();
    for(int i = 0; i < info_new_rho.dim; i++) info_new_rho.dimLength.push_back(new_rho.getShape(i));
    // 规则分区算子初始化
    RegularSlice S1 = RegularSlice("S1", 2, 1);
    S1.setDimId(0);
    S1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(S1,info_rho));

    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_new_rho));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops rho_Ops;
    
    S1.setDimId(0);
    rho_Ops.push_back(S1);


    // 算子组初始化
    Dac_Ops new_rho_Ops;
    
    idx1.setDimId(0);
    new_rho_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    S1.setDimId(0);
    In_Ops.push_back(S1);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);


	
    //生成设备内存分配大小
    int rho_Size = para_gene_tool.init_device_memory_size(info_rho,rho_Ops);

    //生成设备内存分配大小
    int new_rho_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_new_rho);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_new_rho,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(rho_Ops,rho_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,new_rho_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(rho_Ops);

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
    double *d_rho=malloc_device<double>(rho_Size,q);
    // 设备内存分配
    double *d_new_rho=malloc_device<double>(new_rho_Size,q);
    // 归约设备内存分配
    double *reduction_new_rho = malloc_device<double>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<double> rho_tool;
    double* r_rho=(double*)malloc(sizeof(double)*rho_Size);
    
    // 数据算子组初始化
    Dac_Ops rho_ops;
    
    S1.setDimId(0);
    S1.setSplitLength(8);
    rho_ops.push_back(S1);
    rho_tool.init(info_rho,rho_ops);
    rho_tool.Reconstruct(r_rho,rho);
    // 数据重组
    DataReconstructor<double> new_rho_tool;
    double* r_new_rho=(double*)malloc(sizeof(double)*new_rho_Size);
    
    // 数据算子组初始化
    Dac_Ops new_rho_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    new_rho_ops.push_back(idx1);
    new_rho_tool.init(info_new_rho,new_rho_ops);
    new_rho_tool.Reconstruct(r_new_rho,new_rho);
    
    // 数据移动
    q.memcpy(d_rho,r_rho,rho_Size*sizeof(double)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx1_=(item_id+(0))%idx1.split_size;
            const auto S1_=(item_id+(0))%S1.split_size;
            // 嵌入计算
			
            lwr(d_rho+(S1_*SplitLength[0][0]),d_new_rho+(S1_*SplitLength[1][0]));
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_new_rho+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_new_rho[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_new_rho,reduction_new_rho, Reduction_Size*sizeof(double)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_new_rho, d_new_rho, new_rho_Size*sizeof(double)).wait();
    new_rho_tool.UpdateData(r_new_rho,new_rho);

    // 内存释放
    
    sycl::free(d_rho, q);
    sycl::free(d_new_rho, q);
}

int main() {
    // 创建 Tensor 类型对象
    std::vector<double> rho(WIDTH, 0.0);
    std::vector<double> new_rho(WIDTH, 0.0);
    initializeDensity(rho);

    // 使用 LWR 算法
    std::vector<double> middle_points_out;
    for (int i = 1; i <= 98; i++) {
        middle_points_out.push_back(static_cast<double>(new_rho[i]));
    }
    //std::vector<int> shape = {98,1};
    dacpp::Tensor<double, 1> middle_out_tensor(middle_points_out);

    std::vector<double> middle_points_in;
    for (int i = 0; i <= 98; i++) {
        middle_points_in.push_back(static_cast<double>(rho[i]));
    }
    //std::vector<int> shape2 = {99,1};
    dacpp::Tensor<double, 1> middle_in_tensor(middle_points_in);

    for (int t = 0; t < TIME_STEPS; ++t) {



        LWR_shell(middle_in_tensor, middle_out_tensor);


        for (int i = 1; i <= 98; i++) {
            middle_in_tensor[i] = middle_out_tensor[i-1];
        }
        
        // 处理边界条件
        middle_in_tensor[0] = middle_out_tensor[0]; // 左边界无车流

    }

    middle_in_tensor.print();

    // 释放动态分配的内存

    return 0;
}
