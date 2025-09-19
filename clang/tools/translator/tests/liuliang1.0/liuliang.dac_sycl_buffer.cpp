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
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void lwr(double* rho,double* new_rho,sycl::accessor<int, 1, sycl::access::mode::read_write> info_rho_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_new_rho_acc) 
{
    new_rho[0] = rho[1] - (DELTA_T / DELTA_X) * (q(rho[1]) - q(rho[0]));
    new_rho[0] = std::max(0., new_rho[0]);
}


// 生成函数调用
void LWR_shell_lwr(const dacpp::Vector<double> & rho, dacpp::Vector<double> & new_rho) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
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
    int new_rhoReduction_Size = para_gene_tool.init_device_memory_size(info_new_rho,Reduction_Ops);

	
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
    
    // Buffer设备内存分配
    buffer <double> b_rho{rho_Size};

    // Buffer设备内存分配
    buffer <double> b_new_rho{new_rho_Size};

    // 规约Buffer设备内存分配
    std::vector<sycl::buffer<double, 1>> b_reduction_new_rho(new_rhoReduction_Size, buffer<double, 1>{1});
    for(int i = 0; i < new_rhoReduction_Size; i++){
        host_accessor temp_accessor{b_reduction_new_rho[i]};
        temp_accessor[0] = 0;
    }
    // 数据关联计算
    
	
    // 数据移动
    double* h_rho = (double*)malloc(rho.getSize()*sizeof(double));
    rho.tensor2Array(h_rho);
    {
        host_accessor temp_accessor{b_rho};
        for(int i = 0; i < rho_Size; i++){
            temp_accessor[i] = h_rho[i];
        }
    }

    // 数据移动
    double* h_new_rho = (double*)malloc(new_rho.getSize()*sizeof(double));

    // 数据重组
    DataReconstructor<double> rho_tool;
    
    // 数据算子组初始化
    Dac_Ops rho_ops;
    
    S1.setDimId(0);
    rho_ops.push_back(S1);

    rho_tool.init(info_rho,rho_ops);
    buffer<double> r_rho{rho_Size};
    rho_tool.Reconstruct(r_rho,b_rho,q);
	std::vector<int> info_partition_rho=para_gene_tool.init_partition_data_shape(info_rho,rho_ops);
    sycl::buffer<int> info_partition_rho_buffer(info_partition_rho.data(), sycl::range<1>(info_partition_rho.size()));

    // 数据重组
    DataReconstructor<double> new_rho_tool;
    
    // 数据算子组初始化
    Dac_Ops new_rho_ops;
    
    idx1.setDimId(0);
    new_rho_ops.push_back(idx1);

    new_rho_tool.init(info_new_rho,new_rho_ops);
    buffer<double> r_new_rho{new_rho_Size};
    new_rho_tool.Reconstruct(r_new_rho,b_new_rho,q);
	std::vector<int> info_partition_new_rho=para_gene_tool.init_partition_data_shape(info_new_rho,new_rho_ops);
    sycl::buffer<int> info_partition_new_rho_buffer(info_partition_new_rho.data(), sycl::range<1>(info_partition_new_rho.size()));
    
	
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

    //队列提交命令组
    q.submit([&](handler &h) {
    
        accessor acc_rho{r_rho, h};
        accessor acc_new_rho{r_new_rho, h};
    
        auto info_partition_rho_accessor = info_partition_rho_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_new_rho_accessor = info_partition_new_rho_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto idx1_=(item_id+(0))%idx1.split_size;
            const auto S1_=(item_id+(0))%S1.split_size;
            // 获得accessor指针
            
            auto* d_rho = acc_rho.get_multi_ptr<access::decorated::no>().get();
            auto* d_new_rho = acc_new_rho.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            lwr(d_rho+(S1_*SplitLength[0][0]),d_new_rho+(S1_*SplitLength[1][0]),info_partition_rho_accessor,info_partition_new_rho_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<new_rhoReduction_Size;i++) {
            q.submit([&](handler &h) {
                accessor d_new_rho{r_new_rho, h};
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(b_reduction_new_rho[i], h, 
                sycl::plus<>(),property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_new_rho[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_new_rho};
            for(int i = 0; i < new_rhoReduction_Size; i++){
                host_accessor temp_accessor{b_reduction_new_rho[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }


    //结果返回
    new_rho_tool.UpdateData(r_new_rho,b_new_rho,q);
    {
        host_accessor temp_accessor{b_new_rho};
        for(int i = 0; i < new_rho_Size; i++){
            h_new_rho[i] = temp_accessor[i];
        }
    }
    new_rho.array2Tensor(h_new_rho);

	

    // 内存释放
    
}

int main() {
    // 创建 Tensor 类型对象
    std::vector<double> rho(WIDTH, 0.0);
    std::vector<double> new_rho(WIDTH, 0.0);
    initializeDensity(rho);
    dacpp::Vector<double> rho_tensor(rho);
    dacpp::Vector<double> new_rho_tensor(new_rho);
    dacpp::Vector<double> middle_out_tensor = new_rho_tensor[{1,WIDTH-1}];
    dacpp::Vector<double> middle_in_tensor = rho_tensor[{0,WIDTH-1}];
    for (int t = 0; t < TIME_STEPS; ++t) {
        LWR_shell_lwr(middle_in_tensor, middle_out_tensor);
        for (int i = 1; i <= WIDTH-2; i++) {
            middle_in_tensor[i] = middle_out_tensor[i-1];
        }
        
        middle_in_tensor[0] = middle_out_tensor[0]; // 左边界无车流
        //middle_in_tensor[99] = middle_out_tensor[97];

    }
    std::cout << middle_in_tensor[15] << std::endl;
    

    

    // 释放动态分配的内存

    return 0;
}
