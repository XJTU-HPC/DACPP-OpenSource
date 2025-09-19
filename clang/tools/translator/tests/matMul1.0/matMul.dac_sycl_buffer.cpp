#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void matrixMultiply_calc(int* vecA,int* vecB,int* dotProduct,sycl::accessor<int, 1, sycl::access::mode::read_write> info_vecA_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_vecB_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_dotProduct_acc) 
{
    for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}


// 生成函数调用
void matrixMultiply_shell_matrixMultiply_calc(const dacpp::Matrix<int> & matA, const dacpp::Matrix<int> & matB, dacpp::Matrix<int> & matC) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_matA;
    info_matA.dim = matA.getDim();
    for(int i = 0; i < info_matA.dim; i++) info_matA.dimLength.push_back(matA.getShape(i));
	
    // 数据信息初始化
    DataInfo info_matB;
    info_matB.dim = matB.getDim();
    for(int i = 0; i < info_matB.dim; i++) info_matB.dimLength.push_back(matB.getShape(i));
	
    // 数据信息初始化
    DataInfo info_matC;
    info_matC.dim = matC.getDim();
    for(int i = 0; i < info_matC.dim; i++) info_matC.dimLength.push_back(matC.getShape(i));
	
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matA));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matB));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops matA_Ops;
    
    idx1.setDimId(0);
    matA_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops matB_Ops;
    
    idx2.setDimId(1);
    matB_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops matC_Ops;
    
    idx1.setDimId(0);
    matC_Ops.push_back(idx1);

    idx2.setDimId(1);
    matC_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    idx1.setDimId(0);
    In_Ops.push_back(idx1);

    idx2.setDimId(1);
    In_Ops.push_back(idx2);


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
    int matA_Size = para_gene_tool.init_device_memory_size(info_matA,matA_Ops);

    //生成设备内存分配大小
    int matB_Size = para_gene_tool.init_device_memory_size(info_matB,matB_Ops);

    //生成设备内存分配大小
    int matC_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_matC);

    //生成设备内存分配大小
    int matCReduction_Size = para_gene_tool.init_device_memory_size(info_matC,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matA_Ops,matA_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matB_Ops,matB_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,matC_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(matA_Ops);

    ops_s.push_back(matB_Ops);

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
    buffer <int> b_matA{matA_Size};

    // Buffer设备内存分配
    buffer <int> b_matB{matB_Size};

    // Buffer设备内存分配
    buffer <int> b_matC{matC_Size};

    // 规约Buffer设备内存分配
    std::vector<sycl::buffer<int, 1>> b_reduction_matC(matCReduction_Size, buffer<int, 1>{1});
    for(int i = 0; i < matCReduction_Size; i++){
        host_accessor temp_accessor{b_reduction_matC[i]};
        temp_accessor[0] = 0;
    }
    // 数据关联计算
    
	
    // 数据移动
    int* h_matA = (int*)malloc(matA.getSize()*sizeof(int));
    matA.tensor2Array(h_matA);
    {
        host_accessor temp_accessor{b_matA};
        for(int i = 0; i < matA_Size; i++){
            temp_accessor[i] = h_matA[i];
        }
    }

    // 数据移动
    int* h_matB = (int*)malloc(matB.getSize()*sizeof(int));
    matB.tensor2Array(h_matB);
    {
        host_accessor temp_accessor{b_matB};
        for(int i = 0; i < matB_Size; i++){
            temp_accessor[i] = h_matB[i];
        }
    }

    // 数据移动
    int* h_matC = (int*)malloc(matC.getSize()*sizeof(int));

    // 数据重组
    DataReconstructor<int> matA_tool;
    
    // 数据算子组初始化
    Dac_Ops matA_ops;
    
    idx1.setDimId(0);
    matA_ops.push_back(idx1);

    matA_tool.init(info_matA,matA_ops);
    buffer<int> r_matA{matA_Size};
    matA_tool.Reconstruct(r_matA,b_matA,q);
	std::vector<int> info_partition_matA=para_gene_tool.init_partition_data_shape(info_matA,matA_ops);
    sycl::buffer<int> info_partition_matA_buffer(info_partition_matA.data(), sycl::range<1>(info_partition_matA.size()));

    // 数据重组
    DataReconstructor<int> matB_tool;
    
    // 数据算子组初始化
    Dac_Ops matB_ops;
    
    idx2.setDimId(1);
    matB_ops.push_back(idx2);

    matB_tool.init(info_matB,matB_ops);
    buffer<int> r_matB{matB_Size};
    matB_tool.Reconstruct(r_matB,b_matB,q);
	std::vector<int> info_partition_matB=para_gene_tool.init_partition_data_shape(info_matB,matB_ops);
    sycl::buffer<int> info_partition_matB_buffer(info_partition_matB.data(), sycl::range<1>(info_partition_matB.size()));

    // 数据重组
    DataReconstructor<int> matC_tool;
    
    // 数据算子组初始化
    Dac_Ops matC_ops;
    
    idx1.setDimId(0);
    matC_ops.push_back(idx1);
    idx2.setDimId(1);
    matC_ops.push_back(idx2);

    matC_tool.init(info_matC,matC_ops);
    buffer<int> r_matC{matC_Size};
    matC_tool.Reconstruct(r_matC,b_matC,q);
	std::vector<int> info_partition_matC=para_gene_tool.init_partition_data_shape(info_matC,matC_ops);
    sycl::buffer<int> info_partition_matC_buffer(info_partition_matC.data(), sycl::range<1>(info_partition_matC.size()));
    
	
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

    //队列提交命令组
    q.submit([&](handler &h) {
    
        accessor acc_matA{r_matA, h};
        accessor acc_matB{r_matB, h};
        accessor acc_matC{r_matC, h};
    
        auto info_partition_matA_accessor = info_partition_matA_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_matB_accessor = info_partition_matB_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_matC_accessor = info_partition_matC_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 获得accessor指针
            
            auto* d_matA = acc_matA.get_multi_ptr<access::decorated::no>().get();
            auto* d_matB = acc_matB.get_multi_ptr<access::decorated::no>().get();
            auto* d_matC = acc_matC.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            matrixMultiply_calc(d_matA+(idx1_*SplitLength[0][0]),d_matB+(idx2_*SplitLength[1][0]),d_matC+(idx1_*SplitLength[2][0]+idx2_*SplitLength[2][1]),info_partition_matA_accessor,info_partition_matB_accessor,info_partition_matC_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<matCReduction_Size;i++) {
            q.submit([&](handler &h) {
                accessor d_matC{r_matC, h};
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(b_reduction_matC[i], h, 
                sycl::plus<>(),property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_matC[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_matC};
            for(int i = 0; i < matCReduction_Size; i++){
                host_accessor temp_accessor{b_reduction_matC[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }


    //结果返回
    matC_tool.UpdateData(r_matC,b_matC,q);
    {
        host_accessor temp_accessor{b_matC};
        for(int i = 0; i < matC_Size; i++){
            h_matC[i] = temp_accessor[i];
        }
    }
    matC.array2Tensor(h_matC);

	

    // 内存释放
    
}

int main() {
    // 初始化两个矩阵 A 和 B
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    dacpp::Matrix<int> matA({4, 5}, dataA);

    std::vector<int> dataB{1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20};
    dacpp::Matrix<int> matB({5, 4}, dataB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Matrix<int> matC({4, 4}, dataC);

    matrixMultiply_shell_matrixMultiply_calc(matA, matB, matC);
    matC.print();

    return 0;
}
