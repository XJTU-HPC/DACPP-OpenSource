#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;





#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void matrixMultiply_calc(int* vecA, int* vecB, int* dotProduct) 
{
    for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}


// 生成函数调用
void matrixMultiply_shell(const dacpp::Tensor<int, 2> & matA, const dacpp::Tensor<int, 2> & matB, dacpp::Tensor<int, 2> & matC) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    //ParameterGeneration<int,2> para_gene_tool;
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
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_matC,Reduction_Ops);

	
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
    
    // 设备内存分配
    int *d_matA=malloc_device<int>(matA_Size,q);
    // 设备内存分配
    int *d_matB=malloc_device<int>(matB_Size,q);
    // 设备内存分配
    int *d_matC=malloc_device<int>(matC_Size,q);
    // 归约设备内存分配
    int *reduction_matC = malloc_device<int>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<int> matA_tool;
    int* r_matA=(int*)malloc(sizeof(int)*matA_Size);
    
    // 数据算子组初始化
    Dac_Ops matA_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    matA_ops.push_back(idx1);
    matA_tool.init(info_matA,matA_ops);
    matA_tool.Reconstruct(r_matA,matA);
    // 数据重组
    DataReconstructor<int> matB_tool;
    int* r_matB=(int*)malloc(sizeof(int)*matB_Size);
    
    // 数据算子组初始化
    Dac_Ops matB_ops;
    
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    matB_ops.push_back(idx2);
    matB_tool.init(info_matB,matB_ops);
    matB_tool.Reconstruct(r_matB,matB);
    // 数据重组
    DataReconstructor<int> matC_tool;
    int* r_matC=(int*)malloc(sizeof(int)*matC_Size);
    
    // 数据算子组初始化
    Dac_Ops matC_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    matC_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    matC_ops.push_back(idx2);
    matC_tool.init(info_matC,matC_ops);
    matC_tool.Reconstruct(r_matC,matC);
    
    // 数据移动
    q.memcpy(d_matA,r_matA,matA_Size*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_matB,r_matB,matB_Size*sizeof(int)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 嵌入计算
			
            matrixMultiply_calc(d_matA+(idx1_*SplitLength[0][0]),d_matB+(idx2_*SplitLength[1][0]),d_matC+(idx1_*SplitLength[2][0]+idx2_*SplitLength[2][1]));
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_matC+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_matC[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_matC,reduction_matC, Reduction_Size*sizeof(int)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_matC, d_matC, matC_Size*sizeof(int)).wait();
    matC_tool.UpdateData(r_matC,matC);

    // 内存释放
    
    sycl::free(d_matA, q);
    sycl::free(d_matB, q);
    sycl::free(d_matC, q);
}

int main() {
    // 初始化两个矩阵 A 和 B
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    dacpp::Tensor<int, 2> matA({4, 5}, dataA);

    std::vector<int> dataB{1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20};
    dacpp::Tensor<int, 2> matB({5, 4}, dataB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Tensor<int, 2> matC({4, 4}, dataC);

    matrixMultiply_shell(matA, matB, matC);
    matC.print();

    return 0;
}
