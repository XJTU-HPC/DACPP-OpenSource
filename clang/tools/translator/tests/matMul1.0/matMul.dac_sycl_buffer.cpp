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

void matrixMultiply_calc(const int* vecA,const int* vecB,int* dotProduct,int vecA_0,int vecA_1,int vecB_0,int vecB_1,int dotProduct_0,int dotProduct_1,int vecA_0_shape,int vecA_1_shape,int vecB_0_shape,int vecB_1_shape,int dotProduct_0_shape,int dotProduct_1_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_vecA_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_vecB_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_dotProduct_acc) {
    for (int i = 0; i < 5; i++) {
        dotProduct[(0 + dotProduct_0) * dotProduct_1_shape + (0 + dotProduct_1)] += vecA[(0 + vecA_0) * vecA_1_shape + (i + vecA_1)] * vecB[(i + vecB_0) * vecB_1_shape + (0 + vecB_1)];
    }
}


// 生成函数调用
void matrixMultiply_shell_matrixMultiply_calc(const dacpp::Matrix<int> & matA, const dacpp::Matrix<int> & matB, dacpp::Matrix<int> & matC) { 
    using namespace sycl;
    // 设备选择
    auto selector = default_selector_v;
    sycl::queue dacpp_q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_matA;
    info_matA.dim = matA.getDim();
    int info_matA_Shape[2] = {0};
    for(int i = 0; i < info_matA.dim; i++)
    {
        info_matA.dimLength.push_back(matA.getShape(i));
        info_matA_Shape[i] = matA.getShape(i);
    }
	
    // 数据信息初始化
    DataInfo info_matB;
    info_matB.dim = matB.getDim();
    int info_matB_Shape[2] = {0};
    for(int i = 0; i < info_matB.dim; i++)
    {
        info_matB.dimLength.push_back(matB.getShape(i));
        info_matB_Shape[i] = matB.getShape(i);
    }
	
    // 数据信息初始化
    DataInfo info_matC;
    info_matC.dim = matC.getDim();
    int info_matC_Shape[2] = {0};
    for(int i = 0; i < info_matC.dim; i++)
    {
        info_matC.dimLength.push_back(matC.getShape(i));
        info_matC_Shape[i] = matC.getShape(i);
    }
	
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


	
	
	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    // 设备内存分配
    
    // 数据关联计算
    
	
    // 数据移动
    int* h_matA = (int*)malloc(matA.getSize()*sizeof(int));
    matA.tensor2Array(h_matA);
	buffer<int, 1> r_matA(h_matA, range<1>(matA.getSize()));

    // 数据移动
    int* h_matB = (int*)malloc(matB.getSize()*sizeof(int));
    matB.tensor2Array(h_matB);
	buffer<int, 1> r_matB(h_matB, range<1>(matB.getSize()));

    // 数据移动
    int* h_matC = (int*)malloc(matC.getSize()*sizeof(int));

    // 数据重组
    
    
    // 数据算子组初始化
    Dac_Ops matA_ops;
    
    idx1.setDimId(0);
    matA_ops.push_back(idx1);


	std::vector<int> info_partition_matA=para_gene_tool.init_partition_data_shape(info_matA,matA_ops);
    sycl::buffer<int> info_partition_matA_buffer(info_partition_matA.data(), sycl::range<1>(info_partition_matA.size()));

    // 数据重组
    
    
    // 数据算子组初始化
    Dac_Ops matB_ops;
    
    idx2.setDimId(1);
    matB_ops.push_back(idx2);


	std::vector<int> info_partition_matB=para_gene_tool.init_partition_data_shape(info_matB,matB_ops);
    sycl::buffer<int> info_partition_matB_buffer(info_partition_matB.data(), sycl::range<1>(info_partition_matB.size()));

    // 数据重组
    
    // 数据算子组初始化
    Dac_Ops matC_ops;
    
    idx1.setDimId(0);
    matC_ops.push_back(idx1);
    idx2.setDimId(1);
    matC_ops.push_back(idx2);


    auto r_matC = std::make_unique<sycl::buffer<int, 1>>(h_matC,sycl::range<1>(matC.getSize()));
    r_matC->set_final_data(h_matC);

	std::vector<int> info_partition_matC=para_gene_tool.init_partition_data_shape(info_matC,matC_ops);
    sycl::buffer<int> info_partition_matC_buffer(info_partition_matC.data(), sycl::range<1>(info_partition_matC.size()));
    
	
	
    sycl::device device = dacpp_q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];

	// 二维划分（可测试三维拓展）
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);

    // 固定 local 为 16*16,但受设备上限约束
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);

    // 对齐 global 到 local 的整数倍（防止越界）
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);

    //队列提交命令组
    dacpp_q.submit([&](handler &h) {
    
        accessor<int, 1, access::mode::read> acc_matA(r_matA, h);
        r_matA.set_final_data(nullptr);
        
        accessor<int, 1, access::mode::read> acc_matB(r_matB, h);
        r_matB.set_final_data(nullptr);
        
        accessor<int, 1, sycl::access::mode::discard_write> acc_matC(*r_matC, h);
    
        auto info_partition_matA_accessor = info_partition_matA_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matB_accessor = info_partition_matB_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matC_accessor = info_partition_matC_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
			// 获得划分数据单元左上角（第一个元素）的位置
			
			const auto matA_0 = idx1_ * idx1.stride;
			const auto matA_1 = 0;
			const auto matB_0 = 0;
			const auto matB_1 = idx2_ * idx2.stride;
			const auto matC_0 = idx1_ * idx1.stride;
			const auto matC_1 = idx2_ * idx2.stride;
            // 获得accessor指针
            
            auto* d_matA = acc_matA.get_multi_ptr<access::decorated::no>().get();
            auto* d_matB = acc_matB.get_multi_ptr<access::decorated::no>().get();
            auto* d_matC = acc_matC.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            matrixMultiply_calc(d_matA,d_matB,d_matC,matA_0,matA_1,matB_0,matB_1,matC_0,matC_1,info_matA_Shape[0],info_matA_Shape[1],info_matB_Shape[0],info_matB_Shape[1],info_matC_Shape[0],info_matC_Shape[1],info_partition_matA_accessor,info_partition_matB_accessor,info_partition_matC_accessor);
        });
    }).wait();
    

	
    //结果返回语句改为析构语句
    r_matC.reset();
    matC.array2Tensor(h_matC);

	

}

int main() {
    // 初始化两个矩阵 A 和 B
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    dacpp::Matrix<int> matA({4, 5}, dataA);

    std::vector<int> dataB{1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20};
    dacpp::Matrix<int> matB({5, 4}, dataB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Matrix<int> matC({4, 4}, dataC);
    // for(int i=0;i<1;i++){
    matrixMultiply_shell_matrixMultiply_calc(matA, matB, matC);
    // }
    matC.print();

    return 0;
}
