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

const int N = 1024; // 矩阵维度

void matrixMultiply_calc(int* vecA,int* vecB,int* dotProduct,sycl::accessor<int, 1, sycl::access::mode::read_write> info_vecA_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_vecB_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_dotProduct_acc) 
{
    for (int i = 0; i < N; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}


    // 生成函数调用
    void matrixMultiply_shell_matrixMultiply_calc(const dacpp::Matrix<int> & matA, const dacpp::Matrix<int> & matB, dacpp::Matrix<int> & matC) {
        double total_kernel_time = 0.0;
        double total_data_transfer_time = 0.0;
        auto start_time = std::chrono::high_resolution_clock::now(); 
        // 设备选择
        auto devices = device::get_devices(info::device_type::gpu);
        vector<queue> q;
        for (const auto& dev : devices) {
            q.emplace_back(dev, sycl::property::queue::enable_profiling{});
        }
        int numDevices = q.size();
        // printf("Running on %d GPU\n", numDevices);
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
    Dac_Ops matCReduction_Ops;
    
    idx1.setDimId(0);
    matCReduction_Ops.push_back(idx1);

    idx2.setDimId(1);
    matCReduction_Ops.push_back(idx2);


	
    //生成设备内存分配大小
    int matA_Size = para_gene_tool.init_device_memory_size(info_matA,matA_Ops);

    //生成设备内存分配大小
    int matB_Size = para_gene_tool.init_device_memory_size(info_matB,matB_Ops);

    //生成设备内存分配大小
    int matC_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_matC);

    //生成设备内存分配大小
    int matCReduction_Size = para_gene_tool.init_device_memory_size(info_matC,matCReduction_Ops);

	
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
        
    // 数据重组
    DataReconstructor<int> matA_tool;

    
    // 数据算子组初始化
    Dac_Ops matA_ops;
    
    idx1.setDimId(0);
    matA_ops.push_back(idx1);
    matA_tool.init(info_matA,matA_ops);
    int recon_h_matA[matA_Size];
    matA.tensor2Array(recon_h_matA);
    int *recon_d_matA=malloc_device<int>(matA_Size,q[0]);
    q[0].memcpy(recon_d_matA, recon_h_matA, matA_Size*sizeof(int)).wait();
    int *r_matA=malloc_device<int>(matA_Size,q[0]);
    matA_tool.Reconstruct(r_matA, recon_d_matA,q[0]);
    sycl::free(recon_d_matA, q[0]);
    std::vector<int> info_partition_matA=para_gene_tool.init_partition_data_shape(info_matA,matA_ops);
    sycl::buffer<int> info_partition_matA_buffer(info_partition_matA.data(), sycl::range<1>(info_partition_matA.size()));
    // 数据重组
    DataReconstructor<int> matB_tool;

    
    // 数据算子组初始化
    Dac_Ops matB_ops;
    
    idx2.setDimId(1);
    matB_ops.push_back(idx2);
    matB_tool.init(info_matB,matB_ops);
    int recon_h_matB[matB_Size];
    matB.tensor2Array(recon_h_matB);
    int *recon_d_matB=malloc_device<int>(matB_Size,q[0]);
    q[0].memcpy(recon_d_matB, recon_h_matB, matB_Size*sizeof(int)).wait();
    int *r_matB=malloc_device<int>(matB_Size,q[0]);
    matB_tool.Reconstruct(r_matB, recon_d_matB,q[0]);
    sycl::free(recon_d_matB, q[0]);
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
    int *r_matC=malloc_device<int>(matC_Size,q[0]);
    q[0].memset(r_matC, 0, matC_Size * sizeof(int)).wait();
    std::vector<int> info_partition_matC=para_gene_tool.init_partition_data_shape(info_matC,matC_ops);
    sycl::buffer<int> info_partition_matC_buffer(info_partition_matC.data(), sycl::range<1>(info_partition_matC.size()));
        int matA_data_SplitNum = para_gene_tool.init_Data_SplitNum(matA_Ops,info_matA);
        int matA_data_SplitSize = para_gene_tool.init_Data_SplitSize(matA_data_SplitNum,matA_Size);
    
        std::vector<int> matA_Device_Size = para_gene_tool.init_Device_Size(numDevices,matA_data_SplitSize,matA_data_SplitNum);
    
        int matB_data_SplitNum = para_gene_tool.init_Data_SplitNum(matB_Ops,info_matB);
        int matB_data_SplitSize = para_gene_tool.init_Data_SplitSize(matB_data_SplitNum,matB_Size);
    
        std::vector<int> matB_Device_Size(numDevices, matB_Size/matB_data_SplitSize);
    
        int matC_data_SplitNum = para_gene_tool.init_Data_SplitNum(matC_Ops,info_matC);
        int matC_data_SplitSize = para_gene_tool.init_Data_SplitSize(matC_data_SplitNum,matC_Size);
    
        // 数据关联计算
        
    std::vector<sycl::event> events;
    
        std::vector<int*> d_matAs(numDevices);
        
        std::vector<int*> d_matBs(numDevices);
        
        std::vector<int*> d_matCs(numDevices);
        
        std::vector<int*> reduction_matCs(numDevices);
        
        int matA_HaveMalloc = 0;
    
        int matB_HaveMalloc = 0;
    
        int matC_HaveMalloc = 0;
    
        int Result_HaveMalloc = 0;
    
    
    // 设备内存分配
    int *d_mymatC=malloc_device<int>(matC_Size,q[0]);
    for(int numDevice = 0; numDevice < numDevices; numDevice++){
        printf("Running on GPU %d\n", numDevice);
        // 设备内存分配
        int *d_matA=malloc_device<int>(matA_Device_Size[numDevice] * matA_data_SplitSize,q[numDevice]);
        // 设备内存分配
        int *d_matB=malloc_device<int>(matB_Device_Size[numDevice] * matB_data_SplitSize,q[numDevice]);
        // 归约设备内存分配
        int *d_matC = malloc_device<int>(matA_Device_Size[numDevice]*matB_Device_Size[numDevice]*matC_data_SplitSize,q[numDevice]);
        // 归约设备内存分配
        int *reduction_matC = malloc_device<int>(matCReduction_Size,q[numDevice]);
        // 数据移动


        // 数据移动
        q[numDevice].memcpy(d_matA,r_matA + matA_HaveMalloc , matA_Device_Size[numDevice] * matA_data_SplitSize * sizeof(int)).wait();
        matA_HaveMalloc += matA_Device_Size[numDevice] * matA_data_SplitSize;
        // 数据移动
        q[numDevice].memcpy(d_matB,r_matB ,matB_Device_Size[numDevice] * matB_data_SplitSize * sizeof(int)).wait();
        //内核执行
        
        //工作项划分
        sycl::device device = q[numDevice].get_device();
        int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	    //工作项划分
        int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
        Item_Size = matA_Device_Size[numDevice]*matB_Device_Size[numDevice];
        
        sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
        sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
        //队列提交命令组
         events.push_back(q[numDevice].submit([&](handler &h) {
            // 访问器初始化
            
        auto info_partition_matA_accessor = info_partition_matA_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_matB_accessor = info_partition_matB_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_matC_accessor = info_partition_matC_buffer.get_access<sycl::access::mode::read_write>(h);

            h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                const auto item_id = item.get_local_id(2)*item.get_local_range(2)+item.get_local_id(2);;
                // 索引初始化
                
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
                // 嵌入计算
                
        matrixMultiply_calc(d_matA+(idx1_*SplitLength[0][0]),d_matB+(((idx2_+ Result_HaveMalloc)%idx2.split_size)*SplitLength[1][0]),d_matC+(idx1_*SplitLength[2][0]+idx2_*SplitLength[2][1]),info_partition_matA_accessor,info_partition_matB_accessor,info_partition_matC_accessor);
            });
        }));
        Result_HaveMalloc += matA_Device_Size[numDevice]*matB_Device_Size[numDevice];
        
    
        //保存内存地址
        
        //保存内存地址
        d_matAs[numDevice] = d_matA;
        //保存内存地址
        d_matBs[numDevice] = d_matB;
        //保存内存地址
        d_matCs[numDevice] = d_matC;
        //保存内存地址
        reduction_matCs[numDevice] = reduction_matC;
    }
        // 等待所有内核完成并记录时间
    for (auto& e : events) {
        e.wait();
        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        double kernel_time = (end - start) * 1e-6; // 纳秒转毫秒
        total_kernel_time += kernel_time;
    }
    
    for(int numDevice = 0; numDevice < numDevices; numDevice++){
        //获取内存地址
        
        //获取内存地址
        int *d_matA = d_matAs[numDevice];
        //获取内存地址
        int *d_matB = d_matBs[numDevice];
        //获取内存地址
        int *d_matC = d_matCs[numDevice];
        //获取内存地址
        int *reduction_matC = reduction_matCs[numDevice];
        // 归约
        
    // 归约
    if(Reduction_Split_Size > 1)
    {
        events[numDevice].wait();
        for(int i=0;i<matCReduction_Size;i++) {
            q[numDevice].submit([&](handler &h) {
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
        q[numDevice].memcpy(d_matC,reduction_matC, matCReduction_Size*sizeof(int)).wait();
    }

        
    // 归并结果返回
    q[numDevice].memcpy(d_mymatC + matC_HaveMalloc, d_matC, matA_Device_Size[numDevice]*matB_Device_Size[numDevice]*matC_data_SplitSize*sizeof(int)).wait();
    matC_HaveMalloc += matA_Device_Size[numDevice]*matB_Device_Size[numDevice]*matC_data_SplitSize;
        // 内存释放
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> total_duration = end_time - start_time;
    // std::cout << "内核计算时间: " << total_kernel_time << " ms" << std::endl;
    // std::cout << "总执行时间: " << total_duration.count() << " ms" << std::endl;

    sycl::free(d_matA, q[numDevice]);
    sycl::free(d_matB, q[numDevice]);
    sycl::free(d_matC, q[numDevice]);
    sycl::free(reduction_matC, q[numDevice]);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = end_time - start_time;
    std::cout << "内核计算时间: " << total_kernel_time << " ms" << std::endl;
    std::cout << "总执行时间: " << total_duration.count() << " ms" << std::endl;
    // 归并结果返回
	int *d_myTensor2=malloc_device<int>(matC_Size,q[0]);
    matC_tool.UpdateData(d_mymatC,d_myTensor2,q[0]);
	int* res = (int*)malloc(matC_Size*sizeof(int));
	q[0].memcpy(res,d_myTensor2, matC_Size*sizeof(int)).wait();
	matC.array2Tensor(res);
    }

int main() {
    // 初始化两个矩阵 A 和 B
    
    // 生成全1矩阵A (N×N)
    std::vector<int> dataA(N * N, 1);
    dacpp::Matrix<int> matA({N, N}, dataA);
    
    // 生成全1矩阵B (N×N)
    std::vector<int> dataB(N * N, 1);
    dacpp::Matrix<int> matB({N, N}, dataB);
    
    // 生成结果矩阵C (N×N)，初始为全0
    std::vector<int> dataC(N * N, 0);
    dacpp::Matrix<int> matC({N, N}, dataC);
    
    // 执行矩阵乘法
    matrixMultiply_shell_matrixMultiply_calc(matA, matB, matC);
    // matC.print();

    return 0;
}
