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

void PrintTargetInfo(queue& q) {
    auto device = q.get_device();
    auto max_block_size =
        device.get_info<info::device::max_work_group_size>();
  
    auto max_EU_count =
        device.get_info<info::device::max_compute_units>();
  
    cout<< " Running on " << device.get_info<info::device::name>()<<"\n";
    cout<< " The Device Max Work Group Size is : "<< max_block_size<<"\n";
    cout<< " The Device Max EUCount is : " << max_EU_count<<"\n";
  }

// 生成函数调用
void matrixMultiply_shell(const dacpp::Tensor<int, 2> & matA, const dacpp::Tensor<int, 2> & matB, dacpp::Tensor<int, 2> & matC) { 
    // 设备选择
    // auto selector = gpu_selector_v;
    // queue q(selector);
    //声明参数生成工具

    auto devices = device::get_devices(info::device_type::gpu);
    vector<queue> q;
    for (const auto& dev : devices) {
        q.emplace_back(dev);
    }
    int numDevices = q.size();
    printf("Running on %d GPU\n", numDevices);
    

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
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++){
            cout << "Splitlength [" << i << "][" << j << "] = " << SplitLength[i][j] << " ";
        }
        cout << std::endl;
    }
	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


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
    //计算任务分配
    int matA_data_SplitNum = para_gene_tool.init_Data_SplitNum(matA_Ops,info_matA);
    printf("matA_data_SplitNum is %d\n", matA_data_SplitNum);
    int matB_data_SplitNum = para_gene_tool.init_Data_SplitNum(matB_Ops,info_matB);
    printf("matB_data_SplitNum is %d\n", matB_data_SplitNum);
  
    int matA_data_SplitSize = para_gene_tool.init_Data_SplitSize(matA_data_SplitNum,matA_Size);
    printf("matA_data_SplitSize is %d\n", matA_data_SplitSize);
    int matB_data_SplitSize = para_gene_tool.init_Data_SplitSize(matB_data_SplitNum,matB_Size);
    printf("matB_data_SplitSize is %d\n", matB_data_SplitSize);

    std::vector<int> matA_Device_Size = para_gene_tool.init_Device_Size(numDevices,matA_data_SplitSize,matA_data_SplitNum);
    for(int i = 0; i < matA_Device_Size.size(); i++){
        cout << "matA_Device_Size[" << i << "] = " << matA_Device_Size[i] << " ";
    }
    cout << std::endl;
    // std::vector<int> matB_Device_Size = para_gene_tool.init_Device_Size(numDevices,matB_data_SplitSize,matB_data_SplitNum);
    // for(int i = 0; i < matB_Device_Size.size(); i++){
    //     cout << "matB_Device_Size[" << i << "] = " << matB_Device_Size[i] << " ";
    // }
    // cout << std::endl;
    std::vector<int> matB_Device_Size(numDevices, 4);


    //数据划分
    std::vector<sycl::event> events;
    // 设备内存分配
    std::vector<int*> d_matAs(numDevices);
    std::vector<int*> d_matBs(numDevices);
    std::vector<int*> d_matCs(numDevices);
    std::vector<int*> reduction_matCs(numDevices);

    for(int numDevice = 0; numDevice < numDevices; numDevice++){
        // PrintTargetInfo(q[numDevice]);
        // 设备内存分配
        // int *d_matA = malloc_device<int>(matA_NumSplit/2*SplitLength[0][0],q[numDevice]);
        int *d_matA=malloc_device<int>(matA_Device_Size[numDevice] * matA_data_SplitSize,q[numDevice]);
        // 设备内存分配
        int *d_matB=malloc_device<int>(matB_Device_Size[numDevice] * matB_data_SplitSize,q[numDevice]);
        // 设备内存分配
        int *d_matC=malloc_device<int>(matC_Size/2,q[numDevice]);

        // 归约设备内存分配
        int *reduction_matC = malloc_device<int>(Reduction_Size,q[numDevice]);
    
        // 数据移动
        q[numDevice].memcpy(d_matA,r_matA + numDevice * matA_Size/2,matA_Size*sizeof(int)/2).wait();
        // 数据移动
        q[numDevice].memcpy(d_matB,r_matB,matB_Size*sizeof(int)).wait();
        
        // 确认 d_matA 的数据
        int* host_matA = (int*)malloc(sizeof(int) * Item_Size/2);
        q[numDevice].memcpy(host_matA, d_matA, matA_Size * sizeof(int)/2).wait();
        printf("The Device %d d_matA data is: ", numDevice);
        for (int i = 0; i < matA_Size/2; i++) {
            printf("%d ", host_matA[i]);
        }
        printf("\n");
        free(host_matA);

        // int* host_matB = (int*)malloc(sizeof(int) * Item_Size);
        // q[numDevice].memcpy(host_matB, d_matB, matB_Size * sizeof(int)).wait();
        // printf("The Device %d d_matB data is: ", numDevice);
        // for(int i = 0; i < matB_Size; i++){
        //     printf("%d ", host_matB[i]);
        // }
        // printf("\n");
        // free(host_matB);

        sycl::range<3> global(1, 1, Item_Size);
        sycl::range<3> local(1, 1, matA_Device_Size[numDevice]*matB_Device_Size[numDevice]);
        
        // printf("breakpoint 0\n");
        //队列提交命令组
        events.push_back(q[numDevice].submit([&](handler &h) {
            sycl::stream out(1024, 256, h);

            h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
                const auto item_id = item.get_local_id(2);
                
                // 索引初始化
                const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
                // const auto idx2_=(item_id+(0)+8 *numDevice)% idx2.split_size;
                const auto idx2_=(item_id+(0))%idx2.split_size;
                
                // 嵌入计算
                matrixMultiply_calc(d_matA+(idx1_*SplitLength[0][0]),d_matB+(idx2_*SplitLength[1][0]),d_matC+(idx1_*SplitLength[2][0]+idx2_*SplitLength[2][1]));
            });
        }));
        // printf("breakpoint 1\n");
        d_matAs[numDevice] = d_matA;
        d_matBs[numDevice] = d_matB;
        d_matCs[numDevice] = d_matC;
        reduction_matCs[numDevice] = reduction_matC;

    }
    
    for(int numDevice = 0; numDevice < numDevices; numDevice++){
        // 归约
        int *d_matA = d_matAs[numDevice];
        int *d_matB = d_matBs[numDevice];
        int *d_matC = d_matCs[numDevice];
        int *reduction_matC = reduction_matCs[numDevice];
        events[numDevice].wait();
        printf("Device %d is finish\n", numDevice);
        if(Reduction_Split_Size > 1){
            printf("Reduction!\n");
            for(int i=0;i<Reduction_Size;i++) {
                q[numDevice].submit([&](handler &h) {
    	            h.parallel_for(range<1>(Reduction_Split_Size/2),reduction(reduction_matC + i, sycl::plus<>(),property::reduction::initialize_to_identity()),
                    [=](id<1> idx,auto &reducer) {
                        reducer.combine(d_matC[(i/Reduction_Split_Length) * Reduction_Split_Length * Reduction_Split_Size + i % Reduction_Split_Length+idx * Reduction_Split_Length]);
     	            });
            });
            }
            q[numDevice].memcpy(d_matC,reduction_matC, Reduction_Size*sizeof(int)).wait();
        }
        // printf("breakpoint 2\n");
        // 归并结果返回
        q[numDevice].memcpy(r_matC+8 *numDevice, d_matC, matC_Size/2*sizeof(int)).wait();
         // 将 d_matC 的数据复制回主机内存
        //  printf("breakpoint 3\n");

         int* host_matC = (int*)malloc(sizeof(int) * Item_Size);
         q[numDevice].memcpy(host_matC, d_matC, Item_Size * sizeof(int)/2).wait();
         // 打印 d_matC 的值
         printf("The Device %d result is: ", numDevice);
         for (int i = 0; i < Item_Size/2; i++) {
             printf("%d ", host_matC[i]);
         }
         printf("\n");
 
         free(host_matC);

        // 内存释放
        sycl::free(d_matA, q[numDevice]);
        sycl::free(d_matB, q[numDevice]);
        sycl::free(d_matC, q[numDevice]);
        sycl::free(reduction_matC, q[numDevice]);
    }
    // printf("breakpoint 4\n");
    matC_tool.UpdateData(r_matC,matC);
    // for(int i = 0; i < 16; i++){
    //     printf("%d ", r_matC[i]);
    // }
    // printf("\n");
    // matC.print();
    // printf("breakpoint 5\n");
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
    // printf("breakpoint 6\n");
    matC.print();
    // printf("breakpoint 7\n");

    return 0;
}
