#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}
const int N = 8;  // 假设数组的大小为1024


// 交换函数
void swap(vector<int>& array, int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}





// 奇偶归并排序的核心操作
#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void oddeven(int* array,int* array_out,sycl::accessor<int, 1, sycl::access::mode::read_write> info_array_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_array_out_acc) 
{
    if (array[0] > array[1]) {
        array_out[0] = array[1];
        array_out[1] = array[0];
    } else {
        array_out[0] = array[0];
        array_out[1] = array[1];
    }
}


// // 生成函数调用
// void ODDEVEN_oddeven(const dacpp::Vector<int> & array, dacpp::Vector<int> & array_out) { 
//     // 设备选择
//     auto selector = default_selector_v;
//     queue q(selector);
//     //声明参数生成工具
//     ParameterGeneration<int,2> para_gene_tool;
//     // 算子初始化
    
//     // 数据信息初始化
//     DataInfo info_array;
//     info_array.dim = array.getDim();
//     for(int i = 0; i < info_array.dim; i++) info_array.dimLength.push_back(array.getShape(i));
//     // 数据信息初始化
//     DataInfo info_array_out;
//     info_array_out.dim = array_out.getDim();
//     for(int i = 0; i < info_array_out.dim; i++) info_array_out.dimLength.push_back(array_out.getShape(i));
//     // 规则分区算子初始化
//     RegularSlice S1 = RegularSlice("S1", 2, 2);
//     S1.setDimId(0);
//     S1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(S1,info_array));

//     //参数生成
	
//     // 参数生成 提前计算后面需要用到的参数	
	
//     // 算子组初始化
//     Dac_Ops array_Ops;
    
//     S1.setDimId(0);
//     array_Ops.push_back(S1);


//     // 算子组初始化
//     Dac_Ops array_out_Ops;
    
//     S1.setDimId(0);
//     array_out_Ops.push_back(S1);


//     // 算子组初始化
//     Dac_Ops In_Ops;
    
//     S1.setDimId(0);
//     In_Ops.push_back(S1);


//     // 算子组初始化
//     Dac_Ops Out_Ops;
    
//     S1.setDimId(0);
//     Out_Ops.push_back(S1);


//     // 算子组初始化
//     Dac_Ops Reduction_Ops;
    
//     S1.setDimId(0);
//     Reduction_Ops.push_back(S1);


	
//     //生成设备内存分配大小
//     int array_Size = para_gene_tool.init_device_memory_size(info_array,array_Ops);

//     //生成设备内存分配大小
//     int array_out_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_array_out);

//     //生成设备内存分配大小
//     int Reduction_Size = para_gene_tool.init_device_memory_size(info_array_out,Reduction_Ops);

	
//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(array_Ops,array_Size);

//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(In_Ops,array_out_Size);

	
	
//     std::vector<Dac_Ops> ops_s;
	
//     ops_s.push_back(array_Ops);

//     ops_s.push_back(In_Ops);


// 	// 生成划分长度的二维矩阵
//     int SplitLength[2][1] = {0};
//     para_gene_tool.init_split_length_martix(2,1,&SplitLength[0][0],ops_s);

	
//     // 计算工作项的大小
//     int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
//     // 计算归约中split_size的大小
//     int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
//     // 计算归约中split_length的大小
//     int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


//     // 设备内存分配
    
//     // 设备内存分配
//     int *d_array=malloc_device<int>(array_Size,q);
//     // 设备内存分配
//     int *d_array_out=malloc_device<int>(array_out_Size,q);
//     // 归约设备内存分配
//     int *reduction_array_out = malloc_device<int>(Reduction_Size,q);
//     // 数据关联计算
    
    
//     // 数据重组
//     DataReconstructor<int> array_tool;
//     int* r_array=(int*)malloc(sizeof(int)*array_Size);
    
//     // 数据算子组初始化
//     Dac_Ops array_ops;
    
//     S1.setDimId(0);
//     S1.setSplitLength(8);
//     array_ops.push_back(S1);
//     array_tool.init(info_array,array_ops);
//     array_tool.Reconstruct(r_array,array);
// 	std::vector<int> info_partition_array=para_gene_tool.init_partition_data_shape(info_array,array_ops);
//     sycl::buffer<int> info_partition_array_buffer(info_partition_array.data(), sycl::range<1>(info_partition_array.size()));
//     // 数据重组
//     DataReconstructor<int> array_out_tool;
//     int* r_array_out=(int*)malloc(sizeof(int)*array_out_Size);
    
//     // 数据算子组初始化
//     Dac_Ops array_out_ops;
    
//     S1.setDimId(0);
//     S1.setSplitLength(8);
//     array_out_ops.push_back(S1);
//     array_out_tool.init(info_array_out,array_out_ops);
//     array_out_tool.Reconstruct(r_array_out,array_out);
// 	std::vector<int> info_partition_array_out=para_gene_tool.init_partition_data_shape(info_array_out,array_out_ops);
//     sycl::buffer<int> info_partition_array_out_buffer(info_partition_array_out.data(), sycl::range<1>(info_partition_array_out.size()));
    
//     // 数据移动
//     q.memcpy(d_array,r_array,array_Size*sizeof(int)).wait();
	
//     //工作项划分
//     sycl::range<3> local(1, 1, Item_Size);
//     sycl::range<3> global(1, 1, 1);
//     //队列提交命令组
//     q.submit([&](handler &h) {
//         // 访问器初始化
        
//         auto info_partition_array_accessor = info_partition_array_buffer.get_access<sycl::access::mode::read_write>(h);
//         auto info_partition_array_out_accessor = info_partition_array_out_buffer.get_access<sycl::access::mode::read_write>(h);
//         h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
//             const auto item_id = item.get_local_id(2);
//             // 索引初始化
			
//             const auto S1_=(item_id+(0))%S1.split_size;
//             // 嵌入计算
			
//             oddeven(d_array+(S1_*SplitLength[0][0]),d_array_out+(S1_*SplitLength[1][0]),info_partition_array_accessor,info_partition_array_out_accessor);
//         });
//     }).wait();
    

	
//     // 归约
//     if(Reduction_Split_Size > 1)
//     {
//         for(int i=0;i<Reduction_Size;i++) {
//             q.submit([&](handler &h) {
//     	        h.parallel_for(
//                 range<1>(Reduction_Split_Size),
//                 reduction(reduction_array_out+i, 
//                 sycl::plus<>(),
//                 property::reduction::initialize_to_identity()),
//                 [=](id<1> idx,auto &reducer) {
//                     reducer.combine(d_array_out[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
//      	        });
//          }).wait();
//         }
//         q.memcpy(d_array_out,reduction_array_out, Reduction_Size*sizeof(int)).wait();
//     }


	
//     // 归并结果返回
//     q.memcpy(r_array_out, d_array_out, array_out_Size*sizeof(int)).wait();
//     array_out_tool.UpdateData(r_array_out,array_out);

//     // 内存释放
    
//     sycl::free(d_array, q);
//     sycl::free(d_array_out, q);
// }

void oddEvenMergeSort(vector<int>& array, int n) {
    dacpp::Tensor<int, 1> array_tensor(array);
    vector<int> array_out(N);
    dacpp::Tensor<int, 1> array_out_tensor(array_out);

    // 每一轮排序进行多次比较
    for (int phase = 0; phase < N; phase++) {
        // 奇数阶段：比较相邻的奇数索引
        //array_tensor.print();
        // ODDEVEN_oddeven(array_tensor, array_out_tensor);
        //array_out_tensor.print();
        // vector<int> array2(N-2, 0);
        //dacpp::Tensor<int, 1> array2_tensor(array2);

        // for(int i = 0;i < N-2; i++){
        //     array2_tensor[i] = array_out_tensor[i+1];
        // }
        dacpp::Tensor<int, 1> array2_tensor = array_out_tensor[{1,N-1}];
        vector<int> array_out2(N-2, 0);
        dacpp::Tensor<int, 1> array_out2_tensor(array_out2);
        //array2_tensor.print();
        // ODDEVEN_oddeven(array2_tensor, array_out2_tensor);

        for(int i = 1;i < N-1; i++){
            array_tensor[i] = array_out2_tensor[i-1];
        }
        array_tensor[0] = array_out_tensor[0];
        array_tensor[N-1] = array_out_tensor[N-1];
    }
    array_tensor.print();
}

// 主函数：初始化数据并调用奇偶归并排序
int main() {
    vector<int> array(N);

    // 初始化数据
    // srand(time(0));
    // for (int i = 0; i < N; i++) {
    //     array[i] = rand() % 10;  // 随机生成0到1000之间的整数
    // }
    for (int i = N; i > 0; i--) {
        array[i] = N-i;  // 随机生成0到1000之间的整数
    }

    // 打印排序前的数组（前10个）
    std::cout << "Array before sorting (first 10 elements):" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    // 执行奇偶归并排序
    oddEvenMergeSort(array, N);


    return 0;
}
