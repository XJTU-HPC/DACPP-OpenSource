#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "ReconTensor.h"
#define DACPP_TRANSLATE_MODE 1

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
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void oddeven(const int* array,int* array_out,int array_0,int array_out_0,int array_0_shape,int array_out_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_array_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_array_out_acc) {
    if (array[0+array_0] > array[1+array_0]) {
        array_out[0+array_out_0] = array[1+array_0];
        array_out[1+array_out_0] = array[0+array_0];
    } else {
        array_out[0+array_out_0] = array[0+array_0];
        array_out[1+array_out_0] = array[1+array_0];
    }
}


// 生成函数调用
void ODDEVEN_oddeven(const dacpp::Vector<int> & array, dacpp::Vector<int> & array_out) { 
    using namespace sycl;
    // 设备选择
    auto selector = default_selector_v;
    sycl::queue dacpp_q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_array;
    info_array.dim = array.getDim();
    int info_array_Shape[1] = {0};
    for(int i = 0; i < info_array.dim; i++)
    {
        info_array.dimLength.push_back(array.getShape(i));
        info_array_Shape[i] = array.getShape(i);
    }
	
    // 数据信息初始化
    DataInfo info_array_out;
    info_array_out.dim = array_out.getDim();
    int info_array_out_Shape[1] = {0};
    for(int i = 0; i < info_array_out.dim; i++)
    {
        info_array_out.dimLength.push_back(array_out.getShape(i));
        info_array_out_Shape[i] = array_out.getShape(i);
    }
	
    // 规则分区算子初始化
    RegularSlice S1 = RegularSlice("S1", 2, 2);
    S1.setDimId(0);
    S1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(S1,info_array));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops array_Ops;
    
    S1.setDimId(0);
    array_Ops.push_back(S1);


    // 算子组初始化
    Dac_Ops array_out_Ops;
    
    S1.setDimId(0);
    array_out_Ops.push_back(S1);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    S1.setDimId(0);
    In_Ops.push_back(S1);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    S1.setDimId(0);
    Out_Ops.push_back(S1);


	
	
	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    // 设备内存分配
    
    // 数据关联计算
    
	
    // 数据移动
    int* h_array = (int*)malloc(array.getSize()*sizeof(int));
    array.tensor2Array(h_array);
	buffer<int, 1> r_array(h_array, range<1>(array.getSize()));

    // 数据移动
    int* h_array_out = (int*)malloc(array_out.getSize()*sizeof(int));

    // 数据重组
    
    
    // 数据算子组初始化
    Dac_Ops array_ops;
    
    S1.setDimId(0);
    array_ops.push_back(S1);


	std::vector<int> info_partition_array=para_gene_tool.init_partition_data_shape(info_array,array_ops);
    sycl::buffer<int> info_partition_array_buffer(info_partition_array.data(), sycl::range<1>(info_partition_array.size()));

    // 数据重组
    
    // 数据算子组初始化
    Dac_Ops array_out_ops;
    
    S1.setDimId(0);
    array_out_ops.push_back(S1);


    auto r_array_out = std::make_unique<sycl::buffer<int, 1>>(h_array_out,sycl::range<1>(array_out.getSize()));
    r_array_out->set_final_data(h_array_out);

	std::vector<int> info_partition_array_out=para_gene_tool.init_partition_data_shape(info_array_out,array_out_ops);
    sycl::buffer<int> info_partition_array_out_buffer(info_partition_array_out.data(), sycl::range<1>(info_partition_array_out.size()));
    
	
	
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
    
        accessor<int, 1, access::mode::read> acc_array(r_array, h);
        
        accessor<int, 1, sycl::access::mode::discard_write> acc_array_out(*r_array_out, h);
    
        auto info_partition_array_accessor = info_partition_array_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_array_out_accessor = info_partition_array_out_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto S1_=(item_id+(0))%S1.split_size;
			// 获得划分数据单元左上角（第一个元素）的位置
			
			const auto array_0 = S1_ * S1.stride;
			const auto array_out_0 = S1_ * S1.stride;
            // 获得accessor指针
            
            auto* d_array = acc_array.get_multi_ptr<access::decorated::no>().get();
            auto* d_array_out = acc_array_out.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            oddeven(d_array,d_array_out,array_0,array_out_0,info_array_Shape[0],info_array_out_Shape[0],info_partition_array_accessor,info_partition_array_out_accessor);
        });
    }).wait();
    

	
    //结果返回语句改为析构语句
    r_array_out.reset();
    array_out.array2Tensor(h_array_out);

	

}

void oddEvenMergeSort(vector<int>& array, int n) {
    dacpp::Tensor<int, 1> array_tensor(array);
    vector<int> array_out(N);
    dacpp::Tensor<int, 1> array_out_tensor(array_out);

    // 每一轮排序进行多次比较
    for (int phase = 0; phase < N; phase++) {
        // 奇数阶段：比较相邻的奇数索引
        //array_tensor.print();
        ODDEVEN_oddeven(array_tensor, array_out_tensor);
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
        ODDEVEN_oddeven(array2_tensor, array_out2_tensor);

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
    for (int i = 0; i < N; i++) {
        array[i] = N - i;  // 初始化为递减的数组
    }

    // 打印排序前的数组（前10个）
    //std::cout << "Array before sorting (first 10 elements):" << std::endl;
    for (int i = 0; i < N; i++) {
        //std::cout << array[i] << " ";
    }
    //std::cout << std::endl;

    // 执行奇偶归并排序
    oddEvenMergeSort(array, N);


    return 0;
}
