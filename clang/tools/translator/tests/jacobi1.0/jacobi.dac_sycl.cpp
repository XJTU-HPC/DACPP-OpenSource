#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
// 定义矩阵大小
const int N = 100; // 可以修改 N 的值来改变矩阵大小
const int max_iter = 10000;
const float tolerance = 1e-6;
namespace dacpp {
    typedef std::vector<std::any> list;
}






#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void jacobi(float* a,float* b,float* x,float* x_new,int* num,sycl::accessor<int, 1, sycl::access::mode::read_write> info_a_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_b_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_x_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_x_new_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_num_acc) 
{
    floa[0]t sigma[0] = 0;
    for (int i = 0; i < N; ++i) {
        if (i != num[0]) {
            sigma[0] += a[i] * x[i];
        }
    }
    x[0]_new[0] = (b[0] - sigma[0]) / a[num[0]];
}


// 生成函数调用
void jacobiShell_jacobi(const dacpp::Tensor<float, 2> & A, const dacpp::Tensor<float, 1> & b, const dacpp::Tensor<float, 1> & x, dacpp::Tensor<float, 1> & x_new, const dacpp::Tensor<int, 1> & nums) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_A;
    info_A.dim = A.getDim();
    for(int i = 0; i < info_A.dim; i++) info_A.dimLength.push_back(A.getShape(i));
    // 数据信息初始化
    DataInfo info_b;
    info_b.dim = b.getDim();
    for(int i = 0; i < info_b.dim; i++) info_b.dimLength.push_back(b.getShape(i));
    // 数据信息初始化
    DataInfo info_x;
    info_x.dim = x.getDim();
    for(int i = 0; i < info_x.dim; i++) info_x.dimLength.push_back(x.getShape(i));
    // 数据信息初始化
    DataInfo info_x_new;
    info_x_new.dim = x_new.getDim();
    for(int i = 0; i < info_x_new.dim; i++) info_x_new.dimLength.push_back(x_new.getShape(i));
    // 数据信息初始化
    DataInfo info_nums;
    info_nums.dim = nums.getDim();
    for(int i = 0; i < info_nums.dim; i++) info_nums.dimLength.push_back(nums.getShape(i));
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_A));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops A_Ops;
    
    idx1.setDimId(0);
    A_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops b_Ops;
    
    idx1.setDimId(0);
    b_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops x_Ops;
    

    // 算子组初始化
    Dac_Ops x_new_Ops;
    
    idx1.setDimId(0);
    x_new_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops nums_Ops;
    
    idx1.setDimId(0);
    nums_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    idx1.setDimId(0);
    In_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);


	
    //生成设备内存分配大小
    int A_Size = para_gene_tool.init_device_memory_size(info_A,A_Ops);

    //生成设备内存分配大小
    int b_Size = para_gene_tool.init_device_memory_size(info_b,b_Ops);

    //生成设备内存分配大小
    int x_Size = para_gene_tool.init_device_memory_size(info_x,x_Ops);

    //生成设备内存分配大小
    int x_new_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_x_new);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_x_new,Reduction_Ops);

    //生成设备内存分配大小
    int nums_Size = para_gene_tool.init_device_memory_size(info_nums,nums_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(A_Ops,A_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(b_Ops,b_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(x_Ops,x_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,x_new_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(nums_Ops,nums_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(A_Ops);

    ops_s.push_back(b_Ops);

    ops_s.push_back(x_Ops);

    ops_s.push_back(In_Ops);

    ops_s.push_back(nums_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[5][1] = {0};
    para_gene_tool.init_split_length_martix(5,1,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // 设备内存分配
    float *d_A=malloc_device<float>(A_Size,q);
    // 设备内存分配
    float *d_b=malloc_device<float>(b_Size,q);
    // 设备内存分配
    float *d_x=malloc_device<float>(x_Size,q);
    // 设备内存分配
    float *d_x_new=malloc_device<float>(x_new_Size,q);
    // 归约设备内存分配
    float *reduction_x_new = malloc_device<float>(Reduction_Size,q);
    // 设备内存分配
    int *d_nums=malloc_device<int>(nums_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<float> A_tool;
    float* r_A=(float*)malloc(sizeof(float)*A_Size);
    
    // 数据算子组初始化
    Dac_Ops A_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    A_ops.push_back(idx1);
    A_tool.init(info_A,A_ops);
    A_tool.Reconstruct(r_A,A);
	std::vector<int> info_partition_A=para_gene_tool.init_partition_data_shape(info_A,A_ops);
    sycl::buffer<int> info_partition_A_buffer(info_partition_A.data(), sycl::range<1>(info_partition_A.size()));
    // 数据重组
    DataReconstructor<float> b_tool;
    float* r_b=(float*)malloc(sizeof(float)*b_Size);
    
    // 数据算子组初始化
    Dac_Ops b_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    b_ops.push_back(idx1);
    b_tool.init(info_b,b_ops);
    b_tool.Reconstruct(r_b,b);
	std::vector<int> info_partition_b=para_gene_tool.init_partition_data_shape(info_b,b_ops);
    sycl::buffer<int> info_partition_b_buffer(info_partition_b.data(), sycl::range<1>(info_partition_b.size()));
    // 数据重组
    DataReconstructor<float> x_tool;
    float* r_x=(float*)malloc(sizeof(float)*x_Size);
    
    // 数据算子组初始化
    Dac_Ops x_ops;
    
    x_tool.init(info_x,x_ops);
    x_tool.Reconstruct(r_x,x);
	std::vector<int> info_partition_x=para_gene_tool.init_partition_data_shape(info_x,x_ops);
    sycl::buffer<int> info_partition_x_buffer(info_partition_x.data(), sycl::range<1>(info_partition_x.size()));
    // 数据重组
    DataReconstructor<float> x_new_tool;
    float* r_x_new=(float*)malloc(sizeof(float)*x_new_Size);
    
    // 数据算子组初始化
    Dac_Ops x_new_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    x_new_ops.push_back(idx1);
    x_new_tool.init(info_x_new,x_new_ops);
    x_new_tool.Reconstruct(r_x_new,x_new);
	std::vector<int> info_partition_x_new=para_gene_tool.init_partition_data_shape(info_x_new,x_new_ops);
    sycl::buffer<int> info_partition_x_new_buffer(info_partition_x_new.data(), sycl::range<1>(info_partition_x_new.size()));
    // 数据重组
    DataReconstructor<int> nums_tool;
    int* r_nums=(int*)malloc(sizeof(int)*nums_Size);
    
    // 数据算子组初始化
    Dac_Ops nums_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    nums_ops.push_back(idx1);
    nums_tool.init(info_nums,nums_ops);
    nums_tool.Reconstruct(r_nums,nums);
	std::vector<int> info_partition_nums=para_gene_tool.init_partition_data_shape(info_nums,nums_ops);
    sycl::buffer<int> info_partition_nums_buffer(info_partition_nums.data(), sycl::range<1>(info_partition_nums.size()));
    
    // 数据移动
    q.memcpy(d_A,r_A,A_Size*sizeof(float)).wait();
    // 数据移动
    q.memcpy(d_b,r_b,b_Size*sizeof(float)).wait();
    // 数据移动
    q.memcpy(d_x,r_x,x_Size*sizeof(float)).wait();
    // 数据移动
    q.memcpy(d_nums,r_nums,nums_Size*sizeof(int)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_A_accessor = info_partition_A_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_b_accessor = info_partition_b_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_x_accessor = info_partition_x_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_x_new_accessor = info_partition_x_new_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_nums_accessor = info_partition_nums_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx1_=(item_id+(0))%idx1.split_size;
            // 嵌入计算
			
            jacobi(d_A+(idx1_*SplitLength[0][0]),d_b+(idx1_*SplitLength[1][0]),d_x,d_x_new+(idx1_*SplitLength[3][0]),d_nums+(idx1_*SplitLength[4][0]),info_partition_A_accessor,info_partition_b_accessor,info_partition_x_accessor,info_partition_x_new_accessor,info_partition_nums_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_x_new+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_x_new[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_x_new,reduction_x_new, Reduction_Size*sizeof(float)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_x_new, d_x_new, x_new_Size*sizeof(float)).wait();
    x_new_tool.UpdateData(r_x_new,x_new);

    // 内存释放
    
    sycl::free(d_A, q);
    sycl::free(d_b, q);
    sycl::free(d_x, q);
    sycl::free(d_x_new, q);
    sycl::free(d_nums, q);
}

int main() {


    // 初始化系数矩阵 A 和向量 b
    std::vector<float> mat_A(N * N, 0.0f);
    std::vector<float> vec_b(N, 0.0f);
    std::vector<float> vec_x(N, 0.0f);     // 初始解
    std::vector<float> vec_x_new(N, 0.0f); // 更新后的解

    // 自动初始化 A 和 b
    for (int i = 0; i < N; ++i) {
        mat_A[i * N + i] = 4.0f; // 对角线元素，确保对角占优

        if (i > 0) {
            mat_A[i * N + i - 1] = -1.0f; // 下三角元素
        }
        if (i < N - 1) {
            mat_A[i * N + i + 1] = -1.0f; // 上三角元素
        }

        vec_b[i] = 1.0f; // 初始化向量 b，可根据需要修改
    }

    // std::vector<int> A_shape = {100, 100};
    // std::vector<int> b_shape = {100};
    // std::vector<int> x_shape = {100};
    // std::vector<int> x_new_shape = {100};
    dacpp::Tensor<float, 2> A({100, 100}, mat_A);
    dacpp::Tensor<float, 1> b(vec_b);
    dacpp::Tensor<float, 1> x(vec_x);
    dacpp::Tensor<float, 1> x_new(vec_x_new);
    
    bool converged = false;
    int iter = 0;
    std::vector<int> nums(100);
    // 使用 std::iota 填充 nums，值从 0 开始
    for(int i = 0;i < N;  i++){
        nums[i] = i;
    }
    //std::vector<int> nums_shape = {100};
    dacpp::Tensor<int, 1> tensor_nums(nums);
    float* data = new float[1 * 100];
    float* data2 = new float[1 * 100];

    while (!converged && iter < max_iter) {
        jacobiShell_jacobi(A, b, x, x_new, tensor_nums);
        
        x.tensor2Array(data);
        x_new.tensor2Array(data2);

        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(data2[i] - data[i]));
        }

        if (max_error < tolerance) {
            converged = true;
        }

        // 更新 x
        x=x_new;

        ++iter;
    }


    // 输出结果
    std::cout << "迭代次数: " << iter << std::endl;
    std::cout << "解向量 x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << data2[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
