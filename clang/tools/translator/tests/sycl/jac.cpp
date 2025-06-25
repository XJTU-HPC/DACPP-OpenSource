// #include <iostream>
// #include <vector>
// #include <cmath>
// #include "ReconTensor.h"
// // 定义矩阵大小
// const int N = 10000; // 可以修改 N 的值来改变矩阵大小
// const int max_iter = 1;
// const float tolerance = 1e-8;
// namespace dacpp {
//     typedef std::vector<std::any> list;
// }


// #include <sycl/sycl.hpp>
// #include "DataReconstructor.h"
// #include "ParameterGeneration.h"

// using namespace sycl;

// void jacobi(float* a,float* b,float* x,float* x_new,int* num,sycl::accessor<int, 1, sycl::access::mode::read_write> info_a_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_b_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_x_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_x_new_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_num_acc) 
// {
//     float sigma = 0;
//     for (int i = 0; i < N; ++i) {
//         if (i != num[0]) {
//             sigma += a[i] * x[i];
//         }
//     }
//     x_new[0] = (b[0] - sigma) / a[num[0]];
// }


// // 生成函数调用
// void jacobiShell_jacobi(const dacpp::Matrix<float> & A, const dacpp::Vector<float> & b, const dacpp::Vector<float> & x, dacpp::Vector<float> & x_new, const dacpp::Vector<int> & nums) { 
//     // 设备选择
//     auto selector = default_selector_v;
//     //queue q(selector);
//     queue q(selector,{property::queue::enable_profiling()});
//     double total_generation = 0, total_reconstruct = 0,total_memcpy = 0, total_kernel = 0, total_reduction = 0;
//     event e;

//     auto start1 = std::chrono::high_resolution_clock::now();
//     //声明参数生成工具
//     ParameterGeneration para_gene_tool;
//     // 算子初始化
    
//     // 数据信息初始化
//     DataInfo info_A;
//     info_A.dim = A.getDim();
//     for(int i = 0; i < info_A.dim; i++) info_A.dimLength.push_back(A.getShape(i));
//     // 数据信息初始化
//     DataInfo info_b;
//     info_b.dim = b.getDim();
//     for(int i = 0; i < info_b.dim; i++) info_b.dimLength.push_back(b.getShape(i));
//     // 数据信息初始化
//     DataInfo info_x;
//     info_x.dim = x.getDim();
//     for(int i = 0; i < info_x.dim; i++) info_x.dimLength.push_back(x.getShape(i));
//     // 数据信息初始化
//     DataInfo info_x_new;
//     info_x_new.dim = x_new.getDim();
//     for(int i = 0; i < info_x_new.dim; i++) info_x_new.dimLength.push_back(x_new.getShape(i));
//     // 数据信息初始化
//     DataInfo info_nums;
//     info_nums.dim = nums.getDim();
//     for(int i = 0; i < info_nums.dim; i++) info_nums.dimLength.push_back(nums.getShape(i));
//     // 降维算子初始化
//     Index idx1 = Index("idx1");
//     idx1.setDimId(0);
//     idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_A));

//     //参数生成
	
//     // 参数生成 提前计算后面需要用到的参数	
	
//     // 算子组初始化
//     Dac_Ops A_Ops;
    
//     idx1.setDimId(0);
//     A_Ops.push_back(idx1);


//     // 算子组初始化
//     Dac_Ops b_Ops;
    
//     idx1.setDimId(0);
//     b_Ops.push_back(idx1);


//     // 算子组初始化
//     Dac_Ops x_Ops;
    

//     // 算子组初始化
//     Dac_Ops x_new_Ops;
    
//     idx1.setDimId(0);
//     x_new_Ops.push_back(idx1);


//     // 算子组初始化
//     Dac_Ops nums_Ops;
    
//     idx1.setDimId(0);
//     nums_Ops.push_back(idx1);


//     // 算子组初始化
//     Dac_Ops In_Ops;
    
//     idx1.setDimId(0);
//     In_Ops.push_back(idx1);


//     // 算子组初始化
//     Dac_Ops Out_Ops;
    
//     idx1.setDimId(0);
//     Out_Ops.push_back(idx1);


//     // 算子组初始化
//     Dac_Ops Reduction_Ops;
    
//     idx1.setDimId(0);
//     Reduction_Ops.push_back(idx1);


	
//     //生成设备内存分配大小
//     int A_Size = para_gene_tool.init_device_memory_size(info_A,A_Ops);

//     //生成设备内存分配大小
//     int b_Size = para_gene_tool.init_device_memory_size(info_b,b_Ops);

//     //生成设备内存分配大小
//     int x_Size = para_gene_tool.init_device_memory_size(info_x,x_Ops);

//     //生成设备内存分配大小
//     int x_new_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_x_new);

//     //生成设备内存分配大小
//     int x_newReduction_Size = para_gene_tool.init_device_memory_size(info_x_new,Reduction_Ops);

//     //生成设备内存分配大小
//     int nums_Size = para_gene_tool.init_device_memory_size(info_nums,nums_Ops);

	
//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(A_Ops,A_Size);

//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(b_Ops,b_Size);

//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(x_Ops,x_Size);

//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(In_Ops,x_new_Size);

//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(nums_Ops,nums_Size);

	
	
//     std::vector<Dac_Ops> ops_s;
	
//     ops_s.push_back(A_Ops);

//     ops_s.push_back(b_Ops);

//     ops_s.push_back(x_Ops);

//     ops_s.push_back(In_Ops);

//     ops_s.push_back(nums_Ops);


// 	// 生成划分长度的二维矩阵
//     int SplitLength[5][1] = {0};
//     para_gene_tool.init_split_length_martix(5,1,&SplitLength[0][0],ops_s);

	
//     // 计算工作项的大小
//     int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
//     // 计算归约中split_size的大小
//     int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
//     // 计算归约中split_length的大小
//     int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);

//     auto end1 = std::chrono::high_resolution_clock::now();
//     total_generation += std::chrono::duration<double>(end1 - start1).count();
//     std::cout << "total_generation:" << total_generation << "s\n";


//     // 设备内存分配
    
//     // 设备内存分配
//     float *d_A=malloc_device<float>(A_Size,q);
//     // 设备内存分配
//     float *d_b=malloc_device<float>(b_Size,q);
//     // 设备内存分配
//     float *d_x=malloc_device<float>(x_Size,q);
//     // 设备内存分配
//     float *d_x_new=malloc_device<float>(x_new_Size,q);
//     // 归约设备内存分配
//     float *reduction_x_new = malloc_device<float>(x_newReduction_Size,q);
//     // 设备内存分配
//     int *d_nums=malloc_device<int>(nums_Size,q);
//     // 数据关联计算
    
//     auto start2 = std::chrono::high_resolution_clock::now(); 
//     // 数据重组
//     auto start21 = std::chrono::high_resolution_clock::now();
//     DataReconstructor<float> A_tool;
//     float* r_A=(float*)malloc(sizeof(float)*A_Size);
//     auto end21 = std::chrono::high_resolution_clock::now();
//     //std::cout << "21:" << std::chrono::duration<double>(end21 - start21).count() << "s\n";
    
//     auto start22 = std::chrono::high_resolution_clock::now();
//     // 数据算子组初始化
//     Dac_Ops A_ops;
    
//     idx1.setDimId(0);
//     A_ops.push_back(idx1);
//     auto end22 = std::chrono::high_resolution_clock::now();
//     //std::cout << "22:" << std::chrono::duration<double>(end22 - start22).count() << "s\n";

//     auto start23 = std::chrono::high_resolution_clock::now();
//     A_tool.init(info_A,A_ops);
//     auto end23 = std::chrono::high_resolution_clock::now();
//     std::cout << "tool.init:" << std::chrono::duration<double>(end23 - start23).count() << "s\n";

//     auto start24 = std::chrono::high_resolution_clock::now();
//     A_tool.Reconstruct(r_A,A);
//     auto end24 = std::chrono::high_resolution_clock::now();
//     std::cout << "tool.Reconstruct:" << std::chrono::duration<double>(end24 - start24).count() << "s\n";

// 	std::vector<int> info_partition_A=para_gene_tool.init_partition_data_shape(info_A,A_ops);
//     sycl::buffer<int> info_partition_A_buffer(info_partition_A.data(), sycl::range<1>(info_partition_A.size()));
//     // 数据重组
//     DataReconstructor<float> b_tool;
//     float* r_b=(float*)malloc(sizeof(float)*b_Size);
    
//     // 数据算子组初始化
//     Dac_Ops b_ops;
    
//     idx1.setDimId(0);
//     b_ops.push_back(idx1);
//     auto start25 = std::chrono::high_resolution_clock::now();
//     b_tool.init(info_b,b_ops);
//     auto end25 = std::chrono::high_resolution_clock::now();
//     std::cout << "tool.init2:" << std::chrono::duration<double>(end25 - start25).count() << "s\n";
//     auto start26 = std::chrono::high_resolution_clock::now();
//     b_tool.Reconstruct(r_b,b);
//     auto end26 = std::chrono::high_resolution_clock::now();
//     std::cout << "tool.Reconstruct2:" << std::chrono::duration<double>(end26 - start26).count() << "s\n";
// 	std::vector<int> info_partition_b=para_gene_tool.init_partition_data_shape(info_b,b_ops);
//     sycl::buffer<int> info_partition_b_buffer(info_partition_b.data(), sycl::range<1>(info_partition_b.size()));
//     // 数据重组
//     DataReconstructor<float> x_tool;
//     float* r_x=(float*)malloc(sizeof(float)*x_Size);
    
//     // 数据算子组初始化
//     Dac_Ops x_ops;
    
//     x_tool.init(info_x,x_ops);
//     x_tool.Reconstruct(r_x,x);
// 	std::vector<int> info_partition_x=para_gene_tool.init_partition_data_shape(info_x,x_ops);
//     sycl::buffer<int> info_partition_x_buffer(info_partition_x.data(), sycl::range<1>(info_partition_x.size()));
//     // 数据重组
//     DataReconstructor<float> x_new_tool;
//     float* r_x_new=(float*)malloc(sizeof(float)*x_new_Size);
    
//     // 数据算子组初始化
//     Dac_Ops x_new_ops;
    
//     idx1.setDimId(0);
//     x_new_ops.push_back(idx1);
//     x_new_tool.init(info_x_new,x_new_ops);
//     x_new_tool.Reconstruct(r_x_new,x_new);
// 	std::vector<int> info_partition_x_new=para_gene_tool.init_partition_data_shape(info_x_new,x_new_ops);
//     sycl::buffer<int> info_partition_x_new_buffer(info_partition_x_new.data(), sycl::range<1>(info_partition_x_new.size()));
//     // 数据重组
//     DataReconstructor<int> nums_tool;
//     int* r_nums=(int*)malloc(sizeof(int)*nums_Size);
    
//     // 数据算子组初始化
//     Dac_Ops nums_ops;
    
//     idx1.setDimId(0);
//     nums_ops.push_back(idx1);
//     nums_tool.init(info_nums,nums_ops);
//     nums_tool.Reconstruct(r_nums,nums);
// 	std::vector<int> info_partition_nums=para_gene_tool.init_partition_data_shape(info_nums,nums_ops);
//     sycl::buffer<int> info_partition_nums_buffer(info_partition_nums.data(), sycl::range<1>(info_partition_nums.size()));
    
//     auto end2 = std::chrono::high_resolution_clock::now();
//     total_reconstruct += std::chrono::duration<double>(end2 - start2).count();
//     std::cout << "total_reconstruct:" << total_reconstruct << "s\n";

//     auto start3 = std::chrono::high_resolution_clock::now();
//     // 设备数据初始化
//     q.memset(d_A,0,A_Size*sizeof(float)).wait();
//     // 数据移动
//     q.memcpy(d_A,r_A,A_Size*sizeof(float)).wait();
//     // 设备数据初始化
//     q.memset(d_b,0,b_Size*sizeof(float)).wait();
//     // 数据移动
//     q.memcpy(d_b,r_b,b_Size*sizeof(float)).wait();
//     // 设备数据初始化
//     q.memset(d_x,0,x_Size*sizeof(float)).wait();
//     // 数据移动
//     q.memcpy(d_x,r_x,x_Size*sizeof(float)).wait();
//     // 设备数据初始化
//     q.memset(d_x_new,0,x_new_Size*sizeof(float)).wait();
//     // 设备数据初始化
//     q.memset(d_nums,0,nums_Size*sizeof(int)).wait();
//     // 数据移动
//     q.memcpy(d_nums,r_nums,nums_Size*sizeof(int)).wait();

//     auto end3 = std::chrono::high_resolution_clock::now();
//     total_memcpy += std::chrono::duration<double>(end3 - start3).count();
//     std::cout << "total_memcpy:" << total_memcpy << "s\n";
	
//     sycl::device device = q.get_device();
//     int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
// 	//工作项划分
//     int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
//     sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
//     sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
//     //队列提交命令组
//     e = q.submit([&](handler &h) {
//         // 访问器初始化
        
//         auto info_partition_A_accessor = info_partition_A_buffer.get_access<sycl::access::mode::read_write>(h);
//         auto info_partition_b_accessor = info_partition_b_buffer.get_access<sycl::access::mode::read_write>(h);
//         auto info_partition_x_accessor = info_partition_x_buffer.get_access<sycl::access::mode::read_write>(h);
//         auto info_partition_x_new_accessor = info_partition_x_new_buffer.get_access<sycl::access::mode::read_write>(h);
//         auto info_partition_nums_accessor = info_partition_nums_buffer.get_access<sycl::access::mode::read_write>(h);
//         h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
//             const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
//             if(item_id >= Item_Size)
//                 return;
//             // 索引初始化
			
//             const auto idx1_=(item_id+(0))%idx1.split_size;
//             // 嵌入计算
			
//             jacobi(d_A+(idx1_*SplitLength[0][0]),d_b+(idx1_*SplitLength[1][0]),d_x,d_x_new+(idx1_*SplitLength[3][0]),d_nums+(idx1_*SplitLength[4][0]),info_partition_A_accessor,info_partition_b_accessor,info_partition_x_accessor,info_partition_x_new_accessor,info_partition_nums_accessor);
//         });
//     });
//     e.wait();
//     auto start_kernel = e.get_profiling_info<info::event_profiling::command_start>();
//     auto end_kernel = e.get_profiling_info<info::event_profiling::command_end>();
//     total_kernel += (end_kernel - start_kernel) / 1e9;
//     std::cout << "total_kernel:" << total_kernel << "s\n";
    

	
//     // 归约
//     if(Reduction_Split_Size > 1)
//     {
//         for(int i=0;i<x_newReduction_Size;i++) {
//             q.submit([&](handler &h) {
//     	        h.parallel_for(
//                 range<1>(Reduction_Split_Size),
//                 reduction(reduction_x_new+i, 
//                 sycl::plus<>(),
//                 property::reduction::initialize_to_identity()),
//                 [=](id<1> idx,auto &reducer) {
//                     reducer.combine(d_x_new[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
//      	        });
//          }).wait();
//         }
//         q.memcpy(d_x_new,reduction_x_new, x_newReduction_Size*sizeof(float)).wait();
//     }


	
//     // 归并结果返回
//     q.memcpy(r_x_new, d_x_new, x_new_Size*sizeof(float)).wait();
//     x_new_tool.UpdateData(r_x_new,x_new);

//     // 内存释放
    
//     sycl::free(d_A, q);
//     sycl::free(d_b, q);
//     sycl::free(d_x, q);
//     sycl::free(d_x_new, q);
//     sycl::free(d_nums, q);
// }

// int main() {


//     // 初始化系数矩阵 A 和向量 b
//     std::vector<float> mat_A(N * N, 0.0f);
//     std::vector<float> vec_b(N, 0.0f);
//     std::vector<float> vec_x(N, 0.0f);     // 初始解
//     std::vector<float> vec_x_new(N, 0.0f); // 更新后的解

//     // 自动初始化 A 和 b
//     for (int i = 0; i < N; ++i) {
//         mat_A[i * N + i] = 4.0f; // 对角线元素，确保对角占优

//         if (i > 0) {
//             mat_A[i * N + i - 1] = -1.0f; // 下三角元素
//         }
//         if (i < N - 1) {
//             mat_A[i * N + i + 1] = -1.0f; // 上三角元素
//         }

//         vec_b[i] = 1.0f; // 初始化向量 b，可根据需要修改
//     }

//     // std::vector<int> A_shape = {100, 100};
//     // std::vector<int> b_shape = {100};
//     // std::vector<int> x_shape = {100};
//     // std::vector<int> x_new_shape = {100};
//     dacpp::Matrix<float> A({N, N}, mat_A);
//     dacpp::Vector<float> b(vec_b);
//     dacpp::Vector<float> x(vec_x);
//     dacpp::Vector<float> x_new(vec_x_new);
    
//     bool converged = false;
//     int iter = 0;
//     std::vector<int> nums(N);
//     // 使用 std::iota 填充 nums，值从 0 开始
//     for(int i = 0;i < N;  i++){
//         nums[i] = i;
//     }
//     //std::vector<int> nums_shape = {100};
//     dacpp::Vector<int> tensor_nums(nums);
//     float* data = new float[1 * N];
//     float* data2 = new float[1 * N];

//     while (!converged && iter < max_iter) {
//         jacobiShell_jacobi(A, b, x, x_new, tensor_nums);
        
//         x.tensor2Array(data);
//         x_new.tensor2Array(data2);

//         float max_error = 0.0f;
//         for (int i = 0; i < N; ++i) {
//             max_error = std::max(max_error, std::fabs(data2[i] - data[i]));
//         }

//         if (max_error < tolerance) {
//             converged = true;
//         }

//         // 更新 x
//         x=x_new;

//         ++iter;
//     }


//     // 输出结果
//     std::cout << "迭代次数: " << iter << std::endl;
//     std::cout << "解向量 x:" << std::endl;
//     // for (int i = 0; i < N; ++i) {
//     //     std::cout << data2[i] << " ";
//     // }
//     // std::cout << std::endl;

//     return 0;
// }

//测试修改后的usm模板能否运行

// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <cstdlib>
// #include "ReconTensor.h"

// namespace dacpp {
//     typedef std::vector<std::any> list;
// }

// const double dt = 0.1;       // 时间步长
// const double T = 5.0;       // 总时间
// const size_t numIsotopes = 10; // 设定大量同位素（例如，10000个）




// // 计算每种同位素在时间 t 的数量
// #include <sycl/sycl.hpp>
// #include "DataReconstructor1.h"
// #include "ParameterGeneration.h"

// using namespace sycl;

// void decay(double* N0s,double* lambdas,double* local_A,double* t,sycl::accessor<int, 1, sycl::access::mode::read_write> info_N0s_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_lambdas_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_local_A_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_t_acc) 
// {
//     local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);
// }


// // 生成函数调用
// void DECAY_decay(const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) { 
//     // 设备选择
//     auto selector = default_selector_v;
//     queue q(selector);
//     //声明参数生成工具
//     ParameterGeneration para_gene_tool;
//     // 算子初始化
    
//     // 数据信息初始化
//     DataInfo info_N0s;
//     info_N0s.dim = N0s.getDim();
//     for(int i = 0; i < info_N0s.dim; i++) info_N0s.dimLength.push_back(N0s.getShape(i));
	
//     // 数据信息初始化
//     DataInfo info_lambdas;
//     info_lambdas.dim = lambdas.getDim();
//     for(int i = 0; i < info_lambdas.dim; i++) info_lambdas.dimLength.push_back(lambdas.getShape(i));
	
//     // 数据信息初始化
//     DataInfo info_local_A;
//     info_local_A.dim = local_A.getDim();
//     for(int i = 0; i < info_local_A.dim; i++) info_local_A.dimLength.push_back(local_A.getShape(i));
	
//     // 数据信息初始化
//     DataInfo info_t;
//     info_t.dim = t.getDim();
//     for(int i = 0; i < info_t.dim; i++) info_t.dimLength.push_back(t.getShape(i));
	
//     // 降维算子初始化
//     Index i = Index("i");
//     i.setDimId(0);
//     i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_N0s));

//     //参数生成
	
//     // 参数生成 提前计算后面需要用到的参数	
	
//     // 算子组初始化
//     Dac_Ops N0s_Ops;
    
//     i.setDimId(0);
//     N0s_Ops.push_back(i);


//     // 算子组初始化
//     Dac_Ops lambdas_Ops;
    
//     i.setDimId(0);
//     lambdas_Ops.push_back(i);


//     // 算子组初始化
//     Dac_Ops local_A_Ops;
    
//     i.setDimId(0);
//     local_A_Ops.push_back(i);


//     // 算子组初始化
//     Dac_Ops t_Ops;
    

//     // 算子组初始化
//     Dac_Ops In_Ops;
    
//     i.setDimId(0);
//     In_Ops.push_back(i);


//     // 算子组初始化
//     Dac_Ops Out_Ops;
    
//     i.setDimId(0);
//     Out_Ops.push_back(i);


//     // 算子组初始化
//     Dac_Ops Reduction_Ops;
    
//     i.setDimId(0);
//     Reduction_Ops.push_back(i);


	
//     //生成设备内存分配大小
//     int N0s_Size = para_gene_tool.init_device_memory_size(info_N0s,N0s_Ops);

//     //生成设备内存分配大小
//     int lambdas_Size = para_gene_tool.init_device_memory_size(info_lambdas,lambdas_Ops);

//     //生成设备内存分配大小
//     int local_A_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_local_A);

//     //生成设备内存分配大小
//     int local_AReduction_Size = para_gene_tool.init_device_memory_size(info_local_A,Reduction_Ops);

//     //生成设备内存分配大小
//     int t_Size = para_gene_tool.init_device_memory_size(info_t,t_Ops);

	
//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(N0s_Ops,N0s_Size);

//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(lambdas_Ops,lambdas_Size);

//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(In_Ops,local_A_Size);

//     // 计算算子组里面的算子的划分长度
//     para_gene_tool.init_op_split_length(t_Ops,t_Size);

	
	
//     std::vector<Dac_Ops> ops_s;
	
//     ops_s.push_back(N0s_Ops);

//     ops_s.push_back(lambdas_Ops);

//     ops_s.push_back(In_Ops);

//     ops_s.push_back(t_Ops);


// 	// 生成划分长度的二维矩阵
//     int SplitLength[4][1] = {0};
//     para_gene_tool.init_split_length_martix(4,1,&SplitLength[0][0],ops_s);

	
//     // 计算工作项的大小
//     int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
//     // 计算归约中split_size的大小
//     int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
//     // 计算归约中split_length的大小
//     int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


//     // 设备内存分配
    
//     // 归约设备内存分配
//     double *reduction_local_A = malloc_device<double>(local_AReduction_Size,q);
//     // 数据关联计算
    
    
//     // 数据重组
//     DataReconstructor<double> N0s_tool;
//     // double* r_N0s=(double*)malloc(sizeof(double)*N0s_Size);
    
//     // 数据算子组初始化
//     Dac_Ops N0s_ops;
    
//     i.setDimId(0);
//     N0s_ops.push_back(i);
//     N0s_tool.init(info_N0s,N0s_ops);

//     double recon_h_N0s[N0s_Size];
// 	double *recon_d_N0s=malloc_device<double>(N0s_Size,q);
// 	double *d_N0s=malloc_device<double>(N0s_Size,q);
	

//     N0s.tensor2Array(recon_h_N0s);
//     q.memcpy(recon_d_N0s, recon_h_N0s, N0s_Size*sizeof(double)).wait();
//     N0s_tool.Reconstruct(d_N0s, recon_d_N0s,q);

// 	sycl::free(recon_d_N0s, q);
// 	std::vector<int> info_partition_N0s=para_gene_tool.init_partition_data_shape(info_N0s,N0s_ops);
//     sycl::buffer<int> info_partition_N0s_buffer(info_partition_N0s.data(), sycl::range<1>(info_partition_N0s.size()));
//     // 数据重组
//     DataReconstructor<double> lambdas_tool;
//     // double* r_lambdas=(double*)malloc(sizeof(double)*lambdas_Size);
    
//     // 数据算子组初始化
//     Dac_Ops lambdas_ops;
    
//     i.setDimId(0);
//     lambdas_ops.push_back(i);
//     lambdas_tool.init(info_lambdas,lambdas_ops);

//     double recon_h_lambdas[lambdas_Size];
// 	double *recon_d_lambdas=malloc_device<double>(lambdas_Size,q);
// 	double *d_lambdas=malloc_device<double>(lambdas_Size,q);
	

//     lambdas.tensor2Array(recon_h_lambdas);
//     q.memcpy(recon_d_lambdas, recon_h_lambdas, lambdas_Size*sizeof(double)).wait();
//     lambdas_tool.Reconstruct(d_lambdas, recon_d_lambdas,q);

// 	sycl::free(recon_d_lambdas, q);
// 	std::vector<int> info_partition_lambdas=para_gene_tool.init_partition_data_shape(info_lambdas,lambdas_ops);
//     sycl::buffer<int> info_partition_lambdas_buffer(info_partition_lambdas.data(), sycl::range<1>(info_partition_lambdas.size()));
// 	// 数据重组
// 	DataReconstructor<double> local_A_tool;
	
//     // 数据算子组初始化
//     Dac_Ops local_A_ops;
    
//     i.setDimId(0);
//     local_A_ops.push_back(i);
// 	local_A_tool.init(info_local_A,local_A_ops);

// 	double* r_local_A=(double*)malloc(sizeof(double)*local_A_Size);
// 	double *d_local_A=malloc_device<double>(local_A_Size,q);
// 	q.memset(d_local_A, 0, local_A_Size * sizeof(double)).wait();
	
// 	std::vector<int> info_partition_local_A=para_gene_tool.init_partition_data_shape(info_local_A,local_A_ops);
// 	sycl::buffer<int> info_partition_local_A_buffer(info_partition_local_A.data(), sycl::range<1>(info_partition_local_A.size()));
//     // 数据重组
//     DataReconstructor<double> t_tool;
//     // double* r_t=(double*)malloc(sizeof(double)*t_Size);
    
//     // 数据算子组初始化
//     Dac_Ops t_ops;
    
//     t_tool.init(info_t,t_ops);

//     double recon_h_t[t_Size];
// 	double *recon_d_t=malloc_device<double>(t_Size,q);
// 	double *d_t=malloc_device<double>(t_Size,q);
	

//     t.tensor2Array(recon_h_t);
//     q.memcpy(recon_d_t, recon_h_t, t_Size*sizeof(double)).wait();
//     t_tool.Reconstruct(d_t, recon_d_t,q);

// 	sycl::free(recon_d_t, q);
// 	std::vector<int> info_partition_t=para_gene_tool.init_partition_data_shape(info_t,t_ops);
//     sycl::buffer<int> info_partition_t_buffer(info_partition_t.data(), sycl::range<1>(info_partition_t.size()));
    
	
//     sycl::device device = q.get_device();
//     int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
// 	//工作项划分
//     int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
//     sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
//     sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
//     //队列提交命令组
//     q.submit([&](handler &h) {
//         // 访问器初始化
        
//         auto info_partition_N0s_accessor = info_partition_N0s_buffer.get_access<sycl::access::mode::read_write>(h);

//         auto info_partition_lambdas_accessor = info_partition_lambdas_buffer.get_access<sycl::access::mode::read_write>(h);

//         auto info_partition_local_A_accessor = info_partition_local_A_buffer.get_access<sycl::access::mode::read_write>(h);

//         auto info_partition_t_accessor = info_partition_t_buffer.get_access<sycl::access::mode::read_write>(h);

//         h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
//             const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
//             if(item_id >= Item_Size)
//                 return;
//             // 索引初始化
			
//             const auto i_=(item_id+(0))%i.split_size;
//             // 嵌入计算
			
//             decay(d_N0s+(i_*SplitLength[0][0]),d_lambdas+(i_*SplitLength[1][0]),d_local_A+(i_*SplitLength[2][0]),d_t,info_partition_N0s_accessor,info_partition_lambdas_accessor,info_partition_local_A_accessor,info_partition_t_accessor);
//         });
//     }).wait();
    

	
//     // 归约
//     if(Reduction_Split_Size > 1)
//     {
//         for(int i=0;i<local_AReduction_Size;i++) {
//             q.submit([&](handler &h) {
//     	        h.parallel_for(
//                 range<1>(Reduction_Split_Size),
//                 reduction(reduction_local_A+i, 
//                 sycl::plus<>(),
//                 property::reduction::initialize_to_identity()),
//                 [=](id<1> idx,auto &reducer) {
//                     reducer.combine(d_local_A[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
//      	        });
//          }).wait();
//         }
//         q.memcpy(d_local_A,reduction_local_A, local_AReduction_Size*sizeof(double)).wait();
//     }


//     // 归并结果返回
// 	double *d_myTensor2=malloc_device<double>(local_A_Size,q);
//     local_A_tool.UpdateData(d_local_A,d_myTensor2,q);
// 	double* res = (double*)malloc(local_A_Size*sizeof(double));
// 	q.memcpy(res,d_myTensor2, local_A_Size*sizeof(double)).wait();
// 	local_A.array2Tensor(res);
	

//     // 内存释放
    
//     sycl::free(d_N0s, q);
//     sycl::free(d_lambdas, q);
//     sycl::free(d_local_A, q);
//     sycl::free(d_t, q);
// }

// void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
//     size_t numIsotopes = lambdas.size(); // 同位素的数量
//     std::vector<double> A(T/dt*numIsotopes, 0.0);  // 存储每个同位素在不同时间点的数量
//     std::vector<double> time;  // 时间序列
//     std::vector<double> t;
//     t.push_back(static_cast<double>(0));

//     // 串行计算每个同位素的衰变过程
//     std::vector<double> local_A(numIsotopes, 0.0);
//     dacpp::Vector<double> local_A_tensor(local_A);
//     dacpp::Vector<double> N0s_tensor(N0s);
//     dacpp::Vector<double> lambdas_tensor(lambdas);
//     dacpp::Vector<double> t_tensor(t);
//     dacpp::Matrix<double> A_tensor({static_cast<int>(T/dt), static_cast<int>(numIsotopes)}, A);
    

//     while(t_tensor[0] <= T){  
//         DECAY_decay(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
//         A_tensor[10*t_tensor[0]] = local_A_tensor;
//         t_tensor[0] += dt;
//     }
//     A_tensor.print();
// }

// int main() {
    

//     // 随机生成衰变常数和初始数量
//     std::vector<double> lambdas(numIsotopes);
//     std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

//     // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
//     for (size_t i = 0; i < numIsotopes; ++i) {
//         lambdas[i] = 0.01 + 0.01*i;  // lambda 范围 [0.01, 0.2]
//     }


//     //size_t numOutputSteps = 10; // 输出的时间步数量

//     calculateDecay(lambdas, N0s, dt, T);

//     return 0;
// }


#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const double dt = 0.1;       // 时间步长
const double T = 5.0;       // 总时间
const size_t numIsotopes = 10; // 设定大量同位素（例如，10000个）




// 计算每种同位素在时间 t 的数量
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void decay(double* N0s,double* lambdas,double* local_A,double* t,sycl::accessor<int, 1, sycl::access::mode::read_write> info_N0s_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_lambdas_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_local_A_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_t_acc) 
{
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);
}


// 生成函数调用
void DECAY_decay(const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_N0s;
    info_N0s.dim = N0s.getDim();
    for(int i = 0; i < info_N0s.dim; i++) info_N0s.dimLength.push_back(N0s.getShape(i));
	
    // 数据信息初始化
    DataInfo info_lambdas;
    info_lambdas.dim = lambdas.getDim();
    for(int i = 0; i < info_lambdas.dim; i++) info_lambdas.dimLength.push_back(lambdas.getShape(i));
	
    // 数据信息初始化
    DataInfo info_local_A;
    info_local_A.dim = local_A.getDim();
    for(int i = 0; i < info_local_A.dim; i++) info_local_A.dimLength.push_back(local_A.getShape(i));
	
    // 数据信息初始化
    DataInfo info_t;
    info_t.dim = t.getDim();
    for(int i = 0; i < info_t.dim; i++) info_t.dimLength.push_back(t.getShape(i));
	
    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_N0s));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops N0s_Ops;
    
    i.setDimId(0);
    N0s_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops lambdas_Ops;
    
    i.setDimId(0);
    lambdas_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops local_A_Ops;
    
    i.setDimId(0);
    local_A_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops t_Ops;
    

    // 算子组初始化
    Dac_Ops In_Ops;
    
    i.setDimId(0);
    In_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    i.setDimId(0);
    Out_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    i.setDimId(0);
    Reduction_Ops.push_back(i);


	
    //生成设备内存分配大小
    int N0s_Size = para_gene_tool.init_device_memory_size(info_N0s,N0s_Ops);

    //生成设备内存分配大小
    int lambdas_Size = para_gene_tool.init_device_memory_size(info_lambdas,lambdas_Ops);

    //生成设备内存分配大小
    int local_A_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_local_A);

    //生成设备内存分配大小
    int local_AReduction_Size = para_gene_tool.init_device_memory_size(info_local_A,Reduction_Ops);

    //生成设备内存分配大小
    int t_Size = para_gene_tool.init_device_memory_size(info_t,t_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(N0s_Ops,N0s_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(lambdas_Ops,lambdas_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,local_A_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(t_Ops,t_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(N0s_Ops);

    ops_s.push_back(lambdas_Ops);

    ops_s.push_back(In_Ops);

    ops_s.push_back(t_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[4][1] = {0};
    para_gene_tool.init_split_length_martix(4,1,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // 归约设备内存分配
    double *reduction_local_A = malloc_device<double>(local_AReduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<double> N0s_tool;
    // double* r_N0s=(double*)malloc(sizeof(double)*N0s_Size);
    
    // 数据算子组初始化
    Dac_Ops N0s_ops;
    
    i.setDimId(0);
    N0s_ops.push_back(i);
    N0s_tool.init(info_N0s,N0s_ops);

    double recon_h_N0s[N0s_Size];
	double *recon_d_N0s=malloc_device<double>(N0s_Size,q);
	double *d_N0s=malloc_device<double>(N0s_Size,q);
	

    N0s.tensor2Array(recon_h_N0s);
    q.memcpy(recon_d_N0s, recon_h_N0s, N0s_Size*sizeof(double)).wait();
    N0s_tool.Reconstruct(d_N0s, recon_d_N0s,q);

	sycl::free(recon_d_N0s, q);
	std::vector<int> info_partition_N0s=para_gene_tool.init_partition_data_shape(info_N0s,N0s_ops);
    sycl::buffer<int> info_partition_N0s_buffer(info_partition_N0s.data(), sycl::range<1>(info_partition_N0s.size()));
    // 数据重组
    DataReconstructor<double> lambdas_tool;
    // double* r_lambdas=(double*)malloc(sizeof(double)*lambdas_Size);
    
    // 数据算子组初始化
    Dac_Ops lambdas_ops;
    
    i.setDimId(0);
    lambdas_ops.push_back(i);
    lambdas_tool.init(info_lambdas,lambdas_ops);

    double recon_h_lambdas[lambdas_Size];
	double *recon_d_lambdas=malloc_device<double>(lambdas_Size,q);
	double *d_lambdas=malloc_device<double>(lambdas_Size,q);
	

    lambdas.tensor2Array(recon_h_lambdas);
    q.memcpy(recon_d_lambdas, recon_h_lambdas, lambdas_Size*sizeof(double)).wait();
    lambdas_tool.Reconstruct(d_lambdas, recon_d_lambdas,q);

	sycl::free(recon_d_lambdas, q);
	std::vector<int> info_partition_lambdas=para_gene_tool.init_partition_data_shape(info_lambdas,lambdas_ops);
    sycl::buffer<int> info_partition_lambdas_buffer(info_partition_lambdas.data(), sycl::range<1>(info_partition_lambdas.size()));
	// 数据重组
	DataReconstructor<double> local_A_tool;
	
    // 数据算子组初始化
    Dac_Ops local_A_ops;
    
    i.setDimId(0);
    local_A_ops.push_back(i);
	local_A_tool.init(info_local_A,local_A_ops);

	double* r_local_A=(double*)malloc(sizeof(double)*local_A_Size);
	double *d_local_A=malloc_device<double>(local_A_Size,q);
	q.memset(d_local_A, 0, local_A_Size * sizeof(double)).wait();
	
	std::vector<int> info_partition_local_A=para_gene_tool.init_partition_data_shape(info_local_A,local_A_ops);
	sycl::buffer<int> info_partition_local_A_buffer(info_partition_local_A.data(), sycl::range<1>(info_partition_local_A.size()));
    // 数据重组
    DataReconstructor<double> t_tool;
    // double* r_t=(double*)malloc(sizeof(double)*t_Size);
    
    // 数据算子组初始化
    Dac_Ops t_ops;
    
    t_tool.init(info_t,t_ops);

    double recon_h_t[t_Size];
	double *recon_d_t=malloc_device<double>(t_Size,q);
	double *d_t=malloc_device<double>(t_Size,q);
	

    t.tensor2Array(recon_h_t);
    q.memcpy(recon_d_t, recon_h_t, t_Size*sizeof(double)).wait();
    t_tool.Reconstruct(d_t, recon_d_t,q);

	sycl::free(recon_d_t, q);
	std::vector<int> info_partition_t=para_gene_tool.init_partition_data_shape(info_t,t_ops);
    sycl::buffer<int> info_partition_t_buffer(info_partition_t.data(), sycl::range<1>(info_partition_t.size()));
    
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_N0s_accessor = info_partition_N0s_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_lambdas_accessor = info_partition_lambdas_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_local_A_accessor = info_partition_local_A_buffer.get_access<sycl::access::mode::read_write>(h);

        auto info_partition_t_accessor = info_partition_t_buffer.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 嵌入计算
			
            decay(d_N0s+(i_*SplitLength[0][0]),d_lambdas+(i_*SplitLength[1][0]),d_local_A+(i_*SplitLength[2][0]),d_t,info_partition_N0s_accessor,info_partition_lambdas_accessor,info_partition_local_A_accessor,info_partition_t_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<local_AReduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_local_A+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_local_A[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_local_A,reduction_local_A, local_AReduction_Size*sizeof(double)).wait();
    }


    // 归并结果返回
	double *d_myTensor2=malloc_device<double>(local_A_Size,q);
    local_A_tool.UpdateData(d_local_A,d_myTensor2,q);
	double* res = (double*)malloc(local_A_Size*sizeof(double));
	q.memcpy(res,d_myTensor2, local_A_Size*sizeof(double)).wait();
	local_A.array2Tensor(res);
	

    // 内存释放
    
    sycl::free(d_N0s, q);
    sycl::free(d_lambdas, q);
    sycl::free(d_local_A, q);
    sycl::free(d_t, q);
}

void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size(); // 同位素的数量
    std::vector<double> A(T/dt*numIsotopes, 0.0);  // 存储每个同位素在不同时间点的数量
    std::vector<double> time;  // 时间序列
    std::vector<double> t;
    t.push_back(static_cast<double>(0));

    // 串行计算每个同位素的衰变过程
    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({static_cast<int>(T/dt), static_cast<int>(numIsotopes)}, A);
    

    while(t_tensor[0] <= T){  
        DECAY_decay(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    A_tensor.print();
}

int main() {
    

    // 随机生成衰变常数和初始数量
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 随机初始化衰变常数（例如，lambda 在 0.01 到 0.2 之间）
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01*i;  // lambda 范围 [0.01, 0.2]
    }


    //size_t numOutputSteps = 10; // 输出的时间步数量

    calculateDecay(lambdas, N0s, dt, T);

    return 0;
}