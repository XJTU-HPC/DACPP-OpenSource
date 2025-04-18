#include <iostream>
#include <vector>
#include <complex>
#include "ReconTensor.h"
#include <cmath>

namespace dacpp {
    typedef std::vector<std::any> list;
}
using namespace std;

// 全局变量定义
const int row_count = 8, col_count = 8, max_iterations = 1000;
vector<complex<float>> complex_points;  // 一维向量表示复数点
vector<int> mandelbrot_flags;           // 一维数组表示是否属于 Mandelbrot 集
int total_points = 0;                   // 总点数
int mandelbrot_count = 0;               // 属于 Mandelbrot 集的点数量

// 初始化复数点向量
void InitializeComplexPoints() {
    total_points = row_count * col_count;  // 总点数
    complex_points.resize(total_points);

    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;  // 一维向量索引
            float real = -1.5f + (i * (2.0f / row_count));  // 将行索引映射到实部
            float imag = -1.0f + (j * (2.0f / col_count));  // 将列索引映射到虚部
            complex_points[index] = complex<float>(real, imag);
        }
    }
}



void MANDEL_mandel(const dacpp::Tensor<complex<float>, 1> & complex_points, dacpp::Tensor<int, 1> & mandelbrot_flags);


// 计算 Mandelbrot 集
void ComputeMandelbrot() {
    mandelbrot_flags.resize(total_points, 0);  // 初始化一维数组为 0

    dacpp::Tensor<complex<float>, 1> complex_points_tensor(complex_points);
    dacpp::Tensor<int, 1> mandelbrot_flags_tensor(mandelbrot_flags);


    MANDEL_mandel(complex_points_tensor, mandelbrot_flags_tensor);

    // 统计数组中 1 的个数
    mandelbrot_count = 0;
    for (int i = 0; i < total_points; i++){
        if (mandelbrot_flags_tensor[i] == 1) mandelbrot_count++;
    }
}

// 打印统计信息
void PrintStats() {
    cout << "Mandelbrot Set Statistics:\n";
    cout << "Total points: " << total_points << "\n";
    cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
}

#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void mandel(complex<float>* complex_points,int* mandelbrot_flags,sycl::accessor<int, 1, sycl::access::mode::read_write> info_complex_points_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_mandelbrot_flags_acc) 
{
    const complex<float> &c = complex_points[0];
    complex<float> z = 0;
    int iterations = 0;
    for (int i = 0; i < max_iterations; ++i) {
        if (std::sqrt(z.real() * z.real() + z.imag() * z.imag()) > 2.F) {
            iterations = i;
            break;
        }
        z = z * z + c;
        iterations = max_iterations;
    }
    if (iterations == max_iterations) {
        mandelbrot_flags[0] = 1;
    }
}


// 生成函数调用
void MANDEL_mandel(const dacpp::Tensor<complex<float>, 1> & complex_points, dacpp::Tensor<int, 1> & mandelbrot_flags) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_complex_points;
    info_complex_points.dim = complex_points.getDim();
    for(int i = 0; i < info_complex_points.dim; i++) info_complex_points.dimLength.push_back(complex_points.getShape(i));
    // 数据信息初始化
    DataInfo info_mandelbrot_flags;
    info_mandelbrot_flags.dim = mandelbrot_flags.getDim();
    for(int i = 0; i < info_mandelbrot_flags.dim; i++) info_mandelbrot_flags.dimLength.push_back(mandelbrot_flags.getShape(i));
    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_complex_points));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops complex_points_Ops;
    
    i.setDimId(0);
    complex_points_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops mandelbrot_flags_Ops;
    
    i.setDimId(0);
    mandelbrot_flags_Ops.push_back(i);


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
    int complex_points_Size = para_gene_tool.init_device_memory_size(info_complex_points,complex_points_Ops);

    //生成设备内存分配大小
    int mandelbrot_flags_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_mandelbrot_flags);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_mandelbrot_flags,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(complex_points_Ops,complex_points_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,mandelbrot_flags_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(complex_points_Ops);

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
    
    // 设备内存分配
    complex<float> *d_complex_points=malloc_device<complex<float>>(complex_points_Size,q);
    // 设备内存分配
    int *d_mandelbrot_flags=malloc_device<int>(mandelbrot_flags_Size,q);
    // 归约设备内存分配
    int *reduction_mandelbrot_flags = malloc_device<int>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<complex<float>> complex_points_tool;
    complex<float>* r_complex_points=(complex<float>*)malloc(sizeof(complex<float>)*complex_points_Size);
    
    // 数据算子组初始化
    Dac_Ops complex_points_ops;
    
    i.setDimId(0);
    i.setSplitLength(8);
    complex_points_ops.push_back(i);
    complex_points_tool.init(info_complex_points,complex_points_ops);
    complex_points_tool.Reconstruct(r_complex_points,complex_points);
	std::vector<int> info_partition_complex_points=para_gene_tool.init_partition_data_shape(info_complex_points,complex_points_ops);
    sycl::buffer<int> info_partition_complex_points_buffer(info_partition_complex_points.data(), sycl::range<1>(info_partition_complex_points.size()));
    // 数据重组
    DataReconstructor<int> mandelbrot_flags_tool;
    int* r_mandelbrot_flags=(int*)malloc(sizeof(int)*mandelbrot_flags_Size);
    
    // 数据算子组初始化
    Dac_Ops mandelbrot_flags_ops;
    
    i.setDimId(0);
    i.setSplitLength(8);
    mandelbrot_flags_ops.push_back(i);
    mandelbrot_flags_tool.init(info_mandelbrot_flags,mandelbrot_flags_ops);
    mandelbrot_flags_tool.Reconstruct(r_mandelbrot_flags,mandelbrot_flags);
	std::vector<int> info_partition_mandelbrot_flags=para_gene_tool.init_partition_data_shape(info_mandelbrot_flags,mandelbrot_flags_ops);
    sycl::buffer<int> info_partition_mandelbrot_flags_buffer(info_partition_mandelbrot_flags.data(), sycl::range<1>(info_partition_mandelbrot_flags.size()));
    
    // 数据移动
    q.memcpy(d_complex_points,r_complex_points,complex_points_Size*sizeof(complex<float>)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_complex_points_accessor = info_partition_complex_points_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_mandelbrot_flags_accessor = info_partition_mandelbrot_flags_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 嵌入计算
			
            mandel(d_complex_points+(i_*SplitLength[0][0]),d_mandelbrot_flags+(i_*SplitLength[1][0]),info_partition_complex_points_accessor,info_partition_mandelbrot_flags_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_mandelbrot_flags+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_mandelbrot_flags[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_mandelbrot_flags,reduction_mandelbrot_flags, Reduction_Size*sizeof(int)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_mandelbrot_flags, d_mandelbrot_flags, mandelbrot_flags_Size*sizeof(int)).wait();
    mandelbrot_flags_tool.UpdateData(r_mandelbrot_flags,mandelbrot_flags);

    // 内存释放
    
    sycl::free(d_complex_points, q);
    sycl::free(d_mandelbrot_flags, q);
}

int main() {
    // 初始化复数点向量
    InitializeComplexPoints();

    // 计算 Mandelbrot 集
    ComputeMandelbrot();

    // 打印统计信息
    PrintStats();

    return 0;
}
