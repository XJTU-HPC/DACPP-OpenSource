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







// 打印统计信息
void PrintStats() {
    cout << "Mandelbrot Set Statistics:\n";
    cout << "Total points: " << total_points << "\n";
    cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
}

#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void mandel(const complex<float>* complex_points,int* mandelbrot_flags,int complex_points_0,int mandelbrot_flags_0,int complex_points_0_shape,int mandelbrot_flags_0_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_complex_points_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_mandelbrot_flags_acc) {
    const complex<float> &c = complex_points[0+complex_points_0];
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
        mandelbrot_flags[0+mandelbrot_flags_0] = 1;
    }
}


// 生成函数调用
void MANDEL_mandel(const dacpp::Vector<complex<float> > & complex_points, dacpp::Vector<int> & mandelbrot_flags) { 
    using namespace sycl;
    // 设备选择
    auto selector = default_selector_v;
    sycl::queue dacpp_q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_complex_points;
    info_complex_points.dim = complex_points.getDim();
    int info_complex_points_Shape[1] = {0};
    for(int i = 0; i < info_complex_points.dim; i++)
    {
        info_complex_points.dimLength.push_back(complex_points.getShape(i));
        info_complex_points_Shape[i] = complex_points.getShape(i);
    }
	
    // 数据信息初始化
    DataInfo info_mandelbrot_flags;
    info_mandelbrot_flags.dim = mandelbrot_flags.getDim();
    int info_mandelbrot_flags_Shape[1] = {0};
    for(int i = 0; i < info_mandelbrot_flags.dim; i++)
    {
        info_mandelbrot_flags.dimLength.push_back(mandelbrot_flags.getShape(i));
        info_mandelbrot_flags_Shape[i] = mandelbrot_flags.getShape(i);
    }
	
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


	
	
	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);


    // 设备内存分配
    
    // 数据关联计算
    
	
    // 数据移动
    complex<float>* h_complex_points = (complex<float>*)malloc(complex_points.getSize()*sizeof(complex<float>));
    complex_points.tensor2Array(h_complex_points);
	buffer<complex<float>, 1> r_complex_points(h_complex_points, range<1>(complex_points.getSize()));

    // 数据移动
    int* h_mandelbrot_flags = (int*)malloc(mandelbrot_flags.getSize()*sizeof(int));

    // 数据重组
    
    
    // 数据算子组初始化
    Dac_Ops complex_points_ops;
    
    i.setDimId(0);
    complex_points_ops.push_back(i);


	std::vector<int> info_partition_complex_points=para_gene_tool.init_partition_data_shape(info_complex_points,complex_points_ops);
    sycl::buffer<int> info_partition_complex_points_buffer(info_partition_complex_points.data(), sycl::range<1>(info_partition_complex_points.size()));

    // 数据重组
    
    // 数据算子组初始化
    Dac_Ops mandelbrot_flags_ops;
    
    i.setDimId(0);
    mandelbrot_flags_ops.push_back(i);


    auto r_mandelbrot_flags = std::make_unique<sycl::buffer<int, 1>>(h_mandelbrot_flags,sycl::range<1>(mandelbrot_flags.getSize()));
    r_mandelbrot_flags->set_final_data(h_mandelbrot_flags);

	std::vector<int> info_partition_mandelbrot_flags=para_gene_tool.init_partition_data_shape(info_mandelbrot_flags,mandelbrot_flags_ops);
    sycl::buffer<int> info_partition_mandelbrot_flags_buffer(info_partition_mandelbrot_flags.data(), sycl::range<1>(info_partition_mandelbrot_flags.size()));
    
	
	
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
    
        accessor<complex<float>, 1, access::mode::read> acc_complex_points(r_complex_points, h);
        
        accessor<int, 1, sycl::access::mode::discard_write> acc_mandelbrot_flags(*r_mandelbrot_flags, h);
    
        auto info_partition_complex_points_accessor = info_partition_complex_points_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_mandelbrot_flags_accessor = info_partition_mandelbrot_flags_buffer.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
			// 获得划分数据单元左上角（第一个元素）的位置
			
			const auto complex_points_0 = i_ * i.stride;
			const auto mandelbrot_flags_0 = i_ * i.stride;
            // 获得accessor指针
            
            auto* d_complex_points = acc_complex_points.get_multi_ptr<access::decorated::no>().get();
            auto* d_mandelbrot_flags = acc_mandelbrot_flags.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            mandel(d_complex_points,d_mandelbrot_flags,complex_points_0,mandelbrot_flags_0,info_complex_points_Shape[0],info_mandelbrot_flags_Shape[0],info_partition_complex_points_accessor,info_partition_mandelbrot_flags_accessor);
        });
    }).wait();
    

	
    //结果返回语句改为析构语句
    r_mandelbrot_flags.reset();
    mandelbrot_flags.array2Tensor(h_mandelbrot_flags);

	

}

int main() {
    // 初始化复数点向量
    InitializeComplexPoints();

    // 计算 Mandelbrot 集
    mandelbrot_flags.resize(total_points, 0);  // 初始化一维数组为 0

    dacpp::Vector<complex<float>> complex_points_tensor(complex_points);
    dacpp::Vector<int> mandelbrot_flags_tensor(mandelbrot_flags);


    MANDEL_mandel(complex_points_tensor, mandelbrot_flags_tensor);

    // 统计数组中 1 的个数
    mandelbrot_count = 0;
    for (int i = 0; i < total_points; i++){
        if (mandelbrot_flags_tensor[i] == 1) mandelbrot_count++;
    }

    // 打印统计信息
    PrintStats();

    return 0;
}
