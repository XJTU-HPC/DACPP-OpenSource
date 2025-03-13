#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;





#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void mc(double* x,double* y,int* inside_circle,sycl::accessor<int, 1, sycl::access::mode::read_write> info_x_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_y_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_inside_circle_acc) 
{
    x[0] = rand() / (double)2147483647;
    y[0] = rand() / (double)2147483647;
    if (x[0] * x[0] + y[0] * y[0] <= 1.) {
        inside_circle[0] = 1;
    }
}


// 生成函数调用
void MC_mc(const dacpp::Vector<double> & x, const dacpp::Vector<double> & y, const dacpp::Vector<int> & inside_circle) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_x;
    info_x.dim = x.getDim();
    for(int i = 0; i < info_x.dim; i++) info_x.dimLength.push_back(x.getShape(i));
    // 数据信息初始化
    DataInfo info_y;
    info_y.dim = y.getDim();
    for(int i = 0; i < info_y.dim; i++) info_y.dimLength.push_back(y.getShape(i));
    // 数据信息初始化
    DataInfo info_inside_circle;
    info_inside_circle.dim = inside_circle.getDim();
    for(int i = 0; i < info_inside_circle.dim; i++) info_inside_circle.dimLength.push_back(inside_circle.getShape(i));
    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_x));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops x_Ops;
    
    i.setDimId(0);
    x_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops y_Ops;
    
    i.setDimId(0);
    y_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops inside_circle_Ops;
    
    i.setDimId(0);
    inside_circle_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    i.setDimId(0);
    In_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops Out_Ops;
    

    // 算子组初始化
    Dac_Ops Reduction_Ops;
    

	
    //生成设备内存分配大小
    int x_Size = para_gene_tool.init_device_memory_size(info_x,x_Ops);

    //生成设备内存分配大小
    int y_Size = para_gene_tool.init_device_memory_size(info_y,y_Ops);

    //生成设备内存分配大小
    int inside_circle_Size = para_gene_tool.init_device_memory_size(info_inside_circle,inside_circle_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(x_Ops,x_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(y_Ops,y_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(inside_circle_Ops,inside_circle_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(x_Ops);

    ops_s.push_back(y_Ops);

    ops_s.push_back(inside_circle_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[3][1] = {0};
    para_gene_tool.init_split_length_martix(3,1,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // Buffer设备内存分配
    buffer <double> b_x{x_Size};
    // Buffer设备内存分配
    buffer <double> b_y{y_Size};
    // Buffer设备内存分配
    buffer <int> b_inside_circle{inside_circle_Size};
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<double> x_tool;
    double* r_x=(double*)malloc(sizeof(double)*x_Size);
    
    // 数据算子组初始化
    Dac_Ops x_ops;
    
    i.setDimId(0);
    x_ops.push_back(i);
    x_tool.init(info_x,x_ops);
    x_tool.Reconstruct(r_x,x);
	std::vector<int> info_partition_x=para_gene_tool.init_partition_data_shape(info_x,x_ops);
    sycl::buffer<int> info_partition_x_buffer(info_partition_x.data(), sycl::range<1>(info_partition_x.size()));
    // 数据重组
    DataReconstructor<double> y_tool;
    double* r_y=(double*)malloc(sizeof(double)*y_Size);
    
    // 数据算子组初始化
    Dac_Ops y_ops;
    
    i.setDimId(0);
    y_ops.push_back(i);
    y_tool.init(info_y,y_ops);
    y_tool.Reconstruct(r_y,y);
	std::vector<int> info_partition_y=para_gene_tool.init_partition_data_shape(info_y,y_ops);
    sycl::buffer<int> info_partition_y_buffer(info_partition_y.data(), sycl::range<1>(info_partition_y.size()));
    // 数据重组
    DataReconstructor<int> inside_circle_tool;
    int* r_inside_circle=(int*)malloc(sizeof(int)*inside_circle_Size);
    
    // 数据算子组初始化
    Dac_Ops inside_circle_ops;
    
    i.setDimId(0);
    inside_circle_ops.push_back(i);
    inside_circle_tool.init(info_inside_circle,inside_circle_ops);
    inside_circle_tool.Reconstruct(r_inside_circle,inside_circle);
	std::vector<int> info_partition_inside_circle=para_gene_tool.init_partition_data_shape(info_inside_circle,inside_circle_ops);
    sycl::buffer<int> info_partition_inside_circle_buffer(info_partition_inside_circle.data(), sycl::range<1>(info_partition_inside_circle.size()));
    
    { //Buffer主机到设备传输数据
        host_accessor temp_accessor{b_x};
        for(int i = 0; i < x_Size; i++){
            temp_accessor[i] = r_x[i];
        }
    }

    { //Buffer主机到设备传输数据
        host_accessor temp_accessor{b_y};
        for(int i = 0; i < y_Size; i++){
            temp_accessor[i] = r_y[i];
        }
    }

    { //Buffer主机到设备传输数据
        host_accessor temp_accessor{b_inside_circle};
        for(int i = 0; i < inside_circle_Size; i++){
            temp_accessor[i] = r_inside_circle[i];
        }
    }

	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
    
        accessor acc_x{b_x, h};
        accessor acc_y{b_y, h};
        accessor acc_inside_circle{b_inside_circle, h};
    
        auto info_partition_x_accessor = info_partition_x_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_y_accessor = info_partition_y_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_inside_circle_accessor = info_partition_inside_circle_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 获得accessor指针
            
            auto* d_x = acc_x.get_multi_ptr<access::decorated::no>().get();
            auto* d_y = acc_y.get_multi_ptr<access::decorated::no>().get();
            auto* d_inside_circle = acc_inside_circle.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            mc(d_x+(i_*SplitLength[0][0]),d_y+(i_*SplitLength[1][0]),d_inside_circle+(i_*SplitLength[2][0]),info_partition_x_accessor,info_partition_y_accessor,info_partition_inside_circle_accessor);
        });
    }).wait();
    

	
	

    // 内存释放
    
}

double monte_carlo_pi(int num_samples) {
    std::vector<double> x(num_samples,0.0);
    std::vector<double> y(num_samples,0.0);
    std::vector<int> inside_circle(num_samples,0.0);
    int inside_circle_sum = 0;

    dacpp::Vector<double> x_tensor(x);
    dacpp::Vector<double> y_tensor(y);
    dacpp::Vector<int> inside_circle_tensor(inside_circle);

    MC_mc(x_tensor, y_tensor, inside_circle_tensor);
    // 计算π的估算值

    for(int  i = 0; i < num_samples; i++ ){
        if(inside_circle_tensor[i] == 1)   inside_circle_sum++;
    }

    return 4.0 * inside_circle_sum / num_samples;
}

int main() {
    srand(time(0));  // 用当前时间作为随机种子
    int num_samples = 1000000;  // 选择模拟的随机点数量
    double pi_estimate = monte_carlo_pi(num_samples);

    cout << "Estimated Pi: " << pi_estimate << endl;
    return 0;
}
