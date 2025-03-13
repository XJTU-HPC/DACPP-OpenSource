#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "ReconTensor.h"
#include <random>
namespace dacpp {
    typedef std::vector<std::any> list;
}

//线性同余生成器参数
const int m = 2147483647;  // 2^31 - 1, 最大的质数
const int a1 = 16807;       // 常用的增幅1
const int c1 = 0;           // 增量为0
const int a2 = 48271;       // 常用的增幅2
const int c2 = 0;           // 增量为0
const int seed = 12345;     // 固定种子
const int N = 1024;         //模拟点的数量，决定了并行规模
using namespace std;
//一维随机游走参数
const int num_walkers = 1024;//并行规模，即模拟点数
const int num_steps = 10;//随机步数

//二维随机游走参数
const int num_walkers_2D = 1024;//并行规模，D即模拟点数
const int num_steps_2D = 10;//随机步数

bool isInsideUnitCircle(double x, double y) {
    return (x * x + y * y) <= 1.0;
}

// 判断点是否在单位正方形内（默认范围是[0,1] x [0,1]）
bool in_square(double x, double y) {
    return true;  // 单位正方形的条件是点都在[0,1]范围内
}

// 判断点是否在单位三角形内
bool in_triangle(double x, double y) {
    return (x >= 0 && y >= 0 && y <= 1 - x);  // 单位三角形的条件：x + y <= 1, 且x, y >= 0
}

// 估算椭圆的面积
bool in_ellipse(double x, double y) {
    // 假设椭圆方程为 (x^2 / a^2) + (y^2 / b^2) = 1
    double a = 1.0;  // 椭圆的长半轴
    double b = 0.5;  // 椭圆的短半轴
    return (x * x / (a * a)) + (y * y / (b * b)) <= 1.0;
}


// 估算任意多边形的面积,
//使用射线法，从该点向水平方向发射一条射线，若与图形有1个交点则在内，两个则在外，
constexpr int POLYGON_SIZE = 4;
const std::array<std::pair<double, double>, POLYGON_SIZE> square = {{
    {0, 0}, {1, 0}, {1, 1}, {0, 1}
}};

// SYCL 设备代码中使用的 in_polygon,sycl中不能直接使用pair
bool in_polygon(double x, double y, const std::array<std::pair<double, double>, POLYGON_SIZE>& vertices) {
    int n = vertices.size();
    bool inside = false;

    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = vertices[i].first, yi = vertices[i].second;
        double xj = vertices[j].first, yj = vertices[j].second;

        if ((yj - yi) != 0) { // 避免除零错误
            if (((yi > y) != (yj > y)) && 
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
                inside = !inside;
            }
        }
    }
    return inside;
}

// 估算复杂曲线或几何体的面积
bool in_complex_curve(double x, double y) {
    // 使用 parametric 方程来定义一个复杂曲线心形曲线
    // 心形曲线的 parametric 方程：x(t) = 16sin^3(t), y(t) = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)

        return std::pow(x*x + y*y - 1, 3) - x*x*y*y*y <= 0;

    
}


//几何处理shell



//一维随机游走shell


//二维随机游走shell


//几何处理calc

void random_walk_1D_shell_random_walk_1D_calc(const dacpp::Vector<int> & index, dacpp::Vector<int> & final_positions);

//一维随机游走calc


//二维随机游走calc
void MC_pi_mc_pi(const dacpp::Vector<int> & index, dacpp::Vector<int> & inside_circle);

//几何处理函数
double monte_carlo_pi(int num_samples) {

    //std::vector<double> x(num_samples,0.0);
    //std::vector<double> y(num_samples,0.0);
    std::vector<int> inside_circle(num_samples,0);
    //std::cout << inside_circle[5] << std::endl;
    std::vector<int> index(num_samples,0);
    for(int i =  0; i < num_samples; i++) index[i] = i;
    int inside_circle_sum = 0;

    //dacpp::Vector<double> x_tensor(x);
    //dacpp::Vector<double> y_tensor(y);
    dacpp::Vector<int> inside_circle_tensor(inside_circle);
    dacpp::Vector<int> index_tensor(index);

    MC_pi_mc_pi(index_tensor, inside_circle_tensor);
    // 计算π的估算值

    for(int  i = 0; i < num_samples; i++ ){
        if(inside_circle_tensor[i] == 1)   inside_circle_sum++;
    }
    std::cout << inside_circle_sum << std::endl;

    return 4.0 * inside_circle_sum / num_samples;
}


// **一维随机游走**
dacpp::Vector<int>  random_walk_1D(std::vector<int>& final_positions) {
    std::vector<int> index(num_walkers,0);
    for(int i =  0; i < num_walkers; i++) index[i] = i;
    dacpp::Vector<int> final_positions_tensor(final_positions);
    dacpp::Vector<int> index_tensor(index);
    random_walk_1D_shell_random_walk_1D_calc(index_tensor, final_positions_tensor);
    return final_positions_tensor;
}


//二维随机游走
#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void mc_pi(int* index,int* inside_circle,sycl::accessor<int, 1, sycl::access::mode::read_write> info_index_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_inside_circle_acc) 
{
    int x1 = (seed + index[0]) % m;
    x1 = (a1 * x1 + c1) % m;
    int x2 = (seed + index[0] + 1) % m;
    x2 = (a2 * x2 + c2) % m;
    double x = 2. * (x1 / static_cast<double>(m)) - 1.;
    double y = 2. * (x2 / static_cast<double>(m)) - 1.;
    if (in_square(x, y)) {
        inside_circle[0] = 1;
    }
}


// 生成函数调用
void MC_pi_mc_pi(const dacpp::Vector<int> & index, dacpp::Vector<int> & inside_circle) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_index;
    info_index.dim = index.getDim();
    for(int i = 0; i < info_index.dim; i++) info_index.dimLength.push_back(index.getShape(i));
    // 数据信息初始化
    DataInfo info_inside_circle;
    info_inside_circle.dim = inside_circle.getDim();
    for(int i = 0; i < info_inside_circle.dim; i++) info_inside_circle.dimLength.push_back(inside_circle.getShape(i));
    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_index));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops index_Ops;
    
    i.setDimId(0);
    index_Ops.push_back(i);


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
    
    i.setDimId(0);
    Out_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    i.setDimId(0);
    Reduction_Ops.push_back(i);


	
    //生成设备内存分配大小
    int index_Size = para_gene_tool.init_device_memory_size(info_index,index_Ops);

    //生成设备内存分配大小
    int inside_circle_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_inside_circle);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_inside_circle,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(index_Ops,index_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,inside_circle_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(index_Ops);

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
    int *d_index=malloc_device<int>(index_Size,q);
    // 设备内存分配
    int *d_inside_circle=malloc_device<int>(inside_circle_Size,q);
    // 归约设备内存分配
    int *reduction_inside_circle = malloc_device<int>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<int> index_tool;
    int* r_index=(int*)malloc(sizeof(int)*index_Size);
    
    // 数据算子组初始化
    Dac_Ops index_ops;
    
    i.setDimId(0);
    index_ops.push_back(i);
    index_tool.init(info_index,index_ops);
    index_tool.Reconstruct(r_index,index);
	std::vector<int> info_partition_index=para_gene_tool.init_partition_data_shape(info_index,index_ops);
    sycl::buffer<int> info_partition_index_buffer(info_partition_index.data(), sycl::range<1>(info_partition_index.size()));
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
    
    // 设备数据初始化
    q.memset(d_index,0,index_Size*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_index,r_index,index_Size*sizeof(int)).wait();
    // 设备数据初始化
    q.memset(d_inside_circle,0,inside_circle_Size*sizeof(int)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_index_accessor = info_partition_index_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_inside_circle_accessor = info_partition_inside_circle_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 嵌入计算
			
            mc_pi(d_index+(i_*SplitLength[0][0]),d_inside_circle+(i_*SplitLength[1][0]),info_partition_index_accessor,info_partition_inside_circle_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_inside_circle+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_inside_circle[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_inside_circle,reduction_inside_circle, Reduction_Size*sizeof(int)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_inside_circle, d_inside_circle, inside_circle_Size*sizeof(int)).wait();
    inside_circle_tool.UpdateData(r_inside_circle,inside_circle);

    // 内存释放
    
    sycl::free(d_index, q);
    sycl::free(d_inside_circle, q);
}

void random_walk_1D_calc(int* index,int* final_positions,sycl::accessor<int, 1, sycl::access::mode::read_write> info_index_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_final_positions_acc) 
{
    int x = 0;
    int walker_seed = seed + index[0];
    for (int step = 0; step < num_steps; step++) {
        walker_seed = (a1 * walker_seed + c1) % m;
        int rand_val = walker_seed % 2;
        int step_direction = (rand_val * 2) - 1;
        x += step_direction;
    }
    final_positions[0] = x;
}


// 生成函数调用
void random_walk_1D_shell_random_walk_1D_calc(const dacpp::Vector<int> & index, dacpp::Vector<int> & final_positions) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_index;
    info_index.dim = index.getDim();
    for(int i = 0; i < info_index.dim; i++) info_index.dimLength.push_back(index.getShape(i));
    // 数据信息初始化
    DataInfo info_final_positions;
    info_final_positions.dim = final_positions.getDim();
    for(int i = 0; i < info_final_positions.dim; i++) info_final_positions.dimLength.push_back(final_positions.getShape(i));
    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_index));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops index_Ops;
    
    i.setDimId(0);
    index_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops final_positions_Ops;
    
    i.setDimId(0);
    final_positions_Ops.push_back(i);


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
    int index_Size = para_gene_tool.init_device_memory_size(info_index,index_Ops);

    //生成设备内存分配大小
    int final_positions_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_final_positions);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_final_positions,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(index_Ops,index_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,final_positions_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(index_Ops);

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
    int *d_index=malloc_device<int>(index_Size,q);
    // 设备内存分配
    int *d_final_positions=malloc_device<int>(final_positions_Size,q);
    // 归约设备内存分配
    int *reduction_final_positions = malloc_device<int>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<int> index_tool;
    int* r_index=(int*)malloc(sizeof(int)*index_Size);
    
    // 数据算子组初始化
    Dac_Ops index_ops;
    
    i.setDimId(0);
    index_ops.push_back(i);
    index_tool.init(info_index,index_ops);
    index_tool.Reconstruct(r_index,index);
	std::vector<int> info_partition_index=para_gene_tool.init_partition_data_shape(info_index,index_ops);
    sycl::buffer<int> info_partition_index_buffer(info_partition_index.data(), sycl::range<1>(info_partition_index.size()));
    // 数据重组
    DataReconstructor<int> final_positions_tool;
    int* r_final_positions=(int*)malloc(sizeof(int)*final_positions_Size);
    
    // 数据算子组初始化
    Dac_Ops final_positions_ops;
    
    i.setDimId(0);
    final_positions_ops.push_back(i);
    final_positions_tool.init(info_final_positions,final_positions_ops);
    final_positions_tool.Reconstruct(r_final_positions,final_positions);
	std::vector<int> info_partition_final_positions=para_gene_tool.init_partition_data_shape(info_final_positions,final_positions_ops);
    sycl::buffer<int> info_partition_final_positions_buffer(info_partition_final_positions.data(), sycl::range<1>(info_partition_final_positions.size()));
    
    // 设备数据初始化
    q.memset(d_index,0,index_Size*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_index,r_index,index_Size*sizeof(int)).wait();
    // 设备数据初始化
    q.memset(d_final_positions,0,final_positions_Size*sizeof(int)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_index_accessor = info_partition_index_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_final_positions_accessor = info_partition_final_positions_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 嵌入计算
			
            random_walk_1D_calc(d_index+(i_*SplitLength[0][0]),d_final_positions+(i_*SplitLength[1][0]),info_partition_index_accessor,info_partition_final_positions_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_final_positions+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_final_positions[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_final_positions,reduction_final_positions, Reduction_Size*sizeof(int)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_final_positions, d_final_positions, final_positions_Size*sizeof(int)).wait();
    final_positions_tool.UpdateData(r_final_positions,final_positions);

    // 内存释放
    
    sycl::free(d_index, q);
    sycl::free(d_final_positions, q);
}

void random_walk_2D_calc(int* index,int* final_positions_x,int* final_positions_y,sycl::accessor<int, 1, sycl::access::mode::read_write> info_index_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_final_positions_x_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_final_positions_y_acc) 
{
    int x = 0;
    int y = 0;
    int walker_seed = seed + index[0];
    for (int step = 0; step < num_steps; step++) {
        walker_seed = (a1 * walker_seed + c1) % m;
        int rand_val = walker_seed % 4;
        if (rand_val == 0) {
            y += 1;
        } else if (rand_val == 1) {
            y -= 1;
        } else if (rand_val == 2) {
            x -= 1;
        } else {
            x += 1;
        }
    }
    final_positions_x[0] = x;
    final_positions_y[0] = y;
}


// 生成函数调用
void random_walk_2D_shell_random_walk_2D_calc(const dacpp::Vector<int> & index, dacpp::Vector<int> & final_positions_x, dacpp::Vector<int> & final_positions_y) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_index;
    info_index.dim = index.getDim();
    for(int i = 0; i < info_index.dim; i++) info_index.dimLength.push_back(index.getShape(i));
    // 数据信息初始化
    DataInfo info_final_positions_x;
    info_final_positions_x.dim = final_positions_x.getDim();
    for(int i = 0; i < info_final_positions_x.dim; i++) info_final_positions_x.dimLength.push_back(final_positions_x.getShape(i));
    // 数据信息初始化
    DataInfo info_final_positions_y;
    info_final_positions_y.dim = final_positions_y.getDim();
    for(int i = 0; i < info_final_positions_y.dim; i++) info_final_positions_y.dimLength.push_back(final_positions_y.getShape(i));
    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_index));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops index_Ops;
    
    i.setDimId(0);
    index_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops final_positions_x_Ops;
    
    i.setDimId(0);
    final_positions_x_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops final_positions_y_Ops;
    
    i.setDimId(0);
    final_positions_y_Ops.push_back(i);


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
    int index_Size = para_gene_tool.init_device_memory_size(info_index,index_Ops);

    //生成设备内存分配大小
    int final_positions_x_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_final_positions_x);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_final_positions_x,Reduction_Ops);

    //生成设备内存分配大小
    int final_positions_y_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_final_positions_y);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_final_positions_y,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(index_Ops,index_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,final_positions_x_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,final_positions_y_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(index_Ops);

    ops_s.push_back(In_Ops);

    ops_s.push_back(In_Ops);


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
    
    // 设备内存分配
    int *d_index=malloc_device<int>(index_Size,q);
    // 设备内存分配
    int *d_final_positions_x=malloc_device<int>(final_positions_x_Size,q);
    // 归约设备内存分配
    int *reduction_final_positions_x = malloc_device<int>(Reduction_Size,q);
    // 设备内存分配
    int *d_final_positions_y=malloc_device<int>(final_positions_y_Size,q);
    // 归约设备内存分配
    int *reduction_final_positions_y = malloc_device<int>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<int> index_tool;
    int* r_index=(int*)malloc(sizeof(int)*index_Size);
    
    // 数据算子组初始化
    Dac_Ops index_ops;
    
    i.setDimId(0);
    index_ops.push_back(i);
    index_tool.init(info_index,index_ops);
    index_tool.Reconstruct(r_index,index);
	std::vector<int> info_partition_index=para_gene_tool.init_partition_data_shape(info_index,index_ops);
    sycl::buffer<int> info_partition_index_buffer(info_partition_index.data(), sycl::range<1>(info_partition_index.size()));
    // 数据重组
    DataReconstructor<int> final_positions_x_tool;
    int* r_final_positions_x=(int*)malloc(sizeof(int)*final_positions_x_Size);
    
    // 数据算子组初始化
    Dac_Ops final_positions_x_ops;
    
    i.setDimId(0);
    final_positions_x_ops.push_back(i);
    final_positions_x_tool.init(info_final_positions_x,final_positions_x_ops);
    final_positions_x_tool.Reconstruct(r_final_positions_x,final_positions_x);
	std::vector<int> info_partition_final_positions_x=para_gene_tool.init_partition_data_shape(info_final_positions_x,final_positions_x_ops);
    sycl::buffer<int> info_partition_final_positions_x_buffer(info_partition_final_positions_x.data(), sycl::range<1>(info_partition_final_positions_x.size()));
    // 数据重组
    DataReconstructor<int> final_positions_y_tool;
    int* r_final_positions_y=(int*)malloc(sizeof(int)*final_positions_y_Size);
    
    // 数据算子组初始化
    Dac_Ops final_positions_y_ops;
    
    i.setDimId(0);
    final_positions_y_ops.push_back(i);
    final_positions_y_tool.init(info_final_positions_y,final_positions_y_ops);
    final_positions_y_tool.Reconstruct(r_final_positions_y,final_positions_y);
	std::vector<int> info_partition_final_positions_y=para_gene_tool.init_partition_data_shape(info_final_positions_y,final_positions_y_ops);
    sycl::buffer<int> info_partition_final_positions_y_buffer(info_partition_final_positions_y.data(), sycl::range<1>(info_partition_final_positions_y.size()));
    
    // 设备数据初始化
    q.memset(d_index,0,index_Size*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_index,r_index,index_Size*sizeof(int)).wait();
    // 设备数据初始化
    q.memset(d_final_positions_x,0,final_positions_x_Size*sizeof(int)).wait();
    // 设备数据初始化
    q.memset(d_final_positions_y,0,final_positions_y_Size*sizeof(int)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_index_accessor = info_partition_index_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_final_positions_x_accessor = info_partition_final_positions_x_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_final_positions_y_accessor = info_partition_final_positions_y_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 嵌入计算
			
            random_walk_2D_calc(d_index+(i_*SplitLength[0][0]),d_final_positions_x+(i_*SplitLength[1][0]),d_final_positions_y+(i_*SplitLength[2][0]),info_partition_index_accessor,info_partition_final_positions_x_accessor,info_partition_final_positions_y_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_final_positions_y+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_final_positions_y[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_final_positions_y,reduction_final_positions_y, Reduction_Size*sizeof(int)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_final_positions_y, d_final_positions_y, final_positions_y_Size*sizeof(int)).wait();
    final_positions_y_tool.UpdateData(r_final_positions_y,final_positions_y);

    // 内存释放
    
    sycl::free(d_index, q);
    sycl::free(d_final_positions_x, q);
    sycl::free(d_final_positions_y, q);
}

dacpp::Matrix<int>  random_walk_2D(std::vector<int>& final_positions_x,std::vector<int>& final_positions_y) {
    std::vector<int> index(num_walkers,0);
    for(int i =  0; i < num_walkers; i++) index[i] = i;
    dacpp::Vector<int> final_positions_x_tensor(final_positions_x);
    dacpp::Vector<int> final_positions_y_tensor(final_positions_y);
    
    dacpp::Vector<int> index_tensor(index);
    random_walk_2D_shell_random_walk_2D_calc(index_tensor, final_positions_x_tensor, final_positions_y_tensor);
    dacpp::Matrix<int> result({2,num_walkers});
    result[0]= final_positions_x_tensor;
    result[1]= final_positions_y_tensor;
    // = {final_positions_x_tensor,final_positions_y_tensor};
    return result;
}



int main() {

    /*
        蒙特卡洛估算Π
        N：随机点数量
    */
    double pi_estimate = monte_carlo_pi(N);
    std::cout << "Estimated Pi: " << pi_estimate << std::endl;


    /*
        一维随机游走模拟
        num_walkers：随机点数量
    */
    std::vector<int> final_positions(num_walkers);
    std::vector<int> result(num_walkers,0);
    dacpp::Vector<int> result_tensor(result);
    result_tensor =  random_walk_1D(final_positions);
    std::cout << "Final positions of first 10 walkers: ";// 输出前 10 个 walker 的最终位置
    for (int i = 0; i < 10; i++) {
        std::cout << result_tensor[i] << " ";
    }
    std::cout << std::endl;



    /*
        二维随机游走模拟
    */
    std::vector<int> final_positions_x(num_walkers_2D);
    std::vector<int> final_positions_y(num_walkers_2D);
    //std::vector<std::vector<int>> result(num_walkers_2D, std::vector<int>(2, 0));
    dacpp::Matrix<int> result_tensor_2D({2,num_walkers});
    result_tensor_2D =  random_walk_2D(final_positions_x, final_positions_y);
    std::cout << "Final positions of first 10 walkers: ";// 输出前 10 个 walker 的最终位置
    for (int i = 0; i < 10; i++) {
        std::cout << "Walker " << i << ": (" << result_tensor_2D[0][i] << ", " << result_tensor_2D[1][i] << ")\n";
    }
    std::cout << std::endl;





    return 0;
}
