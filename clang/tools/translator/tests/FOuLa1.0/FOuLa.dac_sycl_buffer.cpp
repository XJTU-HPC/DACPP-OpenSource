#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <any>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <queue>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

double phi(double x) { return x*x*x+x; }

double alpha(double t) { return 0.0; }

double beta(double t) { return 1.0+exp(t); }

double f(double x, double t) { return x*exp(t)-6*x; }

double exact(double x, double t) { return x*(x*x+exp(t)); }

//同样的问题，划分时，一个待计算数据和三个计算数据，一共四个数据要划分到一起




#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void pde(double* u_kin,double* u_kout,double* r,sycl::accessor<int, 1, sycl::access::mode::read_write> info_u_kin_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_u_kout_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_r_acc) 
{
    u_kout[0] = r[0] * u_kin[0] + (1 - 2 * r[0]) * u_kin[1] + r[0] * u_kin[2];
}


// 生成函数调用
void PDE_pde(const dacpp::Vector<double> & u_kin, dacpp::Vector<double> & u_kout, const dacpp::Vector<double> & r) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_u_kin;
    info_u_kin.dim = u_kin.getDim();
    for(int i = 0; i < info_u_kin.dim; i++) info_u_kin.dimLength.push_back(u_kin.getShape(i));
	
    // 数据信息初始化
    DataInfo info_u_kout;
    info_u_kout.dim = u_kout.getDim();
    for(int i = 0; i < info_u_kout.dim; i++) info_u_kout.dimLength.push_back(u_kout.getShape(i));
	
    // 数据信息初始化
    DataInfo info_r;
    info_r.dim = r.getDim();
    for(int i = 0; i < info_r.dim; i++) info_r.dimLength.push_back(r.getShape(i));
	
    // 规则分区算子初始化
    RegularSlice s = RegularSlice("s", 3, 1);
    s.setDimId(0);
    s.SetSplitSize(para_gene_tool.init_operetor_splitnumber(s,info_u_kin));

    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_u_kout));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops u_kin_Ops;
    
    s.setDimId(0);
    u_kin_Ops.push_back(s);


    // 算子组初始化
    Dac_Ops u_kout_Ops;
    
    i.setDimId(0);
    u_kout_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops r_Ops;
    

    // 算子组初始化
    Dac_Ops In_Ops;
    
    s.setDimId(0);
    In_Ops.push_back(s);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    i.setDimId(0);
    Out_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    i.setDimId(0);
    Reduction_Ops.push_back(i);


	
    //生成设备内存分配大小
    int u_kin_Size = para_gene_tool.init_device_memory_size(info_u_kin,u_kin_Ops);

    //生成设备内存分配大小
    int u_kout_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_u_kout);

    //生成设备内存分配大小
    int u_koutReduction_Size = para_gene_tool.init_device_memory_size(info_u_kout,Reduction_Ops);

    //生成设备内存分配大小
    int r_Size = para_gene_tool.init_device_memory_size(info_r,r_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(u_kin_Ops,u_kin_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,u_kout_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(r_Ops,r_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(u_kin_Ops);

    ops_s.push_back(In_Ops);

    ops_s.push_back(r_Ops);


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
    buffer <double> b_u_kin{u_kin_Size};

    // Buffer设备内存分配
    buffer <double> b_u_kout{u_kout_Size};

    // 规约Buffer设备内存分配
    std::vector<sycl::buffer<double, 1>> b_reduction_u_kout(u_koutReduction_Size, buffer<double, 1>{1});
    for(int i = 0; i < u_koutReduction_Size; i++){
        host_accessor temp_accessor{b_reduction_u_kout[i]};
        temp_accessor[0] = 0;
    }
    // Buffer设备内存分配
    buffer <double> b_r{r_Size};

    // 数据关联计算
    
	
    // 数据移动
    double* h_u_kin = (double*)malloc(u_kin.getSize()*sizeof(double));
    u_kin.tensor2Array(h_u_kin);
    {
        host_accessor temp_accessor{b_u_kin};
        for(int i = 0; i < u_kin_Size; i++){
            temp_accessor[i] = h_u_kin[i];
        }
    }

    // 数据移动
    double* h_u_kout = (double*)malloc(u_kout.getSize()*sizeof(double));

    // 数据移动
    double* h_r = (double*)malloc(r.getSize()*sizeof(double));
    r.tensor2Array(h_r);
    {
        host_accessor temp_accessor{b_r};
        for(int i = 0; i < r_Size; i++){
            temp_accessor[i] = h_r[i];
        }
    }

    // 数据重组
    DataReconstructor<double> u_kin_tool;
    
    // 数据算子组初始化
    Dac_Ops u_kin_ops;
    
    s.setDimId(0);
    u_kin_ops.push_back(s);

    u_kin_tool.init(info_u_kin,u_kin_ops);
    buffer<double> r_u_kin{u_kin_Size};
    u_kin_tool.Reconstruct(r_u_kin,b_u_kin,q);
	std::vector<int> info_partition_u_kin=para_gene_tool.init_partition_data_shape(info_u_kin,u_kin_ops);
    sycl::buffer<int> info_partition_u_kin_buffer(info_partition_u_kin.data(), sycl::range<1>(info_partition_u_kin.size()));

    // 数据重组
    DataReconstructor<double> u_kout_tool;
    
    // 数据算子组初始化
    Dac_Ops u_kout_ops;
    
    i.setDimId(0);
    u_kout_ops.push_back(i);

    u_kout_tool.init(info_u_kout,u_kout_ops);
    buffer<double> r_u_kout{u_kout_Size};
    u_kout_tool.Reconstruct(r_u_kout,b_u_kout,q);
	std::vector<int> info_partition_u_kout=para_gene_tool.init_partition_data_shape(info_u_kout,u_kout_ops);
    sycl::buffer<int> info_partition_u_kout_buffer(info_partition_u_kout.data(), sycl::range<1>(info_partition_u_kout.size()));

    // 数据重组
    DataReconstructor<double> r_tool;
    
    // 数据算子组初始化
    Dac_Ops r_ops;
    

    r_tool.init(info_r,r_ops);
    buffer<double> r_r{r_Size};
    r_tool.Reconstruct(r_r,b_r,q);
	std::vector<int> info_partition_r=para_gene_tool.init_partition_data_shape(info_r,r_ops);
    sycl::buffer<int> info_partition_r_buffer(info_partition_r.data(), sycl::range<1>(info_partition_r.size()));
    
	
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

    //队列提交命令组
    q.submit([&](handler &h) {
    
        accessor acc_u_kin{r_u_kin, h};
        accessor acc_u_kout{r_u_kout, h};
        accessor acc_r{r_r, h};
    
        auto info_partition_u_kin_accessor = info_partition_u_kin_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_u_kout_accessor = info_partition_u_kout_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_r_accessor = info_partition_r_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            const auto s_=(item_id+(0))%s.split_size;
            // 获得accessor指针
            
            auto* d_u_kin = acc_u_kin.get_multi_ptr<access::decorated::no>().get();
            auto* d_u_kout = acc_u_kout.get_multi_ptr<access::decorated::no>().get();
            auto* d_r = acc_r.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            pde(d_u_kin+(s_*SplitLength[0][0]),d_u_kout+(s_*SplitLength[1][0]),d_r,info_partition_u_kin_accessor,info_partition_u_kout_accessor,info_partition_r_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<u_koutReduction_Size;i++) {
            q.submit([&](handler &h) {
                accessor d_u_kout{r_u_kout, h};
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(b_reduction_u_kout[i], h, 
                sycl::plus<>(),property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_u_kout[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_u_kout};
            for(int i = 0; i < u_koutReduction_Size; i++){
                host_accessor temp_accessor{b_reduction_u_kout[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }


    //结果返回
    u_kout_tool.UpdateData(r_u_kout,b_u_kout,q);
    {
        host_accessor temp_accessor{b_u_kout};
        for(int i = 0; i < u_kout_Size; i++){
            h_u_kout[i] = temp_accessor[i];
        }
    }
    u_kout.array2Tensor(h_u_kout);

	

    // 内存释放
    
}

int main() {
    int n = 100; //时间域n等分
    int m = 5; //空间域m等分
    double r = 0.25;
    double a = 1.0;
    double h = 1.0 / m; //空间步长
    double tau = 1.0 / n; //时间步长
    double *x,*t,**u;
    
    //r=a*tau/(h*h);  //网比
    //printf("r=%.4f.\n",r);
    
    x = (double*)malloc(sizeof(double)*(m+1));
    for (int i=0;i<=m;i++) {
        x[i]=i*h;
    }
    t = (double*)malloc(sizeof(double)*(n+1));
    for (int i = 0; i <= n; i++) {
        t[i]=i*tau;
    }
    u = (double**)malloc(sizeof(double*)*(m+1));
    for (int i=0;i<=m;i++) {
        u[i]=(double*)malloc(sizeof(double)*(n+1));
    }
    for (int i = 0; i <= m; i++)
        u[i][0]=phi(x[i]);
    for (int i = 1; i <= n; i++) {
        u[0][i]=alpha(t[i]);
        u[m][i]=beta(t[i]);
    }
    
    // Flatten the 2D u array into a 1D vector for Tensor creation
    std::vector<double> u_flat;
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            u_flat.push_back(static_cast<double>(u[i][j]));  // Cast if needed
        }
    }

    dacpp::Matrix<double> u_tensor({m+1, n+1}, u_flat);

    for (int k = 0; k < n; k++) {
        dacpp::Vector<double> middle_tensor = u_tensor[{1,m}][k+1];
        std::vector<double> r_data;
        r_data.push_back(r);
        dacpp::Vector<double> R(r_data);
        dacpp::Vector<double> u_test1 = u_tensor[{}][k];
        PDE_pde(u_test1, middle_tensor, R);
        
        //计算完毕后，替换第1到4个点
        for (int i = 1; i <= m-1; i++) {
            u_tensor[i][k+1] = middle_tensor[i-1];
        }

    }

    // 每个位置需要下，左下，右下，三个位置的元素，串行中从下往上，从左往右遍历计算
    // 那么每一行的元素计算是互不相关的，可以并行执行，所有的行从下往上串行执行
    u_tensor[1].print();
    // double* data = new double[6 * 101];
    // u_tensor.tensor2Array(data);

    // // 将一维数组转换为二维 vector
    // std::vector<std::vector<double>> vec2D;
    // vec2D.resize(6, std::vector<double>(101));

    // // 将一维数组的数据填充到二维数组中
    // for (int i = 0; i < 6; ++i) {
    //     for (int j = 0; j < 101; ++j) {
    //         vec2D[i][j] = data[i * 101 + j];
    //     }
    // }


    // int j = int(0.2 / tau);
    // int number = int(0.4 / h);
    // for (int k = j; k <= n; k = k + j) {
    //     printf("(x,t)=(%.1f,%.1f), y=%.2f, exact=%.3f, err=%.3e.\n",x[number],t[k],vec2D[number][k],exact(x[number],t[k]),std::fabs(vec2D[number][k]-exact(x[number],t[k])));
    // }
    // for (int k = j; k <= n; k = k + j) {
    //     printf("(x,t)=(%.1f,%.1f), y=%.2f, exact=%.3f.\n",x[number],t[k],vec2D[number][k],exact(x[number],t[k]));
    // }


    return 0;
}