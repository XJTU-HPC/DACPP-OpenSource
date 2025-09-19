#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
using Complex = std::complex<double>;  // 复数类型别名
const int N = 8;






// 离散傅里叶变换（DFT）
#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void dft(std::complex<double>* input,std::complex<double>* output,int* vec,sycl::accessor<int, 1, sycl::access::mode::read_write> info_input_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_output_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_vec_acc) 
{
    Complex sum(0, 0);
    for (int n = 0; n < N; ++n) {
        double angle = -2. * 3.1415926535897931 * vec[0] * n / N;
        Complex W_n(std::cos(angle), std::sin(angle));
        sum += input[n] * W_n;
    }
    output[0] = sum;
}


// 生成函数调用
void DFT_dft(const dacpp::Vector<std::complex<double> > & input, dacpp::Vector<std::complex<double> > & output, const dacpp::Vector<int> & vec) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_input;
    info_input.dim = input.getDim();
    for(int i = 0; i < info_input.dim; i++) info_input.dimLength.push_back(input.getShape(i));
	
    // 数据信息初始化
    DataInfo info_output;
    info_output.dim = output.getDim();
    for(int i = 0; i < info_output.dim; i++) info_output.dimLength.push_back(output.getShape(i));
	
    // 数据信息初始化
    DataInfo info_vec;
    info_vec.dim = vec.getDim();
    for(int i = 0; i < info_vec.dim; i++) info_vec.dimLength.push_back(vec.getShape(i));
	
    // 降维算子初始化
    Index i = Index("i");
    i.setDimId(0);
    i.SetSplitSize(para_gene_tool.init_operetor_splitnumber(i,info_output));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops input_Ops;
    

    // 算子组初始化
    Dac_Ops output_Ops;
    
    i.setDimId(0);
    output_Ops.push_back(i);


    // 算子组初始化
    Dac_Ops vec_Ops;
    
    i.setDimId(0);
    vec_Ops.push_back(i);


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
    int input_Size = para_gene_tool.init_device_memory_size(info_input,input_Ops);

    //生成设备内存分配大小
    int output_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_output);

    //生成设备内存分配大小
    int outputReduction_Size = para_gene_tool.init_device_memory_size(info_output,Reduction_Ops);

    //生成设备内存分配大小
    int vec_Size = para_gene_tool.init_device_memory_size(info_vec,vec_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(input_Ops,input_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,output_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(vec_Ops,vec_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(input_Ops);

    ops_s.push_back(In_Ops);

    ops_s.push_back(vec_Ops);


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
    buffer <std::complex<double>> b_input{input_Size};

    // Buffer设备内存分配
    buffer <std::complex<double>> b_output{output_Size};

    // 规约Buffer设备内存分配
    std::vector<sycl::buffer<std::complex<double>, 1>> b_reduction_output(outputReduction_Size, buffer<std::complex<double>, 1>{1});
    for(int i = 0; i < outputReduction_Size; i++){
        host_accessor temp_accessor{b_reduction_output[i]};
        temp_accessor[0] = 0;
    }
    // Buffer设备内存分配
    buffer <int> b_vec{vec_Size};

    // 数据关联计算
    
	
    // 数据移动
    std::complex<double>* h_input = (std::complex<double>*)malloc(input.getSize()*sizeof(std::complex<double>));
    input.tensor2Array(h_input);
    {
        host_accessor temp_accessor{b_input};
        for(int i = 0; i < input_Size; i++){
            temp_accessor[i] = h_input[i];
        }
    }

    // 数据移动
    std::complex<double>* h_output = (std::complex<double>*)malloc(output.getSize()*sizeof(std::complex<double>));

    // 数据移动
    int* h_vec = (int*)malloc(vec.getSize()*sizeof(int));
    vec.tensor2Array(h_vec);
    {
        host_accessor temp_accessor{b_vec};
        for(int i = 0; i < vec_Size; i++){
            temp_accessor[i] = h_vec[i];
        }
    }

    // 数据重组
    DataReconstructor<std::complex<double>> input_tool;
    
    // 数据算子组初始化
    Dac_Ops input_ops;
    

    input_tool.init(info_input,input_ops);
    buffer<std::complex<double>> r_input{input_Size};
    input_tool.Reconstruct(r_input,b_input,q);
	std::vector<int> info_partition_input=para_gene_tool.init_partition_data_shape(info_input,input_ops);
    sycl::buffer<int> info_partition_input_buffer(info_partition_input.data(), sycl::range<1>(info_partition_input.size()));

    // 数据重组
    DataReconstructor<std::complex<double>> output_tool;
    
    // 数据算子组初始化
    Dac_Ops output_ops;
    
    i.setDimId(0);
    output_ops.push_back(i);

    output_tool.init(info_output,output_ops);
    buffer<std::complex<double>> r_output{output_Size};
    output_tool.Reconstruct(r_output,b_output,q);
	std::vector<int> info_partition_output=para_gene_tool.init_partition_data_shape(info_output,output_ops);
    sycl::buffer<int> info_partition_output_buffer(info_partition_output.data(), sycl::range<1>(info_partition_output.size()));

    // 数据重组
    DataReconstructor<int> vec_tool;
    
    // 数据算子组初始化
    Dac_Ops vec_ops;
    
    i.setDimId(0);
    vec_ops.push_back(i);

    vec_tool.init(info_vec,vec_ops);
    buffer<int> r_vec{vec_Size};
    vec_tool.Reconstruct(r_vec,b_vec,q);
	std::vector<int> info_partition_vec=para_gene_tool.init_partition_data_shape(info_vec,vec_ops);
    sycl::buffer<int> info_partition_vec_buffer(info_partition_vec.data(), sycl::range<1>(info_partition_vec.size()));
    
	
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

    //队列提交命令组
    q.submit([&](handler &h) {
    
        accessor acc_input{r_input, h};
        accessor acc_output{r_output, h};
        accessor acc_vec{r_vec, h};
    
        auto info_partition_input_accessor = info_partition_input_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_output_accessor = info_partition_output_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_vec_accessor = info_partition_vec_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 获得accessor指针
            
            auto* d_input = acc_input.get_multi_ptr<access::decorated::no>().get();
            auto* d_output = acc_output.get_multi_ptr<access::decorated::no>().get();
            auto* d_vec = acc_vec.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            dft(d_input,d_output+(i_*SplitLength[1][0]),d_vec+(i_*SplitLength[2][0]),info_partition_input_accessor,info_partition_output_accessor,info_partition_vec_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<outputReduction_Size;i++) {
            q.submit([&](handler &h) {
                accessor d_output{r_output, h};
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(b_reduction_output[i], h, 
                sycl::plus<>(),property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_output[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_output};
            for(int i = 0; i < outputReduction_Size; i++){
                host_accessor temp_accessor{b_reduction_output[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }


    //结果返回
    output_tool.UpdateData(r_output,b_output,q);
    {
        host_accessor temp_accessor{b_output};
        for(int i = 0; i < output_Size; i++){
            h_output[i] = temp_accessor[i];
        }
    }
    output.array2Tensor(h_output);

	

    // 内存释放
    
}

void dftfunc(const vector<std::complex<double>>& input, vector<std::complex<double>>& output) {
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);

    // 使用 for 循环初始化 vector，元素从 1 到 N
    for (int i = 0; i < N; ++i) {
        vec[i] = i;  // 赋值为 1 到 N
    }
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);

    // DFT 公式：X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    DFT_dft(input_tensor, output_tensor, vec_tensor);
    output_tensor.print();
}

int main() {
    // 定义一个输入信号（长度为8的复数序列）

    vector<std::complex<double>> input(N);
    
    // 初始化输入数据（可以是任何时间域信号）
    for (int i = 0; i < N; ++i) {
        input[i] = Complex(i, 0);  // 以复数形式填充数据，这里只是简单的填充数据
    }

    // 输出原始数据
    //std::cout << "原始数据（时间域）:" << std::endl;
    // for (const auto& val : input) {
    //     //std::cout << val << std::endl;
    // }

    // 计算离散傅里叶变换
    vector<Complex> output(N);
    dftfunc(input, output);

    // // 输出傅里叶变换后的数据（频域）
    // cout << "\n傅里叶变换后的数据（频域）:" << endl;
    // for (const auto& val : output) {
    //     cout << val << endl;
    // }

    return 0;
}
