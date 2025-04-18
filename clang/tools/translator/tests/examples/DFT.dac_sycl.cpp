#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
using Complex = complex<double>;  // 复数类型别名
const int N = 8;


void DFT(const dacpp::Tensor<Complex, 1> & input, dacpp::Tensor<Complex, 1> & output, const dacpp::Tensor<int, 1> & vec);




// 离散傅里叶变换（DFT）
void dftfunc(const vector<Complex>& input, vector<Complex>& output) {
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);

    // 使用 for 循环初始化 vector，元素从 1 到 N
    for (int i = 0; i < N; ++i) {
        vec[i] = i;  // 赋值为 1 到 N
    }
    dacpp::Tensor<int, 1> vec_tensor(vec);
    dacpp::Tensor<Complex, 1> input_tensor(input);
    dacpp::Tensor<Complex, 1> output_tensor(output);

    // DFT 公式：X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    DFT(input_tensor, output_tensor, vec_tensor);
    output_tensor.print();
}

#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void dft(Complex* input, Complex* output, int* vec) 
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
void DFT(const dacpp::Tensor<Complex, 1> & input, dacpp::Tensor<Complex, 1> & output, const dacpp::Tensor<int, 1> & vec) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
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
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_output,Reduction_Ops);

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
    
    // 设备内存分配
    Complex *d_input=malloc_device<Complex>(input_Size,q);
    // 设备内存分配
    Complex *d_output=malloc_device<Complex>(output_Size,q);
    // 归约设备内存分配
    Complex *reduction_output = malloc_device<Complex>(Reduction_Size,q);
    // 设备内存分配
    int *d_vec=malloc_device<int>(vec_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<Complex> input_tool;
    Complex* r_input=(Complex*)malloc(sizeof(Complex)*input_Size);
    
    // 数据算子组初始化
    Dac_Ops input_ops;
    
    input_tool.init(info_input,input_ops);
    input_tool.Reconstruct(r_input,input);
    // 数据重组
    DataReconstructor<Complex> output_tool;
    Complex* r_output=(Complex*)malloc(sizeof(Complex)*output_Size);
    
    // 数据算子组初始化
    Dac_Ops output_ops;
    
    i.setDimId(0);
    i.setSplitLength(8);
    output_ops.push_back(i);
    output_tool.init(info_output,output_ops);
    output_tool.Reconstruct(r_output,output);
    // 数据重组
    DataReconstructor<int> vec_tool;
    int* r_vec=(int*)malloc(sizeof(int)*vec_Size);
    
    // 数据算子组初始化
    Dac_Ops vec_ops;
    
    i.setDimId(0);
    i.setSplitLength(8);
    vec_ops.push_back(i);
    vec_tool.init(info_vec,vec_ops);
    vec_tool.Reconstruct(r_vec,vec);
    
    // 数据移动
    q.memcpy(d_input,r_input,input_Size*sizeof(Complex)).wait();
    // 数据移动
    q.memcpy(d_vec,r_vec,vec_Size*sizeof(int)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i_=(item_id+(0))%i.split_size;
            // 嵌入计算
			
            dft(d_input,d_output+(i_*SplitLength[1][0]),d_vec+(i_*SplitLength[2][0]));
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_output+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_output[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_output,reduction_output, Reduction_Size*sizeof(Complex)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_output, d_output, output_Size*sizeof(Complex)).wait();
    output_tool.UpdateData(r_output,output);

    // 内存释放
    
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_vec, q);
}

int main() {
    // 定义一个输入信号（长度为8的复数序列）

    vector<Complex> input(N);
    
    // 初始化输入数据（可以是任何时间域信号）
    for (int i = 0; i < N; ++i) {
        input[i] = Complex(i, 0);  // 以复数形式填充数据，这里只是简单的填充数据
    }

    // 输出原始数据
    std::cout << "原始数据（时间域）:" << std::endl;
    for (const auto& val : input) {
        std::cout << val << std::endl;
    }

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
