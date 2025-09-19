#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"

using namespace std;
namespace dacpp {
    typedef std::vector<std::any> list;
}

// 网格参数
const int NX = 32;           // x方向网格数量
const int NY = 32;           // y方向网格数量
const double Lx = 10.0f;       // x方向长度
const double Ly = 10.0f;       // y方向长度
const double alpha = 0.01f;    // 热扩散系数
const int TIME_STEPS = 1000;  // 时间步数
// 空间步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// 稳定性条件
const double dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
const double delta_t = 0.4f * dt_stability; // 选择一个更严格的时间步长以确保稳定性





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void stencil(double* mat,double* out,sycl::accessor<int, 1, sycl::access::mode::read_write> info_mat_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_out_acc) 
{
    out[0] = mat[1*info_mat_acc[1]+1] + alpha * delta_t * (((mat[2*info_mat_acc[1]+1] - 2.F * mat[1*info_mat_acc[1]+1] + mat[0*info_mat_acc[1]+1]) / (dx * dx)) + ((mat[1*info_mat_acc[1]+2] - 2.F * mat[1*info_mat_acc[1]+1] + mat[1*info_mat_acc[1]+0]) / (dy * dy)));
}


// 生成函数调用
void stencilShell_stencil(const dacpp::Matrix<double> & matIn, dacpp::Matrix<double> & matOut) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_matIn;
    info_matIn.dim = matIn.getDim();
    for(int i = 0; i < info_matIn.dim; i++) info_matIn.dimLength.push_back(matIn.getShape(i));
	
    // 数据信息初始化
    DataInfo info_matOut;
    info_matOut.dim = matOut.getDim();
    for(int i = 0; i < info_matOut.dim; i++) info_matOut.dimLength.push_back(matOut.getShape(i));
	
    // 规则分区算子初始化
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.setDimId(0);
    sp1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp1,info_matIn));

    // 规则分区算子初始化
    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.setDimId(1);
    sp2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp2,info_matIn));

    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matOut));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matOut));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops matIn_Ops;
    
    sp1.setDimId(0);
    matIn_Ops.push_back(sp1);

    sp2.setDimId(1);
    matIn_Ops.push_back(sp2);


    // 算子组初始化
    Dac_Ops matOut_Ops;
    
    idx1.setDimId(0);
    matOut_Ops.push_back(idx1);

    idx2.setDimId(1);
    matOut_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    sp1.setDimId(0);
    In_Ops.push_back(sp1);

    sp2.setDimId(1);
    In_Ops.push_back(sp2);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);

    idx2.setDimId(1);
    Reduction_Ops.push_back(idx2);


	
    //生成设备内存分配大小
    int matIn_Size = para_gene_tool.init_device_memory_size(info_matIn,matIn_Ops);

    //生成设备内存分配大小
    int matOut_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_matOut);

    //生成设备内存分配大小
    int matOutReduction_Size = para_gene_tool.init_device_memory_size(info_matOut,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matIn_Ops,matIn_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,matOut_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(matIn_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[2][2] = {0};
    para_gene_tool.init_split_length_martix(2,2,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // Buffer设备内存分配
    buffer <double> b_matIn{matIn_Size};

    // Buffer设备内存分配
    buffer <double> b_matOut{matOut_Size};

    // 规约Buffer设备内存分配
    std::vector<sycl::buffer<double, 1>> b_reduction_matOut(matOutReduction_Size, buffer<double, 1>{1});
    for(int i = 0; i < matOutReduction_Size; i++){
        host_accessor temp_accessor{b_reduction_matOut[i]};
        temp_accessor[0] = 0;
    }
    // 数据关联计算
    
	
    // 数据移动
    double* h_matIn = (double*)malloc(matIn.getSize()*sizeof(double));
    matIn.tensor2Array(h_matIn);
    {
        host_accessor temp_accessor{b_matIn};
        for(int i = 0; i < matIn_Size; i++){
            temp_accessor[i] = h_matIn[i];
        }
    }

    // 数据移动
    double* h_matOut = (double*)malloc(matOut.getSize()*sizeof(double));

    // 数据重组
    DataReconstructor<double> matIn_tool;
    
    // 数据算子组初始化
    Dac_Ops matIn_ops;
    
    sp1.setDimId(0);
    matIn_ops.push_back(sp1);
    sp2.setDimId(1);
    matIn_ops.push_back(sp2);

    matIn_tool.init(info_matIn,matIn_ops);
    buffer<double> r_matIn{matIn_Size};
    matIn_tool.Reconstruct(r_matIn,b_matIn,q);
	std::vector<int> info_partition_matIn=para_gene_tool.init_partition_data_shape(info_matIn,matIn_ops);
    sycl::buffer<int> info_partition_matIn_buffer(info_partition_matIn.data(), sycl::range<1>(info_partition_matIn.size()));

    // 数据重组
    DataReconstructor<double> matOut_tool;
    
    // 数据算子组初始化
    Dac_Ops matOut_ops;
    
    idx1.setDimId(0);
    matOut_ops.push_back(idx1);
    idx2.setDimId(1);
    matOut_ops.push_back(idx2);

    matOut_tool.init(info_matOut,matOut_ops);
    buffer<double> r_matOut{matOut_Size};
    matOut_tool.Reconstruct(r_matOut,b_matOut,q);
	std::vector<int> info_partition_matOut=para_gene_tool.init_partition_data_shape(info_matOut,matOut_ops);
    sycl::buffer<int> info_partition_matOut_buffer(info_partition_matOut.data(), sycl::range<1>(info_partition_matOut.size()));
    
	
	
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

    //队列提交命令组
    q.submit([&](handler &h) {
    
        accessor acc_matIn{r_matIn, h};
        accessor acc_matOut{r_matOut, h};
    
        auto info_partition_matIn_accessor = info_partition_matIn_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_matOut_accessor = info_partition_matOut_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			
            const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id+(0))%sp2.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 获得accessor指针
            
            auto* d_matIn = acc_matIn.get_multi_ptr<access::decorated::no>().get();
            auto* d_matOut = acc_matOut.get_multi_ptr<access::decorated::no>().get();
            // 嵌入计算
			
            stencil(d_matIn+(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),d_matOut+(sp1_*SplitLength[1][0]+sp2_*SplitLength[1][1]),info_partition_matIn_accessor,info_partition_matOut_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<matOutReduction_Size;i++) {
            q.submit([&](handler &h) {
                accessor d_matOut{r_matOut, h};
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(b_reduction_matOut[i], h, 
                sycl::plus<>(),property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_matOut[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_matOut};
            for(int i = 0; i < matOutReduction_Size; i++){
                host_accessor temp_accessor{b_reduction_matOut[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }


    //结果返回
    matOut_tool.UpdateData(r_matOut,b_matOut,q);
    {
        host_accessor temp_accessor{b_matOut};
        for(int i = 0; i < matOut_Size; i++){
            h_matOut[i] = temp_accessor[i];
        }
    }
    matOut.array2Tensor(h_matOut);

	

    // 内存释放
    
}

int main() {


    //cout << "Grid size: " << NX << "x" << NY << "\n";
    //cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // 初始化温度场
    vector<double> u_prev(NX * NY, 0.0f); // 前一步（在热传导方程中，只有当前步和上一步）
    vector<double> u_curr(NX * NY, 0.0f);  // 当前步
    vector<double> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如，中心有一个高斯分布的热源
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            // 高斯分布
            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }

    //std::vector<int> shape = {32, 32};
    dacpp::Matrix<double> u_curr_tensor({NX, NY}, u_curr);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);

    for(int i=0;i<TIME_STEPS;i++) {
        dacpp::Matrix<double> middle_tensor = u_next_tensor[{1,NX-1}][{1,NY-1}];
        stencilShell_stencil(u_curr_tensor, middle_tensor);

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                double* data = new double[1];
                u_curr_tensor[i][j]=middle_tensor[i-1][j-1];
            }
        }

        // 处理边界条件（绝热边界：导数为零）
        for (int j = 0; j < NY; ++j) {
            //double* data = new double[1];
            u_curr_tensor[0][j]=u_curr_tensor[1][j];              // 顶部边界
            u_curr_tensor[NX - 1][j]=u_curr_tensor[NX-2][j];
             // 底部边界
        }
        for (int i = 0; i < NX; ++i) {
            //double* data = new double[1];
            u_curr_tensor[i][0]=u_curr_tensor[i][1];              // 顶部边界
            u_curr_tensor[i][NY-1]=u_curr_tensor[i][NY-2];
        }
        
    }
    u_curr_tensor[0].print();


    // 输出最终结果的某些值作为示例
    //cout << "Final temperature at center: " << vec2D[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
