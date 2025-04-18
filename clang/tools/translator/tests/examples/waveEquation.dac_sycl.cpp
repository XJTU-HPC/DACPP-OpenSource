#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// 网格参数
const int NX = 8;    // x方向网格数量
const int NY = 8;    // y方向网格数量
const double Lx = 10.0f; // x方向长度
const double Ly = 10.0f; // y方向长度
const double c = 1.0f;   // 波速
const int TIME_STEPS = 1000; // 时间步数
// 网格步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// CFL条件
const double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件





#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void waveEq(double* cur,double* prev,double* next,sycl::accessor<int, 1, sycl::access::mode::read_write> info_cur_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_prev_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_next_acc) 
{
    double dt = 0.5F * std::fmin(dx, dy) / c;
    double u_xx = (cur[2*info_cur_acc[1]+1] - 2.F * cur[1*info_cur_acc[1]+1] + cur[0*info_cur_acc[1]+1]) / (dx * dx);
    double u_yy = (cur[1*info_cur_acc[1]+2] - 2.F * cur[1*info_cur_acc[1]+1] + cur[1*info_cur_acc[1]+0]) / (dy * dy);
    next[0] = 2.F * cur[1*info_cur_acc[1]+1] - prev[0] + (c * c) * dt * dt * (u_xx + u_yy);
}


// 生成函数调用
void waveEqShell_waveEq(const dacpp::Tensor<double, 2> & matCur, const dacpp::Tensor<double, 2> & matPrev, dacpp::Tensor<double, 2> & matNext) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_matCur;
    info_matCur.dim = matCur.getDim();
    for(int i = 0; i < info_matCur.dim; i++) info_matCur.dimLength.push_back(matCur.getShape(i));
    // 数据信息初始化
    DataInfo info_matPrev;
    info_matPrev.dim = matPrev.getDim();
    for(int i = 0; i < info_matPrev.dim; i++) info_matPrev.dimLength.push_back(matPrev.getShape(i));
    // 数据信息初始化
    DataInfo info_matNext;
    info_matNext.dim = matNext.getDim();
    for(int i = 0; i < info_matNext.dim; i++) info_matNext.dimLength.push_back(matNext.getShape(i));
    // 规则分区算子初始化
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.setDimId(0);
    sp1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp1,info_matCur));

    // 规则分区算子初始化
    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.setDimId(1);
    sp2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp2,info_matCur));

    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matPrev));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matPrev));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops matCur_Ops;
    
    sp1.setDimId(0);
    matCur_Ops.push_back(sp1);

    sp2.setDimId(1);
    matCur_Ops.push_back(sp2);


    // 算子组初始化
    Dac_Ops matPrev_Ops;
    
    idx1.setDimId(0);
    matPrev_Ops.push_back(idx1);

    idx2.setDimId(1);
    matPrev_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops matNext_Ops;
    
    idx1.setDimId(0);
    matNext_Ops.push_back(idx1);

    idx2.setDimId(1);
    matNext_Ops.push_back(idx2);


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
    int matCur_Size = para_gene_tool.init_device_memory_size(info_matCur,matCur_Ops);

    //生成设备内存分配大小
    int matPrev_Size = para_gene_tool.init_device_memory_size(info_matPrev,matPrev_Ops);

    //生成设备内存分配大小
    int matNext_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_matNext);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_matNext,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matCur_Ops,matCur_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matPrev_Ops,matPrev_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,matNext_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(matCur_Ops);

    ops_s.push_back(matPrev_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[3][2] = {0};
    para_gene_tool.init_split_length_martix(3,2,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // 设备内存分配
    double *d_matCur=malloc_device<double>(matCur_Size,q);
    // 设备内存分配
    double *d_matPrev=malloc_device<double>(matPrev_Size,q);
    // 设备内存分配
    double *d_matNext=malloc_device<double>(matNext_Size,q);
    // 归约设备内存分配
    double *reduction_matNext = malloc_device<double>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<double> matCur_tool;
    double* r_matCur=(double*)malloc(sizeof(double)*matCur_Size);
    
    // 数据算子组初始化
    Dac_Ops matCur_ops;
    
    sp1.setDimId(0);
    sp1.setSplitLength(8);
    matCur_ops.push_back(sp1);
    sp2.setDimId(1);
    sp2.setSplitLength(8);
    matCur_ops.push_back(sp2);
    matCur_tool.init(info_matCur,matCur_ops);
    matCur_tool.Reconstruct(r_matCur,matCur);
	std::vector<int> info_partition_matCur=para_gene_tool.init_partition_data_shape(info_matCur,matCur_ops);
    sycl::buffer<int> info_partition_matCur_buffer(info_partition_matCur.data(), sycl::range<1>(info_partition_matCur.size()));
    // 数据重组
    DataReconstructor<double> matPrev_tool;
    double* r_matPrev=(double*)malloc(sizeof(double)*matPrev_Size);
    
    // 数据算子组初始化
    Dac_Ops matPrev_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    matPrev_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    matPrev_ops.push_back(idx2);
    matPrev_tool.init(info_matPrev,matPrev_ops);
    matPrev_tool.Reconstruct(r_matPrev,matPrev);
	std::vector<int> info_partition_matPrev=para_gene_tool.init_partition_data_shape(info_matPrev,matPrev_ops);
    sycl::buffer<int> info_partition_matPrev_buffer(info_partition_matPrev.data(), sycl::range<1>(info_partition_matPrev.size()));
    // 数据重组
    DataReconstructor<double> matNext_tool;
    double* r_matNext=(double*)malloc(sizeof(double)*matNext_Size);
    
    // 数据算子组初始化
    Dac_Ops matNext_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    matNext_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    matNext_ops.push_back(idx2);
    matNext_tool.init(info_matNext,matNext_ops);
    matNext_tool.Reconstruct(r_matNext,matNext);
	std::vector<int> info_partition_matNext=para_gene_tool.init_partition_data_shape(info_matNext,matNext_ops);
    sycl::buffer<int> info_partition_matNext_buffer(info_partition_matNext.data(), sycl::range<1>(info_partition_matNext.size()));
    
    // 数据移动
    q.memcpy(d_matCur,r_matCur,matCur_Size*sizeof(double)).wait();
    // 数据移动
    q.memcpy(d_matPrev,r_matPrev,matPrev_Size*sizeof(double)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_matCur_accessor = info_partition_matCur_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_matPrev_accessor = info_partition_matPrev_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_matNext_accessor = info_partition_matNext_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id+(0))%sp2.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 嵌入计算
			
            waveEq(d_matCur+(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),d_matPrev+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),d_matNext+(sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]),info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_matNext+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_matNext[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_matNext,reduction_matNext, Reduction_Size*sizeof(double)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_matNext, d_matNext, matNext_Size*sizeof(double)).wait();
    matNext_tool.UpdateData(r_matNext,matNext);

    // 内存释放
    
    sycl::free(d_matCur, q);
    sycl::free(d_matPrev, q);
    sycl::free(d_matNext, q);
}

int main() {

    
    // 初始化波场
    vector<double> u_prev(NX * NY, 0.0f); // 前一步
    vector<double> u_curr(NX * NY, 0.0f);  // 当前步
    vector<double> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如一个高斯脉冲
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
        }
    }

    for(int i = 0; i < 1; i++){
        for(int j = 0; j < NY; j++){
            std::cout << u_prev[i * NX + j] << " " ;
        }
        std::cout << std::endl;
    }

  

    //std::vector<int> shape_u_curr = {8, 8};
    dacpp::Tensor<double, 2> u_curr_tensor({8, 8}, u_curr);

    std::vector<double> u_prev_middle_points;
    for (int i = 1; i <= 6; i++) {
        std::vector<double> row;
        for (int j = 1; j <= 6; j++) {  
            u_prev_middle_points.push_back(static_cast<double>(u_prev[i*NY+j]));  
        }
        
    }
    //std::vector<int> shape2= {6, 6};
    dacpp::Tensor<double, 2> u_prev_middle_tensor({6, 6}, u_prev_middle_points);
    //u_next取点
    std::vector<double> u_next_middle_points;
    for (int i = 1; i <= 6; i++) {
        std::vector<double> row;
        for (int j = 1; j <= 6; j++) {  
            u_next_middle_points.push_back(static_cast<double>(u_next[i*NY+j]));  
        }
        
    }
    
    for(int i = 0;i < TIME_STEPS; i++) {
        
        //std::vector<int> shape1= {6, 6};
        dacpp::Tensor<double, 2> u_next_middle_tensor({6, 6}, u_next_middle_points);


        
        waveEqShell_waveEq(u_curr_tensor, u_prev_middle_tensor, u_next_middle_tensor);

       // u_next_middle_tensor.print();

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                u_prev_middle_tensor[i-1][j-1]=u_curr_tensor[i][j];
            }
        }

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                u_curr_tensor[i][j]=u_next_middle_tensor[i-1][j-1];
            }
        }

        // 处理边界条件（绝热边界：导数为零）


        for (int i = 0; i < NX; ++i) {       
            u_curr_tensor[i][NY-1]=0;
            u_curr_tensor[i][0]=0;
        }
        for (int j = 0; j < NY; ++j) {
            u_curr_tensor[NX - 1][j]=0;
            u_curr_tensor[0][j]=0;
             // 底部边界
        }

        
    }
    u_curr_tensor.print();


    
    // 输出最终结果的某些值作为示例
    //cout << "Final wave state at center: " << u_curr[(NX/2)*NY + (NY/2)] << "\n";
    
    return 0;
}
