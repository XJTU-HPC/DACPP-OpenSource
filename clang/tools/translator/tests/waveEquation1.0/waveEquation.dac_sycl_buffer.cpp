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
const int TIME_STEPS = 10; // 时间步数
// 网格步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// CFL条件
const double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件





#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"

using namespace sycl;

void waveEq(const double* cur,const double* prev,double* next,int cur_0,int cur_1,int prev_0,int prev_1,int next_0,int next_1,int cur_0_shape,int cur_1_shape,int prev_0_shape,int prev_1_shape,int next_0_shape,int next_1_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_cur_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_prev_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_next_acc) {
    double dt = 0.5F * std::fmin(dx, dy) / c;
    double u_xx = (cur[(2 + cur_0) * cur_1_shape + (1 + cur_1)] - 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] + cur[(0 + cur_0) * cur_1_shape + (1 + cur_1)]) / (dx * dx);
    double u_yy = (cur[(1 + cur_0) * cur_1_shape + (2 + cur_1)] - 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] + cur[(1 + cur_0) * cur_1_shape + (0 + cur_1)]) / (dy * dy);
    next[(0 + next_0) * next_1_shape + (0 + next_1)] = 2.F * cur[(1 + cur_0) * cur_1_shape + (1 + cur_1)] - prev[(0 + prev_0) * prev_1_shape + (0 + prev_1)] + (c * c) * dt * dt * (u_xx + u_yy);
}

struct __dacpp_ctx_waveEqShell_waveEq {
    sycl::queue dacpp_q{};
    ParameterGeneration para_gene_tool;
    int Item_Size = 0;
    int dim_x = 1;
    int dim_y = 1;
    int local_x = 1;
    int local_y = 1;
    int global_x = 1;
    int global_y = 1;
    RegularSlice sp1;
    RegularSlice sp2;
    Index idx1;
    Index idx2;
    Dac_Ops In_Ops;
    Dac_Ops Out_Ops;
    DataInfo info_matCur;
    std::vector<int> info_matCur_Shape;
    Dac_Ops matCur_Ops;
    std::vector<double> h_matCur;
    std::unique_ptr<sycl::buffer<double, 1>> r_matCur;
    std::vector<int> info_partition_matCur;
    std::unique_ptr<sycl::buffer<int, 1>> info_partition_matCur_buffer;
    DataInfo info_matPrev;
    std::vector<int> info_matPrev_Shape;
    Dac_Ops matPrev_Ops;
    std::vector<double> h_matPrev;
    std::unique_ptr<sycl::buffer<double, 1>> r_matPrev;
    std::vector<int> info_partition_matPrev;
    std::unique_ptr<sycl::buffer<int, 1>> info_partition_matPrev_buffer;
    DataInfo info_matNext;
    std::vector<int> info_matNext_Shape;
    Dac_Ops matNext_Ops;
    std::vector<double> h_matNext;
    std::unique_ptr<sycl::buffer<double, 1>> r_matNext;
    std::vector<int> info_partition_matNext;
    std::unique_ptr<sycl::buffer<int, 1>> info_partition_matNext_buffer;
};

void __dacpp_init_waveEqShell_waveEq(__dacpp_ctx_waveEqShell_waveEq& ctx, dacpp::Matrix<double> & matCur, dacpp::Matrix<double> & matPrev, dacpp::Matrix<double> & matNext) {
    ctx.dacpp_q = sycl::queue(sycl::default_selector_v);
    ctx.info_matCur = DataInfo{};
    ctx.info_matCur.dim = matCur.getDim();
    ctx.info_matCur.dimLength.clear();
    ctx.info_matCur_Shape.assign(ctx.info_matCur.dim, 0);
    for (int dimIdx = 0; dimIdx < ctx.info_matCur.dim; ++dimIdx) {
        ctx.info_matCur.dimLength.push_back(matCur.getShape(dimIdx));
        ctx.info_matCur_Shape[dimIdx] = matCur.getShape(dimIdx);
    }
    ctx.info_matPrev = DataInfo{};
    ctx.info_matPrev.dim = matPrev.getDim();
    ctx.info_matPrev.dimLength.clear();
    ctx.info_matPrev_Shape.assign(ctx.info_matPrev.dim, 0);
    for (int dimIdx = 0; dimIdx < ctx.info_matPrev.dim; ++dimIdx) {
        ctx.info_matPrev.dimLength.push_back(matPrev.getShape(dimIdx));
        ctx.info_matPrev_Shape[dimIdx] = matPrev.getShape(dimIdx);
    }
    ctx.info_matNext = DataInfo{};
    ctx.info_matNext.dim = matNext.getDim();
    ctx.info_matNext.dimLength.clear();
    ctx.info_matNext_Shape.assign(ctx.info_matNext.dim, 0);
    for (int dimIdx = 0; dimIdx < ctx.info_matNext.dim; ++dimIdx) {
        ctx.info_matNext.dimLength.push_back(matNext.getShape(dimIdx));
        ctx.info_matNext_Shape[dimIdx] = matNext.getShape(dimIdx);
    }
    ctx.sp1 = RegularSlice("sp1", 3, 1);
    ctx.sp1.setDimId(0);
    ctx.sp1.SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx.sp1, ctx.info_matCur));
    ctx.sp2 = RegularSlice("sp2", 3, 1);
    ctx.sp2.setDimId(1);
    ctx.sp2.SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx.sp2, ctx.info_matCur));
    ctx.idx1 = Index("idx1");
    ctx.idx1.setDimId(0);
    ctx.idx1.SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx.idx1, ctx.info_matPrev));
    ctx.idx2 = Index("idx2");
    ctx.idx2.setDimId(1);
    ctx.idx2.SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx.idx2, ctx.info_matPrev));
    ctx.matCur_Ops.clear();
    ctx.sp1.setDimId(0);
    ctx.matCur_Ops.push_back(ctx.sp1);
    ctx.sp2.setDimId(1);
    ctx.matCur_Ops.push_back(ctx.sp2);
    ctx.matPrev_Ops.clear();
    ctx.idx1.setDimId(0);
    ctx.matPrev_Ops.push_back(ctx.idx1);
    ctx.idx2.setDimId(1);
    ctx.matPrev_Ops.push_back(ctx.idx2);
    ctx.matNext_Ops.clear();
    ctx.idx1.setDimId(0);
    ctx.matNext_Ops.push_back(ctx.idx1);
    ctx.idx2.setDimId(1);
    ctx.matNext_Ops.push_back(ctx.idx2);
    ctx.In_Ops.clear();
    ctx.Out_Ops.clear();
    ctx.sp1.setDimId(0);
    ctx.In_Ops.push_back(ctx.sp1);
    ctx.sp2.setDimId(1);
    ctx.In_Ops.push_back(ctx.sp2);
    ctx.idx1.setDimId(0);
    ctx.Out_Ops.push_back(ctx.idx1);
    ctx.idx2.setDimId(1);
    ctx.Out_Ops.push_back(ctx.idx2);
    ctx.Item_Size = ctx.para_gene_tool.init_work_item_size(ctx.In_Ops);
    sycl::device device = ctx.dacpp_q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    ctx.dim_x = static_cast<int>(sycl::ceil(sycl::sqrt(static_cast<float>(ctx.Item_Size))));
    ctx.dim_y = static_cast<int>(sycl::ceil(static_cast<float>(ctx.Item_Size) / ctx.dim_x));
    ctx.local_x = std::min(16, static_cast<int>(max_sizes[0]));
    ctx.local_y = std::min(16, static_cast<int>(max_sizes[1]));
    ctx.global_x = ((ctx.dim_x + ctx.local_x - 1) / ctx.local_x) * ctx.local_x;
    ctx.global_y = ((ctx.dim_y + ctx.local_y - 1) / ctx.local_y) * ctx.local_y;
    ctx.h_matCur.clear();
    matCur.tensor2Array(ctx.h_matCur);
    ctx.r_matCur = std::make_unique<sycl::buffer<double, 1>>(ctx.h_matCur.data(), sycl::range<1>(matCur.getSize()));
    ctx.info_partition_matCur = ctx.para_gene_tool.init_partition_data_shape(ctx.info_matCur, ctx.matCur_Ops);
    ctx.info_partition_matCur_buffer = std::make_unique<sycl::buffer<int, 1>>(ctx.info_partition_matCur.data(), sycl::range<1>(ctx.info_partition_matCur.size()));
    ctx.h_matPrev.clear();
    matPrev.tensor2Array(ctx.h_matPrev);
    ctx.r_matPrev = std::make_unique<sycl::buffer<double, 1>>(ctx.h_matPrev.data(), sycl::range<1>(matPrev.getSize()));
    ctx.info_partition_matPrev = ctx.para_gene_tool.init_partition_data_shape(ctx.info_matPrev, ctx.matPrev_Ops);
    ctx.info_partition_matPrev_buffer = std::make_unique<sycl::buffer<int, 1>>(ctx.info_partition_matPrev.data(), sycl::range<1>(ctx.info_partition_matPrev.size()));
    ctx.h_matNext.clear();
    ctx.r_matNext = std::make_unique<sycl::buffer<double, 1>>(sycl::range<1>(matNext.getSize()));
    ctx.info_partition_matNext = ctx.para_gene_tool.init_partition_data_shape(ctx.info_matNext, ctx.matNext_Ops);
    ctx.info_partition_matNext_buffer = std::make_unique<sycl::buffer<int, 1>>(ctx.info_partition_matNext.data(), sycl::range<1>(ctx.info_partition_matNext.size()));
}

void __dacpp_submit_waveEqShell_waveEq(__dacpp_ctx_waveEqShell_waveEq& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto& sp1 = ctx.sp1;
    auto& sp2 = ctx.sp2;
    auto& idx1 = ctx.idx1;
    auto& idx2 = ctx.idx2;
    auto& r_matCur = *ctx.r_matCur;
    auto& r_matPrev = *ctx.r_matPrev;
    auto* r_matNext = ctx.r_matNext.get();
    auto& info_partition_matCur_buffer = *ctx.info_partition_matCur_buffer;
    auto& info_partition_matPrev_buffer = *ctx.info_partition_matPrev_buffer;
    auto& info_partition_matNext_buffer = *ctx.info_partition_matNext_buffer;
    auto* info_matCur_Shape = ctx.info_matCur_Shape.data();
    auto* info_matPrev_Shape = ctx.info_matPrev_Shape.data();
    auto* info_matNext_Shape = ctx.info_matNext_Shape.data();
    const int Item_Size = ctx.Item_Size;
    sycl::range<2> local(ctx.local_x, ctx.local_y);
    sycl::range<2> global(ctx.global_x, ctx.global_y);
    dacpp_q.submit([&](handler &h) {

        accessor<double, 1, access::mode::read> acc_matCur(r_matCur, h);
        r_matCur.set_final_data(nullptr);
        
        accessor<double, 1, access::mode::read> acc_matPrev(r_matPrev, h);
        r_matPrev.set_final_data(nullptr);
        
        accessor<double, 1, sycl::access::mode::discard_write> acc_matNext(*r_matNext, h);
        auto info_partition_matCur_accessor = info_partition_matCur_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matPrev_accessor = info_partition_matPrev_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matNext_accessor = info_partition_matNext_buffer.get_access<sycl::access::mode::read>(h);        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化

            const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id+(0))%sp2.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;            // 获得划分数据单元左上角（第一个元素）的位置

			const auto matCur_0 = sp1_ * sp1.stride;
			const auto matCur_1 = sp2_ * sp2.stride;
			const auto matPrev_0 = idx1_ * idx1.stride;
			const auto matPrev_1 = idx2_ * idx2.stride;
			const auto matNext_0 = idx1_ * idx1.stride;
			const auto matNext_1 = idx2_ * idx2.stride;            // 获得accessor指针

            auto* d_matCur = acc_matCur.get_multi_ptr<access::decorated::no>().get();
            auto* d_matPrev = acc_matPrev.get_multi_ptr<access::decorated::no>().get();
            auto* d_matNext = acc_matNext.get_multi_ptr<access::decorated::no>().get();            // 嵌入计算

            waveEq(d_matCur,d_matPrev,d_matNext,matCur_0,matCur_1,matPrev_0,matPrev_1,matNext_0,matNext_1,info_matCur_Shape[0],info_matCur_Shape[1],info_matPrev_Shape[0],info_matPrev_Shape[1],info_matNext_Shape[0],info_matNext_Shape[1],info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);        });
    });
}

void __dacpp_submit_region_waveEqShell_waveEq_stmt_0(__dacpp_ctx_waveEqShell_waveEq& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto* r_matCur = ctx.r_matCur.get();
    auto* r_matPrev = ctx.r_matPrev.get();
    auto* r_matNext = ctx.r_matNext.get();
    auto* info_matCur_Shape = ctx.info_matCur_Shape.data();
    auto* info_matPrev_Shape = ctx.info_matPrev_Shape.data();
    auto* info_matNext_Shape = ctx.info_matNext_Shape.data();
{
    int __iL = (1);
    int __iR = (NX-2);
    int __iN = __iR - __iL +1;
    int __jL = (1);
    int __jR = (NY-2);
    int __jN = __jR - __jL + 1;
    dacpp_q.submit([&](sycl::handler& h){
    auto acc_matCur = r_matCur->get_access<sycl::access::mode::read_write>(h);
    auto acc_matPrev = r_matPrev->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matCur = acc_matCur.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_matPrev = acc_matPrev.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int j = __jL + idx[1];
      {
                d_matPrev[(i-1) * (info_matPrev_Shape[0]) + (j-1)]=d_matCur[(i) * (info_matCur_Shape[0]) + (j)];
            }
    });
  });
}
}

void __dacpp_submit_region_waveEqShell_waveEq_stmt_1(__dacpp_ctx_waveEqShell_waveEq& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto* r_matCur = ctx.r_matCur.get();
    auto* r_matPrev = ctx.r_matPrev.get();
    auto* r_matNext = ctx.r_matNext.get();
    auto* info_matCur_Shape = ctx.info_matCur_Shape.data();
    auto* info_matPrev_Shape = ctx.info_matPrev_Shape.data();
    auto* info_matNext_Shape = ctx.info_matNext_Shape.data();
{
    int __iL = (1);
    int __iR = (NX-2);
    int __iN = __iR - __iL +1;
    int __jL = (1);
    int __jR = (NY-2);
    int __jN = __jR - __jL + 1;
    dacpp_q.submit([&](sycl::handler& h){
    auto acc_matCur = r_matCur->get_access<sycl::access::mode::read_write>(h);
    auto acc_matNext = r_matNext->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matCur = acc_matCur.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_matNext = acc_matNext.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int j = __jL + idx[1];
      {
                d_matCur[(i) * (info_matCur_Shape[0]) + (j)]=d_matNext[(i-1) * (info_matPrev_Shape[0]) + (j-1)];
            }
    });
  });
}
}

void __dacpp_submit_region_waveEqShell_waveEq_stmt_2(__dacpp_ctx_waveEqShell_waveEq& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto* r_matCur = ctx.r_matCur.get();
    auto* r_matPrev = ctx.r_matPrev.get();
    auto* r_matNext = ctx.r_matNext.get();
    auto* info_matCur_Shape = ctx.info_matCur_Shape.data();
    auto* info_matPrev_Shape = ctx.info_matPrev_Shape.data();
    auto* info_matNext_Shape = ctx.info_matNext_Shape.data();
{
  int __L = (0);
  int __R = (NX-1);
  int __N = __R - __L +1;
  dacpp_q.submit([&](sycl::handler& h){
    auto acc_matCur = r_matCur->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_matCur = acc_matCur.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __L + idx[0];
      {       
            d_matCur[(i) * (info_matCur_Shape[0]) + (NY-1)]=0;
            d_matCur[(i) * (info_matCur_Shape[0]) + (0)]=0;
        }
    });
  });
}
}

void __dacpp_submit_region_waveEqShell_waveEq_stmt_3(__dacpp_ctx_waveEqShell_waveEq& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto* r_matCur = ctx.r_matCur.get();
    auto* r_matPrev = ctx.r_matPrev.get();
    auto* r_matNext = ctx.r_matNext.get();
    auto* info_matCur_Shape = ctx.info_matCur_Shape.data();
    auto* info_matPrev_Shape = ctx.info_matPrev_Shape.data();
    auto* info_matNext_Shape = ctx.info_matNext_Shape.data();
{
  int __L = (0);
  int __R = (NY-1);
  int __N = __R - __L +1;
  dacpp_q.submit([&](sycl::handler& h){
    auto acc_matCur = r_matCur->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_matCur = acc_matCur.template get_multi_ptr<sycl::access::decorated::no>().get();
      int j = __L + idx[0];
      {
            d_matCur[(NX - 1) * (info_matCur_Shape[0]) + (j)]=0;
            d_matCur[(0) * (info_matCur_Shape[0]) + (j)]=0;
             // 底部边界
        }
    });
  });
}
}

void __dacpp_sync_waveEqShell_waveEq(__dacpp_ctx_waveEqShell_waveEq& ctx, dacpp::Matrix<double> & matCur, dacpp::Matrix<double> & matPrev, dacpp::Matrix<double> & matNext) {
    using namespace sycl;
    ctx.dacpp_q.wait();
    ctx.h_matCur.resize(matCur.getSize());
    {
        host_accessor acc(*ctx.r_matCur);
        for (std::size_t idx = 0; idx < ctx.h_matCur.size(); ++idx) {
            ctx.h_matCur[idx] = acc[idx];
        }
    }
    matCur.array2Tensor(ctx.h_matCur);
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

    // for(int i = 0; i < 1; i++){
    //     for(int j = 0; j < NY; j++){
    //         //std::cout << u_prev[i * NX + j] << " " ;
    //     }
    //     //std::cout << std::endl;
    // }

    dacpp::Matrix<double> matCur({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matPrev = u_prev_tensor[{1,NX-1}][{1,NY-1}];
    dacpp::Matrix<double> matNext = u_next_tensor[{1,NX-1}][{1,NY-1}];
    
        __dacpp_ctx_waveEqShell_waveEq __dacpp_ctx_waveEqShell_waveEq_0;
    __dacpp_init_waveEqShell_waveEq(__dacpp_ctx_waveEqShell_waveEq_0, matCur, matPrev, matNext);
for(int i = 0;i < TIME_STEPS; i++) {
        __dacpp_submit_waveEqShell_waveEq(__dacpp_ctx_waveEqShell_waveEq_0);
        __dacpp_submit_region_waveEqShell_waveEq_stmt_0(__dacpp_ctx_waveEqShell_waveEq_0);

        __dacpp_submit_region_waveEqShell_waveEq_stmt_1(__dacpp_ctx_waveEqShell_waveEq_0);
        // 处理边界条件（绝热边界：导数为零）
        __dacpp_submit_region_waveEqShell_waveEq_stmt_2(__dacpp_ctx_waveEqShell_waveEq_0);
        __dacpp_submit_region_waveEqShell_waveEq_stmt_3(__dacpp_ctx_waveEqShell_waveEq_0);
    }
    __dacpp_sync_waveEqShell_waveEq(__dacpp_ctx_waveEqShell_waveEq_0, matCur, matPrev, matNext);

    //
    matCur.print(); 
    return 0;
}

// #include <iostream>
// #include <vector>
// #include <cmath>
// #include "ReconTensor.h"

// using namespace std;

// namespace dacpp {
//     typedef std::vector<std::any> list;
// }
// // 网格参数
// const int NX = 8;    // x方向网格数量
// const int NY = 8;    // y方向网格数量
// const double Lx = 10.0f; // x方向长度
// const double Ly = 10.0f; // y方向长度
// const double c = 1.0f;   // 波速
// const int TIME_STEPS = 1000; // 时间步数
// // 网格步长
// const double dx = Lx / (NX - 1);
// const double dy = Ly / (NY - 1);

// // CFL条件
// const double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件

// shell dacpp::list waveEqShell(const dacpp::Matrix<double>& matCur, 
//                                 const dacpp::Matrix<double>& matPrev, 
//                                 dacpp::Matrix<double>& matNext) {
//     dacpp::split sp1(3, 1), sp2(3, 1);
//     dacpp::index idx1, idx2;
//     binding(sp1, idx1);
//     binding(sp2, idx2);
//     dacpp::list dataList{matCur[sp1][sp2], matPrev[idx1][idx2], matNext[idx1][idx2]};
//     return dataList;
// }

// calc void waveEq(dacpp::Matrix<double>& cur, double* prev, double* next) {
//     double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件
//     double u_xx = (cur[2][1] - 2.0f * cur[1][1] + cur[0][1])/ (dx * dx);
//     double u_yy = (cur[1][2] - 2.0f * cur[1][1] + cur[1][0])/ (dy * dy);
//     next[0]=2.0f*cur[1][1]-prev[0]+(c * c)*dt*dt*(u_xx+u_yy);
// }

// int main() {
//     // 初始化波场
//     vector<double> u_prev(NX * NY, 0.0f); // 前一步
//     vector<double> u_curr(NX * NY, 0.0f);  // 当前步
//     vector<double> u_next(NX * NY, 0.0f);  // 当前步

//     // 初始条件：例如一个高斯脉冲
//     int cx = NX / 2;
//     int cy = NY / 2;
//     double sigma = 0.5f;
//     for(int i = 0; i < NX; ++i) {
//         for(int j = 0; j < NY; ++j) {
//             double x = i * dx;
//             double y = j * dy;
//             u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
//         }
//     }

//     // for(int i = 0; i < 1; i++){
//     //     for(int j = 0; j < NY; j++){
//     //         std::cout << u_prev[i * NX + j] << " " ;
//     //     }
//     //     std::cout << std::endl;
//     // }

//     dacpp::Matrix<double> matCur({NX, NY}, u_curr);
//     dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
//     dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
//     dacpp::Matrix<double> matPrev = u_prev_tensor[{1,7}][{1,7}];
//     dacpp::Matrix<double> matNext = u_next_tensor[{1,7}][{1,7}];
//         // 使用 dac_for 包裹计算表达式和指针操作代码块
//     dacpp::dac_for(TIME_STEPS, [&](int STEPS) {
//         for(int i = 0; i < STEPS; i++){
//             waveEqShell(matCur, matPrev, matNext) <-> waveEq;
//             dacpp::swap(matPrev, matCur[{1,7}][{1,7}]);
//             dacpp::swap(matCur[{1,7}][{1,7}], matNext);
//         }
//     });
//     for (int i = 1; i <= NX-2; i++) {
//         for(int j = 1; j <=NY-2; j++){
//             matPrev[i-1][j-1]=matCur[i][j];
//         }
//     }
//     for (int i = 1; i <= NX-2; i++) {
//         for(int j = 1; j <=NY-2; j++){
//             matCur[i][j]=matNext[i-1][j-1];
//         }
//     }
//     // 处理边界条件（绝热边界：导数为零）
//     for (int i = 0; i < NX; ++i) {       
//         matCur[i][NY-1]=0;
//         matCur[i][0]=0;
//     }
//     for (int j = 0; j < NY; ++j) {
//         matCur[NX - 1][j]=0;
//         matCur[0][j]=0;
//             // 底部边界
//     }    
//    matCur.print(); 
//     return 0;
// }

