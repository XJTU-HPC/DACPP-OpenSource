#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
// #define DACPP_TRANSLATE_MODE 1

using namespace std;
namespace dacpp {
    typedef std::vector<std::any> list;
}


// 网格参数
const int NX = 2048;          // x方向网格数量
const int NY = 2048;          // y方向网格数量
const double Lx = 10.0f;       // x方向长度
const double Ly = 10.0f;       // y方向长度
const double alpha = 0.01f;    // 热扩散系数
const int TIME_STEPS = 500;  // 时间步数
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

void stencil(const double* mat,double* out,int mat_0,int mat_1,int out_0,int out_1,int mat_0_shape,int mat_1_shape,int out_0_shape,int out_1_shape,sycl::accessor<int, 1, sycl::access::mode::read> info_mat_acc, sycl::accessor<int, 1, sycl::access::mode::read> info_out_acc) {
    out[(0 + out_0) * out_1_shape + (0 + out_1)] = mat[(1 + mat_0) * mat_1_shape + (1 + mat_1)] + alpha * delta_t * (((mat[(2 + mat_0) * mat_1_shape + (1 + mat_1)] - 2.F * mat[(1 + mat_0) * mat_1_shape + (1 + mat_1)] + mat[(0 + mat_0) * mat_1_shape + (1 + mat_1)]) / (dx * dx)) + ((mat[(1 + mat_0) * mat_1_shape + (2 + mat_1)] - 2.F * mat[(1 + mat_0) * mat_1_shape + (1 + mat_1)] + mat[(1 + mat_0) * mat_1_shape + (0 + mat_1)]) / (dy * dy)));
}

struct __dacpp_ctx_stencilShell_stencil {
    sycl::queue dacpp_q{sycl::default_selector_v, {sycl::property::queue::in_order()}};
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
    DataInfo info_matIn;
    std::vector<int> info_matIn_Shape;
    Dac_Ops matIn_Ops;
    std::vector<double> h_matIn;
    std::unique_ptr<sycl::buffer<double, 1>> r_matIn;
    std::vector<int> info_partition_matIn;
    std::unique_ptr<sycl::buffer<int, 1>> info_partition_matIn_buffer;
    DataInfo info_matOut;
    std::vector<int> info_matOut_Shape;
    Dac_Ops matOut_Ops;
    std::vector<double> h_matOut;
    std::unique_ptr<sycl::buffer<double, 1>> r_matOut;
    std::vector<int> info_partition_matOut;
    std::unique_ptr<sycl::buffer<int, 1>> info_partition_matOut_buffer;
};

void __dacpp_init_stencilShell_stencil(__dacpp_ctx_stencilShell_stencil& ctx, dacpp::Matrix<double> & matIn, dacpp::Matrix<double> & matOut) {
    ctx.dacpp_q = sycl::queue(sycl::default_selector_v, {sycl::property::queue::in_order()});
    ctx.info_matIn = DataInfo{};
    ctx.info_matIn.dim = matIn.getDim();
    ctx.info_matIn.dimLength.clear();
    ctx.info_matIn_Shape.assign(ctx.info_matIn.dim, 0);
    for (int dimIdx = 0; dimIdx < ctx.info_matIn.dim; ++dimIdx) {
        ctx.info_matIn.dimLength.push_back(matIn.getShape(dimIdx));
        ctx.info_matIn_Shape[dimIdx] = matIn.getShape(dimIdx);
    }
    ctx.info_matOut = DataInfo{};
    ctx.info_matOut.dim = matOut.getDim();
    ctx.info_matOut.dimLength.clear();
    ctx.info_matOut_Shape.assign(ctx.info_matOut.dim, 0);
    for (int dimIdx = 0; dimIdx < ctx.info_matOut.dim; ++dimIdx) {
        ctx.info_matOut.dimLength.push_back(matOut.getShape(dimIdx));
        ctx.info_matOut_Shape[dimIdx] = matOut.getShape(dimIdx);
    }
    ctx.sp1 = RegularSlice("sp1", 3, 1);
    ctx.sp1.setDimId(0);
    ctx.sp1.SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx.sp1, ctx.info_matIn));
    ctx.sp2 = RegularSlice("sp2", 3, 1);
    ctx.sp2.setDimId(1);
    ctx.sp2.SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx.sp2, ctx.info_matIn));
    ctx.idx1 = Index("idx1");
    ctx.idx1.setDimId(0);
    ctx.idx1.SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx.idx1, ctx.info_matOut));
    ctx.idx2 = Index("idx2");
    ctx.idx2.setDimId(1);
    ctx.idx2.SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx.idx2, ctx.info_matOut));
    ctx.matIn_Ops.clear();
    ctx.sp1.setDimId(0);
    ctx.matIn_Ops.push_back(ctx.sp1);
    ctx.sp2.setDimId(1);
    ctx.matIn_Ops.push_back(ctx.sp2);
    ctx.matOut_Ops.clear();
    ctx.idx1.setDimId(0);
    ctx.matOut_Ops.push_back(ctx.idx1);
    ctx.idx2.setDimId(1);
    ctx.matOut_Ops.push_back(ctx.idx2);
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
    ctx.h_matIn.clear();
    matIn.tensor2Array(ctx.h_matIn);
    ctx.r_matIn = std::make_unique<sycl::buffer<double, 1>>(ctx.h_matIn.data(), sycl::range<1>(matIn.getSize()));
    ctx.r_matIn->set_final_data(nullptr);
    ctx.info_partition_matIn = ctx.para_gene_tool.init_partition_data_shape(ctx.info_matIn, ctx.matIn_Ops);
    ctx.info_partition_matIn_buffer = std::make_unique<sycl::buffer<int, 1>>(ctx.info_partition_matIn.data(), sycl::range<1>(ctx.info_partition_matIn.size()));
    ctx.h_matOut.clear();
    ctx.r_matOut = std::make_unique<sycl::buffer<double, 1>>(sycl::range<1>(matOut.getSize()));
    ctx.info_partition_matOut = ctx.para_gene_tool.init_partition_data_shape(ctx.info_matOut, ctx.matOut_Ops);
    ctx.info_partition_matOut_buffer = std::make_unique<sycl::buffer<int, 1>>(ctx.info_partition_matOut.data(), sycl::range<1>(ctx.info_partition_matOut.size()));
}

void __dacpp_submit_stencilShell_stencil(__dacpp_ctx_stencilShell_stencil& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto& sp1 = ctx.sp1;
    auto& sp2 = ctx.sp2;
    auto& idx1 = ctx.idx1;
    auto& idx2 = ctx.idx2;
    auto& r_matIn = *ctx.r_matIn;
    auto* r_matOut = ctx.r_matOut.get();
    auto& info_partition_matIn_buffer = *ctx.info_partition_matIn_buffer;
    auto& info_partition_matOut_buffer = *ctx.info_partition_matOut_buffer;
    auto* info_matIn_Shape = ctx.info_matIn_Shape.data();
    auto* info_matOut_Shape = ctx.info_matOut_Shape.data();
    const int Item_Size = ctx.Item_Size;
    sycl::range<2> local(ctx.local_x, ctx.local_y);
    sycl::range<2> global(ctx.global_x, ctx.global_y);
    dacpp_q.submit([&](handler &h) {

        accessor<double, 1, access::mode::read> acc_matIn(r_matIn, h);
        
        accessor<double, 1, sycl::access::mode::discard_write> acc_matOut(*r_matOut, h);
        auto info_partition_matIn_accessor = info_partition_matIn_buffer.get_access<sycl::access::mode::read>(h);
        auto info_partition_matOut_accessor = info_partition_matOut_buffer.get_access<sycl::access::mode::read>(h);        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
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

			const auto matIn_0 = sp1_ * sp1.stride;
			const auto matIn_1 = sp2_ * sp2.stride;
			const auto matOut_0 = idx1_ * idx1.stride;
			const auto matOut_1 = idx2_ * idx2.stride;            // 获得accessor指针

            auto* d_matIn = acc_matIn.get_multi_ptr<access::decorated::no>().get();
            auto* d_matOut = acc_matOut.get_multi_ptr<access::decorated::no>().get();            // 嵌入计算

            stencil(d_matIn,d_matOut,matIn_0,matIn_1,matOut_0,matOut_1,info_matIn_Shape[0],info_matIn_Shape[1],info_matOut_Shape[0],info_matOut_Shape[1],info_partition_matIn_accessor,info_partition_matOut_accessor);        });
    });
}

void __dacpp_submit_region_stencilShell_stencil_stmt_0(__dacpp_ctx_stencilShell_stencil& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto* r_matIn = ctx.r_matIn.get();
    auto* r_matOut = ctx.r_matOut.get();
    auto* info_matIn_Shape = ctx.info_matIn_Shape.data();
    auto* info_matOut_Shape = ctx.info_matOut_Shape.data();
{
    int __iL = (1);
    int __iR = (NX-2);
    int __iN = __iR - __iL +1;
    int __jL = (1);
    int __jR = (NY-2);
    int __jN = __jR - __jL + 1;
    dacpp_q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::write>(h);
    auto acc_matOut = r_matOut->get_access<sycl::access::mode::read>(h);
    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      auto* d_matOut = acc_matOut.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __iL + idx[0];
      int j = __jL + idx[1];
      {
                d_matIn[(i) * (info_matIn_Shape[1]) + (j)]=d_matOut[(i-1) * (info_matOut_Shape[1]) + (j-1)];
            }
    });
  });
}
}

void __dacpp_submit_region_stencilShell_stencil_stmt_1(__dacpp_ctx_stencilShell_stencil& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto* r_matIn = ctx.r_matIn.get();
    auto* r_matOut = ctx.r_matOut.get();
    auto* info_matIn_Shape = ctx.info_matIn_Shape.data();
    auto* info_matOut_Shape = ctx.info_matOut_Shape.data();
{
  int __L = (0);
  int __R = (NY-1);
  int __N = __R - __L +1;
  dacpp_q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      int j = __L + idx[0];
      {
            //double* data = new double[1];
            d_matIn[(0) * (info_matIn_Shape[1]) + (j)]=d_matIn[(1) * (info_matIn_Shape[1]) + (j)];              // 顶部边界
            d_matIn[(NX - 1) * (info_matIn_Shape[1]) + (j)]=d_matIn[(NX-2) * (info_matIn_Shape[1]) + (j)];
             // 底部边界
        }
    });
  });
}
}

void __dacpp_submit_region_stencilShell_stencil_stmt_2(__dacpp_ctx_stencilShell_stencil& ctx) {
    using namespace sycl;
    auto& dacpp_q = ctx.dacpp_q;
    auto* r_matIn = ctx.r_matIn.get();
    auto* r_matOut = ctx.r_matOut.get();
    auto* info_matIn_Shape = ctx.info_matIn_Shape.data();
    auto* info_matOut_Shape = ctx.info_matOut_Shape.data();
{
  int __L = (0);
  int __R = (NX-1);
  int __N = __R - __L +1;
  dacpp_q.submit([&](sycl::handler& h){
    auto acc_matIn = r_matIn->get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){
      auto* d_matIn = acc_matIn.template get_multi_ptr<sycl::access::decorated::no>().get();
      int i = __L + idx[0];
      {
            //double* data = new double[1];
            d_matIn[(i) * (info_matIn_Shape[1]) + (0)]=d_matIn[(i) * (info_matIn_Shape[1]) + (1)];              // 顶部边界
            d_matIn[(i) * (info_matIn_Shape[1]) + (NY-1)]=d_matIn[(i) * (info_matIn_Shape[1]) + (NY-2)];
        }
    });
  });
}
}

void __dacpp_sync_stencilShell_stencil(__dacpp_ctx_stencilShell_stencil& ctx, dacpp::Matrix<double> & matIn, dacpp::Matrix<double> & matOut) {
    using namespace sycl;
    ctx.dacpp_q.wait();
    ctx.h_matIn.resize(matIn.getSize());
    {
        host_accessor acc(*ctx.r_matIn);
        for (std::size_t idx = 0; idx < ctx.h_matIn.size(); ++idx) {
            ctx.h_matIn[idx] = acc[idx];
        }
    }
    matIn.array2Tensor(ctx.h_matIn);
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
    dacpp::Matrix<double> matIn({NX, NY}, u_curr);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matOut = u_next_tensor[{1,NX-1}][{1,NY-1}];
        __dacpp_ctx_stencilShell_stencil __dacpp_ctx_stencilShell_stencil_0;
    __dacpp_init_stencilShell_stencil(__dacpp_ctx_stencilShell_stencil_0, matIn, matOut);
for(int i=0;i<TIME_STEPS;i++) {
        __dacpp_submit_stencilShell_stencil(__dacpp_ctx_stencilShell_stencil_0);

        __dacpp_submit_region_stencilShell_stencil_stmt_0(__dacpp_ctx_stencilShell_stencil_0);

        // 处理边界条件（绝热边界：导数为零）
        __dacpp_submit_region_stencilShell_stencil_stmt_1(__dacpp_ctx_stencilShell_stencil_0);
        __dacpp_submit_region_stencilShell_stencil_stmt_2(__dacpp_ctx_stencilShell_stencil_0);
        
    }
    __dacpp_sync_stencilShell_stencil(__dacpp_ctx_stencilShell_stencil_0, matIn, matOut);

    matIn[0].print();


    // 输出最终结果的某些值作为示例
    //cout << "Final temperature at center: " << vec2D[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
