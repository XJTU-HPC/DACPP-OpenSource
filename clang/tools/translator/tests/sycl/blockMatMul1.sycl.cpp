#include <sycl/sycl.hpp>
using namespace sycl;
#include <vector>
#include<iostream>
#include "DataReconstructor.h"

void debug(int x) {
    std::cout << "ok" << x << std::endl;
}
void block_mat_mul(int *matA, int *matB, int *matC) {
    for (int i=0;i<2;i++){
        for (int j=0;j<2;j++){
            for (int k=0;k<2;k++){
                matC[i*2+j] += matA[i*2+k]*matB[k*2+j];
            }
        }
    } 
}
void blockMatMulSplit(dacpp::Tensor<int,2> &matA, dacpp::Tensor<int,2> &matB, dacpp::Tensor<int,2> &matC)
{
    auto selector = default_selector_v;
    queue q(selector);

    DataInfo info_matA;
    info_matA.dim = matA.getDim();
    for(int i = 0; i < info_matA.dim; i++) info_matA.dimLength.push_back(matA.getShape(i));

    DataInfo info_matB;
    info_matB.dim = matB.getDim();
    for(int i = 0; i < info_matB.dim; i++) info_matB.dimLength.push_back(matB.getShape(i));

    DataInfo info_matC;
    info_matC.dim = matC.getDim();
    for(int i = 0; i < info_matC.dim; i++) info_matC.dimLength.push_back(matC.getShape(i));

    RegularSlice si = RegularSlice("si", 2, 2);
	si.SetSplitSize(2);
	RegularSlice sj = RegularSlice("sj", 2, 2);
	sj.SetSplitSize(2);
    RegularSlice sk = RegularSlice("sk", 2, 2);
	sk.SetSplitSize(2);

    DataReconstructor<int> matA_tool;
    int* r_matA=(int*)malloc(sizeof(int)*16);
    Dac_Ops matA_ops;
    si.setDimId(0);
    si.setSplitLength(8);
    matA_ops.push_back(si);
    sk.setDimId(1);
    sk.setSplitLength(4);
    matA_ops.push_back(sk);
    matA_tool.init(info_matA,matA_ops);
    matA_tool.Reconstruct(r_matA,matA);

    std::cout <<  "matA重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matA[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    DataReconstructor<int> matB_tool;
    int* r_matB=(int*)malloc(sizeof(int)*16);
    Dac_Ops matB_ops;
    sk.setDimId(0);
    sk.setSplitLength(8);
    matB_ops.push_back(sk);
    sj.setDimId(1);
    sj.setSplitLength(4);
    matB_ops.push_back(sj);
    matB_tool.init(info_matB,matB_ops);
    matB_tool.Reconstruct(r_matB,matB);

    std::cout <<  "matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    // debug(1);

    DataReconstructor<int> matC_tool;
    int* r_matC=(int*)malloc(sizeof(int)*16*2);
    Dac_Ops matC_ops;
    si.setDimId(0);
    si.setSplitLength(8);
    matC_ops.push_back(si);
    sj.setDimId(1);
    sj.setSplitLength(4);
    matC_ops.push_back(sj);
    // sk.setDimId(2);
    // sk.setSplitLength(4);
    // matC_ops.push_back(sk);
    matC_tool.init(info_matC,matC_ops);
    matC_tool.Reconstruct(r_matC,matC);

    std::cout <<  "matC重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matC[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    // debug(2);

    int *d_matA=malloc_device<int>(16,q);
    int *d_matB=malloc_device<int>(16,q);
    int *d_matC=malloc_device<int>(16*2,q);

    q.memcpy(d_matA,r_matA,16*sizeof(int)).wait();
    q.memcpy(d_matB,r_matB,16*sizeof(int)).wait();
    
    q.memset(d_matC,0,32*sizeof(int)).wait();

    // debug(3);

    sycl::range<3> local(1, 1, 8);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
            const auto si=item_id/2/2%2;
            const auto sj=item_id/2%2;
            const auto sk=item_id%2;



            // 嵌入计算
            block_mat_mul(d_matA+(si*8+sk*4),d_matB+(sk*8+sj*4),d_matC+(si*16+sj*8+sk*4));
        });
    }).wait();

    // debug(4);
    std::cout<<std::endl;
    std::cout <<  "归约前计算结果:\n";
    q.memcpy(r_matC, d_matC, 32*sizeof(int)).wait();
    for(int i=0;i<32;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<std::endl;

    // 归约
    int *reduction_matC = malloc_device<int>(16,q);
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(2 * 16),
        reduction(span<int,16>(reduction_matC,16), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            	reducer[i % 4+i/8*4].combine(d_matC[i]);
     	});
    }).wait();

    // int *res = malloc_shared<int>(16,q);
    q.memcpy(r_matC,reduction_matC, 16*sizeof(int)).wait();
    std::cout <<  "归约后计算结果(未归并):\n";
    for(int i=0;i<16;i++) {
        std::cout << r_matC[i] << " ";
        if(i%4==3) std::cout<<"\n";
    }
    // debug(5);
    matC_tool.UpdateData(r_matC,matC);
    std::cout <<  "归约后计算结果(归并):\n";
    matC.print();
}
int main() {
    std::vector<int> dataA{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    dacpp::Tensor<int,2> matA({4,4},dataA);
    std::vector<int> dataB{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    dacpp::Tensor<int,2> matB({4,4},dataB);
    std::vector<int> dataC{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    dacpp::Tensor<int,2> matC({4,4},dataC);

    blockMatMulSplit(matA,matB,matC);
    
    return 0;
}