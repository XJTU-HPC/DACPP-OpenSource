#include <sycl/sycl.hpp>
using namespace sycl;

#include <vector>
#include<iostream>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

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

void blockMatMul(dacpp::Tensor<int,2> &matA, dacpp::Tensor<int,2> &matB, dacpp::Tensor<int,2> &matC) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //ParameterGeneration<int,2> para_gene_tool; //参数生成工具
    ParameterGeneration para_gene_tool;
    DataInfo info_matA;
    info_matA.dim = matA.getDim();
    for(int i = 0; i < info_matA.dim; i++) info_matA.dimLength.push_back(matA.getShape(i));

    DataInfo info_matB;
    info_matB.dim = matB.getDim();
    for(int i = 0; i < info_matB.dim; i++) info_matB.dimLength.push_back(matB.getShape(i));

    DataInfo info_matC;
    info_matC.dim = matC.getDim();
    for(int i = 0; i < info_matC.dim; i++) info_matC.dimLength.push_back(matC.getShape(i));
    // 算子初始化
    
    // 规则分区算子初始化
    RegularSlice si = RegularSlice("si", 2, 2);
    si.setDimId(0);
    //si.SetSplitSize(para_gene_tool.init_operetor_splitnumber(si,matA));
    si.SetSplitSize(para_gene_tool.init_operetor_splitnumber(si,info_matA));
    std::cout << "si.split_size: " << si.split_size << std::endl;

    // 规则分区算子初始化
    RegularSlice sj = RegularSlice("sj", 2, 2);
    sj.setDimId(1);
    //sj.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sj,matB));
    sj.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sj,info_matB));
    std::cout << "sj.split_size: " << sj.split_size << std::endl;

    // 规则分区算子初始化
    RegularSlice sk = RegularSlice("sk", 2, 2);
    sk.setDimId(1);
    //sk.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sk,matA));
    sk.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sk,info_matA));
    std::cout << "sk.split_size: " << sk.split_size << std::endl;
    // 参数生成 提前计算后面需要用到的参数

    // 算子组初始化
    Dac_Ops matA_ops;
    si.setDimId(0);
    matA_ops.push_back(si);
    sk.setDimId(1);
    matA_ops.push_back(sk);

    // 算子组初始化
    Dac_Ops matB_ops;
    sk.setDimId(0);
    matB_ops.push_back(sk);
    sj.setDimId(1);
    matB_ops.push_back(sj);

    // 算子组初始化
    Dac_Ops In_ops;  
    si.setDimId(0);
    In_ops.push_back(si);
    sj.setDimId(1);
    In_ops.push_back(sj);
    sk.setDimId(2);
    In_ops.push_back(sk);

    // 算子组初始化
    Dac_Ops Out_ops;
    si.setDimId(0);
    Out_ops.push_back(si);
    sj.setDimId(1);
    Out_ops.push_back(sj);

    // 算子组初始化
    Dac_Ops reduction_ops; 
    si.setDimId(0);
    reduction_ops.push_back(si);
    sj.setDimId(1);
    reduction_ops.push_back(sj);

    //生成设备内存分配大小
    //int matA_size = para_gene_tool.init_device_memory_size(matA,matA_ops);
    int matA_size = para_gene_tool.init_device_memory_size(info_matA,matA_ops);
    std::cout << "matA_size: " << matA_size << std::endl;

    //生成设备内存分配大小
    //int matB_size = para_gene_tool.init_device_memory_size(matB,matB_ops);
    int matB_size = para_gene_tool.init_device_memory_size(info_matB,matB_ops);
    std::cout << "matB_size: " << matB_size << std::endl;

    //生成设备内存分配大小
    //int matC_size = para_gene_tool.init_device_memory_size(In_ops,Out_ops,matC);
    int matC_size = para_gene_tool.init_device_memory_size(In_ops,Out_ops,info_matC);
    std::cout << "matC_size: " << matC_size << std::endl;

    //生成设备内存分配大小
    //int reduction_size = para_gene_tool.init_device_memory_size(matC,reduction_ops);
    int reduction_size = para_gene_tool.init_device_memory_size(info_matC,reduction_ops);
    std::cout << "reduction_size: " << reduction_size << std::endl;

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matA_ops,matA_size);
    std::cout << "in matA_ops:" << std::endl;
    for(int i = 0;i < matA_ops.size;i ++)
    {
        std::cout << matA_ops.DacOps[i].name << " split_length:" << matA_ops.DacOps[i].split_length << std::endl;
    }

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matB_ops,matB_size);
    std::cout << "in matB_ops:" << std::endl;
    for(int i = 0;i < matB_ops.size;i ++)
    {
        std::cout << matB_ops.DacOps[i].name << " split_length:" << matB_ops.DacOps[i].split_length << std::endl;
    }

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_ops,matC_size);
    std::cout << "in In_ops:" << std::endl;
    for(int i = 0;i < In_ops.size;i ++)
    {
        std::cout << In_ops.DacOps[i].name << " split_length:" << In_ops.DacOps[i].split_length << std::endl;
    }

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(reduction_ops,reduction_size);
    std::cout << "in reduction_ops:" << std::endl;
    for(int i = 0;i < reduction_ops.size;i ++)
    {
        std::cout << reduction_ops.DacOps[i].name << " split_length:" << reduction_ops.DacOps[i].split_length << std::endl;
    }

    std::vector<Dac_Ops> ops_s;
    ops_s.push_back(matA_ops);
    ops_s.push_back(matB_ops);
    ops_s.push_back(In_ops);

    // 生成划分长度的二维矩阵
    int SplitLength[3][3] = {0};
    para_gene_tool.init_split_length_martix(3,3,&SplitLength[0][0],ops_s);
    std::cout << "split_length_martix: " << std::endl;
    for(int i = 0;i < 3;i ++)
    {
        for(int j = 0;j < 3;j ++)
        {
            std::cout << SplitLength[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // 计算工作项的大小
    int item_size = para_gene_tool.init_work_item_size(In_ops);
    std::cout << "item_size: " << item_size << std::endl;

    // 计算归约中split_size的大小
    int reduction_split_size = para_gene_tool.init_reduction_split_size(In_ops,Out_ops);
    std::cout << "reduction_split_size: " << reduction_split_size << std::endl;

    // 计算归约中split_length的大小
    int reduction_split_length = para_gene_tool.init_reduction_split_length(reduction_ops);
    std::cout << "reduction_split_length: " << reduction_split_length << std::endl;

    
    // 设备内存分配
    int *d_matA=malloc_device<int>(matA_size,q);
    // 设备内存分配
    int *d_matB=malloc_device<int>(matB_size,q);
    // 设备内存分配
    int *d_matC=malloc_device<int>(matC_size,q);
    // 归约设备内存分配
    int *reduction_matC = malloc_device<int>(reduction_size,q);
    // 数据关联计算
    
    // 数据重组
    DataReconstructor<int> matA_tool;
    int* r_matA=(int*)malloc(sizeof(int)*matA_size);   
    // 数据算子组初始化
    Dac_Ops t_matA_ops;  
    si.setDimId(0);
    //si.setSplitLength(8);
    t_matA_ops.push_back(si);
    sk.setDimId(1);
    //sk.setSplitLength(4);
    t_matA_ops.push_back(sk);
    matA_tool.init(info_matA,t_matA_ops);
    matA_tool.Reconstruct(r_matA,matA);
    std::cout <<  "matA重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matA[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }


    // 数据重组
    DataReconstructor<int> matB_tool;
    int* r_matB=(int*)malloc(sizeof(int)*matB_size);  
    // 数据算子组初始化
    Dac_Ops t_matB_ops;   
    sk.setDimId(0);
    //sk.setSplitLength(8);
    t_matB_ops.push_back(sk);
    sj.setDimId(1);
    //sj.setSplitLength(4);
    t_matB_ops.push_back(sj);
    matB_tool.init(info_matB,t_matB_ops);
    matB_tool.Reconstruct(r_matB,matB);
    std::cout <<  "matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }


    // 数据重组
    DataReconstructor<int> matC_tool;
    int* r_matC=(int*)malloc(sizeof(int)*matC_size);   
    // 数据算子组初始化
    Dac_Ops t_matC_ops;   
    si.setDimId(0);
    //si.setSplitLength(16);
    t_matC_ops.push_back(si);
    sj.setDimId(1);
    //sj.setSplitLength(8);
    t_matC_ops.push_back(sj);
    //sk.setDimId(2);
    //sk.setSplitLength(4);
    //注意 假如添加sk算子计算结果将不正确
    //t_matC_ops.push_back(sk); //参照blockMatmul1.sycl.cpp中这一块没有添加sk算子
    matC_tool.init(info_matC,t_matC_ops);
    matC_tool.Reconstruct(r_matC,matC);
    
    // 数据移动
    q.memcpy(d_matA,r_matA,matA_size*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_matB,r_matB,matB_size*sizeof(int)).wait();

    q.memset(d_matC,0,matC_size*sizeof(int)).wait();//这条语句不是模板生成的 需要增加

    //工作项划分
    sycl::range<3> local(1, 1, item_size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化

            const auto si_=(item_id/sj.split_size/sk.split_size+(0))%si.split_size;
            const auto sj_=(item_id/sk.split_size+(0))%sj.split_size;
            const auto sk_=(item_id+(0))%sk.split_size;
            // 嵌入计算

            block_mat_mul(d_matA+(si_*SplitLength[0][0]+sk_*SplitLength[0][1]),d_matB+(sk_*SplitLength[1][0]+sj_*SplitLength[1][1]),d_matC+(si_*SplitLength[2][0]+sj_*SplitLength[2][1]+sk_*SplitLength[2][2]));
        });
    }).wait();

    std::cout<<std::endl;
    std::cout <<  "归约前计算结果:\n";
    q.memcpy(r_matC, d_matC, 32*sizeof(int)).wait();
    for(int i=0;i<32;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<std::endl;   

    // 归约
    for(int i=0;i<reduction_size;i++) {
        q.submit([&](handler &h) {
            h.parallel_for(
            range<1>(reduction_split_size),
            reduction(reduction_matC+i,
            sycl::plus<>(),
            property::reduction::initialize_to_identity()),
            [=](id<1> idx,auto &reducer) {
                reducer.combine(d_matC[(i/reduction_split_length)*reduction_split_length*reduction_split_size+i%reduction_split_length+idx*reduction_split_length]);
            });
    }).wait();
    }
//     q.submit([&](handler &h) {
//         h.parallel_for(
//         range<1>(reduction_split_size * reduction_size),
//         //reduction(span<int,reduction_size>(reduction_matC,reduction_size), 
//         reduction(span<int,16>(reduction_matC,16), 
//         sycl::plus<>(),
//         property::reduction::initialize_to_identity()),
//         [=](id<1> i,auto &reducer) {
//             reducer[i % reduction_split_length + i/(reduction_split_length*reduction_split_size)*reduction_split_length].combine(d_matC[i]);
//         });
//  }).wait();

 //上面现在除了 reduction(span<int,16>(reduction_matC,16)这个16需要填成常量外，其余的计算没有出现问题
 //下面这块参数调用的有些乱
 //start
    // q.memcpy(d_matC,reduction_matC, reduction_size*sizeof(int)).wait();


    // // 归约结果返回
    // q.memcpy(r_matC,d_matC, 1*sizeof(int)).wait();
    // matC_tool.UpdateData(r_matC,matC);
//end

//参照blockMatMul1.sycl.cpp调用
    q.memcpy(d_matC,reduction_matC, 16*sizeof(int)).wait();

    q.memcpy(r_matC,d_matC, 16*sizeof(int)).wait();
    matC_tool.UpdateData(r_matC,matC);
    std::cout <<  "归约后计算结果(归并):\n";
    matC.print();

    // 内存释放
    
    sycl::free(d_matA, q);
    sycl::free(d_matB, q);
    sycl::free(d_matC, q);

}

int main() {
    std::vector<int> dataA{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    dacpp::Tensor<int,2> matA({4,4},dataA);
    std::vector<int> dataB{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    dacpp::Tensor<int,2> matB({4,4},dataB);
    std::vector<int> dataC{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    dacpp::Tensor<int,2> matC({4,4},dataC);

    blockMatMul(matA,matB,matC);
    
    return 0;
}