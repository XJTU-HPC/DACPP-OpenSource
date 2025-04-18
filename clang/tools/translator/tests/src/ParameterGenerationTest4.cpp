#include "ParameterGeneration.h"
#include "sub_template.h"

//测试设备内存空间分配 数据重组的内存 

int main()
{
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Tensor<int,2> matA({4,4},dataA);
    DataInfo info_matA;
    info_matA.dim = matA.getDim();
    for(int i = 0; i < info_matA.dim; i++) info_matA.dimLength.push_back(matA.getShape(i));

    std::vector<int> dataB{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Tensor<int,2> matB({4,4},dataB);
    DataInfo info_matB;
    info_matB.dim = matB.getDim();
    for(int i = 0; i < info_matB.dim; i++) info_matB.dimLength.push_back(matB.getShape(i));

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Tensor<int,2> matC({4,4},dataC);
    DataInfo info_matC;
    info_matC.dim = matC.getDim();
    for(int i = 0; i < info_matC.dim; i++) info_matC.dimLength.push_back(matC.getShape(i));

    matA.print();
    matB.print();
    matC.print();
    ParameterGeneration para_gene_tool;

    //声明算子
    //这两个算子作用于matA
    RegularSlice si("si", 2, 2);
    si.setDimId(0);
    si.SetSplitSize(para_gene_tool.init_operetor_splitnumber(si,info_matA));//设置完维度之后应该马上初始化算子的划分数

    RegularSlice sj("sj", 2, 2);
    sj.setDimId(1);
    sj.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sj,info_matA));

    RegularSlice sk("sk", 2, 2);
    sk.setDimId(1);
    sk.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sk,info_matB));

    //输入的算子组（里面不能有重复的算子 由后端进行去重）
    Dac_Ops ops_in;
    ops_in.push_back(si);
    ops_in.push_back(sj);
    ops_in.push_back(sk);

    //sj.setDimId(0);//这里可以改维度的，之前的已经被存到一个vector里面了

    //输出的算子组
    Dac_Ops ops_out;
    ops_out.push_back(si);
    ops_out.push_back(sk);

    int DataReconstructorSize = para_gene_tool.init_device_memory_size(ops_in,ops_out,info_matC);

    std::cout << "DataReconstructorSize: " << DataReconstructorSize << std::endl;//应该输出32
}



// int main()
// {
//     std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//     dacpp::Tensor<int,2> matA({4,4},dataA);

//     std::vector<int> dataB{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//     dacpp::Tensor<int,2> matB({4,4},dataB);

//     std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//     dacpp::Tensor<int,2> matC({4,4},dataC);

//     matA.print();
//     matB.print();
//     matC.print();
//     ParameterGeneration<int,2> para_gene_tool;

//     //声明算子
//     //这两个算子作用于matA
//     RegularSlice si("si", 2, 2);
//     si.setDimId(0);
//     si.SetSplitSize(para_gene_tool.init_operetor_splitnumber(si,matA));//设置完维度之后应该马上初始化算子的划分数

//     RegularSlice sj("sj", 2, 2);
//     sj.setDimId(1);
//     sj.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sj,matA));

//     RegularSlice sk("sk", 2, 2);
//     sk.setDimId(1);
//     sk.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sk,matB));

//     //输入的算子组（里面不能有重复的算子 由后端进行去重）
//     Dac_Ops ops_in;
//     ops_in.push_back(si);
//     ops_in.push_back(sj);
//     ops_in.push_back(sk);

//     //sj.setDimId(0);//这里可以改维度的，之前的已经被存到一个vector里面了

//     //输出的算子组
//     Dac_Ops ops_out;
//     ops_out.push_back(si);
//     ops_out.push_back(sk);

//     int DataReconstructorSize = para_gene_tool.init_device_memory_size(ops_in,ops_out,matC);

//     std::cout << "DataReconstructorSize: " << DataReconstructorSize << std::endl;//应该输出32
// }