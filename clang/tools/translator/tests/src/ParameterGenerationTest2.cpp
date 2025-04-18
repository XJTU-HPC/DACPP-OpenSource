#include "ParameterGeneration.h"
#include "sub_template.h"


//测试设备内存空间分配 使用新的Tensor

int main()
{
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Tensor<int,2> matA({4,4},dataA);
    DataInfo info_matA;
    info_matA.dim = matA.getDim();
    for(int i = 0; i < info_matA.dim; i++) info_matA.dimLength.push_back(matA.getShape(i));

    matA.print();

    //两个分区算子分别作用于0维和1维
    RegularSlice si("si", 2, 2);
    si.setDimId(0);
    RegularSlice sj("sj", 2, 2);
    sj.setDimId(1);

    //两个降维算子分别作用于0维和1维
    Index sk1("sk1");
    sk1.setDimId(0);
    Index sk2("sk2");
    sk2.setDimId(1);

    //mat[分区][分区]
    Dac_Ops matA_ops1;
    matA_ops1.push_back(si);
    matA_ops1.push_back(sj);

    //mat[降维][降维]
    Dac_Ops matA_ops2;
    matA_ops2.push_back(sk1);
    matA_ops2.push_back(sk2); 

    //mat[分区][降维]
    Dac_Ops matA_ops3;
    matA_ops3.push_back(si);
    matA_ops3.push_back(sk2);

    //mat[分区][]
    Dac_Ops matA_ops4;
    matA_ops4.push_back(si);

    //mat[][降维]
    Dac_Ops matA_ops5;
    matA_ops5.push_back(sk2);

    //mat[][] 没有算子组

    ParameterGeneration para_gene_tool;

    // //下面测试设备内存分配大小的size

    int matA1_size = para_gene_tool.init_device_memory_size(info_matA,matA_ops1);
    std::cout << "matA_size:" << matA1_size << std::endl; //应该输出16 测试没有问题

    int matA2_size = para_gene_tool.init_device_memory_size(info_matA,matA_ops2);
    std::cout << "matA_size:" << matA2_size << std::endl; //应该输出16

    int matA3_size = para_gene_tool.init_device_memory_size(info_matA,matA_ops3);
    std::cout << "matA_size:" << matA3_size << std::endl; //应该输出16

    int matA4_size = para_gene_tool.init_device_memory_size(info_matA,matA_ops4);
    std::cout << "matA_size:" << matA4_size << std::endl; //应该输出16

    int matA5_size = para_gene_tool.init_device_memory_size(info_matA,matA_ops5);
    std::cout << "matA_size:" << matA5_size << std::endl; //应该输出16

    int matA6_size = para_gene_tool.init_device_memory_size(info_matA);
    std::cout << "matA_size:" << matA6_size << std::endl; //应该输出16
}


// int main()
// {
//     std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//     dacpp::Tensor<int,2> matA({4,4},dataA);

//     matA.print();

//     //两个分区算子分别作用于0维和1维
//     RegularSlice si("si", 2, 2);
//     si.setDimId(0);
//     RegularSlice sj("sj", 2, 2);
//     sj.setDimId(1);

//     //两个降维算子分别作用于0维和1维
//     Index sk1("sk1");
//     sk1.setDimId(0);
//     Index sk2("sk2");
//     sk2.setDimId(1);

//     //mat[分区][分区]
//     Dac_Ops matA_ops1;
//     matA_ops1.push_back(si);
//     matA_ops1.push_back(sj);

//     //mat[降维][降维]
//     Dac_Ops matA_ops2;
//     matA_ops2.push_back(sk1);
//     matA_ops2.push_back(sk2); 

//     //mat[分区][降维]
//     Dac_Ops matA_ops3;
//     matA_ops3.push_back(si);
//     matA_ops3.push_back(sk2);

//     //mat[分区][]
//     Dac_Ops matA_ops4;
//     matA_ops4.push_back(si);

//     //mat[][降维]
//     Dac_Ops matA_ops5;
//     matA_ops5.push_back(sk2);

//     //mat[][] 没有算子组

//     ParameterGeneration<int,2> para_gene_tool;

//     // //下面测试设备内存分配大小的size

//     int matA1_size = para_gene_tool.init_device_memory_size(matA,matA_ops1);
//     std::cout << "matA_size:" << matA1_size << std::endl; //应该输出16 测试没有问题

//     int matA2_size = para_gene_tool.init_device_memory_size(matA,matA_ops2);
//     std::cout << "matA_size:" << matA2_size << std::endl; //应该输出16

//     int matA3_size = para_gene_tool.init_device_memory_size(matA,matA_ops3);
//     std::cout << "matA_size:" << matA3_size << std::endl; //应该输出16

//     int matA4_size = para_gene_tool.init_device_memory_size(matA,matA_ops4);
//     std::cout << "matA_size:" << matA4_size << std::endl; //应该输出16

//     int matA5_size = para_gene_tool.init_device_memory_size(matA,matA_ops5);
//     std::cout << "matA_size:" << matA5_size << std::endl; //应该输出16

//     int matA6_size = para_gene_tool.init_device_memory_size(matA);
//     std::cout << "matA_size:" << matA6_size << std::endl; //应该输出16
// }