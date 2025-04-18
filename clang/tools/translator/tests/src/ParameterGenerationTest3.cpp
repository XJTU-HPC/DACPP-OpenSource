#include "ParameterGeneration.h"
#include "sub_template.h"

//测试设备内存空间分配 数据重组的内存 使用新的Tensor 
//这个测试的函数已经弃用了 接口太复杂了

int main()
{
    // std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // //std::vector<int> shapeA{4, 4};
    // dacpp::Tensor<int,2> matA({4,4},dataA);

    // std::vector<int> dataB{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // //std::vector<int> shapeB{4, 4};
    // dacpp::Tensor<int,2> matB({4,4},dataB);

    // std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // //std::vector<int> shapeC{4, 4};
    // dacpp::Tensor<int,2> matC({4,4},dataC);

    // matA.print();
    // matB.print();
    // matC.print();

    // // 输入的Tensor的vector数组 
    // std::vector<dacpp::Tensor<int,2>> tensor_in;
    // tensor_in.push_back(matA);
    // tensor_in.push_back(matB);

    // //声明算子
    // //这两个算子作用于matA
    // RegularSlice si("si", 2, 2);
    // si.setDimId(0);
    // RegularSlice sj("sj", 2, 2);
    // sj.setDimId(1);

    // //matA的算子组
    // Dac_Ops ops_matA;
    // ops_matA.push_back(si);
    // ops_matA.push_back(sj);

    // //声明算子 下面两个算子作用于matB 不能直接修改算子作用的维度 因为一改之后前面的作用的维度也会发生变化
    // RegularSlice sj1("sj", 2, 2);//sj1其实是sj作用于不同维度的算子 这个名字不能改
    // sj1.setDimId(0);
    // RegularSlice sk("sk", 2, 2);
    // sk.setDimId(1);

    // //matB的算子组
    // Dac_Ops ops_matB;
    // ops_matB.push_back(sj1);
    // ops_matB.push_back(sk);

    // //输入的算子组的vector 
    // std::vector<Dac_Ops> ops_in;
    // ops_in.push_back(ops_matA);
    // ops_in.push_back(ops_matB);

    // //输出的算子组
    // Dac_Ops ops_out;
    // ops_out.push_back(si);
    // ops_out.push_back(sk);

    // //两个分区算子分别作用于0维和1维


    // ParameterGeneration<int,2> para_gene_tool;

    //int DataReconstructorSize = para_gene_tool.init_device_memory_size(tensor_in,matC,ops_in,ops_out);

    //std::cout << "DataReconstructorSize: " << DataReconstructorSize << std::endl;//应该输出32
}