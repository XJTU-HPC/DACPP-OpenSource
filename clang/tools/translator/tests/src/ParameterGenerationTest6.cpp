#include "ParameterGeneration.h"
#include "sub_template.h"

//测试算子划分长度的计算

int main()
{
    //4 * 4的矩阵
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Tensor<int,2> matA({4,4},dataA);
    DataInfo info_matA;
    info_matA.dim = matA.getDim();
    for(int i = 0; i < info_matA.dim; i++) info_matA.dimLength.push_back(matA.getShape(i));

    matA.print();

    ParameterGeneration para_gene_tool;

    //声明算子
    //这两个算子作用于matA
    RegularSlice si("si", 2, 1);
    si.setDimId(0);
    si.SetSplitSize(para_gene_tool.init_operetor_splitnumber(si,info_matA));//设置完维度之后应该马上初始化算子的划分数

    RegularSlice sj("sj", 2, 1);
    sj.setDimId(1);
    sj.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sj,info_matA));

    //输入的算子组（里面不能有重复的算子 由后端进行去重）
    Dac_Ops ops_in;
    ops_in.push_back(si);
    ops_in.push_back(sj);

    int DataReconstructorSize = para_gene_tool.init_device_memory_size(info_matA,ops_in);//这个应该是36

    para_gene_tool.init_op_split_length(ops_in,DataReconstructorSize);

    for(int i = 0;i < ops_in.size;i ++)
    {
        std::cout << ops_in.DacOps[i].split_length << std::endl;// 应该是12 4
    }
}