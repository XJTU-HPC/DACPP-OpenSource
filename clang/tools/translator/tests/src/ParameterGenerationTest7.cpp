#include "ParameterGeneration.h"
#include "sub_template.h"

//测试算子划分长度矩阵的生成

int main()
{
    const int row = 3;
    const int col = 3;
    int matrix[row][col] = {0};

    ParameterGeneration para_gene_tool;

    // 规则分区算子初始化
    RegularSlice si = RegularSlice("si", 2, 2);
    si.setSplitLength(4);

    // 规则分区算子初始化
    RegularSlice sj = RegularSlice("sj", 2, 2);
    sj.setSplitLength(5);

    // 规则分区算子初始化
    RegularSlice sk = RegularSlice("sk", 2, 2);
    sk.setSplitLength(6);

    // 算子组初始化
    Dac_Ops matA_ops;
    matA_ops.push_back(si);
    matA_ops.push_back(sk);

    // 算子组初始化
    Dac_Ops matB_ops;
    matB_ops.push_back(sk);
    matB_ops.push_back(sj);

    // 算子组初始化
    Dac_Ops In_ops;
    In_ops.push_back(si);
    In_ops.push_back(sj);
    In_ops.push_back(sk);

    std::vector<Dac_Ops> ops_s;
    ops_s.push_back(matA_ops);
    ops_s.push_back(matB_ops);
    ops_s.push_back(In_ops);

    para_gene_tool.init_split_length_martix(row,col,&matrix[0][0],ops_s);

    for(int i = 0;i < 3;i ++)
    {
        for(int j = 0;j < 3;j ++)
            std::cout << matrix[i][j] << " ";
        std::cout << std::endl;
    }

}