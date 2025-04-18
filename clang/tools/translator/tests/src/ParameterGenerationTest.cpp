#include "ParameterGeneration.h"
#include "sub_template.h"

//测试算子划分数 已经使用新的Tensor

int main()
{
    std::vector<int> data{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}; //在运行时会先从文件中读取
    
    //使用新的Tensor
    dacpp::Tensor<int,2> myTensor({4,5},data);
    // 数据信息初始化
    DataInfo info_data;
    info_data.dim = myTensor.getDim();
    for(int i = 0; i < info_data.dim; i++) info_data.dimLength.push_back(myTensor.getShape(i));

    myTensor.print();
    RegularSlice si = RegularSlice("si", 2, 2); //这些通过dacpp的抽象语法树可以分析得到
    si.setDimId(0); //这个通过分析抽象语法树得到
    Index sj = Index("sj");
    sj.setDimId(1);
    RegularSlice sk = RegularSlice("sk", 3, 1); //这些通过dacpp的抽象语法树可以分析得到
    sk.setDimId(1); //这个通过分析抽象语法树得到

    ParameterGeneration para_gene_tool;
    int si_spilit_number = para_gene_tool.init_operetor_splitnumber(si,info_data);
    std::cout << "si_spilit_number:" << si_spilit_number << std::endl; //应该输出2
    int sj_spilit_number = para_gene_tool.init_operetor_splitnumber(sj,info_data);
    std::cout << "sj_spilit_number:" << sj_spilit_number << std::endl; //应该输出5
    int sk_spilit_number = para_gene_tool.init_operetor_splitnumber(sk,info_data);
    std::cout << "sk_spilit_number:" << sk_spilit_number << std::endl; //应该输出3

}




// 下面这个测试已经改为使用新的参数生成工具测试
// int main()
// {
//     std::vector<int> data{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}; //在运行时会先从文件中读取
//     //std::vector<int> shape{4,5};//在运行时会得到

//     //使用新的Tensor
//     dacpp::Tensor<int,2> myTensor({4,5},data);

//     myTensor.print();
//     RegularSlice si = RegularSlice("si", 2, 2); //这些通过dacpp的抽象语法树可以分析得到
//     si.setDimId(0); //这个通过分析抽象语法树得到
//     Index sj = Index("sj");
//     sj.setDimId(1);
//     RegularSlice sk = RegularSlice("sk", 3, 1); //这些通过dacpp的抽象语法树可以分析得到
//     sk.setDimId(1); //这个通过分析抽象语法树得到

//     ParameterGeneration<int,2> para_gene_tool;
//     int si_spilit_number = para_gene_tool.init_operetor_splitnumber(si,myTensor);
//     std::cout << "si_spilit_number:" << si_spilit_number << std::endl; //应该输出2
//     int sj_spilit_number = para_gene_tool.init_operetor_splitnumber(sj,myTensor);
//     std::cout << "sj_spilit_number:" << sj_spilit_number << std::endl; //应该输出5
//     int sk_spilit_number = para_gene_tool.init_operetor_splitnumber(sk,myTensor);
//     std::cout << "sk_spilit_number:" << sk_spilit_number << std::endl; //应该输出3

// }