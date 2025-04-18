//用来测试如果数据没有算子嵌入计算的情形

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include "sub_template.h"

//文件调用的模板已经废弃 先注释

int main(){
	// std::cout<<"******************dac2sycl CodeGen test******************\n\n";
	// Index i = Index("i");
	// i.SetSplitSize(3);
	// Index j = Index("j");
	// j.SetSplitSize(3);
	
	// Dac_Ops ops;
	// ops.push_back(i);
	// ops.push_back(j);
	// std::string IndexInit = CodeGen_IndexInit(ops);

	// Dac_Ops vecA_ops;
	// i.setDimId(0);
	// i.setSplitLength(1);
	// vecA_ops.push_back(i);
	// DacData d_vecA = DacData("d_vecA",1,vecA_ops);
	// // d_vecA.setDimLength(0,3);

	// Dac_Ops vecB_ops;
	// j.setDimId(0);
	// j.setSplitLength(1);
	// vecB_ops.push_back(j);
	// DacData d_vecB = DacData("d_vecB",1,vecB_ops);
	// // d_vecB.setDimLength(0,3);

	// Dac_Ops dotProduct_ops;
	// i.setSplitLength(3);
	// j.setSplitLength(1);
	// j.setDimId(1);
	// dotProduct_ops.push_back(i);
	// dotProduct_ops.push_back(j);
	// DacData d_dotProduct = DacData("d_dotProduct",2,dotProduct_ops);
	// // d_dotProduct.setDimLength(0,3);
	// // d_dotProduct.setDimLength(1,3);

    // //只修改下面的部分
    // Dac_Ops vecA2_ops;
	// DacData d_vecA2 = DacData("d_vecA2",1,vecA2_ops);

    // Dac_Ops vecB2_ops;
	// DacData d_vecB2 = DacData("d_vecB2",1,vecB2_ops);

	// Dac_Ops d_dotProduct2_ops;
	// DacData d_dotProduct2 = DacData("d_dotProduct2",2,d_dotProduct2_ops);



	// Args args = Args();
	// args.push_back(d_vecA2);
	// args.push_back(d_vecB2);
	// args.push_back(d_dotProduct2);
	// std::string CalcEmbed = CodeGen_CalcEmbed("mat_mul",args);

	// std::string KernelExecute = CodeGen_KernelExecute("9",IndexInit,CalcEmbed);

	// std::cout<<KernelExecute;
	// return 0;
}