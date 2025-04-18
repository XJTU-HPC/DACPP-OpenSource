#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include "sub_template.h"

//这个文件调用的模板已经废弃 先注释了

void debug(int num){
    std::cout<<"ok"<<num<<"\n";
}
int main(){
	// std::cout<<"******************vec*scalar CodeGen test******************\n\n";
	// Index i = Index("i");
	// i.SetSplitSize(3);
	
	// Dac_Ops ops;
	// ops.push_back(i);
	// std::string IndexInit = CodeGen_IndexInit(ops);

	// Dac_Ops vecA_ops;
	// i.setDimId(0);
	// i.setSplitLength(1);
	// vecA_ops.push_back(i);
	// DacData d_vecA = DacData("d_vecA",1,vecA_ops);
	// // debug(0);
	// Dac_Ops alpha_ops;
	// // debug(1);
	// DacData d_alpha = DacData("d_alpha",1,alpha_ops);
	
	// Args args = Args();
	// args.push_back(d_vecA);
	// args.push_back(d_alpha);
	// std::string CalcEmbed = CodeGen_CalcEmbed("mul",args);
	// // debug(2);
	// std::string KernelExecute = CodeGen_KernelExecute("3",IndexInit,CalcEmbed);
	// // debug(3);
	// std::cout<<KernelExecute;
	// return 0;
}