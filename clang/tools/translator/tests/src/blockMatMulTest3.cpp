#include "sub_template.h"
//该文件调用的是废弃的模板 暂时注释掉
int main(){
	// std::cout<<"************************************dac2sycl blockMatMul CodeGen test************************************\n\n";
	
    // // 设备内存分配
    // std::string deviceMemAlloc = CodeGen_DeviceMemAlloc("int","matA", "16");
    // deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matB", "16");
    // deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matC", "16*2*2");
	// deviceMemAlloc += CodeGen_DeviceMemAllocReduction("int","matC", "16*2");
    // // std::cout<<deviceMemAlloc;

	// // 算子初始化
	// std::string opInit = CodeGen_RegularSliceInit("si","2","2","2");
	// opInit += CodeGen_RegularSliceInit("sj","2","2","2");
	// opInit += CodeGen_RegularSliceInit("sk","2","2","2");
	// opInit += CodeGen_IndexInit("i","2");
	// opInit += CodeGen_IndexInit("j","2");
	// opInit += CodeGen_IndexInit("k","2");

	// // 数据关联计算
	// std::string opPushBack_A = CodeGen_OpPushBack2Ops("matA","si","0","8");
	// opPushBack_A += CodeGen_OpPushBack2Ops("matA","sk","1","4");
	// std::string dataOpsInit_A = CodeGen_DataOpsInit("matA", opPushBack_A);
	// std::string dataRecon = CodeGen_DataReconstruct("int","matA", "16", dataOpsInit_A);

	// std::string opPushBack_B = CodeGen_OpPushBack2Ops("matB","sk","0","8");
	// opPushBack_B += CodeGen_OpPushBack2Ops("matB","sj","1","4");
	// std::string dataOpsInit_B = CodeGen_DataOpsInit("matB", opPushBack_B);
	// dataRecon += CodeGen_DataReconstruct("int","matB", "16", dataOpsInit_B);

	// std::string opPushBack_C = CodeGen_OpPushBack2Ops("matC","si","0","16");
	// opPushBack_C += CodeGen_OpPushBack2Ops("matC","sj","1","8");
	// opPushBack_C += CodeGen_OpPushBack2Ops("matC","sk","2","4");
	// std::string dataOpsInit_C = CodeGen_DataOpsInit("matC", opPushBack_C);
	// dataRecon += CodeGen_DataReconstruct("int","matC", "32", dataOpsInit_C);
    
    // std::string dac_1 = CodeGen_DataAssocComp(dataRecon, "", "", "", "");

	// // 数据关联计算
	// std::string toolPushBack_A = CodeGen_OpPushBack2Tool("matA","i","0","2");
	// std::string dataRecon1 = CodeGen_DataReconstructOpPush("matA", toolPushBack_A);
	// // std::cout<<toolPushBack_A;
	// // std::cout<<dataRecon1;
	// std::string toolPushBack_B = CodeGen_OpPushBack2Tool("matB","j","1","2");
	// dataRecon1 += CodeGen_DataReconstructOpPush("matB", toolPushBack_B);
	// std::string toolPushBack_C = CodeGen_OpPushBack2Tool("matC","i","0","2");
	// toolPushBack_C += CodeGen_OpPushBack2Tool("matC","j","1","1");
	// dataRecon1 += CodeGen_DataReconstructOpPush("matC", toolPushBack_C);

	// std::string dac_2 = CodeGen_DataAssocComp(dataRecon1, "", "", "", "");

	// // 数据关联计算
	// std::string toolPushBack_A_1 = CodeGen_OpPushBack2Tool("matA","k","1","1");
	// std::string dataRecon2 = CodeGen_DataReconstructOpPush("matA", toolPushBack_A_1);
	// std::string toolPushBack_B_1 = CodeGen_OpPushBack2Tool("matB","K","0","1");
	// dataRecon2 += CodeGen_DataReconstructOpPush("matB", toolPushBack_B_1);
	
	// // 主机数据移动至设备
    // std::string H2DMemMove = CodeGen_H2DMemMov("int","matA", "16");
    // H2DMemMove += CodeGen_H2DMemMov("int","matB","16");
	// H2DMemMove += CodeGen_H2DMemMov("int","matc","16");
	// // std::cout<<H2DMemMove;

	// // 为了统一，默认降维算子的stride与size均为1
	// RegularSlice si = RegularSlice("si", 2, 2);
	// si.SetSplitSize(2);
	// RegularSlice sj = RegularSlice("sj", 2, 2);
	// sj.SetSplitSize(2);
	// RegularSlice sk = RegularSlice("sk", 2, 2);
	// sk.SetSplitSize(2);
	// Index i = Index("i");
	// i.SetSplitSize(2);
	// Index j = Index("j");
	// j.SetSplitSize(2);
	// Index k = Index("k");
	// k.SetSplitSize(2);
	// Dac_Ops ops;
	// ops.push_back(si);
	// ops.push_back(sj);
	// ops.push_back(sk);
	// ops.push_back(i);
	// ops.push_back(j);
	// ops.push_back(k);
	// std::string IndexInit = CodeGen_IndexInit(ops);

	// // 嵌入计算 需要算子的作用维度 算子的每份划分长度
	// Dac_Ops matA_ops;
	// si.setDimId(0);
	// si.setSplitLength(8);
	// matA_ops.push_back(si);
	// sk.setDimId(1);
	// sk.setSplitLength(4);
	// matA_ops.push_back(sk);
	// i.setDimId(0);
	// i.setSplitLength(2);
	// matA_ops.push_back(i);
	// k.setDimId(1);
	// k.setSplitLength(1);
	// matA_ops.push_back(k);
	// DacData d_matA = DacData("d_matA", 2, matA_ops);

	// Dac_Ops matB_ops;
	// sk.setDimId(0);
	// sk.setSplitLength(8);
	// matB_ops.push_back(sk);
	// sj.setDimId(1);
	// sj.setSplitLength(4);
	// matB_ops.push_back(sj);
	// j.setDimId(1);
	// j.setSplitLength(2);
	// matB_ops.push_back(j);
	// k.setDimId(0);
	// k.setSplitLength(1);
	// matB_ops.push_back(k);
	// DacData d_matB = DacData("d_matB", 2, matB_ops);

	// Dac_Ops matC_ops;
	// si.setDimId(0);
	// si.setSplitLength(32);
	// matC_ops.push_back(si);
	// sj.setDimId(1);
	// sj.setSplitLength(16);
	// matC_ops.push_back(sj);
	// sk.setDimId(2);
	// sk.setSplitLength(8);
	// matC_ops.push_back(sk);
	// i.setDimId(0);
	// i.setSplitLength(4);
	// matC_ops.push_back(i);
	// j.setDimId(1);
	// j.setSplitLength(2);
	// matC_ops.push_back(j);
	// k.setDimId(3);
	// k.setSplitLength(1);
	// matC_ops.push_back(k);
	// DacData d_matC = DacData("d_matC", 4, matC_ops);

	// Args args = Args();
	// args.push_back(d_matA);
	// args.push_back(d_matB);
	// args.push_back(d_matC);

	// std::string CalcEmbed = CodeGen_CalcEmbed("block_mat_mul",args);
	// std::string KernelExecute = CodeGen_KernelExecute("64",IndexInit,CalcEmbed);
	// // std::cout<<KernelExecute;
	// std::string Reduction = CodeGen_Reduction_Span("32","2","1","matC","int","sycl::plus<>()");
	// Reduction += CodeGen_Reduction_Span("16","2","4","matC","int","sycl::plus<>()");
	
	// std::string D2HMemMove = CodeGen_D2HMemMov("matC","int","16",true);

	// std::string dac_3 = CodeGen_DataAssocComp(dataRecon2, H2DMemMove, KernelExecute, Reduction, D2HMemMove);


	// // 数据关联计算
	// std::string toolPopBack_A = CodeGen_OpPopFromTool("matA");
	// std::string dataRecon3 = CodeGen_DataReconstructOpPop("matA", toolPopBack_A);
	// std::string toolPopBack_B = CodeGen_OpPopFromTool("matB");
	// dataRecon3 +=  CodeGen_DataReconstructOpPop("matB", toolPopBack_B);
	// std::string dac_4 = CodeGen_DataAssocComp(dataRecon3, "", "", "", "");

	// // 数据关联计算
	// std::string toolPopBack_A_1 = CodeGen_OpPopFromTool("matA");
	// std::string dataRecon4 = CodeGen_DataReconstructOpPop("matA", toolPopBack_A_1);
	// std::string toolPopBack_B_1 = CodeGen_OpPopFromTool("matB");
	// dataRecon4 +=  CodeGen_DataReconstructOpPop("matB", toolPopBack_B_1);
	// std::string toolPopBack_C_1 = CodeGen_OpPopFromTool("matC");
	// toolPopBack_C_1 += CodeGen_OpPopFromTool("matC");
	// dataRecon4 +=  CodeGen_DataReconstructOpPop("matC", toolPopBack_C_1);
	// std::string dac_5 = CodeGen_DataAssocComp(dataRecon4, "", "", "", "");

	// std::string MemFree = CodeGen_MemFree("d_matA");
	// MemFree += CodeGen_MemFree("d_matB");
	// MemFree += CodeGen_MemFree("d_matC");
	// MemFree += CodeGen_MemFree("reduction_matC");

	// std::string res = CodeGen_DAC2SYCL(
	// 	"blockMatMul",
	// 	"(dacpp::Tensor<int> &matA, dacpp::Tensor<int> &matB, dacpp::Tensor<int> &matC)",
	// 	deviceMemAlloc,
	// 	opInit,
	// 	dac_1+dac_2+dac_3+dac_4+dac_5,
	// 	MemFree);
	// std::cout<<res;
}