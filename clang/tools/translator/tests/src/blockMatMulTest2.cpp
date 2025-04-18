#include "sub_template.h"
//该文件调用的是废弃的模板 暂时注释掉
int main(){
	// std::cout<<"************************************dac2sycl blockMatMul CodeGen test************************************\n\n";
	
    // // 设备内存分配
    // std::string deviceMemAlloc = CodeGen_DeviceMemAlloc("int","matA", "16");
    // deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matB", "16");
    // deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matC", "16*2");
	// deviceMemAlloc += CodeGen_DeviceMemAllocReduction("int","matC", "16");
    // // std::cout<<deviceMemAlloc;
    
    // // 主机数据移动至设备
    // std::string H2DMemMove = CodeGen_H2DMemMov("int","matA", "16");
    // H2DMemMove += CodeGen_H2DMemMov("int","matB","16");
	// // std::cout<<H2DMemMove;

    // // 索引初始化 得到在工作项中的索引值 需要算子的步长stride与size
	// // 为了统一，默认降维算子的stride与size均为1
	// RegularSlice si = RegularSlice("si", 2, 2);
	// si.SetSplitSize(2);
	// RegularSlice sj = RegularSlice("sj", 2, 2);
	// sj.SetSplitSize(2);
	// RegularSlice sk = RegularSlice("sk", 2, 2);
	// sk.SetSplitSize(2);
	// Dac_Ops ops;
	// ops.push_back(si);
	// ops.push_back(sj);
	// ops.push_back(sk);
	// std::string IndexInit = CodeGen_IndexInit(ops);

	// // 嵌入计算 需要算子的作用维度 算子的每份划分长度
	// Dac_Ops matA_ops;
	// si.setDimId(0);
	// si.setSplitLength(8);
	// matA_ops.push_back(si);
	// sk.setDimId(1);
	// sk.setSplitLength(4);
	// matA_ops.push_back(sk);
	// DacData d_matA = DacData("d_matA", 2, matA_ops);

	// Dac_Ops matB_ops;
	// sk.setDimId(0);
	// sk.setSplitLength(8);
	// matB_ops.push_back(sk);
	// sj.setDimId(1);
	// sj.setSplitLength(4);
	// matB_ops.push_back(sj);
	// DacData d_matB = DacData("d_matB", 2, matB_ops);

	// Dac_Ops matC_ops;
	// si.setDimId(0);
	// si.setSplitLength(16);
	// matC_ops.push_back(si);
	// sj.setDimId(1);
	// sj.setSplitLength(8);
	// matC_ops.push_back(sj);
	// sk.setDimId(2);
	// sk.setSplitLength(4);
	// matC_ops.push_back(sk);
	// DacData d_matC = DacData("d_matC", 3, matC_ops);

	// Args args = Args();
	// args.push_back(d_matA);
	// args.push_back(d_matB);
	// args.push_back(d_matC);

	// std::string CalcEmbed = CodeGen_CalcEmbed("block_mat_mul",args);
	// std::string KernelExecute = CodeGen_KernelExecute("8",IndexInit,CalcEmbed);
	// // std::cout<<KernelExecute;
	// std::string Reduction = CodeGen_Reduction("8","matC","int","sycl::plus<>()");
	// std::string D2HMemMove = CodeGen_D2HMemMov("matC","int","1",true);

	// std::string opInit = CodeGen_RegularSliceInit("si","2","2","2");
	// opInit += CodeGen_RegularSliceInit("sj","2","2","2");
	// opInit += CodeGen_RegularSliceInit("sk","2","2","2");

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

	// std::string MemFree = CodeGen_MemFree("d_matA");
	// MemFree += CodeGen_MemFree("d_matB");
	// MemFree += CodeGen_MemFree("d_matC");

	// std::string dac = CodeGen_DataAssocComp(dataRecon, H2DMemMove, KernelExecute, Reduction, D2HMemMove);

	// std::string res = CodeGen_DAC2SYCL(
	// 	"blockMatMul",
	// 	"(dacpp::Tensor<int> &matA, dacpp::Tensor<int> &matB, dacpp::Tensor<int> &matC)",
	// 	deviceMemAlloc,
	// 	opInit,
	// 	dac,
	// 	MemFree);
	// std::cout<<res;
}