// #include "sub_template.h"

// //软编码测试分块矩阵相乘
// int main(){
// 	std::cout<<"************************************dac2sycl blockMatMul CodeGen test************************************\n\n";

//     //dacpp::list dataList{matA[si][sk], matB[sk][sj], matC[si][sj]}; 注意顺序是这样的
//     //先初始化算子
//     std::string opInit = CodeGen_RegularSliceInit2("si","2","2","0","matA");
// 	opInit += CodeGen_RegularSliceInit2("sj","2","2","1","matB");
// 	opInit += CodeGen_RegularSliceInit2("sk","2","2","1","matA");//算子作用于不同的数据时，划分数应该是相同的
//     //std::cout << opInit;

//     //要计算matA数据重组后的内存（matA设备分配的内存的多少），需要得到作用于matA的算子组
//     //所以先计算matA的算子组
//     std::string add2ops_matA = CodeGen_AddOp2Ops("si","0","matA_ops"); //算子名 算子作用维度 算子加到的算子组的名字
//     add2ops_matA += CodeGen_AddOp2Ops("sk","1","matA_ops");
//     std::string matA_ops_ = CodeGen_DataOpsInit2("matA_ops",add2ops_matA); //算子组名 添加算子的一堆代码
//     //std::cout << matA_ops_;

//     //matB的算子组
//     std::string add2ops_matB = CodeGen_AddOp2Ops("sk","0","matB_ops");
//     add2ops_matB += CodeGen_AddOp2Ops("sj","1","matB_ops");
//     std::string matB_ops_ = CodeGen_DataOpsInit2("matB_ops",add2ops_matB);
//     //std::cout << matB_ops_;

//     //计算matC时需要两个算子组 这个是那个需要乘n的大小
//     //Inops 输入的算子组 抽象语法树负责去重
//     //注意顺序 dacpp::list dataList{matA[si][sk], matB[sk][sj], matC[si][sj] 的情况下顺序应该是 i j k
//     //注意将算子添加进算子组的顺序
//     std::string add2ops_Inops = CodeGen_AddOp2Ops("si","0","In_ops");
//     add2ops_Inops += CodeGen_AddOp2Ops("sj","1","In_ops");
//     add2ops_Inops += CodeGen_AddOp2Ops("sk","2","In_ops");//注意这里维度是2 （虽然不知道这个是为什么 和分块矩阵乘有关系）
//     std::string In_ops_ = CodeGen_DataOpsInit2("In_ops",add2ops_Inops);
//     //std::cout << In_ops_;

//     //Outops 输出的算子组 
//     std::string add2ops_Outops = CodeGen_AddOp2Ops("si","0","Out_ops"); 
//     add2ops_Outops += CodeGen_AddOp2Ops("sj","1","Out_ops");
//     std::string Out_ops_ = CodeGen_DataOpsInit2("Out_ops",add2ops_Outops);
//     //std::cout << Out_ops_;    

//     //计算归约时分配的内存大小 使用matC重组后的大小表示 计算原理和matA matB是一样的
//     std::string add2ops_reduction = CodeGen_AddOp2Ops("si","0","recdution_ops");
//     add2ops_reduction += CodeGen_AddOp2Ops("sj","1","reduction_ops");
//     std::string reduction_ops_ = CodeGen_DataOpsInit2("reduction_ops",add2ops_reduction);
//     //std::cout << reduction_ops_;
//     std::string InitOPS = matA_ops_ + matB_ops_ + In_ops_ + Out_ops_ + reduction_ops_;

//     //生成设备分配内存的大小 matA_size matB_size matC_size后面需要反复用到 注意区分
//     std::string divice_memory = CodeGen_DeviceMemSizeGenerate("matA_size","matA","matA_ops");//设备内存分配的名字 对应的tensor的名字 作用在这个tensor的算子组 
//     divice_memory += CodeGen_DeviceMemSizeGenerate("matB_size","matB","matB_ops");
//     divice_memory += CodeGen_DeviceMemSizeGenerate("matC_size","In_ops","Out_ops","matC");
//     divice_memory += CodeGen_DeviceMemSizeGenerate("reduction_size","matC","reduction_ops");
//     //std::cout << divice_memory;

//     //生成算子的划分长度 注意名字要和前面保持一致
//     std::string init_spilitlength = CodeGen_Init_Split_Length("matA_ops","matA_size");
//     init_spilitlength += CodeGen_Init_Split_Length("matB_ops","matB_size");
//     init_spilitlength += CodeGen_Init_Split_Length("In_ops","matC_size");
//     init_spilitlength += CodeGen_Init_Split_Length("reduction_ops","reduction_size");
//     //std::cout << init_spilitlength;

//     //生成算子长度的矩阵
//     std::string AddDacOps2Vector = CodeGen_Add_DacOps2Vector("ops_s","matA_ops");
//     AddDacOps2Vector += CodeGen_Add_DacOps2Vector("ops_s","matB_ops");
//     AddDacOps2Vector += CodeGen_Add_DacOps2Vector("ops_s","In_ops");
//     std::string DeclareDacOpsVector = CodeGen_Declare_DacOps_Vector("ops_s",AddDacOps2Vector);
//     std::string InitSpilitLengthMatrix = CodeGen_Init_Split_Length_Matrix(DeclareDacOpsVector,"3","3","ops_s");//注意行和列由后端进行填充 否则加大了SYCL代码复杂度
//     //std::cout << InitSpilitLengthMatrix;

//     //生成分配工作项数量的多少
//     std::string item_number = CodeGen_Init_Work_Item_Number("item_size","in_ops");
//     //std::cout << item_number;

//     //生成归约中Split_size中的大小
//     std::string Init_Reduction_Split_Size = CodeGen_Init_Reduction_Split_Size("reduction_split_size","In_ops","Out_ops");
//     //std::cout << Init_Reduction_Split_Size;

//     //生成归约中Split_length大小
//     std::string Init_Reduction_Split_Length = CodeGen_Init_Reduction_Split_Length("reduction_split_length","Out_ops");
//     //std::cout << Init_Reduction_Split_Length;

//     std::string ParameterGenerate = CodeGen_ParameterGenerate(InitOPS,divice_memory,init_spilitlength,InitSpilitLengthMatrix,item_number,Init_Reduction_Split_Size,Init_Reduction_Split_Length); 


//     /*blockMatMulTest2.cpp 源码*/
//     // 设备内存分配
//     // std::string deviceMemAlloc = CodeGen_DeviceMemAlloc("int","matA", "16");
//     std::string deviceMemAlloc = CodeGen_DeviceMemAlloc("int","matA", "matA_size");
//     // deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matB", "16");
//     deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matB", "matB_size");
//     // deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matC", "16*2");
//     deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matC", "matC_size");
// 	// deviceMemAlloc += CodeGen_DeviceMemAllocReduction("int","matC", "16");
//     deviceMemAlloc += CodeGen_DeviceMemAllocReduction("int","matC", "reduction_size");
//     //std::cout<<deviceMemAlloc;

//     // 主机数据移动至设备
//     std::string H2DMemMove = CodeGen_H2DMemMov("int","matA", "matA_size");
//     H2DMemMove += CodeGen_H2DMemMov("int","matB","matB_size");
// 	//std::cout<<H2DMemMove;

//     //这些算子的索引和步长由抽象语法树提供，这些算子的划分数在算子初始化时已经算过
//     // 索引初始化 得到在工作项中的索引值 需要算子的步长stride与size
// 	// 为了统一，默认降维算子的stride与size均为1
// 	RegularSlice si = RegularSlice("si", 2, 2);
// 	//si.SetSplitSize(2);//抽象语法树已经无法提供划分数信息
// 	RegularSlice sj = RegularSlice("sj", 2, 2);
// 	//sj.SetSplitSize(2);
// 	RegularSlice sk = RegularSlice("sk", 2, 2);
// 	//sk.SetSplitSize(2);
// 	Dac_Ops ops;
// 	ops.push_back(si);
// 	ops.push_back(sj);
// 	ops.push_back(sk);
// 	//std::string IndexInit = CodeGen_IndexInit(ops);//这个之前测试是使用旧的索引生成模板 现在换成新的
//     //因为这里面没有binding 所以三个算子分别属于三个集合 并且相对于集合的偏移量是零
//     std::vector<std::string> sets = {"id0" , "id1" , "id2"};//抽象语法树提供
//     std::vector<std::string> offsets = {"0" , "0" , "0"};//抽象语法树提供
//     std::string IndexInit = CodeGen_IndexInit2(ops,sets,offsets);
//     //std::cout << IndexInit;

//     //嵌入计算 需要算子的作用维度 算子的每份划分长度
// 	Dac_Ops matA_ops;
// 	si.setDimId(0);
// 	//si.setSplitLength(8); //抽象语法树无法提供划分长度
// 	matA_ops.push_back(si);
// 	sk.setDimId(1);
// 	//sk.setSplitLength(4);
// 	matA_ops.push_back(sk);
// 	DacData d_matA = DacData("d_matA", 2, matA_ops);//数据名称 数据维度 算子组

// 	Dac_Ops matB_ops;
// 	sk.setDimId(0);
// 	//sk.setSplitLength(8);
// 	matB_ops.push_back(sk);
// 	sj.setDimId(1);
// 	//sj.setSplitLength(4);
// 	matB_ops.push_back(sj);
// 	DacData d_matB = DacData("d_matB", 2, matB_ops);

// 	Dac_Ops matC_ops;
// 	si.setDimId(0);
// 	//si.setSplitLength(16);
// 	matC_ops.push_back(si);
// 	sj.setDimId(1);
// 	//sj.setSplitLength(8);
// 	matC_ops.push_back(sj);
// 	sk.setDimId(2);
// 	//sk.setSplitLength(4);
// 	matC_ops.push_back(sk);
// 	DacData d_matC = DacData("d_matC", 3, matC_ops);

// 	Args args = Args();
// 	args.push_back(d_matA);
// 	args.push_back(d_matB);
// 	args.push_back(d_matC);

//     // std::string CalcEmbed = CodeGen_CalcEmbed2("block_mat_mul",args);//注意调用了新的嵌入计算模板
// 	// std::string KernelExecute = CodeGen_KernelExecute("item_size",IndexInit,CalcEmbed);//注意这里面填的size的大小需要是前面算出来的大小
    
//     std::string CalcEmbed = "";
// 	std::string KernelExecute = "";
// 	//std::cout<<KernelExecute;

//     //std::string Reduction = CodeGen_Reduction("8","matC","int","sycl::plus<>()");//这个调用的是已经弃用的归约模板
//     //下面填归约模板注意：span_size就是归约分配的内存的大小 SpilitSize是in_ops的划分数除以out_ops的划分数(前面有计算) SplitLength就是结果数据的最后一个算子的划分长度
//     //存在问题：两次归约的话应该怎么办
//     std::string Reduction = CodeGen_Reduction_Span("reduction_size","reduction_split_size","reduction_split_length","matC","int","sycl::plus<>()");
// 	std::string D2HMemMove = CodeGen_D2HMemMov("matC","int","1",true);

//     //下面这块算子的划分长度还没有去掉 数据重组不太懂还未改
// 	std::string opPushBack_A = CodeGen_OpPushBack2Ops("matA","si","0","8");
// 	opPushBack_A += CodeGen_OpPushBack2Ops("matA","sk","1","4");
// 	std::string dataOpsInit_A = CodeGen_DataOpsInit("matA", opPushBack_A);
// 	std::string dataRecon = CodeGen_DataReconstruct("int","matA", "matA_size", dataOpsInit_A);

// 	std::string opPushBack_B = CodeGen_OpPushBack2Ops("matB","sk","0","8");
// 	opPushBack_B += CodeGen_OpPushBack2Ops("matB","sj","1","4");
// 	std::string dataOpsInit_B = CodeGen_DataOpsInit("matB", opPushBack_B);
// 	dataRecon += CodeGen_DataReconstruct("int","matB", "matB_size", dataOpsInit_B);

// 	std::string opPushBack_C = CodeGen_OpPushBack2Ops("matC","si","0","16");
// 	opPushBack_C += CodeGen_OpPushBack2Ops("matC","sj","1","8");
// 	opPushBack_C += CodeGen_OpPushBack2Ops("matC","sk","2","4");
// 	std::string dataOpsInit_C = CodeGen_DataOpsInit("matC", opPushBack_C);
// 	dataRecon += CodeGen_DataReconstruct("int","matC", "matC_size", dataOpsInit_C);

//     std::string MemFree = CodeGen_MemFree("d_matA");
// 	MemFree += CodeGen_MemFree("d_matB");
// 	MemFree += CodeGen_MemFree("d_matC");

//     std::string dac = CodeGen_DataAssocComp(dataRecon, H2DMemMove, KernelExecute, Reduction, D2HMemMove);

// 	std::string res = CodeGen_DAC2SYCL2(
// 		"blockMatMul",
// 		"(dacpp::Tensor<int,2> &matA, dacpp::Tensor<int,2> &matB, dacpp::Tensor<int,2> &matC)",
//         opInit,
//         ParameterGenerate,
// 		deviceMemAlloc,
// 		dac,
// 		MemFree);
// 	std::cout<<res;
//     /*blockMatMulTest2.cpp 源码*/



// }