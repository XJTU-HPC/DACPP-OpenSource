#ifndef SUB_TEMPLATE_H
#define SUB_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include <unordered_map>
#include <set>
#include"dacInfo.h"

/*
	文本替换。
	将 text 中的 find 换成 replace。
*/
void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);
/*
	文本替换。
	对 templ 做 replacements 中的文本替换。
*/
std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

std::string CodeGen_DataAssocComp(std::string dataRecon, std::string H2DMemMove, std::string kernelExecute, std::string reduction, std::string D2HMemMove);

//std::string CodeGen_DAC2SYCL(std::string dacShellName, std::string dacShellParams, std::string deviceMemAlloc, std::string opInit, std::string dataAssocComp, std::string memFree);

// aborted 下面这个sub.cpp文件中的实现已经注释掉了 这里还没有注释是因为rewriter还在用 但是不影响最终的测试结果
std::string CodeGen_DAC2SYCL(std::string dacShellName,std::string dacShellParams,std::string opInit,std::string dataRecon,std::string deviceMemAlloc,std::string H2DMemMove,std::string kernelExecute,std::string reduction,std::string D2HMemMove,std::string memFree);

std::string CodeGen_DataInfoInit(std::string name);

std::string CodeGen_RegularSliceInit(std::string opName,std::string size,std::string stride,std::string splitSize);

std::string CodeGen_IndexInit(std::string opName,std::string splitSize);

std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId);

std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId, std::string splitLength);

//下面这个参数有一个是算子的划分长度 这个是多余的
//std::string CodeGen_OpPushBack2Tool(std::string name, std::string opName, std::string dimId, std::string splitLength);

//重新写
std::string CodeGen_OpPushBack2Tool(std::string name, std::string opName, std::string dimId);

std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack);

std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit);

std::string CodeGen_DataReconstructOpPush(std::string name,std::string opPushBack2Tool);

std::string CodeGen_OpPopFromTool(std::string name);

std::string CodeGen_DataReconstructOpPop(std::string name,std::string opPopFromTool);

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size);

std::string CodeGen_DeviceMemAllocReduction(std::string  type,std::string name,std::string size);

std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size);

std::string CodeGen_DeviceDataInit(std::string type,std::string name,std::string size);

//aborted 下面这个sub.cpp文件中的实现已经注释掉了 这里还没有注释是因为rewriter还在用 但是不影响最终的测试结果
std::string CodeGen_KernelExecute(std::string SplitSize,std::string IndexInit,std::string CalcEmbed);

std::string CodeGen_KernelExecute(std::string SplitSize,std::string AccessorInit,std::string IndexInit,std::string CalcEmbed);

//aborted
//std::string CodeGen_IndexInit(Dac_Ops ops);

//aborted 下面这个sub.cpp文件中的实现已经注释掉了 这里还没有注释是因为rewriter还在用 但是不影响最终的测试结果
std::string CodeGen_IndexInit(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets);

//aborted 下面这个sub.cpp文件中的实现已经注释掉了 这里还没有注释是因为rewriter还在用 但是不影响最终的测试结果
std::string CodeGen_CalcEmbed(std::string Name,Args args);

std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names);

//aborted
//std::string CodeGen_Reduction(std::string SplitSize,std::string Name,std::string Type,std::string ReductionRule);

std::string CodeGen_Reduction_Span(std::string SPAN_SIZE,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule);

std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction);

std::string CodeGen_MemFree(std::string Name);

//下面是新增的

std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams,std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp, std::string memFree);

std::string CodeGen_RegularSliceInit2(std::string opName,std::string size,std::string stride,std::string dim_id,std::string tensor_name,bool Redefinition);

std::string CodeGen_RegularSliceInit2(std::string opName,std::string size,std::string stride,std::string dim_id,std::string tensor_name);

std::string CodeGen_IndexInit2(std::string opName,std::string dim_id,std::string DATA_INFO_NAME);

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME,std::string DACOPS_NAME);

//std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string TENSOR_NAME,std::string DACOPS_NAME);

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME);

//std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string TENSOR_NAME);

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string IN_DAC_OPS_NAME,std::string OUT_DAC_OPS_NAME,std::string DATA_INFO_NAME);

//std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string IN_DAC_OPS_NAME,std::string OUT_DAC_OPS_NAME,std::string TENSOR_OUT);

std::string CodeGen_AddOp2Ops(std::string OP_NAME,std::string DIM_ID,std::string OPS_NAME);

std::string CodeGen_DataOpsInit2(std::string OPS_NAME,std::string ADD_OP2OPS);

std::string CodeGen_Init_Split_Length(std::string OPS_NAME,std::string SIZE);

std::string CodeGen_Add_DacOps2Vector(std::string OPSS_NAME,std::string OPS_NAME);

std::string CodeGen_Declare_DacOps_Vector(std::string OPSS_NAME,std::string PUSH_BACK_DAC_OPS);

std::string CodeGen_Init_Split_Length_Matrix(std::string DECLARE_DACOPS_VECTOR,std::string ROW,std::string COL,std::string OPS_S_NAME);

std::string CodeGen_IndexInit2(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets);

std::string CodeGen_CalcEmbed2(std::string Name,Args args);

std::string CodeGen_Init_Work_Item_Number(std::string NAME,std::string OPS_NAME);

std::string CodeGen_Init_Reduction_Split_Size(std::string NAME,std::string OPS_IN,std::string OPS_OUT);

std::string CodeGen_Init_Reduction_Split_Length(std::string NAME,std::string OPS_NAME);

std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string InitSpilitLengthMatrix,std::string ItemNumber,std::string InitReductionSplitSize,std::string InitReductionSplitLength);

//std::string CodeGen_IndexInit2(std::string opName,std::string dim_id,std::string TENSOR_NAME);

std::string CodeGen_InitParameterTool(std::string DIM_NUM);

std::string CodeGen_AccessorInit(std::string name);
#endif