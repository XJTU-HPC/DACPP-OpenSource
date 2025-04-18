#ifndef UNIVERSITY_TEMPLATE_H
#define UNIVERSITY_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include <unordered_map>
#include <set>
#include"dacInfo.h"

namespace UNIVERSAL_TEMPLATE{

    extern const char *DATA_INFO_INIT_Template;
    extern const char *PARA_GENE_Template;
    extern const char *OP_REGULAR_SLICE_INIT_Template2;
    extern const char *OP_INDEX_INIT_Template2;
    extern const char *DEVICE_MEM_SIZE_Generate_Template1;
    extern const char *DEVICE_MEM_SIZE_Generate_Template2;
    extern const char *DEVICE_MEM_SIZE_Generate_Template3;
    extern const char *ADD_OP2OPS_Template;
    extern const char *OPS_INIT_Template;
    extern const char *INIT_SPLIT_LENGTH_Template;
    extern const char *ADD_DACOPS2VECTOR_Template;
    extern const char *DECLARE_DACOPS_VECTOR_Template;
    extern const char *INIT_SPLIT_LENGTH_MATRIX_Template;
    extern const char *INIT_WORK_ITEM_NUMBER_Template;
    extern const char *INIT_REDUCTION_SPLIT_SIZE_Template;
    extern const char *INIT_REDUCTION_SPLIT_LENGTH_Template;
    extern const char *DATA_ASSOC_COMP_Template;
    extern const char *DATA_RECON_Template;
    extern const char *OP_PUSH_BACK2OPS_Template;
    extern const char *OP_PUSH_BACK2TOOL_Template;
    extern const char *DATA_OPS_INIT_Template;
    extern const char *DATA_RECON_OP_PUSH_Template;
    extern const char *OP_POP_FROM_TOOL_Template;
    extern const char *DATA_RECON_OP_POP_Template;
    extern const char *INDEX_INIT_Template;
    extern const char *CALC_EMBED_Template;

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);
    
    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

    std::string CodeGen_DataInfoInit(std::string name);

    std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string InitSpilitLengthMatrix,std::string ItemNumber,std::string InitReductionSplitSize,std::string InitReductionSplitLength);

    std::string CodeGen_RegularSliceInit2(std::string opName,std::string size,std::string stride,std::string dim_id,std::string DATA_INFO_NAME);
    
    std::string CodeGen_IndexInit2(std::string opName,std::string dim_id,std::string DATA_INFO_NAME);

    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME,std::string DACOPS_NAME);

    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME);
    
    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string IN_DAC_OPS_NAME,std::string OUT_DAC_OPS_NAME,std::string DATA_INFO_NAME);
    
    std::string CodeGen_AddOp2Ops(std::string OP_NAME,std::string DIM_ID,std::string OPS_NAME);
    
    std::string CodeGen_DataOpsInit2(std::string OPS_NAME,std::string ADD_OP2OPS);

    std::string CodeGen_Init_Split_Length(std::string OPS_NAME,std::string SIZE);
    
    std::string CodeGen_Add_DacOps2Vector(std::string OPSS_NAME,std::string OPS_NAME);
    
    std::string CodeGen_Declare_DacOps_Vector(std::string OPSS_NAME,std::string PUSH_BACK_DAC_OPS);
    
    std::string CodeGen_Init_Split_Length_Matrix(std::string DECLARE_DACOPS_VECTOR,std::string ROW,std::string COL,std::string OPS_S_NAME);
    
    std::string CodeGen_Init_Work_Item_Number(std::string NAME,std::string OPS_NAME);
    
    std::string CodeGen_Init_Reduction_Split_Size(std::string NAME,std::string OPS_IN,std::string OPS_OUT);
    
    std::string CodeGen_Init_Reduction_Split_Length(std::string NAME,std::string OPS_NAME);

    std::string CodeGen_DataAssocComp(std::string dataRecon, std::string H2DMemMove, std::string kernelExecute, std::string reduction, std::string D2HMemMove);

    std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit);

    std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId);

    std::string CodeGen_OpPushBack2Tool(std::string name, std::string opName, std::string dimId);
    
    std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack2Ops);

    std::string CodeGen_DataReconstructOpPush(std::string name,std::string opPushBack2Tool);

    std::string CodeGen_OpPopFromTool(std::string name);

    std::string CodeGen_DataReconstructOpPop(std::string name,std::string opPopFromTool);

    std::string CodeGen_IndexInit2(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets);

    std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names);

}

#endif