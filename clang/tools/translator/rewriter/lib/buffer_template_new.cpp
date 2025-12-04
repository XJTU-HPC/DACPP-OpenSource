#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "buffer_template_new.h"

namespace BUFFER_TEMPLATE {

void replaceTextInString(std::string& text, 
    const std::string &find, 
    const std::string &replace){
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos){
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}
std::string templateString(std::string templ, 
    std::vector<std::pair<std::string, std::string>> replacements){
	for(auto &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

//BUFFER_ACCESSOR_LIST存储了所有访问器的声明
//ACCESSOR_POINTER_LIST存储了所有访问器指针的声明 这些都是在内核中用到的
std::string BUFFER_ACCESSOR_LIST = "";
std::string ACCESSOR_POINTER_LIST = "";
//下面这个由b_name修改为r_name因为现在重组后的数据放到了r_name中
/*
const char *BUFFER_ACCESSOR_Template = R"~~~(
        accessor acc_{{NAME}}{b_{{NAME}}, h};)~~~";
*/
const char *BUFFER_ACCESSOR_Template = R"~~~(
        accessor acc_{{NAME}}{r_{{NAME}}, h};)~~~";
const char *ACCESSOR_POINTER_Template = R"~~~(
            auto* d_{{NAME}} = acc_{{NAME}}.get_multi_ptr<access::decorated::no>().get();)~~~";




//生成函数的总模板 基本包含了大致结构  buffer没有内存释放
const char *DAC2SYCL_Template_2 = R"~~~(
// 生成函数调用
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    {{OP_INIT}}
    //参数生成
	{{ParameterGenerate}}
    // 设备内存分配
    {{DEVICE_MEM_ALLOC}}
    // 数据关联计算
    {{DATA_ASSOC_COMP}}
})~~~";

std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams,std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp, std::string memFree){
    return templateString(DAC2SYCL_Template_2,
	{	
		{"{{DAC_SHELL_NAME}}",    dacShellName},
		{"{{DAC_SHELL_PARAMS}}",  dacShellParams},
		{"{{OP_INIT}}",           opInit},
		{"{{ParameterGenerate}}", parameter_generate},
		{"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
		{"{{DATA_ASSOC_COMP}}",   dataAssocComp}
	});
}




//数据信息初始化模板 这部分应该在上面的ParameterGeneration para_gene_tool;后面
//{{OP_INIT}}的前面 种种原因当时没有把这个写进去 Rewriter调用应该调用
const char *DATA_INFO_INIT_Template = R"~~~(
    // 数据信息初始化
    DataInfo info_{{NAME}};
    info_{{NAME}}.dim = {{NAME}}.getDim();
    int info_{{NAME}}_Shape[{{DIM}}] = {0};
    for(int i = 0; i < info_{{NAME}}.dim; i++)
    {
        info_{{NAME}}.dimLength.push_back({{NAME}}.getShape(i));
        info_{{NAME}}_Shape[i] = {{NAME}}.getShape(i);
    }
	)~~~";
std::string CodeGen_DataInfoInit(std::string name, std::string dim){
    return templateString(DATA_INFO_INIT_Template,
	{
		{"{{NAME}}",    name},
        {"{{DIM}}",    dim}
	});
}




//算子初始化模板 {{OP_INIT}}
//分区算子初始化
const char *OP_REGULAR_SLICE_INIT_Template2 = R"~~~(
    // 规则分区算子初始化
    RegularSlice {{OP_NAME}} = RegularSlice("{{OP_NAME}}", {{SIZE}}, {{STRIDE}});
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.SetSplitSize(para_gene_tool.init_operetor_splitnumber({{OP_NAME}},{{DATA_INFO_NAME}}));
)~~~";

std::string CodeGen_RegularSliceInit2(std::string opName,std::string size,std::string stride,std::string dim_id,std::string DATA_INFO_NAME){
    return templateString(OP_REGULAR_SLICE_INIT_Template2,
	{
		{"{{OP_NAME}}",    opName},
		{"{{SIZE}}",       size},
		{"{{STRIDE}}",     stride},
		{"{{DIM_ID}}",     dim_id}, //需要通过dimId来计算算子的划分数了
		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME}
	});
}

//降维算子初始化
const char *OP_INDEX_INIT_Template2 = R"~~~(
    // 降维算子初始化
    Index {{OP_NAME}} = Index("{{OP_NAME}}");
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.SetSplitSize(para_gene_tool.init_operetor_splitnumber({{OP_NAME}},{{DATA_INFO_NAME}}));
)~~~";

std::string CodeGen_IndexInit2(std::string opName,std::string dim_id,std::string DATA_INFO_NAME){
    return templateString(OP_INDEX_INIT_Template2,
	{
		{"{{OP_NAME}}",    opName},
		{"{{DIM_ID}}", dim_id}, //需要通过dimId来计算算子的划分数
		{"{{DATA_INFO_NAME}}", DATA_INFO_NAME}
	});
}




//参数生成的总模板  {{ParameterGenerate}}
// const char *PARA_GENE_Template = R"~~~(
//     // 参数生成 提前计算后面需要用到的参数	
// 	{{InitOPS}}
// 	{{InitDeviceMemorySize}}
// 	{{InitSplitLength}}
// 	{{InitSpilitLengthMatrix}}
// 	{{ItemNumber}}
// 	{{InitReductionSplitSize}}
// 	{{InitReductionSplitLength}}
// )~~~";
//由于规约功能暂时没用，因此去掉{{InitReductionSplitSize}}和{{InitReductionSplitLength}}：
const char *PARA_GENE_Template = R"~~~(
    // 参数生成 提前计算后面需要用到的参数	
	{{InitOPS}}
	{{InitDeviceMemorySize}}
	{{InitSplitLength}}
	{{ItemNumber}}
)~~~";

// std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string InitSpilitLengthMatrix,std::string ItemNumber,std::string InitReductionSplitSize,std::string InitReductionSplitLength){
//     return templateString(PARA_GENE_Template,
// 	{
// 		{"{{InitOPS}}", InitOPS},
// 		{"{{InitDeviceMemorySize}}", InitDeviceMemorySize},//设备内存的分配大小计算
// 		{"{{InitSplitLength}}",InitSplitLength},
// 		{"{{InitSpilitLengthMatrix}}",InitSpilitLengthMatrix},
// 		{"{{ItemNumber}}",ItemNumber},
// 		{"{{InitReductionSplitSize}}",InitReductionSplitSize},
// 		{"{{InitReductionSplitLength}}",InitReductionSplitLength}
// 	});
// }
//由于规约功能暂时没用，因此去掉{{InitReductionSplitSize}}和{{InitReductionSplitLength}}：
// std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string InitSpilitLengthMatrix,std::string ItemNumber){
//     return templateString(PARA_GENE_Template,
// 	{
// 		{"{{InitOPS}}", InitOPS},
// 		{"{{InitDeviceMemorySize}}", InitDeviceMemorySize},//设备内存的分配大小计算
// 		{"{{InitSplitLength}}",InitSplitLength},
// 		{"{{InitSpilitLengthMatrix}}",InitSpilitLengthMatrix},
// 		{"{{ItemNumber}}",ItemNumber},
// 	});
// }

std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string ItemNumber){
    return templateString(PARA_GENE_Template,
	{
		{"{{InitOPS}}", InitOPS},
		{"{{InitDeviceMemorySize}}", InitDeviceMemorySize},//设备内存的分配大小计算
		{"{{InitSplitLength}}",InitSplitLength},
		{"{{ItemNumber}}",ItemNumber},
	});
}

//{{InitOPS}}
// 算子组初始化 
const char *OPS_INIT_Template = R"~~~(
    // 算子组初始化
    Dac_Ops {{OPS_NAME}};
    {{ADD_OP2OPS}}
)~~~";

std::string CodeGen_DataOpsInit2(std::string OPS_NAME,std::string ADD_OP2OPS){
    return templateString(OPS_INIT_Template,
	{
		{"{{OPS_NAME}}",       OPS_NAME},
		{"{{ADD_OP2OPS}}",    ADD_OP2OPS}
	});
}

//将算子添加到算子组的模板 数据重组时也有添加算子到算子组的模板 每次添加都将要重新设置作用的维度
//{{ADD_OP2OPS}}
const char *ADD_OP2OPS_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OPS_NAME}}.push_back({{OP_NAME}});
)~~~";

std::string CodeGen_AddOp2Ops(std::string OP_NAME,std::string DIM_ID,std::string OPS_NAME){
    return templateString(ADD_OP2OPS_Template,
	{
		{"{{OP_NAME}}",    OP_NAME},
		{"{{DIM_ID}}",     DIM_ID},
		{"{{OPS_NAME}}",   OPS_NAME}
	});
}

//{{InitDeviceMemorySize}}
//生成设备内存分配大小的模板 对应mat[分区][分区] mat[分区][降维] mat[分区][] mat[降维][]
const char *DEVICE_MEM_SIZE_Generate_Template1 = R"~~~(
    //生成设备内存分配大小
    int {{NAME}} = para_gene_tool.init_device_memory_size({{DATA_INFO_NAME}},{{DACOPS_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME,std::string DACOPS_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template1,
	{
        {"{{NAME}}",        NAME}, //设备内存的名字 
		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME}, //tensor的名字
		{"{{DACOPS_NAME}}",        DACOPS_NAME} //算子组的名字
	});
}

//生成设备内存分配大小的模板 对应mat[][]
const char *DEVICE_MEM_SIZE_Generate_Template2 = R"~~~(
    //生成设备内存分配大小
    int {{NAME}} = para_gene_tool.init_device_memory_size({{DATA_INFO_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template2,
	{
        {"{{NAME}}",        NAME}, //设备内存的名字
		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME} //tensor的名字
	});
}

//生成设备内存分配的大小 对应数据重组需要分配的大小 
const char *DEVICE_MEM_SIZE_Generate_Template3 = R"~~~(
    //生成设备内存分配大小
    int {{NAME}} = para_gene_tool.init_device_memory_size({{IN_DAC_OPS_NAME}},{{OUT_DAC_OPS_NAME}},{{DATA_INFO_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string IN_DAC_OPS_NAME,std::string OUT_DAC_OPS_NAME,std::string DATA_INFO_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template3,
	{
		{"{{NAME}}",            NAME}, //这个名字要注意 因为要和后面的名字对应
		{"{{IN_DAC_OPS_NAME}}", IN_DAC_OPS_NAME},//输入算子组的名字
		{"{{OUT_DAC_OPS_NAME}}",OUT_DAC_OPS_NAME},//输出算子组的名字
		{"{{DATA_INFO_NAME}}",      DATA_INFO_NAME}//输出数据TENSOR的名字
	});
}

//{{InitSplitLength}}
//计算算子组里面算子的划分数
const char *INIT_SPLIT_LENGTH_Template = R"~~~(
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length({{OPS_NAME}},{{SIZE}});
)~~~";

std::string CodeGen_Init_Split_Length(std::string OPS_NAME,std::string SIZE){
    return templateString(INIT_SPLIT_LENGTH_Template,
	{
		{"{{OPS_NAME}}",       OPS_NAME},
		{"{{SIZE}}",           SIZE}//这个是重组之后的数据的大小
	});
}

//{{InitSpilitLengthMatrix}}
//生成算子划分长度的二维矩阵
const char *INIT_SPLIT_LENGTH_MATRIX_Template = R"~~~(
	{{DECLARE_DACOPS_VECTOR}}
	// 生成划分长度的二维矩阵
    int SplitLength[{{ROW}}][{{COL}}] = {0};
    para_gene_tool.init_split_length_martix({{ROW}},{{COL}},&SplitLength[0][0],{{OPS_S_NAME}});
)~~~";

std::string CodeGen_Init_Split_Length_Matrix(std::string DECLARE_DACOPS_VECTOR,std::string ROW,std::string COL,std::string OPS_S_NAME){
    return templateString(INIT_SPLIT_LENGTH_MATRIX_Template,
	{
		{"{{DECLARE_DACOPS_VECTOR}}",       DECLARE_DACOPS_VECTOR},
		{"{{ROW}}",       ROW},//行 也就是算子组组的个数 后端可以提供
		{"{{COL}}",       COL},//列 算子组中最多的算子的个数作为列
		{"{{OPS_S_NAME}}",       OPS_S_NAME}//前面声明的算子组组的名字
	});
}

//声明std::vector<Dac_Ops>
//{{DECLARE_DACOPS_VECTOR}}
const char *DECLARE_DACOPS_VECTOR_Template = R"~~~(
    std::vector<Dac_Ops> {{OPSS_NAME}};
	{{PUSH_BACK_DAC_OPS}}
)~~~";

std::string CodeGen_Declare_DacOps_Vector(std::string OPSS_NAME,std::string PUSH_BACK_DAC_OPS){
    return templateString(DECLARE_DACOPS_VECTOR_Template,
	{
		{"{{OPSS_NAME}}",           OPSS_NAME},//声明的DAC_OPS算子组组的名字
		{"{{PUSH_BACK_DAC_OPS}}",   PUSH_BACK_DAC_OPS}//要添加的算子的语句
	});
}

//将算子组添加到std::vector<Dac_ops>这个算子组的vector里面
//{{PUSH_BACK_DAC_OPS}}
const char *ADD_DACOPS2VECTOR_Template = R"~~~(
    {{OPSS_NAME}}.push_back({{OPS_NAME}});
)~~~";

std::string CodeGen_Add_DacOps2Vector(std::string OPSS_NAME,std::string OPS_NAME){
    return templateString(ADD_DACOPS2VECTOR_Template,
	{
		{"{{OPSS_NAME}}",       OPSS_NAME},//算子组vector的名字 std::vector<Dac_ops>的名字
		{"{{OPS_NAME}}",         OPS_NAME}//要添加的算子组的名字
	});
}

//{{ItemNumber}}
//计算工作项的多少
const char *INIT_WORK_ITEM_NUMBER_Template = R"~~~(
    // 计算工作项的大小
    int {{NAME}} = para_gene_tool.init_work_item_size({{OPS_NAME}});
)~~~";

std::string CodeGen_Init_Work_Item_Number(std::string NAME,std::string OPS_NAME){
    return templateString(INIT_WORK_ITEM_NUMBER_Template,
	{
		{"{{NAME}}",           NAME},
		{"{{OPS_NAME}}",       OPS_NAME}//算子组的名字
	});
}

//{{InitReductionSplitSize}}
//计算归约中split_size的大小，由于规约功能暂时没用，因此注释掉
// const char *INIT_REDUCTION_SPLIT_SIZE_Template = R"~~~(
//     // 计算归约中split_size的大小
//     int {{NAME}} = para_gene_tool.init_reduction_split_size({{OPS_IN}},{{OPS_OUT}});
// )~~~";

// std::string CodeGen_Init_Reduction_Split_Size(std::string NAME,std::string OPS_IN,std::string OPS_OUT){
//     return templateString(INIT_REDUCTION_SPLIT_SIZE_Template,
// 	{
// 		{"{{NAME}}",           NAME},//归约中spilitsize的名字
// 		{"{{OPS_IN}}",       OPS_IN},//输入算子组的名字
// 		{"{{OPS_OUT}}",     OPS_OUT}//输出算子组的名字
// 	});
// }

//{{InitReductionSplitLength}}
//计算归约中split_length的大小，由于规约功能暂时没用，因此注释掉
// const char *INIT_REDUCTION_SPLIT_LENGTH_Template = R"~~~(
//     // 计算归约中split_length的大小
//     int {{NAME}} = para_gene_tool.init_reduction_split_length({{OPS_NAME}});
// )~~~";

// std::string CodeGen_Init_Reduction_Split_Length(std::string NAME,std::string OPS_NAME){
//     return templateString(INIT_REDUCTION_SPLIT_LENGTH_Template,
// 	{
// 		{"{{NAME}}",           NAME},//归约中spilitsize的名字
// 		{"{{OPS_NAME}}",   OPS_NAME} //算子组的名字
// 	});
// }




//{{DEVICE_MEM_ALLOC}}
//设备内存分配 使用buffer模拟设备内存
//相当于说每申请一个设备内存，在BUFFER_ACCESSOR_LIST中添加访问这个buffer的访问器声明
//为什么不会多声明呢？因为在数据重组中的buffer设备声明中没有没有往BUFFER_ACCESSOR_Template添加访问器声明
const char *DEVICE_MEM_ALLOC_Template = R"~~~(
    // Buffer设备内存分配
    buffer <{{TYPE}}> b_{{NAME}}{{{SIZE}}};
)~~~";

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size){
    BUFFER_ACCESSOR_LIST += templateString(BUFFER_ACCESSOR_Template,{
        {"{{NAME}}", name}
    });
    ACCESSOR_POINTER_LIST += templateString(ACCESSOR_POINTER_Template,{
        {"{{NAME}}", name}
    });
	string s = templateString(DEVICE_MEM_ALLOC_Template,{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}});

	ACCESSOR_POINTER_LIST = " ";
    return s;
}

//目前归约还未使用
// const char *DEVICE_MEM_ALLOC_REDUCTION_Template = R"~~~(
//     // 规约Buffer设备内存分配
//     std::vector<sycl::buffer<{{TYPE}}, 1>> b_reduction_{{NAME}}({{SIZE}}, buffer<{{TYPE}}, 1>{1});
//     for(int i = 0; i < {{SIZE}}; i++){
//         host_accessor temp_accessor{b_reduction_{{NAME}}[i]};
//         temp_accessor[0] = 0;
//     })~~~";
//由于规约时不能用偏移量指示buffer,所以需要定义一个vector存结果
// std::string CodeGen_DeviceMemAllocReduction(std::string type,std::string name,std::string size){
//     return templateString(DEVICE_MEM_ALLOC_REDUCTION_Template,{
// 		{"{{TYPE}}", type},
// 		{"{{NAME}}", name},
// 		{"{{SIZE}}", size}
// 	});
// }




//{{DATA_ASSOC_COMP}}
//数据关联计算
const char *DATA_ASSOC_COMP_Template = R"~~~(
	{{H2D_MEM_MOV}}    
	{{DATA_RECON}}
	{{KERNEL_EXECUTE}}
	{{REDUCTION}}
	{{D2H_MEM_MOV}}
)~~~";

std::string CodeGen_DataAssocComp(std::string H2DMemMove, std::string dataRecon, std::string kernelExecute, std::string reduction, std::string D2HMemMove){
    return templateString(DATA_ASSOC_COMP_Template,
	{
        {"{{H2D_MEM_MOV}}",       H2DMemMove},//设备端数据重组 先移动数据到设备
		{"{{DATA_RECON}}",        dataRecon},
        {"{{KERNEL_EXECUTE}}",    kernelExecute},
		{"{{REDUCTION}}",         reduction},
        {"{{D2H_MEM_MOV}}",       D2HMemMove}
	});
}

//数据移动
//{{H2D_MEM_MOV}}
// const char *D2B_MOV_BUFFER_Template = R"~~~(
//     // 数据移动
//     {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}.getSize()*sizeof({{TYPE}}));
//     {{NAME}}.tensor2Array(h_{{NAME}});
//     {
//         host_accessor temp_accessor{b_{{NAME}}};
//         for(int i = 0; i < {{SIZE}}; i++){
//             temp_accessor[i] = h_{{NAME}}[i];
//         }
//     }
// )~~~";

// const char *D2B_MOV_BUFFER_Template = R"~~~(
//     // 数据移动
//     {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}_Size*sizeof({{TYPE}}));
//     {{NAME}}.tensor2Array(h_{{NAME}});
// 	buffer<{{TYPE}}, 1> b_{{NAME}}(h_{{NAME}}, range<1>({{NAME}}_Size));
// )~~~";

const char *D2B_MOV_BUFFER_Template = R"~~~(
    // 数据移动
    {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}.getSize()*sizeof({{TYPE}}));
    {{NAME}}.tensor2Array(h_{{NAME}});
	buffer<{{TYPE}}, 1> r_{{NAME}}(h_{{NAME}}, range<1>({{NAME}}.getSize()));
)~~~";

std::string CodeGen_D2B_Mov_Buffer(std::string TYPE,std::string NAME,std::string SIZE)
{
    return templateString(D2B_MOV_BUFFER_Template,
	{
        {"{{TYPE}}",        TYPE},   
		{"{{NAME}}",        NAME},
		{"{{SIZE}}",        SIZE}
	});
}

//用于buffer模板数据重组中申请主机内存
/*
    double* h_matNext=(double*)malloc(matNext.getSize()*sizeof(double));
*/
const char *INIT_HOST_MEMORY_Template = R"~~~(
    // 数据移动
    {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}.getSize()*sizeof({{TYPE}}));
)~~~";
std::string CodeGen_Init_Host_Memory(std::string TYPE,std::string NAME)
{
    return templateString(INIT_HOST_MEMORY_Template,
	{
        {"{{TYPE}}",        TYPE},   
		{"{{NAME}}",        NAME}
	});
}

//执行数据初始化为0
const char *DEVICE_DATA_INIT_Template = R"~~~(
    { //Buffer数据初始化
        host_accessor temp_accessor{b_{{NAME}}};
        for(int i = 0; i < {{SIZE}}; i++){
            temp_accessor[i] = 0;
        }
    }
)~~~";

std::string CodeGen_DeviceDataInit(std::string type,std::string name,std::string size){
    return templateString(DEVICE_DATA_INIT_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

//数据重组
//{{DATA_RECON}}
//数据重组的buffer模板 支持在设备端完成数据重组
//注意逻辑是h_name(主机数据)传输到b_name(设备数据)，然后b_name到r_name(重组之后的数据)
//使用r_name进行计算
// const char *DATA_RECON_BUFFER_Template = R"~~~(
//     // 数据重组
//     DataReconstructor<{{TYPE}}> {{NAME}}_tool;
//     {{DATA_OPS_INIT}}
//     {{NAME}}_tool.init(info_{{NAME}},{{NAME}}_ops,q);
//     buffer<{{TYPE}}> r_{{NAME}}{{{SIZE}}};
//     {{NAME}}_tool.Reconstruct(r_{{NAME}},b_{{NAME}},q);
// 	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
//     sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
// )~~~";

const char *DATA_RECON_BUFFER_Template = R"~~~(
    // 数据重组
    
    {{DATA_OPS_INIT}}

	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
)~~~";

std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit){
    return templateString(DATA_RECON_BUFFER_Template,
	{
		{"{{TYPE}}",       type},
		{"{{NAME}}",       name},
		{"{{SIZE}}",       size},
		{"{{DATA_OPS_INIT}}", dataOpsInit}
	});
}

// const char *DATA_RECON_BUFFER_Template1 = R"~~~(
//     // 数据重组
//     DataReconstructor<{{TYPE}}> {{NAME}}_tool;
//     {{DATA_OPS_INIT}}

//     // buffer<{{TYPE}}> r_{{NAME}}{{{SIZE}}};
//     std::vector<{{TYPE}}> init({{SIZE}}, 0);
//     sycl::buffer<{{TYPE}}> r_{{NAME}}(init.data(), sycl::range<1>({{SIZE}}));

// 	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
//     sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
// )~~~";

// const char *DATA_RECON_BUFFER_Template1 = R"~~~(
//     // 数据重组
//     {{DATA_OPS_INIT}}

//     std::vector<{{TYPE}}> {{NAME}}_init({{NAME}}.getSize(), 0);
//     sycl::buffer<{{TYPE}}> r_{{NAME}}({{NAME}}_init.data(), sycl::range<1>({{NAME}}.getSize()));

// 	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
//     sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
// )~~~";

const char *DATA_RECON_BUFFER_Template1 = R"~~~(
    // 数据重组
    {{DATA_OPS_INIT}}

    auto r_{{NAME}} = std::make_unique<sycl::buffer<{{TYPE}}, 1>>(sycl::range<1>({{NAME}}.getSize()));
    r_{{NAME}}->set_final_data(h_{{NAME}});

	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
)~~~";

std::string CodeGen_DataReconstruct1(std::string type,std::string name,std::string size,std::string dataOpsInit){
    return templateString(DATA_RECON_BUFFER_Template1,
	{
		{"{{TYPE}}",       type},
		{"{{NAME}}",       name},
		{"{{SIZE}}",       size},
		{"{{DATA_OPS_INIT}}", dataOpsInit}
	});
}

//{{DATA_OPS_INIT}}
//数据算子组初始化 用于计算数据重组时的相关数据
const char *DATA_OPS_INIT_Template = R"~~~(
    // 数据算子组初始化
    Dac_Ops {{NAME}}_ops;
    {{OP_PUSH_BACK2OPS}}
)~~~";

std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack2Ops){
    return templateString(DATA_OPS_INIT_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_PUSH_BACK2OPS}}",    opPushBack2Ops},
	});
}

//{{OP_PUSH_BACK2OPS}}
//将需要用到的算子加入到算子组
const char *OP_PUSH_BACK2OPS_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{NAME}}_ops.push_back({{OP_NAME}});)~~~";

std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId){
    return templateString(OP_PUSH_BACK2OPS_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{NAME}}",       name},
		{"{{DIM_ID}}",     dimId}
	});
}

//内核执行
//{{KERNEL_EXECUTE}}
//一维划分的Buffer内核执行模板
//和usm不同的是增加获得访问器以及获得访问器指针的操作
// const char *KERNEL_EXECUTE_Template = R"~~~(
//     sycl::device device = q.get_device();
//     int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
// 	//工作项划分
//     int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
//     sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
//     sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

//     //队列提交命令组
//     q.submit([&](handler &h) {
//     {{ACCESSOR_LIST}}
//     {{ACCESSOR_INIT}}
//         h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
//             const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
//             if(item_id >= Item_Size)
//                 return;
//             // 索引初始化
// 			{{INDEX_INIT}}
//             // 获得accessor指针
//             {{ACCESSOR_POINTER_LIST}}
//             // 嵌入计算
// 			{{CALC_EMBED}}
//         });
//     }).wait();
    
// )~~~";

//二维划分的Buffer内核执行模板
const char *KERNEL_EXECUTE_Template = R"~~~(
    sycl::device device = q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];

	// 二维划分（可测试三维拓展）
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);

    // 固定 local 为 16*16,但受设备上限约束
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);

    // 对齐 global 到 local 的整数倍（防止越界）
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);

    //队列提交命令组
    q.submit([&](handler &h) {
    {{ACCESSOR_LIST}}
    {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			{{INDEX_INIT}}
			// 获得划分数据单元左上角（第一个元素）的位置
			{{GETPOS}}
            // 获得accessor指针
            {{ACCESSOR_POINTER_LIST}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
    
)~~~";
//内核执行中的{{ACCESSOR_LIST}}，需要把A、B、C都传入，以免受到Rewriter_Buffer.cpp中第204行的判断读写的逻辑的干扰
// const char* ACCESSOR_LIST_K =  R"~~~(
//         accessor acc_{{NAME}}{r_{{NAME}}, h};)~~~";
// std::string CodeGen_AccessorInit0(std::string name){
// 	return templateString(ACCESSOR_LIST_K,{
// 		{"{{NAME}}", name}
// 	});
// }
//2025.12.4修改：对于只读的数据，将buffer的访问模式修改为只读 然后禁止数据写回操作。对于只写的数据，访问模式设置为覆盖写 同时注意 写成这样*r_matC。
const char* ACCESSOR_LIST_K_read =  R"~~~(
        accessor<{{TYPE}}, 1, access::mode::read> acc_{{NAME}}(r_{{NAME}}, h);
        r_{{NAME}}.set_final_data(nullptr);
        )~~~";
const char* ACCESSOR_LIST_K_write =  R"~~~(
        accessor<{{TYPE}}, 1, sycl::access::mode::discard_write> acc_{{NAME}}(*r_{{NAME}}, h);)~~~";
std::string CodeGen_AccessorInit0_read(std::string name, std::string type){
	return templateString(ACCESSOR_LIST_K_read,{
		{"{{NAME}}", name},
        {"{{TYPE}}", type}
	});
}
std::string CodeGen_AccessorInit0_write(std::string name, std::string type){
	return templateString(ACCESSOR_LIST_K_write,{
		{"{{NAME}}", name},
        {"{{TYPE}}", type}
	});
}
//内核执行中的{{ACCESSOR_POINTER_LIST}}，需要把A、B、C都传入，以免受到Rewriter_Buffer.cpp中第204行的判断读写的逻辑的干扰
const char* ACCESSOR_POINTER_LIST_K =  R"~~~(
            auto* d_{{NAME}} = acc_{{NAME}}.get_multi_ptr<access::decorated::no>().get();)~~~";
std::string CodeGen_AccessorInit1(std::string name){
	return templateString(ACCESSOR_POINTER_LIST_K,{
		{"{{NAME}}", name}
	});
}

const char* GETPOS_1 =  R"~~~(
			const auto {{NAME}}_{{idx}} = {{splitname}}_ * {{splitname}}.stride;)~~~";

const char* GETPOS_0 =  R"~~~(
			const auto {{NAME}}_{{idx}} = 0;)~~~";

// 获得划分数据单元左上角（第一个元素）的位置
std::string CodeGen_getpos1(std::string name, std::string opsname, std::string idx){
	return templateString(GETPOS_1,{
		{"{{NAME}}", name},
		{"{{splitname}}", opsname},
		{"{{idx}}", idx}
	});
}

std::string CodeGen_getpos0(std::string name, std::string idx){
	return templateString(GETPOS_0,{
		{"{{NAME}}", name},
		{"{{idx}}", idx}
	});
}

// std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit,std::string ACCESSOR_LIST1, std::string ACCESSOR_LIST2, std::string CalcEmbed){
//     return templateString(KERNEL_EXECUTE_Template,
// 	{
// 		{"{{SPLIT_SIZE}}",    SplitSize},
// 		{"{{INDEX_INIT}}",    IndexInit},
// 		{"{{CALC_EMBED}}",    CalcEmbed},
//         {"{{ACCESSOR_INIT}}", AccessorInit},
//         // {"{{ACCESSOR_LIST}}",   BUFFER_ACCESSOR_LIST},
// 		{"{{ACCESSOR_LIST}}",  ACCESSOR_LIST1},
//         {"{{ACCESSOR_POINTER_LIST}}",   ACCESSOR_LIST2}
// 	});
// }

std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit, std::string getpos, std::string ACCESSOR_LIST1, std::string ACCESSOR_LIST2, std::string CalcEmbed){
    return templateString(KERNEL_EXECUTE_Template,
	{
		{"{{SPLIT_SIZE}}",    SplitSize},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed},
        {"{{ACCESSOR_INIT}}", AccessorInit},
		{"{{ACCESSOR_LIST}}",  ACCESSOR_LIST1},
        {"{{ACCESSOR_POINTER_LIST}}",   ACCESSOR_LIST2},
		{"{{GETPOS}}",    getpos}
	});
}

// 访问器初始化
// {{ACCESSOR_INIT}}
// const char *ACCESSOR_INIT_Template = R"~~~(
//         auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read_write>(h);)~~~";
// std::string CodeGen_AccessorInit(std::string name) {
// 	return templateString(ACCESSOR_INIT_Template,
// 	{
// 		{"{{NAME}}",    name}
// 	});
// }
//2025.12.4修改：对于数据单元形状的访问器，将模式从读写模式修改为只读模式
const char *ACCESSOR_INIT_Template = R"~~~(
        auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read>(h);)~~~";
std::string CodeGen_AccessorInit(std::string name) {
	return templateString(ACCESSOR_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}

//索引初始化
//{{INDEX_INIT}}
const char *INDEX_INIT_Template = R"~~~(
            const auto {{NAME}}={{EXPRESSION}};)~~~";
//新的索引生成模板 相当于现在的ops能用的只有算子的名字了 算子的划分数是不会改变的
std::string CodeGen_IndexInit2(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets)//sets表示每个算子属于的集合的名字 offsets表示每个算子相对于集合的偏移量
{ 
    std::set<std::string> sets_map;//用于辅助找到不同的集合的个数
    std::vector<std::string> sets_order;//记录了不同的集合出现的顺序，储存集合的名字： idx idy idz
    std::vector<std::string> sets_split;//记录了不同集合对应的划分数，与集合名相对应： idx的划分数 idy的划分数 idz的划分数 
    for (int i = 0; i < sets.size(); ++i) 
    {
		std::string ops_i_name = ops[i].name;
        if (sets_map.find(sets[i]) == sets_map.end())//如果容器里没有
        {
            sets_map.insert(sets[i]);//将集合插入容器
            sets_order.push_back(sets[i]);//将集合放入到集合的数组中
            sets_split.push_back(ops_i_name + ".split_size");//将集合对应的划分数放入数组中
        }
    }
    
    int sets_size = sets_map.size();//得到各类集合总个数
    std::unordered_map<std::string,std::string> sets_sub_expression;//<集合的名称，集合对应的索引表达式>

    for(int i = 0;i < sets_size; i++)//有几个集合就循环几次
    {
		std::string sub_expression = "item_id";
		for(int j = i + 1;j < sets_size;j ++){
			sub_expression = sub_expression + "/" + sets_split[j];
		}
		//sub_expression = sub_expression + "%" + std::to_string(sets_split[i]);//取模操作应该在偏移之后
        sets_sub_expression[sets_order[i]] = sub_expression;//将子表达式和集合的名字进行关联
	}

    //下面根据偏移量来计算各个算子对应的索引
    int len = ops.size;
	std::vector<std::string> index_expression_vector;
    for(int i = 0;i < len;i ++)
    {
        std::string index_expression = "(";
        index_expression = index_expression + sets_sub_expression[sets[i]];//得到集合的索引
        //index_expression = index_expression + "+" + "(" + offsets[i] + ")" + "+" + std::to_string(ops[i].split_size) + ")";//加上偏移量和划分数 防止出现负数
		index_expression = index_expression + "+" + "(" + offsets[i] + ")" + ")";
		index_expression = index_expression + "%" + ops[i].name + ".split_size";
		index_expression_vector.push_back(index_expression);
    }

	std::string expression = "";
	for(int i=0;i<len;i++){
		std::string opsname = ops[i].name;
		std::string index_i_expression = index_expression_vector[i];
		expression = expression + templateString(INDEX_INIT_Template,
		{
			{"{{NAME}}", opsname + "_"},//注意这里加了下划线
			{"{{EXPRESSION}}", index_i_expression}
		});
	}
	return expression;
}

//嵌入计算
//{{CALC_EMBED}}
// const char *CALC_EMBED_Template = R"~~~(
//             {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

//waveEq(d_matCur+(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),d_matPrev+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),d_matNext+(sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]),info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);
//所有的d_name需要换成r_name
// std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names){
// 	std::string DacCalcArgs = "(";//name(参数) 中的(
// 	int len = args.size;//len 表示有几组数据
// 	for(int i=0;i<len;i++)
// 	{
// 		std::string IndexComb="(";//这个左括号代表数据在长向量中偏移量
// 		for(int j=0;j<args[i].ops.size;j++)//第j组数据中算子组中包含的算子的个数
// 		{
// 			std::string opsname = args[i].ops[j].name;//算子是char[5]类型的，这里需要转换为string后面方便使用。得到第j个算子的名字
// 			IndexComb+= opsname + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";//给算子加一个下划线 这里是为了得到算子的划分长度 具体逻辑忘了
// 			if(j!=args[i].ops.size-1) IndexComb+="+";//还没有到最后一个算子，就继续添加+号
// 		}
// 		IndexComb+=")";//添加偏移量中的右括号
// 		if(IndexComb == "()")//如果是空的话，就不要加IndexComb了
// 		{
// 			DacCalcArgs+=args[i].name;
// 		}
// 		else
// 		{
// 			DacCalcArgs+=args[i].name + "+" + IndexComb;
// 		}		
// 		DacCalcArgs += ",";//因为后面还有其他的参数，所以就继续添加 ,
// 	}
// 	//添加数据访问器的相关参数
// 	for (int z = 0; z < accessor_names.size(); z++) 
// 	{
// 		DacCalcArgs += "info_partition_" + accessor_names[z]+"_accessor";
// 		if (z == accessor_names.size() - 1) 
// 		{
// 			DacCalcArgs += ");";
// 		} 
// 		else 
// 		{
// 			DacCalcArgs += ",";
// 		}
// 	}
// 	return templateString(CALC_EMBED_Template,
// 	{
// 		{"{{DAC_CALC_NAME}}",    Name},
// 		{"{{DAC_CALC_ARGS}}",    DacCalcArgs}
// 	});
// }
const char *CALC_EMBED_Template = R"~~~(
            {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

std::string CodeGen_CalcEmbed2(std::string Name, std::vector<std::string> splits, std::vector<int> splitNum,std::vector<std::string> accessor_names) {

    std::string DacCalcArgs = "("; // calc(

    // 1) 添加d_name
        for (int z = 0; z < accessor_names.size(); z++) {
        DacCalcArgs += "d_" + accessor_names[z] +",";
    }
    // 2) 添加name_dim
        for (int z = 0; z < accessor_names.size(); z++) {
            for(int zz = 0; zz < splitNum[z]; zz++){
                DacCalcArgs += accessor_names[z] + "_" + splits[zz] +",";
            }
    }
    // 3）添加info_name_Shape[dim]
        for (int z = 0; z < accessor_names.size(); z++) {
            for(int zz = 0; zz < splitNum[z]; zz++){
                DacCalcArgs += "info_" + accessor_names[z] + "_Shape[" + splits[zz] + "],";
            }
    }
    // 4) 添加 info_partition_name_accessor
    for (int z = 0; z < accessor_names.size(); z++) {
        DacCalcArgs += "info_partition_" + accessor_names[z] + "_accessor";
        if (z != accessor_names.size() - 1) DacCalcArgs += ",";
    }

    DacCalcArgs += ");";

    return templateString(
        CALC_EMBED_Template,
        {
            {"{{DAC_CALC_NAME}}", Name},
            {"{{DAC_CALC_ARGS}}", DacCalcArgs}
        }
    );
}


//归约
//{{REDUCTION}}
//归约中将b_name修改为了r_name，因为现在是使用r_name进行计算
const char *REDUCTION_Template_Span = R"~~~(
    // 归约
    if({{SPLIT_SIZE}} > 1)
    {
        for(int i=0;i<{{SPAN_SIZE}};i++) {
            q.submit([&](handler &h) {
                accessor d_{{NAME}}{r_{{NAME}}, h};
    	        h.parallel_for(
                range<1>({{SPLIT_SIZE}}),
                reduction(b_reduction_{{NAME}}[i], h, 
                {{REDUCTION_RULE}},property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_{{NAME}}[(i/{{SPLIT_LENGTH}})*{{SPLIT_LENGTH}}*{{SPLIT_SIZE}}+i%{{SPLIT_LENGTH}}+idx*{{SPLIT_LENGTH}}]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_{{NAME}}};
            for(int i = 0; i < {{SPAN_SIZE}}; i++){
                host_accessor temp_accessor{b_reduction_{{NAME}}[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }

)~~~";

std::string CodeGen_Reduction_Span(std::string SpanSize,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule) {
    return templateString(REDUCTION_Template_Span,
	{
        {"{{SPAN_SIZE}}",        SpanSize},   
		{"{{SPLIT_SIZE}}",       SplitSize},
		{"{{SPLIT_LENGTH}}",     SplitLength},
		{"{{TYPE}}",             Type},
		{"{{NAME}}",             Name},
		{"{{REDUCTION_RULE}}",   ReductionRule}
	});
}

//结果返回
//{{D2H_MEM_MOV}}
//先将r_name中的数据逆重组到b_name,再将b_name数据传输到h_name
// const char *RESULT_B2H_MOV_Template = R"~~~(
//     //结果返回
//     {{NAME}}_tool.UpdateData(r_{{NAME}},b_{{NAME}},q,{{NAME}}_Size);
//     {
//         host_accessor temp_accessor{b_{{NAME}}};
//         for(int i = 0; i < {{SIZE}}; i++){
//             h_{{NAME}}[i] = temp_accessor[i];
//         }
//     }
//     {{NAME}}.array2Tensor(h_{{NAME}});
// )~~~";

// const char *RESULT_B2H_MOV_Template = R"~~~(
//     //结果返回
    
//     {
//         host_accessor temp_accessor{r_{{NAME}}};
//         for(int i = 0; i < {{NAME}}.getSize(); i++){
//             h_{{NAME}}[i] = temp_accessor[i];
//         }
//     }
//     {{NAME}}.array2Tensor(h_{{NAME}});
// )~~~";

const char *RESULT_B2H_MOV_Template = R"~~~(
    //结果返回语句改为析构语句
    r_{{NAME}}.reset();
    {{NAME}}.array2Tensor(h_{{NAME}});
)~~~";

std::string CodeGen_Result_B2H_Mov(std::string NAME,std::string SIZE)
{
    return templateString(RESULT_B2H_MOV_Template,
	{
		{"{{NAME}}",             NAME},
		{"{{SIZE}}",             SIZE}
	});
}

}