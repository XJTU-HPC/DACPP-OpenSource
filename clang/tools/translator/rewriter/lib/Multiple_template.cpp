#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include "Multiple_template.h"

namespace MULTIPLE_TEMPLATE {

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace){
	    std::string::size_type pos = 0;
	    while ((pos = text.find(find, pos)) != std::string::npos){
		    text.replace(pos, find.length(), replace);
		    pos += replace.length();
	    }
    }
    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements){
	    for(auto &element : replacements)
		    replaceTextInString(templ, element.first, element.second);
	    return templ;
    }

//------------------------------USM-----------------------------------
//数据信息初始化模板 
const char *DATA_INFO_INIT_Template = R"~~~(
    // 数据信息初始化
    DataInfo info_{{NAME}};
    info_{{NAME}}.dim = {{NAME}}.getDim();
    for(int i = 0; i < info_{{NAME}}.dim; i++) info_{{NAME}}.dimLength.push_back({{NAME}}.getShape(i));
	)~~~";
std::string CodeGen_DataInfoInit(std::string name){
    return templateString(DATA_INFO_INIT_Template,
	{
		{"{{NAME}}",    name}
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

//将算子添加到算子组的模板 数据重组时也有添加算子到算子组的模板 每次添加都将要重新设置作用的维度
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
const char *DATA_OPS_INIT_Template = R"~~~(
    // 数据算子组初始化
    Dac_Ops {{NAME}}_ops;
    {{OP_PUSH_BACK2OPS}})~~~";

std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack2Ops){
    return templateString(DATA_OPS_INIT_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_PUSH_BACK2OPS}}",    opPushBack2Ops},
	});
}
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

//将算子组添加到std::vector<Dac_ops>这个算子组的vector里面
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

//声明std::vector<Dac_Ops>
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

//计算归约中split_size的大小
const char *INIT_REDUCTION_SPLIT_SIZE_Template = R"~~~(
    // 计算归约中split_size的大小
    int {{NAME}} = para_gene_tool.init_reduction_split_size({{OPS_IN}},{{OPS_OUT}});
)~~~";

std::string CodeGen_Init_Reduction_Split_Size(std::string NAME,std::string OPS_IN,std::string OPS_OUT){
    return templateString(INIT_REDUCTION_SPLIT_SIZE_Template,
	{
		{"{{NAME}}",           NAME},//归约中spilitsize的名字
		{"{{OPS_IN}}",       OPS_IN},//输入算子组的名字
		{"{{OPS_OUT}}",     OPS_OUT}//输出算子组的名字
	});
}

//计算归约中split_length的大小
const char *INIT_REDUCTION_SPLIT_LENGTH_Template = R"~~~(
    // 计算归约中split_length的大小
    int {{NAME}} = para_gene_tool.init_reduction_split_length({{OPS_NAME}});
)~~~";

std::string CodeGen_Init_Reduction_Split_Length(std::string NAME,std::string OPS_NAME){
    return templateString(INIT_REDUCTION_SPLIT_LENGTH_Template,
	{
		{"{{NAME}}",           NAME},//归约中spilitsize的名字
		{"{{OPS_NAME}}",   OPS_NAME} //算子组的名字
	});
}

//参数生成的总模板
const char *PARA_GENE_Template = R"~~~(
    // 参数生成 提前计算后面需要用到的参数	
	{{InitOPS}}
	{{InitDeviceMemorySize}}
	{{InitSplitLength}}
	{{InitSpilitLengthMatrix}}
	{{ItemNumber}}
	{{InitReductionSplitSize}}
	{{InitReductionSplitLength}}
)~~~";

std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string InitSpilitLengthMatrix,std::string ItemNumber,std::string InitReductionSplitSize,std::string InitReductionSplitLength){
    return templateString(PARA_GENE_Template,
	{
		{"{{InitOPS}}", InitOPS},
		{"{{InitDeviceMemorySize}}", InitDeviceMemorySize},//设备内存的分配大小计算
		{"{{InitSplitLength}}",InitSplitLength},
		{"{{InitSpilitLengthMatrix}}",InitSpilitLengthMatrix},
		{"{{ItemNumber}}",ItemNumber},
		{"{{InitReductionSplitSize}}",InitReductionSplitSize},
		{"{{InitReductionSplitLength}}",InitReductionSplitLength}
	});
}

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
    const char *DATA_SPLIT = R"~~~(
        int {{NAME}}_data_SplitNum = para_gene_tool.init_Data_SplitNum({{NAME}}_Ops,info_{{NAME}});
        int {{NAME}}_data_SplitSize = para_gene_tool.init_Data_SplitSize({{NAME}}_data_SplitNum,{{NAME}}_Size);
    )~~~";
    std::string CodeGen_DataSplit(std::string Name){
        return templateString(DATA_SPLIT,
        {
            {"{{NAME}}", Name}
        });
    }
    
    const char *DATA_DEVICE_MALLOC = R"~~~(
        std::vector<int> {{NAME}}_Device_Size = para_gene_tool.init_Device_Size(numDevices,{{NAME}}_data_SplitSize,{{NAME}}_data_SplitNum);
    )~~~";
    const char *DATA_DEVICE_MALLOC_FULL = R"~~~(
        std::vector<int> {{NAME}}_Device_Size(numDevices, {{NAME}}_Size/{{NAME}}_data_SplitSize);
    )~~~";
    std::string CodeGen_DataDeviceMalloc(std::string Name,bool isFull){
        if(isFull){
            return templateString(DATA_DEVICE_MALLOC_FULL,
            {
                {"{{NAME}}", Name}
            });
        }else{
            return templateString(DATA_DEVICE_MALLOC,
            {
                {"{{NAME}}", Name}
            });
        }
    }

    const char *DEVICE_Mem = R"~~~(
        std::vector<{{TYPE}}*> d_{{NAME}}s(numDevices);
        )~~~";
    const char *DEVICE_Mem_REDUCTION = R"~~~(
        std::vector<{{TYPE}}*> reduction_{{NAME}}s(numDevices);
        )~~~";

    std::string CodeGen_DeviceMem(std::string type,std::string name){
        return templateString(DEVICE_Mem,
        {
            {"{{TYPE}}", type},
            {"{{NAME}}", name},
        });
    }

    std::string CodeGen_DeviceMemReduction(std::string type,std::string name){
        return templateString(DEVICE_Mem_REDUCTION,
        {
            {"{{TYPE}}", type},
            {"{{NAME}}", name},
        });
    }
    const char *DEVICE_DATA_HAVEMALLOC = R"~~~(
        int {{NAME}}_HaveMalloc = 0;
    )~~~";
    std::string CodeGen_DeviceDataHaveMalloc(std::string name){
        return templateString(DEVICE_DATA_HAVEMALLOC,
        {
            {"{{NAME}}", name}
        });
    }


    const char *DATA_RECON_Template = R"~~~(
    // 数据重组
    DataReconstructor<{{TYPE}}> {{NAME}}_tool;

    {{DATA_OPS_INIT}}
    {{NAME}}_tool.init(info_{{NAME}},{{NAME}}_ops);
    {{TYPE}} recon_h_{{NAME}}[{{NAME}}_Size];
    {{NAME}}.tensor2Array(recon_h_{{NAME}});
    {{TYPE}} *recon_d_{{NAME}}=malloc_device<{{TYPE}}>({{NAME}}_Size,q[0]);
    q[0].memcpy(recon_d_{{NAME}}, recon_h_{{NAME}}, {{NAME}}_Size*sizeof({{TYPE}})).wait();
    {{TYPE}} *r_{{NAME}}=malloc_device<{{TYPE}}>({{NAME}}_Size,q[0]);
    {{NAME}}_tool.Reconstruct(r_{{NAME}}, recon_d_{{NAME}},q[0]);
    sycl::free(recon_d_{{NAME}}, q[0]);
    std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));)~~~";
    
    const char *DATA_RECON_OUT_Template = R"~~~(
    // 数据重组
    DataReconstructor<{{TYPE}}> {{NAME}}_tool;
    {{DATA_OPS_INIT}}
    {{NAME}}_tool.init(info_{{NAME}},{{NAME}}_ops);
    {{TYPE}} *r_{{NAME}}=malloc_device<{{TYPE}}>({{NAME}}_Size,q[0]);
    q[0].memset(r_{{NAME}}, 0, {{NAME}}_Size * sizeof({{TYPE}})).wait();
    std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));)~~~";
    
    std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit, bool isOut){
        if(isOut){
            return templateString(DATA_RECON_OUT_Template,
            {
                {"{{TYPE}}",       type},
                {"{{NAME}}",       name},
                {"{{SIZE}}",       size},
                {"{{DATA_OPS_INIT}}", dataOpsInit}
            });
        }else{
            return templateString(DATA_RECON_Template,
            {
                {"{{TYPE}}",       type},
                {"{{NAME}}",       name},
                {"{{SIZE}}",       size},
                {"{{DATA_OPS_INIT}}", dataOpsInit}
            });
        }
    }


    const char *DEVICE_MEM_ALLOC_Template = R"~~~(
        // 设备内存分配
        {{TYPE}} *d_{{NAME}}=malloc_device<{{TYPE}}>({{NAME}}_Device_Size[numDevice] * {{NAME}}_data_SplitSize,q[numDevice]);)~~~";

    const char *DEVICE_MEM_ALLOC_OutData_Template = R"~~~(
        // 归约设备内存分配
        {{TYPE}} *d_{{NAME}} = malloc_device<{{TYPE}}>({{SIZE}}*{{NAME}}_data_SplitSize,q[numDevice]);)~~~";
    std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size){
        return templateString(DEVICE_MEM_ALLOC_Template,
	    {
		    {"{{TYPE}}", type},
		    {"{{NAME}}", name},
		    {"{{SIZE}}", size}
	    });
    }
    std::string CodeGen_DeviceMemAllocOutData(std::string type,std::string name,std::string size){
        return templateString(DEVICE_MEM_ALLOC_OutData_Template,
        {
            {"{{TYPE}}", type},
            {"{{NAME}}", name},
            {"{{SIZE}}", size}
        });
    }

    const char *DEVICE_MEM_ALLOC_REDUCTION_Template = R"~~~(
        // 归约设备内存分配
        {{TYPE}} *reduction_{{NAME}} = malloc_device<{{TYPE}}>({{SIZE}},q[numDevice]);)~~~";
    
    std::string CodeGen_DeviceMemAllocReduction(std::string  type,std::string name,std::string size){
        return templateString(DEVICE_MEM_ALLOC_REDUCTION_Template,
        {
            {"{{TYPE}}", type},
            {"{{NAME}}", name},
            {"{{SIZE}}", size}
        });
    }
    const char *H2D_MEM_MOV_Template = R"~~~(
        // 数据移动
        q[numDevice].memcpy(d_{{NAME}},r_{{NAME}} + {{NAME}}_HaveMalloc , {{NAME}}_Device_Size[numDevice] * {{NAME}}_data_SplitSize * sizeof({{TYPE}})).wait();
        {{NAME}}_HaveMalloc += {{NAME}}_Device_Size[numDevice] * {{NAME}}_data_SplitSize;)~~~";


    const char *H2D_MEM_MOV_Template_Full = R"~~~(
        // 数据移动
        q[numDevice].memcpy(d_{{NAME}},r_{{NAME}} ,{{NAME}}_Device_Size[numDevice] * {{NAME}}_data_SplitSize * sizeof({{TYPE}})).wait();)~~~";
    
    std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size,bool isFull){
        if(isFull){
            return templateString(H2D_MEM_MOV_Template_Full,
            {
                {"{{TYPE}}", type},
                {"{{NAME}}", name},
                {"{{SIZE}}", size}
            });
        }else{
        return templateString(H2D_MEM_MOV_Template,
        {
            {"{{TYPE}}", type},
            {"{{NAME}}", name},
            {"{{SIZE}}", size}
        });
        }
    }
    const char *KERNEL_EXECUTE_Template = R"~~~(
        //工作项划分
        sycl::range<3> local(1, 1, {{CALC_SIZE}});
        sycl::range<3> global(1, 1, {{SPLIT_SIZE}});
        //队列提交命令组
         events.push_back(q[numDevice].submit([&](handler &h) {
            // 访问器初始化
            {{ACCESSOR_INIT}}
            h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
                const auto item_id = item.get_local_id(2);
                // 索引初始化
                {{INDEX_INIT}}
                // 嵌入计算
                {{CALC_EMBED}}
            });
        }));
        Result_HaveMalloc += {{CALC_SIZE}};
        
    )~~~";
    
    std::string CodeGen_KernelExecute(std::string CALC_SIZE,std::string SplitSize,std::string AccessorInit,std::string IndexInit,std::string CalcEmbed){
        return templateString(KERNEL_EXECUTE_Template,
        {
            {"{{CALC_SIZE}}",     CALC_SIZE},
            {"{{SPLIT_SIZE}}",    SplitSize},
            {"{{ACCESSOR_INIT}}", AccessorInit},
            {"{{INDEX_INIT}}",    IndexInit},
            {"{{CALC_EMBED}}",    CalcEmbed}
        });
    }

    const char *Store_Mem_Template = R"~~~(
        //保存内存地址
        d_{{NAME}}s[numDevice] = d_{{NAME}};)~~~";

    const char *Store_Mem_Template_Reduction = R"~~~(
        //保存内存地址
        reduction_{{NAME}}s[numDevice] = reduction_{{NAME}};)~~~";
    
    std::string CodeGen_StoreDeviceMem(std::string name){
        return templateString(Store_Mem_Template,
        {
            {"{{NAME}}", name}
        });
    }
    std::string CodeGen_StoreDeviceMemReduction(std::string name){
        return templateString(Store_Mem_Template_Reduction,
        {
            {"{{NAME}}", name}
        });
    }
    const char *Load_Mem_Template = R"~~~(
        //获取内存地址
        {{TYPE}} *d_{{NAME}} = d_{{NAME}}s[numDevice];)~~~";
    const char *Load_Mem_Template_Reduction = R"~~~(
        //获取内存地址
        {{TYPE}} *reduction_{{NAME}} = reduction_{{NAME}}s[numDevice];)~~~";
    
    std::string CodeGen_LoadDeviceMem(std::string type,std::string name){
        return templateString(Load_Mem_Template,
        {
            {"{{TYPE}}", type},
            {"{{NAME}}", name}
        });
    }
    std::string CodeGen_LoadDeviceMemReduction(std::string type,std::string name){
        return templateString(Load_Mem_Template_Reduction,
        {
            {"{{TYPE}}", type},
            {"{{NAME}}", name}
        });
    }

const char *CALC_EMBED_Template = R"~~~(
        {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names, std::vector<bool> isFull){
    std::string DacCalcArgs = "(";
    int len = args.size;
    for(int i=0;i<len;i++){
        std::string IndexComb="(";
        for(int j=0;j<args[i].ops.size;j++){
            if(isFull[i]){
                std::string opsname = args[i].ops[j].name;
                if(j != args[i].ops.size-1){
                    // if(args[i].ops[j+1].name == "void") continue;
                    IndexComb+= "(("+opsname + "_+ Result_HaveMalloc/"+args[i].ops[j+1].name+".split_size)%"+opsname+".split_size)" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";
                }else{
                    IndexComb+= "(("+opsname + "_+ Result_HaveMalloc)%"+opsname+".split_size)" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";
                }
                if(j!=args[i].ops.size-1) IndexComb+="+";
            }else{
                std::string opsname = args[i].ops[j].name;
                IndexComb+= opsname + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";
                if(j!=args[i].ops.size-1) IndexComb+="+";
            }
        }
        IndexComb+=")";
        if(IndexComb == "()"){
            DacCalcArgs+=args[i].name;
        }
        else{
            DacCalcArgs+=args[i].name + "+" + IndexComb;
        }		
        DacCalcArgs+=",";
    }
    for (int z = 0; z < accessor_names.size(); z++) {
        DacCalcArgs+="info_partition_"+accessor_names[z]+"_accessor";
        if (z == accessor_names.size() - 1) {
            DacCalcArgs+=");";
        } else {
            DacCalcArgs+=",";
        }
    }
    return templateString(CALC_EMBED_Template,
    {
        {"{{DAC_CALC_NAME}}",    Name},
        {"{{DAC_CALC_ARGS}}",    DacCalcArgs}
    });
}


    const char *REDUCTION_Template_Span = R"~~~(
    // 归约
    if({{SPLIT_SIZE}} > 1)
    {
        events[numDevice].wait();
        for(int i=0;i<{{SPAN_SIZE}};i++) {
            q[numDevice].submit([&](handler &h) {
    	        h.parallel_for(
                range<1>({{SPLIT_SIZE}}),
                reduction(reduction_{{NAME}}+i, 
                {{REDUCTION_RULE}},
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_{{NAME}}[(i/{{SPLIT_LENGTH}})*{{SPLIT_LENGTH}}*{{SPLIT_SIZE}}+i%{{SPLIT_LENGTH}}+idx*{{SPLIT_LENGTH}}]);
     	        });
         }).wait();
        }
        q[numDevice].memcpy(d_{{NAME}},reduction_{{NAME}}, {{SPAN_SIZE}}*sizeof({{TYPE}})).wait();
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

const char *D2H_MEM_MOV_1_Template = R"~~~(
    // 归并结果返回
    q[numDevice].memcpy(d_my{{NAME}} + {{NAME}}_HaveMalloc, d_{{NAME}}, {{SIZE}}*{{NAME}}_data_SplitSize*sizeof({{TYPE}})).wait();
    {{NAME}}_HaveMalloc += {{SIZE}}*{{NAME}}_data_SplitSize;)~~~";

const char *D2H_MEM_MOV_2_Template = R"~~~(
    // 归并结果返回
    q[numDevice].memcpy(d_my{{NAME}} + {{NAME}}_HaveMalloc, d_{{NAME}}, {{SIZE}}*{{NAME}}_data_SplitSize*sizeof({{TYPE}})).wait();
    {{NAME}}_HaveMalloc += {{SIZE}}*{{NAME}}_data_SplitSize;)~~~";

std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction){
    if(isReduction){
		return templateString(D2H_MEM_MOV_2_Template,
		{
			{"{{TYPE}}",            Type},
			{"{{NAME}}",            Name},
			{"{{SIZE}}",            Size}
		});
	}
	else{
		return templateString(D2H_MEM_MOV_1_Template,
		{
			{"{{TYPE}}",            Type},
			{"{{NAME}}",            Name},
			{"{{SIZE}}",            Size}
		});
	}
}

const char *MEM_FREE_Template = R"~~~(
    sycl::free({{NAME}}, q[numDevice]);)~~~";

std::string CodeGen_MemFree(std::string Name){
    return templateString(MEM_FREE_Template,
	{
		{"{{NAME}}",            Name}
	});
}

const char* D2H_MEM_MOV_F_Template = R"~~~(
    // 归并结果返回
	{{TYPE}} *d_myTensor2=malloc_device<{{TYPE}}>({{SIZE}},q[0]);
    {{NAME}}_tool.UpdateData(d_my{{NAME}},d_myTensor2,q[0]);
	{{TYPE}}* res = ({{TYPE}}*)malloc({{SIZE}}*sizeof({{TYPE}}));
	q[0].memcpy(res,d_myTensor2, {{SIZE}}*sizeof({{TYPE}})).wait();
	{{NAME}}.array2Tensor(res);)~~~";

std::string CodeGen_D2HMemMovF(std::string Name,std::string Type,std::string Size){
    return templateString(D2H_MEM_MOV_F_Template,
    {
        {"{{TYPE}}",            Type},
		{"{{NAME}}",            Name},
		{"{{SIZE}}",            Size}
    });
}
const char *just_a_middle = R"~~~(
    // 设备内存分配
    {{TYPE}} *d_my{{NAME}}=malloc_device<{{TYPE}}>({{SIZE}},q[0]);)~~~";

std::string CodeGen_JustAMiddle(std::string type,std::string name,std::string size){
    return templateString(just_a_middle,
    {
        {"{{TYPE}}", type},
        {"{{NAME}}", name},
        {"{{SIZE}}", size}
    });
}

const char* Kernel_Template = R"~~~(
    std::vector<sycl::event> events;
    {{DEVICE_MEM}}
    {{CantFindBetterWay}}
    for(int numDevice = 0; numDevice < numDevices; numDevice++){
        // 设备内存分配
        {{DEVICE_MEM_ALLOC}}
        // 数据移动
        {{H2D_MEM_MOV}}
        //内核执行
        {{KERNEL_EXECUTE}}
        //保存内存地址
        {{STORE_DEVICE_MEM}}
    }
    for(int numDevice = 0; numDevice < numDevices; numDevice++){
        //获取内存地址
        {{LOAD_DEVICE_MEM}}
        // 归约
        {{REDUCTION}}
        // 归并结果返回
        {{D2H_MEM_MOV}}
        // 内存释放
        {{MEM_FREE}}
    })~~~";
//整合模板
std::string CodeGen_Kernel(std::string Device_Mem,std::string CantFindBetterWay,std::string DeviceMemAlloc,std::string H2DMemMove,std::string KernelExecute,std::string Store_Device_Mem,
                           std::string Load_Device_Mem,std::string Reduction,std::string D2HMemMove,std::string MemFree){
    return templateString(Kernel_Template,
    {
        {"{{DEVICE_MEM}}",        Device_Mem},
        {"{{CantFindBetterWay}}", CantFindBetterWay},
        {"{{DEVICE_MEM_ALLOC}}",  DeviceMemAlloc},
        {"{{H2D_MEM_MOV}}",       H2DMemMove},
        {"{{KERNEL_EXECUTE}}",    KernelExecute},
        {"{{STORE_DEVICE_MEM}}",  Store_Device_Mem},
        {"{{LOAD_DEVICE_MEM}}",   Load_Device_Mem},
        {"{{REDUCTION}}",         Reduction},
        {"{{D2H_MEM_MOV}}",       D2HMemMove},
        {"{{MEM_FREE}}",          MemFree}
    });
}

const char *DAC2SYCL_Template_2 = R"~~~(
    // 生成函数调用
    void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
        // 设备选择
        auto devices = device::get_devices(info::device_type::gpu);
        vector<queue> q;
        for (const auto& dev : devices) {
            q.emplace_back(dev);
        }
        int numDevices = q.size();
        // printf("Running on %d GPU\n", numDevices);
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
    
    std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams,std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp){
        return templateString(DAC2SYCL_Template_2,
        {	
            {"{{DAC_SHELL_NAME}}",    dacShellName},
            {"{{DAC_SHELL_PARAMS}}",  dacShellParams},
            {"{{OP_INIT}}",           opInit},
            {"{{ParameterGenerate}}", parameter_generate},
            {"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
            {"{{DATA_ASSOC_COMP}}",   dataAssocComp},
        });
    }
} // namespace Multiple_TEMPLATE