#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"sub_template.h"

/*
	文本替换。
	将 text 中的 find 换成 replace。
*/
void replaceTextInString(std::string& text, 
const std::string &find, 
const std::string &replace){
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos){
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}
/*
	文本替换。
	对 templ 做 replacements 中的文本替换。
*/
std::string templateString(std::string templ, 
std::vector<std::pair<std::string, std::string>> replacements){
	for(auto &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

const char *DATA_ASSOC_COMP_Template = R"~~~(
    {{DATA_RECON}}
    {{H2D_MEM_MOV}}
	{{KERNEL_EXECUTE}}
	{{REDUCTION}}
	{{D2H_MEM_MOV}}
)~~~";

std::string CodeGen_DataAssocComp(std::string dataRecon, std::string H2DMemMove, std::string kernelExecute, std::string reduction, std::string D2HMemMove){
    return templateString(DATA_ASSOC_COMP_Template,
	{
        {"{{DATA_RECON}}",        dataRecon},
        {"{{H2D_MEM_MOV}}",       H2DMemMove},
        {"{{KERNEL_EXECUTE}}",    kernelExecute},
		{"{{REDUCTION}}",         reduction},
        {"{{D2H_MEM_MOV}}",       D2HMemMove}
	});
}

const char *DAC2SYCL_Template_1 = R"~~~(
// 生成函数调用
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 设备内存分配
    {{DEVICE_MEM_ALLOC}}
    // 算子初始化
    {{OP_INIT}}
    // 数据关联计算
    {{DATA_ASSOC_COMP}}
    // 内存释放
    {{MEM_FREE}}
})~~~";

std::string CodeGen_DAC2SYCL(std::string dacShellName, std::string dacShellParams, std::string deviceMemAlloc, std::string opInit, std::string dataAssocComp, std::string memFree){
    return templateString(DAC2SYCL_Template_1,
	{
		{"{{DAC_SHELL_NAME}}",    dacShellName},
		{"{{DAC_SHELL_PARAMS}}",  dacShellParams},
		{"{{OP_INIT}}",           opInit},
		{"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
		{"{{DATA_ASSOC_COMP}}",   dataAssocComp},
        {"{{MEM_FREE}}",          memFree}
	});
}

// aborted
const char *DAC2SYCL_Template = R"~~~(
// 生成函数调用
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    {{OP_INIT}}
    // 数据重组
    {{DATA_RECON}}
    // 设备内存分配
    {{DEVICE_MEM_ALLOC}}
    // 数据移动
    {{H2D_MEM_MOV}}   
    // 内核执行
    {{KERNEL_EXECUTE}}    
    // 归约
    {{REDUCTION}}
    // 返回计算结果
    {{D2H_MEM_MOV}}
    // 内存释放
    {{MEM_FREE}}
})~~~";

// aborted
std::string CodeGen_DAC2SYCL(std::string dacShellName,std::string dacShellParams,std::string opInit,std::string dataRecon,
	std::string deviceMemAlloc,std::string H2DMemMove,std::string kernelExecute,std::string reduction,std::string D2HMemMove,std::string memFree){
    return templateString(DAC2SYCL_Template,
	{
		{"{{DAC_SHELL_NAME}}",    dacShellName},
		{"{{DAC_SHELL_PARAMS}}",  dacShellParams},
		{"{{OP_INIT}}",           opInit},
        {"{{DATA_RECON}}",        dataRecon},
		{"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
        {"{{H2D_MEM_MOV}}",       H2DMemMove},
        {"{{KERNEL_EXECUTE}}",    kernelExecute},
		{"{{REDUCTION}}",         reduction},
        {"{{D2H_MEM_MOV}}",       D2HMemMove},
        {"{{MEM_FREE}}",          memFree}
	});
}

const char *DATA_INFO_INIT_Template = R"~~~(
    // 数据信息初始化
    DataInfo info_{{NAME}};
    info_{{NAME}}.dim = {{NAME}}.getDim();
    for(int i = 0; i < info_{{NAME}}.dim; i++) info_{{NAME}}.dimLength.push_back({{NAME}}.getShape(i));)~~~";
std::string CodeGen_DataInfoInit(std::string name){
    return templateString(DATA_INFO_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}
const char *OP_REGULAR_SLICE_INIT_Template = R"~~~(
    // 规则分区算子初始化
    RegularSlice {{OP_NAME}} = RegularSlice("{{OP_NAME}}", {{SIZE}}, {{STRIDE}});
    {{OP_NAME}}.SetSplitSize({{SPLIT_SIZE}});)~~~";

std::string CodeGen_RegularSliceInit(std::string opName,std::string size,std::string stride,std::string splitSize){
    return templateString(OP_REGULAR_SLICE_INIT_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{SIZE}}",       size},
		{"{{STRIDE}}",     stride},
		{"{{SPLIT_SIZE}}", splitSize}
	});
}

const char *OP_INDEX_INIT_Template = R"~~~(
    // 降维算子初始化
    Index {{OP_NAME}} = Index("{{OP_NAME}}");
    {{OP_NAME}}.SetSplitSize({{SPLIT_SIZE}});)~~~";

std::string CodeGen_IndexInit(std::string opName,std::string splitSize){
    return templateString(OP_INDEX_INIT_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{SPLIT_SIZE}}", splitSize}
	});
}

const char *OP_PUSH_BACK2OPS_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.setSplitLength({{SPLIT_LENGTH}});
    {{NAME}}_ops.push_back({{OP_NAME}});)~~~";

std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId, std::string splitLength){
    return templateString(OP_PUSH_BACK2OPS_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{NAME}}",       name},
		{"{{DIM_ID}}",     dimId},
		{"{{SPLIT_LENGTH}}", splitLength}
	});
}

const char *OP_PUSH_BACK2TOOL_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.setSplitLength({{SPLIT_LENGTH}});
    {{NAME}}_tool.push_back({{OP_NAME}});)~~~";

std::string CodeGen_OpPushBack2Tool(std::string name, std::string opName, std::string dimId, std::string splitLength){
    return templateString(OP_PUSH_BACK2TOOL_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{NAME}}",       name},
		{"{{DIM_ID}}",     dimId},
		{"{{SPLIT_LENGTH}}", splitLength}
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

const char *DATA_RECON_Template = R"~~~(
    // 数据重组
    DataReconstructor<{{TYPE}}> {{NAME}}_tool;
    {{TYPE}}* r_{{NAME}}=({{TYPE}}*)malloc(sizeof({{TYPE}})*{{SIZE}});
    {{DATA_OPS_INIT}}
    {{NAME}}_tool.init(info_{{NAME}},{{NAME}}_ops);
    {{NAME}}_tool.Reconstruct(r_{{NAME}},{{NAME}});
	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));)~~~";

std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit){
    return templateString(DATA_RECON_Template,
	{
		{"{{TYPE}}",       type},
		{"{{NAME}}",       name},
		{"{{SIZE}}",       size},
		{"{{DATA_OPS_INIT}}", dataOpsInit}
	});
}

const char *DATA_RECON_OP_PUSH_Template = R"~~~(
    // 数据重组
    {{OP_PUSH_BACK2TOOL}}
    {{NAME}}_tool.Reconstruct(r_{{NAME}},{{NAME}});)~~~";

std::string CodeGen_DataReconstructOpPush(std::string name,std::string opPushBack2Tool){
    return templateString(DATA_RECON_OP_PUSH_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_PUSH_BACK2TOOL}}", opPushBack2Tool}
	});
}

const char *OP_POP_FROM_TOOL_Template = R"~~~(
    {{NAME}}_tool.pop_back();)~~~";
std::string CodeGen_OpPopFromTool(std::string name){
    return templateString(OP_POP_FROM_TOOL_Template,
	{
		{"{{NAME}}", name}
	});
}

const char *DATA_RECON_OP_POP_Template = R"~~~(
    // 数据重组
    {{OP_POP_FROM_TOOL}}
    {{NAME}}_tool.Reconstruct(r_{{NAME}},{{NAME}});)~~~";

std::string CodeGen_DataReconstructOpPop(std::string name,std::string opPopFromTool){
    return templateString(DATA_RECON_OP_POP_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_POP_FROM_TOOL}}", opPopFromTool}
	});
}

const char *DEVICE_MEM_ALLOC_Template = R"~~~(
    // 设备内存分配
    {{TYPE}} *d_{{NAME}}=malloc_device<{{TYPE}}>({{SIZE}},q);)~~~";

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size){
    return templateString(DEVICE_MEM_ALLOC_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *DEVICE_MEM_ALLOC_REDUCTION_Template = R"~~~(
    // 归约设备内存分配
    {{TYPE}} *reduction_{{NAME}} = malloc_device<{{TYPE}}>({{SIZE}},q);)~~~";

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
    q.memcpy(d_{{NAME}},r_{{NAME}},{{SIZE}}*sizeof({{TYPE}})).wait();)~~~";

std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size){
    return templateString(H2D_MEM_MOV_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *KERNEL_EXECUTE_OLD_Template = R"~~~(
    //工作项划分
    sycl::range<3> local(1, 1, {{SPLIT_SIZE}});
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			{{INDEX_INIT}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
    
)~~~";

std::string CodeGen_KernelExecute(std::string SplitSize,std::string IndexInit,std::string CalcEmbed){
    return templateString(KERNEL_EXECUTE_OLD_Template,
	{
		{"{{SPLIT_SIZE}}",    SplitSize},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed}
	});
}

const char *KERNEL_EXECUTE_Template = R"~~~(
    //工作项划分
    sycl::range<3> local(1, 1, {{SPLIT_SIZE}});
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			{{INDEX_INIT}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
    
)~~~";

std::string CodeGen_KernelExecute(std::string SplitSize,std::string AccessorInit,std::string IndexInit,std::string CalcEmbed){
    return templateString(KERNEL_EXECUTE_Template,
	{
		{"{{SPLIT_SIZE}}",    SplitSize},
		{"{{ACCESSOR_INIT}}", AccessorInit},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed}
	});
}

const char *ACCESSOR_INIT_Template = R"~~~(
        auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read_write>(h);)~~~";
std::string CodeGen_AccessorInit(std::string name) {
	return templateString(ACCESSOR_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}
const char *INDEX_INIT_Template = R"~~~(
            const auto {{NAME}}={{EXPRESSION}};)~~~";

//aborted
std::string CodeGen_IndexInit(Dac_Ops ops)
{
	int len = ops.size;
	std::vector<std::string> index_expression;//增加
	for(int i=0;i<len;i++){
		std::string sub_expression = "item_id";
		for(int j=i+1;j<len;j++){
			sub_expression = sub_expression + "/" + std::to_string(ops[j].split_size);
		}
		sub_expression = sub_expression + "%" + std::to_string(ops[i].split_size);
		index_expression.push_back(sub_expression);//增加
		//ops[i].setExp(sub_expression);
	}

	std::string expression = "";
	for(int i=0;i<len;i++){
		std::string ops_i_name = ops[i].name;
		std::string index_i_expression = index_expression[i];
		expression = expression + templateString(INDEX_INIT_Template,
		{
			//{"{{NAME}}", ops[i].name},
			{"{{NAME}}", ops_i_name},
			//{"{{EXPRESSION}}", ops[i].getExp()}
			{"{{EXPRESSION}}", index_i_expression}
		});
	}

	return expression;
}

std::string CodeGen_IndexInit(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets)//sets表示每个算子属于的集合的名字 offsets表示每个算子相对于集合的偏移量
{ 
    std::set<std::string> sets_map;//用于辅助找到不同的集合的个数
    std::vector<std::string> sets_order;//记录了不同的集合出现的顺序，储存集合的名字： idx idy idz
    std::vector<int> sets_split;//记录了不同集合对应的划分数，与集合名相对应： idx的划分数 idy的划分数 idz的划分数 
    for (int i = 0; i < sets.size(); ++i) 
    {
        if (sets_map.find(sets[i]) == sets_map.end())//如果容器里没有
        {
            sets_map.insert(sets[i]);//将集合插入容器
            sets_order.push_back(sets[i]);//将集合放入到集合的数组中
            sets_split.push_back(ops[i].split_size);//将集合对应的划分数放入数组中
        }
    }
    
    int sets_size = sets_map.size();//得到各类集合总个数
    std::unordered_map<std::string,std::string> sets_sub_expression;//<集合的名称，集合对应的索引表达式>

    for(int i = 0;i < sets_size; i++)//有几个集合就循环几次
    {
		std::string sub_expression = "item_id";
		for(int j = i + 1;j < sets_size;j ++){
			sub_expression = sub_expression + "/" + std::to_string(sets_split[j]);
		}
		//sub_expression = sub_expression + "%" + std::to_string(sets_split[i]);//取模操作应该在偏移之后
        sets_sub_expression[sets_order[i]] = sub_expression;//将子表达式和集合的名字进行关联
	}

    //下面根据偏移量来计算各个算子对应的索引
    int len = ops.size;
	std::vector<std::string> index_expression_vector;//新增加
    for(int i = 0;i < len;i ++)
    {
        std::string index_expression = "(";
        index_expression = index_expression + sets_sub_expression[sets[i]];//得到集合的索引
        index_expression = index_expression + "+" + "(" + offsets[i] + ")" + "+" + std::to_string(ops[i].split_size) + ")";//加上偏移量和划分数 防止出现负数
		index_expression = index_expression + "%" + std::to_string(ops[i].split_size);//偏移之后再取模
        //ops[i].setExp(index_expression);
		index_expression_vector.push_back(index_expression);//新增加
    }

	std::string expression = "";
	for(int i=0;i<len;i++){
		std::string opsname = ops[i].name;
		std::string index_i_expression = index_expression_vector[i];
		expression = expression + templateString(INDEX_INIT_Template,
		{
			//{"{{NAME}}", ops[i].name},
			{"{{NAME}}", opsname},
			//{"{{EXPRESSION}}", ops[i].getExp()}
			{"{{EXPRESSION}}", index_i_expression}
		});
	}
	return expression;
}

const char *CALC_EMBED_Template = R"~~~(
            {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

std::string CodeGen_CalcEmbed(std::string Name,Args args){
	std::string DacCalcArgs = "(";
	int len = args.size;
	for(int i=0;i<len;i++){
		std::string IndexComb="(";
		for(int j=0;j<args[i].ops.size;j++){
			std::string opsname = args[i].ops[j].name;
			//IndexComb+= args[i].ops[j].name + "*" + std::to_string(args[i].ops[j].split_length);
			IndexComb+= opsname + "*" + std::to_string(args[i].ops[j].split_length);
			if(j!=args[i].ops.size-1) IndexComb+="+";
		}
		IndexComb+=")";
		if(IndexComb == "()")
		{
			DacCalcArgs+=args[i].name;
		}
		else{
			DacCalcArgs+=args[i].name + "+" + IndexComb;
		}		
		if(i==len-1){
			DacCalcArgs+=",";
			for(int j=0;j<len;j++) {
				DacCalcArgs+="info_partition_"+args[j].name.substr(2)+"_accessor";
				if(j!=len-1) DacCalcArgs+=",";
			}
			DacCalcArgs+=");";
		}
		else{
			DacCalcArgs+=",";
		}
	}
	return templateString(CALC_EMBED_Template,
	{
		{"{{DAC_CALC_NAME}}",    Name},
		{"{{DAC_CALC_ARGS}}",    DacCalcArgs}
	});
}

std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names){
		std::string DacCalcArgs = "(";
	int len = args.size;
	for(int i=0;i<len;i++){
		std::string IndexComb="(";
		for(int j=0;j<args[i].ops.size;j++){
			std::string opsname = args[i].ops[j].name;
			//IndexComb+= args[i].ops[j].name + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";
			IndexComb+= opsname + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";
			if(j!=args[i].ops.size-1) IndexComb+="+";
		}
		IndexComb+=")";
		if(IndexComb == "()")
		{
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

// aborted
const char *REDUCTION_Template = R"~~~(
    // 归约
    //使用内核函数进行归约
    q.submit([&](handler &h) {
    	h.parallel_for(range<1>({{SPLIT_SIZE}}),reduction(reduction_{{NAME}}, {{REDUCTION_RULE}},property::reduction::initialize_to_identity()),[=](id<1> i,auto &reducer) {
            	reducer.combine(d_{{NAME}}[i]);
     	});
 }).wait();
)~~~";

// aborted
std::string CodeGen_Reduction(std::string SplitSize,std::string Name,std::string Type,std::string ReductionRule) {
    return templateString(REDUCTION_Template,
	{
		{"{{SPLIT_SIZE}}",       SplitSize},
		{"{{TYPE}}",             Type},
		{"{{NAME}}",             Name},
		{"{{REDUCTION_RULE}}",   ReductionRule}
	});
}

const char *REDUCTION_Template_Span = R"~~~(
    // 归约
    if({{SPLIT_SIZE}} > 1)
    {
        for(int i=0;i<{{SPAN_SIZE}};i++) {
            q.submit([&](handler &h) {
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
        q.memcpy(d_{{NAME}},reduction_{{NAME}}, {{SPAN_SIZE}}*sizeof({{TYPE}})).wait();
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
    q.memcpy(r_{{NAME}}, d_{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();
    {{NAME}}_tool.UpdateData(r_{{NAME}},{{NAME}});)~~~";

const char *D2H_MEM_MOV_2_Template = R"~~~(
    // 归约结果返回
    q.memcpy(r_{{NAME}},d__{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();
    {{NAME}}_tool.UpdateData(r_{{NAME}},{{NAME}});)~~~";

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
    sycl::free(d_{{NAME}}, q);)~~~";

std::string CodeGen_MemFree(std::string Name){
    return templateString(MEM_FREE_Template,
	{
		{"{{NAME}}",            Name}
	});
}



/*下面是新增加以及修改的模板 之前的模板未做更改*/

//新的总的生成模板，调整了顺序 先算子初始化再去计算参数

const char *DAC2SYCL_Template_2 = R"~~~(
// 生成函数调用
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    {{OP_INIT}}
    //参数生成
	{{ParameterGenerate}}
    // 设备内存分配
    {{DEVICE_MEM_ALLOC}}
    // 数据关联计算
    {{DATA_ASSOC_COMP}}
    // 内存释放
    {{MEM_FREE}}
})~~~";

std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams,std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp, std::string memFree){
    return templateString(DAC2SYCL_Template_2,
	{	
		{"{{DAC_SHELL_NAME}}",    dacShellName},
		{"{{DAC_SHELL_PARAMS}}",  dacShellParams},
		{"{{OP_INIT}}",           opInit},
		{"{{ParameterGenerate}}", parameter_generate},
		{"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
		{"{{DATA_ASSOC_COMP}}",   dataAssocComp},
        {"{{MEM_FREE}}",          memFree}
	});
}

// const char *DAC2SYCL_Template_2 = R"~~~(
// // 生成函数调用
// void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
//     // 设备选择
//     auto selector = gpu_selector_v;
//     queue q(selector);
//     //声明参数生成工具
//     {{ParameterTool}}
//     // 算子初始化
//     {{OP_INIT}}
//     //参数生成
// 	{{ParameterGenerate}}
//     // 设备内存分配
//     {{DEVICE_MEM_ALLOC}}
//     // 数据关联计算
//     {{DATA_ASSOC_COMP}}
//     // 内存释放
//     {{MEM_FREE}}
// })~~~";

// std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams,std::string parameter_tool, std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp, std::string memFree){
//     return templateString(DAC2SYCL_Template_2,
// 	{	
// 		{"{{DAC_SHELL_NAME}}",    dacShellName},
// 		{"{{DAC_SHELL_PARAMS}}",  dacShellParams},
// 		{"{{ParameterTool}}",     parameter_tool},
// 		{"{{OP_INIT}}",           opInit},
// 		{"{{ParameterGenerate}}", parameter_generate},
// 		{"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
// 		{"{{DATA_ASSOC_COMP}}",   dataAssocComp},
//         {"{{MEM_FREE}}",          memFree}
// 	});
// }

// 下面已经不需要了
// //参数生成工具的声明 Tensor是几维的这个就应该是几维的
// const char *INIT_PARAMETER_TOOL_Template = R"~~~(
//     ParameterGeneration<int,{{DIM_NUM}}> para_gene_tool{{NUM}};
// )~~~";

// std::string CodeGen_InitParameterTool(std::string DIM_NUM){
//     return templateString(INIT_PARAMETER_TOOL_Template,
// 	{
// 		{"{{DIM_NUM}}",    DIM_NUM}
// 	});
// }

//新的 规则分区算子初始化
// const char *OP_REGULAR_SLICE_INIT_Template2 = R"~~~(
//     // 规则分区算子初始化
//     RegularSlice {{OP_NAME}} = RegularSlice("{{OP_NAME}}", {{SIZE}}, {{STRIDE}});
//     {{OP_NAME}}.setDimId({{DIM_ID}});
//     {{OP_NAME}}.SetSplitSize(para_gene_tool.init_operetor_splitnumber({{OP_NAME}},{{TENSOR_NAME}}));
// )~~~";

// std::string CodeGen_RegularSliceInit2(std::string opName,std::string size,std::string stride,std::string dim_id,std::string tensor_name){
//     return templateString(OP_REGULAR_SLICE_INIT_Template2,
// 	{
// 		{"{{OP_NAME}}",    opName},
// 		{"{{SIZE}}",       size},
// 		{"{{STRIDE}}",     stride},
// 		{"{{DIM_ID}}",     dim_id}, //需要通过dimId来计算算子的划分数了
// 		{"{{TENSOR_NAME}}",     tensor_name}
// 	});
// }

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

//新的 降维算子初始化
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

//生成算子划分数的模板 在初始化算子时直接进行划分数的赋值了
// const char *OP_SPILIT_NUMBER_Generate_Template = R"~~~(
// 	//生成算子的划分数
//     int {{OP_NAME}}_spilit_number = para_gene_tool.init_operetor_splitnumber({{OP_NAME}},{{TENSOR_NAME}});
// 	{{OP_NAME}}.SetSplitSize({{OP_NAME}}_spilit_number);
// )~~~";

// std::string CodeGen_OpSpilitNumberGenerate(std::string op_name, std::string tensor_name){
//     return templateString(OP_SPILIT_NUMBER_Generate_Template,
// 	{
//         {"{{OP_NAME}}",        op_name}, //算子的名字 注意这里有一个逗号
// 		{"{{TENSOR_NAME}}",    tensor_name} //存数据的tensor的名字 
// 	});
// }

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

/*下面函数已废弃*/
//构造tensor_in std::vector<dacpp::Tensor<ImplType>> tensor_in
// const char *Tensor_Vector_Declaration_Template = R"~~~(
// 	std::vector<dacpp::Tensor<{{TYPE}}>> {{NAME}};
// )~~~";

// std::string CodeGen_TensorVectorDeclarationGenerate(std::string TYPE, std::string tensor_name){
//     return templateString(Tensor_Vector_Declaration_Template,
// 	{
//         {"{{TYPE}}",        TYPE}, //Tensor里面的数据类型
// 		{"{{NAME}}",    tensor_name} //Tensor vector组的名字 注意前后命名一致性 
// 	});
// }

// //tensor_in.pushback();往vector组里面添加数据
// const char *Tensor_Vector_Add_Template = R"~~~(
// 	{{NAME}}.push_back({{NAME2}});
// )~~~";

// std::string CodeGen_TensorVectorAddGenerate(std::string NAME, std::string NAME2){
//     return templateString(Tensor_Vector_Add_Template,
// 	{
//         {"{{NAME}}",        NAME}, //Tensor vector组的名字
// 		{"{{NAME2}}",       NAME2} //加到里面的Tensor的名字 
// 	});
// }
/*上面函数已废弃*/

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

/*下面函数已废弃*/
//生成设备内存分配的大小 对应数据重组需要分配的大小
// const char *DEVICE_MEM_SIZE_Generate_Template3 = R"~~~(
// 	//生成设备内存分配大小
//     int {{NAME}}_size = para_gene_tool.init_device_memory_size({{TENSOR_IN_NAME}},{{TENSOR_OUT_NAME}},{{IN_DAC_OPS_NAME}},{{OUT_DAC_OPS_NAME}});
// )~~~";

// std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string TENSOR_IN_NAME,std::string TENSOR_OUT_NAME,std::string IN_DAC_OPS_NAME,std::string OUT_DAC_OPS_NAME){
//     return templateString(DEVICE_MEM_SIZE_Generate_Template3,
// 	{
//         {"{{NAME}}",        NAME}, //设备内存的名字
// 		{"{{TENSOR_IN_NAME}}",     TENSOR_IN_NAME}, //输入tensor组的名字
// 		{"{{TENSOR_OUT_NAME}}", TENSOR_OUT_NAME},//输出Tensor的名字 这个就一个Tensor
// 		{"{{IN_DAC_OPS_NAME}}", IN_DAC_OPS_NAME},//输入算子组的组的名字
// 		{"{{OUT_DAC_OPS_NAME}}",OUT_DAC_OPS_NAME}//输出算子组的名字
// 	});
// }
/*上面函数已废弃*/

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

//将算子添加到算子组的模板 之前数据重组时也有添加算子到算子组的模板 每次添加都将要重新设置作用的维度
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

//更改算子作用维度  暂时没有用到
// const char *SET_OP_DIMID_Template = R"~~~(
// 	//更改算子作用维度
//     {{OP_NAME}}.setDimId({{DIMID}});
// )~~~";

// std::string CodeGen_SetOpDimId(std::string OP_NAME, std::string DIMID){
//     return templateString(SET_OP_DIMID_Template,
// 	{
// 		{"{{OP_NAME}}",    OP_NAME},
// 		{"{{DIMID}}",      DIMID}
// 	});
// }

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

//aborted 上面弃用的那个索引生成的修改
// std::string CodeGen_IndexInit2(Dac_Ops ops)
// {
// 	int len = ops.size;
// 	for(int i=0;i<len;i++){
// 		std::string sub_expression = "item_id";
// 		for(int j=i+1;j<len;j++){
// 			sub_expression = sub_expression + "/" + ops[j].name + ".spilit_size";
// 		}
// 		sub_expression = sub_expression + "%" + ops[i].name + ".spilit_size";
// 		ops[i].setExp(sub_expression);
// 	}

// 	std::string expression = "";
// 	for(int i=0;i<len;i++){
// 		expression = expression + templateString(INDEX_INIT_Template,
// 		{
// 			{"{{NAME}}", ops[i].name + "_"},
// 			{"{{EXPRESSION}}", ops[i].getExp()}
// 		});
// 	}

// 	return expression;
// }

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
        //ops[i].setExp(index_expression);
    }

	std::string expression = "";
	for(int i=0;i<len;i++){
		std::string opsname = ops[i].name;
		std::string index_i_expression = index_expression_vector[i];
		expression = expression + templateString(INDEX_INIT_Template,
		{
			{"{{NAME}}", opsname + "_"},//注意这里加了下划线
			//{"{{EXPRESSION}}", ops[i].getExp()}
			{"{{EXPRESSION}}", index_i_expression}
		});
	}
	return expression;
}

//新的嵌入计算的模板 
std::string CodeGen_CalcEmbed2(std::string Name,Args args){
	std::string DacCalcArgs = "(";
	int len = args.size;
	for(int i=0;i<len;i++){
		std::string IndexComb="(";
		for(int j=0;j<args[i].ops.size;j++){
			std::string opsname = args[i].ops[j].name;
			//IndexComb+= args[i].ops[j].name + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";
			IndexComb+= opsname + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";
			if(j!=args[i].ops.size-1) IndexComb+="+";
		}
		IndexComb+=")";
		if(IndexComb == "()")
		{
			DacCalcArgs+=args[i].name;
		}
		else{
			DacCalcArgs+=args[i].name + "+" + IndexComb;
		}		
		if(i==len-1){
			DacCalcArgs+=",";
			for(int j=0;j<len;j++) {
				DacCalcArgs+="info_partition_"+args[j].name.substr(2)+"_accessor";
				if(j!=len-1) DacCalcArgs+=",";
			}
			DacCalcArgs+=");";
		}
		else{
			DacCalcArgs+=",";
		}
	}
	return templateString(CALC_EMBED_Template,
	{
		{"{{DAC_CALC_NAME}}",    Name},
		{"{{DAC_CALC_ARGS}}",    DacCalcArgs}
	});
}