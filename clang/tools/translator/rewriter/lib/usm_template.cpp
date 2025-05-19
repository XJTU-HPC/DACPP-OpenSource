#include "usm_template.h"

namespace USM_TEMPLATE {

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
const char *H2D_MEM_MOV_Template3 = R"~~~(
    // 数据移动
	{{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{SIZE}}*sizeof({{TYPE}}));
	{{NAME}}.tensor2Array(h_{{NAME}});
    q.memcpy(d_{{NAME}},h_{{NAME}},{{SIZE}}*sizeof({{TYPE}})).wait();)~~~";

std::string CodeGen_H2DMemMov3(std::string type,std::string name,std::string size){
    return templateString(H2D_MEM_MOV_Template3,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}
const char *H2D_MEM_MOV_Template_Out = R"~~~(
    // 数据移动
	{{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{SIZE}}*sizeof({{TYPE}}));
	// {{NAME}}.tensor2Array(h_{{NAME}});
    q.memset(d_{{NAME}}, 0, {{SIZE}}*sizeof({{TYPE}})).wait();)~~~";

std::string CodeGen_H2DMemMov_Out(std::string type,std::string name,std::string size){
    return templateString(H2D_MEM_MOV_Template_Out,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}


const char *DEVICE_DATA_INIT_Template = R"~~~(
    // 设备数据初始化
    q.memset(d_{{NAME}},0,{{SIZE}}*sizeof({{TYPE}})).wait();)~~~";

std::string CodeGen_DeviceDataInit(std::string type,std::string name,std::string size){
    return templateString(DEVICE_DATA_INIT_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

// const char *KERNEL_EXECUTE_Template = R"~~~(
//     //工作项划分
//     sycl::range<3> local(1, 1, {{SPLIT_SIZE}});
//     sycl::range<3> global(1, 1, 1);
//     //队列提交命令组
//     q.submit([&](handler &h) {
//         // 访问器初始化
//         {{ACCESSOR_INIT}}
//         h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
//             const auto item_id = item.get_local_id(2);
//             // 索引初始化
// 			{{INDEX_INIT}}
//             // 嵌入计算
// 			{{CALC_EMBED}}
//         });
//     }).wait();
    
// )~~~";

// std::string CodeGen_KernelExecute(std::string SplitSize,std::string AccessorInit,std::string IndexInit,std::string CalcEmbed){
//     return templateString(KERNEL_EXECUTE_Template,
// 	{
// 		{"{{SPLIT_SIZE}}",    SplitSize},
// 		{"{{ACCESSOR_INIT}}", AccessorInit},
// 		{"{{INDEX_INIT}}",    IndexInit},
// 		{"{{CALC_EMBED}}",    CalcEmbed}
// 	});
// }

//普通的USM内核
const char *KERNEL_EXECUTE_Template1 = R"~~~(
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			{{INDEX_INIT}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
    
)~~~";

std::string CodeGen_KernelExecute(std::string SplitSize,std::string AccessorInit,std::string IndexInit,std::string CalcEmbed){
    return templateString(KERNEL_EXECUTE_Template1,
	{
		{"{{ACCESSOR_INIT}}", AccessorInit},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed}
	});
}

//Template2模板只要是为了支持dac_for这种特殊的结构
const char *KERNEL_EXECUTE_Template2 = R"~~~(
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
	{{START_OFFSET_INIT}}
    for(int step = 0;step < {{TIME_STEPS}};step ++)
    {
        //队列提交命令组
		q.submit([&](handler &h) {
			// 访问器初始化
			{{ACCESSOR_INIT}}
			{{VIRTUALMAP_INIT}}
			h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
				const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
				if(item_id >= Item_Size)
					return;
				// 索引初始化
				{{INDEX_INIT}}
				// 嵌入计算
				{{CALC_EMBED}}
			});
		}).wait();
		if(step < {{TIME_STEPS}} - 1)
		{
			{{SWAP_OPERATE}}
			{{START_OFFSET_OPERATE}}
			{{NEW_DATA_INIT}}
		}
    }
)~~~";

std::string CodeGen_KernelExecute(std::string StartOffsetInit,std::string TimeSteps,std::string AccessorInit,std::string VirtualMapInit,std::string IndexInit,std::string CalcEmbed,std::string SwapOperate,std::string StartOffsetOperate,std::string NewDataInit){
    return templateString(KERNEL_EXECUTE_Template2,
	{
		{"{{START_OFFSET_INIT}}", StartOffsetInit},
		{"{{TIME_STEPS}}",        TimeSteps},
		{"{{ACCESSOR_INIT}}",     AccessorInit},
		{"{{VIRTUALMAP_INIT}}",   VirtualMapInit},
		{"{{INDEX_INIT}}",        IndexInit},
		{"{{CALC_EMBED}}",        CalcEmbed},
		{"{{SWAP_OPERATE}}",      SwapOperate},
		{"{{START_OFFSET_OPERATE}}" ,StartOffsetOperate},
		{"{{NEW_DATA_INIT}}",     NewDataInit}
	});
}

//内核中会用到
const char *START_OFFSET_INIT_Template = R"~~~(
        std::vector<int> start_offset(info_{{NAME}}.dim);
		for(int i = 0;i < info_{{NAME}}.dim;i ++)
		{
			start_offset[i] = {{NUM}};
		}
)~~~";
std::string CodeGen_StartOffsetInit(std::string name,std::string num) {
	return templateString(START_OFFSET_INIT_Template,
	{
		{"{{NAME}}",    name},
		{"{{NUM}}",     num}
	});
}

//内核中会用到
const char *ACCESSOR_INIT_Template = R"~~~(
        auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read_write>(h);
)~~~";
std::string CodeGen_AccessorInit(std::string name) {
	return templateString(ACCESSOR_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}

//内核中会用到
const char *VIRTUALMAP_INIT_Template = R"~~~(
        VirtualMapParams {{NAME}}_params {
			.block_size = {{NAME}}_tool.block_size,
			.dim_num = {{NAME}}_tool.dim_num,
			.start = {{NAME}}_tool.start_buffer.get_access<sycl::access::mode::read_write>(h),
			.data_shape = {{NAME}}_tool.data_shape_buffer.get_access<sycl::access::mode::read_write>(h),
			.data_stride = {{NAME}}_tool.data_stride_buffer.get_access<sycl::access::mode::read_write>(h),
			.view_shape = {{NAME}}_tool.view_shape_buffer.get_access<sycl::access::mode::read_write>(h),
			.view_stride = {{NAME}}_tool.view_stride_buffer.get_access<sycl::access::mode::read_write>(h),
			.block_shape = {{NAME}}_tool.block_shape_buffer.get_access<sycl::access::mode::read_write>(h),
			.block_stride = {{NAME}}_tool.block_stride_buffer.get_access<sycl::access::mode::read_write>(h),
			.block_move = {{NAME}}_tool.block_move_buffer.get_access<sycl::access::mode::read_write>(h),
			.grid_shape = {{NAME}}_tool.grid_shape_buffer.get_access<sycl::access::mode::read_write>(h),
			.grid_stride = {{NAME}}_tool.grid_stride_buffer.get_access<sycl::access::mode::read_write>(h),
		};
)~~~";
std::string CodeGen_VirtualMapInit(std::string name) {
	return templateString(VIRTUALMAP_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}


const char *CALC_EMBED_Template = R"~~~(
            {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";
//内核中会用到 大多数例子用到
//waveEq(d_matCur+(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),d_matPrev+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),d_matNext+(sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]),info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);

std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names){
	std::string DacCalcArgs = "(";//name(参数) 中的(
	int len = args.size;//len 表示有几组数据
	for(int i=0;i<len;i++)
	{
		std::string IndexComb="(";//这个左括号代表数据在长向量中偏移量
		for(int j=0;j<args[i].ops.size;j++)//第j组数据中算子组中包含的算子的个数
		{
			std::string opsname = args[i].ops[j].name;//算子是char[5]类型的，这里需要转换为string后面方便使用。得到第j个算子的名字
			IndexComb+= opsname + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";//给算子加一个下划线 这里是为了得到算子的划分长度 具体逻辑忘了
			if(j!=args[i].ops.size-1) IndexComb+="+";//还没有到最后一个算子，就继续添加+号
		}
		IndexComb+=")";//添加偏移量中的右括号
		if(IndexComb == "()")//如果是空的话，就不要加IndexComb了
		{
			DacCalcArgs+=args[i].name;
		}
		else
		{
			DacCalcArgs+=args[i].name + "+" + IndexComb;
		}		
		DacCalcArgs += ",";//因为后面还有其他的参数，所以就继续添加 ,
	}
	//添加数据访问器的相关参数
	for (int z = 0; z < accessor_names.size(); z++) 
	{
		DacCalcArgs += "info_partition_" + accessor_names[z]+"_accessor";
		if (z == accessor_names.size() - 1) 
		{
			DacCalcArgs += ");";
		} 
		else 
		{
			DacCalcArgs += ",";
		}
	}
	return templateString(CALC_EMBED_Template,
	{
		{"{{DAC_CALC_NAME}}",    Name},
		{"{{DAC_CALC_ARGS}}",    DacCalcArgs}
	});
}

//内核中用到 dac_for专用
//dac_for的嵌入计算样子和别的不同
//waveEq(d_matCur, d_matPrev, d_matNext, (sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]), (idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]), (sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]), matCur_params, matPrev_params, matNext_params, info_partition_matCur_accessor, info_partition_matPrev_accessor, info_partition_matNext_accessor);

std::string CodeGen_CalcEmbed3(std::string Name,Args args,std::vector<std::string> accessor_names){
	std::string DacCalcArgs = "(";//name(参数) 中的(
	int len = args.size;//len 表示有几组数据 len必须是一个大于0的整数
	for(int i = 0;i < len;i ++)//(d_matCur, d_matPrev, d_matNext,
	{
		DacCalcArgs += args[i].name;
		DacCalcArgs += ",";
	}

	for(int i = 0;i < len;i ++)//(d_matCur, d_matPrev, d_matNext, (sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]), (idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]), (sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]), 
	{
		std::string IndexComb="(";//这个左括号代表数据在长向量中偏移量
		for(int j = 0;j < args[i].ops.size;j ++)//第j组数据中算子组中包含的算子的个数
		{
			std::string opsname = args[i].ops[j].name;//算子是char[5]类型的，这里需要转换为string后面方便使用。得到第j个算子的名字
			IndexComb+= opsname + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";//给算子加一个下划线 这里是为了得到算子的划分长度 具体逻辑忘了
			if(j != args[i].ops.size - 1) IndexComb += "+";//还没有到最后一个算子，就继续添加+号
		}
		IndexComb += ")";//添加偏移量中的右括号
		if(IndexComb == "()")//如果是空的话，就不要加IndexComb了
		{
			DacCalcArgs += "0";
		}
		else
		{
			DacCalcArgs += IndexComb;
		}		
		DacCalcArgs += ",";//因为后面还有其他的参数，所以就继续添加 ,
	}

	//添加params的相关参数 accessor_names中的name恰好也是params的名字
	for(int i = 0;i < accessor_names.size();i ++)//(d_matCur, d_matPrev, d_matNext, (sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]), (idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]), (sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]), matCur_params, matPrev_params, matNext_params,
	{
		DacCalcArgs += accessor_names[i] + "_params,";
	}

	//添加数据访问器的相关参数
	for (int z = 0; z < accessor_names.size(); z++)//(d_matCur, d_matPrev, d_matNext, (sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]), (idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]), (sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]), matCur_params, matPrev_params, matNext_params, info_partition_matCur_accessor, info_partition_matPrev_accessor, info_partition_matNext_accessor);
	{
		DacCalcArgs += "info_partition_" + accessor_names[z]+"_accessor";
		if (z == accessor_names.size() - 1) 
		{
			DacCalcArgs += ");";
		} 
		else 
		{
			DacCalcArgs += ",";
		}
	}
	return templateString(CALC_EMBED_Template,
	{
		{"{{DAC_CALC_NAME}}",    Name},
		{"{{DAC_CALC_ARGS}}",    DacCalcArgs}
	});
}

//内核中用到 dacfor专用 交换数据操作
//这块要写成d_的形式吗？
const char *SWAP_OPERATE_INIT_Template = R"~~~(
	swap(d_{{NAME1}},d_{{NAME2}});
	swap({{NAME1}}_tool.data_stride_buffer, {{NAME2}}_tool.data_stride_buffer);
	swap({{NAME1}}_tool.data_shape_buffer, {{NAME2}}_tool.data_shape_buffer);
)~~~";
std::string CodeGen_SwapOperateInit(std::string name1,std::string name2) {
	return templateString(SWAP_OPERATE_INIT_Template,
	{
		{"{{NAME1}}",    name1},
		{"{{NAME2}}",    name2}
	});
}

//内核中要用到 dacfor专用 数据视图与数据内存偏移量操作
//感觉这一块和波动方程这个例子高度耦合 先这样去操作吧
const char *START_OFFSET_OPERATE_INIT_Template = R"~~~(
	{{SWAP_START}}
	{{ADD_START}}
	{{SUB_START}}
)~~~";
std::string CodeGen_StartOffsetOperateInit(std::string swap_start,std::string add_start,std::string sub_start) {
	return templateString(START_OFFSET_OPERATE_INIT_Template,
	{
		{"{{SWAP_START}}",    swap_start},
		{"{{ADD_START}}",     add_start},
		{"{{SUB_START}}",     sub_start}
	});
}

//START_OFFSET_OPERATE_INIT_Template 的子模板
const char *SWAP_START_INIT_Template = R"~~~(
	swap({{NAME1}}_tool.start, {{NAME2}}_tool.start);
)~~~";
std::string CodeGen_SwapStartInit(std::string name1,std::string name2) {
	return templateString(SWAP_START_INIT_Template,
	{
		{"{{NAME1}}",    name1},
		{"{{NAME2}}",    name2}
	});
}

//START_OFFSET_OPERATE_INIT_Template的子模板
const char *ADD_START_INIT_Template = R"~~~(
	{{NAME}}_tool.add_start(start_offset);
)~~~";
std::string CodeGen_AddStartInit(std::string name) {
	return templateString(ADD_START_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}

//START_OFFSET_OPERATE_INIT_Template的子模板
const char *SUB_START_INIT_Template = R"~~~(
	{{NAME}}_tool.sub_start(start_offset);
)~~~";
std::string CodeGen_SubStartInit(std::string name) {
	return templateString(SUB_START_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}

//内核中要用到 dacfor专用 用于对输出的不断初始化
const char *NEW_DATA_INIT_Template = R"~~~(
	{{NAME}}_tool.init(info_{{NAME}},{{NAME}}_ops);
	sycl::free(d_{{NAME}},q);
	d_{{NAME}} = malloc_device<{{TYPE}}>({{NAME}}.getSize(),q);
	q.memset(d_{{NAME}},0,{{NAME}}.getSize() * sizeof({{TYPE}})).wait();
)~~~";
std::string CodeGen_NewDataInit(std::string name,std::string type) {
	return templateString(NEW_DATA_INIT_Template,
	{
		{"{{NAME}}",    name},
		{"{{TYPE}}",    type}
	});
}


//归约之后再去考虑吧 现在基本用不到归约
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

// const char *D2H_MEM_MOV_1_Template = R"~~~(
//     // 归并结果返回
//     q.memcpy(r_{{NAME}}, d_{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();
//     {{NAME}}_tool.UpdateData(r_{{NAME}},{{NAME}});)~~~";

// const char *D2H_MEM_MOV_2_Template = R"~~~(
//     // 归约结果返回
//     q.memcpy(r_{{NAME}},d__{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();
//     {{NAME}}_tool.UpdateData(r_{{NAME}},{{NAME}});)~~~";

// std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction){
//     if(isReduction){
// 		return templateString(D2H_MEM_MOV_2_Template,
// 		{
// 			{"{{TYPE}}",            Type},
// 			{"{{NAME}}",            Name},
// 			{"{{SIZE}}",            Size}
// 		});
// 	}
// 	else{
// 		return templateString(D2H_MEM_MOV_1_Template,
// 		{
// 			{"{{TYPE}}",            Type},
// 			{"{{NAME}}",            Name},
// 			{"{{SIZE}}",            Size}
// 		});
// 	}
// }

const char *D2H_MEM_MOV_3_Template = R"~~~(
    // 归并结果返回
	{{TYPE}} *d_myTensor2=malloc_device<{{TYPE}}>({{SIZE}},q);
    {{NAME}}_tool.UpdateData(d_{{NAME}},d_myTensor2,q);
	{{TYPE}}* res = ({{TYPE}}*)malloc({{SIZE}}*sizeof({{TYPE}}));
	q.memcpy(res,d_myTensor2, {{SIZE}}*sizeof({{TYPE}})).wait();
	{{NAME}}.array2Tensor(res);)~~~";

const char *D2H_MEM_MOV_4_Template = R"~~~(
    // 归并结果返回
	{{TYPE}} *d_myTensor2=malloc_device<{{TYPE}}>({{SIZE}},q);
    {{NAME}}_tool.UpdateData(d_{{NAME}},d_myTensor2,q);
	{{TYPE}}* res = ({{TYPE}}*)malloc({{SIZE}}*sizeof({{TYPE}}));
	q.memcpy(res,d_myTensor2, {{SIZE}}*sizeof({{TYPE}})).wait();
	{{NAME}}.array2Tensor(res);)~~~";


std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction){
    if(isReduction){
		return templateString(D2H_MEM_MOV_4_Template,
		{
			{"{{TYPE}}",            Type},
			{"{{NAME}}",            Name},
			{"{{SIZE}}",            Size}
		});
	}
	else{
		return templateString(D2H_MEM_MOV_3_Template,
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

//dac_for专用的数据重组
const char *DATA_RECON_Template3 = R"~~~(
    // 数据重组
    DataReconstructor<{{TYPE}}> {{NAME}}_tool;
    {{DATA_OPS_INIT}}
    {{NAME}}_tool.init(info_{{NAME}},{{NAME}}_ops);
	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
)~~~";

std::string CodeGen_DataReconstruct_Dacfor(std::string type,std::string name,std::string size,std::string dataOpsInit){
    return templateString(DATA_RECON_Template3,
	{
		{"{{TYPE}}",       type},
		{"{{NAME}}",       name},
		{"{{SIZE}}",       size},
		{"{{DATA_OPS_INIT}}", dataOpsInit}
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

const char *OP_PUSH_BACK2TOOL_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{NAME}}_tool.push_back({{OP_NAME}});)~~~";

std::string CodeGen_OpPushBack2Tool(std::string name, std::string opName, std::string dimId){
    return templateString(OP_PUSH_BACK2TOOL_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{NAME}}",       name},
		{"{{DIM_ID}}",     dimId}
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



}