#include "usm_time_template.h"

namespace USM_TIME_TEMPLATE {

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
//生成函数的总模板，基本包含了生成SYCL程序的大致结构
const char *DAC2SYCL_Template_2 = R"~~~(


// 生成函数调用
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) {
    auto start_time = std::chrono::high_resolution_clock::now();
    double total_kernel_time = 0.0;//内核执行时间
    double total_recon_time = 0.0; //数据重组时间
    double total_memcpy = 0.0; //数据移动时间
    double total_parameter_time = 0.0;  //参数生成时间
	//设备初始化计时
	auto device_init_start=std::chrono::high_resolution_clock::now();

    // 设备选择
    auto selector = default_selector_v;
    queue q(selector,sycl::property::queue::enable_profiling{});
	auto device_init_end=std::chrono::high_resolution_clock::now();
	double deviceInit_time = std::chrono::duration_cast<std::chrono::microseconds>(
    device_init_end - device_init_start).count() * 1e-3; //设备初始化时间

    
	//声明参数生成工具
	auto parameter_start = std::chrono::high_resolution_clock::now();
    ParameterGeneration para_gene_tool;
    // 算子初始化
    {{OP_INIT}}
    //参数生成
	{{ParameterGenerate}}
    // 设备内存分配
	auto parameter_end = std::chrono::high_resolution_clock::now();
    total_parameter_time = std::chrono::duration_cast<std::chrono::microseconds>(parameter_end - parameter_start).count() * 1e-3;
	auto start_memory = std::chrono::high_resolution_clock::now();
    {{DEVICE_MEM_ALLOC}}
    // 数据关联计算
    {{DATA_ASSOC_COMP}}
    // 内存释放
    {{MEM_FREE}}
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = end_time - start_time;
    std::cout << "设备初始化时间: " << deviceInit_time<< " ms" << std::endl;
    std::cout << "参数生成时间: " << total_parameter_time << " ms" << std::endl;
    std::cout << "数据重组时间: " << total_recon_time << " ms" << std::endl;
    std::cout << "数据移动时间: " << total_memcpy << " ms" << std::endl;
    std::cout << "内核执行时间: " << total_kernel_time << " ms" << std::endl;
    std::cout << "总执行时间: " << total_duration.count() << " ms" << std::endl;
} )~~~";
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

//{{InitReductionSplitLength}}
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

//{{DEVICE_MEM_ALLOC}}
//设备内存分配
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
//数据关联计算 {{DATA_ASSOC_COMP}}
const char *DATA_ASSOC_COMP_Template = R"~~~(
	{{H2D_MEM_MOV}}  
	auto memcpy_end = std::chrono::high_resolution_clock::now();
    total_memcpy = std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;
	std::chrono::high_resolution_clock::time_point recon_start = std::chrono::high_resolution_clock::now();
	{{DATA_RECON}}
	std::chrono::high_resolution_clock::time_point recon_end = std::chrono::high_resolution_clock::now();
    total_recon_time = std::chrono::duration_cast<std::chrono::microseconds>(recon_end - recon_start).count() * 1e-3;
	{{KERNEL_EXECUTE}}
	{{REDUCTION}}
	start_memory = std::chrono::high_resolution_clock::now();
	{{D2H_MEM_MOV}}
	memcpy_end = std::chrono::high_resolution_clock::now();
    total_memcpy += std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;
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
const char *H2D_MEM_MOV_Template3 = R"~~~(
    // 数据移动
	{{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{SIZE}}*sizeof({{TYPE}}));
	{{NAME}}.tensor2Array(h_{{NAME}});
    q.memcpy(d_{{NAME}},h_{{NAME}},{{SIZE}}*sizeof({{TYPE}})).wait();
)~~~";

std::string CodeGen_H2DMemMov3(std::string type,std::string name,std::string size){
    return templateString(H2D_MEM_MOV_Template3,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

//这个和上面那个不同 最后一步没有执行数据移动操作 执行赋零操作
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
//下面这个疑似弃用
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

//数据重组
//{{DATA_RECON}}
//这个数据重组是绝大多数USM例子用到的数据重组 在设备端完成并行赋值
const char *DATA_RECON_Template2 = R"~~~(
    // 数据重组
    DataReconstructor<{{TYPE}}> {{NAME}}_tool;
    {{DATA_OPS_INIT}}
    {{NAME}}_tool.init(info_{{NAME}},{{NAME}}_ops);
	start_memory = std::chrono::high_resolution_clock::now();
	{{TYPE}} *r_{{NAME}}=malloc_device<{{TYPE}}>({{NAME}}_Size,q);
	memcpy_end = std::chrono::high_resolution_clock::now();
	total_memcpy += std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - start_memory).count() * 1e-3;
    {{NAME}}_tool.Reconstruct(r_{{NAME}},d_{{NAME}},q);
	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
)~~~";

std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit){
    return templateString(DATA_RECON_Template2,
	{
		{"{{TYPE}}",       type},
		{"{{NAME}}",       name},
		{"{{SIZE}}",       size},
		{"{{DATA_OPS_INIT}}", dataOpsInit}
	});
}
//dac_for专用的数据重组
//目前只有波动方程例子使用 逻辑上重组实际物理地址不重组
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
//通用的USM 非dac_for
const char *KERNEL_EXECUTE_Template1 = R"~~~(
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
	//工作项划分
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
    //队列提交命令组
    sycl::event e = q.submit([&](handler &h) {
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
    });
	e.wait();
	auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    total_kernel_time = (end - start) * 1e-6; // 纳秒转毫秒
    
)~~~";

std::string CodeGen_KernelExecute(std::string SplitSize,std::string AccessorInit,std::string IndexInit,std::string CalcEmbed){
    return templateString(KERNEL_EXECUTE_Template1,
	{
		{"{{ACCESSOR_INIT}}", AccessorInit},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed}
	});
}
//dac_for的内核结构
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
		sycl::event e = q.submit([&](handler &h) {
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
		});
		e.wait();
		auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        total_kernel_time += (end - start) * 1e-6; // 纳秒转毫秒
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
//访问器初始化
//{{ACCESSOR_INIT}}
const char *ACCESSOR_INIT_Template = R"~~~(
        auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read_write>(h);
)~~~";
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
const char *CALC_EMBED_Template = R"~~~(
            {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

//通用的模板 非dac_for 下一行是一个例子
//waveEq(d_matCur+(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),d_matPrev+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),d_matNext+(sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]),info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);
//所有的d_name需要换成r_name
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
//dac_for专用的嵌入计算模板 下一行是一个例子
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
//下面内核中的模板均为dac_for专用的模板

//视图和内存偏移量初始化 dac_for专用
//{{START_OFFSET_INIT}} 
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

//映射信息初始化 dac_for专用
//{{VIRTUALMAP_INIT}}
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

//交换数据操作 dac_for专用
//{{SWAP_OPERATE}}
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
//这个推理需要完善
//{{START_OFFSET_OPERATE}}
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
//{{SWAP_START}}
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
//{{ADD_START}}
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
//{{SUB_START}}
const char *SUB_START_INIT_Template = R"~~~(
	{{NAME}}_tool.sub_start(start_offset);
)~~~";
std::string CodeGen_SubStartInit(std::string name) {
	return templateString(SUB_START_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}

//用于对输出的不断初始化 
//{{NEW_DATA_INIT}}
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
//归约模板
//{{REDUCTION}}
//现在的应用似乎都无需用到归约
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
                    reducer.combine(r_{{NAME}}[(i/{{SPLIT_LENGTH}})*{{SPLIT_LENGTH}}*{{SPLIT_SIZE}}+i%{{SPLIT_LENGTH}}+idx*{{SPLIT_LENGTH}}]);
     	        });
         }).wait();
        }
        q.memcpy(r_{{NAME}},reduction_{{NAME}}, {{SPAN_SIZE}}*sizeof({{TYPE}})).wait();
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
//和上面的区别是这里的是d_name 现在内核中计算使用的是r_name
//dac_for使用这个
const char *REDUCTION_Template_Span1 = R"~~~(
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

std::string CodeGen_Reduction_Span1(std::string SpanSize,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule) {
    return templateString(REDUCTION_Template_Span1,
	{
        {"{{SPAN_SIZE}}",        SpanSize},   
		{"{{SPLIT_SIZE}}",       SplitSize},
		{"{{SPLIT_LENGTH}}",     SplitLength},
		{"{{TYPE}}",             Type},
		{"{{NAME}}",             Name},
		{"{{REDUCTION_RULE}}",   ReductionRule}
	});
}
//设备数据传回主机
//{{D2H_MEM_MOV}}
const char *D2H_MEM_MOV_3_Template = R"~~~(
    // 归并结果返回
    {{NAME}}_tool.UpdateData(r_{{NAME}},d_{{NAME}},q);
	q.memcpy(h_{{NAME}},d_{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();
	{{NAME}}.array2Tensor(h_{{NAME}});
)~~~";

const char *D2H_MEM_MOV_4_Template = R"~~~(
    // 归并结果返回
	q.memcpy(h_{{NAME}},d_{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();
	{{NAME}}.array2Tensor(h_{{NAME}});
)~~~";

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

//{{MEM_FREE}}
//内存释放
const char *MEM_FREE_Template = R"~~~(
    sycl::free(d_{{NAME}}, q);)~~~";

std::string CodeGen_MemFree(std::string Name){
    return templateString(MEM_FREE_Template,
	{
		{"{{NAME}}",            Name}
	});
}

//其他用到的模板
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

}