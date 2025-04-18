#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "buffer_template.h"

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

std::string BUFFER_ACCESSOR_LIST = "";
std::string ACCESSOR_POINTER_LIST = "";
const char *BUFFER_ACCESSOR_Template = R"~~~(
        accessor acc_{{NAME}}{b_{{NAME}}, h};)~~~";
const char *ACCESSOR_POINTER_Template = R"~~~(
            auto* d_{{NAME}} = acc_{{NAME}}.get_multi_ptr<access::decorated::no>().get();)~~~";

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
})~~~";

std::string CodeGen_DAC2SYCL(std::string dacShellName,std::string dacShellParams,std::string opInit,std::string dataRecon,
	std::string deviceMemAlloc,std::string H2DMemMove,std::string kernelExecute,std::string reduction,std::string D2HMemMove){
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
        {"{{D2H_MEM_MOV}}",       D2HMemMove}
	});
}

const char *DEVICE_MEM_ALLOC_Template = R"~~~(
    // Buffer设备内存分配
    buffer <{{TYPE}}> b_{{NAME}}{{{SIZE}}};)~~~";

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size){
    BUFFER_ACCESSOR_LIST += templateString(BUFFER_ACCESSOR_Template,{
        {"{{NAME}}", name}
    });
    ACCESSOR_POINTER_LIST += templateString(ACCESSOR_POINTER_Template,{
        {"{{NAME}}", name}
    });
    return templateString(DEVICE_MEM_ALLOC_Template,{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *DEVICE_MEM_ALLOC_REDUCTION_Template = R"~~~(
    // 规约Buffer设备内存分配
    std::vector<sycl::buffer<{{TYPE}}, 1>> b_reduction_{{NAME}}({{SIZE}}, buffer<{{TYPE}}, 1>{1});
    for(int i = 0; i < {{SIZE}}; i++){
        host_accessor temp_accessor{b_reduction_{{NAME}}[i]};
        temp_accessor[0] = 0;
    })~~~";
//由于规约时不能用偏移量指示buffer,所以需要定义一个vector存结果
std::string CodeGen_DeviceMemAllocReduction(std::string type,std::string name,std::string size){
    return templateString(DEVICE_MEM_ALLOC_REDUCTION_Template,{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *H2D_MEM_MOV_Template = R"~~~(
    { //Buffer主机到设备传输数据
        host_accessor temp_accessor{b_{{NAME}}};
        for(int i = 0; i < {{SIZE}}; i++){
            temp_accessor[i] = r_{{NAME}}[i];
        }
    }
)~~~";

std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size){
    return templateString(H2D_MEM_MOV_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

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

const char *ACCESSOR_INIT_Template = R"~~~(
        auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read_write>(h);)~~~";
std::string CodeGen_AccessorInit(std::string name) {
	return templateString(ACCESSOR_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}

const char *KERNEL_EXECUTE_Template = R"~~~(
    //工作项划分
    sycl::range<3> local(1, 1, {{SPLIT_SIZE}});
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
    {{ACCESSOR_LIST}}
    {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			{{INDEX_INIT}}
            // 获得accessor指针
            {{ACCESSOR_POINTER_LIST}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
    
)~~~";

string CodeGen_KernelExecute_ArrayList(string SplitSize, std::string AccessorInit, string IndexInit, string CalcEmbed, std::initializer_list<string> values){
    
    std::string USE_ACCESSOR_List="";
    std::string USE_ACCESSOR_POINTER_LIST="";
    for(string value : values){
        USE_ACCESSOR_List += templateString(BUFFER_ACCESSOR_Template,{
            {"{{NAME}}", value}
        });
        USE_ACCESSOR_POINTER_LIST += templateString(ACCESSOR_POINTER_Template,{
            {"{{NAME}}", value}
        });
    }    
    return templateString(KERNEL_EXECUTE_Template,
    {
        {"{{SPLIT_SIZE}}",    SplitSize},
        {"{{INDEX_INIT}}",    IndexInit},
        {"{{CALC_EMBED}}",    CalcEmbed},
        {"{{ACCESSOR_INIT}}", AccessorInit},
        {"{{ACCESSOR_LIST}}",   USE_ACCESSOR_List},
        {"{{ACCESSOR_POINTER_LIST}}",   USE_ACCESSOR_POINTER_LIST}
    });
}

std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit, std::string CalcEmbed){
    return templateString(KERNEL_EXECUTE_Template,
	{
		{"{{SPLIT_SIZE}}",    SplitSize},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed},
        {"{{ACCESSOR_INIT}}", AccessorInit},
        {"{{ACCESSOR_LIST}}",   BUFFER_ACCESSOR_LIST},
        {"{{ACCESSOR_POINTER_LIST}}",   ACCESSOR_POINTER_LIST}
	});
}

const char *REDUCTION_Template_Span = R"~~~(
    // 归约
    if({{SPLIT_SIZE}} > 1)
    {
        for(int i=0;i<{{SPAN_SIZE}};i++) {
            q.submit([&](handler &h) {
                accessor d_{{NAME}}{b_{{NAME}}, h};
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
            host_accessor b_acc{b_{{NAME}}};
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

const char *D2H_MEM_MOV_1_Template = R"~~~(
    // 归并结果返回
    {
        host_accessor temp_accessor{b_{{NAME}}};
        for(int i = 0; i < {{SIZE}}; i++){
            r_{{NAME}}[i] = temp_accessor[i];
        }
    }
    {{NAME}}_tool.UpdateData(r_{{NAME}},{{NAME}});)~~~";

const char *D2H_MEM_MOV_2_Template = R"~~~(
    // 归约结果返回
    {
        host_accessor temp_accessor{b_{{NAME}}};
        for(int i = 0; i < {{SIZE}}; i++){
            r_{{NAME}}[i] = temp_accessor[i];
        }
    }
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

}