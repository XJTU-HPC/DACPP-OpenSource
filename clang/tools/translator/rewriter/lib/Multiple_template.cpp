#include <string>
#include <iostream>
#include <fstream>
#include <vector>
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
    q[numDevice].memcpy(r_{{NAME}} + {{NAME}}_HaveMalloc, d_{{NAME}}, {{SIZE}}*{{NAME}}_data_SplitSize*sizeof({{TYPE}})).wait();
    {{NAME}}_HaveMalloc += {{SIZE}}*{{NAME}}_data_SplitSize;
    // {{NAME}}_tool.UpdateData(r_{{NAME}},{{NAME}});)~~~";

const char *D2H_MEM_MOV_2_Template = R"~~~(
    // 归约结果返回
    q[numDevice].memcpy(r_{{NAME}} + {{NAME}}_HaveMalloc, d_{{NAME}}, {{SIZE}}*{{NAME}}_data_SplitSize*sizeof({{TYPE}})).wait();
    {{NAME}}_HaveMalloc += {{SIZE}}*{{NAME}}_data_SplitSize;
    // {{NAME}}_tool.UpdateData(r_{{NAME}},{{NAME}});)~~~";

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
    sycl::free(d_{{NAME}}, q[numDevice]);)~~~";

std::string CodeGen_MemFree(std::string Name){
    return templateString(MEM_FREE_Template,
	{
		{"{{NAME}}",            Name}
	});
}

const char* D2H_MEM_MOV_F_Template = R"~~~(
    // 归并结果返回
    {{NAME}}_tool.UpdateData(r_{{NAME}},{{NAME}});)~~~";

std::string CodeGen_D2HMemMovF(std::string Name){
    return templateString(D2H_MEM_MOV_F_Template,
    {
        {"{{NAME}}",            Name},
    });
}

const char* Kernel_Template = R"~~~(
    std::vector<sycl::event> events;
    {{DEVICE_MEM}}
    
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
std::string CodeGen_Kernel(std::string Device_Mem,std::string DeviceMemAlloc,std::string H2DMemMove,std::string KernelExecute,std::string Store_Device_Mem,
                           std::string Load_Device_Mem,std::string Reduction,std::string D2HMemMove,std::string MemFree){
    return templateString(Kernel_Template,
    {
        {"{{DEVICE_MEM}}",        Device_Mem},
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