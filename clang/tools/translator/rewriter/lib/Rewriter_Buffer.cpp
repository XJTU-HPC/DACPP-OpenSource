#include <set>
#include <iostream>
#include <sstream>
#include "clang/Rewrite/Core/Rewriter.h"
#include "Rewriter.h"
#include "Split.h"
#include "Param.h"
#include "dacInfo.h"
#include "universal_template.h"
#include "buffer_template.h"
#include "usm_template.h"

// void DFS(int node, std::vector<bool>& visited, Dac_Ops ops, Dac_Ops component, int componentID, std::vector<std::string>& sets) {
//             visited[node] = true;
//             sets[node] = std::to_string(componentID);  // 给当前节点标记一个连通分量的ID

//             // 将当前节点加入到当前连通分量
//             component.push_back(ops[node]);

//             // 遍历所有与当前节点相连的节点
//             for (int i = 0; i < ops.size; ++i) {
//                 int k;
//                 if (!visited[i] && GetBindInfo(node, i,k)) {
//                     DFS(i, visited, ops, component, componentID, sets);
//                 }
//             }   
// }



void dacppTranslator::Rewriter::rewriteDac_Buffer() {
    std::cout << "软编码BUFFER版本翻译" << std::endl;

    std::string code = "";
    
    // 添加头文件
    for(int i = 0; i < dacppFile->getNumHeaderFile(); i++) {
        code += "#include " + dacppFile->getHeaderFile(i)->getName() + "\n";
    }
    code += "\n";
    
    // 添加命名空间
    for(int i = 0; i < dacppFile->getNumNameSpace(); i++) {
        code += "using namespace " + dacppFile->getNameSpace(i)->getName() + ";\n";
    }
    code += "\n";
    
    // 添加数据关联表达式对应的划分结构和计算结构
    for(int i = 0; i < dacppFile->getNumExpression(); i++) {
        Expression* expr = dacppFile->getExpression(i);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        

        std::vector<BINDINFO> info;
        shell->GetBindInfo(&info);//获取binding信息
         Dac_Ops ops;
        std::vector<std::string> sets;  // 存储每个节点所属的连通分量的ID
        std::vector<std::string> bindoffset;  // 存储每个节点相对于其连通分量代表节点的偏移
        int componentID = 1;                               // 连通分量的编号
        // std::string s;                                     // 用于存储 GetBindInfo 的偏移量字符串
        for(int i = 0; i < info.size(); i++){
            // std::cout << shell->search_symbol(info[i].v)->getId() << std::endl;
            // std::cout << info[i].icls << std::endl;
            if(info[i].offset.empty())
                info[i].offset = "0"; 
            // std::cout << info[i].offset << std::endl;
        }
        for(int i = 0; i < info.size(); i++){
            if(shell->search_symbol(info[i].v)->type.compare("IndexSplit") == 0) {
                dacppTranslator::IndexSplit* index = static_cast<dacppTranslator::IndexSplit*>(shell->search_symbol(info[i].v));
                Index tmp = Index(index->getId());
                tmp.SetSplitSize(index->getSplitNumber());
                tmp.setDimId(index->getDimIdx());
                ops.push_back(tmp);
                sets.push_back("id"+std::to_string(info[i].icls));
                bindoffset.push_back(info[i].offset);
            }else if(shell->search_symbol(info[i].v)->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit* r = static_cast<dacppTranslator::RegularSplit*>(shell->search_symbol(info[i].v));
                RegularSlice tmp = RegularSlice(r->getId(), r->getSplitSize(), r->getSplitStride());
                //std::cout << r->getId() << " " << r->getSplitSize() << " " << r->getSplitStride() << std::endl;
                tmp.SetSplitSize(r->getSplitNumber());
                tmp.setDimId(r->getDimIdx());
                ops.push_back(tmp);
                sets.push_back("id"+std::to_string(info[i].icls));
                bindoffset.push_back(info[i].offset);
            }
        }

        // 计算结构
        code += "void " + calc->getName() + "(";
        for(int count = 0; count < calc->getNumParams(); count++) {
            code += calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName() + ",";
        }
        for(int count = 0; count < calc->getNumParams(); count++) {
            code += "sycl::accessor<int, 1, sycl::access::mode::read_write> info_" + calc->getParam(count)->getName() + "_acc";
            if(count != calc->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ") \n";
        for(int count = 0; count < calc->getNumBody(); count++) {
            
            code += calc->getBody(count) + "\n";
        }

        // std::cout << code;

        std::string dacShellName = shell->getName() + "_" + calc->getName();
        std::string dacShellParams = "";
        for (int count = 0; count < shell->getNumParams(); count++) {
            Param* param = shell->getParam(count);
            dacShellParams += param->getType() + " " + param->getName();
            if(count != shell->getNumParams() - 1) { 
                dacShellParams += ", ";
            }
        }
        //算子初始化
        std::string opInit = "";
        std::string infoInit = "";
        std::set<std::string> HadInit;
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){//遍历每个参数
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            infoInit += UNIVERSAL_TEMPLATE::CodeGen_DataInfoInit(shellParam->getName());
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){//遍历每个参数的划分
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}//保形算子会被跳过
                Split* split = shellParam->getSplit(NumSplit);
                if(split->type.compare("IndexSplit") == 0){
                    if(HadInit.count(split->getId()) == 1){//已经初始化过了
                        continue;
                    }
                    IndexSplit* indexSplit = static_cast<IndexSplit*>(split);
                    opInit += UNIVERSAL_TEMPLATE::CodeGen_IndexInit2(indexSplit->getId(),std::to_string(NumSplit),"info_"+shellParam->getName());//初始化降维算子
                    HadInit.insert(indexSplit->getId());

                }
                else if(split->type.compare("RegularSplit") == 0){
                    if(HadInit.count(split->getId()) == 1){//已经初始化过了
                        continue;
                    }
                    RegularSplit* regularSplit = static_cast<RegularSplit*>(split);
                    opInit += UNIVERSAL_TEMPLATE::CodeGen_RegularSliceInit2(regularSplit->getId(),std::to_string(regularSplit->getSplitSize()),
                    std::to_string(regularSplit->getSplitStride()),std::to_string(NumSplit),"info_"+shellParam->getName());//初始化规则分区算子
                    HadInit.insert(regularSplit->getId());
                }
            }
        }
        opInit = infoInit + opInit;
        // std::cout << opInit;
        //获取算子与作用数据块的关系
        std::string add2Op = "";
        std::string dataOpsInit = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                add2Op += UNIVERSAL_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(NumSplit),shellParam->getName()+"_Ops");//将算子添加到他作用的参数中去
            }
            dataOpsInit += UNIVERSAL_TEMPLATE::CodeGen_DataOpsInit2(shellParam->getName()+"_Ops",add2Op);
            add2Op = "";
        }

        //划分输入输出算子
        std::string add2Op_inops = "";
        std::string add2Op_outops = "";
        std::string add2Op_reductions = "";
        std::set<std::string> setIn;
        std::set<std::string> setOut;
        Dac_Ops Inops;
        std::string set;
        int inflag = 0;
        int outflag = 0;
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                if(shellParam->getRw() == 1){
                    for(int i = 0; i < info.size(); i++){
                        if(shell->search_symbol(info[i].v)->getId() == split->getId())
                                set = std::to_string(info[i].icls);
                    }
                    if(setOut.count(set) == 1) {
                        continue;
                    }
                    add2Op_outops += UNIVERSAL_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(outflag),"Out_Ops");//将算子添加到输出算子组中去
                    add2Op_reductions += UNIVERSAL_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(outflag),"Reduction_Ops");//将算子添加到归约算子组中去
                    outflag++;
                    setOut.insert(set);
                }
                else{
                    for(int i = 0; i < info.size(); i++){
                        if(shell->search_symbol(info[i].v)->getId() == split->getId())
                            set = std::to_string(info[i].icls);
                    }
                    if(setIn.count(set) == 1) {
                        continue;
                    }
                    add2Op_inops += UNIVERSAL_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(inflag),"In_Ops");//将算子添加到输入算子组中去
                    Dac_Op op = Dac_Op(split->getId(),0,inflag);
                    inflag++;
                    // std::cout<<inflag<<std::endl;
                    Inops.push_back(op);
                    setIn.insert(set);
                }
            }
        }
        std::string dataOpsInit_inops = UNIVERSAL_TEMPLATE::CodeGen_DataOpsInit2("In_Ops",add2Op_inops);
        std::string dataOpsInit_outops = UNIVERSAL_TEMPLATE::CodeGen_DataOpsInit2("Out_Ops",add2Op_outops);
        std::string dataOpsInit_reductions = UNIVERSAL_TEMPLATE::CodeGen_DataOpsInit2("Reduction_Ops",add2Op_reductions);
        // std::cout << dataOpsInit_inops;
        // std::cout << dataOpsInit_outops;
        // std::cout << dataOpsInit_reductions;

        //生成设备内存分配大小
        std::string divice_memory = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(shellParam->getRw() == 1){
                divice_memory += UNIVERSAL_TEMPLATE::CodeGen_DeviceMemSizeGenerate(shellParam->getName()+"_Size","In_Ops","Out_Ops","info_"+shellParam->getName());
                divice_memory += UNIVERSAL_TEMPLATE::CodeGen_DeviceMemSizeGenerate("Reduction_Size","info_"+shellParam->getName(),"Reduction_Ops");
            }
            else{
                divice_memory += UNIVERSAL_TEMPLATE::CodeGen_DeviceMemSizeGenerate(shellParam->getName()+"_Size","info_"+shellParam->getName(),shellParam->getName()+"_Ops");
            }
        }
        // std::cout << divice_memory;

        //生成算子的划分长度
        std::string splitLength = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(shellParam->getRw() == 1){
               splitLength += UNIVERSAL_TEMPLATE::CodeGen_Init_Split_Length("In_Ops",shellParam->getName()+"_Size");
            }
            else{
                splitLength += UNIVERSAL_TEMPLATE::CodeGen_Init_Split_Length(shellParam->getName()+"_Ops",shellParam->getName()+"_Size");
            }
        }
        // std::cout << splitLength;

        //AddDacOps2Vector
        std::string AddDacOps2Vector = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(shellParam->getRw() == 1){
                AddDacOps2Vector += UNIVERSAL_TEMPLATE::CodeGen_Add_DacOps2Vector("ops_s","In_Ops");
            }
            else{
                AddDacOps2Vector += UNIVERSAL_TEMPLATE::CodeGen_Add_DacOps2Vector("ops_s",shellParam->getName()+"_Ops");
            }
        }
        std::string DeclareDacOpsVector = UNIVERSAL_TEMPLATE::CodeGen_Declare_DacOps_Vector("ops_s",AddDacOps2Vector);
        // std::cout << std::to_string(shell->getNumShellParams()) << std::to_string(inflag) << std::endl;
        std::string InitSplitLengthMatrix = UNIVERSAL_TEMPLATE::CodeGen_Init_Split_Length_Matrix(DeclareDacOpsVector,std::to_string(shell->getNumShellParams()),std::to_string(inflag),"ops_s");

        //计算工作分配数量
        std::string item_number = UNIVERSAL_TEMPLATE::CodeGen_Init_Work_Item_Number("Item_Size","In_Ops");
        // std::cout << item_number;

        std::string InitOPS =  dataOpsInit + dataOpsInit_inops + dataOpsInit_outops + dataOpsInit_reductions;

            //生成归约中Split_size中的大小
        std::string Init_Reduction_Split_Size = UNIVERSAL_TEMPLATE::CodeGen_Init_Reduction_Split_Size("Reduction_Split_Size","In_Ops","Out_Ops");
        //std::cout << Init_Reduction_Split_Size;

        //生成归约中Split_length大小
        std::string Init_Reduction_Split_Length = UNIVERSAL_TEMPLATE::CodeGen_Init_Reduction_Split_Length("Reduction_Split_Length","Out_Ops");

        std::string ParameterGenerate = UNIVERSAL_TEMPLATE::CodeGen_ParameterGenerate(InitOPS,divice_memory,splitLength,InitSplitLengthMatrix,item_number,Init_Reduction_Split_Size,Init_Reduction_Split_Length); 
        // std::cout << ParameterGenerate;
        //设置内存分配
        std::string deviceMemAlloc = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(shellParam->getRw() == 1){
                deviceMemAlloc += BUFFER_TEMPLATE::CodeGen_DeviceMemAlloc(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");
                deviceMemAlloc += BUFFER_TEMPLATE::CodeGen_DeviceMemAllocReduction(shellParam->getBasicType(),shellParam->getName(),"Reduction_Size");
            }
            else{
                deviceMemAlloc += BUFFER_TEMPLATE::CodeGen_DeviceMemAlloc(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");
            }
        }
        // std::cout << deviceMemAlloc;

        //主机数据移动至设备
        std::string H2DMemMove = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            H2DMemMove += BUFFER_TEMPLATE::CodeGen_DeviceDataInit(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");
            if(shellParam->getRw() == 0){
                H2DMemMove += BUFFER_TEMPLATE::CodeGen_H2DMemMov(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");
            }
        }
        // std::cout << H2DMemMove;

       
        std::string BindingInit = UNIVERSAL_TEMPLATE::CodeGen_IndexInit2(ops,sets,bindoffset);

        // 嵌入计算
        Args args = Args();
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            Dac_Ops ops;
            if(shellParam->getRw()==0){
                for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                    if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                    Dac_Op op = Dac_Op(shellParam->getSplit(NumSplit)->getId(), 0, NumSplit);
                    ops.push_back(op);
          
                }
            }else{
                for(int Countin = 0; Countin < Inops.size; Countin++){
                    ops.push_back(Inops[Countin]);
                }
            }
            DacData data = DacData("d_"+shellParam->getName(), 0, ops);
            args.push_back(data);
        }
        // std::string CalcEmbed = CodeGen_CalcEmbed2(calc->getName(),args);//注意调用了新的嵌入计算模板
        std::vector<std::string> accessor_names;
        for (int z = 0; z < shell->getNumParams(); z++) {
            accessor_names.push_back(shell->getParam(z)->getName());
        }
        std::string CalcEmbed = UNIVERSAL_TEMPLATE::CodeGen_CalcEmbed2(calc->getName(),args, accessor_names);
        std::string AccessorInit = "";
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            AccessorInit += BUFFER_TEMPLATE::CodeGen_AccessorInit(shell->getShellParam(argCount)->getName());
        }
	    std::string KernelExecute = BUFFER_TEMPLATE::CodeGen_KernelExecute("Item_Size",AccessorInit,BindingInit,CalcEmbed);//注意这里面填的size的大小需要是前面算出来的大小
        // std::cout << KernelExecute;

        std::string Reduction;
        std::string D2HMemMove;
        std::string ReductionRule = "sycl::plus<>()";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(shellParam->getRw() == 1){
                Reduction = BUFFER_TEMPLATE::CodeGen_Reduction_Span("Reduction_Size","Reduction_Split_Size","Reduction_Split_Length",shellParam->getName(),shellParam->getBasicType(),ReductionRule);
	            D2HMemMove = BUFFER_TEMPLATE::CodeGen_D2HMemMov(shellParam->getName(),shellParam->getBasicType(),shellParam->getName()+"_Size",false);
            }
        }
        // std::cout << Reduction;
        // std::cout << D2HMemMove;
       
       //dataRecon
       std::string dataRecon = "";
       for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            std::string opPushBack = "";
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                opPushBack += UNIVERSAL_TEMPLATE::CodeGen_OpPushBack2Ops(shellParam->getName(),split->getId(),std::to_string(split->getDimIdx()));
            }
            std::string dataOpsInit = UNIVERSAL_TEMPLATE::CodeGen_DataOpsInit(shellParam->getName(),opPushBack);
            dataRecon += UNIVERSAL_TEMPLATE::CodeGen_DataReconstruct(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size",dataOpsInit);
       }
        // std::cout << dataRecon;
        // std::cout << "-----------------------------------------------------------------------";
        // 内存释放
        std::string MemFree = "";
        // for(int j = 0; j < shell->getNumParams(); j++) {
        //     ShellParam* shellParam = shell->getShellParam(j);
        //     MemFree += BUFFER_TEMPLATE::CodeGen_MemFree(shellParam->getName());
        // }

        std::string dac = UNIVERSAL_TEMPLATE::CodeGen_DataAssocComp(dataRecon, H2DMemMove, KernelExecute, Reduction, D2HMemMove);
	    std::string res = USM_TEMPLATE::CodeGen_DAC2SYCL2(
		dacShellName,
		dacShellParams,
        opInit,
        ParameterGenerate,
		deviceMemAlloc,
		dac,
		MemFree);
        code += res;
        code += "\n\n";
        rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
        rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
        // std::cout << code;  
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(),code);
    //rewriter->InsertText(dacppFile->getMainFuncLoc()->getBeginLoc(), code);
}
