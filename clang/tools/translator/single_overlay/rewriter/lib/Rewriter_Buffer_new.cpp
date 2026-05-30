#include <set>
#include <iostream>
#include <sstream>
#include "clang/Rewrite/Core/Rewriter.h"
#include "Rewriter.h"
#include "Split.h"
#include "Param.h"
#include "dacInfo.h"
#include "buffer_template_new.h"
#include "Calc.h"
#include "ASTParse.h"
std::vector<int> dim;
std::vector<dacppTranslator::clacparam> clacparams;

// Emit the buffer-mode wrapper and generated helper code for each DACPP expression.
void dacppTranslator::Rewriter::rewriteDac_Buffer() {

    std::string code = "";
    for(int i = 0; i < dacppFile->getNumHeaderFile(); i++) {
        code += "#include " + dacppFile->getHeaderFile(i)->getName() + "\n";
    }
    code += "\n";
    for(int i = 0; i < dacppFile->getNumNameSpace(); i++) {
        code += "using namespace " + dacppFile->getNameSpace(i)->getName() + ";\n";
    }
    code += "\n";
    for(int i = 0; i < dacppFile->getNumExpression(); i++) {
        Expression* expr = dacppFile->getExpression(i);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        

        std::vector<BINDINFO> info;
        shell->GetBindInfo(&info);

        // Group bindings by connected component and record the relative offsets.
        Dac_Ops ops;
        std::vector<std::string> sets;
        std::vector<std::string> bindoffset;
        for(int i = 0; i < info.size(); i++){
            if(info[i].offset.empty())
                info[i].offset = "0"; 
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
                tmp.SetSplitSize(r->getSplitNumber());
                tmp.setDimId(r->getDimIdx());
                ops.push_back(tmp);
                sets.push_back("id"+std::to_string(info[i].icls));
                bindoffset.push_back(info[i].offset);
            }
        }
        code += "void " + calc->getName() + "(";
        for(int count = 0; count < calc->getNumParams(); count++) {
            if(shell->getShellParam(count)->getRw()!=IOTYPE::READ)code += calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName() + ",";
            else code += "const " + calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName() + ",";
        }
          for(int count = 0;count < shell->getNumShellParams(); count ++){
            for(int i = 0;i < shell->getShellParam(count)->getDimension();i++){
               code += "int " + calc->getParam(count)->getName()+"_"+std::to_string(i)+",";
            }
        }
          for(int count = 0;count < shell->getNumShellParams(); count ++){
            for(int i = 0;i < shell->getShellParam(count)->getDimension();i++){
               code += "int " + calc->getParam(count)->getName()+"_"+std::to_string(i)+"_shape,";
            }
        }
        for(int count = 0; count < calc->getNumParams(); count++) {
            code += "sycl::accessor<int, 1, sycl::access::mode::read> info_" + calc->getParam(count)->getName() + "_acc";
            if(count != calc->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ") ";
    for (int i = 0; i < shell->getNumShellParams(); i++) {
        clacparam temp;
    temp.name = calc->getParam(i)->getName();
    temp.dimesion = shell->getShellParam(i)->getDimension();
    dim.push_back(shell->getShellParam(i)->getDimension());

    for (int count = 0; count < shell->getShellParam(i)->getNumSplit(); count++) {
        Split *a = shell->getShellParam(i)->getSplit(count);
        temp.dimid.push_back(a->getDimIdx());

        if (a->type == "IndexSplit") {
            temp.flag.push_back(1);
        } else {
            temp.flag.push_back(0);
        }
    }

    clacparams.push_back(temp);
}
        for(int count = 0; count < calc->getNumBody(); count++) {
            
            code += calc->getBody(count,clacparams) + "\n";
        }

        std::string dacShellName = shell->getName() + "_" + calc->getName();
        std::string dacShellParams = "";
for (int count = 0; count < shell->getNumParams(); count++) {
    Param* param = shell->getParam(count);
    dacShellParams += param->getType() + " " + param->getName();
    if (count != shell->getNumParams() - 1) {
        dacShellParams += ", ";
    }
}
if(dacppFile->mode!=1){
bool needComma = (shell->getNumParams() > 0);
auto Vars=dacppFile->getForStatementVars();
for (const auto &var : Vars) {
    const std::string &type = var.second;
    const std::string &name = var.first;
    bool inShellVars = false;
    for (const auto &shellVar : dacppFile->shellVars) {
        if (shellVar.first == name) {
            inShellVars = true;
            break;
        }
    }
    if (inShellVars) continue;

    if (needComma) {
        dacShellParams += ", ";
    }

    dacShellParams += type + " " + name;
    needComma = true;
}
}
        std::string opInit = "";
        std::string infoInit = "";
        std::set<std::string> HadInit;
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            infoInit += BUFFER_TEMPLATE::CodeGen_DataInfoInit(shellParam->getName(), std::to_string(shellParam->getDimension()));
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                if(split->type.compare("IndexSplit") == 0){
                    if(HadInit.count(split->getId()) == 1){
                        continue;
                    }
                    IndexSplit* indexSplit = static_cast<IndexSplit*>(split);
                    opInit += BUFFER_TEMPLATE::CodeGen_IndexInit2(indexSplit->getId(),std::to_string(NumSplit),"info_"+shellParam->getName());
                    HadInit.insert(indexSplit->getId());

                }
                else if(split->type.compare("RegularSplit") == 0){
                    if(HadInit.count(split->getId()) == 1){
                        continue;
                    }
                    RegularSplit* regularSplit = static_cast<RegularSplit*>(split);
                    opInit += BUFFER_TEMPLATE::CodeGen_RegularSliceInit2(regularSplit->getId(),std::to_string(regularSplit->getSplitSize()),
                    std::to_string(regularSplit->getSplitStride()),std::to_string(NumSplit),"info_"+shellParam->getName());
                    HadInit.insert(regularSplit->getId());
                }
            }
        }
        opInit = infoInit + opInit;
        std::string add2Op = "";
        std::string dataOpsInit = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                add2Op += BUFFER_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(NumSplit),shellParam->getName()+"_Ops");
            }
            dataOpsInit += BUFFER_TEMPLATE::CodeGen_DataOpsInit2(shellParam->getName()+"_Ops",add2Op);
            add2Op = "";
        }
        std::string add2Op_inops = "";
        std::string add2Op_outops = "";
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
                if(shellParam->getRw() == IOTYPE:: WRITE){
                    for(int i = 0; i < info.size(); i++){
                        if(shell->search_symbol(info[i].v)->getId() == split->getId())
                                set = std::to_string(info[i].icls);
                    }
                    if(setOut.count(set) == 1) {
                        continue;
                    }
                    add2Op_outops += BUFFER_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(outflag),"Out_Ops");
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
                    add2Op_inops += BUFFER_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(inflag),"In_Ops");
                    Dac_Op op = Dac_Op(split->getId(),0,inflag);
                    inflag++;
                    Inops.push_back(op);
                    setIn.insert(set);
                }
            }
        }
        std::string dataOpsInit_inops = BUFFER_TEMPLATE::CodeGen_DataOpsInit2("In_Ops",add2Op_inops);
        std::string dataOpsInit_outops = BUFFER_TEMPLATE::CodeGen_DataOpsInit2("Out_Ops",add2Op_outops);
        std::string divice_memory = "";
        std::string splitLength = "";
        std::string AddDacOps2Vector = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(shellParam->getRw() != IOTYPE::READ){
                AddDacOps2Vector += BUFFER_TEMPLATE::CodeGen_Add_DacOps2Vector("ops_s","In_Ops");
            }
            else{
                AddDacOps2Vector += BUFFER_TEMPLATE::CodeGen_Add_DacOps2Vector("ops_s",shellParam->getName()+"_Ops");
            }
        }
        std::string DeclareDacOpsVector = BUFFER_TEMPLATE::CodeGen_Declare_DacOps_Vector("ops_s",AddDacOps2Vector);
        std::string item_number = BUFFER_TEMPLATE::CodeGen_Init_Work_Item_Number("Item_Size","In_Ops");
        std::string InitOPS =  dataOpsInit + dataOpsInit_inops + dataOpsInit_outops;
        std::string ParameterGenerate = BUFFER_TEMPLATE::CodeGen_ParameterGenerate(InitOPS,divice_memory,splitLength,item_number); 
        std::string deviceMemAlloc = "";
        std::string H2DMemMove = "";

       
        std::string BindingInit = BUFFER_TEMPLATE::CodeGen_IndexInit2(ops,sets,bindoffset);
        Args args = Args();
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            Dac_Ops ops;
            if(shellParam->getRw()==IOTYPE::READ){
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
        std::vector<std::string> splits;
        std::vector<int> splitNum;
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            splitNum.push_back(shellParam->getNumSplit());
            for (int y = 0; y < shellParam->getNumSplit(); y++) {
                splits.push_back(std::to_string(y));
            }
        }
        std::vector<std::string> accessor_names;
        for (int z = 0; z < shell->getNumParams(); z++) {
            accessor_names.push_back(shell->getParam(z)->getName());
        }
        std::string CalcEmbed = BUFFER_TEMPLATE::CodeGen_CalcEmbed2(calc->getName(), splits, splitNum, accessor_names);

        
        std::string AccessorInit = "";
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            AccessorInit += BUFFER_TEMPLATE::CodeGen_AccessorInit(shell->getShellParam(argCount)->getName());
        }
        std::string Accessor_List = "";
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            if(shell->getShellParam(argCount)->getRw() == IOTYPE::READ){
                Accessor_List += BUFFER_TEMPLATE::CodeGen_AccessorInit0_read(shell->getShellParam(argCount)->getName(), shell->getShellParam(argCount)->getBasicType());
            }
            else if(shell->getShellParam(argCount)->getRw() == IOTYPE::WRITE){
                Accessor_List += BUFFER_TEMPLATE::CodeGen_AccessorInit0_write(shell->getShellParam(argCount)->getName(), shell->getShellParam(argCount)->getBasicType());
            }
            else if(shell->getShellParam(argCount)->getRw() == IOTYPE::READ_WRITE){
                Accessor_List += BUFFER_TEMPLATE::CodeGen_AccessorInit0_read_write(shell->getShellParam(argCount)->getName(), shell->getShellParam(argCount)->getBasicType());
            }
        }
        std::string Accessor_Pointer_List = "";
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            Accessor_Pointer_List += BUFFER_TEMPLATE::CodeGen_AccessorInit1(shell->getShellParam(argCount)->getName());
        }
        std::string getpos = "";
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            ShellParam* shellParam = shell->getShellParam(argCount);
             for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                Split* split = shellParam->getSplit(NumSplit);
                if(split->type.compare("RegularSplit") == 0 || split->type.compare("IndexSplit") == 0){
                    getpos += BUFFER_TEMPLATE::CodeGen_getpos1(shell->getShellParam(argCount)->getName(), split->getId(), std::to_string(NumSplit));
                }
                else 
                    getpos += BUFFER_TEMPLATE::CodeGen_getpos0(shell->getShellParam(argCount)->getName(), std::to_string(NumSplit));
            }  
        }
	    std::string KernelExecute;
        if(dacppFile->getForStatementCtrl()&&dacppFile->mode==0){
            KernelExecute = BUFFER_TEMPLATE::CodeGen_KernelExecute2("Item_Size",AccessorInit,BindingInit,getpos,Accessor_List,Accessor_Pointer_List,CalcEmbed,dacppFile);
        }else{
            KernelExecute = BUFFER_TEMPLATE::CodeGen_KernelExecute("Item_Size",AccessorInit,BindingInit,getpos,Accessor_List,Accessor_Pointer_List,CalcEmbed);
        }

        std::string Reduction;
        std::string D2HMemMove;
        std::string ReductionRule = "sycl::plus<>()";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(shellParam->getRw() != IOTYPE::READ){
	            Reduction += BUFFER_TEMPLATE::CodeGen_Result_B2H_Mov(shellParam->getName(),shellParam->getName()+"_Size");
            }
        }
       
        std::string dataRecon = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(shellParam->getRw() == IOTYPE::WRITE){
                dataRecon += BUFFER_TEMPLATE::CodeGen_Init_Host_Memory(shellParam->getBasicType(),shellParam->getName());
            }else if(shellParam->getRw() == IOTYPE::READ){
                dataRecon += BUFFER_TEMPLATE::CodeGen_D2B_Mov_Buffer(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");}
            else if(shellParam->getRw() == IOTYPE::READ_WRITE){
                 dataRecon += BUFFER_TEMPLATE::CodeGen_D2B_Mov_Buffer_read_write(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");}
            }
       for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            std::string opPushBack = "";
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                opPushBack += BUFFER_TEMPLATE::CodeGen_OpPushBack2Ops(shellParam->getName(),split->getId(),std::to_string(split->getDimIdx()));
            }
            std::string dataOpsInit = BUFFER_TEMPLATE::CodeGen_DataOpsInit(shellParam->getName(),opPushBack);
            if(shellParam->getRw() != IOTYPE::READ){
                dataRecon += BUFFER_TEMPLATE::CodeGen_DataReconstruct1(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size",dataOpsInit);
            }
            else{
                dataRecon += BUFFER_TEMPLATE::CodeGen_DataReconstruct(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size",dataOpsInit);
            }
       }

        std::string dac = BUFFER_TEMPLATE::CodeGen_DataAssocComp(dataRecon, H2DMemMove, KernelExecute, Reduction, D2HMemMove);
	    std::string res = BUFFER_TEMPLATE::CodeGen_DAC2SYCL2(
		dacShellName,
		dacShellParams,
        opInit,
        ParameterGenerate,
		deviceMemAlloc,
		dac);
        code += res;
        code += "\n\n";
        rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
        rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(),code);
}
