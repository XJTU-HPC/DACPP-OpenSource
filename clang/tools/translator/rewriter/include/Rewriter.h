//定义了rewriter类
#ifndef TRANSLATOR_REWRITER_REWRITER_H
#define TRANSLATOR_REWRITER_REWRITER_H

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "DacppStructure.h"
#include "Param.h"
#include "dacInfo.h"
#include <set>
#include <iostream>
#include <sstream>
#include "Split.h"
#include "ASTParse.h"
#include <string>
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceLocation.h"

namespace dacppTranslator {

class Rewriter {
private:
    clang::Rewriter* rewriter;
    DacppFile* dacppFile;
    std::string generateCalc(std::string code, Expression* expr) {
        Calc* calc = expr->getCalc();
        // 计算结构
        code += "void " + calc->getName() + "(";
        for(int paramCount = 0; paramCount < calc->getNumParams(); paramCount++) {
            Param* param = calc->getParam(paramCount);
            code += param->getBasicType() + "* " + param->getName();
            if(paramCount != calc->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ") {\n";
        int exprCount = 0;
        for(int bodyCount = 0; bodyCount < calc->getNumBody(); bodyCount++) {
            if (calc->getBody(bodyCount).compare("Expression") != 0) {
                code += "   " + calc->getBody(bodyCount) + "\n";
                continue;
            }
            code += "}\n\n";
            generateCalc(code, calc->getExpr(exprCount++));
            if (bodyCount + 1 == calc->getNumBody()) {
                continue;
            }
            code += "void " + calc->getName() + "(";
            for(int paramCount = 0; paramCount < calc->getNumParams(); paramCount++) {
                Param* param = calc->getParam(paramCount);
                code += param->getBasicType() + "* " + param->getName();
                if(paramCount != calc->getNumParams() - 1) {
                    code += ", ";
                }
            }
            code += ") {\n";
        }
        code += "}\n\n";
        return code;
    }
 
public:
    void setRewriter(clang::Rewriter* rewriter) {
        this->rewriter = rewriter;
    }

    void setDacppFile(DacppFile* dacppFile) {
        this->dacppFile = dacppFile;
    }

    std::string generateCalc(Calc* calc);

    void addSplit(std::vector<std::vector<int>>& shapes, std::vector<std::vector<Split*>>& splits,
                  Expression* expr);
    
    std::string generateChildExpr(Expression* ancestor, Expression* expr, Dac_Ops& index, std::vector<Dac_Ops>& tensorOps,
                                  int* mem);
    
    std::string generateChildExpr2(Expression* ancestor, Expression* expr, Dac_Ops& index, std::vector<Dac_Ops>& tensorOps,
                                  int* mem);
    
    std::string generateSyclFunc(Expression* expr);


    void rewriteDac_Usm();
    void rewriteDac_Usm_time();

    void rewriteDac_Buffer();

    void rewriteDac_Multiple();
/*2025.12.2重写，仅支持一个“<->”的情况，多<->有待扩充*/
    void rewriteMain() {

    const FunctionDecl* mainFunc = dacppFile->getMainFunction();
    const Stmt* mainBody = dacppFile->getMainBody();
    clang::ASTContext* ctx = dacppFile->getContext();
    const ForStmt* targetFor = dacppFile->getForStatement();

    if (!mainFunc || !mainBody) {
        llvm::errs() << "rewriteMain: main function not captured.\n";
        return;
    }

    const SourceManager& SM = rewriter->getSourceMgr();
    const LangOptions& LO = rewriter->getLangOpts();

    //===========================
    // 1. 获取 main 原始代码
    //===========================
    std::string mainOriginal =
        Lexer::getSourceText(
            CharSourceRange::getTokenRange(mainBody->getSourceRange()),
            SM, LO
        ).str();

    if (mainOriginal.empty()) {
        llvm::errs() << "rewriteMain: main body source empty.\n";
        return;
    }

    //===========================
    // 2. 如果没有 forStatement，保持原逻辑即可
    //===========================
    if (!dacppFile->getForStatementCtrl()||dacppFile->mode==1) {
        // 执行原来的替换 <-> 的逻辑
        for (int exprCount = 0; exprCount < dacppFile->dacExprs.size(); exprCount++) {
            const BinaryOperator* dacExpr = dacppFile->dacExprs[exprCount];

            CallExpr* shellCall = dacppTranslator::getNode<CallExpr>(dacExpr->getLHS());
            DeclRefExpr* declRefExpr = nullptr;
            
            if (isa<DeclRefExpr>(dacExpr->getRHS()))
                declRefExpr = dyn_cast<DeclRefExpr>(dacExpr->getRHS());
            else
                declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());

            llvm::raw_string_ostream rso(*(new std::string()));
            clang::PrintingPolicy policy(LO);
            shellCall->printPretty(rso, nullptr, policy);
            std::string code = rso.str();

            code.replace(
                code.find(shellCall->getDirectCallee()->getNameAsString()),
                shellCall->getDirectCallee()->getNameAsString().size(),
                shellCall->getDirectCallee()->getNameAsString() + "_" + declRefExpr->getDecl()->getNameAsString()
            );

            rewriter->ReplaceText(dacExpr->getSourceRange(), code);
        }
        return;
    }

    //===========================
    // 3. 有 forStatement → 需要执行新的替换流程
    //===========================

    //-----------------------------
    // 3.1 把 for 循环全文 text 删除
    //-----------------------------
    rewriter->RemoveText(targetFor->getSourceRange());

    //-----------------------------
    // 3.2 生成 closureInit 代码
    //-----------------------------
    // std::string closureInit;
    // closureInit += "{{closureInit}}\n";

    // 如用户定义的 closure 结构体在别处生成，这里只插入初始化

    //-----------------------------
    // 3.3 获取 <-> 那一句
    //-----------------------------
    // std::string dacExprCode = "";
    // if (dacppFile->dacExprs.size() > 0) {
    //     const BinaryOperator* dacExpr = dacppFile->dacExprs[0];
    //     CallExpr* shellCall = dacppTranslator::getNode<CallExpr>(dacExpr->getLHS());

    //     DeclRefExpr* declRefExpr = nullptr;
    //     if (isa<DeclRefExpr>(dacExpr->getRHS()))
    //         declRefExpr = dyn_cast<DeclRefExpr>(dacExpr->getRHS());
    //     else
    //         declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());

    //     llvm::raw_string_ostream rso(dacExprCode);
    //     clang::PrintingPolicy policy(LO);
    //     shellCall->printPretty(rso, nullptr, policy);

    //     // 改 shell 为 shell_calc
    //     std::string oldName = shellCall->getDirectCallee()->getNameAsString();
    //     std::string newName = oldName + "_" + declRefExpr->getDecl()->getNameAsString();
    //     size_t pos = dacExprCode.find(oldName);
    //     if (pos != std::string::npos) {
    //         dacExprCode.replace(pos, oldName.size(), newName);
    //     }
    // }

    //-----------------------------
    // 3.4 在 for 循环原位置插入新代码
    //-----------------------------
    // std::string newCode;
    // newCode += closureInit;
    // newCode += dacExprCode + ";\n";
    // newCode += "{{closureWriteback}}\n";
    // auto vars = dacppFile->getForStatementVars();
    // std::string closureInitCode = generateClosureInit(vars);
    // std::string closureWritebackCode = generateClosureWriteback(vars);
    // size_t tagPos = newCode.find("{{closureInit}}");
    // if (tagPos != std::string::npos) {
    //     newCode.replace(tagPos, strlen("{{closureInit}}"), closureInitCode);
    // }
    // tagPos = newCode.find("{{closureWriteback}}");
    // if (tagPos != std::string::npos) {
    //     newCode.replace(tagPos, strlen("{{closureWriteback}}"), closureWritebackCode);
    // }
    // rewriter->InsertText(targetFor->getBeginLoc(), newCode, true, true);
    //===========================
    // 3.5 main 其他代码保持不动
    //===========================

        //=============================
// 3.3 获取 <-> 那一句，并生成带所有 forStatementVars 的函数调用
//=============================
std::string dacExprCode = "";
if (dacppFile->dacExprs.size() > 0) {
    const BinaryOperator* dacExpr = dacppFile->dacExprs[0];
    CallExpr* shellCall = dacppTranslator::getNode<CallExpr>(dacExpr->getLHS());

    llvm::raw_string_ostream rso(dacExprCode);
    clang::PrintingPolicy policy(LO);
    shellCall->printPretty(rso, nullptr, policy);

    // 构造新的函数名（_calc 或者原来的逻辑）
    std::string oldName = shellCall->getDirectCallee()->getNameAsString();
    std::string newName;
    if (isa<DeclRefExpr>(dacExpr->getRHS())) {
        newName = oldName + "_" + dyn_cast<DeclRefExpr>(dacExpr->getRHS())->getDecl()->getNameAsString();
    } else {
        DeclRefExpr* declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());
        newName = oldName + "_" + declRefExpr->getDecl()->getNameAsString();
    }

    // 替换原函数名
    size_t pos = dacExprCode.find(oldName);
    if (pos != std::string::npos) {
        dacExprCode.replace(pos, oldName.size(), newName);
    }

    //=============================
    // 将 forStatementVars 作为参数添加
    //=============================
auto vars = dacppFile->getForStatementVars(); // vector<pair<string,type>>
auto shellVars = dacppFile->getShellVars();   // vector<pair<string,type>>

if (!vars.empty()) {
    std::string paramList;
    for (const auto& var : vars) {
        const std::string& varName = var.first;

        // 检查 varName 是否在 shellVars 里
        bool isShellVar = false;
        for (const auto& shellVar : shellVars) {
            if (shellVar.first == varName) {
                isShellVar = true;
                break;
            }
        }
        if (isShellVar) continue; // 跳过 shellVar

        // 加入 paramList
        if (!paramList.empty()) paramList += ", ";
        paramList += varName;
    }

    // 在函数调用末尾插入参数
    size_t callEnd = dacExprCode.find(")");
    if (callEnd != std::string::npos && !paramList.empty()) {
        dacExprCode.insert(callEnd, ", " + paramList);
    }
}

}

//=============================
// 3.4 在 for 循环原位置插入新代码
//=============================
std::string newCode = dacExprCode + ";\n";
rewriter->InsertText(targetFor->getBeginLoc(), newCode, true, true);

}
// /*2025.12.2新增，用于辅助rewritemain，实现closure成员的赋值*/
// std::string generateClosureInit(const std::vector<std::pair<std::string, std::string>>& vars)
// {
//     std::string result;
//     result+= "Closure closure;\n";
//     for (const auto& p : vars) {
//         const std::string& varName = p.first;
//         const std::string& varType  = p.second;
//         if (varType.find("const") != std::string::npos)
//         continue;
//         // closure.var = var;
//         result += "closure." + varName + " = " + varName + ";\n";
//     }

//     return result;
// }
// std::string generateClosureWriteback(const std::vector<std::pair<std::string, std::string>>& vars)
// {
//     std::string result;
//     for (const auto& p : vars) {
//         const std::string& varName = p.first;
//         const std::string& varType  = p.second;

//         // 若类型中包含 const，则跳过
//         if (varType.find("const") != std::string::npos)
//             continue;
//         // closure.var = var;
//         result +=  varName + " = " + "closure." + varName+ ";\n";
//     }

//     return result;
// }
//     // 根据 forStatementVars 生成 Closure 结构体
// std::string CodeGen_ClosureStruct(
//     const std::vector<std::pair<std::string, std::string>>& vars)
// {
//     std::string code;
//     code += "typedef struct Closure {\n";

//     for (const auto& p : vars) {
//         const std::string& varName = p.first;
//         const std::string& typeName = p.second;

//         code += "    " + typeName + " " + varName + ";\n";
//     }

//     code += "} closure;\n\n";
//     return code;
// }


    void rewriteMPI();
};

// namespace dacppTranslator
}

# endif