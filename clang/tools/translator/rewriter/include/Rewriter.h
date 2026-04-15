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

    if (!mainFunc || !mainBody) {
        llvm::errs() << "rewriteMain: main function not captured.\n";
        return;
    }

    const LangOptions& LO = rewriter->getLangOpts();

    if (dacppFile->hasBufferRegionPlan()) {
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
    // 3. 有 forStatement → 保留原循环结构，仅替换其中的 <-> 表达式
    //===========================
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
