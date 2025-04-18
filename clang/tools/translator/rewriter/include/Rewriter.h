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
#include "sub_template.h"
#include "ASTParse.h"
#include <string>

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

    void rewriteDac_Buffer();

    void rewriteMain() {
for (int exprCount = 0; exprCount < dacppFile->dacExprs.size(); exprCount++) {
        const BinaryOperator* dacExpr = dacppFile->dacExprs[exprCount];
        CallExpr* shellCall = dacppTranslator::getNode<CallExpr>(dacExpr->getLHS());
        DeclRefExpr* declRefExpr;
        if(isa<DeclRefExpr>(dacExpr->getRHS())) {
            declRefExpr = dyn_cast<DeclRefExpr>(dacExpr->getRHS());
        }
        else {
            declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());
        }
        std::string str;
        llvm::raw_string_ostream rso(str);
        clang::LangOptions langOpts;
        langOpts.CPlusPlus = true; // 启用C++选项（根据需要配置）
        clang::PrintingPolicy policy(langOpts);
        shellCall->printPretty(rso, nullptr, policy);
        std::string code = rso.str();
        code.replace(code.find(shellCall->getDirectCallee()->getNameAsString()), 
        shellCall->getDirectCallee()->getNameAsString().size(), shellCall->getDirectCallee()->getNameAsString()
         + "_" + declRefExpr->getDecl()->getNameAsString());
        rewriter->ReplaceText(dacExpr->getSourceRange(), code);
    }
}
};

// namespace dacppTranslator
}

# endif