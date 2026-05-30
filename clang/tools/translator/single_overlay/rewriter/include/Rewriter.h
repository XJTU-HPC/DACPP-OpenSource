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
    // Rewrite main either with direct <-> replacement or loop-aware injection.
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
    std::string mainOriginal =
        Lexer::getSourceText(
            CharSourceRange::getTokenRange(mainBody->getSourceRange()),
            SM, LO
        ).str();

    if (mainOriginal.empty()) {
        llvm::errs() << "rewriteMain: main body source empty.\n";
        return;
    }
    if (!dacppFile->getForStatementCtrl()||dacppFile->mode==1) {
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
    rewriter->RemoveText(targetFor->getSourceRange());
std::string dacExprCode = "";
if (dacppFile->dacExprs.size() > 0) {
    const BinaryOperator* dacExpr = dacppFile->dacExprs[0];
    CallExpr* shellCall = dacppTranslator::getNode<CallExpr>(dacExpr->getLHS());

    llvm::raw_string_ostream rso(dacExprCode);
    clang::PrintingPolicy policy(LO);
    shellCall->printPretty(rso, nullptr, policy);
    std::string oldName = shellCall->getDirectCallee()->getNameAsString();
    std::string newName;
    if (isa<DeclRefExpr>(dacExpr->getRHS())) {
        newName = oldName + "_" + dyn_cast<DeclRefExpr>(dacExpr->getRHS())->getDecl()->getNameAsString();
    } else {
        DeclRefExpr* declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());
        newName = oldName + "_" + declRefExpr->getDecl()->getNameAsString();
    }
    size_t pos = dacExprCode.find(oldName);
    if (pos != std::string::npos) {
        dacExprCode.replace(pos, oldName.size(), newName);
    }
auto vars = dacppFile->getForStatementVars();
auto shellVars = dacppFile->getShellVars();

if (!vars.empty()) {
    std::string paramList;
    for (const auto& var : vars) {
        const std::string& varName = var.first;
        bool isShellVar = false;
        for (const auto& shellVar : shellVars) {
            if (shellVar.first == varName) {
                isShellVar = true;
                break;
            }
        }
        if (isShellVar) continue;
        if (!paramList.empty()) paramList += ", ";
        paramList += varName;
    }
    size_t callEnd = dacExprCode.find(")");
    if (callEnd != std::string::npos && !paramList.empty()) {
        dacExprCode.insert(callEnd, ", " + paramList);
    }
}

}
std::string newCode = dacExprCode + ";\n";
rewriter->InsertText(targetFor->getBeginLoc(), newCode, true, true);

}


    void rewriteMPI();
};
}

# endif
