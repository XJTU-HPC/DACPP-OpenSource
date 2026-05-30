
#ifndef TRANSLATOR_PARSER_DACPPSTRUCTURE_H
#define TRANSLATOR_PARSER_DACPPSTRUCTURE_H

#include <string>
#include <vector>

#include "clang/AST/AST.h"
#include "Split.h"
#include "Param.h"
#include "Shell.h"
#include "Calc.h"
#include "Dacfor.h"

using namespace clang;

namespace dacppTranslator {

class HeaderFile {
private:
    std::string name;

public:
    HeaderFile();
    HeaderFile(std::string name);

    void setName(std::string name);
    std::string getName();
};

class NameSpace {
private:
    std::string name;

public:
    NameSpace();
    NameSpace(std::string name);
    void setName(std::string name);
    std::string getName();
};

class Expression {
private:
    Shell* shell;
    Calc* calc;
    const clang::BinaryOperator* dacExpr;

public:
    Expression();
    void setShell(Shell* shell);
    Shell* getShell();
    void setCalc(Calc* calc);
    Calc* getCalc();
    void setDacExpr(const clang::BinaryOperator* dacExpr);
    const clang::BinaryOperator* getDacExpr();
    static bool shellLHS_p(const BinaryOperator *dacExpr);
};

// Top-level parse result for a DACPP translation unit.
class DacppFile {
public:
    std::vector<const clang::BinaryOperator*> dacExprs;
    std::vector<HeaderFile*> headerFiles;
    std::vector<NameSpace*> nameSpaces;

    std::vector<Expression*> exprs;
    const FunctionDecl* mainFuncLoc;

    clang::TranslationUnitDecl* decl;
    ControlBlock* block;
    const CallExpr* FS;

    clang::ASTContext *Context = nullptr;

    const clang::ForStmt* forStatement;
    bool forStatementCtrl = false;
    std::vector<std::pair<std::string, std::string>> forStatementVars;

    const clang::FunctionDecl* mainFunctionDecl = nullptr;
    const clang::Stmt* mainStmt = nullptr;

    std::vector<const clang::ForStmt*> innerForStatements;
    std::vector<std::pair<std::string, std::string>> shellVars;
    int mode = 0;
public:
    const FunctionDecl* node;
    DacppFile();

    void setTranslationUnitDecl(clang::TranslationUnitDecl* decl) {
        this->decl = decl;
    }

    clang::TranslationUnitDecl* getTranslationUnitDecl() {
        return decl;
    }

    void setHeaderFile(std::string headerfile);
    HeaderFile* getHeaderFile(int idx);
    int getNumHeaderFile();

    void setNameSpace(std::string nameSpace);
    NameSpace* getNameSpace(int idx);
    int getNumNameSpace();

    void setExpression(const BinaryOperator* dacExpr);
    Expression* getExpression(int idx);
    int getNumExpression();

    void setMainFuncLoc(const FunctionDecl* mainFuncLoc);
    const FunctionDecl* getMainFuncLoc();

    void setBlock(ControlBlock* block) {
        this->block = block;
    }
    ControlBlock* getBlock() {
        return block;
    }
    void setForStmt(const CallExpr* FS) {
        this->FS = FS;
    }
    const CallExpr* getForStmt() {
        return FS;
    }

    void setForStatement(const clang::ForStmt* FS) {
        this->forStatement = FS;
        this->forStatementCtrl = true;
    }
    const clang::ForStmt* getForStatement() {
        return forStatement;
    }
    bool getForStatementCtrl() {
        return forStatementCtrl;
    }
    void collectVarsFromForStatement();
    std::vector<std::pair<std::string, std::string>> getForStatementVars();
    std::vector<std::pair<std::string, std::string>> getShellVars(){
    std::vector<std::pair<std::string, std::string>> result = shellVars;
    for (auto &var : result) {
        if (var.second == "_Bool") {
            var.second = "bool";
        }
    }
    return result;
}
    void setContext(clang::ASTContext* ctx) { this->Context = ctx; }
    clang::ASTContext* getContext() { return this->Context; }
    void setMainFunction(const clang::FunctionDecl* func) { this->mainFunctionDecl = func; }
    const clang::FunctionDecl* getMainFunction() { return this->mainFunctionDecl; }
    void setMainBody(const clang::Stmt* body) { this->mainStmt = body; }
    const clang::Stmt* getMainBody() { return this->mainStmt; }
    void collectInnerForStmts();

};
}
#endif
