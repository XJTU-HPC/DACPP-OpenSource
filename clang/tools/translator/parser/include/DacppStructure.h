#ifndef TRANSLATOR_PARSER_DACPPSTRUCTURE_H
#define TRANSLATOR_PARSER_DACPPSTRUCTURE_H


#include <string>
#include <vector>

#include "clang/AST/AST.h"

#include "Split.h"
#include "Param.h"
#include "Shell.h"
#include "Calc.h"


using namespace clang;


namespace dacppTranslator {


/**
 * 存储头文件信息
 */
class HeaderFile {
private:
    std::string name; // 头文件名

public:
    HeaderFile();
    HeaderFile(std::string name);

    void setName(std::string name);
    std::string getName();
};


/**
 * 存储命名空间信息
 */
class NameSpace {
private:
    std::string name; // 命名空间名

public:
    NameSpace();
    NameSpace(std::string name);

    void setName(std::string name);
    std::string getName();
};


/**
 * 存储数据关联计算表达式信息
 */
class Expression {
private:
    Shell* shell; // 数据关联表达式对应的shell
    Calc* calc; // 数据关联计算表达式对应的calc
    const clang::BinaryOperator* dacExpr; // AST中数据关联计算表达式节点的位置

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


/**
 * 存储DACPP文件信息
 */
class DacppFile {
public:
    std::vector<const clang::BinaryOperator*> dacExprs;
private:
    std::vector<HeaderFile*> headerFiles; // 头文件
    std::vector<NameSpace*> nameSpaces; // 命名空间
    std::vector<Expression*> exprs; // 数据关联计算表达式
    const FunctionDecl* mainFuncLoc; // AST中主函数节点位置
    clang::TranslationUnitDecl* decl;


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
};


}

#endif