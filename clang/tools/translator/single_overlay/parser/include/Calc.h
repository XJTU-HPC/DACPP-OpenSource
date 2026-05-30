#ifndef TRANSLATOR_PARSER_CALC_H
#define TRANSLATOR_PARSER_CALC_H

#include <string>
#include <vector>

#include "clang/AST/AST.h"

#include "Split.h"
#include "Param.h"

using namespace clang;

namespace dacppTranslator {
struct clacparam{
    std::string name;
    int dimesion;
    std::vector<int> dimid;
    std::vector<int> flag;
};

class Expression;

class Calc {
private:
    std::string name;
    std::vector<Param*> params;

    std::vector<Expression*> exprs;
    Expression* father;
    FunctionDecl* calcLoc;

public:
    Calc();

    void setName(std::string name);
    std::string getName();

    void setParam(Param* param);
    Param* getParam(int idx);
    int getNumParams();

    void setBody(Stmt* body);
    std::string getBody(int idx);
    std::string getBody(int idx,std::vector<dacppTranslator::clacparam>& dim);
    std::string dacfor_getBody(int idx);
    int getNumBody();

    void setExpr(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes);
    Expression* getExpr(int idx);
    int getNumExprs();

    void setFather(Expression* expr);
    Expression* getFather();

    void setCalcLoc(FunctionDecl* calcLoc);
    FunctionDecl* getCalcLoc();

    void parseCalc(const BinaryOperator* dacExpr);
    std::vector<std::string> body;

    std::vector<std::string> blocks;
};

}

#endif