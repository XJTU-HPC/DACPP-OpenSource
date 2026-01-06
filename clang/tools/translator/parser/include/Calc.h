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
    int dimesion;//数据自身维度
    std::vector<int> dimid;//算子作用维度；
    std::vector<int> flag; //用来标记是否是降维算子；
};



class Expression;


/**
 * 存储计算结构信息
 */
class Calc {
private:
    std::string name; // 函数名
    std::vector<Param*> params; // 参数
    /*
        计算函数中可能也包含数据关联计算表达式
        存在数据关联表达式的嵌套这种情况需要以树的形式将其保存起来
    */
    std::vector<Expression*> exprs; // 数据关联计算表达式
    Expression* father; // 所属的数据关联计算表达式
    FunctionDecl* calcLoc; // AST中Calc节点的位置

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
    std::vector<std::string> body; // 函数体

    // 存储划分后的函数体
    std::vector<std::string> blocks;
};


}

#endif