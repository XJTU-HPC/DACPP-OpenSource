#ifndef TRANSLATOR_PARSER_SHELL_H
#define TRANSLATOR_PARSER_SHELL_H

#include <string>
#include <vector>

#include "clang/AST/AST.h"

#include "Param.h"
#include "Split.h"

using namespace clang;

namespace dacppTranslator {

class Expression;

typedef struct _tagBINDINFO
    {
    int icls;
    VNode *v;
    std::string offset;
    } 	BINDINFO;

class Shell {

private:
    std::string name;
    std::vector<Param*> params;
    std::vector<Split*> splits;
    std::vector<ShellParam*> shellParams;
    Expression* father;
    FunctionDecl* shellLoc;

public:
    Shell();
    ~Shell();

    void setName(std::string name);
    std::string getName();

    void setParam(Param* param);
    Param* getParam(int idx);
    int getNumParams();

    void setSplit(Split* split);
    Split* getSplit(int idx);
    int getNumSplits();

    void setShellParam(ShellParam* param);
    ShellParam* getShellParam(int idx);
    int getNumShellParams();

    void setFather(Expression* expr);
    Expression* getFather();

    void setShellLoc(FunctionDecl* expr);
    FunctionDecl* getShellLoc();

    void parseShell(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes);
    void GetBindInfo(std::vector<BINDINFO> *pbindInfo) ;
    dacppTranslator::Split *search_symbol(VNode *v);

    ALGraph *G;
};

}

#endif