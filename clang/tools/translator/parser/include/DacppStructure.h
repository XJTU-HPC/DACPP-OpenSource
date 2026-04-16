//定义dacppfile的文件
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

struct BufferRegionPlan {
    bool enabled = false;
    int exprIndex = -1;
    const clang::FunctionDecl* parentFunction = nullptr;
    const clang::ForStmt* outerFor = nullptr;
    const clang::BinaryOperator* dacExpr = nullptr;
    std::vector<const clang::Stmt*> siblingStmts;
    std::vector<std::pair<std::string, std::string>> capturedVars;
    std::vector<std::pair<std::string, std::string>> capturedNonShellVars;
    std::string disableReason;
};


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
    std::vector<HeaderFile*> headerFiles; // 头文件
    std::vector<NameSpace*> nameSpaces; // 命名空间
    
    std::vector<Expression*> exprs; // 数据关联计算表达式
    const FunctionDecl* mainFuncLoc; // AST中主函数节点位置
    /****************dacfor相关*****************/
    clang::TranslationUnitDecl* decl;
    ControlBlock* block;//用于存dacfor
    const CallExpr* FS;// SourceRange forRange;
    /*2025.12.1新增↓*/
    clang::ASTContext *Context = nullptr;//用于存储本代码的ASTContext
    /****************for循环相关*****************/
    //这里的for循环是指包含了表达式的for循环 形如for(...) {shell<->clac;}
    const clang::ForStmt* forStatement;  // 用于存储包裹了表达式的循环语句
    bool forStatementCtrl = false; // 用于标记是否检测到了包裹表达式的循环语句
    std::vector<std::pair<std::string, std::string>> forStatementVars; // 记录前文提及的for循环中用到的、但是在for循环外声明的变量的   变量名及其类型,第一个表示变量名，第二个表示变量类型
    /****************main函数相关*****************/
    const clang::FunctionDecl* mainFunctionDecl = nullptr; // main 函数声明
    const clang::Stmt* mainStmt = nullptr;             // main 函数体
    std::vector<std::pair<std::string, std::string>> shellVars; // shell参数的变量名及其类型,第一个表示变量名，第二个表示变量类型
    int mode = 0; //用于存储翻译模式,0表示使用新版本，1表示强制用老版本
    bool enableMPI = false;
    bool mpiBroadcastOutputs = true;
    bool mainAlreadyRewritten = false;
    BufferRegionPlan bufferRegionPlan;
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
    // void setForRange(SourceRange range) { forRange = range; }
    // SourceRange getForRange() const { return forRange; }
    /*2025.12.01新增*/
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
    void collectVarsFromForStatement();//找出从for语句中提取的在for循环外定义并在for循环内使用的变量名及其类型
    std::vector<std::pair<std::string, std::string>> getForStatementVars();//返回从for语句中提取的在for循环外定义并在for循环内使用的变量名及其类型
    std::vector<std::pair<std::string, std::string>> getShellVars(){    
    std::vector<std::pair<std::string, std::string>> result = shellVars;
    for (auto &var : result) {
        if (var.second == "_Bool") {
            var.second = "bool";
        }
    }
    return result;
}//返回从for语句中提取的在for循环外定义并在for循环内使用的变量名及其类型
    void setContext(clang::ASTContext* ctx) { this->Context = ctx; }//设置上下文
    clang::ASTContext* getContext() { return this->Context; }//获取上下文
    void setMainFunction(const clang::FunctionDecl* func) { this->mainFunctionDecl = func; }//设置主函数decl
    const clang::FunctionDecl* getMainFunction() { return this->mainFunctionDecl; }//获取主函数decl
    void setMainBody(const clang::Stmt* body) { this->mainStmt = body; }//设置主函数体
    const clang::Stmt* getMainBody() { return this->mainStmt; }//获取主函数体
    void analyzeBufferRegionPlan();
    const BufferRegionPlan& getBufferRegionPlan() const { return this->bufferRegionPlan; }
    bool hasBufferRegionPlan() const { return this->bufferRegionPlan.enabled; }
    void setEnableMPI(bool enabled) { this->enableMPI = enabled; }
    bool getEnableMPI() const { return this->enableMPI; }
    void setMPIBroadcastOutputs(bool enabled) { this->mpiBroadcastOutputs = enabled; }
    bool getMPIBroadcastOutputs() const { return this->mpiBroadcastOutputs; }
    void setMainAlreadyRewritten(bool rewritten) { this->mainAlreadyRewritten = rewritten; }
    bool getMainAlreadyRewritten() const { return this->mainAlreadyRewritten; }

};
}
#endif
