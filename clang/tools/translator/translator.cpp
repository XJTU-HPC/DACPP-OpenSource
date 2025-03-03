#include <string>
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include <unordered_map>
#include <set>

#include "Split.h"
#include "Param.h"
#include "DacppStructure.h"
#include "Rewriter.h"
#include "ASTParse.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;


dacppTranslator::DacppFile* dacppFile = new dacppTranslator::DacppFile();

std::unordered_map<FunctionDecl*, std::set<FunctionDecl*>> dacExprMap;

/*
  ASTMatcher 匹配 DACPP 文件中符合要求的节点
*/
class DacHandler : public MatchFinder::MatchCallback {
  
public:
    DacHandler() {}

    virtual void run(const MatchFinder::MatchResult &Result) override {
        // 匹配数据关联计算表达式
        if (const BinaryOperator* dacExpr = Result.Nodes.getNodeAs<clang::BinaryOperator>("dac_expr")) {
            dacppFile->dacExprs.push_back(dacExpr);
            /*
                对匹配到的数据关联计算表达式进行过滤，只解析顶级数据关联计算表达式
                解析数据关联计算表达式时需要知道数据的维度，将其硬编码到生成的SYCL文件中
                而子数据关联计算表达式在编译期无法从得到该信息
                顶级数据关联计算表达式在编译期可以找到数据的定义位置，从其构造函数中得到数据的维度信息
            */
            if (dyn_cast<BinaryOperator>(dacExpr->getLHS()) ||
                dyn_cast<BinaryOperator>(dacExpr->getRHS())) {
              llvm::errs() << "error: multi binary operator in a dac statement"
                           << "\n";
              return;
            }
            // 获取 DAC 数据关联表达式左值
            Expr* dacExprLHS = dacppTranslator::Expression::shellLHS_p (dacExpr) ? dacExpr->getLHS() : dacExpr->getRHS();
            CallExpr* shellCall = dacppTranslator::getNode<CallExpr>(dacExprLHS);
            FunctionDecl* functionDecl = shellCall->getDirectCallee();
            Expr* curExpr = shellCall->getArg(0);
            DeclRefExpr* declRefExpr;
            if(isa<DeclRefExpr>(curExpr)) {
                declRefExpr = dyn_cast<DeclRefExpr>(curExpr);
            }
            else {
                declRefExpr = dacppTranslator::getNode<DeclRefExpr>(curExpr);
            }
            if(isa<ParmVarDecl>(declRefExpr->getDecl())) {
                return;
            }

            if(isa<DeclRefExpr>(dacExpr->getRHS())) {
                declRefExpr = dyn_cast<DeclRefExpr>(dacExpr->getRHS());
            }
            else {
                declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());
            }
            FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(declRefExpr->getDecl());
            if (dacExprMap.find(functionDecl) != dacExprMap.end()) {
                if (dacExprMap[functionDecl].count(calcFunc) == 1) {
                    return;
                }
            } else {
                dacExprMap.emplace(functionDecl, std::set<FunctionDecl*>());
            }
            dacExprMap[functionDecl].emplace(calcFunc);
            // 解析 DACPP 文件中的顶级数据关联计算表达式
            dacppFile->setExpression(dacExpr);
        }
        // 匹配主函数
        else if (const FunctionDecl* mainFunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("main")) {
            dacppFile->setMainFuncLoc(mainFunc);
        } else if (const FunctionDecl* mainFunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("dac_expr_father")) {
            dacppFile->node = mainFunc;
        }
  
    }

};

// ASTConsumer 是一个抽象接口，由需要遍历 AST 的用户来实现
// 源码文件解析得到的 Clang AST 会通过 ASTConsumer 回传给外部工具
// 可以重载两个函数：
// HandleTopLevelDecl()：解析顶级声明时被调用
// HandleTranslationUnit()：在整个文件都解析完后调用
// ASTConsumer 是一个用来在AST上写一些通用动作的接口
class MyASTConsumer : public ASTConsumer {

private:
    DacHandler HandleForDac;
    MatchFinder Matcher;

public:
    MyASTConsumer() {
        // 可以通过 addMatcher 添加用户构造的匹配器到 MatchFinder中
        // Matcher.addMatcher(binaryOperator(hasOperatorName("<->")).bind("dac_expr"), &HandleForDac);
        Matcher.addMatcher(functionDecl(hasDescendant(binaryOperator(hasOperatorName("<->")))).bind("dac_expr_father"), &HandleForDac);
        Matcher.addMatcher(binaryOperator(hasOperatorName("<->")).bind("dac_expr"), &HandleForDac);
        Matcher.addMatcher(functionDecl(hasName("main")).bind("main"), &HandleForDac);
    }

    void HandleTranslationUnit(ASTContext &Context) override {
        // Run the matchers when we have the whole TU parsed.
        Matcher.matchAST(Context);
        dacppFile->setTranslationUnitDecl(Context.getTranslationUnitDecl());
    }

};

// ASTFrontendAction 是为使用基于 AST consumer 的前端动作的抽象基类
// FrontendAction 是允许用户特定动作作为编译的一部分执行的接口
// 需要实现 CreateASTConsumer 方法，该方法对于每个翻译单元返回一个 ASTConsumer
class MyFrontendAction : public ASTFrontendAction {

private:
    Rewriter* clangRewriter;

public:
    MyFrontendAction() {
        clangRewriter = new Rewriter();
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        clangRewriter->setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<MyASTConsumer>();
    }

    void EndSourceFileAction() override {
        dacppTranslator::Rewriter* rewriter = new dacppTranslator::Rewriter();
        rewriter->setRewriter(clangRewriter);
        rewriter->setDacppFile(dacppFile);
        // dacppTranslator::printDacppFileInfo(dacppFile);
        // rewriter->rewriteDac_Usm();
        rewriter->rewriteDac_Multiple();
        rewriter->rewriteMain();

        // // this will output to screen as what you got.
        // clangRewriter->getEditBuffer(clangRewriter->getSourceMgr().getMainFileID())
        //     .write(llvm::outs());
        
        // 生成 SYCL 文件
        std::error_code error_code;
        std::string file = getCurrentFile().str();
        file.replace(file.find(".cpp"), 4, "_sycl.cpp");
        llvm::raw_fd_ostream outFile(file, error_code, llvm::sys::fs::OF_None);
        // this will write the result to outFile
        clangRewriter->getEditBuffer(clangRewriter->getSourceMgr().getMainFileID()).write(outFile);
        outFile.close();
  }

};

// 命令行选项的帮助信息
static llvm::cl::OptionCategory translator("translator options");

int main(int argc, const char **argv) {
    // 所有命令行 Clang 工具通用的选项解析器
    auto ExpectedParser =
        CommonOptionsParser::create(argc, argv, translator);
    if (!ExpectedParser) {
      llvm::errs() << ExpectedParser.takeError();
      return 1;
    }
    CommonOptionsParser &op = ExpectedParser.get();

    // 运行一个前端动作的工具
    ClangTool tool(op.getCompilations(), op.getSourcePathList());

    // 在命令行指定的所有文件上运行一个动作
    return tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}