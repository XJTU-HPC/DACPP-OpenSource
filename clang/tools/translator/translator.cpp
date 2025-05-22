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
#include <regex>

#include "ReconTensor.h"
#include "Split.h"
#include "Param.h"
#include "DacppStructure.h"
#include "Rewriter.h"
#include "ASTParse.h"
#include "Dacfor.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
// #define dac_for for

dacppTranslator::DacppFile* dacppFile = new dacppTranslator::DacppFile();

std::unordered_map<FunctionDecl*, std::set<FunctionDecl*>> dacExprMap;


/*
  ASTMatcher 匹配 DACPP 文件中符合要求的节点
*/
class DacHandler : public MatchFinder::MatchCallback {
  
public:
    DacHandler() {}

    dacppTranslator::ControlBlock* block = new dacppTranslator::ControlBlock();
    virtual void run(const MatchFinder::MatchResult &Result) override {
        // printf("Found a match\n");
        // 匹配数据关联计算表达式

        const SourceManager& SM = *Result.SourceManager;
        const LangOptions& LangOpts = Result.Context->getLangOpts();
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
            // mainFunc->dump();
            if(!dacppFile->node){
                dacppFile->node = mainFunc;
            }
        } else if (const CallExpr* call = Result.Nodes.getNodeAs<CallExpr>("dac_for_call")) {
            if (!call) return;
            dacppFile->setForStmt(call);
            const SourceManager& SM = *Result.SourceManager;
            const LangOptions& LangOpts = Result.Context->getLangOpts();
            clang::PrintingPolicy policy(LangOpts);
            policy.adjustForCPlusPlus();
        
            // 提取 loop bound（dac_for 的第一个参数）
            if (call->getNumArgs() < 2) return;
            const Expr* boundExpr = call->getArg(0)->IgnoreImpCasts();
            std::string loopBoundStr = Lexer::getSourceText(CharSourceRange::getTokenRange(boundExpr->getSourceRange()), SM, LangOpts).str();
            llvm::outs() << "Loop bound: " << loopBoundStr << "\n";
        
            // 设置 control block
            block->setLoopBound(loopBoundStr);
        
            // 提取 lambda（第二个参数）
            const Expr* lambdaExpr = call->getArg(1);
            lambdaExpr = lambdaExpr->IgnoreImplicit();   // 忽略 MaterializeTemporaryExpr
            lambdaExpr = lambdaExpr->IgnoreParenCasts(); // 忽略括号和显式/隐式 cast
            lambdaExpr = lambdaExpr->IgnoreCasts();      // 进一步忽略 cast（可选）

            const LambdaExpr* lambda = dyn_cast<LambdaExpr>(lambdaExpr);
            if (!lambda) return;
            printf("Lambda is exits\n");
            const Stmt* lambdaBody = lambda->getBody();
            if (!lambdaBody) return;
            // llvm::outs() << "Lambda body is: " << lambdaBody->getStmtClassName() << "\n";
            // 寻找 Lambda 中的 ForStmt 并处理其子语句
            for (const Stmt* stmt : lambdaBody->children()) {
                // printf("寻找ForStmt\n");
                if (const ForStmt* forStmt = dyn_cast<ForStmt>(stmt)) {
                    processStmt(forStmt->getBody(), SM, LangOpts, block);
                }
            }
            block->setLoopBound(loopBoundStr);
            dacppFile->setBlock(block);
            // dacppFile->setForRange(call->getSourceRange());
        
            printResult();
        }
        
    }

    std::string getSourceText(const Expr* expr, const SourceManager& SM, const LangOptions& LangOpts) {
        SourceRange range = expr->getSourceRange();
        return Lexer::getSourceText(CharSourceRange::getTokenRange(range), SM, LangOpts).str();
    }

    void printResult() const {
        std::cout << "Loop Count: " << block->getLoopBound() << "\n";
        for (const auto& pc : block->getParamControls()) {
            std::cout << "  Swap: "
                      << pc->getParamA()->getName() << " <=> " << pc->getParamB()->getName() << "\n";
        }
    }
    std::vector<std::string> extractShape(const std::string& exprStr) {
        std::vector<std::string> shapeList;
        std::regex slice_regex(R"(\{([^{}]+)\})");  // 匹配 {内容}
    
        auto begin = std::sregex_iterator(exprStr.begin(), exprStr.end(), slice_regex);
        auto end = std::sregex_iterator();
    
        for (auto it = begin; it != end; ++it) {
            std::string insideBraces = (*it)[1].str();  // 括号里的内容，比如 "1,7"
    
            // 用逗号分割字符串
            std::stringstream ss(insideBraces);
            std::string item;
            while (std::getline(ss, item, ',')) {
                // 去除两端空白
                size_t start = item.find_first_not_of(" \t\n\r");
                size_t end = item.find_last_not_of(" \t\n\r");
                if (start != std::string::npos && end != std::string::npos) {
                    shapeList.push_back(item.substr(start, end - start + 1));
                } else if (!item.empty()) {
                    shapeList.push_back(item);
                }
            }
        }
    
        return shapeList;
    }
    std::string extractBaseName(const std::string& exprStr) {
        size_t pos = exprStr.find("[{");
        if (pos != std::string::npos) {
            return exprStr.substr(0, pos);
        }
        return exprStr;
    }
    void processStmt(const Stmt* stmt, const SourceManager& SM, const LangOptions& LangOpts, dacppTranslator::ControlBlock* block) {
        if (!stmt) return;
        // printf("find a loop");
        if (const CallExpr* call = dyn_cast<CallExpr>(stmt)) {
            const FunctionDecl* callee = call->getDirectCallee();
            if (!callee) return;
    
            std::string funcName = callee->getQualifiedNameAsString();
            if (funcName != "dacpp::swap") return;
    
            // 解析 swap 的两个参数
            if (call->getNumArgs() < 2) return;
    
            const Expr* argA = call->getArg(0)->IgnoreImpCasts();
            const Expr* argB = call->getArg(1)->IgnoreImpCasts();
    
            std::string argA_str = Lexer::getSourceText(CharSourceRange::getTokenRange(argA->getSourceRange()), SM, LangOpts).str();
            std::string argB_str = Lexer::getSourceText(CharSourceRange::getTokenRange(argB->getSourceRange()), SM, LangOpts).str();
    
            llvm::outs() << "Found swap: " << argA_str << " <=> " << argB_str << "\n";
    
            // 构建 ParamControl
            auto pc = std::make_shared<dacppTranslator::ParamControl>();
    
            auto paramA = new dacppTranslator::Param();
            paramA->setName(extractBaseName(argA_str));
            pc->setParamA(paramA);
    
            auto paramB = new dacppTranslator::Param();
            paramB->setName(extractBaseName(argB_str));
            pc->setParamB(paramB);
    
            pc->setOperation(dacppTranslator::SemanticOperation::Swap);
    
            pc->setShapeA(extractShape(argA_str));
            pc->setShapeB(extractShape(argB_str));
    
            block->addParamControl(pc);
        }
    
        // 递归遍历所有子语句
        for (const Stmt* child : stmt->children()) {
            processStmt(child, SM, LangOpts, block);
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
        Matcher.addMatcher(callExpr(callee(functionDecl(hasName("dac_for")))).bind("dac_for_call"), &HandleForDac);
        
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
        if(dacppFile->getBlock()){
            dacppFile->setHeaderFile("\"DataReconstructor.new.h\"");
            dacppFile->setHeaderFile("\"ParameterGeneration.h\"");
            dacppFile->setHeaderFile("\"utils.h\"");
        }else{
            dacppFile->setHeaderFile("\"DataReconstructor1.h\"");
            dacppFile->setHeaderFile("\"ParameterGeneration.h\"");
        }
        rewriter->setRewriter(clangRewriter);
        rewriter->setDacppFile(dacppFile);

        // dacppTranslator::printDacppFileInfo(dacppFile);
        // rewriter->rewriteDac_Usm();
        //rewriter->rewriteDac_Buffer();
        rewriter->rewriteDac_Multiple();

        // rewriter->rewriteMPI();
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