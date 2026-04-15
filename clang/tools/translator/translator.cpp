#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include "clang/AST/ASTTypeTraits.h"

#include "ASTParse.h"
#include "Dacfor.h"
#include "DacppStructure.h"
#include "Param.h"
#include "ReconTensor.h"
#include "Rewriter.h"
#include "Split.h"
#include "llvm/Support/CommandLine.h"
// 为了处理宏定义
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Lexer.h"
// LLVM 命令行选项
//宏定义检测器
using namespace clang;

namespace {

class DacppMacroDetector : public PPCallbacks {
public:
    DacppMacroDetector(SourceManager &SM, int &Mode)
        : SM(SM), Mode(Mode) {}

    void MacroDefined(const Token &Tok,
                      const MacroDirective *MD) override {

        auto *II = Tok.getIdentifierInfo();
        if (!II)
            return;

        if (II->getName() != "DACPP_TRANSLATE_MODE")
            return;

        if (!SM.isWrittenInMainFile(Tok.getLocation()))
            return;

        const MacroInfo *MI = MD->getMacroInfo();
        if (!MI || MI->getNumTokens() != 1)
            return;

        const Token &ValTok = MI->getReplacementToken(0);
        if (!ValTok.is(tok::numeric_constant))
            return;

        StringRef Val =
            Lexer::getSpelling(ValTok, SM, LangOptions());

        Mode = std::stoi(Val.str());

        llvm::errs()
            << "[DACPP] DACPP_TRANSLATE_MODE = "
            << Mode << "\n";
    }

private:
    SourceManager &SM;
    int &Mode;
};

} // anonymous namespace
static llvm::cl::opt<std::string>
    ModeOpt("mode",
            llvm::cl::desc("Choose backend mode: usm or buffer or usm_time"),
            llvm::cl::init("buffer"));

static llvm::cl::alias UsmMode("usm", llvm::cl::desc("Alias for --mode=usm"),
                               llvm::cl::aliasopt(ModeOpt));
static llvm::cl::alias BufferMode("buffer",
                                  llvm::cl::desc("Alias for --mode=buffer"),
                                  llvm::cl::aliasopt(ModeOpt));
static llvm::cl::alias Usm_timeMode("usm_time",
                                    llvm::cl::desc("Alias for --mode=usm_time"),
                                    llvm::cl::aliasopt(ModeOpt));
static llvm::cl::opt<bool>
    MpiOpt("mpi",
           llvm::cl::desc("Enable V1 MPI code generation for non-stencil programs"),
           llvm::cl::init(false));

enum class MpiOutputSyncMode {
  AllRanks,
  RootOnly
};

static llvm::cl::opt<MpiOutputSyncMode>
    MpiOutputSyncModeOpt(
        "mpi-output-sync",
        llvm::cl::desc("How WRITE/READ_WRITE outputs are synchronized after root gather"),
        llvm::cl::values(
            clEnumValN(MpiOutputSyncMode::AllRanks, "all-ranks",
                       "Gather on root, then broadcast updated outputs back to every rank"),
            clEnumValN(MpiOutputSyncMode::RootOnly, "root-only",
                       "Gather on root only; skip the final output broadcast")),
        llvm::cl::init(MpiOutputSyncMode::AllRanks));

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
// #define dac_for for

dacppTranslator::DacppFile *dacppFile = new dacppTranslator::DacppFile();

std::unordered_map<FunctionDecl *, std::set<FunctionDecl *>> dacExprMap;

/*
  ASTMatcher 匹配 DACPP 文件中符合要求的节点
*/
class DacHandler : public MatchFinder::MatchCallback {

public:
  DacHandler() {}

  dacppTranslator::ControlBlock *block =
      new dacppTranslator::ControlBlock(); // 循环控制块
  virtual void run(const MatchFinder::MatchResult &Result) override {
    std::cout<<"运行匹配处理程序"<<std::endl;
    //  第一步：保存 ASTContext 
    if (!dacppFile->Context)
        dacppFile->Context = Result.Context;

    // printf("Found a match\n");
    // 匹配数据关联计算表达式
    
    const SourceManager &SM = *Result.SourceManager;
    const LangOptions &LangOpts = Result.Context->getLangOpts();
    if (const BinaryOperator *dacExpr =
        Result.Nodes.getNodeAs<clang::BinaryOperator>("dac_expr")) {

    dacppFile->dacExprs.push_back(dacExpr);

    if (dyn_cast<BinaryOperator>(dacExpr->getLHS()) ||
        dyn_cast<BinaryOperator>(dacExpr->getRHS())) {
      llvm::errs() << "error: multi binary operator in a dac statement\n";
      return;
    }

    // 获取 DAC 数据关联表达式左值（shell侧）
    Expr *dacExprLHS = dacppTranslator::Expression::shellLHS_p(dacExpr)
                           ? dacExpr->getLHS()
                           : dacExpr->getRHS();
    CallExpr *shellCall = dacppTranslator::getNode<CallExpr>(dacExprLHS);
    FunctionDecl *functionDecl = shellCall->getDirectCallee();

    // === ★★★ 新增：记录 shell 参数 ===
    dacppFile->shellVars.clear();  // 若一个源文件多次解析，可清空
    for (unsigned i = 0; i < functionDecl->getNumParams(); i++) {
        ParmVarDecl* param = functionDecl->getParamDecl(i);
        std::string paramName = param->getNameAsString();
        std::string paramType = param->getType().getAsString();

        dacppFile->shellVars.emplace_back(paramName, paramType);
    }

    llvm::outs() << "[SHELL] collected params of shell function: "
                 << functionDecl->getNameAsString() << "\n";
    for (auto& p : dacppFile->shellVars) {
        llvm::outs() << "    param " << p.first
                     << " type=" << p.second << "\n";
    }
    // === ★★★ 新增结束 ===


    Expr *curExpr = shellCall->getArg(0);
    DeclRefExpr *declRefExpr;
    if (isa<DeclRefExpr>(curExpr)) {
      declRefExpr = dyn_cast<DeclRefExpr>(curExpr);
    } else {
      declRefExpr = dacppTranslator::getNode<DeclRefExpr>(curExpr);
    }
    if (isa<ParmVarDecl>(declRefExpr->getDecl())) {
      return;
    }

    if (isa<DeclRefExpr>(dacExpr->getRHS())) {
      declRefExpr = dyn_cast<DeclRefExpr>(dacExpr->getRHS());
    } else {
      declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());
    }

    FunctionDecl *calcFunc = dyn_cast<FunctionDecl>(declRefExpr->getDecl());
    if (dacExprMap.find(functionDecl) != dacExprMap.end()) {
      if (dacExprMap[functionDecl].count(calcFunc) == 1) {
        return;
      }
    } else {
      dacExprMap.emplace(functionDecl, std::set<FunctionDecl *>());
    }
    dacExprMap[functionDecl].emplace(calcFunc);

    dacppFile->setExpression(dacExpr);

    // ------------------ 查找最外层循环 --------------------
    ASTContext *Ctx = Result.Context;
    clang::DynTypedNode cur = clang::DynTypedNode::create(*dacExpr);
    const Stmt *inner = nullptr;
    const Stmt *outer = nullptr;

    while (true) {
      auto parents = Ctx->getParents(cur);
      if (parents.empty())
          break;
      const auto &p = parents[0];
      if (const Stmt *loop = p.get<ForStmt>() ? (const Stmt *)p.get<ForStmt>()
                               : p.get<WhileStmt>()
                                   ? (const Stmt *)p.get<WhileStmt>()
                                   : nullptr) {
        if (!inner) inner = loop;
        outer = loop;
      }
      if (p.get<FunctionDecl>()) break;
      cur = p;
    }
    if (outer) {
    auto *FS = llvm::dyn_cast<clang::ForStmt>(outer);
    if (FS) {
      dacppFile->setForStatement(FS);
      dacppFile->collectVarsFromForStatement();
      llvm::outs() << "[DAC-Translator] Found outermost loop for <->\n";
    } else {
      llvm::outs() << "[DAC-Translator] <-> NOT inside any loop\n";
    }
  }
}

    /*2025.12.1修改，在匹配到主函数的情况下记录主函数的delc\stmt\代码的context*/
    // 匹配主函数
    else if (const FunctionDecl* mainFunc =
             Result.Nodes.getNodeAs<clang::FunctionDecl>("main")) {

    // 存储主函数的 FunctionDecl
    dacppFile->setMainFunction(mainFunc);

    // 存储主函数的函数体 Stmt*
    dacppFile->setMainBody(mainFunc->getBody());

    // 存储当前 ASTContext
    dacppFile->setContext(Result.Context);

    // 保留旧接口（如果仍依赖 mainFuncLoc）
    dacppFile->setMainFuncLoc(mainFunc);
}else if (const FunctionDecl *mainFunc =
                   Result.Nodes.getNodeAs<clang::FunctionDecl>(
                       "dac_expr_father")) {
      // mainFunc->dump();
      if (!dacppFile->node) {
        dacppFile->node = mainFunc;
      }
    } else if (const CallExpr *call =
                   Result.Nodes.getNodeAs<CallExpr>("dac_for_call")) {
      if (!call)
        return;
      dacppFile->setForStmt(call);
      const SourceManager &SM = *Result.SourceManager;
      const LangOptions &LangOpts = Result.Context->getLangOpts();
      clang::PrintingPolicy policy(LangOpts);
      policy.adjustForCPlusPlus();

      // 提取 loop bound（dac_for 的第一个参数）
      if (call->getNumArgs() < 2)
        return;
      const Expr *boundExpr = call->getArg(0)->IgnoreImpCasts();
      std::string loopBoundStr =
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(boundExpr->getSourceRange()), SM,
              LangOpts)
              .str();
      llvm::outs() << "Loop bound: " << loopBoundStr << "\n";

      // 设置 control block
      block->setLoopBound(loopBoundStr);

      // 提取 lambda（第二个参数）
      const Expr *lambdaExpr = call->getArg(1);
      lambdaExpr =
          lambdaExpr->IgnoreImplicit(); // 忽略 MaterializeTemporaryExpr
      lambdaExpr = lambdaExpr->IgnoreParenCasts(); // 忽略括号和显式/隐式 cast
      lambdaExpr = lambdaExpr->IgnoreCasts();      // 进一步忽略 cast（可选）

      const LambdaExpr *lambda = dyn_cast<LambdaExpr>(lambdaExpr);
      if (!lambda)
        return;
      printf("Lambda is exits\n");
      const Stmt *lambdaBody = lambda->getBody();
      if (!lambdaBody)
        return;
      // llvm::outs() << "Lambda body is: " << lambdaBody->getStmtClassName() <<
      // "\n"; 寻找 Lambda 中的 ForStmt 并处理其子语句
      for (const Stmt *stmt : lambdaBody->children()) {
        // printf("寻找ForStmt\n");
        if (const ForStmt *forStmt = dyn_cast<ForStmt>(stmt)) {
          processStmt(forStmt->getBody(), SM, LangOpts, block);
        }
      }
      block->setLoopBound(loopBoundStr);
      dacppFile->setBlock(block);
      // dacppFile->setForRange(call->getSourceRange());

      printResult();
    }
  }

  std::string getSourceText(const Expr *expr, const SourceManager &SM,
                            const LangOptions &LangOpts) {
    SourceRange range = expr->getSourceRange();
    return Lexer::getSourceText(CharSourceRange::getTokenRange(range), SM,
                                LangOpts)
        .str();
  }

  void printResult() const {
    std::cout << "Loop Count: " << block->getLoopBound() << "\n";
    for (const auto &pc : block->getParamControls()) {
      std::cout << "  Swap: " << pc->getParamA()->getName() << " <=> "
                << pc->getParamB()->getName() << "\n";
    }
  }
  std::vector<std::string> extractShape(const std::string &exprStr) {
    std::vector<std::string> shapeList;
    std::regex slice_regex(R"(\{([^{}]+)\})"); // 匹配 {内容}

    auto begin =
        std::sregex_iterator(exprStr.begin(), exprStr.end(), slice_regex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
      std::string insideBraces = (*it)[1].str(); // 括号里的内容，比如 "1,7"

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
  std::string extractBaseName(const std::string &exprStr) {
    size_t pos = exprStr.find("[{");
    if (pos != std::string::npos) {
      return exprStr.substr(0, pos);
    }
    return exprStr;
  }
  void processStmt(const Stmt *stmt, const SourceManager &SM,
                   const LangOptions &LangOpts,
                   dacppTranslator::ControlBlock *block) {
    if (!stmt)
      return;
    // printf("find a loop");
    if (const CallExpr *call = dyn_cast<CallExpr>(stmt)) {
      const FunctionDecl *callee = call->getDirectCallee();
      if (!callee)
        return;

      std::string funcName = callee->getQualifiedNameAsString();
      if (funcName != "dacpp::swap")
        return;

      // 解析 swap 的两个参数
      if (call->getNumArgs() < 2)
        return;

      const Expr *argA = call->getArg(0)->IgnoreImpCasts();
      const Expr *argB = call->getArg(1)->IgnoreImpCasts();

      std::string argA_str =
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(argA->getSourceRange()), SM,
              LangOpts)
              .str();
      std::string argB_str =
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(argB->getSourceRange()), SM,
              LangOpts)
              .str();

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
    for (const Stmt *child : stmt->children()) {
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
    // Matcher.addMatcher(binaryOperator(hasOperatorName("<->")).bind("dac_expr"),
    // &HandleForDac);
    Matcher.addMatcher(
        functionDecl(hasDescendant(binaryOperator(hasOperatorName("<->"))))
            .bind("dac_expr_father"),
        &HandleForDac);
    Matcher.addMatcher(binaryOperator(hasOperatorName("<->")).bind("dac_expr"),
                       &HandleForDac);
    Matcher.addMatcher(functionDecl(hasName("main")).bind("main"),
                       &HandleForDac);
    Matcher.addMatcher(
        functionDecl(hasName("shell")).bind("shellDecl"),
        &HandleForDac
      );
    // Matcher.addMatcher(forStmt(hasDescendant(binaryOperator(hasOperatorName("<->")).bind("dac_expr_in_for"))).bind("for_with_dac_expr"),
    // &HandleForDac);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    std::cout<<"处理翻译单元"<<std::endl;
    Matcher.matchAST(Context);
    std::cout<<"处理翻译单元"<<std::endl;
    dacppFile->setTranslationUnitDecl(Context.getTranslationUnitDecl());
  }
};

// ASTFrontendAction 是为使用基于 AST consumer 的前端动作的抽象基类
// FrontendAction 是允许用户特定动作作为编译的一部分执行的接口
// 需要实现 CreateASTConsumer 方法，该方法对于每个翻译单元返回一个 ASTConsumer
class MyFrontendAction : public ASTFrontendAction {

private:
  Rewriter *clangRewriter;
  int TranslateMode = 0; 
public:
  MyFrontendAction() { 

    clangRewriter = new Rewriter(); 
  }
  bool BeginSourceFileAction(CompilerInstance &CI) override {
    CI.getPreprocessor().addPPCallbacks(
        std::make_unique<DacppMacroDetector>(
            CI.getSourceManager(),
            TranslateMode
        )
    );
    return true;
}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    llvm::outs() << "[DAC-Translator--node1] Creating AST consumer for file: "
                 << file.str() << "\n";
    std::cout<<"为文件创建AST consumer: "<<file.str()<<std::endl;
    clangRewriter->setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<MyASTConsumer>();
  }

  void EndSourceFileAction() override {
    std::cout << "处理翻译单元" << std::endl;
    dacppTranslator::Rewriter *rewriter = new dacppTranslator::Rewriter();
    dacppFile->mode = TranslateMode;
    dacppFile->setEnableMPI(MpiOpt);
    dacppFile->setMPIBroadcastOutputs(
        MpiOutputSyncModeOpt == MpiOutputSyncMode::AllRanks);
    if (dacppFile->getBlock()) {
      dacppFile->setHeaderFile("\"DataReconstructor.new.h\"");
      dacppFile->setHeaderFile("\"ParameterGeneration.h\"");
      dacppFile->setHeaderFile("\"utils.h\"");
    } else {
      dacppFile->setHeaderFile("\"DataReconstructor1.h\"");
      dacppFile->setHeaderFile("\"ParameterGeneration.h\"");
    }
    if (MpiOpt) {
      dacppFile->setHeaderFile("<mpi.h>");
      dacppFile->setHeaderFile("<cstdio>");
      dacppFile->setHeaderFile("\"MPIPlanner.h\"");
    }
    rewriter->setRewriter(clangRewriter);
    rewriter->setDacppFile(dacppFile);

    if (ModeOpt == "buffer" || MpiOpt) {
      dacppFile->analyzeBufferRegionPlan();
      if (dacppFile->hasBufferRegionPlan()) {
        if (MpiOpt &&
            !dacppFile->getBufferRegionPlan().siblingForStmts.empty()) {
          llvm::outs() << "[DACPP] MPI region optimization deferred: "
                       << "sibling loops are not deviceized yet; "
                       << "falling back to stable MPI wrapper\n";
        } else {
          llvm::outs() << "[DACPP] "
                       << (MpiOpt ? "MPI" : "buffer")
                       << " region optimization enabled\n";
        }
      } else if (!dacppFile->getBufferRegionPlan().disableReason.empty()) {
        llvm::outs() << "[DACPP] "
                     << (MpiOpt ? "MPI" : "buffer")
                     << " region optimization skipped: "
                     << dacppFile->getBufferRegionPlan().disableReason << "\n";
      }
    }

    // dacppTranslator::printDacppFileInfo(dacppFile);

if (MpiOpt) {
    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    if (dacppFile->hasBufferRegionPlan() &&
        regionPlan.siblingForStmts.empty()) {
        rewriter->rewriteMPI_Region();
    } else {
        rewriter->rewriteMPI();
    }
} else if (ModeOpt == "usm") {
  //  rewriter->rewriteDac_Usm();
} else if (ModeOpt == "buffer") {
    std::cout<<"使用BUFFER版本翻译"<<std::endl;
    rewriter->rewriteDac_Buffer();
} else if(ModeOpt == "usm_time"){
    //rewriter->rewriteDac_Usm_time();
}else {
    // 意外情况 fallback
   // rewriter->rewriteDac_Usm();
}
    rewriter->rewriteMain();

    std::error_code error_code;
    std::string file = getCurrentFile().str();
    size_t pos = file.find(".cpp");
    if (pos != std::string::npos) {
      file.replace(pos, 4, "_sycl_" + ModeOpt + ".cpp");
    }

    // rewriter->rewriteDac_Usm();
    // rewriter->rewriteDac_Buffer();
    //  rewriter->rewriteDac_Multiple();

    // rewriter->rewriteMPI();
    // rewriter->rewriteMain();

    // // this will output to screen as what you got.
    // clangRewriter->getEditBuffer(clangRewriter->getSourceMgr().getMainFileID())
    //     .write(llvm::outs());

    // 生成 SYCL 文件
    // std::error_code error_code;
    // std::string file = getCurrentFile().str();
    // file.replace(file.find(".cpp"), 4, "_sycl.cpp");

    llvm::raw_fd_ostream outFile(file, error_code, llvm::sys::fs::OF_None);
    // this will write the result to outFile
    clangRewriter->getEditBuffer(clangRewriter->getSourceMgr().getMainFileID())
        .write(outFile);
    outFile.close();
  }
};

// 命令行选项的帮助信息
static llvm::cl::OptionCategory translator("translator options");

int main(int argc, const char **argv) {
  // 所有命令行 Clang 工具通用的选项解析器
  // auto ExpectedParser =
  //     CommonOptionsParser::create(argc, argv, translator);
  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, llvm::cl::getGeneralCategory());
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &op = ExpectedParser.get();
  // 运行一个前端动作的工具
  ClangTool tool(op.getCompilations(), op.getSourcePathList());
  // 在命令行指定的所有文件上运行一个动作
  // std::cout << "Mode: " << ModeOpt << std::endl;
  return tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
