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
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Lexer.h"
using namespace clang;

namespace {

constexpr char kBufferOutputSuffix[] = "_sycl_buffer.cpp";

// Preserve the source-level DACPP translation hint when present.
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

    }

private:
    SourceManager &SM;
    int &Mode;
};

} // anonymous namespace

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;

dacppTranslator::DacppFile *dacppFile = new dacppTranslator::DacppFile();

std::unordered_map<FunctionDecl *, std::set<FunctionDecl *>> dacExprMap;

class DacHandler : public MatchFinder::MatchCallback {

public:
  DacHandler() {}

  dacppTranslator::ControlBlock *block = new dacppTranslator::ControlBlock();
  virtual void run(const MatchFinder::MatchResult &Result) override {
    if (!dacppFile->Context)
        dacppFile->Context = Result.Context;

    if (const BinaryOperator *dacExpr =
        Result.Nodes.getNodeAs<clang::BinaryOperator>("dac_expr")) {

    dacppFile->dacExprs.push_back(dacExpr);

    if (dyn_cast<BinaryOperator>(dacExpr->getLHS()) ||
        dyn_cast<BinaryOperator>(dacExpr->getRHS())) {
      llvm::errs() << "error: multi binary operator in a dac statement\n";
      return;
    }

    Expr *dacExprLHS = dacppTranslator::Expression::shellLHS_p(dacExpr)
                           ? dacExpr->getLHS()
                           : dacExpr->getRHS();
    CallExpr *shellCall = dacppTranslator::getNode<CallExpr>(dacExprLHS);
    FunctionDecl *functionDecl = shellCall->getDirectCallee();

    dacppFile->shellVars.clear();
    for (unsigned i = 0; i < functionDecl->getNumParams(); i++) {
        ParmVarDecl* param = functionDecl->getParamDecl(i);
        std::string paramName = param->getNameAsString();
        std::string paramType = param->getType().getAsString();

        dacppFile->shellVars.emplace_back(paramName, paramType);
    }

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

    ASTContext *Ctx = Result.Context;
    clang::DynTypedNode cur = clang::DynTypedNode::create(*dacExpr);
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
    }
  }
}
    else if (const FunctionDecl* mainFunc =
             Result.Nodes.getNodeAs<clang::FunctionDecl>("main")) {
    dacppFile->setMainFunction(mainFunc);
    dacppFile->setMainBody(mainFunc->getBody());
    dacppFile->setContext(Result.Context);
    dacppFile->setMainFuncLoc(mainFunc);
}else if (const FunctionDecl *mainFunc =
                   Result.Nodes.getNodeAs<clang::FunctionDecl>(
                       "dac_expr_father")) {
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

      if (call->getNumArgs() < 2)
        return;
      const Expr *boundExpr = call->getArg(0)->IgnoreImpCasts();
      std::string loopBoundStr =
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(boundExpr->getSourceRange()), SM,
              LangOpts)
              .str();

      block->setLoopBound(loopBoundStr);

      const Expr *lambdaExpr = call->getArg(1);
      lambdaExpr = lambdaExpr->IgnoreImplicit();
      lambdaExpr = lambdaExpr->IgnoreParenCasts();
      lambdaExpr = lambdaExpr->IgnoreCasts();

      const LambdaExpr *lambda = dyn_cast<LambdaExpr>(lambdaExpr);
      if (!lambda)
        return;
      const Stmt *lambdaBody = lambda->getBody();
      if (!lambdaBody)
        return;
      for (const Stmt *stmt : lambdaBody->children()) {
        if (const ForStmt *forStmt = dyn_cast<ForStmt>(stmt)) {
          processStmt(forStmt->getBody(), SM, LangOpts, block);
        }
      }
      block->setLoopBound(loopBoundStr);
      dacppFile->setBlock(block);
    }
  }
  std::vector<std::string> extractShape(const std::string &exprStr) {
    std::vector<std::string> shapeList;
    std::regex slice_regex(R"(\{([^{}]+)\})");

    auto begin =
        std::sregex_iterator(exprStr.begin(), exprStr.end(), slice_regex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
      std::string insideBraces = (*it)[1].str();

      std::stringstream ss(insideBraces);
      std::string item;
      while (std::getline(ss, item, ',')) {
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
    if (const CallExpr *call = dyn_cast<CallExpr>(stmt)) {
      const FunctionDecl *callee = call->getDirectCallee();
      if (!callee)
        return;

      std::string funcName = callee->getQualifiedNameAsString();
      if (funcName != "dacpp::swap")
        return;

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

    for (const Stmt *child : stmt->children()) {
      processStmt(child, SM, LangOpts, block);
    }
  }
};

class MyASTConsumer : public ASTConsumer {

private:
  DacHandler HandleForDac;
  MatchFinder Matcher;

public:
  MyASTConsumer() {
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
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    Matcher.matchAST(Context);
    dacppFile->setTranslationUnitDecl(Context.getTranslationUnitDecl());
  }
};

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
    clangRewriter->setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<MyASTConsumer>();
  }

  void EndSourceFileAction() override {
    dacppTranslator::Rewriter *rewriter = new dacppTranslator::Rewriter();
    dacppFile->mode = TranslateMode;
    if (dacppFile->getBlock()) {
      dacppFile->setHeaderFile("\"DataReconstructor.new.h\"");
      dacppFile->setHeaderFile("\"ParameterGeneration.h\"");
      dacppFile->setHeaderFile("\"utils.h\"");
    } else {
      dacppFile->setHeaderFile("\"DataReconstructor1.h\"");
      dacppFile->setHeaderFile("\"ParameterGeneration.h\"");
    }
    rewriter->setRewriter(clangRewriter);
    rewriter->setDacppFile(dacppFile);
    rewriter->rewriteDac_Buffer();
    rewriter->rewriteMain();

    std::error_code error_code;
    std::string file = getCurrentFile().str();
    size_t pos = file.find(".cpp");
    if (pos != std::string::npos) {
      file.replace(pos, 4, kBufferOutputSuffix);
    }

    llvm::raw_fd_ostream outFile(file, error_code, llvm::sys::fs::OF_None);
    clangRewriter->getEditBuffer(clangRewriter->getSourceMgr().getMainFileID())
        .write(outFile);
    outFile.close();
  }
};

int main(int argc, const char **argv) {
  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, llvm::cl::getGeneralCategory());
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &op = ExpectedParser.get();
  ClangTool tool(op.getCompilations(), op.getSourcePathList());
  return tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
