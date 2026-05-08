//实现dacppfile的文件
#include <algorithm>
#include <string>
#include <regex>
#include <set>

#include "clang/AST/Attr.h"
#include "llvm/ADT/StringExtras.h"

#include "DacppStructure.h"
#include "ASTParse.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/AST.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/STLExtras.h"


using namespace clang;

static bool rangeContains(const clang::SourceManager& SM,
                          clang::SourceRange outer,
                          clang::SourceRange inner) {
    if (outer.isInvalid() || inner.isInvalid()) {
        return false;
    }

    auto beforeOrEqual = [&](clang::SourceLocation lhs, clang::SourceLocation rhs) {
        return lhs == rhs || SM.isBeforeInTranslationUnit(lhs, rhs);
    };

    return beforeOrEqual(outer.getBegin(), inner.getBegin()) &&
           beforeOrEqual(inner.getEnd(), outer.getEnd());
}

static bool containsWord(const std::string& text, const std::string& word) {
    std::regex pattern("\\b" + word + "\\b");
    return std::regex_search(text, pattern);
}

static bool isSupportedRegionLoop(const clang::ForStmt* FS,
                                  clang::ASTContext* Context) {
    if (!FS || !Context) {
        return false;
    }

    const std::string loopText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(FS->getSourceRange()),
        Context->getSourceManager(),
        Context->getLangOpts()).str();

    if (loopText.empty() || loopText.find("<->") != std::string::npos) {
        return false;
    }

    static const std::vector<std::string> kRejectedKeywords = {
        "while", "switch", "return", "break", "continue", "goto"
    };
    for (const auto& keyword : kRejectedKeywords) {
        if (containsWord(loopText, keyword)) {
            return false;
        }
    }

    return true;
}

static bool buildBufferRegionPlanForExprImpl(
    dacppTranslator::DacppFile* dacppFile,
    dacppTranslator::Shell* shell,
    const clang::BinaryOperator* dacExpr,
    const clang::Stmt* outerLoop,
    dacppTranslator::BufferRegionPlan& bufferRegionPlan,
    std::string* disableReason) {
    bufferRegionPlan = dacppTranslator::BufferRegionPlan{};
    if (!dacppFile || !shell || !dacppFile->getContext() || !outerLoop ||
        !dacExpr) {
        const std::string reason = "missing loop or DAC expression";
        bufferRegionPlan.disableReason = reason;
        if (disableReason) {
            *disableReason = reason;
        }
        return false;
    }

    const clang::Stmt* rawBody = nullptr;
    if (const auto* FS = llvm::dyn_cast<clang::ForStmt>(outerLoop)) {
        rawBody = FS->getBody();
    } else if (const auto* WS = llvm::dyn_cast<clang::WhileStmt>(outerLoop)) {
        rawBody = WS->getBody();
    } else {
        const std::string reason = "outer loop must be for or while";
        bufferRegionPlan.disableReason = reason;
        if (disableReason) {
            *disableReason = reason;
        }
        return false;
    }

    const auto* body = llvm::dyn_cast_or_null<clang::CompoundStmt>(rawBody);
    if (!body) {
        const std::string reason = "outer loop body must be a compound statement";
        bufferRegionPlan.disableReason = reason;
        if (disableReason) {
            *disableReason = reason;
        }
        return false;
    }

    const auto& SM = dacppFile->getContext()->getSourceManager();
    int exprCountInOuterLoop = 0;
    for (const auto* candidate : dacppFile->dacExprs) {
        if (candidate && rangeContains(SM, outerLoop->getSourceRange(),
                                       candidate->getSourceRange())) {
            ++exprCountInOuterLoop;
        }
    }
    if (exprCountInOuterLoop != 1) {
        const std::string reason =
            "outer loop contains multiple DAC expressions";
        bufferRegionPlan.disableReason = reason;
        if (disableReason) {
            *disableReason = reason;
        }
        return false;
    }

    int dacStmtIndex = -1;
    for (std::size_t idx = 0; idx < body->size(); ++idx) {
        const auto* stmt = body->body_begin()[idx];
        if (stmt && rangeContains(SM, stmt->getSourceRange(),
                                  dacExpr->getSourceRange())) {
            dacStmtIndex = static_cast<int>(idx);
            break;
        }
    }
    if (dacStmtIndex < 0) {
        const std::string reason =
            "failed to locate top-level DAC expression statement";
        bufferRegionPlan.disableReason = reason;
        if (disableReason) {
            *disableReason = reason;
        }
        return false;
    }
    if (dacStmtIndex != 0) {
        const std::string reason =
            "only loops with DAC expression as the first top-level statement are supported";
        bufferRegionPlan.disableReason = reason;
        if (disableReason) {
            *disableReason = reason;
        }
        return false;
    }

    std::vector<std::pair<std::string, std::string>> capturedVars;
    {
        clang::ASTContext& Ctx = *dacppFile->getContext();
        clang::SourceManager& SourceMgr = Ctx.getSourceManager();
        llvm::SmallVector<const clang::VarDecl*, 16> usedVars;
        std::function<void(const clang::Stmt*)> collectRefs =
            [&](const clang::Stmt* S) {
                if (!S) {
                    return;
                }
                if (auto* DRE = llvm::dyn_cast<clang::DeclRefExpr>(S)) {
                    if (auto* VD = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl())) {
                        usedVars.push_back(VD);
                    }
                }
                for (const clang::Stmt* child : S->children()) {
                    collectRefs(child);
                }
            };
        collectRefs(outerLoop);

        llvm::SmallPtrSet<const clang::VarDecl*, 16> varsDeclaredInside;
        auto collectInnerDecls = [&](const clang::Stmt* init,
                                     const clang::Stmt* loopBody) {
            if (auto* DS = llvm::dyn_cast_or_null<const clang::DeclStmt>(init)) {
                for (auto it = DS->decl_begin(); it != DS->decl_end(); ++it) {
                    if (auto* VD = llvm::dyn_cast<clang::VarDecl>(*it)) {
                        varsDeclaredInside.insert(VD);
                    }
                }
            }

            std::function<void(const clang::Stmt*)> scan =
                [&](const clang::Stmt* S) {
                    if (!S) {
                        return;
                    }
                    if (auto* DS = llvm::dyn_cast<const clang::DeclStmt>(S)) {
                        for (auto it = DS->decl_begin(); it != DS->decl_end();
                             ++it) {
                            if (auto* VD = llvm::dyn_cast<clang::VarDecl>(*it)) {
                                varsDeclaredInside.insert(VD);
                            }
                        }
                    }
                    for (const clang::Stmt* child : S->children()) {
                        scan(child);
                    }
                };

            scan(loopBody);
        };

        const clang::Stmt* loopInit = nullptr;
        const clang::Stmt* loopBody = outerLoop;
        if (const auto* FS = llvm::dyn_cast<clang::ForStmt>(outerLoop)) {
            loopInit = FS->getInit();
            loopBody = FS->getBody();
        } else if (const auto* WS = llvm::dyn_cast<clang::WhileStmt>(outerLoop)) {
            loopBody = WS->getBody();
        }
        collectInnerDecls(loopInit, loopBody);

        llvm::SmallPtrSet<const clang::VarDecl*, 16> uniqueVars;
        for (const clang::VarDecl* VD : usedVars) {
            if (!VD || varsDeclaredInside.count(VD) || VD->isFileVarDecl()) {
                continue;
            }
            if (SourceMgr.isBeforeInTranslationUnit(VD->getBeginLoc(),
                                                    outerLoop->getBeginLoc())) {
                uniqueVars.insert(VD);
            }
        }

        for (const clang::VarDecl* VD : uniqueVars) {
            std::string name = VD->getNameAsString();
            std::string type = VD->getType().getAsString();
            if (type == "_Bool") {
                type = "bool";
            }
            capturedVars.emplace_back(name, type);
        }
    }

    std::set<std::string> shellVarNames;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        shellVarNames.insert(shell->getParam(paramIdx)->getName());
    }

    std::vector<const clang::Stmt*> siblingStmts;
    for (std::size_t idx = static_cast<std::size_t>(dacStmtIndex + 1);
         idx < body->size(); ++idx) {
        const auto* stmt = body->body_begin()[idx];
        if (!stmt) {
            const std::string reason =
                "unexpected null statement after DAC expression";
            bufferRegionPlan.disableReason = reason;
            if (disableReason) {
                *disableReason = reason;
            }
            return false;
        }

        const auto* siblingFor = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
        if (siblingFor) {
            if (!isSupportedRegionLoop(siblingFor, dacppFile->getContext())) {
                const std::string reason =
                    "sibling loop contains unsupported control flow";
                bufferRegionPlan.disableReason = reason;
                if (disableReason) {
                    *disableReason = reason;
                }
                return false;
            }
        } else {
            const std::string stmtText = Lexer::getSourceText(
                CharSourceRange::getTokenRange(stmt->getSourceRange()),
                dacppFile->getContext()->getSourceManager(),
                dacppFile->getContext()->getLangOpts()).str();
            if (stmtText.empty() || stmtText.find("<->") != std::string::npos) {
                const std::string reason =
                    "sibling statement contains unsupported syntax";
                bufferRegionPlan.disableReason = reason;
                if (disableReason) {
                    *disableReason = reason;
                }
                return false;
            }
            static const std::vector<std::string> kRejectedKeywordsForStmt = {
                "while", "switch", "return", "break", "continue", "goto"};
            for (const auto& keyword : kRejectedKeywordsForStmt) {
                if (containsWord(stmtText, keyword)) {
                    const std::string reason =
                        "sibling statement contains unsupported control flow";
                    bufferRegionPlan.disableReason = reason;
                    if (disableReason) {
                        *disableReason = reason;
                    }
                    return false;
                }
            }
        }

        siblingStmts.push_back(stmt);
    }

    std::set<std::string> allNonShellVarNames;
    for (const auto* stmt : siblingStmts) {
        const std::string stmtText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(stmt->getSourceRange()),
            dacppFile->getContext()->getSourceManager(),
            dacppFile->getContext()->getLangOpts()).str();
        for (const auto& captured : capturedVars) {
            if (shellVarNames.count(captured.first) != 0) {
                continue;
            }
            if (containsWord(stmtText, captured.first)) {
                allNonShellVarNames.insert(captured.first);
            }
        }
    }
    std::vector<std::pair<std::string, std::string>> capturedNonShellVars;
    for (const auto& captured : capturedVars) {
        if (allNonShellVarNames.count(captured.first) != 0) {
            capturedNonShellVars.push_back(captured);
        }
    }

    bufferRegionPlan.enabled = true;
    bufferRegionPlan.exprIndex = -1;
    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        auto* expr = dacppFile->getExpression(exprIdx);
        if (expr && expr->getDacExpr() == dacExpr) {
            bufferRegionPlan.exprIndex = exprIdx;
            break;
        }
    }
    bufferRegionPlan.parentFunction = dacppFile->node;
    bufferRegionPlan.outerLoop = outerLoop;
    bufferRegionPlan.outerFor = llvm::dyn_cast<clang::ForStmt>(outerLoop);
    bufferRegionPlan.dacExpr = dacExpr;
    bufferRegionPlan.siblingStmts = std::move(siblingStmts);
    bufferRegionPlan.capturedVars = std::move(capturedVars);
    bufferRegionPlan.capturedNonShellVars = std::move(capturedNonShellVars);
    bufferRegionPlan.disableReason.clear();
    if (disableReason) {
        disableReason->clear();
    }
    return true;
}

/**
 * 存储头文件信息类实现
 */
dacppTranslator::HeaderFile::HeaderFile() {
}

dacppTranslator::HeaderFile::HeaderFile(std::string name) {
    this->name = name;
}

void dacppTranslator::HeaderFile::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::HeaderFile::getName() {
    return name;
}


/**
 * 存储命名空间信息类实现
 */
dacppTranslator::NameSpace::NameSpace() {
}

dacppTranslator::NameSpace::NameSpace(std::string name) {
    this->name = name;
}

void dacppTranslator::NameSpace::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::NameSpace::getName() {
    return name;
}


/**
 * 存储数据关联计算表达式信息类实现
 */
dacppTranslator::Expression::Expression() {
}

void dacppTranslator::Expression::setShell(Shell* shell) {
    this->shell = shell;
}

dacppTranslator::Shell* dacppTranslator::Expression::getShell() {
    return shell;
}

void dacppTranslator::Expression::setCalc(Calc* calc) {
    this->calc = calc;
}

dacppTranslator::Calc* dacppTranslator::Expression::getCalc() {
    return calc;
}

void dacppTranslator::Expression::setDacExpr(const clang::BinaryOperator* dacExpr) {
    this->dacExpr = dacExpr;
}

const clang::BinaryOperator* dacppTranslator::Expression::getDacExpr() {
    return dacExpr;
}

bool dacppTranslator::Expression::shellLHS_p(const BinaryOperator *dacExpr) {
  bool found_p = false;
  Expr *LHS;
  const CXXBindTemporaryExpr *BTE;
  const CallExpr *Call;
  const ImplicitCastExpr *ICE;
  const DeclRefExpr *DRE;
  const FunctionDecl *FD;
  do {
    LHS = dacExpr->getLHS();
    BTE = dyn_cast<CXXBindTemporaryExpr>(LHS);
    if (!BTE)
      break;

    Call = dyn_cast<CallExpr>(BTE->getSubExpr());
    if (!Call)
      break;

    ICE = dyn_cast<ImplicitCastExpr>(Call->getCallee());
    if (!ICE)
      break;

    DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
    if (!DRE)
      break;

    FD = dyn_cast<FunctionDecl>(DRE->getDecl());
    if (!FD)
      break;

    if (!FD->getDescribedFunctionTemplate() &&
        !FD->isFunctionTemplateSpecialization()) {
      if (FD->hasAttrs()) {
        const AttrVec &Attrs = FD->getAttrs();
        for (auto *A : Attrs) {
          std::string BSBuf;
          llvm::raw_string_ostream BSStream(BSBuf);
          A->printPretty(BSStream, PrintingPolicy(LangOptions()));
          if (!strcmp(BSBuf.c_str(), "shell")) {
            found_p = true;
            break;
          }
        }
      }
    }
  } while (0);
  return found_p;
}

/**
 * 存储DACPP文件信息类实现
 */
dacppTranslator::DacppFile::DacppFile() {
    setHeaderFile("<sycl/sycl.hpp>");
    setNameSpace("sycl");
}

void dacppTranslator::DacppFile::setHeaderFile(std::string headerFile) {
    headerFiles.push_back(new HeaderFile(headerFile));
}

dacppTranslator::HeaderFile* dacppTranslator::DacppFile::getHeaderFile(int idx) {
    return headerFiles[idx];
}

int dacppTranslator::DacppFile::getNumHeaderFile() {
    return headerFiles.size();
}

void dacppTranslator::DacppFile::setNameSpace(std::string nameSpace) {
    nameSpaces.push_back(new NameSpace(nameSpace));
}

dacppTranslator::NameSpace* dacppTranslator::DacppFile::getNameSpace(int idx) {
    return nameSpaces[idx];
}

int dacppTranslator::DacppFile::getNumNameSpace() {
    return nameSpaces.size();
}

void dacppTranslator::DacppFile::setExpression(const BinaryOperator* dacExpr) {
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacppTranslator::Expression::shellLHS_p (dacExpr) ?  dacExpr->getLHS() : dacExpr->getRHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    // 获取实参的形状
    // 这里的实参目前指的是 DACPP:Tensor
    // TODO 
    // 实参形状这里直接在定义实参的位置找到了实参的初始化列表，在翻译的时候将实参形状硬编码到了函数中，如果多次调用同一函数，如果实参不同，会生成多个SYCL函数
    // Tensor的初始化用到了两个std::vector，如果在生成之后对其进行了push_back，则不能得到正确的形状，会出现bug
    // 如果Tensor初始化列表中的vector是从文件中读取的，也无法获得正确形状，这种情况需要在SYCL文件中把形状修改为软编码，比如Tensor.getShape(idx)
    std::vector<std::vector<int>> shapes(shellCall->getNumArgs());
    for(unsigned int paramsCount = 0; paramsCount < shellCall->getNumArgs(); paramsCount++) {
        Expr* curExpr = shellCall->getArg(paramsCount);
        DeclRefExpr* declRefExpr;
        if(isa<DeclRefExpr>(curExpr)) {
            declRefExpr = dyn_cast<DeclRefExpr>(curExpr);
        } else if(isa<ImplicitCastExpr>(curExpr)) {
            declRefExpr = getNode<DeclRefExpr>(curExpr);
        } else {
            // 带切片的Tensor
            while(getNode<CXXOperatorCallExpr>(curExpr)) {
                curExpr = getNode<CXXOperatorCallExpr>(curExpr);
            }
            int count = 0;
            for(Stmt::child_iterator it = curExpr->child_begin(); it != curExpr->child_end(); it++, count++) {
                if(count != 1) {
                    continue;
                }
                declRefExpr = getNode<DeclRefExpr>(*it);
            }
        }
        int count = 0;
        for(Stmt::child_iterator it = dyn_cast<VarDecl>(declRefExpr->getDecl())->getInit()->child_begin(); it != dyn_cast<VarDecl>(declRefExpr->getDecl())->getInit()->child_end(); it++) {
            if(count != 1) {
                count++;
                continue;
            }
            VarDecl* shapeDecl = dyn_cast<VarDecl>(getNode<DeclRefExpr>(*it)->getDecl());
            InitListExpr* initListExpr = getNode<InitListExpr>(shapeDecl->getInit());
            for(Stmt::child_iterator it = initListExpr->child_begin(); it != initListExpr->child_end(); it++) {
                IntegerLiteral* integer = dyn_cast<IntegerLiteral>(*it);
                shapes[paramsCount].push_back(std::stoi(toString(integer->getValue(), 10, true)));
            }
            count++;
        }
    }
    Shell* shell = new Shell();
    Calc* calc = new Calc();
    Expression* expr = new Expression();
    expr->setDacExpr(dacExpr);
    expr->setShell(shell);
    expr->setCalc(calc);
    shell->setFather(expr);
    calc->setFather(expr);
    shell->parseShell(dacExpr, shapes);
    calc->parseCalc(dacExpr);
    exprs.push_back(expr);
}

dacppTranslator::Expression* dacppTranslator::DacppFile::getExpression(int idx) {
    return exprs[idx];
}

int dacppTranslator::DacppFile::getNumExpression() {
    return exprs.size();
}

void dacppTranslator::DacppFile::setMainFuncLoc(const FunctionDecl* mainFuncLoc) {
    this->mainFuncLoc = mainFuncLoc;
}

const FunctionDecl* dacppTranslator::DacppFile::getMainFuncLoc() {
    return mainFuncLoc;
}
void dacppTranslator::DacppFile::collectVarsFromForStatement() {
    forStatementVars.clear();
    const clang::Stmt* loop = loopStatement ? loopStatement : forStatement;
    if (!loop) {
        return;
    }

    clang::ASTContext &Ctx = *Context;
    clang::SourceManager &SM = Ctx.getSourceManager();

    // --------------------------------------
    // Step 1：收集 for 循环体内所有被引用的变量（DeclRefExpr）
    // --------------------------------------
    llvm::SmallVector<const clang::VarDecl*, 16> usedVars;

    std::function<void(const clang::Stmt*)> collectRefs = [&](const clang::Stmt* S) {
        if (!S) return;

        if (auto* DRE = llvm::dyn_cast<clang::DeclRefExpr>(S)) {
            if (auto* VD = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl())) {
                usedVars.push_back(VD);
            }
        }

        for (const clang::Stmt* child : S->children())
            collectRefs(child);
    };

    collectRefs(loop);

    // --------------------------------------
    // Step 2：收集 for 循环内部声明的所有变量（用于排除）
    // --------------------------------------
    llvm::SmallPtrSet<const clang::VarDecl*, 16> varsDeclaredInside;

    auto collectInnerDecls = [&](const clang::Stmt* init, const clang::Stmt* body) {
        if (auto* DS = llvm::dyn_cast_or_null<const clang::DeclStmt>(init)) {
            for (auto it = DS->decl_begin(); it != DS->decl_end(); ++it)
                if (auto* VD = llvm::dyn_cast<clang::VarDecl>(*it))
                    varsDeclaredInside.insert(VD);
        }

        std::function<void(const clang::Stmt*)> scan = [&](const clang::Stmt* S) {
            if (!S) return;
            if (auto* DS = llvm::dyn_cast<const clang::DeclStmt>(S)) {
                for (auto it = DS->decl_begin(); it != DS->decl_end(); ++it)
                    if (auto* VD = llvm::dyn_cast<clang::VarDecl>(*it))
                        varsDeclaredInside.insert(VD);
            }
            for (auto* c : S->children()) scan(c);
        };

        scan(body);
    };

    const clang::Stmt* loopInit = nullptr;
    const clang::Stmt* loopBody = loop;
    if (const auto* FS = llvm::dyn_cast<clang::ForStmt>(loop)) {
        loopInit = FS->getInit();
        loopBody = FS->getBody();
    } else if (const auto* WS = llvm::dyn_cast<clang::WhileStmt>(loop)) {
        loopBody = WS->getBody();
    }
    collectInnerDecls(loopInit, loopBody);

    // --------------------------------------
    // Step 3：过滤：只保留在 for 外声明、但在 for 内使用的非全局变量
    // --------------------------------------
    llvm::SmallPtrSet<const clang::VarDecl*, 16> uniqueVars;

    for (const clang::VarDecl* VD : usedVars) {
        if (!VD) continue;

        // 排除循环内部声明的变量
        if (varsDeclaredInside.count(VD))
            continue;

        // 排除全局变量
        if (VD->isFileVarDecl())
            continue;

        // 判断声明位置是否在循环语句之前
        clang::SourceLocation declLoc = VD->getBeginLoc();
        clang::SourceLocation forLoc = loop->getBeginLoc();

        if (SM.isBeforeInTranslationUnit(declLoc, forLoc)) {
            uniqueVars.insert(VD);
        }
    }

    // --------------------------------------
    // Step 4：保存变量（名字 + 类型）
    // --------------------------------------
    for (const clang::VarDecl* VD : uniqueVars) {
        std::string name  = VD->getNameAsString();
        std::string type  = VD->getType().getAsString();
        forStatementVars.emplace_back(name, type);
    }
}

//    std::vector<std::pair<std::string, std::string>> forStatementVars; // 记录前文提及的for循环中用到的、但是在for循环外声明的变量的   变量名及其类型,第一个表示变量名，第二个表示变量类型
std::vector<std::pair<std::string, std::string>> dacppTranslator::DacppFile::getForStatementVars() {
    std::vector<std::pair<std::string, std::string>> result = forStatementVars;

    for (auto &var : result) {
        if (var.second == "_Bool") {
            var.second = "bool";
        }
    }

    return result;
}

void dacppTranslator::DacppFile::analyzeBufferRegionPlan() {
    bufferRegionPlan = BufferRegionPlan{};

    const clang::Stmt* outerLoop = this->loopStatement ? this->loopStatement :
        static_cast<const clang::Stmt*>(this->forStatement);
    if (!this->Context || !outerLoop || this->exprs.size() != 1 ||
        this->dacExprs.size() != 1) {
        bufferRegionPlan.disableReason =
            "requires exactly one DAC expression inside one outer loop";
        return;
    }

    const clang::Stmt* rawBody = nullptr;
    if (const auto* FS = llvm::dyn_cast<clang::ForStmt>(outerLoop)) {
        rawBody = FS->getBody();
    } else if (const auto* WS = llvm::dyn_cast<clang::WhileStmt>(outerLoop)) {
        rawBody = WS->getBody();
    } else {
        bufferRegionPlan.disableReason = "outer loop must be for or while";
        return;
    }

    const auto* body = llvm::dyn_cast_or_null<clang::CompoundStmt>(rawBody);
    if (!body) {
        bufferRegionPlan.disableReason = "outer loop body must be a compound statement";
        return;
    }

    auto* expr = this->exprs.front();
    if (!expr || !expr->getShell() || !expr->getDacExpr()) {
        bufferRegionPlan.disableReason = "missing DAC expression";
        return;
    }

    buildBufferRegionPlanForExprImpl(this, expr->getShell(), expr->getDacExpr(),
                                     outerLoop, bufferRegionPlan,
                                     &bufferRegionPlan.disableReason);
}

namespace dacppTranslator {
namespace mpi_rewriter {

bool buildBufferRegionPlanForDacExpr(
    DacppFile* dacppFile,
    Shell* shell,
    const clang::BinaryOperator* dacExpr,
    BufferRegionPlan& plan,
    std::string* disableReason) {
    if (!dacppFile || !shell || !dacExpr) {
        const std::string reason = "missing loop or DAC expression";
        plan = BufferRegionPlan{};
        plan.disableReason = reason;
        if (disableReason) {
            *disableReason = reason;
        }
        return false;
    }

    const clang::Stmt* outerLoop = nullptr;
    for (const auto& site : dacppFile->getMPIStencilSites()) {
        if (site.dacExpr == dacExpr) {
            outerLoop = site.outerLoop;
            break;
        }
    }
    if (!outerLoop) {
        const std::string reason = "missing stencil outer loop";
        plan = BufferRegionPlan{};
        plan.disableReason = reason;
        if (disableReason) {
            *disableReason = reason;
        }
        return false;
    }

    return buildBufferRegionPlanForExprImpl(dacppFile, shell, dacExpr, outerLoop,
                                            plan, disableReason);
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
