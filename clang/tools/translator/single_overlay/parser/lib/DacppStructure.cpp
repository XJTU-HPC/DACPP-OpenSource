
#include <string>

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

using namespace clang;

class InnerForCollector : public clang::RecursiveASTVisitor<InnerForCollector> {
public:
    const clang::ForStmt* outer;
    std::vector<const clang::ForStmt*> results;

    InnerForCollector(const clang::ForStmt* outerLoop)
        : outer(outerLoop) {}

    bool VisitForStmt(const clang::ForStmt* FS) {
        if (FS == outer)
            return true;

        results.push_back(FS);
        return true;
    }
};

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

    Expr* dacExprLHS = dacppTranslator::Expression::shellLHS_p (dacExpr) ?  dacExpr->getLHS() : dacExpr->getRHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);

    std::vector<std::vector<int>> shapes(shellCall->getNumArgs());
    for(unsigned int paramsCount = 0; paramsCount < shellCall->getNumArgs(); paramsCount++) {
        Expr* curExpr = shellCall->getArg(paramsCount);
        DeclRefExpr* declRefExpr;
        if(isa<DeclRefExpr>(curExpr)) {
            declRefExpr = dyn_cast<DeclRefExpr>(curExpr);
        } else if(isa<ImplicitCastExpr>(curExpr)) {
            declRefExpr = getNode<DeclRefExpr>(curExpr);
        } else {

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

    clang::ASTContext &Ctx = *Context;
    clang::SourceManager &SM = Ctx.getSourceManager();

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

    collectRefs(forStatement->getBody());

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

    collectInnerDecls(forStatement->getInit(), forStatement->getBody());

    llvm::SmallPtrSet<const clang::VarDecl*, 16> uniqueVars;

    for (const clang::VarDecl* VD : usedVars) {
        if (!VD) continue;

        if (varsDeclaredInside.count(VD))
            continue;

        if (VD->isFileVarDecl())
            continue;

        clang::SourceLocation declLoc = VD->getBeginLoc();
        clang::SourceLocation forLoc = forStatement->getBeginLoc();

        if (SM.isBeforeInTranslationUnit(declLoc, forLoc)) {
            uniqueVars.insert(VD);
        }
    }

    for (const clang::VarDecl* VD : uniqueVars) {
        std::string name  = VD->getNameAsString();
        std::string type  = VD->getType().getAsString();
        forStatementVars.emplace_back(name, type);
    }
}

std::vector<std::pair<std::string, std::string>> dacppTranslator::DacppFile::getForStatementVars() {
    std::vector<std::pair<std::string, std::string>> result = forStatementVars;

    for (auto &var : result) {
        if (var.second == "_Bool") {
            var.second = "bool";
        }
    }

    return result;
}

void dacppTranslator::DacppFile::collectInnerForStmts() {

    if (!this->forStatement) {
        llvm::errs() << "[DacppFile] collectInnerForStmts failed: forStatement is null.\n";
        return;
    }
    if (!this->Context) {
        llvm::errs() << "[DacppFile] collectInnerForStmts failed: Context is null.\n";
        return;
    }

    innerForStatements.clear();

    InnerForCollector collector(this->forStatement);

    collector.TraverseStmt(const_cast<clang::Stmt*>(this->forStatement->getBody()));

    innerForStatements = collector.results;
}
