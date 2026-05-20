#include "mpi/operator_resident/LoopLocalStencilOwnerLoop.h"

#include <algorithm>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

std::string loopVarName(const clang::ForStmt* forStmt);

std::string getExprSourceText(const clang::Expr* expr, DacppFile* dacppFile) {
    if (!expr || !dacppFile || !dacppFile->getContext()) {
        return "";
    }
    const auto& context = *dacppFile->getContext();
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(expr->getSourceRange()),
               context.getSourceManager(), context.getLangOpts())
        .str();
}

const clang::Expr* stripExpr(const clang::Expr* expr) {
    while (expr) {
        expr = expr->IgnoreParenImpCasts();
        if (const auto* cleanup =
                llvm::dyn_cast<clang::ExprWithCleanups>(expr)) {
            expr = cleanup->getSubExpr();
            continue;
        }
        if (const auto* materialized =
                llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
            expr = materialized->getSubExpr();
            continue;
        }
        if (const auto* temporary =
                llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
            expr = temporary->getSubExpr();
            continue;
        }
        return expr;
    }
    return nullptr;
}

bool evaluateInt64Constant(const clang::Expr* expr,
                           clang::ASTContext* context,
                           int64_t* value) {
    expr = stripExpr(expr);
    if (!expr || !context || !value) {
        return false;
    }
    if (const auto* literal = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
        *value = literal->getValue().getSExtValue();
        return true;
    }
    if (!expr->getType()->isIntegerType()) {
        return false;
    }
    clang::Expr::EvalResult evalResult;
    if (!expr->EvaluateAsInt(evalResult, *context) ||
        !evalResult.Val.isInt()) {
        return false;
    }
    *value = evalResult.Val.getInt().getSExtValue();
    return true;
}

bool isDeclRefToName(const clang::Expr* expr, const std::string& name) {
    expr = stripExpr(expr);
    const auto* declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
    return declRef && declRef->getDecl() &&
           declRef->getDecl()->getNameAsString() == name;
}

const clang::ValueDecl* valueDeclFromDeclRef(const clang::Expr* expr) {
    expr = stripExpr(expr);
    const auto* declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
    return declRef ? declRef->getDecl() : nullptr;
}

bool isDeclRefToExpectedValue(const clang::Expr* expr,
                              const std::string& fallbackName,
                              const clang::ValueDecl* expectedDecl) {
    const auto* actualDecl = valueDeclFromDeclRef(expr);
    if (!actualDecl) {
        return false;
    }
    if (expectedDecl) {
        return actualDecl == expectedDecl;
    }
    return !fallbackName.empty() &&
           actualDecl->getNameAsString() == fallbackName;
}

bool isIntegerLiteralValue(const clang::Expr* expr, int value) {
    expr = stripExpr(expr);
    const auto* literal =
        llvm::dyn_cast_or_null<clang::IntegerLiteral>(expr);
    return literal && literal->getValue().getSExtValue() == value;
}

bool isLoopVarPlusOne(const clang::Expr* expr, const std::string& loopVar) {
    expr = stripExpr(expr);
    const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(expr);
    return binary && binary->getOpcode() == clang::BO_Add &&
           isDeclRefToName(binary->getLHS(), loopVar) &&
           isIntegerLiteralValue(binary->getRHS(), 1);
}

bool isLoopVarMinusOne(const clang::Expr* expr, const std::string& loopVar) {
    expr = stripExpr(expr);
    const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(expr);
    return binary && binary->getOpcode() == clang::BO_Sub &&
           isDeclRefToName(binary->getLHS(), loopVar) &&
           isIntegerLiteralValue(binary->getRHS(), 1);
}

class DeclRefNameVisitor
    : public clang::RecursiveASTVisitor<DeclRefNameVisitor> {
public:
    explicit DeclRefNameVisitor(std::string targetName)
        : targetName_(std::move(targetName)) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* declRef) {
        if (declRef && declRef->getDecl() &&
            declRef->getDecl()->getNameAsString() == targetName_) {
            found_ = true;
            return false;
        }
        return true;
    }

    bool found() const { return found_; }

private:
    std::string targetName_;
    bool found_ = false;
};

bool exprReferencesVar(const clang::Expr* expr, const std::string& name) {
    if (!expr || name.empty()) {
        return false;
    }
    DeclRefNameVisitor visitor(name);
    visitor.TraverseStmt(const_cast<clang::Expr*>(expr));
    return visitor.found();
}

bool stmtContainsCoutExpr(const clang::Stmt* stmt) {
    if (!stmt) {
        return false;
    }
    if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
        return dre->getDecl() && dre->getDecl()->getNameAsString() == "cout";
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (stmtContainsCoutExpr(child)) {
            return true;
        }
    }
    return false;
}

bool stmtContainsPrintCall(const clang::Stmt* stmt) {
    if (!stmt) {
        return false;
    }
    if (const auto* call =
            llvm::dyn_cast<clang::CXXMemberCallExpr>(stmt)) {
        const clang::CXXMethodDecl* method = call->getMethodDecl();
        if (method && method->getNameAsString() == "print") {
            return true;
        }
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (stmtContainsPrintCall(child)) {
            return true;
        }
    }
    return false;
}

bool stmtContainsRootObservableOutput(const clang::Stmt* stmt) {
    return stmtContainsCoutExpr(stmt) || stmtContainsPrintCall(stmt);
}

const clang::Stmt* topLevelStmtForExpr(DacppFile* dacppFile,
                                       const clang::CompoundStmt* loopBody,
                                       const clang::BinaryOperator* dacExpr) {
    if (!dacppFile || !dacppFile->getContext() || !loopBody || !dacExpr) {
        return nullptr;
    }
    clang::DynTypedNode current = clang::DynTypedNode::create(*dacExpr);
    const clang::Stmt* currentStmt = dacExpr;
    while (true) {
        auto parents = dacppFile->getContext()->getParents(current);
        if (parents.empty()) {
            break;
        }
        const auto& parent = parents[0];
        if (const auto* parentCompound =
                parent.get<clang::CompoundStmt>()) {
            if (parentCompound == loopBody) {
                return currentStmt;
            }
        }
        if (parent.get<clang::FunctionDecl>()) {
            break;
        }
        if (const auto* parentStmt = parent.get<clang::Stmt>()) {
            currentStmt = parentStmt;
        }
        current = parent;
    }
    return nullptr;
}

std::vector<const clang::Stmt*> collectTopLevelBodyStmts(
    const clang::CompoundStmt* loopBody) {
    std::vector<const clang::Stmt*> result;
    if (!loopBody) {
        return result;
    }
    for (const clang::Stmt* stmt : loopBody->body()) {
        result.push_back(stmt);
    }
    return result;
}

std::string singleDeclName(const clang::Stmt* stmt) {
    const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(stmt);
    if (!declStmt || !declStmt->isSingleDecl()) {
        return "";
    }
    const auto* varDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    return varDecl ? varDecl->getNameAsString() : "";
}

const clang::Expr* singleDeclInit(const clang::Stmt* stmt) {
    const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(stmt);
    if (!declStmt || !declStmt->isSingleDecl()) {
        return nullptr;
    }
    const auto* varDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    return varDecl ? varDecl->getInit() : nullptr;
}

bool isForwardUnitLoopFromInteger(const clang::ForStmt* forStmt,
                                  const std::string& loopVar,
                                  int startValue) {
    if (!forStmt || loopVar.empty()) {
        return false;
    }
    const auto* declStmt =
        llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
    if (!declStmt || !declStmt->isSingleDecl()) {
        return false;
    }
    const auto* varDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    if (!varDecl || varDecl->getNameAsString() != loopVar ||
        !isIntegerLiteralValue(varDecl->getInit(), startValue)) {
        return false;
    }

    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
    if (!cond || !isDeclRefToName(cond->getLHS(), loopVar) ||
        (cond->getOpcode() != clang::BO_LT &&
         cond->getOpcode() != clang::BO_LE)) {
        return false;
    }

    const clang::Expr* inc = stripExpr(forStmt->getInc());
    const auto* unary = llvm::dyn_cast_or_null<clang::UnaryOperator>(inc);
    return unary &&
           (unary->getOpcode() == clang::UO_PostInc ||
            unary->getOpcode() == clang::UO_PreInc) &&
           isDeclRefToName(unary->getSubExpr(), loopVar);
}

bool isForwardUnitLoopFromZero(const clang::ForStmt* forStmt,
                               const std::string& loopVar) {
    return isForwardUnitLoopFromInteger(forStmt, loopVar, 0);
}

bool isForwardUnitLoopFromOne(const clang::ForStmt* forStmt,
                              const std::string& loopVar) {
    return isForwardUnitLoopFromInteger(forStmt, loopVar, 1);
}

struct TensorSliceExpr {
    std::string baseName;
    const clang::ValueDecl* baseDecl = nullptr;
    const clang::Expr* firstIndex = nullptr;
    const clang::Expr* secondIndex = nullptr;
};

struct TensorSubscriptExpr {
    std::string baseName;
    const clang::Expr* index = nullptr;
};

bool findInitListExpr(const clang::Stmt* stmt,
                      const clang::InitListExpr** result) {
    if (!stmt || !result) {
        return false;
    }
    if (const auto* init = llvm::dyn_cast<clang::InitListExpr>(stmt)) {
        *result = init;
        return true;
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (findInitListExpr(child, result)) {
            return true;
        }
    }
    return false;
}

bool extractTwoDimTensorSlice(const clang::Expr* expr,
                              TensorSliceExpr* slice) {
    expr = stripExpr(expr);
    const auto* secondSubscript =
        llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr);
    if (!secondSubscript ||
        secondSubscript->getOperator() != clang::OO_Subscript ||
        secondSubscript->getNumArgs() < 2) {
        return false;
    }
    const clang::Expr* firstSubscriptExpr = stripExpr(secondSubscript->getArg(0));
    const auto* firstSubscript =
        llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(
            firstSubscriptExpr);
    if (!firstSubscript ||
        firstSubscript->getOperator() != clang::OO_Subscript ||
        firstSubscript->getNumArgs() < 2) {
        return false;
    }
    const clang::Expr* baseExpr = stripExpr(firstSubscript->getArg(0));
    const auto* baseRef =
        llvm::dyn_cast_or_null<clang::DeclRefExpr>(baseExpr);
    if (!baseRef || !baseRef->getDecl()) {
        return false;
    }
    slice->baseName = baseRef->getDecl()->getNameAsString();
    slice->baseDecl = baseRef->getDecl();
    slice->firstIndex = firstSubscript->getArg(1);
    slice->secondIndex = secondSubscript->getArg(1);
    return true;
}

bool firstIndexForTensorUse(const clang::Expr* expr,
                            const std::string& tensorName,
                            clang::ASTContext* context,
                            int64_t* row) {
    TensorSliceExpr slice;
    const clang::Expr* firstIndex = nullptr;
    if (extractTwoDimTensorSlice(expr, &slice) &&
        slice.baseName == tensorName) {
        firstIndex = slice.firstIndex;
    } else {
        expr = stripExpr(expr);
        const auto* opCall =
            llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr);
        if (!opCall || opCall->getOperator() != clang::OO_Subscript ||
            opCall->getNumArgs() < 2 ||
            !isDeclRefToName(opCall->getArg(0), tensorName)) {
            return false;
        }
        firstIndex = opCall->getArg(1);
    }
    int64_t candidate = -1;
    if (!evaluateInt64Constant(firstIndex, context, &candidate) ||
        candidate < 0) {
        return false;
    }
    if (row) {
        *row = candidate;
    }
    return true;
}

bool extractTensorSliceFromVectorInit(const clang::Stmt* stmt,
                                      TensorSliceExpr* slice) {
    const clang::Expr* init = stripExpr(singleDeclInit(stmt));
    if (!init || !slice) {
        return false;
    }
    if (extractTwoDimTensorSlice(init, slice)) {
        return true;
    }
    const auto* construct =
        llvm::dyn_cast_or_null<clang::CXXConstructExpr>(init);
    return construct && construct->getNumArgs() == 1 &&
           extractTwoDimTensorSlice(construct->getArg(0), slice);
}

bool extractOneDimTensorSubscript(const clang::Expr* expr,
                                  TensorSubscriptExpr* subscript) {
    expr = stripExpr(expr);
    const auto* opCall =
        llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr);
    if (!opCall || opCall->getOperator() != clang::OO_Subscript ||
        opCall->getNumArgs() < 2) {
        return false;
    }
    const clang::Expr* baseExpr = stripExpr(opCall->getArg(0));
    const auto* baseRef =
        llvm::dyn_cast_or_null<clang::DeclRefExpr>(baseExpr);
    if (!baseRef || !baseRef->getDecl()) {
        return false;
    }
    subscript->baseName = baseRef->getDecl()->getNameAsString();
    subscript->index = opCall->getArg(1);
    return true;
}

bool isEmptyInitializerListExpr(const clang::Expr* expr) {
    expr = stripExpr(expr);
    if (const auto* construct =
            llvm::dyn_cast_or_null<clang::CXXConstructExpr>(expr)) {
        return construct->getNumArgs() == 0 &&
               construct->getType().getAsString().find("initializer_list") !=
                   std::string::npos;
    }
    const clang::InitListExpr* init = nullptr;
    return findInitListExpr(expr, &init) && init && init->getNumInits() == 0;
}

bool isOwnerRangeOneToExpectedEnd(const clang::Expr* expr,
                                  const std::string& expectedEndName,
                                  const clang::ValueDecl* expectedEndDecl) {
    const clang::InitListExpr* init = nullptr;
    if (!findInitListExpr(expr, &init) || !init || init->getNumInits() != 2) {
        return false;
    }
    return isIntegerLiteralValue(init->getInit(0), 1) &&
           isDeclRefToExpectedValue(init->getInit(1), expectedEndName,
                                    expectedEndDecl);
}

struct OwnerMatrixShapeProof {
    std::string rowInteriorEndName;
    const clang::ValueDecl* rowInteriorEndDecl = nullptr;
    std::string timeStepCountName;
    const clang::ValueDecl* timeStepCountDecl = nullptr;
};

bool extractDeclPlusOne(const clang::Expr* expr,
                        std::string* name,
                        const clang::ValueDecl** decl) {
    expr = stripExpr(expr);
    const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(expr);
    if (!binary || binary->getOpcode() != clang::BO_Add) {
        return false;
    }
    const clang::Expr* declExpr = nullptr;
    if (isIntegerLiteralValue(binary->getLHS(), 1)) {
        declExpr = binary->getRHS();
    } else if (isIntegerLiteralValue(binary->getRHS(), 1)) {
        declExpr = binary->getLHS();
    }
    const auto* valueDecl = valueDeclFromDeclRef(declExpr);
    if (!valueDecl) {
        return false;
    }
    if (name) {
        *name = valueDecl->getNameAsString();
    }
    if (decl) {
        *decl = valueDecl;
    }
    return true;
}

bool isExpectedValueMinusOne(const clang::Expr* expr,
                             const std::string& expectedName,
                             const clang::ValueDecl* expectedDecl) {
    expr = stripExpr(expr);
    const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(expr);
    return binary && binary->getOpcode() == clang::BO_Sub &&
           isDeclRefToExpectedValue(binary->getLHS(), expectedName,
                                    expectedDecl) &&
           isIntegerLiteralValue(binary->getRHS(), 1);
}

bool extractOwnerMatrixShapeProof(const clang::ValueDecl* ownerDecl,
                                  OwnerMatrixShapeProof* proof) {
    const auto* varDecl = llvm::dyn_cast_or_null<clang::VarDecl>(ownerDecl);
    if (!varDecl || !varDecl->getInit() || !proof) {
        return false;
    }
    const clang::InitListExpr* shapeInit = nullptr;
    if (!findInitListExpr(varDecl->getInit(), &shapeInit) || !shapeInit ||
        shapeInit->getNumInits() != 2) {
        return false;
    }
    OwnerMatrixShapeProof candidate;
    if (!extractDeclPlusOne(shapeInit->getInit(0),
                            &candidate.rowInteriorEndName,
                            &candidate.rowInteriorEndDecl) ||
        !extractDeclPlusOne(shapeInit->getInit(1),
                            &candidate.timeStepCountName,
                            &candidate.timeStepCountDecl)) {
        return false;
    }
    *proof = candidate;
    return true;
}

bool hasOwnerTimeLoopBound(const clang::ForStmt* forStmt,
                           const std::string& loopVar,
                           const OwnerMatrixShapeProof& proof) {
    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt
                                                          ? forStmt->getCond()
                                                          : nullptr);
    return cond && cond->getOpcode() == clang::BO_LE &&
           isDeclRefToName(cond->getLHS(), loopVar) &&
           isExpectedValueMinusOne(cond->getRHS(),
                                   proof.timeStepCountName,
                                   proof.timeStepCountDecl);
}

bool hasOwnerWritebackLoopBound(const clang::ForStmt* forStmt,
                                const std::string& loopVar,
                                const OwnerMatrixShapeProof& proof) {
    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt
                                                          ? forStmt->getCond()
                                                          : nullptr);
    return cond && cond->getOpcode() == clang::BO_LE &&
           isDeclRefToName(cond->getLHS(), loopVar) &&
           isExpectedValueMinusOne(cond->getRHS(),
                                   proof.rowInteriorEndName,
                                   proof.rowInteriorEndDecl);
}

bool isReaderSliceStmt(const clang::Stmt* stmt,
                       const std::string& readerName,
                       const std::string& loopVar,
                       std::string* ownerName,
                       const clang::ValueDecl** ownerDecl) {
    if (singleDeclName(stmt) != readerName) {
        return false;
    }
    TensorSliceExpr slice;
    if (!extractTensorSliceFromVectorInit(stmt, &slice) ||
        slice.baseName.empty() ||
        !isEmptyInitializerListExpr(slice.firstIndex) ||
        !isDeclRefToName(slice.secondIndex, loopVar)) {
        return false;
    }
    if (ownerName) {
        *ownerName = slice.baseName;
    }
    if (ownerDecl) {
        *ownerDecl = slice.baseDecl;
    }
    return true;
}

bool isWriterSliceStmt(const clang::Stmt* stmt,
                       const std::string& writerName,
                       const std::string& ownerName,
                       const std::string& loopVar,
                       const OwnerMatrixShapeProof& proof) {
    if (singleDeclName(stmt) != writerName) {
        return false;
    }
    TensorSliceExpr slice;
    return extractTensorSliceFromVectorInit(stmt, &slice) &&
           slice.baseName == ownerName &&
           isOwnerRangeOneToExpectedEnd(slice.firstIndex,
                                        proof.rowInteriorEndName,
                                        proof.rowInteriorEndDecl) &&
           isLoopVarPlusOne(slice.secondIndex, loopVar);
}

bool isAnyWriterSliceFromOwnerAtNextStep(const clang::Stmt* stmt,
                                         const std::string& ownerName,
                                         const std::string& loopVar) {
    TensorSliceExpr slice;
    return extractTensorSliceFromVectorInit(stmt, &slice) &&
           slice.baseName == ownerName &&
           isLoopVarPlusOne(slice.secondIndex, loopVar);
}

bool findReaderOwnerNameInBody(const std::vector<const clang::Stmt*>& stmts,
                               const std::string& readerName,
                               const std::string& loopVar,
                               std::string* ownerName,
                               const clang::ValueDecl** ownerDecl) {
    for (const clang::Stmt* stmt : stmts) {
        std::string candidateOwner;
        const clang::ValueDecl* candidateDecl = nullptr;
        if (isReaderSliceStmt(stmt, readerName, loopVar, &candidateOwner,
                              &candidateDecl)) {
            if (ownerName) {
                *ownerName = candidateOwner;
            }
            if (ownerDecl) {
                *ownerDecl = candidateDecl;
            }
            return true;
        }
    }
    return false;
}

int countWriterSlicesFromOwnerAtNextStep(
    const std::vector<const clang::Stmt*>& stmts,
    const std::string& ownerName,
    const std::string& loopVar) {
    int count = 0;
    for (const clang::Stmt* stmt : stmts) {
        if (isAnyWriterSliceFromOwnerAtNextStep(stmt, ownerName, loopVar)) {
            ++count;
        }
    }
    return count;
}

const clang::CXXMemberCallExpr* memberCallExprFromStmt(
    const clang::Stmt* stmt) {
    const auto* expr = llvm::dyn_cast_or_null<clang::Expr>(stmt);
    expr = stripExpr(expr);
    return llvm::dyn_cast_or_null<clang::CXXMemberCallExpr>(expr);
}

bool sourceMatchesRegex(const std::string& text, const std::regex& pattern) {
    std::smatch match;
    return std::regex_search(text, match, pattern);
}

bool scalarPayloadSourceIsCurrentStrictShape(const std::string& text) {
    static const std::regex strictPayload(R"(^[^() ;]+$)");
    return sourceMatchesRegex(text, strictPayload);
}

bool scalarPayloadAstIsCurrentStrictShape(const clang::Expr* expr) {
    expr = stripExpr(expr);
    return llvm::isa_and_nonnull<clang::DeclRefExpr>(expr) ||
           llvm::isa_and_nonnull<clang::IntegerLiteral>(expr) ||
           llvm::isa_and_nonnull<clang::FloatingLiteral>(expr) ||
           llvm::isa_and_nonnull<clang::CXXBoolLiteralExpr>(expr) ||
           llvm::isa_and_nonnull<clang::CharacterLiteral>(expr);
}

bool isVectorStorageDeclStmt(const clang::Stmt* stmt,
                             const std::string& vectorName) {
    if (vectorName.empty() || singleDeclName(stmt) != vectorName) {
        return false;
    }
    const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(stmt);
    const auto* varDecl =
        declStmt && declStmt->isSingleDecl()
            ? llvm::dyn_cast_or_null<clang::VarDecl>(
                  declStmt->getSingleDecl())
            : nullptr;
    if (!varDecl) {
        return false;
    }
    const std::string typeName = varDecl->getType().getAsString();
    if (typeName.find("std::vector") == std::string::npos) {
        return false;
    }
    const clang::Expr* init = stripExpr(varDecl->getInit());
    if (!init) {
        return true;
    }
    const auto* construct =
        llvm::dyn_cast_or_null<clang::CXXConstructExpr>(init);
    return construct && construct->getNumArgs() == 0;
}

std::string implicitObjectName(const clang::CXXMemberCallExpr* call) {
    if (!call) {
        return "";
    }
    const clang::Expr* object = stripExpr(call->getImplicitObjectArgument());
    const auto* ref = llvm::dyn_cast_or_null<clang::DeclRefExpr>(object);
    return ref && ref->getDecl() ? ref->getDecl()->getNameAsString() : "";
}

bool isScalarPayloadStmt(const clang::Stmt* stmt,
                         const std::string& loopVar,
                         std::string* vectorName,
                         std::string* scalarExpr,
                         DacppFile* dacppFile) {
    const auto* call = memberCallExprFromStmt(stmt);
    if (!call || !call->getMethodDecl() ||
        call->getMethodDecl()->getNameAsString() != "push_back" ||
        call->getNumArgs() != 1) {
        return false;
    }
    const clang::Expr* payload = call->getArg(0);
    if (!payload || exprReferencesVar(payload, loopVar) ||
        !scalarPayloadAstIsCurrentStrictShape(payload)) {
        return false;
    }
    const std::string payloadText = getExprSourceText(payload, dacppFile);
    if (payloadText.empty() ||
        !scalarPayloadSourceIsCurrentStrictShape(payloadText)) {
        return false;
    }
    const std::string storageName = implicitObjectName(call);
    if (storageName.empty()) {
        return false;
    }
    if (vectorName) {
        *vectorName = storageName;
    }
    if (scalarExpr) {
        *scalarExpr = payloadText;
    }
    return true;
}

bool isScalarShellArgStmt(const clang::Stmt* stmt,
                          const std::string& scalarName,
                          const std::string& vectorName,
                          DacppFile* dacppFile) {
    if (scalarName.empty() || vectorName.empty() ||
        singleDeclName(stmt) != scalarName) {
        return false;
    }
    const clang::Expr* init = stripExpr(singleDeclInit(stmt));
    const auto* construct =
        llvm::dyn_cast_or_null<clang::CXXConstructExpr>(init);
    if (!construct || construct->getNumArgs() != 1) {
        return false;
    }
    if (getExprSourceText(construct->getArg(0), dacppFile) != vectorName) {
        return false;
    }
    const clang::Expr* arg = stripExpr(construct->getArg(0));
    while (const auto* copyConstruct =
               llvm::dyn_cast_or_null<clang::CXXConstructExpr>(arg)) {
        if (copyConstruct->getNumArgs() != 1) {
            return false;
        }
        arg = stripExpr(copyConstruct->getArg(0));
    }
    return isDeclRefToName(arg, vectorName);
}

bool isOwnerWritebackAssignment(const clang::Expr* expr,
                                const std::string& ownerName,
                                const std::string& writerName,
                                const std::string& outerLoopVar,
                                const std::string& writebackLoopVar) {
    expr = stripExpr(expr);
    const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(expr);
    if (!binary || binary->getOpcode() != clang::BO_Assign) {
        return false;
    }
    TensorSliceExpr lhsSlice;
    if (!extractTwoDimTensorSlice(binary->getLHS(), &lhsSlice) ||
        lhsSlice.baseName != ownerName ||
        !isDeclRefToName(lhsSlice.firstIndex, writebackLoopVar) ||
        !isLoopVarPlusOne(lhsSlice.secondIndex, outerLoopVar)) {
        return false;
    }
    TensorSubscriptExpr rhsSubscript;
    return extractOneDimTensorSubscript(binary->getRHS(), &rhsSubscript) &&
           rhsSubscript.baseName == writerName &&
           isLoopVarMinusOne(rhsSubscript.index, writebackLoopVar);
}

const clang::Expr* singleExprLoopBody(const clang::Stmt* stmt) {
    if (!stmt) {
        return nullptr;
    }
    if (const auto* compound =
            llvm::dyn_cast_or_null<clang::CompoundStmt>(stmt)) {
        if (compound->size() != 1) {
            return nullptr;
        }
        stmt = *compound->body_begin();
    }
    return llvm::dyn_cast_or_null<clang::Expr>(stmt);
}

bool isOwnerWritebackLoopStmt(const clang::Stmt* stmt,
                              const std::string& ownerName,
                              const std::string& writerName,
                              const std::string& loopVar,
                              const OwnerMatrixShapeProof& proof) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt || !forStmt->getBody()) {
        return false;
    }
    const std::string writebackLoopVar = loopVarName(forStmt);
    if (!isForwardUnitLoopFromOne(forStmt, writebackLoopVar) ||
        !hasOwnerWritebackLoopBound(forStmt, writebackLoopVar, proof)) {
        return false;
    }
    return isOwnerWritebackAssignment(singleExprLoopBody(forStmt->getBody()),
                                      ownerName, writerName, loopVar,
                                      writebackLoopVar);
}

class OwnerMutationVisitor
    : public clang::RecursiveASTVisitor<OwnerMutationVisitor> {
public:
    OwnerMutationVisitor(std::string ownerName, std::string loopVar)
        : ownerName_(std::move(ownerName)), loopVar_(std::move(loopVar)) {}

    bool VisitBinaryOperator(clang::BinaryOperator* binary) {
        if (!binary || !binary->isAssignmentOp()) {
            return true;
        }
        TensorSliceExpr lhsSlice;
        if (extractTwoDimTensorSlice(binary->getLHS(), &lhsSlice) &&
            lhsSlice.baseName == ownerName_ &&
            isLoopVarPlusOne(lhsSlice.secondIndex, loopVar_)) {
            ++nextStepAssignments_;
        }
        return true;
    }

    int nextStepAssignments() const { return nextStepAssignments_; }

private:
    std::string ownerName_;
    std::string loopVar_;
    int nextStepAssignments_ = 0;
};

int countOwnerNextStepAssignments(const clang::Stmt* stmt,
                                  const std::string& ownerName,
                                  const std::string& loopVar) {
    if (!stmt || ownerName.empty() || loopVar.empty()) {
        return 0;
    }
    OwnerMutationVisitor visitor(ownerName, loopVar);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    return visitor.nextStepAssignments();
}

bool stmtRangeContains(const clang::SourceManager& sourceManager,
                       const clang::Stmt* outer,
                       const clang::Stmt* inner) {
    if (!outer || !inner || outer->getSourceRange().isInvalid() ||
        inner->getSourceRange().isInvalid()) {
        return false;
    }
    const auto beforeOrEqual = [&](clang::SourceLocation lhs,
                                   clang::SourceLocation rhs) {
        return lhs == rhs || sourceManager.isBeforeInTranslationUnit(lhs, rhs);
    };
    return beforeOrEqual(outer->getBeginLoc(), inner->getBeginLoc()) &&
           beforeOrEqual(inner->getEndLoc(), outer->getEndLoc());
}

class FixedOwnerRowPostUseVisitor
    : public clang::RecursiveASTVisitor<FixedOwnerRowPostUseVisitor> {
public:
    FixedOwnerRowPostUseVisitor(std::string ownerName,
                                clang::ASTContext* context)
        : OwnerName(std::move(ownerName)), Context(context) {}

    bool FullUse = false;
    std::string Reason;
    std::set<int64_t> Rows;

    bool TraverseBinaryOperator(clang::BinaryOperator* binary) {
        if (!binary) {
            return true;
        }
        if (binary->isAssignmentOp()) {
            ++WriteDepth;
            TraverseStmt(binary->getLHS());
            --WriteDepth;
            if (exprMentionsOwner(binary->getLHS())) {
                recordFull("owner tensor written after owner loop");
            }
            TraverseStmt(binary->getRHS());
            return true;
        }
        return clang::RecursiveASTVisitor<
            FixedOwnerRowPostUseVisitor>::TraverseBinaryOperator(binary);
    }

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* binary) {
        if (!binary) {
            return true;
        }
        ++WriteDepth;
        TraverseStmt(binary->getLHS());
        --WriteDepth;
        if (exprMentionsOwner(binary->getLHS())) {
            recordFull("owner tensor compound-written after owner loop");
        }
        TraverseStmt(binary->getRHS());
        return true;
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* unary) {
        if (!unary) {
            return true;
        }
        if (unary->isIncrementDecrementOp()) {
            ++WriteDepth;
            TraverseStmt(unary->getSubExpr());
            --WriteDepth;
            if (exprMentionsOwner(unary->getSubExpr())) {
                recordFull("owner tensor increment/decrement after owner loop");
            }
            return true;
        }
        if (unary->getOpcode() == clang::UO_AddrOf &&
            exprMentionsOwner(unary->getSubExpr())) {
            recordFull("owner tensor address escapes after owner loop");
            return true;
        }
        return clang::RecursiveASTVisitor<
            FixedOwnerRowPostUseVisitor>::TraverseUnaryOperator(unary);
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call) {
            return true;
        }
        const clang::CXXMethodDecl* method = call->getMethodDecl();
        if (method && method->getNameAsString() == "print") {
            int64_t row = -1;
            if (firstIndexForTensorUse(call->getImplicitObjectArgument(),
                                       OwnerName, Context, &row)) {
                Rows.insert(row);
            } else if (exprMentionsOwner(call->getImplicitObjectArgument())) {
                recordFull("owner tensor print is not fixed-row");
            }
            return true;
        }
        if (exprMentionsOwner(call->getImplicitObjectArgument())) {
            recordFull("member call on owner tensor after owner loop");
        }
        return true;
    }

    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* call) {
        if (!call) {
            return true;
        }
        if (call->getOperator() == clang::OO_Subscript && WriteDepth == 0) {
            int64_t row = -1;
            if (firstIndexForTensorUse(call, OwnerName, Context, &row)) {
                Rows.insert(row);
                return true;
            }
        }
        if (call->getOperator() == clang::OO_LessLess) {
            return true;
        }
        if (call->isAssignmentOp()) {
            return true;
        }
        if (exprMentionsOwner(call)) {
            recordFull("owner tensor used in unsupported operator call");
        }
        return true;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (!dre || !dre->getDecl() ||
            dre->getDecl()->getNameAsString() != OwnerName ||
            WriteDepth > 0) {
            return true;
        }
        int64_t row = -1;
        if (isInsideFixedOwnerRow(dre, &row)) {
            Rows.insert(row);
            return true;
        }
        recordFull("owner tensor use is not fixed-row");
        return true;
    }

    bool VisitCallExpr(clang::CallExpr* call) {
        if (!call) {
            return true;
        }
        if (llvm::isa<clang::CXXMemberCallExpr>(call) ||
            llvm::isa<clang::CXXOperatorCallExpr>(call)) {
            return true;
        }
        for (const clang::Expr* arg : call->arguments()) {
            if (exprMentionsOwner(arg)) {
                recordFull("owner tensor passed to function after owner loop");
                break;
            }
        }
        return true;
    }

private:
    std::string OwnerName;
    clang::ASTContext* Context = nullptr;
    int WriteDepth = 0;

    void recordFull(const std::string& reason) {
        if (!FullUse) {
            Reason = reason;
        }
        FullUse = true;
    }

    bool exprIsFixedOwnerRow(const clang::Expr* expr) const {
        int64_t row = -1;
        return firstIndexForTensorUse(expr, OwnerName, Context, &row);
    }

    bool exprMentionsOwner(const clang::Expr* expr) const {
        if (!expr) {
            return false;
        }
        TensorSliceExpr slice;
        if (extractTwoDimTensorSlice(expr, &slice) &&
            slice.baseName == OwnerName) {
            return true;
        }
        if (isDeclRefToName(expr, OwnerName)) {
            return true;
        }
        for (const clang::Stmt* child : expr->children()) {
            if (const auto* childExpr =
                    llvm::dyn_cast_or_null<clang::Expr>(child)) {
                if (exprMentionsOwner(childExpr)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool isInsideFixedOwnerRow(const clang::DeclRefExpr* dre,
                               int64_t* row) const {
        if (!dre || !Context) {
            return false;
        }
        clang::DynTypedNode current = clang::DynTypedNode::create(*dre);
        for (int depth = 0; depth < 10; ++depth) {
            auto parents = Context->getParents(current);
            if (parents.empty()) {
                return false;
            }
            const clang::DynTypedNode& parent = parents[0];
            if (const auto* expr = parent.get<clang::Expr>()) {
                if (firstIndexForTensorUse(expr, OwnerName, Context, row)) {
                    return true;
                }
                current = parent;
                continue;
            }
            return false;
        }
        return false;
    }
};

struct FixedOwnerRowPostUsePlan {
    bool supported = false;
    int64_t row = -1;
    std::string reason;
};

FixedOwnerRowPostUsePlan analyzeFixedOwnerRowPostUse(
    DacppFile* dacppFile,
    const clang::ForStmt* outerLoop,
    const std::string& ownerName) {
    FixedOwnerRowPostUsePlan result;
    if (!dacppFile || !dacppFile->getContext() || !outerLoop ||
        ownerName.empty()) {
        result.reason = "post-use analysis context unavailable";
        return result;
    }
    auto parents = dacppFile->getContext()->getParents(*outerLoop);
    if (parents.empty()) {
        result.reason = "owner loop parent unavailable";
        return result;
    }
    const auto* parentCompound = parents[0].get<clang::CompoundStmt>();
    if (!parentCompound) {
        parentCompound = llvm::dyn_cast_or_null<clang::CompoundStmt>(
            parents[0].get<clang::Stmt>());
    }
    if (!parentCompound) {
        result.reason = "owner loop parent compound unavailable";
        return result;
    }

    bool sawLoop = false;
    std::set<int64_t> rows;
    for (const clang::Stmt* child : parentCompound->body()) {
        if (!child) {
            continue;
        }
        if (child == outerLoop ||
            stmtRangeContains(dacppFile->getContext()->getSourceManager(),
                              child, outerLoop)) {
            sawLoop = true;
            continue;
        }
        if (!sawLoop) {
            continue;
        }
        FixedOwnerRowPostUseVisitor visitor(ownerName,
                                            dacppFile->getContext());
        visitor.TraverseStmt(const_cast<clang::Stmt*>(child));
        if (visitor.FullUse) {
            result.reason = visitor.Reason.empty()
                                ? "unknown owner tensor post-loop use"
                                : visitor.Reason;
            return result;
        }
        if (!visitor.Rows.empty() &&
            !stmtContainsRootObservableOutput(child)) {
            result.reason =
                "fixed owner row read is not root-observable output";
            return result;
        }
        rows.insert(visitor.Rows.begin(), visitor.Rows.end());
    }
    if (rows.size() != 1) {
        result.reason = rows.empty() ? "no fixed owner row post-loop use"
                                     : "multiple fixed owner rows post-loop";
        return result;
    }
    result.supported = true;
    result.row = *rows.begin();
    result.reason = "fixed owner row post-loop use";
    return result;
}

struct OwnerLoopBodyShape {
    const clang::Stmt* writerSliceStmt = nullptr;
    const clang::Stmt* scalarStorageStmt = nullptr;
    const clang::Stmt* scalarPayloadStmt = nullptr;
    const clang::Stmt* scalarShellArgStmt = nullptr;
    const clang::Stmt* readerSliceStmt = nullptr;
    const clang::Stmt* dacExprStmt = nullptr;
    const clang::Stmt* ownerWritebackStmt = nullptr;
    std::string ownerName;
    std::string scalarVectorName;
    std::string scalarExpr;
};

struct OwnerLoopContractCheck {
    bool passed = false;
    std::string reason;
};

OwnerLoopContractCheck checkOwnerLoopContractConsistency(
    const LoopLocalStencilOwnerLoopContract& contract) {
    if (!contract.enabled || !contract.replacementStmt) {
        return {false, "owner-loop-contract-disabled"};
    }
    int replaceCount = 0;
    std::set<std::string> removeRoles;
    for (const auto& statement : contract.lowering.statements) {
        if (statement.action == LoweringContractStmtAction::Replace &&
            statement.stmt == contract.replacementStmt &&
            statement.role == "owner-loop") {
            ++replaceCount;
        } else if (statement.action ==
                   LoweringContractStmtAction::Remove) {
            removeRoles.insert(statement.role);
        }
    }
    static const char* requiredRoles[] = {
        "writer-slice", "scalar-vector-storage", "scalar-payload",
        "scalar-shell-arg", "reader-slice", "dac-expression",
        "owner-writeback"};
    if (replaceCount != 1) {
        return {false, "owner-loop-contract-missing-replace-loop"};
    }
    for (const char* role : requiredRoles) {
        if (removeRoles.count(role) == 0) {
            return {false, std::string("owner-loop-contract-missing-") +
                               role};
        }
    }
    if (contract.lowering.residentTensors.size() != 3) {
        return {false, "owner-loop-contract-resident-count-mismatch"};
    }
    bool hasLoopExitMaterialization = false;
    for (const auto& materialization : contract.lowering.materializations) {
        if (materialization.tensorName == contract.ownerTensorName &&
            materialization.timing ==
                LoweringContractMaterializeTiming::LoopExit) {
            hasLoopExitMaterialization = true;
            break;
        }
    }
    if (!hasLoopExitMaterialization) {
        return {false, "owner-loop-contract-missing-loop-exit-materialize"};
    }
    bool hasCompileGuard = false;
    bool hasRuntimeGuard = false;
    for (const auto& guard : contract.lowering.guards) {
        if (guard.reason.empty()) {
            return {false, "owner-loop-contract-guard-missing-reason"};
        }
        hasCompileGuard =
            hasCompileGuard ||
            guard.disposition ==
                LoweringContractGuardDisposition::CompileTimeFallback;
        hasRuntimeGuard =
            hasRuntimeGuard ||
            guard.disposition ==
                LoweringContractGuardDisposition::RuntimeAbort;
    }
    if (!hasCompileGuard || !hasRuntimeGuard) {
        return {false, "owner-loop-contract-missing-guard-class"};
    }
    return {true, "owner-loop-contract-consistent"};
}

const clang::ForStmt* outerForLoopForExpr(DacppFile* dacppFile,
                                          const clang::BinaryOperator* dacExpr) {
    if (!dacppFile || !dacppFile->getContext() || !dacExpr) {
        return nullptr;
    }
    clang::DynTypedNode current = clang::DynTypedNode::create(*dacExpr);
    const clang::ForStmt* outerFor = nullptr;
    while (true) {
        auto parents = dacppFile->getContext()->getParents(current);
        if (parents.empty()) {
            break;
        }
        const auto& parent = parents[0];
        if (const auto* forStmt = parent.get<clang::ForStmt>()) {
            outerFor = forStmt;
        }
        if (parent.get<clang::FunctionDecl>()) {
            break;
        }
        current = parent;
    }
    return outerFor;
}

std::string loopVarName(const clang::ForStmt* forStmt) {
    const auto* declStmt =
        forStmt ? llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit())
                : nullptr;
    if (!declStmt || !declStmt->isSingleDecl()) {
        return "";
    }
    const auto* varDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    return varDecl ? varDecl->getNameAsString() : "";
}

const ParamAccessPlan* stencil1DReaderParam(
    const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::StencilWindow && param.reads &&
            !param.writes) {
            return &param;
        }
    }
    return nullptr;
}

const ParamAccessPlan* stencil1DWriterParam(
    const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::OutputDirect && param.writes &&
            !param.reads) {
            return &param;
        }
    }
    return nullptr;
}

const ParamAccessPlan* scalarReaderParam(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar &&
            param.reads && !param.writes) {
            return &param;
        }
    }
    return nullptr;
}

LoopLoweringContract buildOwnerLoopLoweringContract(
    const ShellPartitionPlan& exprPlan,
    const clang::ForStmt* outerLoop,
    const clang::CompoundStmt* loopBody,
    const std::string& ownerName,
    const ParamAccessPlan& reader,
    const ParamAccessPlan& writer) {
    LoopLoweringContract contract;
    contract.enabled = true;
    contract.loweringName = "LoopLocalStencilOwnerLoop";
    contract.acceptedReason =
        "loop-local stencil owner-loop contract accepted current strict shape";
    contract.statements.push_back(
        {outerLoop, LoweringContractStmtAction::Replace, "owner-loop",
         "replace whole source loop with generated owner-loop call"});

    const char* roles[] = {
        "writer-slice", "scalar-vector-storage", "scalar-payload",
        "scalar-shell-arg", "reader-slice", "dac-expression",
        "owner-writeback"};
    int stmtIndex = 0;
    for (const clang::Stmt* stmt : loopBody->body()) {
        if (stmt == exprPlan.exprNode.dacExpr) {
            contract.statements.push_back(
                {stmt, LoweringContractStmtAction::Remove, "dac-expression",
                 "absorbed into generated owner-loop kernel"});
            ++stmtIndex;
            continue;
        }
        const std::string role =
            stmtIndex < 7 ? roles[stmtIndex] : "owner-loop-statement";
        contract.statements.push_back(
            {stmt, LoweringContractStmtAction::Remove, role,
             "absorbed by owner-loop replacement"});
        ++stmtIndex;
    }
    contract.residentTensors.push_back(
        {ownerName, "owner matrix provides resident stencil state"});
    contract.residentTensors.push_back(
        {reader.actualTensorName, "loop-local reader slice"});
    contract.residentTensors.push_back(
        {writer.actualTensorName, "loop-local writer slice"});
    contract.materializations.push_back(
        {ownerName, LoweringContractMaterializeTiming::LoopExit,
         "gather owned history once and update owner matrix on root"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "strict loop-local owner/slice/scalar/writeback shape proof"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::RuntimeAbort,
         "MPI count narrowing guards for history gather"});
    return contract;
}

LoopLocalStencilOwnerLoopContract rejectContract(
    const ShellPartitionPlan& exprPlan,
    const std::string& reason) {
    LoopLocalStencilOwnerLoopContract contract;
    contract.exprIndex = exprPlan.exprIndex;
    contract.rejectedReason = reason;
    return contract;
}

}  // namespace

LoopLocalStencilOwnerLoopContract detectLoopLocalStencilOwnerLoop(
    DacppFile* dacppFile,
    const ShellPartitionPlan& exprPlan) {
    if (!dacppFile || !exprPlan.exprNode.dacExpr || !exprPlan.exprNode.calc ||
        exprPlan.signature.layout != LocalLayoutKind::StencilWindow1D) {
        return rejectContract(exprPlan, "requires stencil window 1d plan");
    }
    if (exprPlan.exprNode.calc->getNumParams() != 3 ||
        exprPlan.params.size() != 3) {
        return rejectContract(
            exprPlan,
            "requires exactly three calc/shell parameters");
    }
    const auto* reader = stencil1DReaderParam(exprPlan);
    const auto* writer = stencil1DWriterParam(exprPlan);
    const auto* scalar = scalarReaderParam(exprPlan);
    if (!reader || !writer || !scalar) {
        return rejectContract(
            exprPlan,
            "requires one stencil reader, one write-only direct writer, and one replicated scalar reader");
    }
    const std::string elemType =
        exprPlan.exprNode.calc->getParam(reader->paramIndex)->getBasicType();
    const std::string writerType =
        exprPlan.exprNode.calc->getParam(writer->paramIndex)->getBasicType();
    const std::string scalarType =
        exprPlan.exprNode.calc->getParam(scalar->paramIndex)->getBasicType();
    if (elemType != writerType || elemType != scalarType ||
        usesByteTransport(elemType)) {
        return rejectContract(
            exprPlan,
            "requires matching native MPI element types for reader/writer/scalar");
    }

    const clang::ForStmt* outerLoop =
        outerForLoopForExpr(dacppFile, exprPlan.exprNode.dacExpr);
    const std::string loopVar = loopVarName(outerLoop);
    if (!outerLoop || loopVar.empty()) {
        return rejectContract(
            exprPlan,
            "requires enclosing for loop with simple induction variable");
    }
    if (!isForwardUnitLoopFromZero(outerLoop, loopVar)) {
        return rejectContract(
            exprPlan,
            "requires enclosing for loop with simple forward unit induction");
    }
    const auto* loopBody =
        llvm::dyn_cast_or_null<clang::CompoundStmt>(outerLoop->getBody());
    if (!loopBody) {
        return rejectContract(exprPlan, "requires compound loop body");
    }
    const std::vector<const clang::Stmt*> bodyStmts =
        collectTopLevelBodyStmts(loopBody);
    const std::string readerName = reader->actualTensorName;
    const std::string writerName = writer->actualTensorName;
    const std::string scalarName = scalar->actualTensorName;

    std::string ownerName;
    const clang::ValueDecl* ownerDecl = nullptr;
    if (!findReaderOwnerNameInBody(bodyStmts, readerName, loopVar,
                                   &ownerName, &ownerDecl)) {
        return rejectContract(exprPlan,
                              "requires reader slice owner[{}][loop]");
    }
    OwnerMatrixShapeProof ownerShapeProof;
    if (!extractOwnerMatrixShapeProof(ownerDecl, &ownerShapeProof)) {
        return rejectContract(
            exprPlan,
            "requires owner matrix shape {interior+1,time+1}");
    }
    if (!hasOwnerTimeLoopBound(outerLoop, loopVar, ownerShapeProof)) {
        return rejectContract(
            exprPlan,
            "requires owner loop bound to cover matrix time columns");
    }
    if (countWriterSlicesFromOwnerAtNextStep(bodyStmts, ownerName, loopVar) >
        1) {
        return rejectContract(exprPlan,
                              "reject multiple writer slices");
    }
    int ownerNextStepAssignmentCount = 0;
    for (const clang::Stmt* stmt : bodyStmts) {
        ownerNextStepAssignmentCount +=
            countOwnerNextStepAssignments(stmt, ownerName, loopVar);
    }
    if (ownerNextStepAssignmentCount > 1) {
        return rejectContract(exprPlan,
                              "reject ambiguous owner mutation or multiple writer slices");
    }

    if (bodyStmts.size() != 7) {
        return rejectContract(
            exprPlan,
            "requires current seven-statement owner-loop body shape");
    }

    OwnerLoopBodyShape shape;
    shape.writerSliceStmt = bodyStmts[0];
    shape.scalarStorageStmt = bodyStmts[1];
    shape.scalarPayloadStmt = bodyStmts[2];
    shape.scalarShellArgStmt = bodyStmts[3];
    shape.readerSliceStmt = bodyStmts[4];
    shape.dacExprStmt = bodyStmts[5];
    shape.ownerWritebackStmt = bodyStmts[6];

    const clang::Stmt* dacTopLevelStmt =
        topLevelStmtForExpr(dacppFile, loopBody, exprPlan.exprNode.dacExpr);
    if (!dacTopLevelStmt || dacTopLevelStmt != shape.dacExprStmt) {
        return rejectContract(
            exprPlan,
            "requires DAC expression as sixth owner-loop statement");
    }

    const clang::ValueDecl* shapeOwnerDecl = nullptr;
    if (!isReaderSliceStmt(shape.readerSliceStmt, readerName, loopVar,
                           &shape.ownerName, &shapeOwnerDecl) ||
        shapeOwnerDecl != ownerDecl) {
        return rejectContract(exprPlan,
                              "requires reader slice owner[{}][loop]");
    }
    if (!isWriterSliceStmt(shape.writerSliceStmt, writerName,
                           shape.ownerName, loopVar, ownerShapeProof)) {
        if (isAnyWriterSliceFromOwnerAtNextStep(shape.writerSliceStmt,
                                               shape.ownerName, loopVar)) {
            return rejectContract(
                exprPlan,
                "reject wrong owner writer slice; expected owner[{1,...}][loop+1]");
        }
        return rejectContract(
            exprPlan,
            "requires writer slice owner[{1,...}][loop+1]");
    }
    if (!isScalarPayloadStmt(shape.scalarPayloadStmt, loopVar,
                             &shape.scalarVectorName, &shape.scalarExpr,
                             dacppFile)) {
        return rejectContract(
            exprPlan,
            "reject variant scalar payload; expected one invariant loop-local payload expression");
    }
    if (!isVectorStorageDeclStmt(shape.scalarStorageStmt,
                                 shape.scalarVectorName) ||
        !isScalarShellArgStmt(shape.scalarShellArgStmt, scalarName,
                              shape.scalarVectorName, dacppFile)) {
        return rejectContract(
            exprPlan,
            "requires loop-local scalar vector with one invariant payload");
    }
    if (!isOwnerWritebackLoopStmt(shape.ownerWritebackStmt, shape.ownerName,
                                  writerName, loopVar, ownerShapeProof)) {
        return rejectContract(
            exprPlan,
            "requires post-DAC writer slice writeback to owner[...][loop+1]");
    }

    LoopLocalStencilOwnerLoopContract contract;
    contract.enabled = true;
    contract.exprIndex = exprPlan.exprIndex;
    contract.replacementStmt = outerLoop;
    contract.ownerTensorName = shape.ownerName;
    contract.scalarExpr = shape.scalarExpr;
    contract.functionName =
        operatorResidentWrapperName(exprPlan.exprNode.shell,
                                    exprPlan.exprNode.calc,
                                    exprPlan.exprIndex) +
        "_owner_loop";
    contract.elementType = elemType;
    contract.mpiType = mpiDatatypeFor(elemType);
    const FixedOwnerRowPostUsePlan fixedPostUse =
        analyzeFixedOwnerRowPostUse(dacppFile, outerLoop,
                                    contract.ownerTensorName);
    contract.fixedPostUseRow = fixedPostUse.supported;
    contract.postUseRow = fixedPostUse.row;
    contract.postUseReason = fixedPostUse.reason;
    contract.lowering = buildOwnerLoopLoweringContract(
        exprPlan, outerLoop, loopBody, shape.ownerName, *reader, *writer);
    const OwnerLoopContractCheck consistency =
        checkOwnerLoopContractConsistency(contract);
    contract.contractConsistencyCheckPassed = consistency.passed;
    contract.contractConsistencyCheckReason = consistency.reason;
    if (!contract.contractConsistencyCheckPassed) {
        return rejectContract(exprPlan, contract.contractConsistencyCheckReason);
    }
    return contract;
}

std::string buildLoopLocalStencilOwnerLoopCode(
    const LoopLocalStencilOwnerLoopContract& contract,
    const ShellPartitionPlan& exprPlan) {
    if (!contract.enabled || !exprPlan.exprNode.calc) {
        return "";
    }
    const std::string& type = contract.elementType;
    const std::string& mpiType = contract.mpiType;
    const std::string calcName = exprPlan.exprNode.calc->getName();
    const auto* reader = stencil1DReaderParam(exprPlan);
    const auto* writer = stencil1DWriterParam(exprPlan);
    const auto* scalar = scalarReaderParam(exprPlan);
    if (!reader || !writer || !scalar) {
        return "";
    }
    std::string code;
    code += "void " + contract.functionName + "(dacpp::Matrix<" + type +
            ">& __or_owner, " + type + " __or_scalar_value) {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);\n";
    code += "    dacpp::mpi::SegmentedProfile dacpp_profile;\n";
    code +=
        "    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    auto& q = dacpp::mpi::operator_resident::default_queue();\n";
    code += "    const int64_t __or_rows = __or_owner.getShape(0);\n";
    code += "    const int64_t __or_cols = __or_owner.getShape(1);\n";
    code += "    if (__or_rows < 3 || __or_cols < 2) {\n";
    code += "        return;\n";
    code += "    }\n";
    code += "    const int64_t __or_output_size = __or_rows - 2;\n";
    code += "    const int64_t __or_steps = __or_cols - 1;\n";
    code += "    const int __or_window_size = 3;\n";
    if (contract.fixedPostUseRow) {
        code += "    const int64_t __or_fixed_postuse_row = " +
                std::to_string(contract.postUseRow) + ";\n";
        code += "    const int64_t __or_fixed_postuse_out = __or_fixed_postuse_row - 1;\n";
    }
    code +=
        "    const auto __or_range = dacpp::mpi::operator_resident::rank_range_1d(__or_output_size, mpi_rank, mpi_size);\n";
    code += "    const int64_t __or_local_item_count = __or_range.count;\n";
    code +=
        "    const auto __or_halo_layout = dacpp::mpi::operator_resident::resident_halo_1d_layout(__or_output_size, mpi_rank, mpi_size, __or_window_size);\n";
    code += "    std::vector<" + type + "> __or_initial_col;\n";
    code +=
        "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);\n";
    code +=
        "    auto dacpp_profile_scatter_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (mpi_rank == 0) {\n";
    code +=
        "        __or_initial_col.resize(static_cast<std::size_t>(__or_rows));\n";
    code +=
        "        for (int64_t __or_row = 0; __or_row < __or_rows; ++__or_row) {\n";
    code +=
        "            __or_initial_col[static_cast<std::size_t>(__or_row)] = __or_owner.getElement({static_cast<int>(__or_row), 0});\n";
    code += "        }\n";
    code += "    }\n";
    code += "    std::vector<" + type + "> __or_curr;\n";
    code +=
        "    dacpp::mpi::operator_resident::scatter_window_1d(__or_initial_col, __or_curr, __or_output_size, __or_rows, __or_window_size, __or_halo_layout, mpi_rank, mpi_size, " +
        mpiType + ");\n";
    code +=
        "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start);\n";
    code += "    std::vector<" + type + "> __or_next(__or_curr.size(), " +
            type + "{});\n";
    code +=
        "    std::vector<" + type +
        "> __or_scalar_vec(1, __or_scalar_value);\n";
    code += "    std::vector<" + type + "> __or_left_boundary_values;\n";
    code += "    std::vector<" + type + "> __or_right_boundary_values;\n";
    code += "    auto dacpp_profile_bcast_start_boundaries = dacpp::mpi::profileSegmentStart();\n";
    code += "    __or_left_boundary_values.resize(static_cast<std::size_t>(__or_cols));\n";
    code += "    __or_right_boundary_values.resize(static_cast<std::size_t>(__or_cols));\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        for (int64_t __or_col = 0; __or_col < __or_cols; ++__or_col) {\n";
    code += "            __or_left_boundary_values[static_cast<std::size_t>(__or_col)] = __or_owner.getElement({0, static_cast<int>(__or_col)});\n";
    code += "            __or_right_boundary_values[static_cast<std::size_t>(__or_col)] = __or_owner.getElement({static_cast<int>(__or_rows - 1), static_cast<int>(__or_col)});\n";
    code += "        }\n";
    code += "    }\n";
    code += "    MPI_Bcast(__or_left_boundary_values.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_cols, \"[DACPP][MPI][OR][FOuLa] left boundary bcast count exceeds MPI int range\"), " +
            mpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    MPI_Bcast(__or_right_boundary_values.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_cols, \"[DACPP][MPI][OR][FOuLa] right boundary bcast count exceeds MPI int range\"), " +
            mpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_boundaries);\n";
    if (contract.fixedPostUseRow) {
        code += "    const bool __or_fixed_row_valid = __or_fixed_postuse_row > 0 && __or_fixed_postuse_row < __or_rows;\n";
        code += "    const bool __or_owns_fixed_row = __or_fixed_row_valid && __or_fixed_postuse_out >= __or_range.begin && __or_fixed_postuse_out < __or_range.begin + __or_range.count;\n";
        code += "    const int64_t __or_local_fixed_out = __or_owns_fixed_row ? (__or_fixed_postuse_out - __or_range.begin) : -1;\n";
        code += "    std::vector<" + type + "> __or_selected_history;\n";
        code += "    if (__or_owns_fixed_row) {\n";
        code += "        __or_selected_history.assign(static_cast<std::size_t>(__or_cols), " + type + "{});\n";
        code += "        __or_selected_history[0] = __or_curr[static_cast<std::size_t>(__or_local_fixed_out + 1)];\n";
        code += "    }\n";
    } else {
        code +=
            "    std::vector<" + type +
            "> __or_local_history(static_cast<std::size_t>(__or_local_item_count * __or_cols), " +
            type + "{});\n";
        code +=
            "    for (int64_t __or_i = 0; __or_i < __or_local_item_count; ++__or_i) {\n";
        code +=
            "        __or_local_history[static_cast<std::size_t>(__or_i * __or_cols)] = __or_curr[static_cast<std::size_t>(__or_i + 1)];\n";
        code += "    }\n";
    }
    code +=
        "    const int __or_last_owner_rank = dacpp::mpi::operator_resident::nearest_nonempty_rank_1d(__or_output_size, mpi_size, mpi_size, -1);\n";
    code +=
        "    for (int64_t __or_step = 0; __or_step < __or_steps; ++__or_step) {\n";
    code += "        __or_scalar_vec[0] = __or_scalar_value;\n";
    code +=
        "        auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
    code += "        if (__or_local_item_count > 0) {\n";
    code += "            sycl::buffer<" + type +
            ", 1> __or_reader_buf(__or_curr.data(), sycl::range<1>(__or_curr.size()));\n";
    code += "            sycl::buffer<" + type +
            ", 1> __or_writer_buf(__or_next.data(), sycl::range<1>(__or_next.size()));\n";
    code += "            sycl::buffer<" + type +
            ", 1> __or_scalar_buf(__or_scalar_vec.data(), sycl::range<1>(__or_scalar_vec.size()));\n";
    code += "            q.submit([&](sycl::handler& h) {\n";
    code +=
        "                auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    code +=
        "                auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code +=
        "                auto __or_scalar_acc = __or_scalar_buf.get_access<sycl::access::mode::read>(h);\n";
    code +=
        "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code +=
        "                    const int item_linear = static_cast<int>(idx[0]);\n";
    code +=
        "                    auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    code +=
        "                    auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    code +=
        "                    auto* __or_scalar_data = __or_scalar_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : exprPlan.params) {
        const std::string paramType =
            exprPlan.exprNode.calc->getParam(param.paramIndex)->getBasicType();
        if (param.paramIndex == reader->paramIndex) {
            code += "                    dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_reader_data, item_linear};\n";
            continue;
        }
        if (param.paramIndex == writer->paramIndex) {
            code += "                    dacpp::mpi::ContiguousView1D<" +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_writer_data, item_linear + 1};\n";
            continue;
        }
        if (param.paramIndex == scalar->paramIndex) {
            code += "                    dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_scalar_data, 0};\n";
            continue;
        }
        return "";
    }
    code += "                    " + calcName + "_mpi_local(view_" +
            exprPlan.exprNode.calc->getParam(0)->getName() + ", view_" +
            exprPlan.exprNode.calc->getParam(1)->getName() + ", view_" +
            exprPlan.exprNode.calc->getParam(2)->getName() + ");\n";
    code += "                });\n";
    code += "            });\n";
    code += "            q.wait();\n";
    code += "        }\n";
    code +=
        "        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
    code += "        " + type + " __or_left_boundary{};\n";
    code += "        " + type + " __or_right_boundary{};\n";
    code += "        __or_left_boundary = __or_left_boundary_values[static_cast<std::size_t>(__or_step + 1)];\n";
    code += "        __or_right_boundary = __or_right_boundary_values[static_cast<std::size_t>(__or_step + 1)];\n";
    code += "        if (mpi_rank == 0 && !__or_next.empty()) {\n";
    code += "            __or_next[0] = __or_left_boundary;\n";
    code += "        }\n";
    code +=
        "        if (mpi_rank == __or_last_owner_rank && __or_local_item_count > 0 && static_cast<std::size_t>(__or_local_item_count + 1) < __or_next.size()) {\n";
    code +=
        "            __or_next[static_cast<std::size_t>(__or_local_item_count + 1)] = __or_right_boundary;\n";
    code += "        }\n";
    code +=
        "        auto dacpp_profile_halo_start = dacpp::mpi::profileSegmentStart();\n";
    code +=
        "        dacpp::mpi::operator_resident::exchange_halo_1d_inplace(__or_next, __or_halo_layout, __or_output_size, __or_window_size, 1, mpi_rank, mpi_size, " +
        mpiType + ");\n";
    code +=
        "        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Halo, dacpp_profile_halo_start);\n";
    code += "        __or_curr.swap(__or_next);\n";
    if (contract.fixedPostUseRow) {
        code += "        if (__or_owns_fixed_row) {\n";
        code += "            __or_selected_history[static_cast<std::size_t>(__or_step + 1)] = __or_curr[static_cast<std::size_t>(__or_local_fixed_out + 1)];\n";
        code += "        }\n";
    } else {
        code +=
            "        for (int64_t __or_i = 0; __or_i < __or_local_item_count; ++__or_i) {\n";
        code +=
            "            __or_local_history[static_cast<std::size_t>(__or_i * __or_cols + (__or_step + 1))] = __or_curr[static_cast<std::size_t>(__or_i + 1)];\n";
        code += "        }\n";
    }
    code += "    }\n";
    if (contract.fixedPostUseRow) {
        code += "    int __or_fixed_owner_rank = 0;\n";
        code += "    if (__or_fixed_row_valid) {\n";
        code += "        __or_fixed_owner_rank = MPI_PROC_NULL;\n";
        code += "        for (int __or_r = 0; __or_r < mpi_size; ++__or_r) {\n";
        code += "            const auto __or_r_range = dacpp::mpi::operator_resident::rank_range_1d(__or_output_size, __or_r, mpi_size);\n";
        code += "            if (__or_fixed_postuse_out >= __or_r_range.begin && __or_fixed_postuse_out < __or_r_range.begin + __or_r_range.count) {\n";
        code += "                __or_fixed_owner_rank = __or_r;\n";
        code += "                break;\n";
        code += "            }\n";
        code += "        }\n";
        code += "    }\n";
        code += "    std::vector<" + type + "> __or_selected_global_history;\n";
        code += "    if (mpi_rank == 0 && __or_fixed_row_valid) {\n";
        code += "        __or_selected_global_history.resize(static_cast<std::size_t>(__or_cols));\n";
        code += "    }\n";
        code += "    auto dacpp_profile_gather_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    const int __or_selected_history_count = __or_fixed_row_valid ? dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_cols, \"[DACPP][MPI][OR][FOuLa] selected row history count exceeds MPI int range\") : 0;\n";
        code += "    if (__or_fixed_row_valid) {\n";
        code += "        if (__or_fixed_owner_rank == MPI_PROC_NULL) {\n";
        code += "            if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR][FOuLa] fixed owner row has no rank owner\\n\");\n";
        code += "            MPI_Abort(MPI_COMM_WORLD, 4);\n";
        code += "        }\n";
        code += "        if (mpi_rank == __or_fixed_owner_rank) {\n";
        code += "            if (mpi_rank == 0) {\n";
        code += "                __or_selected_global_history = __or_selected_history;\n";
        code += "            } else {\n";
        code += "                MPI_Send(__or_selected_history.data(), __or_selected_history_count, " + mpiType + ", 0, 4511, MPI_COMM_WORLD);\n";
        code += "            }\n";
        code += "        } else if (mpi_rank == 0) {\n";
        code += "            MPI_Recv(__or_selected_global_history.data(), __or_selected_history_count, " + mpiType + ", __or_fixed_owner_rank, 4511, MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n";
        code += "        }\n";
        code += "    }\n";
        code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start);\n";
        code += "    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    if (mpi_rank == 0 && __or_fixed_row_valid) {\n";
        code += "        for (int64_t __or_col = 0; __or_col < __or_cols; ++__or_col) {\n";
        code += "            __or_owner.reviseValue(__or_selected_global_history[static_cast<std::size_t>(__or_col)], {static_cast<int>(__or_fixed_postuse_row), static_cast<int>(__or_col)});\n";
        code += "        }\n";
        code += "    }\n";
        code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    } else {
        code += "    std::vector<int> __or_hist_counts(mpi_size, 0);\n";
        code += "    std::vector<int> __or_hist_displs(mpi_size, 0);\n";
        code += "    for (int __or_r = 0; __or_r < mpi_size; ++__or_r) {\n";
        code +=
            "        const auto __or_r_range = dacpp::mpi::operator_resident::rank_range_1d(__or_output_size, __or_r, mpi_size);\n";
        code +=
            "        __or_hist_counts[__or_r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_r_range.count * __or_cols, \"[DACPP][MPI][OR][FOuLa] history gather count exceeds MPI int range\");\n";
        code +=
            "        __or_hist_displs[__or_r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_r_range.begin * __or_cols, \"[DACPP][MPI][OR][FOuLa] history gather displacement exceeds MPI int range\");\n";
        code += "    }\n";
        code += "    std::vector<" + type + "> __or_global_history;\n";
        code += "    if (mpi_rank == 0) {\n";
        code +=
            "        __or_global_history.resize(static_cast<std::size_t>(__or_output_size * __or_cols));\n";
        code += "    }\n";
        code +=
            "    auto dacpp_profile_gather_start = dacpp::mpi::profileSegmentStart();\n";
        code +=
            "    MPI_Gatherv(__or_local_history.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_local_item_count * __or_cols, \"[DACPP][MPI][OR][FOuLa] local history gather count exceeds MPI int range\"), " +
            mpiType +
            ", mpi_rank == 0 ? __or_global_history.data() : nullptr, mpi_rank == 0 ? __or_hist_counts.data() : nullptr, mpi_rank == 0 ? __or_hist_displs.data() : nullptr, " +
            mpiType + ", 0, MPI_COMM_WORLD);\n";
        code +=
            "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start);\n";
        code +=
            "    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    if (mpi_rank == 0) {\n";
        code +=
            "        for (int64_t __or_out = 0; __or_out < __or_output_size; ++__or_out) {\n";
        code +=
            "            for (int64_t __or_col = 0; __or_col < __or_cols; ++__or_col) {\n";
        code +=
            "                __or_owner.reviseValue(__or_global_history[static_cast<std::size_t>(__or_out * __or_cols + __or_col)], {static_cast<int>(__or_out + 1), static_cast<int>(__or_col)});\n";
        code += "            }\n";
        code += "        }\n";
        code += "    }\n";
        code +=
            "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    }
    code += "    dacpp::mpi::reportSegmentedProfile(\"" +
            contract.functionName + "\", dacpp_profile, MPI_COMM_WORLD);\n";
    code += "}\n";
    return code;
}

void logLoopLocalStencilOwnerLoopAccepted(
    const LoopLocalStencilOwnerLoopContract& contract) {
    if (!contract.enabled) {
        return;
    }
    llvm::outs() << "[DACPP][MPI][OR][OwnerLoop] expr="
                 << contract.exprIndex
                 << " owner-loop=accepted contract="
                 << contract.lowering.loweringName
                 << " contract-source=replace-loop"
                 << " contract-remove=writer-slice,scalar-vector-storage,scalar-payload,scalar-shell-arg,reader-slice,dac-expression,owner-writeback"
                 << " contract-resident="
                 << contract.lowering.residentTensors.size()
                 << " contract-materialize=loop-exit"
                 << " materialize="
                 << (contract.fixedPostUseRow ? "fixed-row" : "full-history");
    if (contract.fixedPostUseRow) {
        llvm::outs() << " row=" << contract.postUseRow;
    } else if (!contract.postUseReason.empty()) {
        llvm::outs() << " fixed-row=fallback reason="
                     << contract.postUseReason;
    }
    llvm::outs()
                 << " guard-compile=fallback guard-runtime=count-or-shape"
                 << " owner=" << contract.ownerTensorName
                 << " contract-check="
                 << (contract.contractConsistencyCheckPassed ? "pass"
                                                              : "fail")
                 << " reason="
                 << contract.contractConsistencyCheckReason
                 << " accepted-reason="
                 << contract.lowering.acceptedReason << "\n";
}

void logLoopLocalStencilOwnerLoopRejected(
    const LoopLocalStencilOwnerLoopContract& contract) {
    if (contract.enabled || contract.rejectedReason.empty()) {
        return;
    }
    llvm::outs() << "[DACPP][MPI][OR][OwnerLoop] expr="
                 << contract.exprIndex
                 << " owner-loop=rejected contract=LoopLocalStencilOwnerLoop"
                 << " reason=" << contract.rejectedReason << "\n";
}

void logLoopLocalStencilOwnerLoopRewriteEnabled(
    const LoopLocalStencilOwnerLoopContract& contract) {
    if (!contract.enabled) {
        return;
    }
    llvm::outs() << "[DACPP][MPI][OR][OwnerLoop] expr="
                 << contract.exprIndex
                 << " owner-loop=rewrite-enabled contract="
                 << contract.lowering.loweringName << "\n";
}

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
