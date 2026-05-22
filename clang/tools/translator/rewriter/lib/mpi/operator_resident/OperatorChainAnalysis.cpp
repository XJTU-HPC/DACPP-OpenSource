#include <algorithm>
#include <cctype>
#include <sstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

#include "ASTParse.h"
#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"
#include "ShellPartitionAnalysis_Internal.h"
#include "Split.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

const ParamAccessPlan* stencilWindowReaderParam(
    const ShellPartitionPlan& plan);
const ParamAccessPlan* stencilOutputDirectWriterParam(
    const ShellPartitionPlan& plan);

const clang::CallExpr* getShellCallExpr(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }
    clang::Expr* shellExpr =
        Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                        : dacExpr->getRHS();
    return getNode<clang::CallExpr>(shellExpr);
}

std::string trim(std::string text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string compactExprText(std::string text) {
    text.erase(std::remove_if(text.begin(), text.end(),
                              [](unsigned char c) {
                                  return std::isspace(c) != 0;
                              }),
               text.end());
    return text;
}

bool endsWith(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) ==
               0;
}

std::string exprSource(const clang::Expr* expr, clang::ASTContext* context) {
    if (!expr || !context) {
        return "";
    }
    return trim(clang::Lexer::getSourceText(
                    clang::CharSourceRange::getTokenRange(
                        expr->getSourceRange()),
                    context->getSourceManager(),
                    context->getLangOpts())
                    .str());
}

const clang::Expr* unwrapExtentExpr(const clang::Expr* expr) {
    if (!expr) {
        return nullptr;
    }
    while (true) {
        const clang::Expr* next = expr->IgnoreParenImpCasts();
        if (const auto* cleanups =
                llvm::dyn_cast<clang::ExprWithCleanups>(next)) {
            next = cleanups->getSubExpr();
        } else if (const auto* materialized =
                       llvm::dyn_cast<clang::MaterializeTemporaryExpr>(next)) {
            next = materialized->getSubExpr();
        } else if (const auto* temporary =
                       llvm::dyn_cast<clang::CXXBindTemporaryExpr>(next)) {
            next = temporary->getSubExpr();
        } else if (const auto* construct =
                       llvm::dyn_cast<clang::CXXConstructExpr>(next)) {
            if (construct->getNumArgs() == 1) {
                next = construct->getArg(0);
            } else {
                return next;
            }
        }
        if (!next || next == expr) {
            return next;
        }
        expr = next;
    }
}

bool evaluateInt64Expr(const clang::Expr* expr,
                       clang::ASTContext* context,
                       int64_t& value) {
    expr = unwrapExtentExpr(expr);
    if (!expr || !context) {
        return false;
    }
    if (const auto* lit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
        value = lit->getValue().getSExtValue();
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
    value = evalResult.Val.getInt().getSExtValue();
    return true;
}

int64_t staticVectorVarSize(const clang::VarDecl* varDecl,
                            clang::ASTContext* context) {
    if (!varDecl || !varDecl->hasInit() || !context) {
        return -1;
    }
    const std::string typeName = varDecl->getType().getAsString();
    if (typeName.find("vector") == std::string::npos &&
        typeName.find("Vector") == std::string::npos) {
        return -1;
    }
    const clang::Expr* init = unwrapExtentExpr(varDecl->getInit());
    if (const auto* construct =
            llvm::dyn_cast_or_null<clang::CXXConstructExpr>(init)) {
        if (construct->getNumArgs() >= 1) {
            int64_t size = -1;
            if (evaluateInt64Expr(construct->getArg(0), context, size) &&
                size > 0) {
                return size;
            }
        }
    }
    if (const auto* initList =
            llvm::dyn_cast_or_null<clang::InitListExpr>(init)) {
        return initList->getNumInits() > 0
                   ? static_cast<int64_t>(initList->getNumInits())
                   : -1;
    }
    return -1;
}

int64_t staticOneDimExtentForExpr(
    const clang::Expr* expr,
    clang::ASTContext* context,
    std::set<const clang::ValueDecl*>& seenDecls);

int64_t staticOneDimExtentForVar(
    const clang::VarDecl* varDecl,
    clang::ASTContext* context,
    std::set<const clang::ValueDecl*>& seenDecls) {
    if (!varDecl || !context) {
        return -1;
    }
    const int64_t vectorSize = staticVectorVarSize(varDecl, context);
    if (vectorSize > 0) {
        return vectorSize;
    }
    if (!varDecl->hasInit() || seenDecls.count(varDecl) != 0) {
        return -1;
    }
    seenDecls.insert(varDecl);
    const int64_t initExtent =
        staticOneDimExtentForExpr(varDecl->getInit(), context, seenDecls);
    seenDecls.erase(varDecl);
    return initExtent;
}

int64_t staticSliceExtent(const clang::Expr* indexExpr,
                          int64_t baseExtent,
                          clang::ASTContext* context) {
    indexExpr = unwrapExtentExpr(indexExpr);
    if (!indexExpr || !context) {
        return -1;
    }
    const clang::InitListExpr* listExpr =
        llvm::dyn_cast<clang::InitListExpr>(indexExpr);
    if (!listExpr) {
        if (const auto* stdList =
                llvm::dyn_cast<clang::CXXStdInitializerListExpr>(indexExpr)) {
            const clang::Expr* sub = unwrapExtentExpr(stdList->getSubExpr());
            listExpr = llvm::dyn_cast_or_null<clang::InitListExpr>(sub);
        }
    }
    if (!listExpr) {
        if (const auto* construct =
                llvm::dyn_cast<clang::CXXConstructExpr>(indexExpr)) {
            if (construct->getNumArgs() == 0) {
                return baseExtent;
            }
            if (construct->getNumArgs() == 1) {
                return 1;
            }
            int64_t start = -1;
            int64_t end = -1;
            int64_t stride = 1;
            if (!evaluateInt64Expr(construct->getArg(0), context, start) ||
                !evaluateInt64Expr(construct->getArg(1), context, end)) {
                return -1;
            }
            if (construct->getNumArgs() >= 3 &&
                !evaluateInt64Expr(construct->getArg(2), context, stride)) {
                return -1;
            }
            if (start < 0 || end < start || stride <= 0) {
                return -1;
            }
            return ((end - start - 1) / stride) + 1;
        }
        return -1;
    }
    if (listExpr->getNumInits() == 0) {
        return baseExtent;
    }
    if (listExpr->getNumInits() == 1) {
        return 1;
    }
    int64_t start = -1;
    int64_t end = -1;
    int64_t stride = 1;
    if (!evaluateInt64Expr(listExpr->getInit(0), context, start) ||
        !evaluateInt64Expr(listExpr->getInit(1), context, end)) {
        return -1;
    }
    if (listExpr->getNumInits() >= 3 &&
        !evaluateInt64Expr(listExpr->getInit(2), context, stride)) {
        return -1;
    }
    if (start < 0 || end < start || stride <= 0) {
        return -1;
    }
    return ((end - start - 1) / stride) + 1;
}

int64_t staticOneDimExtentForExpr(
    const clang::Expr* expr,
    clang::ASTContext* context,
    std::set<const clang::ValueDecl*>& seenDecls) {
    expr = unwrapExtentExpr(expr);
    if (!expr || !context) {
        return -1;
    }
    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        if (const auto* varDecl =
                llvm::dyn_cast_or_null<clang::VarDecl>(declRef->getDecl())) {
            return staticOneDimExtentForVar(varDecl, context, seenDecls);
        }
        return -1;
    }
    if (const auto* opCall =
            llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() >= 2) {
            const int64_t baseExtent = staticOneDimExtentForExpr(
                opCall->getArg(0), context, seenDecls);
            return staticSliceExtent(opCall->getArg(1), baseExtent, context);
        }
    }
    if (const auto* construct =
            llvm::dyn_cast<clang::CXXConstructExpr>(expr)) {
        if (construct->getNumArgs() == 1) {
            return staticOneDimExtentForExpr(construct->getArg(0), context,
                                             seenDecls);
        }
    }
    return -1;
}

int64_t staticOneDimShellArgExtent(DacppFile* dacppFile,
                                   const ShellPartitionPlan& plan,
                                   const ParamAccessPlan& param) {
    if (!dacppFile || !dacppFile->getContext() || !plan.exprNode.dacExpr) {
        return operator_resident::shapeValueFor(plan.exprNode.shell,
                                                param.paramIndex, 0);
    }
    const clang::CallExpr* shellCall = getShellCallExpr(plan.exprNode.dacExpr);
    if (shellCall && param.paramIndex >= 0 &&
        param.paramIndex < static_cast<int>(shellCall->getNumArgs())) {
        std::set<const clang::ValueDecl*> seenDecls;
        const int64_t extent = staticOneDimExtentForExpr(
            shellCall->getArg(param.paramIndex), dacppFile->getContext(),
            seenDecls);
        if (extent > 0) {
            return extent;
        }
    }
    return operator_resident::shapeValueFor(plan.exprNode.shell,
                                            param.paramIndex, 0);
}

std::string baseNameFromExpr(const clang::Expr* expr) {
    if (!expr) {
        return "";
    }
    expr = expr->IgnoreParenImpCasts();
    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        if (declRef->getDecl()) {
            return declRef->getDecl()->getNameAsString();
        }
    }
    if (const auto* member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
        return baseNameFromExpr(member->getBase());
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() > 0) {
            return baseNameFromExpr(opCall->getArg(0));
        }
    }
    return "";
}

struct TensorAliasKey {
    std::string name;
    bool precise = false;
};

TensorAliasKey tensorAliasKeyForExpr(
    const clang::Expr* expr,
    clang::ASTContext* context,
    std::set<const clang::ValueDecl*>& seenDecls) {
    if (!expr) {
        return {};
    }
    expr = expr->IgnoreParenImpCasts();
    if (const auto* materialized =
            llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
        return tensorAliasKeyForExpr(materialized->getSubExpr(), context,
                                     seenDecls);
    }
    if (const auto* temporary =
            llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
        return tensorAliasKeyForExpr(temporary->getSubExpr(), context,
                                     seenDecls);
    }
    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        const clang::ValueDecl* decl = declRef->getDecl();
        if (!decl) {
            return {exprSource(expr, context), false};
        }
        if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl)) {
            if (varDecl->getType()->isReferenceType()) {
                if (seenDecls.count(varDecl) != 0) {
                    return {varDecl->getNameAsString(), false};
                }
                if (const clang::Expr* init = varDecl->getInit()) {
                    seenDecls.insert(varDecl);
                    TensorAliasKey key =
                        tensorAliasKeyForExpr(init, context, seenDecls);
                    seenDecls.erase(varDecl);
                    if (!key.name.empty()) {
                        return key;
                    }
                }
                return {varDecl->getNameAsString(), false};
            }
        }
        return {decl->getNameAsString(), true};
    }
    if (const auto* member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
        TensorAliasKey key =
            tensorAliasKeyForExpr(member->getBase(), context, seenDecls);
        if (!key.name.empty()) {
            return key;
        }
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() > 0) {
            TensorAliasKey key =
                tensorAliasKeyForExpr(opCall->getArg(0), context, seenDecls);
            if (!key.name.empty()) {
                return key;
            }
        }
    }
    const std::string baseName = baseNameFromExpr(expr);
    if (!baseName.empty()) {
        return {baseName, false};
    }
    return {exprSource(expr, context), false};
}

bool sourceRangeContains(const clang::SourceManager& sourceManager,
                         clang::SourceRange outer,
                         clang::SourceRange inner) {
    if (outer.isInvalid() || inner.isInvalid()) {
        return false;
    }
    auto beforeOrEqual = [&](clang::SourceLocation lhs,
                             clang::SourceLocation rhs) {
        return lhs == rhs || sourceManager.isBeforeInTranslationUnit(lhs, rhs);
    };
    return beforeOrEqual(outer.getBegin(), inner.getBegin()) &&
           beforeOrEqual(inner.getEnd(), outer.getEnd());
}

bool stmtSourceRangeContains(const clang::SourceManager& sourceManager,
                             const clang::Stmt* outer,
                             const clang::Stmt* inner) {
    return outer && inner &&
           sourceRangeContains(sourceManager, outer->getSourceRange(),
                               inner->getSourceRange());
}

TensorAliasKey actualTensorKeyForArg(const clang::Expr* expr,
                                     clang::ASTContext* context) {
    std::set<const clang::ValueDecl*> seenDecls;
    TensorAliasKey key = tensorAliasKeyForExpr(expr, context, seenDecls);
    if (!key.name.empty()) {
        return key;
    }
    return {exprSource(expr, context), false};
}

std::string actualTensorNameForArg(const clang::Expr* expr,
                                   clang::ASTContext* context) {
    std::string name = baseNameFromExpr(expr);
    if (!name.empty()) {
        return name;
    }
    return exprSource(expr, context);
}

void fillActualTensorNames(ShellPartitionPlan& plan, DacppFile* dacppFile) {
    if (!dacppFile || !plan.exprNode.dacExpr) {
        return;
    }
    const clang::CallExpr* shellCall = getShellCallExpr(plan.exprNode.dacExpr);
    if (!shellCall) {
        return;
    }
    for (auto& param : plan.params) {
        const int idx = param.paramIndex;
        if (idx < 0 || idx >= static_cast<int>(shellCall->getNumArgs())) {
            continue;
        }
        const std::string actual =
            actualTensorNameForArg(shellCall->getArg(idx),
                                   dacppFile->getContext());
        if (!actual.empty()) {
            param.actualTensorName = actual;
        }
        const TensorAliasKey aliasKey =
            actualTensorKeyForArg(shellCall->getArg(idx),
                                  dacppFile->getContext());
        if (!aliasKey.name.empty()) {
            param.actualTensorAliasKey = aliasKey.name;
            param.actualTensorAliasKeyPrecise = aliasKey.precise;
        }
    }
}

bool paramMayUseConstantInit(const ParamAccessPlan& param) {
    if (!param.reads || param.writes || param.actualTensorName.empty()) {
        return false;
    }
    return param.access == ParamAccessKind::DirectMapped ||
           param.access == ParamAccessKind::RowPartitionFullRow ||
           param.access == ParamAccessKind::StencilWindow ||
           param.access == ParamAccessKind::ReplicatedFullTensor;
}

bool paramMayUseGeneratedIndexInit(const ShellPartitionPlan& plan,
                                   const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::DirectMapped &&
           (plan.signature.layout == LocalLayoutKind::Contiguous1D ||
            plan.signature.layout == LocalLayoutKind::ReplicatedFullTensor) &&
           param.tensorDims.size() == 1 && param.tensorDims[0] == 0;
}

ConstantInitPlan analyzeConstantInitForShellArg(
    DacppFile* dacppFile,
    const clang::Expr* shellArg,
    const std::string& targetType);

std::string analysisElemType(const ShellPartitionPlan& plan,
                             const ParamAccessPlan& param) {
    if (plan.exprNode.calc && param.paramIndex >= 0 &&
        param.paramIndex < plan.exprNode.calc->getNumParams()) {
        return plan.exprNode.calc->getParam(param.paramIndex)->getBasicType();
    }
    return "double";
}

void annotateConstantInputInit(ShellPartitionPlan& plan, DacppFile* dacppFile) {
    if (!dacppFile || !plan.exprNode.dacExpr) {
        return;
    }
    const clang::CallExpr* shellCall = getShellCallExpr(plan.exprNode.dacExpr);
    if (!shellCall) {
        return;
    }
    for (auto& param : plan.params) {
        if (!paramMayUseConstantInit(param)) {
            continue;
        }
        if (param.paramIndex < 0 ||
            param.paramIndex >= static_cast<int>(shellCall->getNumArgs())) {
            param.constantInit.reason = "shell argument unavailable";
        } else {
            param.constantInit = analyzeConstantInitForShellArg(
                dacppFile, shellCall->getArg(param.paramIndex),
                analysisElemType(plan, param));
            if (param.constantInit.supported &&
                param.constantInit.indexExpr &&
                !paramMayUseGeneratedIndexInit(plan, param)) {
                param.constantInit.supported = false;
                param.constantInit.reason =
                    "index initializer requires 1D direct-mapped layout";
            }
        }
        llvm::outs() << "[DACPP][MPI][OR] input " << param.actualTensorName
                     << " init-sync=";
        if (param.constantInit.supported) {
            llvm::outs() << (param.constantInit.indexExpr
                                 ? "index-local expr="
                                 : "constant-local value=")
                         << (param.constantInit.logValue.empty()
                                 ? param.constantInit.valueExpr
                                 : param.constantInit.logValue);
        } else {
            llvm::outs() << "scatter";
            if (!param.constantInit.reason.empty()) {
                llvm::outs() << " fallback reason="
                             << param.constantInit.reason;
            }
        }
        llvm::outs() << "\n";
    }
}

bool paramMayNeedHostPostUseSync(const ParamAccessPlan& param) {
    if (param.writes &&
        (param.access == ParamAccessKind::OutputDirect ||
         param.access == ParamAccessKind::FixedBlock)) {
        return true;
    }
    if (param.access == ParamAccessKind::StencilWindow ||
        param.access == ParamAccessKind::DirectMapped) {
        return param.reads && !param.actualTensorName.empty();
    }
    return false;
}

bool layoutSupportsBoundedPostUseSync(const ShellPartitionPlan& plan,
                                      const ParamAccessPlan& param) {
    if (param.postUseSync.kind != PostUseSyncKind::BoundedIndexedRootRead) {
        return true;
    }
    for (const auto& bounded : param.postUseSync.boundedIndices) {
        if (bounded.indices.empty() || bounded.indices.size() > 2) {
            return false;
        }
        if (plan.signature.layout == LocalLayoutKind::StencilWindow2D ||
            (param.access == ParamAccessKind::StencilWindow &&
             plan.signature.layout == LocalLayoutKind::StencilWindow2D)) {
            if (bounded.indices.size() != 2) {
                return false;
            }
            continue;
        }
        if (param.access == ParamAccessKind::StencilWindow &&
            plan.signature.layout == LocalLayoutKind::StencilWindow1D) {
            if (bounded.indices.size() != 1) {
                return false;
            }
            continue;
        }
        if (plan.signature.layout == LocalLayoutKind::Contiguous1D ||
            plan.signature.layout == LocalLayoutKind::ReplicatedFullTensor) {
            if (bounded.indices.size() != 1) {
                return false;
            }
            continue;
        }
        if (plan.signature.layout == LocalLayoutKind::RowBlock2D ||
            plan.signature.layout == LocalLayoutKind::RowPartitionFullRow) {
            if (bounded.indices.size() != 2) {
                return false;
            }
            if (plan.signature.layout == LocalLayoutKind::RowPartitionFullRow &&
                (param.tensorDims.size() != 2 || param.tensorDims[0] != 0 ||
                 param.tensorDims[1] != 1)) {
                return false;
            }
            continue;
        }
        return false;
    }
    return true;
}

bool stencilWindow1DBoundedPostUseCoveredByFollowup(
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& param) {
    if (param.postUseSync.kind != PostUseSyncKind::BoundedIndexedRootRead ||
        param.access != ParamAccessKind::StencilWindow ||
        plan.signature.layout != LocalLayoutKind::StencilWindow1D) {
        return true;
    }
    const ParamAccessPlan* reader = stencilWindowReaderParam(plan);
    const ParamAccessPlan* writer = stencilOutputDirectWriterParam(plan);
    if (!reader || !writer || reader->paramIndex != param.paramIndex) {
        return false;
    }
    const int64_t writerExtent =
        staticOneDimShellArgExtent(dacppFile, plan, *writer);
    if (writerExtent <= 0) {
        return false;
    }
    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (!sitePlan.supported || sitePlan.followupMappings.size() != 1) {
        return false;
    }
    const auto& mapping = sitePlan.followupMappings.front();
    if (mapping.writerParamIndex != writer->paramIndex ||
        mapping.readerParamIndex != reader->paramIndex ||
        mapping.targetOffset < 0) {
        return false;
    }
    for (const auto& bounded : param.postUseSync.boundedIndices) {
        if (bounded.indices.size() != 1) {
            return false;
        }
        const int64_t outputIndex =
            bounded.indices[0] - static_cast<int64_t>(mapping.targetOffset);
        if (outputIndex < 0 || outputIndex >= writerExtent) {
            return false;
        }
    }
    return true;
}

void downgradeUnsupportedBoundedPostUse(ShellPartitionPlan& plan,
                                        ParamAccessPlan& param,
                                        DacppFile* dacppFile) {
    if (param.postUseSync.kind != PostUseSyncKind::BoundedIndexedRootRead) {
        return;
    }
    std::string reason;
    if (!layoutSupportsBoundedPostUseSync(plan, param)) {
        reason = "bounded indexed read unsupported for layout";
    } else if (!stencilWindow1DBoundedPostUseCoveredByFollowup(
                   dacppFile, plan, param)) {
        reason = "bounded indexed read outside 1D followup-owned range";
    }
    if (reason.empty()) {
        return;
    }
    param.postUseSync.kind = PostUseSyncKind::FullTensor;
    param.postUseSync.reason = reason;
    param.postUseSync.boundedIndices.clear();
    param.postUseSync.tensor2ArrayStmt = nullptr;
    param.postUseSync.tensor2ArrayTargetName.clear();
}

void logPostUseSyncDecision(const ParamAccessPlan& param) {
    llvm::outs() << "[DACPP][MPI][OR] output " << param.actualTensorName
                 << " post-use="
                 << postUseSyncKindName(param.postUseSync.kind);
    if (param.postUseSync.kind == PostUseSyncKind::BoundedIndexedRootRead) {
        llvm::outs() << " count=" << param.postUseSync.boundedIndices.size();
    }
    if (param.postUseSync.kind == PostUseSyncKind::FullTensor &&
        !param.postUseSync.reason.empty()) {
        llvm::outs() << " fallback reason=" << param.postUseSync.reason;
    } else if (!param.postUseSync.reason.empty()) {
        llvm::outs() << " reason=" << param.postUseSync.reason;
    }
    llvm::outs() << "\n";
}

void annotateOutputSync(ShellPartitionPlan& plan, DacppFile* dacppFile) {
    if (!dacppFile) {
        return;
    }
    for (auto& param : plan.params) {
        if (!paramMayNeedHostPostUseSync(param)) {
            continue;
        }
        param.postUseSync = analyzePostUseSync(
            dacppFile, plan.exprNode.shell, plan.exprNode.calc,
            plan.exprNode.dacExpr, param.actualTensorName);
        downgradeUnsupportedBoundedPostUse(plan, param, dacppFile);
        const OutputSyncRequirement syncRequirement =
            classifyOutputSyncRequirement(dacppFile, param.actualTensorName,
                                          plan.exprNode.dacExpr);
        const bool orStencilDistributedFollowupLowered =
            syncRequirement == OutputSyncRequirement::DistributedFollowup &&
            isShellDerivedStencilLayout(plan.signature.layout);
        if (param.writes &&
            (param.access == ParamAccessKind::OutputDirect ||
             param.access == ParamAccessKind::FixedBlock)) {
            param.broadcastMaterializedOutput =
                syncRequirement != OutputSyncRequirement::RootOnly &&
                syncRequirement != OutputSyncRequirement::RootCentricFollowup &&
                !orStencilDistributedFollowupLowered;
        }
        if ((syncRequirement == OutputSyncRequirement::RootOnly ||
             syncRequirement == OutputSyncRequirement::RootCentricFollowup) &&
            param.access == ParamAccessKind::OutputDirect &&
            plan.signature.layout == LocalLayoutKind::Contiguous1D) {
            const PostUseReductionPlan reduction =
                analyzePostUseReduction(dacppFile, plan.exprNode.dacExpr,
                                        param.actualTensorName);
            if (reduction.supported) {
                param.postUseReductionCountEqOne = true;
                param.postUseReductionResetStmt = reduction.resetStmt;
                param.postUseReductionLoopStmt = reduction.loopStmt;
                param.postUseReductionScalarName = reduction.scalarName;
                param.materializeAfterWrite = false;
                param.broadcastMaterializedOutput = false;
                llvm::outs() << "[DACPP][MPI][OR] output "
                             << param.actualTensorName
                             << " post-use-reduction=accepted kind=count-eq-one scalar="
                             << reduction.scalarName << "\n";
            } else if (!reduction.reason.empty()) {
                llvm::outs() << "[DACPP][MPI][OR] output "
                             << param.actualTensorName
                             << " post-use-reduction=fallback reason="
                             << reduction.reason << "\n";
            }
        }
        if (!param.postUseReductionCountEqOne &&
            (param.postUseSync.kind == PostUseSyncKind::None ||
             param.postUseSync.kind == PostUseSyncKind::BoundedIndexedRootRead)) {
            param.materializeAfterWrite = false;
            param.broadcastMaterializedOutput = false;
        }
        logPostUseSyncDecision(param);
        llvm::outs() << "[DACPP][MPI] output " << param.actualTensorName
                     << " sync="
                     << outputSyncRequirementName(syncRequirement) << "\n";
    }
}

void logCodegenDisabledFallback(const ShellPartitionPlan& plan) {
    const std::string shellName =
        plan.exprNode.shell ? plan.exprNode.shell->getName() : "<null>";
    llvm::outs() << "[DACPP][MPI][OR] expr=" << plan.exprIndex
                 << " shell=" << shellName
                 << " layout=" << localLayoutKindName(plan.signature.layout)
                 << " codegen=disabled fallback=legacy\n";
}

std::vector<std::string> outputTensorNames(const ShellPartitionPlan& plan) {
    std::vector<std::string> outputs;
    for (const auto& param : plan.params) {
        if (param.writes &&
            (param.access == ParamAccessKind::OutputDirect ||
             param.access == ParamAccessKind::FixedBlock)) {
            outputs.push_back(param.actualTensorName);
        }
    }
    return outputs;
}

bool readsTensor(const ShellPartitionPlan& plan, const std::string& tensorName) {
    for (const auto& param : plan.params) {
        if (param.reads && param.actualTensorName == tensorName &&
            param.access != ParamAccessKind::ReplicatedScalar) {
            return true;
        }
    }
    return false;
}

bool paramsProvenDistinct(const ParamAccessPlan& lhs,
                          const ParamAccessPlan& rhs);

bool canAppendToChain(const OperatorResidentChainPlan& chain,
                      const ShellPartitionPlan& next) {
    if (!chain.supported || !next.supported || chain.exprPlans.empty()) {
        return false;
    }
    if (!isCompatibleForChain(chain.signature, next.signature)) {
        return false;
    }
    if (chain.signature.layout != next.signature.layout) {
        return false;
    }
    for (const std::string& output :
         outputTensorNames(chain.exprPlans.back())) {
        if (readsTensor(next, output)) {
            return true;
        }
    }
    return false;
}

const ParamAccessPlan* singleDirectMappedReader(const ShellPartitionPlan& plan) {
    const ParamAccessPlan* result = nullptr;
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::DirectMapped && param.reads &&
            !param.writes) {
            if (result) {
                return nullptr;
            }
            result = &param;
        }
    }
    return result;
}

const ParamAccessPlan* singleOutputDirectWriter(const ShellPartitionPlan& plan) {
    const ParamAccessPlan* result = nullptr;
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::OutputDirect && param.writes &&
            !param.reads) {
            if (result) {
                return nullptr;
            }
            result = &param;
        }
    }
    return result;
}

bool calcBodyIsSimplePointwise(Calc* calc) {
    if (!calc || !calc->getCalcLoc() || !calc->getCalcLoc()->getBody()) {
        return false;
    }
    const auto* compound =
        llvm::dyn_cast_or_null<clang::CompoundStmt>(
            calc->getCalcLoc()->getBody());
    if (!compound || compound->size() == 0) {
        return false;
    }
    for (const clang::Stmt* stmt : compound->body()) {
        if (!stmt) {
            return false;
        }
        const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt);
        if (declStmt) {
            continue;
        }
        const auto* exprStmt = llvm::dyn_cast<clang::Expr>(stmt);
        const clang::Expr* expr = unwrapExtentExpr(exprStmt);
        if (llvm::isa_and_nonnull<clang::BinaryOperator>(expr) ||
            llvm::isa_and_nonnull<clang::CallExpr>(expr) ||
            llvm::isa_and_nonnull<clang::CXXOperatorCallExpr>(expr)) {
            continue;
        }
        return false;
    }
    return true;
}

int downstreamReadCount(const std::vector<ShellPartitionPlan>& plans,
                        std::size_t startPos,
                        const std::string& tensorName) {
    int count = 0;
    for (std::size_t idx = startPos; idx < plans.size(); ++idx) {
        for (const auto& param : plans[idx].params) {
            if (param.reads && param.actualTensorName == tensorName &&
                param.access != ParamAccessKind::ReplicatedScalar) {
                ++count;
            }
        }
    }
    return count;
}

int downstreamReadCountAfterExpr(const std::vector<ShellPartitionPlan>& plans,
                                 int writerExprIndex,
                                 const std::string& tensorName) {
    int count = 0;
    for (const auto& plan : plans) {
        if (plan.exprIndex <= writerExprIndex) {
            continue;
        }
        for (const auto& param : plan.params) {
            if (param.reads && param.actualTensorName == tensorName &&
                param.access != ParamAccessKind::ReplicatedScalar) {
                ++count;
            }
        }
    }
    return count;
}

std::string rowBlock2DPointwiseFusionRejectReason(
    const OperatorResidentChainPlan& chain,
    const std::vector<ShellPartitionPlan>* allPlans = nullptr) {
    if (!chain.supported) {
        return "chain unsupported";
    }
    if (chain.exprPlans.size() < 2) {
        return "requires at least two shells";
    }
    const ShellPartitionPlan& first = chain.exprPlans[0];
    if (first.signature.layout != LocalLayoutKind::RowBlock2D) {
        return "layout is not RowBlock2D";
    }
    for (std::size_t idx = 0; idx < chain.exprPlans.size(); ++idx) {
        const ShellPartitionPlan& plan = chain.exprPlans[idx];
        if (plan.signature.layout != LocalLayoutKind::RowBlock2D) {
            return "layout is not RowBlock2D";
        }
        if (!isCompatibleForChain(first.signature, plan.signature)) {
            return "partition signatures differ";
        }
        const ParamAccessPlan* reader = singleDirectMappedReader(plan);
        const ParamAccessPlan* writer = singleOutputDirectWriter(plan);
        if (!reader || !writer) {
            return "requires one direct reader and one output-only writer per shell";
        }
        if (!paramsProvenDistinct(*reader, *writer)) {
            return "reader/writer aliasing is not proven distinct";
        }
        if (!calcBodyIsSimplePointwise(plan.exprNode.calc)) {
            return "calc body is not simple pointwise";
        }
        if (idx + 1 >= chain.exprPlans.size()) {
            continue;
        }
        const ShellPartitionPlan& next = chain.exprPlans[idx + 1];
        const ParamAccessPlan* nextReader = singleDirectMappedReader(next);
        if (!nextReader || writer->actualTensorName.empty() ||
            writer->actualTensorName != nextReader->actualTensorName) {
            return "intermediate tensor is not the sole downstream reader input";
        }
        const int downstreamReads =
            allPlans ? downstreamReadCountAfterExpr(*allPlans, plan.exprIndex,
                                                    writer->actualTensorName)
                     : downstreamReadCount(chain.exprPlans, idx + 1,
                                           writer->actualTensorName);
        if (downstreamReads != 1) {
            return "intermediate tensor has multiple downstream consumers";
        }
        const bool intermediateOnlyFeedsFollower =
            writer->retainResidentAfterWrite &&
            writer->postUseSync.kind == PostUseSyncKind::FullTensor &&
            writer->postUseSync.reason == "tensor passed to function";
        if ((writer->materializeAfterWrite ||
             writer->postUseSync.kind != PostUseSyncKind::None) &&
            !intermediateOnlyFeedsFollower) {
            return "intermediate tensor needs host sync/materialization";
        }
    }
    return "";
}

void annotateRowBlock2DPointwiseFusion(
    OperatorResidentChainPlan& chain,
    const std::vector<ShellPartitionPlan>* allPlans = nullptr) {
    const std::string reason =
        rowBlock2DPointwiseFusionRejectReason(chain, allPlans);
    chain.fusePointwiseRowBlock2D = reason.empty();
    chain.fuseRejectReason = reason;
    if (chain.signature.layout == LocalLayoutKind::RowBlock2D &&
        chain.exprPlans.size() >= 2) {
        llvm::outs() << "[DACPP][MPI][OR] chain=" << chain.chainId
                     << " rowblock-pointwise-fusion="
                     << (chain.fusePointwiseRowBlock2D ? "accepted"
                                                       : "rejected");
        if (!chain.fusePointwiseRowBlock2D && !reason.empty()) {
            llvm::outs() << " reason=" << reason;
        }
        llvm::outs() << "\n";
    }
}

bool supportedPhaseLayout(LocalLayoutKind layout) {
    // Phase 1/2 plus the currently implemented Phase 3 DFT/gradientSum layouts.
    // Future Phase 3 shapes still fall back through analysis/codegen gating.
    return layout == LocalLayoutKind::Contiguous1D ||
           layout == LocalLayoutKind::RowBlock2D ||
           layout == LocalLayoutKind::ReplicatedFullTensor ||
           layout == LocalLayoutKind::RowPartitionFullRow ||
           layout == LocalLayoutKind::StencilWindow1D ||
           layout == LocalLayoutKind::StencilWindow2D ||
           layout == LocalLayoutKind::FixedBlock;
}

bool directResidentLoopLayout(LocalLayoutKind layout) {
    return layout == LocalLayoutKind::Contiguous1D;
}

bool stencilFullSyncLoopLayout(LocalLayoutKind layout) {
    return layout == LocalLayoutKind::StencilWindow1D ||
           layout == LocalLayoutKind::StencilWindow2D;
}

const char* orLoopLowerKindName(OrLoopLowerKind kind) {
    switch (kind) {
    case OrLoopLowerKind::None:
        return "None";
    case OrLoopLowerKind::Direct1D:
        return "Direct1D";
    case OrLoopLowerKind::RowBlock2D:
        return "RowBlock2D";
    case OrLoopLowerKind::StencilFullSync:
        return "StencilFullSync";
    case OrLoopLowerKind::StencilResidentHalo:
        return "StencilResidentHalo";
    case OrLoopLowerKind::FixedBlockPhaseExchange:
        return "FixedBlockPhaseExchange";
    case OrLoopLowerKind::FixedBlockPhaseExchangeFollower:
        return "FixedBlockPhaseExchangeFollower";
    }
    return "None";
}

const char* loopKindName(const clang::Stmt* stmt) {
    if (llvm::isa_and_nonnull<clang::ForStmt>(stmt)) {
        return "for";
    }
    if (llvm::isa_and_nonnull<clang::WhileStmt>(stmt)) {
        return "while";
    }
    return "unknown";
}

const clang::Stmt* stableOuterLoopForExpr(DacppFile* dacppFile,
                                          const ShellPartitionPlan& plan) {
    if (!dacppFile || !plan.exprNode.dacExpr) {
        return nullptr;
    }
    for (const auto& site : dacppFile->getMPIStencilSites()) {
        if (site.dacExpr == plan.exprNode.dacExpr) {
            return site.outerLoop;
        }
    }
    return nullptr;
}

bool hasOutputDirectParam(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.writes && param.access == ParamAccessKind::OutputDirect) {
            return true;
        }
    }
    return false;
}

bool hasReadWriteOutputDirectParam(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.reads && param.writes &&
            param.access == ParamAccessKind::OutputDirect) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> directReadInputTensorNames(
    const ShellPartitionPlan& plan) {
    std::vector<std::string> names;
    for (const auto& param : plan.params) {
        if (param.reads && !param.writes &&
            param.access == ParamAccessKind::DirectMapped) {
            names.push_back(param.actualTensorName);
        }
    }
    return names;
}

bool directReadAliasesOutputDirectWrite(const ShellPartitionPlan& plan) {
    std::set<std::string> directReads;
    for (const auto& name : directReadInputTensorNames(plan)) {
        if (!name.empty()) {
            directReads.insert(name);
        }
    }
    if (directReads.empty()) {
        return false;
    }
    for (const auto& param : plan.params) {
        if (param.writes && param.access == ParamAccessKind::OutputDirect &&
            directReads.count(param.actualTensorName) != 0) {
            return true;
        }
    }
    return false;
}

const ParamAccessPlan* singleOutputDirectParamAnyMode(
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* result = nullptr;
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::OutputDirect && param.writes) {
            if (result) {
                return nullptr;
            }
            result = &param;
        }
    }
    return result;
}

bool hasResidentReaderOrFullMaterializeContract(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.retainResidentAfterWrite || param.readFromResident) {
            return true;
        }
        if (param.writes &&
            param.access == ParamAccessKind::OutputDirect &&
            param.materializeAfterWrite) {
            return true;
        }
        if (param.writes &&
            param.access == ParamAccessKind::OutputDirect &&
            param.postUseSync.kind == PostUseSyncKind::FullTensor) {
            return true;
        }
    }
    return false;
}

std::string contiguous1DDistributionRejectReason(
    const ShellPartitionPlan& plan,
    std::string* acceptReason) {
    if (acceptReason) {
        acceptReason->clear();
    }
    if (plan.signature.layout != LocalLayoutKind::Contiguous1D) {
        return "requires Contiguous1D layout";
    }
    if (plan.loopLowerCandidate ||
        plan.orLoopLower.kind != OrLoopLowerKind::None) {
        return "loop-lowered/followup contracts require contiguous ownership";
    }
    if (plan.signature.bindOrder.size() != 1 ||
        plan.signature.bindSizes.size() != 1) {
        return "requires single 1D ownership domain";
    }
    const ParamAccessPlan* writer = singleOutputDirectParamAnyMode(plan);
    if (!writer) {
        return "requires exactly one direct output writer";
    }
    if (writer->reads) {
        return "read_write output direct requires contiguous ownership";
    }
    if (directReadAliasesOutputDirectWrite(plan)) {
        return "direct read aliases output direct write";
    }
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            continue;
        }
        if (param.access != ParamAccessKind::DirectMapped &&
            param.access != ParamAccessKind::OutputDirect) {
            return "non-direct parameter requires contiguous ownership";
        }
        if (usesByteTransport(analysisElemType(plan, param))) {
            return "byte-transport element type requires contiguous ownership";
        }
        if (param.bindOrder.size() != 1 ||
            !operator_resident::sameOrder(param.bindOrder,
                                          plan.signature.bindOrder)) {
            return "parameter bind order mismatch";
        }
        if (param.access == ParamAccessKind::DirectMapped &&
            param.tensorDims.size() != 1) {
            return "direct reader extent is not one-to-one 1D";
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.tensorDims.size() != 1) {
            return "output writer extent is not one-to-one 1D";
        }
    }
    if (hasResidentReaderOrFullMaterializeContract(plan)) {
        return "downstream/full-materialize contract requires contiguous ownership";
    }
    if (writer->postUseReductionCountEqOne) {
        if (acceptReason) {
            *acceptReason = "independent-map scalar-reduction";
        }
        return "";
    }
    if (writer->postUseSync.kind == PostUseSyncKind::BoundedIndexedRootRead &&
        !writer->postUseSync.boundedIndices.empty()) {
        if (acceptReason) {
            *acceptReason = "independent-map bounded/small-root-use";
        }
        return "";
    }
    return "unsupported post-use/reduction";
}

bool hasReplicatedScalarParam(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            return true;
        }
    }
    return false;
}

bool exprWritesTensor(const clang::Expr* expr,
                      const std::set<std::string>& tensorNames);
bool stmtWritesAnyTensor(const clang::Stmt* stmt,
                         const std::set<std::string>& tensorNames);

std::vector<std::string> replicatedScalarTensorNames(
    const ShellPartitionPlan& plan) {
    std::vector<std::string> names;
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar &&
            !param.actualTensorName.empty()) {
            names.push_back(param.actualTensorName);
        }
    }
    return names;
}

bool isIntegerZeroExpr(const clang::Expr* expr,
                       clang::ASTContext* context) {
    expr = expr ? expr->IgnoreParenImpCasts() : nullptr;
    if (!expr) {
        return false;
    }
    if (const auto* integer = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
        return integer->getValue().isZero();
    }
    return compactExprText(exprSource(expr, context)) == "0";
}

bool isRootRankGuardExpr(const clang::Expr* expr,
                         clang::ASTContext* context) {
    if (!expr) {
        return false;
    }
    expr = expr->IgnoreParenImpCasts();
    if (const auto* call = llvm::dyn_cast<clang::CallExpr>(expr)) {
        if (const clang::FunctionDecl* callee = call->getDirectCallee()) {
            if (callee->getNameAsString() == "__dacpp_mpi_is_root_rank") {
                return true;
            }
        }
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
        if (binary->isComparisonOp()) {
            const std::string lhs = compactExprText(baseNameFromExpr(binary->getLHS()));
            const std::string rhs = compactExprText(baseNameFromExpr(binary->getRHS()));
            const std::string lhsText =
                compactExprText(exprSource(binary->getLHS(), context));
            const std::string rhsText =
                compactExprText(exprSource(binary->getRHS(), context));
            const auto mentionsRank = [](const std::string& text) {
                return text == "mpi_rank" || text == "__dacpp_mpi_rank" ||
                       text == "rank" ||
                       text.find("mpi_rank") != std::string::npos;
            };
            if ((mentionsRank(lhs) || mentionsRank(lhsText)) &&
                isIntegerZeroExpr(binary->getRHS(), context)) {
                return true;
            }
            if ((mentionsRank(rhs) || mentionsRank(rhsText)) &&
                isIntegerZeroExpr(binary->getLHS(), context)) {
                return true;
            }
        }
    }
    return false;
}

bool stmtWritesAnyNamedTensorAtTopLevel(const clang::Stmt* stmt,
                                        const std::set<std::string>& names) {
    if (!stmt || names.empty()) {
        return false;
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
        return binary->isAssignmentOp() &&
               exprWritesTensor(binary->getLHS(), names);
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->isAssignmentOp() && opCall->getNumArgs() > 0) {
            return exprWritesTensor(opCall->getArg(0), names);
        }
        if ((opCall->getOperator() == clang::OO_PlusPlus ||
             opCall->getOperator() == clang::OO_MinusMinus) &&
            opCall->getNumArgs() > 0) {
            return exprWritesTensor(opCall->getArg(0), names);
        }
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(stmt)) {
        return unary->isIncrementDecrementOp() &&
               exprWritesTensor(unary->getSubExpr(), names);
    }
    return false;
}

bool stmtContainsRootOnlyGuard(const clang::Stmt* stmt,
                               clang::ASTContext* context) {
    if (!stmt) {
        return false;
    }
    if (const auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
        if (isRootRankGuardExpr(ifStmt->getCond(), context)) {
            return true;
        }
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (stmtContainsRootOnlyGuard(child, context)) {
            return true;
        }
    }
    return false;
}

bool loopBodyHasOnlyUnguardedTopLevelScalarWrites(
    DacppFile* dacppFile,
    const clang::Stmt* loop,
    const ShellPartitionPlan& plan,
    std::string& reason) {
    if (!dacppFile || !dacppFile->getContext() || !loop ||
        !plan.exprNode.dacExpr) {
        reason = "analysis context unavailable";
        return false;
    }
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    const auto scalarNames = replicatedScalarTensorNames(plan);
    if (scalarNames.empty()) {
        reason = "no replicated scalar";
        return false;
    }
    std::set<std::string> scalarSet(scalarNames.begin(), scalarNames.end());
    const clang::CompoundStmt* compound = nullptr;
    if (const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(loop)) {
        compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(
            forStmt->getBody());
    } else if (const auto* whileStmt = llvm::dyn_cast<clang::WhileStmt>(loop)) {
        compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(
            whileStmt->getBody());
    }
    if (!compound) {
        reason = "loop body is not compound";
        return false;
    }
    bool sawScalarWrite = false;
    for (const clang::Stmt* child : compound->body()) {
        if (!child) {
            continue;
        }
        if (sourceRangeContains(sourceManager, child->getSourceRange(),
                                plan.exprNode.dacExpr->getSourceRange())) {
            continue;
        }
        if (!stmtWritesAnyTensor(child, scalarSet)) {
            continue;
        }
        if (!stmtWritesAnyNamedTensorAtTopLevel(child, scalarSet)) {
            reason = "scalar tensor write is nested or compound";
            return false;
        }
        if (stmtContainsRootOnlyGuard(child, dacppFile->getContext())) {
            reason = "scalar tensor write is root-guarded";
            return false;
        }
        sawScalarWrite = true;
    }
    if (!sawScalarWrite) {
        reason = "no scalar tensor write in loop body";
        return false;
    }
    reason = "all ranks update replicated scalar in loop";
    return true;
}

const ParamAccessPlan* singleReplicatedScalarParam(
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* result = nullptr;
    for (const auto& param : plan.params) {
        if (param.access != ParamAccessKind::ReplicatedScalar) {
            continue;
        }
        if (result) {
            return nullptr;
        }
        result = &param;
    }
    return result;
}

const clang::CompoundStmt* compoundBodyForLoop(const clang::Stmt* loop) {
    if (const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(loop)) {
        return llvm::dyn_cast_or_null<clang::CompoundStmt>(forStmt->getBody());
    }
    if (const auto* whileStmt =
            llvm::dyn_cast_or_null<clang::WhileStmt>(loop)) {
        return llvm::dyn_cast_or_null<clang::CompoundStmt>(whileStmt->getBody());
    }
    return llvm::dyn_cast_or_null<clang::CompoundStmt>(loop);
}

const clang::Expr* loopConditionExpr(const clang::Stmt* loop) {
    if (const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(loop)) {
        return forStmt->getCond();
    }
    if (const auto* whileStmt =
            llvm::dyn_cast_or_null<clang::WhileStmt>(loop)) {
        return whileStmt->getCond();
    }
    return nullptr;
}

std::string rewriteScalarTensorExprForLocal(std::string text,
                                            const std::string& tensorName,
                                            const std::string& localName) {
    if (tensorName.empty() || localName.empty()) {
        return text;
    }
    text = compactExprText(text);
    auto replaceAll = [](std::string& target,
                         const std::string& from,
                         const std::string& to) {
        if (from.empty()) {
            return;
        }
        std::size_t pos = 0;
        while ((pos = target.find(from, pos)) != std::string::npos) {
            target.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replaceAll(text, tensorName + "[0]", localName);
    replaceAll(text, tensorName, localName);
    return text;
}

bool isTopLevelScalarWriteStmt(const clang::Stmt* stmt,
                               const std::string& scalarTensorName,
                               std::string& sourceText,
                               clang::ASTContext* context) {
    sourceText.clear();
    if (!stmt || scalarTensorName.empty() || !context) {
        return false;
    }
    std::set<std::string> names{scalarTensorName};
    if (!stmtWritesAnyNamedTensorAtTopLevel(stmt, names)) {
        return false;
    }
    sourceText = clang::Lexer::getSourceText(
                     clang::CharSourceRange::getTokenRange(
                         stmt->getSourceRange()),
                     context->getSourceManager(), context->getLangOpts())
                     .str();
    sourceText = trim(sourceText);
    if (sourceText.empty()) {
        return false;
    }
    if (!sourceText.empty() && sourceText.back() != ';') {
        sourceText += ";";
    }
    return true;
}

const ParamAccessPlan* loopLowerOutputDirectParam(
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* result = nullptr;
    for (const auto& param : plan.params) {
        if (!param.writes || param.access != ParamAccessKind::OutputDirect ||
            param.actualTensorName.empty()) {
            continue;
        }
        if (result) {
            return nullptr;
        }
        result = &param;
    }
    return result;
}

bool decomposeTopLevelSubscript(const clang::Expr* expr,
                                std::string& baseName,
                                const clang::Expr*& indexExpr) {
    baseName.clear();
    indexExpr = nullptr;
    expr = unwrapExtentExpr(expr);
    const auto* opCall = llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr);
    if (!opCall || opCall->getOperator() != clang::OO_Subscript ||
        opCall->getNumArgs() < 2) {
        return false;
    }
    baseName = actualTensorNameForArg(opCall->getArg(0), nullptr);
    if (baseName.empty()) {
        baseName = baseNameFromExpr(opCall->getArg(0));
    }
    indexExpr = opCall->getArg(1);
    return !baseName.empty() && indexExpr;
}

bool firstSubscriptIndexForBase(const clang::Expr* expr,
                                std::string& baseName,
                                const clang::Expr*& firstIndexExpr) {
    baseName.clear();
    firstIndexExpr = nullptr;
    std::vector<const clang::Expr*> reversedIndices;
    const clang::Expr* current = unwrapExtentExpr(expr);
    while (current) {
        const auto* opCall =
            llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(current);
        if (!opCall || opCall->getOperator() != clang::OO_Subscript ||
            opCall->getNumArgs() < 2) {
            baseName = actualTensorNameForArg(current, nullptr);
            if (baseName.empty()) {
                baseName = baseNameFromExpr(current);
            }
            break;
        }
        reversedIndices.push_back(opCall->getArg(1));
        current = unwrapExtentExpr(opCall->getArg(0));
    }
    if (baseName.empty() || reversedIndices.empty()) {
        return false;
    }
    firstIndexExpr = reversedIndices.back();
    return firstIndexExpr != nullptr;
}

const ParamAccessPlan* findParamByActualTensorName(
    const ShellPartitionPlan& plan,
    const std::string& tensorName) {
    if (tensorName.empty()) {
        return nullptr;
    }
    for (const auto& param : plan.params) {
        if (param.actualTensorName == tensorName) {
            return &param;
        }
    }
    return nullptr;
}

bool findLoopOutputWriteback(const ShellPartitionPlan& plan,
                             const clang::CompoundStmt* loopBody,
                             clang::ASTContext* context,
                             std::string& hostTensorName,
                             std::string& hostTensorType,
                             std::string& rowIndexExpr,
                             std::string& reason) {
    hostTensorName.clear();
    hostTensorType.clear();
    rowIndexExpr.clear();
    const ParamAccessPlan* output = loopLowerOutputDirectParam(plan);
    if (!output) {
        reason = "requires exactly one output direct tensor";
        return false;
    }
    if (!loopBody || !context || !plan.exprNode.dacExpr) {
        reason = "loop body unavailable";
        return false;
    }
    const auto& sourceManager = context->getSourceManager();
    bool sawDac = false;
    for (const clang::Stmt* child : loopBody->body()) {
        if (!child) {
            continue;
        }
        if (sourceRangeContains(sourceManager, child->getSourceRange(),
                                plan.exprNode.dacExpr->getSourceRange())) {
            sawDac = true;
            continue;
        }
        if (!sawDac) {
            continue;
        }

        const clang::Stmt* candidateStmt = child;
        if (const auto* exprStmt = llvm::dyn_cast<clang::Expr>(candidateStmt)) {
            if (const clang::Expr* unwrapped = unwrapExtentExpr(exprStmt)) {
                candidateStmt = unwrapped;
            }
        }
        const clang::Expr* lhs = nullptr;
        const clang::Expr* rhs = nullptr;
        if (const auto* binary =
                llvm::dyn_cast<clang::BinaryOperator>(candidateStmt)) {
            if (!binary->isAssignmentOp()) {
                continue;
            }
            lhs = binary->getLHS();
            rhs = binary->getRHS();
        } else if (const auto* opCall =
                       llvm::dyn_cast<clang::CXXOperatorCallExpr>(
                           candidateStmt)) {
            if (!opCall->isAssignmentOp() || opCall->getNumArgs() < 2) {
                continue;
            }
            lhs = opCall->getArg(0);
            rhs = opCall->getArg(1);
        } else {
            continue;
        }

        if (actualTensorKeyForArg(rhs, context).name !=
            output->actualTensorName) {
            continue;
        }
        const clang::Expr* indexExpr = nullptr;
        std::string candidateHost;
        if (!decomposeTopLevelSubscript(lhs, candidateHost, indexExpr)) {
            reason = "writeback lhs is not a tensor row subscript";
            return false;
        }
        if (candidateHost == output->actualTensorName) {
            reason = "writeback host aliases output tensor";
            return false;
        }
        const std::string indexText = exprSource(indexExpr, context);
        if (candidateHost.empty() || indexText.empty()) {
            reason = "writeback row expression unavailable";
            return false;
        }
        hostTensorType =
            lhs ? lhs->IgnoreParenImpCasts()->getType().getAsString() : "";
        class RowExprDeclVisitor
            : public clang::RecursiveASTVisitor<RowExprDeclVisitor> {
        public:
            RowExprDeclVisitor(const ShellPartitionPlan& plan,
                               clang::ASTContext* context)
                : Plan(plan), Context(context) {}

            std::set<std::string> AllowedNames;
            bool Valid = true;
            std::string RejectReason;

            bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* call) {
                if (!call || call->getOperator() != clang::OO_Subscript ||
                    call->getNumArgs() < 2) {
                    return true;
                }
                const std::string base = actualTensorNameForArg(call->getArg(0),
                                                                nullptr);
                if (base.empty()) {
                    return true;
                }
                const ParamAccessPlan* param =
                    findParamByActualTensorName(Plan, base);
                if (!param) {
                    return true;
                }
                int64_t index = -1;
                if (param->access == ParamAccessKind::ReplicatedScalar &&
                    evaluateInt64Expr(call->getArg(1), Context, index) &&
                    index == 0) {
                    return true;
                }
                Valid = false;
                RejectReason =
                    "writeback row uses non-replicated tensor subscript";
                return true;
            }

            bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
                if (!dre || !dre->getDecl()) {
                    return true;
                }
                const std::string name = dre->getDecl()->getNameAsString();
                if (AllowedNames.count(name) != 0) {
                    return true;
                }
                if (const auto* varDecl =
                        llvm::dyn_cast<clang::VarDecl>(dre->getDecl())) {
                    if (varDecl->isFileVarDecl()) {
                        return true;
                    }
                }
                if (llvm::isa<clang::FunctionDecl>(dre->getDecl())) {
                    return true;
                }
                Valid = false;
                RejectReason = "writeback row references non-shell local " + name;
                return true;
            }

            bool VisitCallExpr(clang::CallExpr* call) {
                if (!call) {
                    return true;
                }
                if (const auto* opCall =
                        llvm::dyn_cast<clang::CXXOperatorCallExpr>(call)) {
                    if (opCall->getOperator() == clang::OO_Subscript) {
                        return true;
                    }
                }
                Valid = false;
                RejectReason = "writeback row uses function call";
                return true;
            }

            const ShellPartitionPlan& Plan;
            clang::ASTContext* Context = nullptr;
        };
        RowExprDeclVisitor declVisitor(plan, context);
        for (const auto& param : plan.params) {
            if (!param.actualTensorName.empty()) {
                declVisitor.AllowedNames.insert(param.actualTensorName);
            }
        }
        declVisitor.TraverseStmt(const_cast<clang::Expr*>(indexExpr));
        if (!declVisitor.Valid) {
            reason = declVisitor.RejectReason;
            return false;
        }
        hostTensorName = candidateHost;
        rowIndexExpr = indexText;
        reason = "matched loop row writeback";
        return true;
    }
    reason = "loop output row writeback not found";
    return false;
}

bool matchLoopOutputWritebackStmt(const ShellPartitionPlan& plan,
                                  const clang::Stmt* child,
                                  clang::ASTContext* context,
                                  std::string& hostTensorName,
                                  std::string& hostTensorType,
                                  std::string& rowIndexExpr,
                                  std::string& reason) {
    hostTensorName.clear();
    hostTensorType.clear();
    rowIndexExpr.clear();
    const ParamAccessPlan* output = loopLowerOutputDirectParam(plan);
    if (!output) {
        reason = "requires exactly one output direct tensor";
        return false;
    }
    if (!child || !context) {
        reason = "loop body statement unavailable";
        return false;
    }

    const clang::Stmt* candidateStmt = child;
    if (const auto* exprStmt = llvm::dyn_cast<clang::Expr>(candidateStmt)) {
        if (const clang::Expr* unwrapped = unwrapExtentExpr(exprStmt)) {
            candidateStmt = unwrapped;
        }
    }
    const clang::Expr* lhs = nullptr;
    const clang::Expr* rhs = nullptr;
    if (const auto* binary =
            llvm::dyn_cast<clang::BinaryOperator>(candidateStmt)) {
        if (!binary->isAssignmentOp()) {
            reason = "statement is not assignment";
            return false;
        }
        lhs = binary->getLHS();
        rhs = binary->getRHS();
    } else if (const auto* opCall =
                   llvm::dyn_cast<clang::CXXOperatorCallExpr>(
                       candidateStmt)) {
        if (!opCall->isAssignmentOp() || opCall->getNumArgs() < 2) {
            reason = "statement is not assignment";
            return false;
        }
        lhs = opCall->getArg(0);
        rhs = opCall->getArg(1);
    } else {
        reason = "statement is not assignment";
        return false;
    }

    if (actualTensorKeyForArg(rhs, context).name !=
        output->actualTensorName) {
        reason = "assignment rhs is not output tensor";
        return false;
    }
    const clang::Expr* indexExpr = nullptr;
    std::string candidateHost;
    if (!decomposeTopLevelSubscript(lhs, candidateHost, indexExpr)) {
        reason = "writeback lhs is not a tensor row subscript";
        return false;
    }
    if (candidateHost == output->actualTensorName) {
        reason = "writeback host aliases output tensor";
        return false;
    }
    const std::string indexText = exprSource(indexExpr, context);
    if (candidateHost.empty() || indexText.empty()) {
        reason = "writeback row expression unavailable";
        return false;
    }
    hostTensorType =
        lhs ? lhs->IgnoreParenImpCasts()->getType().getAsString() : "";
    class RowExprDeclVisitor
        : public clang::RecursiveASTVisitor<RowExprDeclVisitor> {
    public:
        RowExprDeclVisitor(const ShellPartitionPlan& plan,
                           clang::ASTContext* context)
            : Plan(plan), Context(context) {}

        std::set<std::string> AllowedNames;
        bool Valid = true;
        std::string RejectReason;

        bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* call) {
            if (!call || call->getOperator() != clang::OO_Subscript ||
                call->getNumArgs() < 2) {
                return true;
            }
            const std::string base =
                actualTensorNameForArg(call->getArg(0), nullptr);
            if (base.empty()) {
                return true;
            }
            const ParamAccessPlan* param =
                findParamByActualTensorName(Plan, base);
            if (!param) {
                return true;
            }
            int64_t index = -1;
            if (param->access == ParamAccessKind::ReplicatedScalar &&
                evaluateInt64Expr(call->getArg(1), Context, index) &&
                index == 0) {
                return true;
            }
            Valid = false;
            RejectReason =
                "writeback row uses non-replicated tensor subscript";
            return true;
        }

        bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
            if (!dre || !dre->getDecl()) {
                return true;
            }
            const std::string name = dre->getDecl()->getNameAsString();
            if (AllowedNames.count(name) != 0) {
                return true;
            }
            if (const auto* varDecl =
                    llvm::dyn_cast<clang::VarDecl>(dre->getDecl())) {
                if (varDecl->isFileVarDecl()) {
                    return true;
                }
            }
            if (llvm::isa<clang::FunctionDecl>(dre->getDecl())) {
                return true;
            }
            Valid = false;
            RejectReason = "writeback row references non-shell local " + name;
            return true;
        }

        bool VisitCallExpr(clang::CallExpr* call) {
            if (!call) {
                return true;
            }
            if (const auto* opCall =
                    llvm::dyn_cast<clang::CXXOperatorCallExpr>(call)) {
                if (opCall->getOperator() == clang::OO_Subscript) {
                    return true;
                }
            }
            Valid = false;
            RejectReason = "writeback row uses function call";
            return true;
        }

        const ShellPartitionPlan& Plan;
        clang::ASTContext* Context = nullptr;
    };
    RowExprDeclVisitor declVisitor(plan, context);
    for (const auto& param : plan.params) {
        if (!param.actualTensorName.empty()) {
            declVisitor.AllowedNames.insert(param.actualTensorName);
        }
    }
    declVisitor.TraverseStmt(const_cast<clang::Expr*>(indexExpr));
    if (!declVisitor.Valid) {
        reason = declVisitor.RejectReason;
        return false;
    }
    hostTensorName = candidateHost;
    rowIndexExpr = indexText;
    reason = "matched loop row writeback";
    return true;
}

class FixedHostTensorRowUseVisitor
    : public clang::RecursiveASTVisitor<FixedHostTensorRowUseVisitor> {
public:
    FixedHostTensorRowUseVisitor(std::string hostTensorName,
                                 clang::ASTContext* context)
        : HostTensorName(std::move(hostTensorName)), Context(context) {}

    bool FullUse = false;
    std::string Reason;
    std::set<int64_t> Rows;

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call) {
            return true;
        }
        const clang::CXXMethodDecl* method = call->getMethodDecl();
        if (!method || method->getNameAsString() != "print") {
            return true;
        }
        int64_t row = -1;
        if (extractFixedRow(call->getImplicitObjectArgument(), row)) {
            Rows.insert(row);
        } else if (exprMentionsHostTensor(call->getImplicitObjectArgument())) {
            recordFull("host tensor print is not fixed-row");
        }
        return true;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (!dre || !dre->getDecl() ||
            dre->getDecl()->getNameAsString() != HostTensorName) {
            return true;
        }
        int64_t row = -1;
        if (isInsideAcceptedFixedRow(dre, row)) {
            Rows.insert(row);
            return true;
        }
        recordFull("host tensor use is not fixed-row");
        return true;
    }

    bool VisitCallExpr(clang::CallExpr* call) {
        if (!call) {
            return true;
        }
        if (const auto* opCall =
                llvm::dyn_cast<clang::CXXOperatorCallExpr>(call)) {
            if (opCall->getOperator() == clang::OO_Subscript ||
                opCall->getOperator() == clang::OO_LessLess) {
                return true;
            }
        }
        if (const auto* member =
                llvm::dyn_cast<clang::CXXMemberCallExpr>(call)) {
            const clang::CXXMethodDecl* method = member->getMethodDecl();
            if (method && method->getNameAsString() == "print") {
                return true;
            }
        }
        for (const clang::Expr* arg : call->arguments()) {
            if (exprMentionsHostTensor(arg)) {
                recordFull("host tensor passed to function");
                break;
            }
        }
        return true;
    }

private:
    std::string HostTensorName;
    clang::ASTContext* Context = nullptr;

    void recordFull(const std::string& reason) {
        if (!FullUse) {
            Reason = reason;
        }
        FullUse = true;
    }

    bool extractFixedRow(const clang::Expr* expr, int64_t& row) const {
        row = -1;
        const clang::Expr* firstIndex = nullptr;
        std::string base;
        if (!firstSubscriptIndexForBase(expr, base, firstIndex) ||
            base != HostTensorName) {
            return false;
        }
        int64_t value = -1;
        if (!evaluateInt64Expr(firstIndex, Context, value) || value < 0) {
            return false;
        }
        row = value;
        return true;
    }

    bool exprIsAcceptedFixedRow(const clang::Expr* expr) const {
        int64_t row = -1;
        if (extractFixedRow(expr, row)) {
            return true;
        }
        return false;
    }

    bool exprMentionsHostTensor(const clang::Expr* expr) const {
        if (!expr) {
            return false;
        }
        if (actualTensorNameForArg(expr, Context) == HostTensorName ||
            baseNameFromExpr(expr) == HostTensorName) {
            return true;
        }
        for (const clang::Stmt* child : expr->children()) {
            if (const auto* childExpr =
                    llvm::dyn_cast_or_null<clang::Expr>(child)) {
                if (exprMentionsHostTensor(childExpr)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool isInsideAcceptedFixedRow(const clang::DeclRefExpr* dre,
                                  int64_t& row) const {
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
                if (extractFixedRow(expr, row)) {
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

LoopLoweredSelectiveMaterializePlan analyzeLoopLowerSelectiveMaterialize(
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan,
    const clang::Stmt* outerLoop) {
    LoopLoweredSelectiveMaterializePlan result;
    if (!dacppFile || !dacppFile->getContext() || !outerLoop ||
        !plan.exprNode.dacExpr) {
        result.reason = "analysis context unavailable";
        return result;
    }
    const ParamAccessPlan* output = loopLowerOutputDirectParam(plan);
    if (!output) {
        result.reason = "requires exactly one output direct tensor";
        return result;
    }
    const clang::CompoundStmt* loopBody = compoundBodyForLoop(outerLoop);
    std::string hostTensorName;
    std::string hostTensorType;
    std::string rowIndexExpr;
    std::string writebackReason;
    if (!findLoopOutputWriteback(plan, loopBody, dacppFile->getContext(),
                                 hostTensorName, hostTensorType, rowIndexExpr,
                                 writebackReason)) {
        result.reason = writebackReason;
        return result;
    }

    const auto parents = dacppFile->getContext()->getParents(*outerLoop);
    if (parents.empty()) {
        result.reason = "loop parent unavailable";
        return result;
    }
    const auto* parentCompound = parents[0].get<clang::CompoundStmt>();
    if (!parentCompound) {
        parentCompound =
            llvm::dyn_cast_or_null<clang::CompoundStmt>(
                parents[0].get<clang::Stmt>());
    }
    if (!parentCompound) {
        result.reason = "loop parent compound unavailable";
        return result;
    }
    bool sawLoop = false;
    std::set<int64_t> rows;
    for (const clang::Stmt* child : parentCompound->body()) {
        if (!child) {
            continue;
        }
        if (child == outerLoop ||
            stmtSourceRangeContains(dacppFile->getContext()->getSourceManager(),
                                    child, outerLoop)) {
            sawLoop = true;
            continue;
        }
        if (!sawLoop) {
            continue;
        }
        FixedHostTensorRowUseVisitor visitor(hostTensorName,
                                             dacppFile->getContext());
        visitor.TraverseStmt(const_cast<clang::Stmt*>(child));
        if (visitor.FullUse) {
            result.reason = visitor.Reason.empty()
                                ? "unknown host tensor post-loop use"
                                : visitor.Reason;
            return result;
        }
        rows.insert(visitor.Rows.begin(), visitor.Rows.end());
    }
    if (rows.size() != 1) {
        result.reason = rows.empty() ? "no fixed host row post-loop use"
                                     : "multiple fixed host rows post-loop";
        return result;
    }
    result.enabled = true;
    result.outputTensorName = output->actualTensorName;
    result.hostTensorName = hostTensorName;
    result.hostTensorType = hostTensorType;
    result.rowIndexExpr = rowIndexExpr;
    result.targetRow = *rows.begin();
    result.reason = "loop row writeback and fixed post-loop row use";
    return result;
}

LoopLoweredDeviceTimeLoopPlan analyzeLoopLowerDeviceTimeLoop(
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan,
    const clang::Stmt* outerLoop) {
    LoopLoweredDeviceTimeLoopPlan result;
    if (!dacppFile || !dacppFile->getContext() || !outerLoop ||
        !plan.exprNode.dacExpr) {
        result.reason = "analysis context unavailable";
        return result;
    }
    if (!plan.loopLowerCandidate) {
        result.reason = "loop-lowered direct candidate required";
        return result;
    }
    if (plan.signature.layout != LocalLayoutKind::Contiguous1D) {
        result.reason = "requires Contiguous1D layout";
        return result;
    }
    if (!plan.loopLowerReplicatedScalarLocalRefresh) {
        result.reason = plan.loopLowerScalarRefreshReason.empty()
                            ? "requires local replicated scalar refresh"
                            : plan.loopLowerScalarRefreshReason;
        return result;
    }
    if (!plan.loopLowerSelectiveMaterialize.enabled) {
        result.reason =
            plan.loopLowerSelectiveMaterialize.reason.empty()
                ? "requires selective-row materialization proof"
                : plan.loopLowerSelectiveMaterialize.reason;
        return result;
    }
    const ParamAccessPlan* scalar = singleReplicatedScalarParam(plan);
    if (!scalar || scalar->actualTensorName.empty()) {
        result.reason = "requires exactly one replicated scalar";
        return result;
    }
    const clang::CompoundStmt* loopBody = compoundBodyForLoop(outerLoop);
    if (!loopBody) {
        result.reason = "loop body is not compound";
        return result;
    }
    const clang::Expr* cond = loopConditionExpr(outerLoop);
    const std::string condText = exprSource(cond, dacppFile->getContext());
    if (condText.empty()) {
        result.reason = "loop condition unavailable";
        return result;
    }

    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    bool sawDac = false;
    bool sawWriteback = false;
    bool sawScalarUpdate = false;
    std::string scalarUpdate;
    for (const clang::Stmt* child : loopBody->body()) {
        if (!child) {
            continue;
        }
        if (sourceRangeContains(sourceManager, child->getSourceRange(),
                                plan.exprNode.dacExpr->getSourceRange())) {
            if (sawDac) {
                result.reason = "multiple DAC statements in loop body";
                return result;
            }
            sawDac = true;
            continue;
        }
        std::string hostTensorName;
        std::string hostTensorType;
        std::string rowIndexExpr;
        std::string writebackReason;
        if (sawDac &&
            matchLoopOutputWritebackStmt(plan, child, dacppFile->getContext(),
                                         hostTensorName, hostTensorType,
                                         rowIndexExpr,
                                         writebackReason) &&
            hostTensorName ==
                plan.loopLowerSelectiveMaterialize.hostTensorName &&
            compactExprText(rowIndexExpr) ==
                compactExprText(
                    plan.loopLowerSelectiveMaterialize.rowIndexExpr)) {
            if (sawWriteback) {
                result.reason = "multiple output writebacks in loop body";
                return result;
            }
            sawWriteback = true;
            continue;
        }
        std::string updateText;
        if (isTopLevelScalarWriteStmt(child, scalar->actualTensorName,
                                      updateText, dacppFile->getContext())) {
            if (!sawWriteback) {
                result.reason = "scalar update appears before writeback";
                return result;
            }
            if (sawScalarUpdate) {
                result.reason = "multiple replicated scalar updates";
                return result;
            }
            scalarUpdate = updateText;
            sawScalarUpdate = true;
            continue;
        }
        result.reason = "unsupported statement inside scalar-refresh loop";
        return result;
    }
    if (!sawDac || !sawWriteback || !sawScalarUpdate) {
        result.reason = "requires DAC, output writeback, and scalar update";
        return result;
    }
    const std::string scalarLocal = "__or_time_scalar_" + scalar->calcParamName;
    result.enabled = true;
    result.scalarTensorName = scalar->actualTensorName;
    result.conditionExpr = rewriteScalarTensorExprForLocal(
        condText, scalar->actualTensorName, scalarLocal);
    result.updateStmt = rewriteScalarTensorExprForLocal(
        scalarUpdate, scalar->actualTensorName, scalarLocal);
    result.reason =
        "scalar-refresh independent items with selective-row materialization";
    return result;
}

bool loopContainsMultipleDacExprs(DacppFile* dacppFile,
                                  const clang::Stmt* loop,
                                  const ShellPartitionPlan& plan) {
    if (!dacppFile || !dacppFile->getContext() || !loop) {
        return true;
    }
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    int count = 0;
    for (const auto* candidate : dacppFile->dacExprs) {
        if (candidate &&
            sourceRangeContains(sourceManager, loop->getSourceRange(),
                                candidate->getSourceRange())) {
            ++count;
        }
    }
    return count != 1 || !sourceRangeContains(
                             sourceManager, loop->getSourceRange(),
                             plan.exprNode.dacExpr->getSourceRange());
}

bool loopContainsOnlyDacAndLoweredStencilPostStmts(
    DacppFile* dacppFile,
    const clang::Stmt* loop,
    const ShellPartitionPlan& plan) {
    if (!dacppFile || !loop || !plan.exprNode.dacExpr ||
        !plan.exprNode.shell || !plan.exprNode.calc) {
        return false;
    }
    const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(loop);
    if (const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(loop)) {
        compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(
            forStmt->getBody());
    } else if (const auto* whileStmt = llvm::dyn_cast<clang::WhileStmt>(loop)) {
        compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(
            whileStmt->getBody());
    }
    if (!compound || !dacppFile->getContext()) {
        return false;
    }
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    std::set<const clang::Stmt*> allowedPostStmts;
    for (const auto& region : collectDistributedFollowupRegions(
             dacppFile, plan.exprNode.shell, plan.exprNode.calc,
             plan.exprNode.dacExpr)) {
        allowedPostStmts.insert(region.stmt);
    }
    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (sitePlan.supported && !sitePlan.hasRootBridge) {
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            allowedPostStmts.insert(stmt);
        }
    }
    for (const clang::Stmt* child : compound->body()) {
        if (!child) {
            continue;
        }
        if (allowedPostStmts.count(child) != 0) {
            continue;
        }
        if (sourceRangeContains(sourceManager, child->getSourceRange(),
                                plan.exprNode.dacExpr->getSourceRange())) {
            continue;
        }
        return false;
    }
    return true;
}

bool shellArgsDeclaredBeforeLoop(DacppFile* dacppFile,
                                 const clang::Stmt* loop,
                                 const ShellPartitionPlan& plan) {
    if (!dacppFile || !dacppFile->getContext() || !loop ||
        !plan.exprNode.dacExpr) {
        return false;
    }
    const clang::CallExpr* shellCall = getShellCallExpr(plan.exprNode.dacExpr);
    if (!shellCall) {
        return false;
    }
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    for (const clang::Expr* arg : shellCall->arguments()) {
        if (!arg) {
            return false;
        }
        const auto* declRef =
            llvm::dyn_cast<clang::DeclRefExpr>(arg->IgnoreParenImpCasts());
        if (!declRef) {
            return false;
        }
        const auto* varDecl =
            llvm::dyn_cast_or_null<clang::VarDecl>(declRef->getDecl());
        if (!varDecl) {
            return false;
        }
        if (!varDecl->isFileVarDecl() &&
            !sourceManager.isBeforeInTranslationUnit(
                varDecl->getBeginLoc(), loop->getBeginLoc())) {
            return false;
        }
    }
    return true;
}

bool isStencilWindow1DReader(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::StencilWindow &&
           param.reads &&
           !param.writes &&
           param.tensorDims.size() == 1 &&
           param.tensorDims[0] == 0;
}

bool isStencilWindow2DReader(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::StencilWindow &&
           param.reads &&
           !param.writes &&
           param.tensorDims.size() == 2 &&
           param.tensorDims[0] == 0 &&
           param.tensorDims[1] == 1;
}

bool isStencilWindowReaderForLayout(const ParamAccessPlan& param,
                                    LocalLayoutKind layout) {
    if (layout == LocalLayoutKind::StencilWindow1D) {
        return isStencilWindow1DReader(param);
    }
    if (layout == LocalLayoutKind::StencilWindow2D) {
        return isStencilWindow2DReader(param);
    }
    return false;
}

bool isStencilOutputDirectWriter(const ParamAccessPlan& param,
                                 LocalLayoutKind layout) {
    if (layout == LocalLayoutKind::StencilWindow1D) {
        return param.access == ParamAccessKind::OutputDirect &&
               param.writes &&
               !param.reads &&
               param.tensorDims.size() == 1 &&
               param.tensorDims[0] == 0;
    }
    if (layout == LocalLayoutKind::StencilWindow2D) {
        return param.access == ParamAccessKind::OutputDirect &&
               param.writes &&
               !param.reads &&
               param.tensorDims.size() == 2 &&
               param.tensorDims[0] == 0 &&
               param.tensorDims[1] == 1;
    }
    return param.access == ParamAccessKind::OutputDirect &&
           param.writes &&
           !param.reads;
}

bool isSupportedStencilScalarReader(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::ReplicatedScalar &&
           param.reads &&
           !param.writes;
}

bool isSupportedStencilDirectReader(const ParamAccessPlan& param) {
    return param.access == ParamAccessKind::DirectMapped &&
           param.reads &&
           !param.writes;
}

bool exprWritesTensor(const clang::Expr* expr,
                      const std::set<std::string>& tensorNames);

bool stmtWritesAnyTensor(const clang::Stmt* stmt,
                         const std::set<std::string>& tensorNames);

bool stmtWritesAnyTensorExcept(
    const clang::Stmt* stmt,
    const std::set<std::string>& tensorNames,
    const std::set<const clang::Stmt*>& ignoredStmts) {
    if (!stmt || tensorNames.empty() || ignoredStmts.count(stmt) != 0) {
        return false;
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
        if (binary->isAssignmentOp() &&
            exprWritesTensor(binary->getLHS(), tensorNames)) {
            return true;
        }
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->isAssignmentOp() && opCall->getNumArgs() > 0 &&
            exprWritesTensor(opCall->getArg(0), tensorNames)) {
            return true;
        }
        if ((opCall->getOperator() == clang::OO_PlusPlus ||
             opCall->getOperator() == clang::OO_MinusMinus) &&
            opCall->getNumArgs() > 0 &&
            exprWritesTensor(opCall->getArg(0), tensorNames)) {
            return true;
        }
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(stmt)) {
        if (unary->isIncrementDecrementOp() &&
            exprWritesTensor(unary->getSubExpr(), tensorNames)) {
            return true;
        }
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (stmtWritesAnyTensorExcept(child, tensorNames, ignoredStmts)) {
            return true;
        }
    }
    return false;
}

bool stencilLoopWritesReaderOutsideOrPath(const clang::Stmt* loop,
                                          DacppFile* dacppFile,
                                          const ShellPartitionPlan& plan) {
    std::set<std::string> readerTensors;
    for (const auto& param : plan.params) {
        if (isStencilWindowReaderForLayout(param, plan.signature.layout) &&
            !param.actualTensorName.empty()) {
            readerTensors.insert(param.actualTensorName);
        }
    }
    if (readerTensors.empty()) {
        return false;
    }
    std::set<const clang::Stmt*> ignoredStmts;
    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (sitePlan.supported && !sitePlan.hasRootBridge) {
        for (const auto& region : collectDistributedFollowupRegions(
                 dacppFile, plan.exprNode.shell, plan.exprNode.calc,
                 plan.exprNode.dacExpr)) {
            ignoredStmts.insert(region.stmt);
        }
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            ignoredStmts.insert(stmt);
        }
    }
    return stmtWritesAnyTensorExcept(loop, readerTensors, ignoredStmts);
}

bool stencilLoopWritesDirectReaderOutsideOrPath(const clang::Stmt* loop,
                                                DacppFile* dacppFile,
                                                const ShellPartitionPlan& plan) {
    std::set<std::string> directReaderTensors;
    for (const auto& param : plan.params) {
        if (isSupportedStencilDirectReader(param) &&
            !param.actualTensorName.empty()) {
            directReaderTensors.insert(param.actualTensorName);
        }
    }
    if (directReaderTensors.empty()) {
        return false;
    }
    std::set<const clang::Stmt*> ignoredStmts;
    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (sitePlan.supported && !sitePlan.hasRootBridge) {
        for (const auto& region : collectDistributedFollowupRegions(
                 dacppFile, plan.exprNode.shell, plan.exprNode.calc,
                 plan.exprNode.dacExpr)) {
            ignoredStmts.insert(region.stmt);
        }
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            ignoredStmts.insert(stmt);
        }
    }
    return stmtWritesAnyTensorExcept(loop, directReaderTensors, ignoredStmts);
}

const ParamAccessPlan* stencilWindowReaderParam(
    const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (isStencilWindowReaderForLayout(param, plan.signature.layout)) {
            return &param;
        }
    }
    return nullptr;
}

Split* splitForShellParam(Shell* shell, int paramIndex) {
    if (!shell || paramIndex < 0 ||
        paramIndex >= shell->getNumShellParams()) {
        return nullptr;
    }
    ShellParam* shellParam = shell->getShellParam(paramIndex);
    if (!shellParam) {
        return nullptr;
    }
    for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
        Split* split = shellParam->getSplit(splitIdx);
        if (split && split->type == "RegularSplit") {
            return split;
        }
    }
    return nullptr;
}

std::vector<RegularSplit*> regularSplitsForShellParam(Shell* shell,
                                                      int paramIndex) {
    std::vector<RegularSplit*> splits;
    if (!shell || paramIndex < 0 ||
        paramIndex >= shell->getNumShellParams()) {
        return splits;
    }
    ShellParam* shellParam = shell->getShellParam(paramIndex);
    if (!shellParam) {
        return splits;
    }
    for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
        Split* split = shellParam->getSplit(splitIdx);
        if (split && split->type == "RegularSplit") {
            splits.push_back(static_cast<RegularSplit*>(split));
        }
    }
    return splits;
}

const ParamAccessPlan* stencilOutputDirectWriterParam(
    const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (isStencilOutputDirectWriter(param, plan.signature.layout)) {
            return &param;
        }
    }
    return nullptr;
}

const ParamAccessPlan* singleStencilDirectReaderParam(
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* directReader = nullptr;
    for (const auto& param : plan.params) {
        if (!isSupportedStencilDirectReader(param)) {
            continue;
        }
        if (directReader) {
            return nullptr;
        }
        directReader = &param;
    }
    return directReader;
}

const ParamAccessPlan* paramByIndex(const ShellPartitionPlan& plan,
                                    int paramIndex) {
    for (const auto& param : plan.params) {
        if (param.paramIndex == paramIndex) {
            return &param;
        }
    }
    return nullptr;
}

bool paramsAlias(const ParamAccessPlan& lhs, const ParamAccessPlan& rhs) {
    if (!lhs.actualTensorAliasKey.empty() &&
        lhs.actualTensorAliasKey == rhs.actualTensorAliasKey) {
        return true;
    }
    return !lhs.actualTensorName.empty() &&
           lhs.actualTensorName == rhs.actualTensorName;
}

bool paramsProvenDistinct(const ParamAccessPlan& lhs,
                          const ParamAccessPlan& rhs) {
    return lhs.actualTensorAliasKeyPrecise &&
           rhs.actualTensorAliasKeyPrecise &&
           !lhs.actualTensorAliasKey.empty() &&
           !rhs.actualTensorAliasKey.empty() &&
           lhs.actualTensorAliasKey != rhs.actualTensorAliasKey;
}

bool stencilReaderAliasesOutputWriter(const ShellPartitionPlan& plan,
                                      const ParamAccessPlan& reader) {
    const ParamAccessPlan* writer = stencilOutputDirectWriterParam(plan);
    return writer && paramsAlias(reader, *writer);
}

bool stencilReaderWriterProvenDistinct(const ShellPartitionPlan& plan,
                                       const ParamAccessPlan& reader) {
    const ParamAccessPlan* writer = stencilOutputDirectWriterParam(plan);
    return writer && paramsProvenDistinct(reader, *writer);
}

bool siteUpdatesWindowReader(const DistributedStencilSitePlan& sitePlan,
                             const ParamAccessPlan& reader) {
    if (!sitePlan.supported) {
        return false;
    }
    for (const auto& mapping : sitePlan.followupMappings) {
        if (mapping.readerParamIndex == reader.paramIndex ||
            mapping.readerTensor == reader.actualTensorName ||
            mapping.readerTensor == reader.shellParamName) {
            return true;
        }
    }
    for (const auto& transition : sitePlan.readCacheTransitions) {
        if (transition.readerParamIndex == reader.paramIndex) {
            return true;
        }
    }
    for (const auto& update : sitePlan.boundaryLocalUpdates) {
        if (update.paramIndex == reader.paramIndex) {
            return true;
        }
    }
    return false;
}

bool canHoistStencilReaderSync(DacppFile* dacppFile,
                               const clang::Stmt* loop,
                               const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = stencilWindowReaderParam(plan);
    if (!reader || !loop) {
        return false;
    }
    if (!stencilReaderWriterProvenDistinct(plan, *reader) ||
        stencilReaderAliasesOutputWriter(plan, *reader)) {
        return false;
    }
    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (sitePlan.hasRootBridge || siteUpdatesWindowReader(sitePlan, *reader)) {
        return false;
    }
    return !stencilLoopWritesReaderOutsideOrPath(loop, dacppFile, plan);
}

int stmtOrderInLoopBody(const clang::Stmt* loop,
                       const clang::Stmt* stmt,
                       const clang::SourceManager& sourceManager) {
    if (!loop || !stmt) {
        return -1;
    }
    const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(loop);
    if (const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(loop)) {
        compound =
            llvm::dyn_cast_or_null<clang::CompoundStmt>(forStmt->getBody());
    } else if (const auto* whileStmt =
                   llvm::dyn_cast<clang::WhileStmt>(loop)) {
        compound =
            llvm::dyn_cast_or_null<clang::CompoundStmt>(whileStmt->getBody());
    }
    if (!compound) {
        return -1;
    }
    int index = 0;
    for (const clang::Stmt* child : compound->body()) {
        if (child &&
            sourceRangeContains(sourceManager, child->getSourceRange(),
                                stmt->getSourceRange())) {
            return index;
        }
        ++index;
    }
    return -1;
}

bool hasCurrentResidentHaloB3StmtOrder(const clang::Stmt* loop,
                                       DacppFile* dacppFile,
                                       const ShellPartitionPlan& plan,
                                       const DistributedStencilSitePlan& sitePlan) {
    if (!loop || !dacppFile || !dacppFile->getContext() || !plan.exprNode.dacExpr ||
        sitePlan.readCacheTransitions.size() != 1 ||
        sitePlan.followupMappings.size() != 1 ||
        sitePlan.boundaryLocalStmts.empty()) {
        return false;
    }
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    const int dacIndex =
        stmtOrderInLoopBody(loop, plan.exprNode.dacExpr, sourceManager);
    const int readCacheIndex =
        stmtOrderInLoopBody(loop, sitePlan.readCacheTransitions.front().stmt,
                            sourceManager);
    const int followupIndex =
        stmtOrderInLoopBody(loop, sitePlan.followupMappings.front().stmt,
                            sourceManager);
    if (dacIndex < 0 || readCacheIndex < 0 || followupIndex < 0) {
        return false;
    }
    if (!(dacIndex < readCacheIndex && readCacheIndex < followupIndex)) {
        return false;
    }
    for (const clang::Stmt* boundaryStmt : sitePlan.boundaryLocalStmts) {
        const int boundaryIndex =
            stmtOrderInLoopBody(loop, boundaryStmt, sourceManager);
        if (boundaryIndex < 0 || boundaryIndex <= followupIndex) {
            return false;
        }
    }
    return true;
}

void addContractResidentTensor(LoopLoweringContract& contract,
                               const ParamAccessPlan* param,
                               const std::string& role) {
    if (!param || param->actualTensorName.empty()) {
        return;
    }
    for (const auto& tensor : contract.residentTensors) {
        if (tensor.tensorName == param->actualTensorName &&
            tensor.role == role) {
            return;
        }
    }
    contract.residentTensors.push_back({param->actualTensorName, role});
}

void addContractMaterialization(LoopLoweringContract& contract,
                                const ParamAccessPlan* param,
                                LoweringContractMaterializeTiming timing,
                                const std::string& reason) {
    if (!param || param->actualTensorName.empty()) {
        return;
    }
    for (const auto& materialization : contract.materializations) {
        if (materialization.tensorName == param->actualTensorName &&
            materialization.timing == timing) {
            return;
        }
    }
    contract.materializations.push_back(
        {param->actualTensorName, timing, reason});
}

std::string stencilContractRemovalRole(
    const clang::Stmt* stmt,
    const DistributedStencilSitePlan& sitePlan,
    const clang::SourceManager* sourceManager) {
    bool removesReadCache = false;
    bool removesFollowup = false;
    if (stmt && sourceManager) {
        for (const auto& transition : sitePlan.readCacheTransitions) {
            if (stmtSourceRangeContains(*sourceManager, stmt,
                                        transition.stmt)) {
                removesReadCache = true;
                break;
            }
        }
        for (const auto& mapping : sitePlan.followupMappings) {
            if (stmtSourceRangeContains(*sourceManager, stmt, mapping.stmt)) {
                removesFollowup = true;
                break;
            }
        }
    }
    if (removesReadCache && removesFollowup) {
        return "read-cache+followup";
    }
    if (removesReadCache) {
        return "read-cache";
    }
    if (removesFollowup) {
        return "followup";
    }
    return "followup/read-cache";
}

void addContractRemoveStmt(
    LoopLoweringContract& contract,
    std::set<const clang::Stmt*>& seenRemovedStmts,
    const clang::Stmt* stmt,
    const std::string& role,
    const std::string& reason) {
    if (!stmt || seenRemovedStmts.count(stmt) != 0) {
        return;
    }
    seenRemovedStmts.insert(stmt);
    contract.statements.push_back(
        {stmt, LoweringContractStmtAction::Remove, role, reason});
}

std::set<const clang::Stmt*> legacyStencilLoopRemovalSet(
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan) {
    std::set<const clang::Stmt*> result;
    if (!dacppFile || !plan.exprNode.shell || !plan.exprNode.calc ||
        !plan.exprNode.dacExpr) {
        return result;
    }
    for (const auto& region : collectDistributedFollowupRegions(
             dacppFile, plan.exprNode.shell, plan.exprNode.calc,
             plan.exprNode.dacExpr)) {
        if (region.stmt) {
            result.insert(region.stmt);
        }
    }

    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (sitePlan.supported && !sitePlan.hasRootBridge) {
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            if (stmt) {
                result.insert(stmt);
            }
        }
    }
    return result;
}

std::string stencilRemovalSetMismatchReason(
    const std::set<const clang::Stmt*>& legacySet,
    const std::set<const clang::Stmt*>& contractSet) {
    if (legacySet == contractSet) {
        return "";
    }
    int missingFromContract = 0;
    int extraInContract = 0;
    for (const clang::Stmt* stmt : legacySet) {
        if (contractSet.count(stmt) == 0) {
            ++missingFromContract;
        }
    }
    for (const clang::Stmt* stmt : contractSet) {
        if (legacySet.count(stmt) == 0) {
            ++extraInContract;
        }
    }
    return "contract removal set mismatch legacy=" +
           std::to_string(legacySet.size()) + " contract=" +
           std::to_string(contractSet.size()) + " missing=" +
           std::to_string(missingFromContract) + " extra=" +
           std::to_string(extraInContract);
}

void annotateStencilContractRemovalSetEquivalence(
    DacppFile* dacppFile,
    ShellPartitionPlan& plan) {
    if (!plan.orLoopLower.contract.enabled ||
        (plan.orLoopLower.kind != OrLoopLowerKind::StencilFullSync &&
         plan.orLoopLower.kind != OrLoopLowerKind::StencilResidentHalo)) {
        return;
    }

    const std::set<const clang::Stmt*> legacySet =
        legacyStencilLoopRemovalSet(dacppFile, plan);
    const std::set<const clang::Stmt*> contractSet =
        loweringContractRemoveStmtSet(plan.orLoopLower.contract);
    const std::string mismatchReason =
        stencilRemovalSetMismatchReason(legacySet, contractSet);
    plan.orLoopLower.contractRemovalSetMatchesLegacy =
        mismatchReason.empty();
    plan.orLoopLower.contractRemovalSetReason =
        mismatchReason.empty() ? "match" : mismatchReason;
}

void populateStencilLoopLoweringContract(
    DacppFile* dacppFile,
    ShellPartitionPlan& plan,
    const std::string& residentHaloRejectReason) {
    if (!plan.exprNode.dacExpr ||
        (plan.orLoopLower.kind != OrLoopLowerKind::StencilFullSync &&
         plan.orLoopLower.kind != OrLoopLowerKind::StencilResidentHalo)) {
        return;
    }

    const bool residentHalo =
        plan.orLoopLower.kind == OrLoopLowerKind::StencilResidentHalo;
    const ParamAccessPlan* reader = stencilWindowReaderParam(plan);
    const ParamAccessPlan* writer = stencilOutputDirectWriterParam(plan);
    const ParamAccessPlan* directReader = singleStencilDirectReaderParam(plan);
    const DistributedStencilSitePlan sitePlan =
        analyzeDistributedStencilSite(dacppFile, plan.exprNode.shell,
                                      plan.exprNode.calc,
                                      plan.exprNode.dacExpr);

    LoopLoweringContract contract;
    contract.enabled = true;
    contract.loweringName =
        residentHalo ? "StencilResidentHalo" : "StencilFullSync";
    contract.acceptedReason =
        residentHalo ? "stencil resident-halo accepted current P4.6 proof"
                     : "stencil full-sync accepted current P4.6 proof";
    if (!residentHalo && !residentHaloRejectReason.empty()) {
        contract.rejectedReason = residentHaloRejectReason;
    }

    contract.statements.push_back(
        {plan.exprNode.dacExpr, LoweringContractStmtAction::Replace,
         "source-dac", "replace source DAC with loop-lowered OR run call"});

    const clang::SourceManager* sourceManager =
        dacppFile && dacppFile->getContext()
            ? &dacppFile->getContext()->getSourceManager()
            : nullptr;
    std::set<const clang::Stmt*> seenRemovedStmts;
    if (sitePlan.supported && !sitePlan.hasRootBridge) {
        for (const clang::Stmt* stmt : sitePlan.distributedFollowupStmts) {
            const std::string role =
                stencilContractRemovalRole(stmt, sitePlan, sourceManager);
            addContractRemoveStmt(
                contract, seenRemovedStmts, stmt, role,
                "removed source stmt is absorbed by P4.6 " + role +
                    " materialization");
        }
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            addContractRemoveStmt(
                contract, seenRemovedStmts, stmt, "boundary-local",
                "removed source stmt is absorbed by P4.6 boundary-local materialization");
        }
    }

    if (residentHalo) {
        addContractResidentTensor(contract, reader,
                                  "resident halo reader state");
        addContractResidentTensor(contract, writer,
                                  "rank-owned halo writer buffer");
        addContractResidentTensor(contract, directReader,
                                  "resident halo direct-reader state");
        addContractMaterialization(
            contract, writer, LoweringContractMaterializeTiming::LoopExit,
            "materialize writer tensor after the outer loop");
        addContractMaterialization(
            contract, reader, LoweringContractMaterializeTiming::LoopExit,
            "materialize resident reader after followup/boundary updates");
        addContractMaterialization(
            contract, directReader,
            LoweringContractMaterializeTiming::LoopExit,
            "materialize resident direct reader after loop-exit rotation");
    } else {
        addContractResidentTensor(contract, reader,
                                  "full-sync reader broadcast state");
        addContractResidentTensor(contract, writer,
                                  "rank-owned writer slice");
        addContractResidentTensor(contract, directReader,
                                  "full-sync direct-reader broadcast state");
        addContractMaterialization(
            contract, writer, LoweringContractMaterializeTiming::EveryRun,
            "full-sync gathers writer tensor inside each run call");
        for (const auto& mapping : sitePlan.followupMappings) {
            addContractMaterialization(
                contract, paramByIndex(plan, mapping.readerParamIndex),
                LoweringContractMaterializeTiming::EveryRun,
                "full-sync applies followup materialization inside each run call");
        }
        for (const auto& transition : sitePlan.readCacheTransitions) {
            addContractMaterialization(
                contract, paramByIndex(plan, transition.readerParamIndex),
                LoweringContractMaterializeTiming::EveryRun,
                "full-sync applies read-cache materialization inside each run call");
        }
    }

    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "P4.6 loop lifetime guard requires shell arguments declared before the loop"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "P4.6 parameter gate rejects unsupported order, scalar-reader, and stencil shapes"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "P4.6 boundary gate accepts only current followup/read-cache/boundary-local forms"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "P4.6 write/alias gate rejects reader or direct-reader writes outside the current OR path"});
    if (residentHalo) {
        contract.guards.push_back(
            {LoweringContractGuardDisposition::RuntimeAbort,
             "resident-halo runtime MPI count narrowing and overflow guards"});
        if (plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
            contract.guards.push_back(
                {LoweringContractGuardDisposition::RuntimeAbort,
                 "temporal-blocked resident halo runtime block tail and widened halo guards"});
        }
        if (directReader) {
            contract.guards.push_back(
                {LoweringContractGuardDisposition::RuntimeAbort,
                 "resident-halo direct-reader runtime shape/count guard"});
        }
    } else if (plan.signature.layout == LocalLayoutKind::StencilWindow2D) {
        contract.guards.push_back(
            {LoweringContractGuardDisposition::RuntimeAbort,
             "StencilWindow2D full-sync MPI count narrowing and direct-reader shape guards"});
    } else if (hasReplicatedScalarParam(plan)) {
        contract.guards.push_back(
            {LoweringContractGuardDisposition::RuntimeAbort,
             "StencilWindow1D full-sync scalar reader size guard"});
    }

    plan.orLoopLower.contract = contract;
}

LoweringContractMaterializeTiming primaryMaterializeTiming(
    const LoopLoweringContract& contract) {
    for (const auto& materialization : contract.materializations) {
        if (materialization.timing != LoweringContractMaterializeTiming::None) {
            return materialization.timing;
        }
    }
    return LoweringContractMaterializeTiming::None;
}

std::string contractRemoveRoleSummary(
    const LoopLoweringContract& contract) {
    std::set<std::string> roles;
    for (const auto& stmtContract : contract.statements) {
        if (stmtContract.action == LoweringContractStmtAction::Remove &&
            !stmtContract.role.empty()) {
            roles.insert(stmtContract.role);
        }
    }
    if (roles.empty()) {
        return "none";
    }
    std::string result;
    for (const auto& role : roles) {
        if (!result.empty()) {
            result += ",";
        }
        result += role;
    }
    return result;
}

bool contractHasGuardDisposition(
    const LoopLoweringContract& contract,
    LoweringContractGuardDisposition disposition) {
    for (const auto& guard : contract.guards) {
        if (guard.disposition == disposition) {
            return true;
        }
    }
    return false;
}

int contractReplaceStmtCount(const LoopLoweringContract& contract,
                             const clang::Stmt* stmt) {
    if (!stmt) {
        return 0;
    }
    int count = 0;
    for (const auto& statement : contract.statements) {
        if (statement.stmt == stmt &&
            statement.action == LoweringContractStmtAction::Replace) {
            ++count;
        }
    }
    return count;
}

bool contractStatementsHaveRolesAndReasons(
    const LoopLoweringContract& contract) {
    for (const auto& statement : contract.statements) {
        if (!statement.stmt || statement.role.empty() ||
            statement.reason.empty()) {
            return false;
        }
    }
    return true;
}

bool contractHasMaterializeTiming(
    const LoopLoweringContract& contract,
    LoweringContractMaterializeTiming timing) {
    for (const auto& materialization : contract.materializations) {
        if (materialization.timing == timing) {
            return true;
        }
    }
    return false;
}

bool contractMaterializationsUseTiming(
    const LoopLoweringContract& contract,
    LoweringContractMaterializeTiming timing) {
    if (contract.materializations.empty()) {
        return false;
    }
    for (const auto& materialization : contract.materializations) {
        if (materialization.tensorName.empty() ||
            materialization.reason.empty() ||
            materialization.timing != timing) {
            return false;
        }
    }
    return true;
}

bool contractMaterializedTensorsAreResident(
    const LoopLoweringContract& contract) {
    std::set<std::string> residentTensorNames;
    for (const auto& tensor : contract.residentTensors) {
        if (tensor.tensorName.empty() || tensor.role.empty()) {
            return false;
        }
        residentTensorNames.insert(tensor.tensorName);
    }
    for (const auto& materialization : contract.materializations) {
        if (residentTensorNames.count(materialization.tensorName) == 0) {
            return false;
        }
    }
    return true;
}

int contractGuardDispositionCount(
    const LoopLoweringContract& contract,
    LoweringContractGuardDisposition disposition) {
    int count = 0;
    for (const auto& guard : contract.guards) {
        if (guard.disposition == disposition) {
            ++count;
        }
    }
    return count;
}

bool contractGuardsHaveReasons(const LoopLoweringContract& contract) {
    for (const auto& guard : contract.guards) {
        if (guard.reason.empty()) {
            return false;
        }
    }
    return true;
}

std::string checkLoopLoweringContractConsistency(
    const ShellPartitionPlan& plan) {
    const OrLoopLowerPlan& loopLower = plan.orLoopLower;
    const LoopLoweringContract& contract = loopLower.contract;
    if (!contract.enabled) {
        return "contract-disabled";
    }
    if (contract.loweringName != orLoopLowerKindName(loopLower.kind)) {
        return "kind-mismatch";
    }
    if (contractReplaceStmtCount(contract, plan.exprNode.dacExpr) != 1) {
        return "missing-replace-stmt";
    }
    if (!contractStatementsHaveRolesAndReasons(contract)) {
        return "stmt-metadata-missing";
    }
    if (!contractGuardsHaveReasons(contract)) {
        return "guard-reason-missing";
    }
    const int compileGuardCount = contractGuardDispositionCount(
        contract, LoweringContractGuardDisposition::CompileTimeFallback);
    if (compileGuardCount < 4) {
        return "missing-compile-guard";
    }

    const bool hasRuntimeGuard = contractHasGuardDisposition(
        contract, LoweringContractGuardDisposition::RuntimeAbort);
    switch (loopLower.kind) {
    case OrLoopLowerKind::StencilResidentHalo:
        if (!loopLower.contractRemovalSetMatchesLegacy) {
            return "remove-list-mismatch";
        }
        if (contract.residentTensors.empty()) {
            return "missing-resident-tensor";
        }
        if (!contractMaterializedTensorsAreResident(contract)) {
            return "materialized-tensor-not-resident";
        }
        if (!contractHasMaterializeTiming(
                contract, LoweringContractMaterializeTiming::LoopExit)) {
            return "materialize-timing-mismatch";
        }
        if (!contractMaterializationsUseTiming(
                contract, LoweringContractMaterializeTiming::LoopExit)) {
            return "materialize-timing-mismatch";
        }
        if (!hasRuntimeGuard) {
            return "missing-runtime-guard";
        }
        return "p4.6-contract-consistent";
    case OrLoopLowerKind::StencilFullSync:
        if (!loopLower.contractRemovalSetMatchesLegacy) {
            return "remove-list-mismatch";
        }
        if (contract.residentTensors.empty()) {
            return "missing-resident-tensor";
        }
        if (!contractMaterializedTensorsAreResident(contract)) {
            return "materialized-tensor-not-resident";
        }
        if (!contractHasMaterializeTiming(
                contract, LoweringContractMaterializeTiming::EveryRun)) {
            return "materialize-timing-mismatch";
        }
        if (!contractMaterializationsUseTiming(
                contract, LoweringContractMaterializeTiming::EveryRun)) {
            return "materialize-timing-mismatch";
        }
        if ((plan.signature.layout == LocalLayoutKind::StencilWindow2D ||
             hasReplicatedScalarParam(plan)) &&
            !hasRuntimeGuard) {
            return "missing-runtime-guard";
        }
        return "p4.6-contract-consistent";
    case OrLoopLowerKind::FixedBlockPhaseExchange: {
        std::set<const clang::Stmt*> metadataRemoveSet;
        for (const clang::Stmt* stmt :
             loopLower.fixedBlockPhaseExchange.followerStmtsToRemove) {
            if (stmt) {
                metadataRemoveSet.insert(stmt);
            }
        }
        if (loweringContractRemoveStmtSet(contract) != metadataRemoveSet) {
            return "remove-list-mismatch";
        }
        if (contract.residentTensors.empty()) {
            return "missing-resident-tensor";
        }
        if (!contractMaterializedTensorsAreResident(contract)) {
            return "materialized-tensor-not-resident";
        }
        if (!contractHasMaterializeTiming(
                contract, LoweringContractMaterializeTiming::LoopExit)) {
            return "materialize-timing-mismatch";
        }
        if (!contractMaterializationsUseTiming(
                contract, LoweringContractMaterializeTiming::LoopExit)) {
            return "materialize-timing-mismatch";
        }
        if (!hasRuntimeGuard) {
            return "missing-runtime-guard";
        }
        return "p5-contract-consistent";
    }
    default:
        return "unsupported-kind";
    }
}

void annotateLoopLoweringContractConsistency(ShellPartitionPlan& plan) {
    if (!plan.orLoopLower.contract.enabled) {
        return;
    }
    const std::string reason = checkLoopLoweringContractConsistency(plan);
    plan.orLoopLower.contractConsistencyCheckPassed =
        reason == "p4.6-contract-consistent" ||
        reason == "p5-contract-consistent";
    plan.orLoopLower.contractConsistencyCheckReason = reason;
}

void logLoopLoweringContractSummary(
    const LoopLoweringContract& contract) {
    if (!contract.enabled) {
        return;
    }
    llvm::outs() << " contract=" << contract.loweringName
                 << " contract-source=replace"
                 << " contract-remove=" << contractRemoveRoleSummary(contract)
                 << " contract-resident="
                 << contract.residentTensors.size()
                 << " contract-materialize="
                 << loweringContractMaterializeTimingName(
                        primaryMaterializeTiming(contract));
    if (contractHasGuardDisposition(
            contract,
            LoweringContractGuardDisposition::CompileTimeFallback)) {
        llvm::outs() << " guard-compile=fallback";
    }
    if (contractHasGuardDisposition(
            contract, LoweringContractGuardDisposition::RuntimeAbort)) {
        llvm::outs() << " guard-runtime=count-or-shape";
    }
    if (!contract.acceptedReason.empty()) {
        llvm::outs() << " accepted-reason=" << contract.acceptedReason;
    }
    if (!contract.rejectedReason.empty()) {
        llvm::outs() << " rejected-reason=" << contract.rejectedReason;
    }
}

void logContractConsistencyCheck(const OrLoopLowerPlan& plan) {
    if (!plan.contract.enabled ||
        plan.contractConsistencyCheckReason.empty()) {
        return;
    }
    llvm::outs() << " contract-check="
                 << (plan.contractConsistencyCheckPassed ? "pass" : "fail")
                 << " reason=" << plan.contractConsistencyCheckReason;
}

void logContractRemovalSetEquivalence(const OrLoopLowerPlan& plan) {
    if (!plan.contract.enabled ||
        (plan.kind != OrLoopLowerKind::StencilFullSync &&
         plan.kind != OrLoopLowerKind::StencilResidentHalo)) {
        return;
    }
    llvm::outs() << " contract-removal-set="
                 << (plan.contractRemovalSetMatchesLegacy ? "match"
                                                          : "mismatch");
    if (!plan.contractRemovalSetMatchesLegacy &&
        !plan.contractRemovalSetReason.empty()) {
        llvm::outs() << " reason=" << plan.contractRemovalSetReason;
    }
}

bool exprWritesTensor(const clang::Expr* expr,
                      const std::set<std::string>& tensorNames) {
    const std::string name = baseNameFromExpr(expr);
    return !name.empty() && tensorNames.count(name) != 0;
}

bool stmtWritesAnyTensor(const clang::Stmt* stmt,
                         const std::set<std::string>& tensorNames) {
    if (!stmt || tensorNames.empty()) {
        return false;
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
        if (binary->isAssignmentOp() &&
            exprWritesTensor(binary->getLHS(), tensorNames)) {
            return true;
        }
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->isAssignmentOp() && opCall->getNumArgs() > 0 &&
            exprWritesTensor(opCall->getArg(0), tensorNames)) {
            return true;
        }
        if ((opCall->getOperator() == clang::OO_PlusPlus ||
             opCall->getOperator() == clang::OO_MinusMinus) &&
            opCall->getNumArgs() > 0 &&
            exprWritesTensor(opCall->getArg(0), tensorNames)) {
            return true;
        }
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(stmt)) {
        if (unary->isIncrementDecrementOp() &&
            exprWritesTensor(unary->getSubExpr(), tensorNames)) {
            return true;
        }
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (stmtWritesAnyTensor(child, tensorNames)) {
            return true;
        }
    }
    return false;
}

bool loopWritesAnyDirectReadInput(const clang::Stmt* loop,
                                  const ShellPartitionPlan& plan) {
    std::set<std::string> directReadInputs;
    for (const auto& name : directReadInputTensorNames(plan)) {
        if (!name.empty()) {
            directReadInputs.insert(name);
        }
    }
    return stmtWritesAnyTensor(loop, directReadInputs);
}

std::string loopLowerRejectReason(DacppFile* dacppFile,
                                  const ShellPartitionPlan& plan,
                                  const clang::Stmt** outerLoop) {
    if (outerLoop) {
        *outerLoop = stableOuterLoopForExpr(dacppFile, plan);
    }
    if (!directResidentLoopLayout(plan.signature.layout)) {
        return std::string("layout ") +
               localLayoutKindName(plan.signature.layout) +
               " is outside Phase 4.5 direct/resident scope";
    }
    const clang::Stmt* loop = outerLoop ? *outerLoop
                                        : stableOuterLoopForExpr(dacppFile, plan);
    if (!loop) {
        return "not inside a stable loop site";
    }
    if (!llvm::isa<clang::ForStmt>(loop) && !llvm::isa<clang::WhileStmt>(loop)) {
        return "stable site is not for/while";
    }
    if (loopContainsMultipleDacExprs(dacppFile, loop, plan)) {
        return "outer loop must contain exactly one DAC expression";
    }
    if (!hasOutputDirectParam(plan)) {
        return "missing direct output";
    }
    if (hasReadWriteOutputDirectParam(plan)) {
        return "read_write output direct unsupported for loop lowering";
    }
    if (directReadInputTensorNames(plan).empty()) {
        return "no loop-invariant direct read input to hoist";
    }
    if (directReadAliasesOutputDirectWrite(plan)) {
        return "direct read input aliases output direct write";
    }
    if (loopWritesAnyDirectReadInput(loop, plan)) {
        return "direct read input is modified inside loop";
    }
    return "";
}

std::string stencilFullSyncLoopRejectReason(DacppFile* dacppFile,
                                            const ShellPartitionPlan& plan,
                                            const clang::Stmt** outerLoop) {
    if (outerLoop) {
        *outerLoop = stableOuterLoopForExpr(dacppFile, plan);
    }
    if (!stencilFullSyncLoopLayout(plan.signature.layout)) {
        return std::string("layout ") +
               localLayoutKindName(plan.signature.layout) +
               " is outside Phase 4.6 stencil full-sync scope";
    }
    const clang::Stmt* loop = outerLoop ? *outerLoop
                                        : stableOuterLoopForExpr(dacppFile, plan);
    if (!loop) {
        return "not inside a stable loop site";
    }
    if (!llvm::isa<clang::ForStmt>(loop) && !llvm::isa<clang::WhileStmt>(loop)) {
        return "stable site is not for/while";
    }
    if (loopContainsMultipleDacExprs(dacppFile, loop, plan)) {
        return "outer loop must contain exactly one DAC expression";
    }
    if (!shellArgsDeclaredBeforeLoop(dacppFile, loop, plan)) {
        return "shell arguments must be declared before the loop";
    }

    int windowReaderCount = 0;
    int writerCount = 0;
    int directReaderCount = 0;
    for (const auto& param : plan.params) {
        if (isStencilWindowReaderForLayout(param, plan.signature.layout)) {
            ++windowReaderCount;
            continue;
        }
        if (isStencilOutputDirectWriter(param, plan.signature.layout)) {
            ++writerCount;
            continue;
        }
        if (isSupportedStencilDirectReader(param)) {
            ++directReaderCount;
            continue;
        }
        if (isSupportedStencilScalarReader(param)) {
            if (plan.signature.layout == LocalLayoutKind::StencilWindow2D) {
                return "StencilWindow2D loop lowering does not yet support replicated scalar readers";
            }
            continue;
        }
        return "unsupported stencil parameter shape for loop lowering";
    }
    const int maxDirectReaders =
        plan.signature.layout == LocalLayoutKind::StencilWindow2D ? 1 : 0;
    if (windowReaderCount != 1 || writerCount != 1 ||
        directReaderCount > maxDirectReaders) {
        return plan.signature.layout == LocalLayoutKind::StencilWindow2D
                   ? "requires one 2D window reader, one WRITE-only direct writer, and at most one direct reader"
                   : "requires one 1D window reader, one WRITE-only direct writer, and no direct readers";
    }

    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (sitePlan.supported) {
        if (sitePlan.hasRootBridge) {
            return "root-bridge stencil sites are outside Phase 4.6 A1";
        }
        if (plan.signature.layout == LocalLayoutKind::StencilWindow1D &&
            !sitePlan.readCacheTransitions.empty()) {
            return "read-cache transitions are outside Phase 4.6 A1 StencilWindow1D";
        }
    }
    if (stencilLoopWritesReaderOutsideOrPath(loop, dacppFile, plan)) {
        return "reader modified in loop outside current OR path";
    }
    return "";
}

std::string stencilResidentHaloRejectReason(
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan,
    const clang::Stmt* loop,
    OrLoopLowerPlan::StencilResidentHaloMetadata& metadata) {
    metadata = OrLoopLowerPlan::StencilResidentHaloMetadata{};
    if (!loop) {
        return "not inside a stable loop site";
    }
    const ParamAccessPlan* reader = stencilWindowReaderParam(plan);
    const ParamAccessPlan* writer = stencilOutputDirectWriterParam(plan);
    if (!reader || !writer) {
        return "requires one window reader and one direct writer";
    }
    const std::string readerType =
        plan.exprNode.calc->getParam(reader->paramIndex)->getBasicType();
    const std::string writerType =
        plan.exprNode.calc->getParam(writer->paramIndex)->getBasicType();
    if (readerType != writerType) {
        return "resident halo B1 requires reader/writer element types to match";
    }
    if (usesByteTransport(readerType) || usesByteTransport(writerType)) {
        return "resident halo B1 requires native MPI element datatypes";
    }
    if (!loopContainsOnlyDacAndLoweredStencilPostStmts(dacppFile, loop,
                                                       plan)) {
        return plan.signature.layout == LocalLayoutKind::StencilWindow2D
                   ? "resident halo B2 requires loop body to contain only DAC and lowered post statements"
                   : "resident halo B1 requires loop body to contain only DAC and lowered post statements";
    }

    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (!sitePlan.supported) {
        return plan.signature.layout == LocalLayoutKind::StencilWindow2D
                   ? "resident halo B2 requires distributed site analysis"
                   : "resident halo B1 requires distributed site analysis";
    }
    if (sitePlan.hasRootBridge) {
        return plan.signature.layout == LocalLayoutKind::StencilWindow2D
                   ? "root-bridge stencil sites are outside resident halo B2"
                   : "root-bridge stencil sites are outside resident halo B1";
    }
    if (plan.signature.layout == LocalLayoutKind::StencilWindow1D) {
        if (!sitePlan.readCacheTransitions.empty()) {
            return "read-cache transitions are outside resident halo B1";
        }
        Split* split = splitForShellParam(plan.exprNode.shell,
                                          reader->paramIndex);
        auto* regular =
            split && split->type == "RegularSplit"
                ? static_cast<RegularSplit*>(split)
                : nullptr;
        if (!regular) {
            return "resident halo B1 requires a regular split window";
        }
        metadata.windowSize = regular->getSplitSize();
        metadata.windowStride = regular->getSplitStride();
        if (metadata.windowSize < 2 || metadata.windowSize > 3 ||
            metadata.windowStride != 1) {
            return "resident halo B1 requires stride-1 window size 2 or 3";
        }
        if (sitePlan.followupMappings.size() != 1) {
            return "resident halo B1 requires exactly one followup mapping";
        }
        const auto& mapping = sitePlan.followupMappings.front();
        if (mapping.rank != 1 ||
            mapping.writerParamIndex != writer->paramIndex ||
            mapping.readerParamIndex != reader->paramIndex ||
            mapping.targetOffset != 1) {
            return "resident halo B1 requires writer->reader followup offset +1";
        }
        metadata.followupTargetOffset = mapping.targetOffset;
        metadata.leftHalo = 0;
        metadata.rightHalo = metadata.windowSize - 1;
        const int64_t readerExtent =
            staticOneDimShellArgExtent(dacppFile, plan, *reader);
        const int64_t writerExtent =
            staticOneDimShellArgExtent(dacppFile, plan, *writer);
        const int64_t requiredReaderExtent =
            writerExtent > 0
                ? writerExtent + static_cast<int64_t>(metadata.windowSize) - 1
                : -1;
        if (readerExtent <= 0 || writerExtent <= 0 ||
            readerExtent != requiredReaderExtent) {
            return "resident halo B1 requires reader extent to cover writer slice plus halo";
        }

        if (sitePlan.boundaryLocalUpdates.size() > 1) {
            return "resident halo B1 supports at most one boundary-local update";
        }
        if (!sitePlan.boundaryLocalUpdates.empty()) {
            const auto& update = sitePlan.boundaryLocalUpdates.front();
            if (update.rank != 1 ||
                update.paramIndex != reader->paramIndex ||
                update.targetRowUsesLoop || update.sourceRowUsesLoop) {
                return "resident halo B1 only supports constant-index 1D boundary updates";
            }
            const auto isExactZeroIndex = [](int index,
                                             const std::string& exprText) {
                return index == 0 && trim(exprText) == "0";
            };
            if (!isExactZeroIndex(update.targetRow, update.targetRowExpr)) {
                return "resident halo B1 boundary update target must be left boundary index 0";
            }
            if (!update.constantRhs &&
                update.sourceParamIndex == writer->paramIndex &&
                !isExactZeroIndex(update.sourceRow, update.sourceRowExpr)) {
                return "resident halo B1 boundary writer copy must read source index 0";
            }
            metadata.hasBoundaryLocalUpdate = true;
            metadata.boundaryTargetIndex = update.targetRow;
            metadata.boundarySourceIndex = update.sourceRow;
            metadata.boundaryConstantValue = update.constantValue;
            metadata.boundaryCopiesWriter =
                !update.constantRhs &&
                update.sourceParamIndex == writer->paramIndex;
            if (!metadata.boundaryCopiesWriter && !update.constantRhs) {
                return "resident halo B1 boundary update must copy writer or constant";
            }
        }

        metadata.enabled = true;
        return "";
    }

    if (plan.signature.layout != LocalLayoutKind::StencilWindow2D) {
        return "resident halo is limited to StencilWindow1D/B2 row-block StencilWindow2D";
    }
    const ParamAccessPlan* directReader = singleStencilDirectReaderParam(plan);
    for (const auto& param : plan.params) {
        if (isSupportedStencilScalarReader(param)) {
            return "resident halo B2 excludes scalar readers";
        }
    }
    const auto regularSplits =
        regularSplitsForShellParam(plan.exprNode.shell, reader->paramIndex);
    if (regularSplits.size() != 2) {
        return "resident halo B2 requires exactly two regular split dimensions";
    }
    metadata.windowRows = regularSplits[0]->getSplitSize();
    metadata.windowCols = regularSplits[1]->getSplitSize();
    metadata.windowRowStride = regularSplits[0]->getSplitStride();
    metadata.windowColStride = regularSplits[1]->getSplitStride();
    if (metadata.windowRows != 3 || metadata.windowCols != 3 ||
        metadata.windowRowStride != 1 || metadata.windowColStride != 1) {
        return "resident halo B2 currently requires a 3x3 stride-1 window";
    }
    const int64_t readerRows = operator_resident::shapeValueFor(
        plan.exprNode.shell, reader->paramIndex, 0);
    const int64_t readerCols = operator_resident::shapeValueFor(
        plan.exprNode.shell, reader->paramIndex, 1);
    const int64_t writerRows = operator_resident::shapeValueFor(
        plan.exprNode.shell, writer->paramIndex, 0);
    const int64_t writerCols = operator_resident::shapeValueFor(
        plan.exprNode.shell, writer->paramIndex, 1);
    if ((readerRows > 0 && writerRows > 0 &&
         readerRows != writerRows + 2) ||
        (readerCols > 0 && writerCols > 0 &&
         readerCols != writerCols + 2)) {
        return "resident halo B2 requires reader/writer shapes with a 1-cell border";
    }
    if (stencilLoopWritesReaderOutsideOrPath(loop, dacppFile, plan)) {
        return "resident halo B2/B3 reader modified outside current OR path";
    }
    if (directReader &&
        stencilLoopWritesDirectReaderOutsideOrPath(loop, dacppFile, plan)) {
        return "resident halo B3 direct reader modified outside current OR path";
    }
    if (sitePlan.followupMappings.size() != 1) {
        return "resident halo B2 requires exactly one followup mapping";
    }
    const auto& mapping = sitePlan.followupMappings.front();
    if (mapping.rank != 2 ||
        mapping.writerParamIndex != writer->paramIndex ||
        mapping.readerParamIndex != reader->paramIndex ||
        mapping.targetRowOffset != 1 ||
        mapping.targetColOffset != 1) {
        return "resident halo B2 requires writer->reader followup offset (+1,+1)";
    }
    metadata.followupTargetRowOffset = mapping.targetRowOffset;
    metadata.followupTargetColOffset = mapping.targetColOffset;

    if (sitePlan.boundaryLocalUpdates.size() != 4) {
        return "resident halo B2 currently requires the canonical 4-loop boundary-local updates";
    }
    bool sawTop = false;
    bool sawBottom = false;
    bool sawLeft = false;
    bool sawRight = false;
    auto isZeroExpr = [](const std::string& exprText) {
        return compactExprText(exprText) == "0";
    };
    auto isOneExpr = [](const std::string& exprText) {
        return compactExprText(exprText) == "1";
    };
    auto isLastIndexExpr = [](const std::string& exprText, int64_t size) {
        const std::string compact = compactExprText(exprText);
        return (size > 0 && compact == std::to_string(size - 1)) ||
               endsWith(compact, "-1");
    };
    auto isPenultimateIndexExpr = [](const std::string& exprText,
                                     int64_t size) {
        const std::string compact = compactExprText(exprText);
        return (size > 1 && compact == std::to_string(size - 2)) ||
               endsWith(compact, "-2");
    };
    auto isZeroConstantExpr = [](const std::string& exprText) {
        const std::string compact = compactExprText(exprText);
        return compact == "0" || compact == "0.0" || compact == "0.0f" ||
               compact == "0.0F";
    };
    auto loopCoversBoundarySpan = [](const BoundaryLocalUpdate& update,
                                     int64_t extent) {
        const std::string loopLower = compactExprText(update.loopLowerExpr);
        const std::string loopUpper = compactExprText(update.loopUpperExpr);
        return loopLower == "0" &&
               ((extent > 0 && loopUpper == std::to_string(extent - 1)) ||
                endsWith(loopUpper, "-1"));
    };
    if (directReader) {
        const std::string directReaderType =
            plan.exprNode.calc->getParam(directReader->paramIndex)
                ->getBasicType();
        if (directReaderType != readerType || directReaderType != writerType) {
            return "resident halo B3 requires direct reader element type to match reader/writer";
        }
        if (usesByteTransport(directReaderType)) {
            return "resident halo B3 requires native MPI element datatypes";
        }
        if (!paramsProvenDistinct(*reader, *directReader) ||
            !paramsProvenDistinct(*writer, *directReader) ||
            paramsAlias(*reader, *directReader) ||
            paramsAlias(*writer, *directReader)) {
            return "resident halo B3 requires distinct precise tensor keys for window/direct-reader/writer";
        }
        const int64_t directReaderRows = operator_resident::shapeValueFor(
            plan.exprNode.shell, directReader->paramIndex, 0);
        const int64_t directReaderCols = operator_resident::shapeValueFor(
            plan.exprNode.shell, directReader->paramIndex, 1);
        if ((directReaderRows > 0 && writerRows > 0 &&
             directReaderRows != writerRows) ||
            (directReaderCols > 0 && writerCols > 0 &&
             directReaderCols != writerCols)) {
            return "resident halo B3 requires direct reader shape to match writer";
        }
        if (sitePlan.readCacheTransitions.size() != 1) {
            return "resident halo B3 requires exactly one read-cache transition";
        }
        const auto& transition = sitePlan.readCacheTransitions.front();
        if (transition.rank != 2 ||
            transition.writerParamIndex != reader->paramIndex ||
            transition.readerParamIndex != directReader->paramIndex ||
            transition.targetRowOffset != -1 ||
            transition.targetColOffset != -1) {
            return "resident halo B3 requires window->direct-reader read-cache offset (-1,-1)";
        }
        if (!hasCurrentResidentHaloB3StmtOrder(loop, dacppFile, plan,
                                               sitePlan)) {
            return "resident halo B3 requires the current DAC -> read-cache -> followup -> boundary statement order";
        }

        for (const auto& update : sitePlan.boundaryLocalUpdates) {
            if (update.rank != 2 ||
                update.paramIndex != reader->paramIndex ||
                !update.constantRhs ||
                !isZeroConstantExpr(update.constantValue)) {
                return "resident halo B3 boundary updates must be zero-valued reader-local writes";
            }
            const std::string loopLower = compactExprText(update.loopLowerExpr);
            const std::string loopUpper = compactExprText(update.loopUpperExpr);
            if (loopLower != "0" || !update.loopUpperInclusive) {
                return "resident halo B3 boundary updates require inclusive full-span loops";
            }
            if (!update.targetRowUsesLoop && update.targetColUsesLoop &&
                isLastIndexExpr(loopUpper, readerCols) &&
                isZeroExpr(update.targetRowExpr)) {
                sawTop = true;
                continue;
            }
            if (!update.targetRowUsesLoop && update.targetColUsesLoop &&
                isLastIndexExpr(loopUpper, readerCols) &&
                isLastIndexExpr(update.targetRowExpr, readerRows)) {
                sawBottom = true;
                continue;
            }
            if (update.targetRowUsesLoop && !update.targetColUsesLoop &&
                isLastIndexExpr(loopUpper, readerRows) &&
                isZeroExpr(update.targetColExpr)) {
                sawLeft = true;
                continue;
            }
            if (update.targetRowUsesLoop && !update.targetColUsesLoop &&
                isLastIndexExpr(loopUpper, readerRows) &&
                isLastIndexExpr(update.targetColExpr, readerCols)) {
                sawRight = true;
                continue;
            }
            return "resident halo B3 only supports the current zero-valued top/bottom/left/right boundary updates";
        }
        if (!sawTop || !sawBottom || !sawLeft || !sawRight) {
            return "resident halo B3 requires canonical zero-valued top/bottom/left/right boundary updates";
        }

        metadata.enabled = true;
        metadata.hasDirectReader = true;
        metadata.readCacheTargetRowOffset = transition.targetRowOffset;
        metadata.readCacheTargetColOffset = transition.targetColOffset;
        return "";
    }
    if (!sitePlan.readCacheTransitions.empty()) {
        return "read-cache transitions are outside resident halo B2";
    }
    for (const auto& update : sitePlan.boundaryLocalUpdates) {
        if (update.rank != 2 ||
            update.paramIndex != reader->paramIndex ||
            update.sourceParamIndex != reader->paramIndex ||
            update.constantRhs) {
            return "resident halo B2 boundary updates must be reader-local self copies";
        }
        const std::string loopLower = compactExprText(update.loopLowerExpr);
        const std::string loopUpper = compactExprText(update.loopUpperExpr);
        const bool sameRowLoopExpr =
            compactExprText(update.targetRowExpr) ==
            compactExprText(update.sourceRowExpr);
        const bool sameColLoopExpr =
            compactExprText(update.targetColExpr) ==
            compactExprText(update.sourceColExpr);

        if (!update.targetRowUsesLoop && !update.sourceRowUsesLoop &&
            update.targetColUsesLoop && update.sourceColUsesLoop &&
            sameColLoopExpr && loopCoversBoundarySpan(update, readerCols) &&
            isZeroExpr(update.targetRowExpr) && isOneExpr(update.sourceRowExpr)) {
            sawTop = true;
            continue;
        }
        if (!update.targetRowUsesLoop && !update.sourceRowUsesLoop &&
            update.targetColUsesLoop && update.sourceColUsesLoop &&
            sameColLoopExpr && loopCoversBoundarySpan(update, readerCols) &&
            isLastIndexExpr(update.targetRowExpr, readerRows) &&
            isPenultimateIndexExpr(update.sourceRowExpr, readerRows)) {
            sawBottom = true;
            continue;
        }
        if (update.targetRowUsesLoop && update.sourceRowUsesLoop &&
            !update.targetColUsesLoop && !update.sourceColUsesLoop &&
            sameRowLoopExpr && loopCoversBoundarySpan(update, readerRows) &&
            isZeroExpr(update.targetColExpr) && isOneExpr(update.sourceColExpr)) {
            sawLeft = true;
            continue;
        }
        if (update.targetRowUsesLoop && update.sourceRowUsesLoop &&
            !update.targetColUsesLoop && !update.sourceColUsesLoop &&
            sameRowLoopExpr && loopCoversBoundarySpan(update, readerRows) &&
            isLastIndexExpr(update.targetColExpr, readerCols) &&
            isPenultimateIndexExpr(update.sourceColExpr, readerCols)) {
            sawRight = true;
            continue;
        }
        return "resident halo B2 only supports the current top/bottom/left/right boundary-local updates";
    }
    if (!sawTop || !sawBottom || !sawLeft || !sawRight) {
        return "resident halo B2 requires canonical top/bottom/left/right boundary-local updates";
    }

    metadata.enabled = true;
    return "";
}

std::string loopIterationLimitExpr(const clang::Stmt* loop,
                                   clang::ASTContext* context,
                                   bool& inclusive) {
    inclusive = false;
    const clang::ForStmt* forStmt =
        llvm::dyn_cast_or_null<clang::ForStmt>(loop);
    if (!forStmt || !context) {
        return "";
    }
    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
    if (!cond || (cond->getOpcode() != clang::BO_LT &&
                  cond->getOpcode() != clang::BO_LE)) {
        return "";
    }
    const auto* declStmt =
        llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
    if (!declStmt || !declStmt->isSingleDecl()) {
        return "";
    }
    const auto* loopVarDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    if (!loopVarDecl) {
        return "";
    }
    const auto* lhsRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
        cond->getLHS()->IgnoreParenImpCasts());
    if (!lhsRef || lhsRef->getDecl() != loopVarDecl) {
        return "";
    }
    int64_t lower = -1;
    if (!evaluateInt64Expr(loopVarDecl->getInit(), context, lower) ||
        lower != 0) {
        return "";
    }
    inclusive = cond->getOpcode() == clang::BO_LE;
    return exprSource(cond->getRHS(), context);
}

void annotateResidentHaloTemporalBlocking(
    DacppFile* dacppFile,
    ShellPartitionPlan& plan,
    const clang::Stmt* loop,
    const OrLoopLowerPlan::StencilResidentHaloMetadata& metadata) {
    plan.orLoopLower.stencilResidentHalo.temporalBlockSize = 0;
    plan.orLoopLower.stencilResidentHalo.temporalLoopLimitExpr.clear();
    plan.orLoopLower.stencilResidentHalo.temporalLoopLimitInclusive = false;
    plan.orLoopLower.stencilResidentHalo.temporalBlockRejectReason.clear();
    auto reject = [&](const std::string& reason) {
        plan.orLoopLower.stencilResidentHalo.temporalBlockRejectReason = reason;
        plan.orLoopLower.stencilResidentHalo.temporalBlockAcceptReason.clear();
    };
    if (!dacppFile || !dacppFile->getContext() || !loop) {
        reject("analysis context unavailable");
        return;
    }
    if (!metadata.enabled) {
        reject(metadata.rejectReason.empty() ? "resident halo not accepted"
                                             : metadata.rejectReason);
        return;
    }
    if (plan.signature.layout == LocalLayoutKind::StencilWindow1D) {
        if (metadata.hasDirectReader) {
            reject("direct readers are outside StencilWindow1D resident halo");
            return;
        }
        if ((metadata.windowSize != 2 && metadata.windowSize != 3) ||
            metadata.followupTargetOffset != 1) {
            reject("requires canonical 1D window size 2/3 and followup offset 1");
            return;
        }
        for (const auto& param : plan.params) {
            if (isSupportedStencilScalarReader(param)) {
                reject("scalar readers are not enabled for 1D k=2 replay");
                return;
            }
        }
        const ParamAccessPlan* reader = stencilWindowReaderParam(plan);
        const ParamAccessPlan* writer = stencilOutputDirectWriterParam(plan);
        if (!reader || !writer) {
            reject("requires one window reader and one direct writer");
            return;
        }
        const int64_t readerExtent =
            staticOneDimShellArgExtent(dacppFile, plan, *reader);
        const int64_t writerExtent =
            staticOneDimShellArgExtent(dacppFile, plan, *writer);
        if (readerExtent <= 0 || writerExtent <= 0 ||
            readerExtent != writerExtent + metadata.windowSize - 1 ||
            writerExtent < 2) {
            reject("requires proven 1D reader extent writer+halo with room for k=2");
            return;
        }
        bool inclusive = false;
        const std::string limitExpr =
            loopIterationLimitExpr(loop, dacppFile->getContext(), inclusive);
        if (limitExpr.empty()) {
            reject("requires canonical zero-based for-loop bound");
            return;
        }
        const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
            dacppFile, plan.exprNode.shell, plan.exprNode.calc,
            plan.exprNode.dacExpr);
        if (!sitePlan.supported || !sitePlan.readCacheTransitions.empty()) {
            reject("requires canonical B1 followup with no read-cache");
            return;
        }
        const bool hasBoundaryReplay = metadata.hasBoundaryLocalUpdate;
        if (!hasBoundaryReplay && !sitePlan.boundaryLocalUpdates.empty()) {
            reject("requires canonical replayable 1D boundary-local update");
            return;
        }
        if (hasBoundaryReplay &&
            (sitePlan.boundaryLocalUpdates.size() != 1 ||
             !metadata.boundaryCopiesWriter ||
             metadata.boundaryTargetIndex != 0 ||
             metadata.boundarySourceIndex != 0)) {
            reject("requires left boundary-local writer copy for 1D k=2 replay");
            return;
        }
        plan.orLoopLower.stencilResidentHalo.temporalBlockSize = 2;
        plan.orLoopLower.stencilResidentHalo.temporalLoopLimitExpr = limitExpr;
        plan.orLoopLower.stencilResidentHalo.temporalLoopLimitInclusive = inclusive;
        plan.orLoopLower.stencilResidentHalo.temporalBlockAcceptReason =
            hasBoundaryReplay ? "boundary-local replay" : "canonical";
        return;
    }
    if (plan.signature.layout != LocalLayoutKind::StencilWindow2D) {
        reject("requires StencilWindow1D or StencilWindow2D resident halo");
        return;
    }
    if (metadata.windowRows != 3 || metadata.windowCols != 3 ||
        metadata.followupTargetRowOffset != 1 ||
        metadata.followupTargetColOffset != 1) {
        reject("requires canonical 3x3 window and followup offset (1,1)");
        return;
    }
    bool inclusive = false;
    const std::string limitExpr =
        loopIterationLimitExpr(loop, dacppFile->getContext(), inclusive);
    if (limitExpr.empty()) {
        reject("requires canonical zero-based for-loop bound");
        return;
    }
    const DistributedStencilSitePlan sitePlan = analyzeDistributedStencilSite(
        dacppFile, plan.exprNode.shell, plan.exprNode.calc,
        plan.exprNode.dacExpr);
    if (metadata.hasDirectReader) {
        const ParamAccessPlan* directReader =
            singleStencilDirectReaderParam(plan);
        if (!directReader) {
            reject("requires exactly one direct reader for direct-reader recurrence");
            return;
        }
        for (const auto& param : plan.params) {
            if (isSupportedStencilScalarReader(param)) {
                reject("scalar readers are not enabled for direct-reader k=2 replay");
                return;
            }
        }
        if (metadata.readCacheTargetRowOffset != -1 ||
            metadata.readCacheTargetColOffset != -1) {
            reject("requires direct-reader read-cache offset (-1,-1)");
            return;
        }
        if (!sitePlan.supported ||
            sitePlan.readCacheTransitions.size() != 1 ||
            sitePlan.followupMappings.size() != 1 ||
            sitePlan.boundaryLocalUpdates.size() != 4) {
            reject("requires canonical B3 read-cache, followup, and boundary-local updates");
            return;
        }
        const auto& transition = sitePlan.readCacheTransitions.front();
        const auto& mapping = sitePlan.followupMappings.front();
        const ParamAccessPlan* reader = stencilWindowReaderParam(plan);
        const ParamAccessPlan* writer = stencilOutputDirectWriterParam(plan);
        if (!reader || !writer ||
            transition.writerParamIndex != reader->paramIndex ||
            transition.readerParamIndex != directReader->paramIndex ||
            mapping.writerParamIndex != writer->paramIndex ||
            mapping.readerParamIndex != reader->paramIndex ||
            transition.targetRowOffset != -1 ||
            transition.targetColOffset != -1 ||
            mapping.targetRowOffset != 1 ||
            mapping.targetColOffset != 1) {
            reject("requires canonical B3 direct-reader recurrence edges");
            return;
        }
        if (!hasCurrentResidentHaloB3StmtOrder(loop, dacppFile, plan,
                                               sitePlan)) {
            reject("requires DAC -> read-cache -> followup -> boundary order for direct-reader recurrence");
            return;
        }
    } else if (!sitePlan.supported ||
               sitePlan.boundaryLocalUpdates.size() != 4 ||
               !sitePlan.readCacheTransitions.empty()) {
        reject("requires canonical B2 followup and boundary-local updates");
        return;
    }
    plan.orLoopLower.stencilResidentHalo.temporalBlockSize = 2;
    plan.orLoopLower.stencilResidentHalo.temporalLoopLimitExpr = limitExpr;
    plan.orLoopLower.stencilResidentHalo.temporalLoopLimitInclusive = inclusive;
}

void annotateResidentHaloSpatial2D(
    ShellPartitionPlan& plan,
    bool haloCandidate,
    const std::string& haloRejectReason) {
    auto& metadata = plan.orLoopLower.stencilResidentHalo;
    metadata.spatial2DEnabled = false;
    metadata.spatial2DHaloWidth = 0;
    metadata.spatial2DAcceptReason.clear();
    metadata.spatial2DRejectReason.clear();
    auto reject = [&](const std::string& reason) {
        metadata.spatial2DRejectReason = reason;
    };
    if (plan.signature.layout != LocalLayoutKind::StencilWindow2D) {
        return;
    }
    if (!haloCandidate || !metadata.enabled) {
        reject(haloRejectReason.empty() ? "resident halo not accepted"
                                        : haloRejectReason);
        return;
    }
    if (metadata.temporalBlockSize > 2) {
        reject("spatial temporal-block>2 unsupported; row-temporal retained");
        return;
    }
    if (metadata.windowRows != 3 || metadata.windowCols != 3 ||
        metadata.windowRowStride != 1 || metadata.windowColStride != 1) {
        reject("requires static 3x3 stride-1 stencil window");
        return;
    }
    if (metadata.followupTargetRowOffset != 1 ||
        metadata.followupTargetColOffset != 1) {
        reject("requires writer->reader followup offset (1,1)");
        return;
    }
    for (const auto& param : plan.params) {
        if (isSupportedStencilScalarReader(param)) {
            reject("scalar readers are outside conservative spatial-2d contract");
            return;
        }
        if (!metadata.hasDirectReader &&
            param.access == ParamAccessKind::DirectMapped &&
            param.reads && !param.writes) {
            reject("direct readers are outside conservative spatial-2d contract");
            return;
        }
        if (param.postUseSync.kind == PostUseSyncKind::FullTensor &&
            !param.broadcastMaterializedOutput &&
            param.access != ParamAccessKind::StencilWindow) {
            reject("unsupported post-use contract for spatial-2d");
            return;
        }
        if (!metadata.hasDirectReader &&
            metadata.temporalBlockSize > 1 &&
            param.postUseSync.kind != PostUseSyncKind::None) {
            reject("spatial temporal-block=2 with host post-use is not profitable in the current rectangular buffer path; row-temporal retained");
            return;
        }
    }
    metadata.spatial2DEnabled = true;
    metadata.spatial2DHaloWidth = metadata.temporalBlockSize > 1 ? 2 : 1;
    if (metadata.hasDirectReader) {
        metadata.spatial2DAcceptReason =
            metadata.temporalBlockSize > 1
                ? "canonical B3 3x3 direct-reader temporal-block=2 spatial role rotation"
                : "canonical B3 3x3 direct-reader spatial role rotation";
    } else {
        metadata.spatial2DAcceptReason =
            metadata.temporalBlockSize > 1
                ? "canonical B2 3x3 stencil temporal-block=2 rectangular-owned writer cells"
                : "canonical B2 3x3 stencil rectangular-owned writer cells";
    }
}

struct PhaseExchangeDetection {
    bool valid = false;
    std::size_t planAIndex = 0;
    std::size_t planBIndex = 0;
    const clang::Stmt* outerLoop = nullptr;
    const clang::BinaryOperator* followerDacExpr = nullptr;
    std::string sourceTensorName;
    std::string phaseAOutputTensorName;
    std::string elementType;
    std::vector<const clang::Stmt*> followerStmts;
    int phaseShiftOffset = 1;
    int blockSize = 2;
    int blockStride = 2;
    int64_t provenEvenTotal = 0;
    std::string rejectReason;
};

void populateFixedBlockPhaseExchangeContract(
    OrLoopLowerPlan& plan,
    const PhaseExchangeDetection& detection,
    const clang::Stmt* phaseADacExpr) {
    LoopLoweringContract contract;
    contract.enabled = true;
    contract.loweringName = "FixedBlockPhaseExchange";
    contract.acceptedReason =
        "phase-exchange accepted canonical fixed-block loop contract";

    contract.statements.push_back(
        {phaseADacExpr, LoweringContractStmtAction::Replace,
         "phase-a-dac", "replace phase-A DAC with resident run call"});
    for (const clang::Stmt* stmt : detection.followerStmts) {
        if (!stmt || stmt == phaseADacExpr) {
            continue;
        }
        contract.statements.push_back(
            {stmt, LoweringContractStmtAction::Remove, "phase-exchange-follower",
             "removed source stmt is absorbed by resident phase exchange"});
    }

    contract.residentTensors.push_back(
        {detection.sourceTensorName, "rank-contiguous resident state"});
    contract.materializations.push_back(
        {detection.sourceTensorName,
         LoweringContractMaterializeTiming::LoopExit,
         "materialize host-visible source tensor after outer loop"});

    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "phase-exchange removed statements must not be referenced after rewrite"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "phase-exchange phase-A output is used after the outer loop"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "phase-exchange unexpected write to resident source tensor"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::CompileTimeFallback,
         "phase-exchange requires a statically proven even total"});
    contract.guards.push_back(
        {LoweringContractGuardDisposition::RuntimeAbort,
         "phase-exchange runtime total mismatch"});

    plan.contract = contract;
}

const clang::Expr* ignoreParenImp(const clang::Expr* expr) {
    return expr ? expr->IgnoreParenImpCasts() : nullptr;
}

const clang::ValueDecl* declRefTarget(const clang::Expr* expr) {
    if (const auto* declRef =
            llvm::dyn_cast_or_null<clang::DeclRefExpr>(ignoreParenImp(expr))) {
        return declRef->getDecl();
    }
    return nullptr;
}

const clang::Expr* unwrapExprFully(const clang::Expr* expr);
const clang::Expr* unwrapVectorInitExpr(const clang::Expr* init);
bool constantInitValueExpr(const clang::Expr* expr,
                           clang::ASTContext* context,
                           const std::string& targetType,
                           bool allowAggregate,
                           std::string& valueExpr,
                           std::string& logValue);
bool stmtReferencesDecl(const clang::Stmt* stmt,
                        const clang::ValueDecl* decl);

const clang::VarDecl* declRefVarTarget(const clang::Expr* expr) {
    return llvm::dyn_cast_or_null<clang::VarDecl>(declRefTarget(expr));
}

const clang::Expr* stripAddressOf(const clang::Expr* expr) {
    expr = unwrapExprFully(expr);
    if (const auto* unary = llvm::dyn_cast_or_null<clang::UnaryOperator>(expr)) {
        if (unary->getOpcode() == clang::UO_AddrOf) {
            return unwrapExprFully(unary->getSubExpr());
        }
    }
    return expr;
}

bool isStdVectorLikeType(clang::QualType type) {
    const std::string typeName = type.getAsString();
    return typeName.find("vector") != std::string::npos ||
           typeName.find("Vector") != std::string::npos;
}

bool isArithmeticLikeType(clang::QualType type) {
    const clang::Type* raw = type.getTypePtrOrNull();
    return raw && raw->isArithmeticType();
}

bool isConstantFillElementType(clang::QualType type,
                               clang::ASTContext* context) {
    if (isArithmeticLikeType(type)) {
        return true;
    }
    return context && !type.isNull() && type.isTriviallyCopyableType(*context);
}

clang::QualType vectorElementType(const clang::VarDecl* vectorVar) {
    if (!vectorVar) {
        return {};
    }
    clang::QualType type = vectorVar->getType();
    const clang::Type* raw = type.getTypePtrOrNull();
    if (!raw) {
        return {};
    }
    if (const auto* specialization =
            raw->getAs<clang::TemplateSpecializationType>()) {
        const auto args = specialization->template_arguments();
        if (args.size() >= 1) {
            const clang::TemplateArgument& arg = args[0];
            if (arg.getKind() == clang::TemplateArgument::Type) {
                return arg.getAsType();
            }
        }
    }
    if (const auto* recordDecl = type->getAsCXXRecordDecl()) {
        if (const auto* specialization =
                llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                    recordDecl)) {
            const auto& args = specialization->getTemplateArgs();
            if (args.size() >= 1 &&
                args[0].getKind() == clang::TemplateArgument::Type) {
                return args[0].getAsType();
            }
        }
    }
    return {};
}

bool constantInitValueExpr(const clang::Expr* expr,
                           clang::ASTContext* context,
                           const std::string& targetType,
                           bool allowAggregate,
                           std::string& valueExpr,
                           std::string& logValue) {
    expr = unwrapExprFully(expr);
    if (!expr || !context) {
        return false;
    }
    if (const auto* construct =
            llvm::dyn_cast<clang::CXXConstructExpr>(expr)) {
        if (construct->getNumArgs() == 0) {
            valueExpr = targetType + "{}";
            logValue = "0";
            return true;
        }
        if (construct->getNumArgs() == 1) {
            return constantInitValueExpr(construct->getArg(0), context,
                                         targetType, allowAggregate,
                                         valueExpr, logValue);
        }
        return false;
    }
    if (llvm::isa<clang::CXXScalarValueInitExpr>(expr) ||
        llvm::isa<clang::ImplicitValueInitExpr>(expr)) {
        valueExpr = targetType + "{}";
        logValue = "0";
        return true;
    }
    if (const auto* initList = llvm::dyn_cast<clang::InitListExpr>(expr)) {
        if (initList->getNumInits() == 0) {
            valueExpr = targetType + "{}";
            logValue = "0";
            return true;
        }
        if (initList->getNumInits() == 1) {
            return constantInitValueExpr(initList->getInit(0), context,
                                         targetType, allowAggregate,
                                         valueExpr, logValue);
        }
        if (allowAggregate) {
            for (unsigned idx = 0; idx < initList->getNumInits(); ++idx) {
                std::string ignoredValue;
                std::string ignoredLog;
                if (!constantInitValueExpr(initList->getInit(idx), context,
                                           targetType, false, ignoredValue,
                                           ignoredLog)) {
                    return false;
                }
            }
            const std::string source = exprSource(initList, context);
            if (source.empty()) {
                return false;
            }
            valueExpr = targetType + source;
            logValue = source;
            return true;
        }
        return false;
    }
    if (const auto* intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
        valueExpr = exprSource(expr, context);
        logValue = intLit->getValue().isZero() ? "0" : valueExpr;
        return true;
    }
    if (const auto* floatLit =
            llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
        valueExpr = exprSource(expr, context);
        logValue = floatLit->getValue().isZero() ? "0" : valueExpr;
        return true;
    }
    if (const auto* boolLit = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
        valueExpr = boolLit->getValue() ? "true" : "false";
        logValue = valueExpr;
        return true;
    }
    if (llvm::isa<clang::CharacterLiteral>(expr)) {
        valueExpr = exprSource(expr, context);
        logValue = valueExpr;
        return !valueExpr.empty();
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
        if (unary->getOpcode() == clang::UO_Minus ||
            unary->getOpcode() == clang::UO_Plus) {
            std::string subValue;
            std::string subLog;
            if (constantInitValueExpr(unary->getSubExpr(), context,
                                      targetType, false, subValue, subLog)) {
                valueExpr = (unary->getOpcode() == clang::UO_Minus ? "-" : "+") +
                            subValue;
                logValue = valueExpr;
                return true;
            }
        }
    }
    return false;
}

bool constantInitValueLogEquivalent(const std::string& lhs,
                                    const std::string& rhs) {
    return compactExprText(lhs) == compactExprText(rhs);
}

bool vectorInitializerListUniformConstant(
    const clang::Expr* init,
    clang::ASTContext* context,
    const std::string& targetType,
    bool allowAggregate,
    std::string& valueExpr,
    std::string& logValue) {
    init = unwrapExprFully(init);
    const auto* initList = llvm::dyn_cast_or_null<clang::InitListExpr>(init);
    if (!initList) {
        if (const auto* stdList =
                llvm::dyn_cast_or_null<clang::CXXStdInitializerListExpr>(
                    init)) {
            initList = llvm::dyn_cast_or_null<clang::InitListExpr>(
                unwrapExprFully(stdList->getSubExpr()));
        }
    }
    if (!initList || initList->getNumInits() == 0) {
        return false;
    }
    std::string firstValue;
    std::string firstLog;
    if (!constantInitValueExpr(initList->getInit(0), context, targetType,
                               allowAggregate, firstValue, firstLog)) {
        return false;
    }
    for (unsigned idx = 1; idx < initList->getNumInits(); ++idx) {
        std::string itemValue;
        std::string itemLog;
        if (!constantInitValueExpr(initList->getInit(idx), context,
                                   targetType, allowAggregate, itemValue,
                                   itemLog)) {
            return false;
        }
        if (!constantInitValueLogEquivalent(firstLog, itemLog)) {
            return false;
        }
    }
    valueExpr = firstValue;
    logValue = firstLog;
    return true;
}

struct VectorConstantInit {
    bool supported = false;
    std::string valueExpr;
    std::string logValue;
    std::string reason;
};

struct VectorIndexFillInit {
    bool supported = false;
    std::string valueExpr;
    std::string logValue;
    std::string loopVarName;
    std::string reason;
    const clang::Stmt* loopStmt = nullptr;
    const clang::Stmt* assignmentStmt = nullptr;
};

VectorConstantInit analyzeVectorConstantConstructor(
    const clang::VarDecl* vectorVar,
    clang::ASTContext* context,
    const std::string& targetType) {
    VectorConstantInit result;
    if (!vectorVar || !context || !vectorVar->hasInit()) {
        result.reason = "vector initializer unavailable";
        return result;
    }
    if (!isStdVectorLikeType(vectorVar->getType())) {
        result.reason = "source initializer is not std::vector";
        return result;
    }
    clang::QualType elemType = vectorElementType(vectorVar);
    if (elemType.isNull()) {
        result.reason = "vector element type unavailable";
        return result;
    }
    if (!isConstantFillElementType(elemType, context)) {
        result.reason = "vector element type is not trivially-copyable";
        return result;
    }
    const bool allowAggregate = !isArithmeticLikeType(elemType);
    const clang::Expr* init = unwrapVectorInitExpr(vectorVar->getInit());
    const auto* construct = llvm::dyn_cast_or_null<clang::CXXConstructExpr>(init);
    if (const auto* parenInit =
            llvm::dyn_cast_or_null<clang::CXXParenListInitExpr>(init)) {
        const auto args = parenInit->getInitExprs();
        if (args.size() == 1) {
            const clang::Expr* sizeArg = unwrapExprFully(args[0]);
            if (!sizeArg || !sizeArg->getType()->isIntegerType()) {
                result.reason = "single-argument vector initializer is not a size constructor";
                return result;
            }
            result.supported = true;
            result.valueExpr = targetType + "{}";
            result.logValue = "0";
            return result;
        }
        if (args.size() >= 2) {
            if (constantInitValueExpr(args[1], context, targetType,
                                      allowAggregate, result.valueExpr,
                                      result.logValue)) {
                result.supported = true;
                return result;
            }
            result.reason = "vector fill value is not a supported constant";
            return result;
        }
        result.reason = "unsupported vector constructor arity";
        return result;
    }
    if (!construct) {
        if (vectorInitializerListUniformConstant(
                init, context, targetType, allowAggregate, result.valueExpr,
                result.logValue)) {
            result.supported = true;
            return result;
        }
        result.reason = "vector initializer is not a constructor";
        return result;
    }
    if (construct->getNumArgs() == 1) {
        const clang::Expr* sizeArg = unwrapExprFully(construct->getArg(0));
        if (!sizeArg || !sizeArg->getType()->isIntegerType()) {
            if (vectorInitializerListUniformConstant(
                    construct->getArg(0), context, targetType, allowAggregate,
                    result.valueExpr, result.logValue)) {
                result.supported = true;
                return result;
            }
            result.reason = "single-argument vector initializer is not a size constructor";
            return result;
        }
        result.supported = true;
        result.valueExpr = targetType + "{}";
        result.logValue = "0";
        return result;
    }
    if (construct->getNumArgs() >= 2) {
        if (constantInitValueExpr(construct->getArg(1), context, targetType,
                                  allowAggregate, result.valueExpr,
                                  result.logValue)) {
            result.supported = true;
            return result;
        }
        result.reason = "vector fill value is not a supported constant";
        return result;
    }
    result.reason = "unsupported vector constructor arity";
    return result;
}

const clang::Expr* vectorSizeConstructorArg(const clang::VarDecl* vectorVar) {
    if (!vectorVar || !vectorVar->hasInit()) {
        return nullptr;
    }
    const clang::Expr* init = unwrapVectorInitExpr(vectorVar->getInit());
    const auto* construct = llvm::dyn_cast_or_null<clang::CXXConstructExpr>(init);
    if (!construct || construct->getNumArgs() < 1) {
        if (const auto* parenInit =
                llvm::dyn_cast_or_null<clang::CXXParenListInitExpr>(init)) {
            const auto args = parenInit->getInitExprs();
            const clang::Expr* firstArg =
                args.empty() ? nullptr : unwrapExprFully(args[0]);
            if (firstArg && firstArg->getType()->isIntegerType()) {
                return args[0];
            }
        }
        return nullptr;
    }
    const clang::Expr* firstArg = unwrapExprFully(construct->getArg(0));
    if (!firstArg || !firstArg->getType()->isIntegerType()) {
        return nullptr;
    }
    return construct->getArg(0);
}

const clang::Expr* unwrapVectorInitExpr(const clang::Expr* init) {
    while (init) {
        init = init->IgnoreParenImpCasts();
        if (const auto* cleanups =
                llvm::dyn_cast<clang::ExprWithCleanups>(init)) {
            init = cleanups->getSubExpr();
            continue;
        }
        if (const auto* materialized =
                llvm::dyn_cast<clang::MaterializeTemporaryExpr>(init)) {
            init = materialized->getSubExpr();
            continue;
        }
        if (const auto* temporary =
                llvm::dyn_cast<clang::CXXBindTemporaryExpr>(init)) {
            init = temporary->getSubExpr();
            continue;
        }
        return init;
    }
    return nullptr;
}

bool isDeclRefToDecl(const clang::Expr* expr, const clang::ValueDecl* decl) {
    expr = unwrapExprFully(expr);
    const auto* declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
    return declRef && declRef->getDecl() == decl;
}

const clang::VarDecl* canonicalZeroBasedLoopVar(const clang::ForStmt* forStmt) {
    const auto* declStmt =
        llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt ? forStmt->getInit()
                                                        : nullptr);
    if (!declStmt || !declStmt->isSingleDecl()) {
        return nullptr;
    }
    const auto* loopVar =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    int64_t initValue = -1;
    if (!loopVar || !evaluateInt64Expr(loopVar->getInit(),
                                       &loopVar->getASTContext(), initValue) ||
        initValue != 0) {
        return nullptr;
    }
    const clang::Expr* inc = unwrapExprFully(forStmt->getInc());
    const auto* unary = llvm::dyn_cast_or_null<clang::UnaryOperator>(inc);
    if (!unary || (unary->getOpcode() != clang::UO_PostInc &&
                   unary->getOpcode() != clang::UO_PreInc) ||
        !isDeclRefToDecl(unary->getSubExpr(), loopVar)) {
        return nullptr;
    }
    return loopVar;
}

bool loopBoundMatchesVectorSize(const clang::ForStmt* forStmt,
                                const clang::VarDecl* loopVar,
                                const clang::Expr* vectorSize,
                                clang::ASTContext* context) {
    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt
                                                          ? forStmt->getCond()
                                                          : nullptr);
    if (!cond || cond->getOpcode() != clang::BO_LT ||
        !isDeclRefToDecl(cond->getLHS(), loopVar) || !vectorSize || !context) {
        return false;
    }
    return compactExprText(exprSource(cond->getRHS(), context)) ==
           compactExprText(exprSource(vectorSize, context));
}

bool lhsIsVectorAtLoopIndex(const clang::Expr* lhs,
                            const clang::VarDecl* vectorVar,
                            const clang::VarDecl* loopVar) {
    lhs = unwrapExprFully(lhs);
    const auto* subscript =
        llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(lhs);
    if (!subscript || subscript->getOperator() != clang::OO_Subscript ||
        subscript->getNumArgs() < 2 ||
        !isDeclRefToDecl(subscript->getArg(1), loopVar)) {
        return false;
    }
    const clang::VarDecl* baseVar = declRefVarTarget(subscript->getArg(0));
    if (!baseVar) {
        return false;
    }
    if (vectorVar) {
        return baseVar == vectorVar;
    }
    return isStdVectorLikeType(baseVar->getType());
}

class GeneratedIndexExprSafetyVisitor
    : public clang::RecursiveASTVisitor<GeneratedIndexExprSafetyVisitor> {
public:
    explicit GeneratedIndexExprSafetyVisitor(const clang::VarDecl* loopVar)
        : LoopVar(loopVar) {}

    bool Safe = true;
    std::string Reason;

    bool VisitDeclRefExpr(clang::DeclRefExpr* expr) {
        if (!expr || !expr->getDecl()) {
            return true;
        }
        const clang::ValueDecl* decl = expr->getDecl();
        if (decl == LoopVar || llvm::isa<clang::FunctionDecl>(decl) ||
            llvm::isa<clang::EnumConstantDecl>(decl)) {
            return true;
        }
        if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl)) {
            if (varDecl->hasGlobalStorage()) {
                return true;
            }
        }
        Safe = false;
        if (Reason.empty()) {
            Reason = "index initializer references local value not visible in wrapper";
        }
        return true;
    }

private:
    const clang::VarDecl* LoopVar = nullptr;
};

bool exprIsSafeGeneratedIndexValue(const clang::Expr* expr,
                                   const clang::VarDecl* loopVar,
                                   std::string& reason) {
    if (!expr || !loopVar) {
        reason = "index initializer expression unavailable";
        return false;
    }
    GeneratedIndexExprSafetyVisitor visitor(loopVar);
    visitor.TraverseStmt(const_cast<clang::Expr*>(expr));
    if (!visitor.Safe) {
        reason = visitor.Reason;
        return false;
    }
    return true;
}

bool stmtIsVectorIndexAssignment(const clang::Stmt* stmt,
                                 const clang::VarDecl* vectorVar,
                                 const clang::VarDecl* loopVar,
                                 clang::ASTContext* context,
                                 std::string& valueExpr,
                                 std::string& reason) {
    const auto* expr = llvm::dyn_cast_or_null<clang::Expr>(stmt);
    expr = unwrapExprFully(expr);
    const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(expr);
    if (!binary || binary->getOpcode() != clang::BO_Assign ||
        !lhsIsVectorAtLoopIndex(binary->getLHS(), vectorVar, loopVar)) {
        return false;
    }
    if (stmtReferencesDecl(binary->getRHS(), vectorVar)) {
        reason = "index initializer reads the vector being initialized";
        return true;
    }
    if (!exprIsSafeGeneratedIndexValue(binary->getRHS(), loopVar, reason)) {
        return true;
    }
    valueExpr = exprSource(binary->getRHS(), context);
    if (valueExpr.empty()) {
        reason = "index initializer source unavailable";
    }
    return true;
}

std::vector<const clang::Stmt*> loopBodyStatements(const clang::Stmt* body) {
    std::vector<const clang::Stmt*> result;
    if (!body) {
        return result;
    }
    if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(body)) {
        for (const clang::Stmt* child : compound->body()) {
            result.push_back(child);
        }
    } else {
        result.push_back(body);
    }
    return result;
}

VectorIndexFillInit analyzeVectorIndexFillLoop(
    const clang::Stmt* stmt,
    const clang::VarDecl* vectorVar,
    const clang::Expr* vectorSize,
    clang::ASTContext* context) {
    VectorIndexFillInit result;
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    const clang::VarDecl* loopVar = canonicalZeroBasedLoopVar(forStmt);
    if (!forStmt || !loopVar ||
        !loopBoundMatchesVectorSize(forStmt, loopVar, vectorSize, context)) {
        result.reason = "not a full zero-based vector fill loop";
        return result;
    }
    int assignmentCount = 0;
    std::string valueExpr;
    const clang::Stmt* assignmentStmt = nullptr;
    for (const clang::Stmt* child : loopBodyStatements(forStmt->getBody())) {
        std::string childValue;
        std::string childReason;
        if (stmtIsVectorIndexAssignment(child, vectorVar, loopVar, context,
                                        childValue, childReason)) {
            if (!childReason.empty()) {
                result.reason = childReason;
                return result;
            }
            ++assignmentCount;
            valueExpr = childValue;
            assignmentStmt = child;
            continue;
        }
        std::string ignoredValue;
        std::string ignoredReason;
        if (stmtIsVectorIndexAssignment(child, nullptr, loopVar, context,
                                        ignoredValue, ignoredReason) &&
            !stmtReferencesDecl(child, vectorVar)) {
            continue;
        }
        if (stmtReferencesDecl(child, vectorVar)) {
            result.reason = "unsupported vector use inside index fill loop";
            return result;
        }
    }
    if (assignmentCount != 1 || valueExpr.empty()) {
        result.reason = "index fill loop does not assign the vector exactly once";
        return result;
    }
    result.supported = true;
    result.valueExpr = valueExpr;
    result.logValue = valueExpr;
    result.loopVarName = loopVar->getNameAsString();
    result.loopStmt = stmt;
    result.assignmentStmt = assignmentStmt;
    return result;
}

const clang::VarDecl* findVectorDeclInTensorInitializer(
    const clang::VarDecl* tensorVar) {
    if (!tensorVar || !tensorVar->hasInit()) {
        return nullptr;
    }
    const clang::Stmt* root = tensorVar->getInit();
    std::vector<const clang::Stmt*> worklist;
    worklist.push_back(root);
    while (!worklist.empty()) {
        const clang::Stmt* current = worklist.back();
        worklist.pop_back();
        if (!current) {
            continue;
        }
        if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(current)) {
            const auto* varDecl =
                llvm::dyn_cast_or_null<clang::VarDecl>(declRef->getDecl());
            if (varDecl && isStdVectorLikeType(varDecl->getType())) {
                return varDecl;
            }
        }
        for (const clang::Stmt* child : current->children()) {
            worklist.push_back(child);
        }
    }
    return nullptr;
}

bool stmtIsDeclForVar(const clang::Stmt* stmt, const clang::VarDecl* varDecl) {
    const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(stmt);
    if (!declStmt || !varDecl) {
        return false;
    }
    for (const clang::Decl* decl : declStmt->decls()) {
        if (decl == varDecl) {
            return true;
        }
    }
    return false;
}

const clang::Expr* calleeObjectExpr(const clang::CXXMemberCallExpr* call) {
    if (!call) {
        return nullptr;
    }
    return call->getImplicitObjectArgument();
}

bool callIsKnownReadOnlyVectorMember(const clang::CXXMemberCallExpr* call) {
    if (!call || !call->getMethodDecl()) {
        return false;
    }
    const std::string name = call->getMethodDecl()->getNameAsString();
    return name == "size" || name == "empty" || name == "capacity" ||
           name == "max_size";
}

class VectorUseSafetyVisitor
    : public clang::RecursiveASTVisitor<VectorUseSafetyVisitor> {
public:
    const clang::VarDecl* Target = nullptr;
    bool SawUse = false;
    bool Unsafe = false;
    std::string Reason;

    explicit VectorUseSafetyVisitor(const clang::VarDecl* target)
        : Target(target) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* expr) {
        if (!expr || expr->getDecl() != Target) {
            return true;
        }
        SawUse = true;
        if (AllowedRefs.count(expr) == 0) {
            Unsafe = true;
            if (Reason.empty()) {
                Reason = "vector alias/function escape before tensor construction";
            }
        }
        return true;
    }

    bool VisitBinaryOperator(clang::BinaryOperator* op) {
        if (!op || !op->isAssignmentOp()) {
            return true;
        }
        if (exprReferencesTarget(op->getLHS())) {
            Unsafe = true;
            if (Reason.empty()) {
                Reason = "vector write before tensor construction";
            }
        }
        return true;
    }

    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* call) {
        if (!call || call->getNumArgs() == 0) {
            return true;
        }
        if (call->isAssignmentOp() && exprReferencesTarget(call->getArg(0))) {
            Unsafe = true;
            if (Reason.empty()) {
                Reason = "vector write before tensor construction";
            }
        }
        if ((call->getOperator() == clang::OO_Subscript ||
             call->getOperator() == clang::OO_Call) &&
            exprReferencesTarget(call->getArg(0))) {
            Unsafe = true;
            if (Reason.empty()) {
                Reason = "vector element access before tensor construction";
            }
        }
        return true;
    }

    bool VisitUnaryOperator(clang::UnaryOperator* op) {
        if (!op) {
            return true;
        }
        if (op->isIncrementDecrementOp() && exprReferencesTarget(op->getSubExpr())) {
            Unsafe = true;
            if (Reason.empty()) {
                Reason = "vector write before tensor construction";
            }
        }
        if (op->getOpcode() == clang::UO_AddrOf &&
            exprReferencesTarget(op->getSubExpr())) {
            Unsafe = true;
            if (Reason.empty()) {
                Reason = "vector address escape before tensor construction";
            }
        }
        return true;
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call || !exprReferencesTarget(calleeObjectExpr(call))) {
            return true;
        }
        if (!callIsKnownReadOnlyVectorMember(call)) {
            Unsafe = true;
            if (Reason.empty()) {
                Reason = "vector mutating/unknown member call before tensor construction";
            }
        }
        return true;
    }

private:
    std::set<const clang::DeclRefExpr*> AllowedRefs;

    bool exprReferencesTarget(const clang::Expr* expr) {
        expr = stripAddressOf(expr);
        if (!expr) {
            return false;
        }
        if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
            if (declRef->getDecl() == Target) {
                AllowedRefs.insert(declRef);
                return true;
            }
        }
        bool found = false;
        std::vector<const clang::Stmt*> worklist;
        worklist.push_back(expr);
        while (!worklist.empty()) {
            const clang::Stmt* current = worklist.back();
            worklist.pop_back();
            if (!current) {
                continue;
            }
            if (const auto* declRef =
                    llvm::dyn_cast<clang::DeclRefExpr>(current)) {
                if (declRef->getDecl() == Target) {
                    AllowedRefs.insert(declRef);
                    found = true;
                }
            }
            for (const clang::Stmt* child : current->children()) {
                worklist.push_back(child);
            }
        }
        return found;
    }
};

bool stmtReferencesDecl(const clang::Stmt* stmt,
                        const clang::ValueDecl* decl) {
    if (!stmt || !decl) {
        return false;
    }
    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
        if (declRef->getDecl() == decl) {
            return true;
        }
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (stmtReferencesDecl(child, decl)) {
            return true;
        }
    }
    return false;
}

bool stmtContainsDeclReference(const clang::Stmt* stmt,
                               const clang::ValueDecl* decl) {
    return stmtReferencesDecl(stmt, decl);
}

std::string tensorReferenceEscapeReason(const clang::Stmt* stmt,
                                        const clang::VarDecl* tensorVar) {
    if (!stmt || !tensorVar || !stmtReferencesDecl(stmt, tensorVar)) {
        return "";
    }
    if (stmtIsDeclForVar(stmt, tensorVar)) {
        return "";
    }
    if (const auto* expr = llvm::dyn_cast<clang::Expr>(stmt)) {
        const clang::Expr* unwrapped = unwrapExprFully(expr);
        if (const auto* declRef =
                llvm::dyn_cast_or_null<clang::DeclRefExpr>(unwrapped)) {
            if (declRef->getDecl() == tensorVar) {
                return "tensor referenced before shell call";
            }
        }
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
        if (binary->isAssignmentOp() && stmtReferencesDecl(binary->getLHS(), tensorVar)) {
            return "tensor write before shell call";
        }
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->isAssignmentOp() && opCall->getNumArgs() > 0 &&
            stmtReferencesDecl(opCall->getArg(0), tensorVar)) {
            return "tensor write before shell call";
        }
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() > 0 &&
            declRefVarTarget(opCall->getArg(0)) == tensorVar) {
            return "tensor view/slice before shell call";
        }
    }
    if (const auto* call = llvm::dyn_cast<clang::CallExpr>(stmt)) {
        for (const clang::Expr* arg : call->arguments()) {
            if (stmtReferencesDecl(arg, tensorVar)) {
                return "tensor function escape before shell call";
            }
        }
    }
    if (const auto* memberCall =
            llvm::dyn_cast<clang::CXXMemberCallExpr>(stmt)) {
        if (stmtReferencesDecl(calleeObjectExpr(memberCall), tensorVar)) {
            return "tensor member call before shell call";
        }
    }
    return "tensor referenced before shell call";
}

bool findTensorReferenceBeforeLimit(const clang::Stmt* stmt,
                                    const clang::VarDecl* tensorVar,
                                    const clang::Expr* limitExpr,
                                    const clang::SourceManager& sourceManager,
                                    std::string& reason) {
    if (!stmt || !tensorVar || !limitExpr ||
        stmt->getBeginLoc().isInvalid()) {
        return false;
    }
    if (!sourceManager.isBeforeInTranslationUnit(stmt->getBeginLoc(),
                                                 limitExpr->getBeginLoc())) {
        return false;
    }
    if (sourceRangeContains(sourceManager, stmt->getSourceRange(),
                            limitExpr->getSourceRange())) {
        for (const clang::Stmt* child : stmt->children()) {
            if (findTensorReferenceBeforeLimit(child, tensorVar, limitExpr,
                                               sourceManager, reason)) {
                return true;
            }
        }
        return false;
    }
    reason = tensorReferenceEscapeReason(stmt, tensorVar);
    return !reason.empty();
}

bool sourceBefore(const clang::SourceManager& sourceManager,
                  clang::SourceLocation lhs,
                  clang::SourceLocation rhs) {
    if (lhs.isInvalid() || rhs.isInvalid()) {
        return false;
    }
    return sourceManager.isBeforeInTranslationUnit(lhs, rhs);
}

const clang::CompoundStmt* enclosingFunctionBodyForExpr(
    clang::ASTContext* context,
    const clang::Expr* expr) {
    if (!context || !expr) {
        return nullptr;
    }
    clang::DynTypedNode current = clang::DynTypedNode::create(*expr);
    for (int depth = 0; depth < 64; ++depth) {
        auto parents = context->getParents(current);
        if (parents.empty()) {
            return nullptr;
        }
        const clang::DynTypedNode& parent = parents[0];
        if (const auto* functionDecl = parent.get<clang::FunctionDecl>()) {
            return llvm::dyn_cast_or_null<clang::CompoundStmt>(
                functionDecl->getBody());
        }
        if (const auto* stmt = parent.get<clang::Stmt>()) {
            current = clang::DynTypedNode::create(*stmt);
            continue;
        }
        return nullptr;
    }
    return nullptr;
}

ConstantInitPlan analyzeConstantInitForShellArg(
    DacppFile* dacppFile,
    const clang::Expr* shellArg,
    const std::string& targetType) {
    ConstantInitPlan plan;
    if (!dacppFile || !dacppFile->getContext() || !shellArg) {
        plan.reason = "analysis context unavailable";
        return plan;
    }
    const auto* tensorVar = declRefVarTarget(shellArg);
    if (!tensorVar) {
        plan.reason = "shell argument is not a direct tensor variable";
        return plan;
    }
    if (!tensorVar->hasInit()) {
        plan.reason = "tensor initializer unavailable";
        return plan;
    }
    const clang::VarDecl* vectorVar =
        findVectorDeclInTensorInitializer(tensorVar);
    if (!vectorVar) {
        plan.reason = "tensor is not constructed from a vector variable";
        return plan;
    }
    const VectorConstantInit vectorInit =
        analyzeVectorConstantConstructor(vectorVar, dacppFile->getContext(),
                                         targetType);
    const clang::Expr* vectorSize = vectorSizeConstructorArg(vectorVar);

    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    if (!sourceBefore(sourceManager, vectorVar->getBeginLoc(),
                      tensorVar->getBeginLoc())) {
        plan.reason = "vector declaration is not before tensor construction";
        return plan;
    }
    if (!sourceBefore(sourceManager, tensorVar->getBeginLoc(),
                      shellArg->getBeginLoc())) {
        plan.reason = "tensor declaration is not before shell call";
        return plan;
    }

    bool sawVectorDecl = false;
    bool sawTensorDecl = false;
    bool checkedVectorWindow = false;
    bool checkedTensorWindow = false;
    bool sawIndexFill = false;
    VectorIndexFillInit indexFill;
    const auto* compound =
        enclosingFunctionBodyForExpr(dacppFile->getContext(), shellArg);
    if (!compound && dacppFile->getMainBody()) {
        compound = llvm::dyn_cast<clang::CompoundStmt>(
            dacppFile->getMainBody());
    }
    if (!compound) {
        plan.reason = "enclosing function body is not compound";
        return plan;
    }
    for (const clang::Stmt* child : compound->body()) {
        if (!child) {
            continue;
        }
        if (!sawVectorDecl) {
            if (stmtIsDeclForVar(child, vectorVar)) {
                sawVectorDecl = true;
            }
            continue;
        }
        if (!sawTensorDecl) {
            if (stmtIsDeclForVar(child, tensorVar)) {
                sawTensorDecl = true;
                checkedVectorWindow = true;
                continue;
            }
            if (!sawIndexFill) {
                indexFill = analyzeVectorIndexFillLoop(
                    child, vectorVar, vectorSize, dacppFile->getContext());
                if (indexFill.supported) {
                    sawIndexFill = true;
                    continue;
                }
            }
            VectorUseSafetyVisitor visitor(vectorVar);
            visitor.TraverseStmt(const_cast<clang::Stmt*>(child));
            if (visitor.Unsafe) {
                plan.reason = visitor.Reason.empty()
                                  ? "vector unsafe use before tensor construction"
                                  : visitor.Reason;
                return plan;
            }
            continue;
        }
        std::string tensorReason;
        if (findTensorReferenceBeforeLimit(child, tensorVar, shellArg,
                                           sourceManager, tensorReason)) {
            plan.reason = tensorReason;
            return plan;
        }
        if (sourceRangeContains(sourceManager, child->getSourceRange(),
                                shellArg->getSourceRange())) {
            break;
        }
        tensorReason = tensorReferenceEscapeReason(child, tensorVar);
        if (!tensorReason.empty()) {
            plan.reason = tensorReason;
            return plan;
        }
    }
    if (!sawVectorDecl) {
        plan.reason = "vector declaration not found in enclosing function body";
        return plan;
    }
    if (!sawTensorDecl || !checkedVectorWindow) {
        plan.reason = "tensor declaration not found after vector";
        return plan;
    }
    (void)checkedTensorWindow;
    if (sawIndexFill && indexFill.supported) {
        plan.supported = true;
        plan.indexExpr = true;
        plan.valueExpr = indexFill.valueExpr;
        plan.logValue = indexFill.logValue.empty() ? indexFill.valueExpr
                                                   : indexFill.logValue;
        plan.globalIndexName = indexFill.loopVarName.empty()
                                   ? "__or_global_index"
                                   : indexFill.loopVarName;
        plan.indexFillLoopStmt = indexFill.loopStmt;
        plan.indexFillAssignmentStmt = indexFill.assignmentStmt;
        return plan;
    }
    if (vectorInit.supported) {
        plan.supported = true;
        plan.valueExpr = vectorInit.valueExpr;
        plan.logValue = vectorInit.logValue.empty() ? vectorInit.valueExpr
                                                    : vectorInit.logValue;
        return plan;
    }
    plan.reason = vectorInit.reason.empty()
                      ? "vector constant initializer unsupported"
                      : vectorInit.reason;
    return plan;
}

struct DefaultInitAnalysis {
    bool supported = false;
    std::string valueExpr;
    std::string logValue;
    std::string reason;
};

DefaultInitAnalysis analyzeDefaultOutputInitForShellArg(
    DacppFile* dacppFile,
    const clang::Expr* shellArg,
    const std::string& targetType) {
    DefaultInitAnalysis result;
    if (!dacppFile || !dacppFile->getContext() || !shellArg) {
        result.reason = "analysis context unavailable";
        return result;
    }
    const auto* tensorVar = declRefVarTarget(shellArg);
    if (!tensorVar || !tensorVar->hasInit()) {
        result.reason = "output argument is not a direct initialized tensor";
        return result;
    }
    const clang::VarDecl* vectorVar =
        findVectorDeclInTensorInitializer(tensorVar);
    if (!vectorVar) {
        result.reason = "output tensor is not constructed from a vector";
        return result;
    }
    const VectorConstantInit vectorInit =
        analyzeVectorConstantConstructor(vectorVar, dacppFile->getContext(),
                                         targetType);
    if (!vectorInit.supported) {
        result.reason = vectorInit.reason.empty()
                            ? "output vector initializer unsupported"
                            : vectorInit.reason;
        return result;
    }
    const std::string compactValue = compactExprText(vectorInit.logValue.empty()
                                                         ? vectorInit.valueExpr
                                                         : vectorInit.logValue);
    const std::string compactDefault =
        compactExprText(targetType + "{}");
    if (compactValue != "0" && compactValue != "0.0" &&
        compactValue != "0.0f" && compactValue != compactDefault &&
        compactValue != "{}") {
        result.reason = "output initializer is not default/zero";
        return result;
    }
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    if (!sourceBefore(sourceManager, vectorVar->getBeginLoc(),
                      tensorVar->getBeginLoc())) {
        result.reason = "output vector declaration is not before tensor construction";
        return result;
    }
    if (!sourceBefore(sourceManager, tensorVar->getBeginLoc(),
                      shellArg->getBeginLoc())) {
        result.reason = "output tensor declaration is not before shell call";
        return result;
    }
    const auto* compound =
        enclosingFunctionBodyForExpr(dacppFile->getContext(), shellArg);
    if (!compound && dacppFile->getMainBody()) {
        compound = llvm::dyn_cast<clang::CompoundStmt>(
            dacppFile->getMainBody());
    }
    if (!compound) {
        result.reason = "output enclosing function body is not compound";
        return result;
    }
    bool sawTensorDecl = false;
    for (const clang::Stmt* child : compound->body()) {
        if (!child) {
            continue;
        }
        if (!sawTensorDecl) {
            if (stmtIsDeclForVar(child, tensorVar)) {
                sawTensorDecl = true;
            }
            continue;
        }
        std::string tensorReason;
        if (findTensorReferenceBeforeLimit(child, tensorVar, shellArg,
                                           sourceManager, tensorReason)) {
            result.reason = tensorReason;
            return result;
        }
        if (sourceRangeContains(sourceManager, child->getSourceRange(),
                                shellArg->getSourceRange())) {
            break;
        }
        tensorReason = tensorReferenceEscapeReason(child, tensorVar);
        if (!tensorReason.empty()) {
            result.reason = tensorReason;
            return result;
        }
    }
    if (!sawTensorDecl) {
        result.reason = "output tensor declaration not found in enclosing function body";
        return result;
    }
    result.supported = true;
    result.valueExpr = vectorInit.valueExpr.empty() ? targetType + "{}"
                                                    : vectorInit.valueExpr;
    result.logValue = vectorInit.logValue.empty() ? "0" : vectorInit.logValue;
    return result;
}

bool exprIsSubscriptOfParam(const clang::Expr* expr,
                            const clang::ValueDecl* paramDecl) {
    expr = unwrapExprFully(expr);
    if (!expr || !paramDecl) {
        return false;
    }
    if (const auto* arraySub =
            llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        return declRefTarget(arraySub->getBase()) == paramDecl;
    }
    if (const auto* opCall =
            llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        return opCall->getOperator() == clang::OO_Subscript &&
               opCall->getNumArgs() > 0 &&
               declRefTarget(opCall->getArg(0)) == paramDecl;
    }
    return false;
}

class LocalAccumOutputVisitor
    : public clang::RecursiveASTVisitor<LocalAccumOutputVisitor> {
public:
    explicit LocalAccumOutputVisitor(const clang::ValueDecl* paramDecl)
        : ParamDecl(paramDecl) {}

    bool Safe = true;
    std::string Reason;
    int AccumWrites = 0;

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* op) {
        if (!op || !Safe) {
            return true;
        }
        if (!exprIsSubscriptOfParam(op->getLHS(), ParamDecl)) {
            if (stmtContainsDeclReference(op, ParamDecl)) {
                reject("unsupported output compound assignment");
            }
            return clang::RecursiveASTVisitor<
                LocalAccumOutputVisitor>::TraverseCompoundAssignOperator(op);
        }
        if (op->getOpcode() != clang::BO_AddAssign) {
            reject("output compound assignment is not +=");
            return true;
        }
        if (stmtContainsDeclReference(op->getRHS(), ParamDecl)) {
            reject("output accumulation RHS reads output");
            return true;
        }
        ++AccumWrites;
        TraverseStmt(op->getRHS());
        return true;
    }

    bool TraverseBinaryOperator(clang::BinaryOperator* op) {
        if (!op || !Safe) {
            return true;
        }
        if (op->isAssignmentOp() &&
            stmtContainsDeclReference(op->getLHS(), ParamDecl)) {
            if (op->getOpcode() == clang::BO_Assign) {
                reject("output uses direct assignment, no initial sync needed");
            } else {
                reject("unsupported output assignment operator");
            }
            return true;
        }
        return clang::RecursiveASTVisitor<
            LocalAccumOutputVisitor>::TraverseBinaryOperator(op);
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* op) {
        if (!op || !Safe) {
            return true;
        }
        if (op->isIncrementDecrementOp() &&
            stmtContainsDeclReference(op->getSubExpr(), ParamDecl)) {
            reject("output increment/decrement requires old value");
            return true;
        }
        return clang::RecursiveASTVisitor<
            LocalAccumOutputVisitor>::TraverseUnaryOperator(op);
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* expr) {
        if (expr && expr->getDecl() == ParamDecl) {
            reject("output read outside supported local accumulation");
        }
        return true;
    }

private:
    const clang::ValueDecl* ParamDecl = nullptr;

    void reject(const std::string& reason) {
        Safe = false;
        if (Reason.empty()) {
            Reason = reason;
        }
    }
};

bool calcSupportsDefaultLocalAccumulation(const ShellPartitionPlan& plan,
                                          const ParamAccessPlan& param,
                                          std::string& reason) {
    reason.clear();
    if (!plan.exprNode.calc || !plan.exprNode.calc->getCalcLoc() ||
        !plan.exprNode.calc->getCalcLoc()->getBody()) {
        reason = "calc body unavailable";
        return false;
    }
    clang::FunctionDecl* calcLoc = plan.exprNode.calc->getCalcLoc();
    if (param.paramIndex < 0 ||
        param.paramIndex >= static_cast<int>(calcLoc->getNumParams())) {
        reason = "calc parameter unavailable";
        return false;
    }
    const clang::ValueDecl* paramDecl =
        calcLoc->getParamDecl(param.paramIndex);
    LocalAccumOutputVisitor visitor(paramDecl);
    visitor.TraverseStmt(calcLoc->getBody());
    if (!visitor.Safe) {
        reason = visitor.Reason.empty()
                     ? "output accumulation pattern unsupported"
                     : visitor.Reason;
        return false;
    }
    if (visitor.AccumWrites == 0) {
        reason = "no supported local accumulation write";
        return false;
    }
    return true;
}

void annotateOutputDirectInit(ShellPartitionPlan& plan, DacppFile* dacppFile) {
    if (!dacppFile || !plan.exprNode.dacExpr) {
        return;
    }
    const clang::CallExpr* shellCall = getShellCallExpr(plan.exprNode.dacExpr);
    for (auto& param : plan.params) {
        if (!param.writes || param.access != ParamAccessKind::OutputDirect ||
            !param.reads) {
            if (param.writes &&
                param.access == ParamAccessKind::OutputDirect) {
                param.outputInit.skipInitialSync = true;
                param.outputInit.valueExpr = analysisElemType(plan, param) + "{}";
                param.outputInit.logValue = "0";
                param.outputInit.reason = "write-only output";
                llvm::outs() << "[DACPP][MPI][OR] output "
                             << param.actualTensorName
                             << " init-sync=local-default reason=write-only output\n";
            }
            continue;
        }
        std::string reason;
        if (!calcSupportsDefaultLocalAccumulation(plan, param, reason)) {
            param.outputInit.reason = reason;
            llvm::outs() << "[DACPP][MPI][OR] output "
                         << param.actualTensorName
                         << " init-sync=scatter fallback reason=" << reason
                         << "\n";
            continue;
        }
        if (!shellCall || param.paramIndex < 0 ||
            param.paramIndex >= static_cast<int>(shellCall->getNumArgs())) {
            param.outputInit.reason = "shell argument unavailable";
            llvm::outs() << "[DACPP][MPI][OR] output "
                         << param.actualTensorName
                         << " init-sync=scatter fallback reason="
                         << param.outputInit.reason << "\n";
            continue;
        }
        const DefaultInitAnalysis init =
            analyzeDefaultOutputInitForShellArg(
                dacppFile, shellCall->getArg(param.paramIndex),
                analysisElemType(plan, param));
        if (!init.supported) {
            param.outputInit.reason = init.reason;
            llvm::outs() << "[DACPP][MPI][OR] output "
                         << param.actualTensorName
                         << " init-sync=scatter fallback reason="
                         << param.outputInit.reason << "\n";
            continue;
        }
        param.outputInit.skipInitialSync = true;
        param.outputInit.valueExpr = init.valueExpr;
        param.outputInit.logValue = init.logValue;
        llvm::outs() << "[DACPP][MPI][OR] output " << param.actualTensorName
                     << " init-sync=local-default value="
                     << (param.outputInit.logValue.empty()
                             ? param.outputInit.valueExpr
                             : param.outputInit.logValue)
                     << " reason=default-initialized local accumulation\n";
    }
}

bool isFixedBlockPlanWithBlockSize(const ShellPartitionPlan& plan,
                                   int blockSize,
                                   int blockStride) {
    if (!plan.supported ||
        plan.signature.layout != LocalLayoutKind::FixedBlock) {
        return false;
    }
    for (const auto& param : plan.params) {
        if (param.access != ParamAccessKind::FixedBlock) {
            return false;
        }
        if (param.fixedBlockSize != blockSize ||
            param.fixedBlockStride != blockStride) {
            return false;
        }
    }
    return true;
}

const ParamAccessPlan* fixedBlockReaderParam(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::FixedBlock && param.reads &&
            !param.writes) {
            return &param;
        }
    }
    return nullptr;
}

const ParamAccessPlan* fixedBlockWriterParam(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::FixedBlock && param.writes &&
            !param.reads) {
            return &param;
        }
    }
    return nullptr;
}

const clang::Expr* unwrapExprFully(const clang::Expr* expr) {
    if (!expr) {
        return nullptr;
    }
    while (true) {
        const clang::Expr* next = expr->IgnoreParenImpCasts();
        if (const auto* cleanups =
                llvm::dyn_cast<clang::ExprWithCleanups>(next)) {
            next = cleanups->getSubExpr();
        } else if (const auto* materialized =
                       llvm::dyn_cast<clang::MaterializeTemporaryExpr>(next)) {
            next = materialized->getSubExpr();
        } else if (const auto* temporary =
                       llvm::dyn_cast<clang::CXXBindTemporaryExpr>(next)) {
            next = temporary->getSubExpr();
        } else if (const auto* construct =
                       llvm::dyn_cast<clang::CXXConstructExpr>(next)) {
            // Strip single-argument constructors (copy/move/conversion).
            if (construct->getNumArgs() == 1) {
                next = construct->getArg(0);
            } else {
                return next;
            }
        }
        if (!next || next == expr) {
            return next;
        }
        expr = next;
    }
}

// Recognizes `T[{offsetExpr, endExpr}]` and returns offset value if it parses
// as an integer literal. Returns -1 otherwise. On match, *baseDecl is set
// to the decl of T.
int recognizePhaseShiftSlice(const clang::Expr* expr,
                             const clang::ValueDecl** baseDecl) {
    expr = unwrapExprFully(expr);
    if (!expr) {
        return -1;
    }
    const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr);
    if (!opCall || opCall->getOperator() != clang::OO_Subscript ||
        opCall->getNumArgs() < 2) {
        return -1;
    }
    const clang::Expr* base = unwrapExprFully(opCall->getArg(0));
    const clang::Expr* index = unwrapExprFully(opCall->getArg(1));
    if (!base || !index) {
        return -1;
    }
    const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(base);
    if (!declRef) {
        return -1;
    }
    if (baseDecl) {
        *baseDecl = declRef->getDecl();
    }
    // index should be {offset, end}: an InitListExpr, CXXStdInitializerListExpr,
    // or CXXConstructExpr.
    const clang::Expr* offsetArg = nullptr;
    const auto* listExpr = llvm::dyn_cast<clang::InitListExpr>(index);
    if (!listExpr) {
        if (const auto* stdList =
                llvm::dyn_cast<clang::CXXStdInitializerListExpr>(index)) {
            const clang::Expr* sub = unwrapExprFully(stdList->getSubExpr());
            listExpr = llvm::dyn_cast_or_null<clang::InitListExpr>(sub);
        }
    }
    if (listExpr) {
        if (listExpr->getNumInits() != 2) {
            return -1;
        }
        offsetArg = unwrapExprFully(listExpr->getInit(0));
    } else if (const auto* construct =
                   llvm::dyn_cast<clang::CXXConstructExpr>(index)) {
        if (construct->getNumArgs() < 2) {
            return -1;
        }
        offsetArg = unwrapExprFully(construct->getArg(0));
    } else {
        return -1;
    }
    if (!offsetArg) {
        return -1;
    }
    if (const auto* intLit =
            llvm::dyn_cast<clang::IntegerLiteral>(offsetArg)) {
        return static_cast<int>(intLit->getValue().getSExtValue());
    }
    return -1;
}

const clang::Expr* skipImplicitWrappers(const clang::Expr* expr) {
    return unwrapExprFully(expr);
}

bool isTensorIndexExpr(const clang::Expr* expr,
                       const clang::ValueDecl* tensorDecl,
                       const clang::Expr** indexOut) {
    expr = skipImplicitWrappers(expr);
    if (!expr) {
        return false;
    }
    if (const auto* opCall =
            llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() != clang::OO_Subscript ||
            opCall->getNumArgs() < 2) {
            return false;
        }
        if (declRefTarget(opCall->getArg(0)) != tensorDecl) {
            return false;
        }
        if (indexOut) {
            *indexOut = ignoreParenImp(opCall->getArg(1));
        }
        return true;
    }
    return false;
}

bool isIntegerLiteralValue(const clang::Expr* expr, int64_t value) {
    expr = ignoreParenImp(expr);
    if (const auto* lit = llvm::dyn_cast_or_null<clang::IntegerLiteral>(expr)) {
        return lit->getValue().getSExtValue() == value;
    }
    return false;
}

bool isLastIndexExpr(const clang::Expr* expr) {
    expr = ignoreParenImp(expr);
    if (!expr) {
        return false;
    }
    if (const auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
        if (binOp->getOpcode() == clang::BO_Sub &&
            isIntegerLiteralValue(binOp->getRHS(), 1)) {
            return true;
        }
    }
    return false;
}

bool isLoopVarPlusOrMinus(const clang::Expr* expr,
                          const clang::ValueDecl* loopVar,
                          int delta) {
    expr = ignoreParenImp(expr);
    if (!expr) {
        return false;
    }
    if (delta == 0) {
        return declRefTarget(expr) == loopVar;
    }
    if (const auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
        if (binOp->getOpcode() == clang::BO_Sub && delta == -1 &&
            declRefTarget(binOp->getLHS()) == loopVar &&
            isIntegerLiteralValue(binOp->getRHS(), 1)) {
            return true;
        }
        if (binOp->getOpcode() == clang::BO_Add && delta == 1 &&
            declRefTarget(binOp->getLHS()) == loopVar &&
            isIntegerLiteralValue(binOp->getRHS(), 1)) {
            return true;
        }
    }
    return false;
}

// Look at the slice that initialized array2_tensor. We accept tensor decl
// `dacpp::Tensor<E,1> array2_tensor = T_out[{1, end}];`
bool tryRecognizePhaseSliceDecl(const clang::Stmt* stmt,
                                const clang::ValueDecl* expectedBase,
                                const clang::ValueDecl** sliceVarOut,
                                int* offsetOut) {
    const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(stmt);
    if (!declStmt || !declStmt->isSingleDecl()) {
        return false;
    }
    const auto* varDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    if (!varDecl || !varDecl->hasInit()) {
        return false;
    }
    const clang::ValueDecl* baseDecl = nullptr;
    int offset = recognizePhaseShiftSlice(varDecl->getInit(), &baseDecl);
    if (offset < 0 || baseDecl != expectedBase) {
        return false;
    }
    if (sliceVarOut) {
        *sliceVarOut = varDecl;
    }
    if (offsetOut) {
        *offsetOut = offset;
    }
    return true;
}

bool isVectorOrTensorDeclStmt(const clang::Stmt* stmt,
                              const clang::ValueDecl** declOut) {
    const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(stmt);
    if (!declStmt || !declStmt->isSingleDecl()) {
        return false;
    }
    const auto* varDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    if (!varDecl) {
        return false;
    }
    if (declOut) {
        *declOut = varDecl;
    }
    return true;
}

bool isInteriorCopyForLoop(const clang::Stmt* stmt,
                           const clang::ValueDecl* sourceTensor,
                           const clang::ValueDecl* phaseBWriter) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt) {
        return false;
    }
    // Init must declare `int i = 1` (DeclStmt with init = 1)
    const auto* initDecl =
        llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
    if (!initDecl || !initDecl->isSingleDecl()) {
        return false;
    }
    const auto* loopVar =
        llvm::dyn_cast_or_null<clang::VarDecl>(initDecl->getSingleDecl());
    if (!loopVar || !loopVar->hasInit() ||
        !isIntegerLiteralValue(loopVar->getInit(), 1)) {
        return false;
    }
    // Cond must be `i < N - 1` (BinaryOperator with LT and RHS = last index).
    const auto* cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(
        ignoreParenImp(forStmt->getCond()));
    if (!cond || cond->getOpcode() != clang::BO_LT) {
        return false;
    }
    if (declRefTarget(cond->getLHS()) != loopVar) {
        return false;
    }
    if (!isLastIndexExpr(cond->getRHS())) {
        return false;
    }
    // Body must be a single assignment: `sourceTensor[i] = phaseBWriter[i-1]`.
    const clang::Stmt* body = forStmt->getBody();
    if (const auto* compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(body)) {
        if (compound->size() != 1) {
            return false;
        }
        body = *compound->body_begin();
    }
    const clang::Expr* bodyExpr =
        llvm::dyn_cast_or_null<clang::Expr>(body);
    if (!bodyExpr) {
        return false;
    }
    bodyExpr = skipImplicitWrappers(bodyExpr);
    const clang::Expr* lhs = nullptr;
    const clang::Expr* rhs = nullptr;
    if (const auto* binAssign = llvm::dyn_cast<clang::BinaryOperator>(bodyExpr)) {
        if (binAssign->getOpcode() != clang::BO_Assign) {
            return false;
        }
        lhs = binAssign->getLHS();
        rhs = binAssign->getRHS();
    } else if (const auto* opCall =
                   llvm::dyn_cast<clang::CXXOperatorCallExpr>(bodyExpr)) {
        if (!opCall->isAssignmentOp() || opCall->getNumArgs() < 2) {
            return false;
        }
        lhs = opCall->getArg(0);
        rhs = opCall->getArg(1);
    } else {
        return false;
    }
    const clang::Expr* indexL = nullptr;
    if (!isTensorIndexExpr(lhs, sourceTensor, &indexL)) {
        return false;
    }
    if (!isLoopVarPlusOrMinus(indexL, loopVar, 0)) {
        return false;
    }
    const clang::Expr* indexR = nullptr;
    if (!isTensorIndexExpr(rhs, phaseBWriter, &indexR)) {
        return false;
    }
    if (!isLoopVarPlusOrMinus(indexR, loopVar, -1)) {
        return false;
    }
    return true;
}

bool isBoundaryAssign(const clang::Stmt* stmt,
                      const clang::ValueDecl* sourceTensor,
                      const clang::ValueDecl* phaseAOutput,
                      bool wantLastIndex) {
    const clang::Expr* expr =
        skipImplicitWrappers(llvm::dyn_cast_or_null<clang::Expr>(stmt));
    if (!expr) {
        return false;
    }
    const clang::Expr* lhs = nullptr;
    const clang::Expr* rhs = nullptr;
    if (const auto* binAssign = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
        if (binAssign->getOpcode() != clang::BO_Assign) {
            return false;
        }
        lhs = binAssign->getLHS();
        rhs = binAssign->getRHS();
    } else if (const auto* opCall =
                   llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (!opCall->isAssignmentOp() || opCall->getNumArgs() < 2) {
            return false;
        }
        lhs = opCall->getArg(0);
        rhs = opCall->getArg(1);
    } else {
        return false;
    }
    const clang::Expr* indexL = nullptr;
    if (!isTensorIndexExpr(lhs, sourceTensor, &indexL)) {
        return false;
    }
    const clang::Expr* indexR = nullptr;
    if (!isTensorIndexExpr(rhs, phaseAOutput, &indexR)) {
        return false;
    }
    if (wantLastIndex) {
        return isLastIndexExpr(indexL) && isLastIndexExpr(indexR);
    }
    return isIntegerLiteralValue(indexL, 0) && isIntegerLiteralValue(indexR, 0);
}

bool stmtUsesAnyDeclAfterLoop(
    const clang::Stmt* stmt,
    const std::set<const clang::ValueDecl*>& targetDecls,
    const clang::Stmt* outerLoop,
    const clang::SourceManager& sourceManager) {
    if (!stmt || targetDecls.empty() || !outerLoop) {
        return false;
    }
    if (sourceRangeContains(sourceManager, outerLoop->getSourceRange(),
                            stmt->getSourceRange())) {
        return false;
    }
    if (stmt->getEndLoc().isValid() &&
        sourceManager.isBeforeInTranslationUnit(stmt->getEndLoc(),
                                                outerLoop->getEndLoc())) {
        return false;
    }
    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
        if (declRef->getDecl() &&
            targetDecls.count(declRef->getDecl()) != 0 &&
            declRef->getExprLoc().isValid() &&
            sourceManager.isBeforeInTranslationUnit(outerLoop->getEndLoc(),
                                                    declRef->getExprLoc())) {
            return true;
        }
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (stmtUsesAnyDeclAfterLoop(child, targetDecls, outerLoop,
                                     sourceManager)) {
            return true;
        }
    }
    return false;
}

bool exprReferencesAnyDecl(const clang::Stmt* stmt,
                           const std::set<const clang::ValueDecl*>& decls) {
    if (!stmt || decls.empty()) {
        return false;
    }
    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
        if (declRef->getDecl() && decls.count(declRef->getDecl()) != 0) {
            return true;
        }
    }
    for (const clang::Stmt* child : stmt->children()) {
        if (exprReferencesAnyDecl(child, decls)) {
            return true;
        }
    }
    return false;
}

void collectDeclStmtsRecursive(const clang::Stmt* stmt,
                               std::vector<const clang::DeclStmt*>& out) {
    if (!stmt) {
        return;
    }
    if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
        out.push_back(declStmt);
    }
    for (const clang::Stmt* child : stmt->children()) {
        collectDeclStmtsRecursive(child, out);
    }
}

// Collect VarDecls declared before `outerLoop` whose initializer transitively
// references the phase-A output. Narrowly covers patterns like
// `auto& alias = array_out_tensor;` or `auto* p = &array_out_tensor;` and
// chains through other pre-loop aliases.
std::set<const clang::ValueDecl*> collectPhaseAOutputAliases(
    DacppFile* dacppFile,
    const clang::ValueDecl* phaseAOutput,
    const clang::Stmt* outerLoop) {
    std::set<const clang::ValueDecl*> aliases;
    if (!phaseAOutput) {
        return aliases;
    }
    aliases.insert(phaseAOutput);
    if (!dacppFile || !outerLoop ||
        !dacppFile->getTranslationUnitDecl() || !dacppFile->getContext()) {
        return aliases;
    }
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    std::vector<const clang::DeclStmt*> declStmts;
    for (const clang::Decl* decl :
         dacppFile->getTranslationUnitDecl()->decls()) {
        const auto* functionDecl =
            llvm::dyn_cast_or_null<clang::FunctionDecl>(decl);
        if (!functionDecl || !functionDecl->hasBody()) {
            continue;
        }
        collectDeclStmtsRecursive(functionDecl->getBody(), declStmts);
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (const clang::DeclStmt* declStmt : declStmts) {
            if (!declStmt) {
                continue;
            }
            // Pre-loop decls only.
            if (declStmt->getBeginLoc().isInvalid() ||
                !sourceManager.isBeforeInTranslationUnit(
                    declStmt->getBeginLoc(), outerLoop->getBeginLoc())) {
                continue;
            }
            for (const clang::Decl* d : declStmt->decls()) {
                const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(d);
                if (!varDecl || !varDecl->hasInit()) {
                    continue;
                }
                if (aliases.count(varDecl) != 0) {
                    continue;
                }
                if (exprReferencesAnyDecl(varDecl->getInit(), aliases)) {
                    aliases.insert(varDecl);
                    changed = true;
                }
            }
        }
    }
    return aliases;
}

bool phaseAOutputUsedAfterLoop(DacppFile* dacppFile,
                               const clang::ValueDecl* phaseAOutput,
                               const clang::Stmt* outerLoop) {
    if (!dacppFile || !phaseAOutput || !outerLoop ||
        !dacppFile->getTranslationUnitDecl() || !dacppFile->getContext()) {
        return true;
    }
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    const std::set<const clang::ValueDecl*> aliases =
        collectPhaseAOutputAliases(dacppFile, phaseAOutput, outerLoop);
    for (const clang::Decl* decl : dacppFile->getTranslationUnitDecl()->decls()) {
        const auto* functionDecl = llvm::dyn_cast_or_null<clang::FunctionDecl>(decl);
        if (!functionDecl || !functionDecl->hasBody()) {
            continue;
        }
        if (stmtUsesAnyDeclAfterLoop(functionDecl->getBody(), aliases,
                                     outerLoop, sourceManager)) {
            return true;
        }
    }
    return false;
}

// Walks the tensor VarDecl's initializer chain to the underlying vector and
// returns that vector's size if it is a compile-time constant. Returns -1
// otherwise. Covers the canonical `std::vector<T> v(N)` / `std::vector<T>{...}`
// patterns used by the accepted phase-exchange source tensor.
int64_t staticTensorTotalSize(const clang::ValueDecl* sourceDecl,
                              clang::ASTContext* context) {
    if (!sourceDecl || !context) {
        return -1;
    }
    const auto* tensorVar = llvm::dyn_cast<clang::VarDecl>(sourceDecl);
    if (!tensorVar || !tensorVar->hasInit()) {
        return -1;
    }
    // Locate the std::vector VarDecl referenced inside the tensor's init.
    const clang::VarDecl* vectorVar = nullptr;
    std::vector<const clang::Stmt*> worklist;
    worklist.push_back(tensorVar->getInit());
    while (!worklist.empty() && !vectorVar) {
        const clang::Stmt* cur = worklist.back();
        worklist.pop_back();
        if (!cur) {
            continue;
        }
        if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(cur)) {
            if (const auto* vd = llvm::dyn_cast_or_null<clang::VarDecl>(
                    declRef->getDecl())) {
                const std::string typeName = vd->getType().getAsString();
                if (typeName.find("vector") != std::string::npos ||
                    typeName.find("Vector") != std::string::npos) {
                    vectorVar = vd;
                    break;
                }
            }
        }
        for (const clang::Stmt* child : cur->children()) {
            worklist.push_back(child);
        }
    }
    if (!vectorVar || !vectorVar->hasInit()) {
        return -1;
    }
    // Try `std::vector<T> v(N)` — first integer argument of the ctor.
    const clang::Expr* vecInit = vectorVar->getInit();
    if (!vecInit) {
        return -1;
    }
    const clang::Expr* unwrapped = vecInit->IgnoreImplicit();
    if (const auto* cce =
            llvm::dyn_cast_or_null<clang::CXXConstructExpr>(unwrapped)) {
        // Only the first positional argument of a std::vector ctor carries the
        // size. Looking past it would let `std::vector<T>(runtimeN, 2)` fold
        // the fill value `2` into a bogus size proof.
        if (cce->getNumArgs() >= 1) {
            const clang::Expr* arg = cce->getArg(0);
            if (arg && arg->getType()->isIntegerType()) {
                clang::Expr::EvalResult evalResult;
                if (arg->EvaluateAsInt(evalResult, *context) &&
                    evalResult.Val.isInt()) {
                    const int64_t value =
                        evalResult.Val.getInt().getSExtValue();
                    if (value > 0) {
                        return value;
                    }
                }
            }
        }
    }
    // Try `std::vector<T>{a, b, c, ...}` — count of integer initializers.
    if (const auto* initList =
            llvm::dyn_cast_or_null<clang::InitListExpr>(unwrapped)) {
        int64_t count = 0;
        for (unsigned idx = 0; idx < initList->getNumInits(); ++idx) {
            const clang::Expr* init = initList->getInit(idx);
            if (init && init->getType()->isIntegerType()) {
                ++count;
            }
        }
        if (count > 0) {
            return count;
        }
    }
    return -1;
}

PhaseExchangeDetection detectPhaseExchange(
    DacppFile* dacppFile,
    const std::vector<ShellPartitionPlan>& plans,
    std::size_t planAIdx) {
    PhaseExchangeDetection result;
    result.planAIndex = planAIdx;
    if (planAIdx >= plans.size() || !dacppFile || !dacppFile->getContext()) {
        result.rejectReason = "phase-exchange context unavailable";
        return result;
    }
    const ShellPartitionPlan& planA = plans[planAIdx];
    if (!isFixedBlockPlanWithBlockSize(planA, 2, 2)) {
        result.rejectReason = "phase-exchange A is not FixedBlock(2,2)";
        return result;
    }
    const clang::Stmt* outerLoop =
        stableOuterLoopForExpr(dacppFile, planA);
    if (!outerLoop ||
        (!llvm::isa<clang::ForStmt>(outerLoop) &&
         !llvm::isa<clang::WhileStmt>(outerLoop))) {
        result.rejectReason = "phase-exchange A not in stable for/while loop";
        return result;
    }
    if (!shellArgsDeclaredBeforeLoop(dacppFile, outerLoop, planA)) {
        result.rejectReason =
            "phase-exchange A shell args must be declared before the loop";
        return result;
    }
    const ParamAccessPlan* readerA = fixedBlockReaderParam(planA);
    const ParamAccessPlan* writerA = fixedBlockWriterParam(planA);
    if (!readerA || !writerA) {
        result.rejectReason = "phase-exchange A missing reader or writer";
        return result;
    }
    const clang::CallExpr* shellCallA =
        getShellCallExpr(planA.exprNode.dacExpr);
    if (!shellCallA) {
        result.rejectReason = "phase-exchange A shell call unavailable";
        return result;
    }
    const clang::ValueDecl* readerDecl =
        declRefTarget(shellCallA->getArg(readerA->paramIndex));
    const clang::ValueDecl* writerDecl =
        declRefTarget(shellCallA->getArg(writerA->paramIndex));
    if (!readerDecl || !writerDecl || readerDecl == writerDecl) {
        result.rejectReason =
            "phase-exchange A reader/writer must be distinct decl-ref tensors";
        return result;
    }

    // Find planB: next FixedBlock(2,2) plan inside the same outer loop.
    const ShellPartitionPlan* planBPtr = nullptr;
    std::size_t planBIdx = planAIdx;
    const auto& sourceManager = dacppFile->getContext()->getSourceManager();
    for (std::size_t idx = planAIdx + 1; idx < plans.size(); ++idx) {
        const ShellPartitionPlan& candidate = plans[idx];
        if (!candidate.exprNode.dacExpr) {
            continue;
        }
        if (!isFixedBlockPlanWithBlockSize(candidate, 2, 2)) {
            continue;
        }
        if (!sourceRangeContains(
                sourceManager, outerLoop->getSourceRange(),
                candidate.exprNode.dacExpr->getSourceRange())) {
            continue;
        }
        planBPtr = &candidate;
        planBIdx = idx;
        break;
    }
    if (!planBPtr) {
        result.rejectReason =
            "phase-exchange B not found inside outer loop";
        return result;
    }
    const ShellPartitionPlan& planB = *planBPtr;
    const ParamAccessPlan* readerB = fixedBlockReaderParam(planB);
    const ParamAccessPlan* writerB = fixedBlockWriterParam(planB);
    if (!readerB || !writerB) {
        result.rejectReason = "phase-exchange B missing reader or writer";
        return result;
    }
    const clang::CallExpr* shellCallB =
        getShellCallExpr(planB.exprNode.dacExpr);
    if (!shellCallB) {
        result.rejectReason = "phase-exchange B shell call unavailable";
        return result;
    }
    if (!planA.exprNode.calc || !planB.exprNode.calc ||
        !planA.exprNode.shell || !planB.exprNode.shell ||
        planA.exprNode.calc->getName() != planB.exprNode.calc->getName() ||
        planA.exprNode.shell->getName() != planB.exprNode.shell->getName()) {
        result.rejectReason =
            "phase-exchange A and B must share the same shell and calc";
        return result;
    }
    if (planA.signature.layout != planB.signature.layout) {
        result.rejectReason = "phase-exchange A/B layouts differ";
        return result;
    }

    // Ensure no other DAC expression sits between A and B inside the loop body.
    int dacExprCountInLoop = 0;
    for (const auto* candidate : dacppFile->dacExprs) {
        if (candidate &&
            sourceRangeContains(sourceManager, outerLoop->getSourceRange(),
                                candidate->getSourceRange())) {
            ++dacExprCountInLoop;
        }
    }
    if (dacExprCountInLoop != 2) {
        result.rejectReason =
            "phase-exchange requires exactly two DAC expressions in the loop body";
        return result;
    }

    // Now walk the loop body to verify the exact statement order.
    const clang::CompoundStmt* compound = nullptr;
    if (const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(outerLoop)) {
        compound =
            llvm::dyn_cast_or_null<clang::CompoundStmt>(forStmt->getBody());
    } else if (const auto* whileStmt =
                   llvm::dyn_cast<clang::WhileStmt>(outerLoop)) {
        compound =
            llvm::dyn_cast_or_null<clang::CompoundStmt>(whileStmt->getBody());
    }
    if (!compound) {
        result.rejectReason = "phase-exchange loop body not a compound stmt";
        return result;
    }

    enum class WalkPhase {
        ExpectDacA,
        AfterA_BeforeB,
        ExpectDacB,
        AfterB,
        Done
    };
    WalkPhase phase = WalkPhase::ExpectDacA;
    const clang::ValueDecl* sliceVarDecl = nullptr;
    const clang::ValueDecl* phaseBWriterDecl = nullptr;
    bool sawSliceDecl = false;
    bool sawIntermediateContainerDecl = false;
    bool sawPhaseBWriterDecl = false;
    bool sawInteriorCopyLoop = false;
    bool sawBoundaryFirst = false;
    bool sawBoundaryLast = false;

    for (const clang::Stmt* child : compound->body()) {
        if (!child) {
            result.rejectReason = "phase-exchange loop body has null stmt";
            return result;
        }
        switch (phase) {
        case WalkPhase::ExpectDacA: {
            if (sourceRangeContains(sourceManager, child->getSourceRange(),
                                    planA.exprNode.dacExpr->getSourceRange())) {
                result.followerStmts.push_back(nullptr);
                phase = WalkPhase::AfterA_BeforeB;
                continue;
            }
            result.rejectReason =
                "phase-exchange first stmt must be DAC expression A";
            return result;
        }
        case WalkPhase::AfterA_BeforeB: {
            if (sourceRangeContains(sourceManager, child->getSourceRange(),
                                    planB.exprNode.dacExpr->getSourceRange())) {
                if (!sawSliceDecl || !sawIntermediateContainerDecl ||
                    !sawPhaseBWriterDecl) {
                    result.rejectReason =
                        "phase-exchange follower decls before B are incomplete";
                    return result;
                }
                result.followerStmts.push_back(child);
                phase = WalkPhase::AfterB;
                continue;
            }
            int sliceOffset = -1;
            const clang::ValueDecl* candidateSlice = nullptr;
            if (!sawSliceDecl &&
                tryRecognizePhaseSliceDecl(child, writerDecl,
                                           &candidateSlice, &sliceOffset)) {
                if (sliceOffset != 1) {
                    result.rejectReason =
                        "phase-exchange slice offset must be 1";
                    return result;
                }
                sliceVarDecl = candidateSlice;
                sawSliceDecl = true;
                result.followerStmts.push_back(child);
                continue;
            }
            const clang::ValueDecl* otherDecl = nullptr;
            if (sawSliceDecl && !sawIntermediateContainerDecl &&
                isVectorOrTensorDeclStmt(child, &otherDecl)) {
                sawIntermediateContainerDecl = true;
                result.followerStmts.push_back(child);
                continue;
            }
            if (sawIntermediateContainerDecl && !sawPhaseBWriterDecl &&
                isVectorOrTensorDeclStmt(child, &otherDecl)) {
                phaseBWriterDecl = otherDecl;
                sawPhaseBWriterDecl = true;
                result.followerStmts.push_back(child);
                continue;
            }
            result.rejectReason =
                "phase-exchange unexpected stmt between DAC A and DAC B";
            return result;
        }
        case WalkPhase::AfterB: {
            if (!sawInteriorCopyLoop) {
                if (isInteriorCopyForLoop(child, readerDecl, phaseBWriterDecl)) {
                    sawInteriorCopyLoop = true;
                    result.followerStmts.push_back(child);
                    continue;
                }
                result.rejectReason =
                    "phase-exchange missing interior copy for-loop";
                return result;
            }
            if (!sawBoundaryFirst) {
                if (isBoundaryAssign(child, readerDecl, writerDecl, false)) {
                    sawBoundaryFirst = true;
                    result.followerStmts.push_back(child);
                    continue;
                }
                result.rejectReason =
                    "phase-exchange missing boundary[0] copy";
                return result;
            }
            if (!sawBoundaryLast) {
                if (isBoundaryAssign(child, readerDecl, writerDecl, true)) {
                    sawBoundaryLast = true;
                    result.followerStmts.push_back(child);
                    phase = WalkPhase::Done;
                    continue;
                }
                result.rejectReason =
                    "phase-exchange missing boundary[N-1] copy";
                return result;
            }
            result.rejectReason =
                "phase-exchange unexpected trailing stmt after boundary";
            return result;
        }
        case WalkPhase::Done:
            {
                std::set<const clang::ValueDecl*> removedFollowerDecls;
                if (sliceVarDecl) {
                    removedFollowerDecls.insert(sliceVarDecl);
                }
                if (phaseBWriterDecl) {
                    removedFollowerDecls.insert(phaseBWriterDecl);
                }
                if (exprReferencesAnyDecl(child, removedFollowerDecls)) {
                    result.rejectReason =
                        "phase-exchange removed follower stmt is referenced after canonical rewrite region";
                    return result;
                }
                const std::set<std::string> sourceTensors{
                    readerDecl->getNameAsString()};
                if (stmtWritesAnyTensor(child, sourceTensors)) {
                    result.rejectReason =
                        "phase-exchange unexpected write to resident source tensor";
                    return result;
                }
                result.rejectReason =
                    "phase-exchange unexpected trailing stmt";
            }
            return result;
        case WalkPhase::ExpectDacB:
            return result;
        }
    }
    if (phase != WalkPhase::Done || !sawInteriorCopyLoop ||
        !sawBoundaryFirst || !sawBoundaryLast) {
        result.rejectReason =
            "phase-exchange loop body did not match expected structure";
        return result;
    }
    if (!sliceVarDecl || !phaseBWriterDecl) {
        result.rejectReason =
            "phase-exchange B helper decls were not recognized";
        return result;
    }
    if (readerB->paramIndex < 0 ||
        readerB->paramIndex >= static_cast<int>(shellCallB->getNumArgs()) ||
        writerB->paramIndex < 0 ||
        writerB->paramIndex >= static_cast<int>(shellCallB->getNumArgs())) {
        result.rejectReason = "phase-exchange B shell args unavailable";
        return result;
    }
    if (declRefTarget(shellCallB->getArg(readerB->paramIndex)) !=
            sliceVarDecl ||
        declRefTarget(shellCallB->getArg(writerB->paramIndex)) !=
            phaseBWriterDecl) {
        result.rejectReason =
            "phase-exchange B shell args must use the recognized helper decls";
        return result;
    }
    if (phaseAOutputUsedAfterLoop(dacppFile, writerDecl, outerLoop)) {
        result.rejectReason =
            "phase-exchange phase-A output is used after the outer loop";
        return result;
    }
    // Require the writer (phase-A output) to be a statically proven even-length
    // vector. The writer is the locally declared phase-A output tensor whose
    // underlying vector size is the canonical even N. The reader's total may
    // still be dynamic (e.g. a function-parameter vector reference), so the
    // generated init also emits a runtime guard that the scattered total
    // equals this proven value — see LoopLoweredFixedBlockPhaseExchangeCodegen.
    const int64_t writerTotal =
        staticTensorTotalSize(writerDecl, dacppFile->getContext());
    auto provenEven = [](int64_t n) { return n > 0 && (n % 2) == 0; };
    if (!provenEven(writerTotal)) {
        result.rejectReason =
            "phase-exchange requires a statically proven even total";
        return result;
    }

    result.valid = true;
    result.planAIndex = planAIdx;
    result.planBIndex = planBIdx;
    result.outerLoop = outerLoop;
    result.followerDacExpr = planB.exprNode.dacExpr;
    result.sourceTensorName = readerDecl->getNameAsString();
    result.phaseAOutputTensorName = writerDecl->getNameAsString();
    result.provenEvenTotal = writerTotal;
    if (planA.exprNode.calc &&
        readerA->paramIndex < planA.exprNode.calc->getNumParams()) {
        result.elementType =
            planA.exprNode.calc->getParam(readerA->paramIndex)->getBasicType();
    }
    return result;
}

void detectAndAnnotateFixedBlockPhaseExchange(
    DacppFile* dacppFile,
    std::vector<ShellPartitionPlan>& plans) {
    if (!dacppFile) {
        return;
    }
    for (std::size_t idx = 0; idx < plans.size(); ++idx) {
        ShellPartitionPlan& planA = plans[idx];
        if (planA.orLoopLower.kind != OrLoopLowerKind::None) {
            continue;
        }
        if (!isFixedBlockPlanWithBlockSize(planA, 2, 2)) {
            continue;
        }
        PhaseExchangeDetection detection =
            detectPhaseExchange(dacppFile, plans, idx);
        const std::string shellName =
            planA.exprNode.shell ? planA.exprNode.shell->getName() : "<null>";
        if (!detection.valid) {
            llvm::outs()
                << "[DACPP][MPI][OR][P5][PhaseExchange] expr="
                << planA.exprIndex << " shell=" << shellName
                << " phase-exchange=rejected reason="
                << detection.rejectReason << "\n";
            continue;
        }
        ShellPartitionPlan& planB = plans[detection.planBIndex];
        // Mark plan A
        planA.orLoopLower.kind = OrLoopLowerKind::FixedBlockPhaseExchange;
        planA.orLoopLower.outerLoop = detection.outerLoop;
        planA.orLoopLower.finalMaterializeRequired = true;
        planA.orLoopLower.runMaterializeEveryStep = false;
        planA.orLoopLower.fixedBlockPhaseExchange.enabled = true;
        planA.orLoopLower.fixedBlockPhaseExchange.blockSize =
            detection.blockSize;
        planA.orLoopLower.fixedBlockPhaseExchange.blockStride =
            detection.blockStride;
        planA.orLoopLower.fixedBlockPhaseExchange.phaseShiftOffset =
            detection.phaseShiftOffset;
        planA.orLoopLower.fixedBlockPhaseExchange.provenEvenTotal =
            detection.provenEvenTotal;
        planA.orLoopLower.fixedBlockPhaseExchange.sourceTensorName =
            detection.sourceTensorName;
        planA.orLoopLower.fixedBlockPhaseExchange.phaseAOutputTensorName =
            detection.phaseAOutputTensorName;
        planA.orLoopLower.fixedBlockPhaseExchange.elementType =
            detection.elementType;
        planA.orLoopLower.fixedBlockPhaseExchange.followerStmtsToRemove =
            detection.followerStmts;
        planA.orLoopLower.fixedBlockPhaseExchange.followerDacExpr =
            detection.followerDacExpr;
        populateFixedBlockPhaseExchangeContract(
            planA.orLoopLower, detection, planA.exprNode.dacExpr);
        annotateLoopLoweringContractConsistency(planA);

        // Mark plan B
        planB.orLoopLower.kind =
            OrLoopLowerKind::FixedBlockPhaseExchangeFollower;
        planB.orLoopLower.outerLoop = detection.outerLoop;
        planB.orLoopLower.fixedBlockPhaseExchange.enabled = true;
        planB.orLoopLower.contract.enabled = true;
        planB.orLoopLower.contract.loweringName =
            "FixedBlockPhaseExchangeFollower";
        planB.orLoopLower.contract.acceptedReason =
            "phase-exchange follower removed by leader contract";

        llvm::outs()
            << "[DACPP][MPI][OR][P5][PhaseExchange] expr="
            << planA.exprIndex << " shell=" << shellName
            << " phase-exchange=accepted block-size=" << detection.blockSize
            << " phase-shift=" << detection.phaseShiftOffset
            << " contract=FixedBlockPhaseExchange"
            << " materialize="
            << loweringContractMaterializeTimingName(
                   LoweringContractMaterializeTiming::LoopExit)
            << " guard-runtime=total-mismatch"
            << " contract-check="
            << (planA.orLoopLower.contractConsistencyCheckPassed ? "pass"
                                                                 : "fail")
            << " reason="
            << planA.orLoopLower.contractConsistencyCheckReason
            << " follower-expr=" << planB.exprIndex
            << " source=" << detection.sourceTensorName
            << " phase-a-output=" << detection.phaseAOutputTensorName
            << "\n";
    }
}

void annotateLoopLowerCandidates(DacppFile* dacppFile,
                                 std::vector<ShellPartitionPlan>& plans) {
    for (auto& plan : plans) {
        if (!plan.supported) {
            continue;
        }
        const clang::Stmt* outerLoop = nullptr;
        const std::string reason =
            loopLowerRejectReason(dacppFile, plan, &outerLoop);
        plan.loopLowerOuterLoop = outerLoop;
        plan.loopLowerRejectReason = reason;
        plan.loopLowerCandidate = reason.empty();
        plan.loopLowerMaterializeEveryRun =
            plan.loopLowerCandidate && hasReplicatedScalarParam(plan);
        plan.loopLowerReplicatedScalarLocalRefresh = false;
        plan.loopLowerScalarRefreshReason.clear();
        plan.loopLowerSelectiveMaterialize =
            LoopLoweredSelectiveMaterializePlan{};
        plan.loopLowerDeviceTimeLoop = LoopLoweredDeviceTimeLoopPlan{};
        if (plan.loopLowerMaterializeEveryRun) {
            plan.loopLowerReplicatedScalarLocalRefresh =
                loopBodyHasOnlyUnguardedTopLevelScalarWrites(
                    dacppFile, outerLoop, plan,
                    plan.loopLowerScalarRefreshReason);
            plan.loopLowerSelectiveMaterialize =
                analyzeLoopLowerSelectiveMaterialize(dacppFile, plan,
                                                     outerLoop);
            plan.loopLowerDeviceTimeLoop =
                analyzeLoopLowerDeviceTimeLoop(dacppFile, plan, outerLoop);
        }

        const std::string shellName =
            plan.exprNode.shell ? plan.exprNode.shell->getName() : "<null>";
        llvm::outs() << "[DACPP][MPI][OR][P4.5] expr=" << plan.exprIndex
                     << " shell=" << shellName
                     << " layout=" << localLayoutKindName(plan.signature.layout)
                     << " loop-lower="
                     << (plan.loopLowerCandidate ? "candidate" : "rejected");
        if (outerLoop) {
            llvm::outs() << " loop=" << loopKindName(outerLoop);
        }
        if (plan.loopLowerCandidate) {
            llvm::outs() << " structure=init/run/materialize";
            if (plan.loopLowerMaterializeEveryRun) {
                if (plan.loopLowerSelectiveMaterialize.enabled) {
                    llvm::outs()
                        << " materialize=selective-row host="
                        << plan.loopLowerSelectiveMaterialize.hostTensorName
                        << " row="
                        << plan.loopLowerSelectiveMaterialize.targetRow
                        << " output="
                        << plan.loopLowerSelectiveMaterialize.outputTensorName;
                } else {
                    llvm::outs() << " materialize=per-run";
                    if (!plan.loopLowerSelectiveMaterialize.reason.empty()) {
                        llvm::outs() << " selective-row=fallback reason="
                                     << plan.loopLowerSelectiveMaterialize.reason;
                    }
                }
                if (plan.loopLowerReplicatedScalarLocalRefresh) {
                    llvm::outs() << " scalar-refresh=local-replicated";
                } else if (!plan.loopLowerScalarRefreshReason.empty()) {
                    llvm::outs() << " scalar-refresh=bcast reason="
                                 << plan.loopLowerScalarRefreshReason;
                }
                if (plan.loopLowerDeviceTimeLoop.enabled) {
                    llvm::outs()
                        << " device-time-loop=accepted reason="
                        << plan.loopLowerDeviceTimeLoop.reason;
                } else if (!plan.loopLowerDeviceTimeLoop.reason.empty()) {
                    llvm::outs()
                        << " device-time-loop=rejected reason="
                        << plan.loopLowerDeviceTimeLoop.reason;
                }
            }
        } else if (!reason.empty()) {
            llvm::outs() << " reason=" << reason;
        }
        llvm::outs() << "\n";

        const clang::Stmt* p46OuterLoop = nullptr;
        const std::string p46Reason =
            stencilFullSyncLoopRejectReason(dacppFile, plan, &p46OuterLoop);
        const bool p46Candidate = p46Reason.empty();
        plan.orLoopLower = OrLoopLowerPlan{};
        OrLoopLowerPlan::StencilResidentHaloMetadata haloMetadata;
        const std::string haloReason =
            p46Candidate
                ? stencilResidentHaloRejectReason(dacppFile, plan,
                                                  p46OuterLoop, haloMetadata)
                : "";
        const bool haloCandidate = p46Candidate && haloReason.empty();
        plan.orLoopLower.kind =
            haloCandidate ? OrLoopLowerKind::StencilResidentHalo
                          : (p46Candidate ? OrLoopLowerKind::StencilFullSync
                                          : OrLoopLowerKind::None);
        plan.orLoopLower.outerLoop = p46OuterLoop;
        plan.orLoopLower.rejectReason = p46Reason;
        plan.orLoopLower.hoistReaderSync =
            p46Candidate && !haloCandidate &&
            canHoistStencilReaderSync(dacppFile, p46OuterLoop, plan);
        plan.orLoopLower.runMaterializeEveryStep =
            p46Candidate && !haloCandidate;
        plan.orLoopLower.finalMaterializeRequired = p46Candidate;
        plan.orLoopLower.stencilResidentHalo = haloMetadata;
        if (!haloCandidate) {
            plan.orLoopLower.stencilResidentHalo.rejectReason = haloReason;
            plan.orLoopLower.stencilResidentHalo.temporalBlockRejectReason =
                haloReason.empty() ? "resident halo not accepted" : haloReason;
        } else {
            annotateResidentHaloTemporalBlocking(dacppFile, plan,
                                                 p46OuterLoop, haloMetadata);
            annotateResidentHaloSpatial2D(plan, haloCandidate, haloReason);
        }
        if (p46Candidate) {
            populateStencilLoopLoweringContract(dacppFile, plan, haloReason);
            annotateStencilContractRemovalSetEquivalence(dacppFile, plan);
            annotateLoopLoweringContractConsistency(plan);
        }

        llvm::outs() << "[DACPP][MPI][OR][P4.6][Loop] expr="
                     << plan.exprIndex << " shell=" << shellName
                     << " layout=" << localLayoutKindName(plan.signature.layout)
                     << " loop-lower="
                     << (p46Reason.empty() ? "candidate" : "rejected");
        if (p46OuterLoop) {
            llvm::outs() << " loop=" << loopKindName(p46OuterLoop);
        }
        if (p46Reason.empty()) {
            llvm::outs()
                << " kind=" << orLoopLowerKindName(plan.orLoopLower.kind)
                << " structure=ctx/init/run/materialize";
            if (haloCandidate) {
                llvm::outs() << " resident-halo=true";
                if (plan.signature.layout == LocalLayoutKind::StencilWindow2D) {
                    llvm::outs()
                        << " window-shape="
                        << plan.orLoopLower.stencilResidentHalo.windowRows
                        << "x"
                        << plan.orLoopLower.stencilResidentHalo.windowCols
                        << " followup-offset=("
                        << plan.orLoopLower.stencilResidentHalo
                               .followupTargetRowOffset
                        << ","
                        << plan.orLoopLower.stencilResidentHalo
                               .followupTargetColOffset
                        << ")";
                    if (plan.orLoopLower.stencilResidentHalo.hasDirectReader) {
                        llvm::outs()
                            << " direct-reader=true read-cache-offset=("
                            << plan.orLoopLower.stencilResidentHalo
                                   .readCacheTargetRowOffset
                            << ","
                            << plan.orLoopLower.stencilResidentHalo
                                   .readCacheTargetColOffset
                            << ") role-rotation=buffer-swap";
                    }
                    if (plan.orLoopLower.stencilResidentHalo
                            .spatial2DEnabled) {
                        llvm::outs()
                            << " distribution=spatial-2d accepted"
                            << " stencil halo-width="
                            << plan.orLoopLower.stencilResidentHalo
                                   .spatial2DHaloWidth
                            << " reason="
                            << plan.orLoopLower.stencilResidentHalo
                                   .spatial2DAcceptReason;
                    } else if (!plan.orLoopLower.stencilResidentHalo
                                    .spatial2DRejectReason.empty()) {
                        llvm::outs()
                            << " distribution=spatial-2d rejected reason="
                            << plan.orLoopLower.stencilResidentHalo
                                   .spatial2DRejectReason;
                    }
                } else {
                    llvm::outs()
                        << " window-size="
                        << plan.orLoopLower.stencilResidentHalo.windowSize
                        << " followup-offset="
                        << plan.orLoopLower.stencilResidentHalo
                               .followupTargetOffset;
                }
                const ParamAccessPlan* readerParam =
                    stencilWindowReaderParam(plan);
                const bool finalBoundedSmall =
                    plan.signature.layout == LocalLayoutKind::StencilWindow1D &&
                    readerParam &&
                    readerParam->postUseSync.kind ==
                        PostUseSyncKind::BoundedIndexedRootRead &&
                    !readerParam->postUseSync.boundedIndices.empty();
                llvm::outs() << " materialize=final";
                if (finalBoundedSmall) {
                    llvm::outs()
                        << " final-materialize=bounded/small";
                }
                if (plan.orLoopLower.stencilResidentHalo.temporalBlockSize >
                    1) {
                    llvm::outs()
                        << " temporal-block="
                        << plan.orLoopLower.stencilResidentHalo
                               .temporalBlockSize
                        << " accepted";
                    if (plan.orLoopLower.stencilResidentHalo.hasDirectReader) {
                        llvm::outs() << " direct-reader recurrence";
                    } else if (!plan.orLoopLower.stencilResidentHalo
                                    .temporalBlockAcceptReason.empty() &&
                               plan.orLoopLower.stencilResidentHalo
                                       .temporalBlockAcceptReason !=
                                   "canonical") {
                        llvm::outs() << " "
                                     << plan.orLoopLower.stencilResidentHalo
                                            .temporalBlockAcceptReason;
                    }
                } else if (!plan.orLoopLower.stencilResidentHalo
                                .temporalBlockRejectReason.empty()) {
                    llvm::outs()
                        << " temporal-block=2 rejected reason="
                        << plan.orLoopLower.stencilResidentHalo
                               .temporalBlockRejectReason;
                }
            } else {
                llvm::outs()
                    << " hoist-reader-sync="
                    << (plan.orLoopLower.hoistReaderSync ? "true" : "false")
                    << " materialize=per-run";
                if (!haloReason.empty()) {
                    llvm::outs() << " resident-halo-reject=" << haloReason;
                }
            }
            logLoopLoweringContractSummary(plan.orLoopLower.contract);
            logContractConsistencyCheck(plan.orLoopLower);
            logContractRemovalSetEquivalence(plan.orLoopLower);
        } else {
            llvm::outs() << " reason=" << p46Reason;
        }
        llvm::outs() << "\n";
    }
}

void finalizeChain(OperatorResidentChainPlan& chain,
                   const std::vector<ShellPartitionPlan>* allPlans = nullptr) {
    if (!chain.supported) {
        return;
    }
    analyzeResidency(chain);
    if (chain.exprPlans.size() == 1) {
        ShellPartitionPlan& plan = chain.exprPlans.front();
        std::string acceptReason;
        const std::string rejectReason =
            contiguous1DDistributionRejectReason(plan, &acceptReason);
        if (rejectReason.empty()) {
            plan.signature.contiguous1DDistribution.kind =
                Contiguous1DDistributionKind::BlockCyclic;
            plan.signature.contiguous1DDistribution.blockSize = 64;
            plan.signature.contiguous1DDistribution.reason = acceptReason;
            chain.signature = plan.signature;
            llvm::outs()
                << "[DACPP][MPI][OR] distribution=block-cyclic accepted "
                << acceptReason << " block-size=64\n";
        } else if (plan.signature.layout == LocalLayoutKind::Contiguous1D) {
            llvm::outs()
                << "[DACPP][MPI][OR] distribution=block-cyclic rejected reason="
                << rejectReason << "\n";
        }
    }
    annotateRowBlock2DPointwiseFusion(chain, allPlans);
    // Note: chain accepted does not mean OR codegen is enabled for this layout
    // Check supportedPhaseLayout() to see which layouts actually generate OR code
    std::ostringstream log;
    log << "[DACPP][MPI][OR] chain=" << chain.chainId
        << " layout=" << localLayoutKindName(chain.signature.layout)
        << " length=" << chain.exprPlans.size()
        << " chain=accepted codegen="
        << (supportedPhaseLayout(chain.signature.layout) ? "enabled"
                                                         : "disabled")
        << "\n";
    llvm::outs().flush();
    llvm::outs() << log.str();
    llvm::outs().flush();
}

} // namespace

OperatorResidentChainPlan buildSingleOperatorResidentChain(
    const ShellPartitionPlan& shellPlan,
    int chainId) {
    OperatorResidentChainPlan chain;
    chain.chainId = chainId;
    if (!shellPlan.supported) {
        chain.supported = false;
        chain.rejectReason = shellPlan.rejectReason;
        return chain;
    }
    chain.supported = true;
    chain.signature = shellPlan.signature;
    chain.exprs.push_back(shellPlan.exprNode);
    chain.exprPlans.push_back(shellPlan);
    finalizeChain(chain);
    return chain;
}

std::vector<OperatorResidentChainPlan> buildOperatorResidentChains(
    DacppFile* dacppFile,
    const std::vector<DacExprNode>& exprNodes,
    const std::vector<ShellPartitionPlan>& shellPlans) {
    std::vector<ShellPartitionPlan> plans = shellPlans;
    for (auto& plan : plans) {
        fillActualTensorNames(plan, dacppFile);
        annotateConstantInputInit(plan, dacppFile);
        annotateOutputDirectInit(plan, dacppFile);
        annotateOutputSync(plan, dacppFile);
    }
    annotateLoopLowerCandidates(dacppFile, plans);
    detectAndAnnotateFixedBlockPhaseExchange(dacppFile, plans);

    std::vector<OperatorResidentChainPlan> chains;
    OperatorResidentChainPlan current;
    current.chainId = -1;

    auto closeCurrent = [&]() {
        if (current.supported && !current.exprPlans.empty()) {
            finalizeChain(current, &plans);
            chains.push_back(current);
        }
        current = OperatorResidentChainPlan{};
        current.chainId = -1;
    };

    for (std::size_t idx = 0; idx < plans.size(); ++idx) {
        const ShellPartitionPlan& plan = plans[idx];
        (void)exprNodes;
        if (!plan.supported || !supportedPhaseLayout(plan.signature.layout)) {
            closeCurrent();
            if (plan.supported && !supportedPhaseLayout(plan.signature.layout)) {
                logCodegenDisabledFallback(plan);
            }
            continue;
        }

        if (!current.supported || current.exprPlans.empty()) {
            current = OperatorResidentChainPlan{};
            current.supported = true;
            current.chainId = static_cast<int>(chains.size());
            current.signature = plan.signature;
            current.exprs.push_back(plan.exprNode);
            current.exprPlans.push_back(plan);
            continue;
        }

        if (canAppendToChain(current, plan)) {
            current.exprs.push_back(plan.exprNode);
            current.exprPlans.push_back(plan);
            continue;
        }

        closeCurrent();
        current.supported = true;
        current.chainId = static_cast<int>(chains.size());
        current.signature = plan.signature;
        current.exprs.push_back(plan.exprNode);
        current.exprPlans.push_back(plan);
    }

    closeCurrent();

    return chains;
}

std::string joinShellCallArgs(const clang::BinaryOperator* dacExpr,
                              DacppFile* dacppFile) {
    if (!dacExpr || !dacppFile) {
        return "";
    }
    const clang::CallExpr* shellCall = getShellCallExpr(dacExpr);
    if (!shellCall) {
        return "";
    }
    std::string args;
    for (unsigned argIdx = 0; argIdx < shellCall->getNumArgs(); ++argIdx) {
        if (!args.empty()) {
            args += ", ";
        }
        args += exprSource(shellCall->getArg(argIdx), dacppFile->getContext());
    }
    return args;
}

std::string buildWrapperCallForDacExpr(const std::string& wrapperName,
                                       const clang::BinaryOperator* dacExpr,
                                       DacppFile* dacppFile) {
    std::string call = wrapperName + "(";
    const std::string argText = joinShellCallArgs(dacExpr, dacppFile);
    if (!argText.empty()) {
        call += argText;
    }
    call += ")";
    return call;
}

std::string fusedRowBlock2DWrapperName(const OperatorResidentChainPlan& chain) {
    if (!chain.fusePointwiseRowBlock2D || chain.exprPlans.empty()) {
        return "";
    }
    Shell* shell = chain.exprPlans.front().exprNode.shell;
    Calc* firstCalc = chain.exprPlans.front().exprNode.calc;
    Calc* lastCalc = chain.exprPlans.back().exprNode.calc;
    if (!shell || !firstCalc || !lastCalc) {
        return "";
    }
    return "__dacpp_mpi_or_fused_rowblock_" + shell->getName() + "_" +
           firstCalc->getName() + "_" + lastCalc->getName() + "_" +
           std::to_string(chain.chainId);
}

bool isFusedRowBlock2DLeader(const OperatorResidentChainPlan& chain,
                             int exprIndex) {
    return chain.fusePointwiseRowBlock2D && !chain.exprPlans.empty() &&
           chain.exprPlans.front().exprIndex == exprIndex;
}

bool isFusedRowBlock2DFollower(const OperatorResidentChainPlan& chain,
                               int exprIndex) {
    return chain.fusePointwiseRowBlock2D && chain.exprPlans.size() >= 2 &&
           chain.exprPlans.front().exprIndex != exprIndex &&
           chain.exprPlans.back().exprIndex == exprIndex;
}

bool isFusedRowBlock2DInterior(const OperatorResidentChainPlan& chain,
                               int exprIndex) {
    if (!chain.fusePointwiseRowBlock2D || chain.exprPlans.size() < 3 ||
        chain.exprPlans.front().exprIndex == exprIndex ||
        chain.exprPlans.back().exprIndex == exprIndex) {
        return false;
    }
    for (const auto& plan : chain.exprPlans) {
        if (plan.exprIndex == exprIndex) {
            return true;
        }
    }
    return false;
}

std::string buildFusedRowBlock2DWrapperCall(
    const OperatorResidentChainPlan& chain,
    DacppFile* dacppFile) {
    if (!chain.fusePointwiseRowBlock2D || chain.exprPlans.size() < 2) {
        return "";
    }
    const std::string wrapperName = fusedRowBlock2DWrapperName(chain);
    if (wrapperName.empty()) {
        return "";
    }
    const ShellPartitionPlan& first = chain.exprPlans.front();
    const ShellPartitionPlan& last = chain.exprPlans.back();
    const ParamAccessPlan* firstReader = singleDirectMappedReader(first);
    const ParamAccessPlan* lastWriter = singleOutputDirectWriter(last);
    const clang::CallExpr* firstCall = getShellCallExpr(first.exprNode.dacExpr);
    const clang::CallExpr* lastCall = getShellCallExpr(last.exprNode.dacExpr);
    if (!firstReader || !lastWriter || !firstCall || !lastCall ||
        firstReader->paramIndex < 0 ||
        firstReader->paramIndex >= static_cast<int>(firstCall->getNumArgs()) ||
        lastWriter->paramIndex < 0 ||
        lastWriter->paramIndex >= static_cast<int>(lastCall->getNumArgs())) {
        return "";
    }
    const std::string firstArg = exprSource(
        firstCall->getArg(firstReader->paramIndex), dacppFile->getContext());
    const std::string lastArg = exprSource(
        lastCall->getArg(lastWriter->paramIndex), dacppFile->getContext());
    if (firstArg.empty() || lastArg.empty()) {
        return "";
    }
    return wrapperName + "(" + firstArg + ", " + lastArg + ")";
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
