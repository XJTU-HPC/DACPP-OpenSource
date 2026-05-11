#include <algorithm>
#include <cctype>
#include <sstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
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

void annotateOutputSync(ShellPartitionPlan& plan, DacppFile* dacppFile) {
    if (!dacppFile) {
        return;
    }
    for (auto& param : plan.params) {
        if (!param.writes ||
            (param.access != ParamAccessKind::OutputDirect &&
             param.access != ParamAccessKind::FixedBlock)) {
            continue;
        }
        const OutputSyncRequirement syncRequirement =
            classifyOutputSyncRequirement(dacppFile, param.actualTensorName,
                                          plan.exprNode.dacExpr);
        const bool orStencilDistributedFollowupLowered =
            syncRequirement == OutputSyncRequirement::DistributedFollowup &&
            isShellDerivedStencilLayout(plan.signature.layout);
        param.broadcastMaterializedOutput =
            syncRequirement != OutputSyncRequirement::RootOnly &&
            !orStencilDistributedFollowupLowered;
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

bool hasReplicatedScalarParam(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            return true;
        }
    }
    return false;
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
            sameColLoopExpr && loopLower == "0" && update.loopUpperInclusive &&
            isZeroExpr(update.targetRowExpr) && isOneExpr(update.sourceRowExpr)) {
            sawTop = true;
            continue;
        }
        if (!update.targetRowUsesLoop && !update.sourceRowUsesLoop &&
            update.targetColUsesLoop && update.sourceColUsesLoop &&
            sameColLoopExpr && loopLower == "0" && update.loopUpperInclusive &&
            isLastIndexExpr(update.targetRowExpr, readerRows) &&
            isPenultimateIndexExpr(update.sourceRowExpr, readerRows)) {
            sawBottom = true;
            continue;
        }
        if (update.targetRowUsesLoop && update.sourceRowUsesLoop &&
            !update.targetColUsesLoop && !update.sourceColUsesLoop &&
            sameRowLoopExpr && loopLower == "0" &&
            !update.loopUpperInclusive &&
            isZeroExpr(update.targetColExpr) && isOneExpr(update.sourceColExpr)) {
            sawLeft = true;
            continue;
        }
        if (update.targetRowUsesLoop && update.sourceRowUsesLoop &&
            !update.targetColUsesLoop && !update.sourceColUsesLoop &&
            sameRowLoopExpr && loopLower == "0" &&
            !update.loopUpperInclusive &&
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
                llvm::outs() << " materialize=per-run";
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
                            << ")";
                    }
                } else {
                    llvm::outs()
                        << " window-size="
                        << plan.orLoopLower.stencilResidentHalo.windowSize
                        << " followup-offset="
                        << plan.orLoopLower.stencilResidentHalo
                               .followupTargetOffset;
                }
                llvm::outs() << " materialize=final";
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

void finalizeChain(OperatorResidentChainPlan& chain) {
    if (!chain.supported) {
        return;
    }
    analyzeResidency(chain);
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
    analyzeResidency(chain);
    return chain;
}

std::vector<OperatorResidentChainPlan> buildOperatorResidentChains(
    DacppFile* dacppFile,
    const std::vector<DacExprNode>& exprNodes,
    const std::vector<ShellPartitionPlan>& shellPlans) {
    std::vector<ShellPartitionPlan> plans = shellPlans;
    for (auto& plan : plans) {
        fillActualTensorNames(plan, dacppFile);
        annotateOutputSync(plan, dacppFile);
    }
    annotateLoopLowerCandidates(dacppFile, plans);
    detectAndAnnotateFixedBlockPhaseExchange(dacppFile, plans);

    std::vector<OperatorResidentChainPlan> chains;
    OperatorResidentChainPlan current;
    current.chainId = -1;

    auto closeCurrent = [&]() {
        if (current.supported && !current.exprPlans.empty()) {
            finalizeChain(current);
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

} // namespace mpi_rewriter
} // namespace dacppTranslator
