#include <algorithm>
#include <cctype>
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
    llvm::outs() << "[DACPP][MPI][OR] chain=" << chain.chainId
                 << " layout=" << localLayoutKindName(chain.signature.layout)
                 << " length=" << chain.exprPlans.size() << " chain=accepted codegen="
                 << (supportedPhaseLayout(chain.signature.layout) ? "enabled" : "disabled")
                 << "\n";
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
