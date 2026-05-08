#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/ASTContext.h"
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
    }
}

void annotateOutputSync(ShellPartitionPlan& plan, DacppFile* dacppFile) {
    if (!dacppFile) {
        return;
    }
    for (auto& param : plan.params) {
        if (!param.writes || param.access != ParamAccessKind::OutputDirect) {
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
        if (param.writes && param.access == ParamAccessKind::OutputDirect) {
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
           layout == LocalLayoutKind::StencilWindow2D;
}

bool directResidentLoopLayout(LocalLayoutKind layout) {
    return layout == LocalLayoutKind::Contiguous1D;
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
