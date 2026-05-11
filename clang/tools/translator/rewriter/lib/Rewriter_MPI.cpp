#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Stmt.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"
#include "Rewriter_MPI_Plan.h"
#include "Rewriter_MPI_Stencil_Common.h"
#include "mpi/operator_resident/LoopLocalStencilOwnerLoop.h"
#include "mpi/shared/LoopLoweredRewrite.h"

namespace dacppTranslator {

namespace {

const clang::CallExpr* getShellCallExpr(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }
    clang::Expr* shellExpr =
        Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS() : dacExpr->getRHS();
    return dacppTranslator::getNode<clang::CallExpr>(shellExpr);
}

std::string getDeclRefName(const clang::Expr* expr) {
    if (!expr) {
        return "";
    }
    const auto* declRef = dacppTranslator::getNode<clang::DeclRefExpr>(
        const_cast<clang::Expr*>(expr));
    if (!declRef || !declRef->getDecl()) {
        return "";
    }
    return declRef->getDecl()->getNameAsString();
}

std::string wrapperNameForDacExpr(const clang::BinaryOperator* dacExpr) {
    const clang::CallExpr* shellCall = getShellCallExpr(dacExpr);
    if (!shellCall || !shellCall->getDirectCallee()) {
        return "";
    }
    const clang::Expr* calcExpr =
        Expression::shellLHS_p(dacExpr) ? dacExpr->getRHS() : dacExpr->getLHS();
    const std::string calcName = getDeclRefName(calcExpr);
    if (calcName.empty()) {
        return "";
    }
    return shellCall->getDirectCallee()->getNameAsString() + "_" + calcName;
}

const mpi_rewriter::ShellPartitionPlan* findOperatorResidentExprPlan(
    const mpi_rewriter::MpiLoweringPlan& plan,
    int exprIdx) {
    if (exprIdx < 0 ||
        exprIdx >= static_cast<int>(plan.exprResults.size())) {
        return nullptr;
    }
    const int chainId = plan.exprResults[exprIdx].operatorResidentChainId;
    if (chainId < 0 || chainId >= static_cast<int>(plan.residentChains.size())) {
        return nullptr;
    }
    for (const auto& candidate : plan.residentChains[chainId].exprPlans) {
        if (candidate.exprIndex == exprIdx) {
            return &candidate;
        }
    }
    return nullptr;
}

void removeOperatorResidentLoweredPostStmts(
    DacppFile* dacppFile,
    clang::Rewriter* rewriter,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    if (!dacppFile || !rewriter || !shell || !calc || !dacExpr) {
        return;
    }
    std::set<const clang::Stmt*> stmtsToRemove;
    const auto distributedRegions =
        mpi_rewriter::collectDistributedFollowupRegions(dacppFile, shell, calc,
                                                        dacExpr);
    for (const auto& region : distributedRegions) {
        stmtsToRemove.insert(region.stmt);
    }

    const auto sitePlan =
        mpi_rewriter::analyzeDistributedStencilSite(dacppFile, shell, calc,
                                                    dacExpr);
    if (sitePlan.supported && !sitePlan.hasRootBridge) {
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            stmtsToRemove.insert(stmt);
        }
    }

    for (const clang::Stmt* stmt : stmtsToRemove) {
        if (stmt) {
            rewriter->RemoveText(stmt->getSourceRange());
        }
    }
}

void removeContractStmts(clang::Rewriter* rewriter,
                         const mpi_rewriter::LoopLoweringContract& contract) {
    if (!rewriter) {
        return;
    }
    for (const clang::Stmt* stmt :
         mpi_rewriter::loweringContractRemoveStmtSet(contract)) {
        if (stmt) {
            rewriter->RemoveText(stmt->getSourceRange());
        }
    }
}

bool isLoopLoweredOperatorResidentExpr(
    const mpi_rewriter::MpiLoweringPlan& plan,
    int exprIdx,
    const mpi_rewriter::ShellPartitionPlan** exprPlanOut) {
    const auto* exprPlan = findOperatorResidentExprPlan(plan, exprIdx);
    if (!exprPlan ||
        !mpi_rewriter::isLoopLoweredOperatorResidentPlan(*exprPlan)) {
        return false;
    }
    if (exprPlanOut) {
        *exprPlanOut = exprPlan;
    }
    return true;
}

void rewriteLoopLoweredOperatorResidentSite(
    DacppFile* dacppFile,
    clang::Rewriter* rewriter,
    const mpi_rewriter::ShellPartitionPlan& exprPlan) {
    if (!dacppFile || !rewriter || !exprPlan.exprNode.dacExpr ||
        !exprPlan.exprNode.shell ||
        !exprPlan.exprNode.calc) {
        return;
    }

    Shell* shell = exprPlan.exprNode.shell;
    Calc* calc = exprPlan.exprNode.calc;
    const int exprIdx = exprPlan.exprIndex;
    const clang::BinaryOperator* dacExpr = exprPlan.exprNode.dacExpr;
    const clang::Stmt* outerLoop =
        exprPlan.orLoopLower.outerLoop ? exprPlan.orLoopLower.outerLoop
                                       : exprPlan.loopLowerOuterLoop;
    if (!outerLoop) {
        return;
    }

    // FixedBlockPhaseExchangeFollower: the follower's DAC expression and
    // surrounding helper statements were already removed when plan A was
    // rewritten. Skip to avoid double rewriting.
    if (exprPlan.orLoopLower.kind ==
        mpi_rewriter::OrLoopLowerKind::FixedBlockPhaseExchangeFollower) {
        return;
    }

    const std::string argText =
        mpi_rewriter::joinShellCallArgs(dacExpr, dacppFile);
    const std::string ctxVar =
        "__dacpp_mpi_or_ctx_" + std::to_string(exprIdx);

    mpi_rewriter::LoopLoweredRewriteSpec rewriteSpec;
    rewriteSpec.outerLoop = outerLoop;
    rewriteSpec.dacExpr = dacExpr;
    rewriteSpec.contextTypeName =
        mpi_rewriter::operatorResidentContextTypeName(shell, calc, exprIdx);
    rewriteSpec.contextVariableName = ctxVar;
    rewriteSpec.initFunctionName =
        mpi_rewriter::operatorResidentInitFunctionName(shell, calc, exprIdx);
    rewriteSpec.runFunctionName =
        mpi_rewriter::operatorResidentRunFunctionName(shell, calc, exprIdx);
    rewriteSpec.materializeFunctionName =
        mpi_rewriter::operatorResidentMaterializeFunctionName(shell, calc,
                                                              exprIdx);
    rewriteSpec.argumentText = argText;
    mpi_rewriter::rewriteLoopLoweredDacExpr(rewriter, rewriteSpec);

    if (exprPlan.orLoopLower.kind ==
            mpi_rewriter::OrLoopLowerKind::StencilFullSync ||
        exprPlan.orLoopLower.kind ==
            mpi_rewriter::OrLoopLowerKind::StencilResidentHalo) {
        if (exprPlan.orLoopLower.contractRemovalSetMatchesLegacy) {
            removeContractStmts(rewriter, exprPlan.orLoopLower.contract);
        } else {
            llvm::outs()
                << "[DACPP][MPI][OR][P4.6][ContractRemoval] expr="
                << exprIdx
                << " fallback=legacy reason="
                << (exprPlan.orLoopLower.contractRemovalSetReason.empty()
                        ? "contract removal set mismatch"
                        : exprPlan.orLoopLower.contractRemovalSetReason)
                << "\n";
            removeOperatorResidentLoweredPostStmts(dacppFile, rewriter, shell,
                                                   calc, dacExpr);
        }
    }

    if (exprPlan.orLoopLower.kind ==
        mpi_rewriter::OrLoopLowerKind::FixedBlockPhaseExchange) {
        removeContractStmts(rewriter, exprPlan.orLoopLower.contract);
    }
}

}  // namespace

void Rewriter::rewriteMPI() {
    auto plan = mpi_rewriter::buildMpiLoweringPlan(dacppFile);

    std::string generated = mpi_rewriter::buildPrelude(dacppFile);
    std::set<std::string> generatedWrappers;
    std::set<std::string> generatedLocalCalcs;
    std::set<const clang::FunctionDecl*> removedDecls;
    std::unordered_map<const clang::BinaryOperator*, MpiStencilSite> siteByExpr;
    std::unordered_map<int, mpi_rewriter::operator_resident::
                                LoopLocalStencilOwnerLoopContract>
        ownerLoopContracts;
    std::set<const clang::Stmt*> rewrittenFoulaLoops;

    for (const auto& site : dacppFile->getMPIStencilSites()) {
        if (!site.dacExpr || !site.outerLoop) {
            continue;
        }
        siteByExpr.emplace(site.dacExpr, site);
    }

    for (const auto& chain : plan.residentChains) {
        if (!chain.supported) {
            continue;
        }
        for (const auto& exprPlan : chain.exprPlans) {
            auto contract =
                mpi_rewriter::operator_resident::
                    detectLoopLocalStencilOwnerLoop(dacppFile, exprPlan);
            if (contract.enabled) {
                ownerLoopContracts[exprPlan.exprIndex] = contract;
                llvm::outs() << "[DACPP][MPI][OR][FOuLa] expr="
                             << exprPlan.exprIndex
                             << " owner-loop=candidate owner="
                             << contract.ownerTensorName << "\n";
                mpi_rewriter::operator_resident::
                    logLoopLocalStencilOwnerLoopAccepted(contract);
            } else if (
                exprPlan.signature.layout ==
                    mpi_rewriter::LocalLayoutKind::StencilWindow1D &&
                exprPlan.params.size() == 3) {
                mpi_rewriter::operator_resident::
                    logLoopLocalStencilOwnerLoopRejected(contract);
            }
        }
    }

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        const bool hasPlanResult =
            exprIdx < static_cast<int>(plan.exprResults.size());
        const mpi_rewriter::MpiPlanKind planKind =
            hasPlanResult ? plan.exprResults[exprIdx].kind
                          : mpi_rewriter::MpiPlanKind::Unsupported;
        const bool isOperatorResident =
            planKind == mpi_rewriter::MpiPlanKind::OperatorResident;
        const bool isStencilPhaseC =
            planKind == mpi_rewriter::MpiPlanKind::StencilPhaseC;
        std::string wrapperName;
        if (isOperatorResident) {
            wrapperName =
                mpi_rewriter::operatorResidentWrapperName(shell, calc, exprIdx);
        } else if (isStencilPhaseC) {
            wrapperName =
                mpi_stencil_rewriter::wrapperName(shell, calc, exprIdx);
        } else {
            wrapperName = shell->getName() + "_" + calc->getName();
        }
        if (generatedWrappers.insert(wrapperName).second) {
            const std::string localCalcKey = calc->getName();
            if (generatedLocalCalcs.insert(localCalcKey).second) {
                generated += mpi_rewriter::buildLocalCalcCode(shell, calc);
                generated += "\n";
            }
            if (isOperatorResident) {
                const int chainId =
                    plan.exprResults[exprIdx].operatorResidentChainId;
                const auto& chain = plan.residentChains[chainId];
                const mpi_rewriter::ShellPartitionPlan* exprPlan = nullptr;
                for (const auto& candidate : chain.exprPlans) {
                    if (candidate.exprIndex == exprIdx) {
                        exprPlan = &candidate;
                        break;
                    }
                }
                if (!exprPlan) {
                    continue;
                }
                auto ownerLoopIt = ownerLoopContracts.find(exprIdx);
                if (ownerLoopIt != ownerLoopContracts.end()) {
                    generated +=
                        mpi_rewriter::operator_resident::
                            buildLoopLocalStencilOwnerLoopCode(
                                ownerLoopIt->second, *exprPlan);
                } else {
                    generated += mpi_rewriter::buildOperatorResidentWrapperCode(
                        dacppFile, chain, *exprPlan);
                }
            } else if (isStencilPhaseC) {
                generated += mpi_stencil_rewriter::buildStencilWrapperCode(
                    dacppFile, shell, calc, exprIdx, expr->getDacExpr());
            } else {
                generated += mpi_rewriter::buildWrapperCode(
                    dacppFile, shell, calc, expr->getDacExpr());
            }
            generated += "\n";

            if (removedDecls.insert(shell->getShellLoc()).second) {
                rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
            }
            if (removedDecls.insert(calc->getCalcLoc()).second) {
                rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
            }
        }
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(), generated);

    std::set<const clang::BinaryOperator*> rewrittenDacExprs;
    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        const clang::BinaryOperator* dacExpr = expr->getDacExpr();
        const bool hasPlanResult =
            exprIdx < static_cast<int>(plan.exprResults.size());
        const mpi_rewriter::MpiPlanKind planKind =
            hasPlanResult ? plan.exprResults[exprIdx].kind
                          : mpi_rewriter::MpiPlanKind::Unsupported;
        auto ownerLoopIt = ownerLoopContracts.find(exprIdx);
        if (ownerLoopIt != ownerLoopContracts.end() &&
            ownerLoopIt->second.replacementStmt &&
            rewrittenFoulaLoops.insert(ownerLoopIt->second.replacementStmt)
                .second) {
            rewriter->ReplaceText(
                ownerLoopIt->second.replacementStmt->getSourceRange(),
                ownerLoopIt->second.functionName + "(" +
                    ownerLoopIt->second.ownerTensorName + ", " +
                    ownerLoopIt->second.scalarExpr + ");");
            rewrittenDacExprs.insert(dacExpr);
            llvm::outs() << "[DACPP][MPI][OR][FOuLa] expr=" << exprIdx
                         << " owner-loop=rewrite-enabled\n";
            mpi_rewriter::operator_resident::
                logLoopLocalStencilOwnerLoopRewriteEnabled(
                    ownerLoopIt->second);
            continue;
        }
        if (planKind == mpi_rewriter::MpiPlanKind::StencilPhaseC) {
            auto siteIt = siteByExpr.find(dacExpr);
            if (siteIt != siteByExpr.end()) {
                mpi_stencil_rewriter::rewriteStencilPhaseCSite(
                    dacppFile, rewriter, siteIt->second, exprIdx, shell, calc);
                rewrittenDacExprs.insert(dacExpr);
                continue;
            }
        }
        const mpi_rewriter::ShellPartitionPlan* loopLowerPlan = nullptr;
        if (planKind == mpi_rewriter::MpiPlanKind::OperatorResident &&
            isLoopLoweredOperatorResidentExpr(plan, exprIdx, &loopLowerPlan)) {
            rewriteLoopLoweredOperatorResidentSite(dacppFile, rewriter,
                                                   *loopLowerPlan);
            rewrittenDacExprs.insert(dacExpr);
            continue;
        }

        const std::string wrapperName =
            planKind == mpi_rewriter::MpiPlanKind::OperatorResident
                ? mpi_rewriter::operatorResidentWrapperName(shell, calc, exprIdx)
                : shell->getName() + "_" + calc->getName();
        rewriter->ReplaceText(
            dacExpr->getSourceRange(),
            mpi_rewriter::buildWrapperCallForDacExpr(wrapperName, dacExpr,
                                                     dacppFile));
        if (planKind == mpi_rewriter::MpiPlanKind::OperatorResident) {
            const auto* exprPlan = findOperatorResidentExprPlan(plan, exprIdx);
            if (exprPlan &&
                mpi_rewriter::isShellDerivedStencilLayout(
                    exprPlan->signature.layout)) {
                removeOperatorResidentLoweredPostStmts(
                    dacppFile, rewriter, shell, calc, dacExpr);
            }
        }
        rewrittenDacExprs.insert(dacExpr);
    }

    for (const clang::BinaryOperator* dacExpr : dacppFile->dacExprs) {
        if (!dacExpr || rewrittenDacExprs.count(dacExpr) != 0) {
            continue;
        }
        const std::string wrapperName = wrapperNameForDacExpr(dacExpr);
        if (wrapperName.empty()) {
            continue;
        }
        rewriter->ReplaceText(
            dacExpr->getSourceRange(),
            mpi_rewriter::buildWrapperCallForDacExpr(wrapperName, dacExpr,
                                                     dacppFile));
    }

    const FunctionDecl* mainFunc = dacppFile->getMainFunction();
    if (!mainFunc) {
        return;
    }

    mpi_stencil_rewriter::insertMainMPISetup(dacppFile, rewriter, mainFunc);

    dacppFile->setMainAlreadyRewritten(true);
}

}  // namespace dacppTranslator
