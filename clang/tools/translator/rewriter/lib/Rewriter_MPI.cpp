#include <set>
#include <string>
#include <vector>

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"
#include "Rewriter_MPI_Plan.h"

namespace dacppTranslator {

void Rewriter::rewriteMPI() {
    auto plan = mpi_rewriter::buildMpiLoweringPlan(dacppFile);
    if (plan.overallKind == mpi_rewriter::MpiPlanKind::StencilPhaseC) {
        rewriteMPIStencil();
        return;
    }

    std::string generated = mpi_rewriter::buildPrelude(dacppFile);
    std::set<std::string> generatedWrappers;
    std::set<std::string> generatedLocalCalcs;
    std::set<const clang::FunctionDecl*> removedDecls;

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        const bool isOperatorResident =
            exprIdx < static_cast<int>(plan.exprResults.size()) &&
            plan.exprResults[exprIdx].kind ==
                mpi_rewriter::MpiPlanKind::OperatorResident;
        std::string wrapperName =
            isOperatorResident
                ? mpi_rewriter::operatorResidentWrapperName(shell, calc, exprIdx)
                : shell->getName() + "_" + calc->getName();
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
                generated += mpi_rewriter::buildOperatorResidentWrapperCode(
                    dacppFile, chain, *exprPlan);
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
        const bool isOperatorResident =
            exprIdx < static_cast<int>(plan.exprResults.size()) &&
            plan.exprResults[exprIdx].kind ==
                mpi_rewriter::MpiPlanKind::OperatorResident;
        const std::string wrapperName =
            isOperatorResident
                ? mpi_rewriter::operatorResidentWrapperName(shell, calc, exprIdx)
                : shell->getName() + "_" + calc->getName();
        rewriter->ReplaceText(
            dacExpr->getSourceRange(),
            mpi_rewriter::buildWrapperCallForDacExpr(wrapperName, dacExpr,
                                                     dacppFile));
        rewrittenDacExprs.insert(dacExpr);
    }

    const FunctionDecl* mainFunc = dacppFile->getMainFunction();
    if (!mainFunc) {
        return;
    }

    const auto* body = llvm::dyn_cast<CompoundStmt>(mainFunc->getBody());
    if (!body) {
        return;
    }

    mpi_rewriter::rewritePrintCallsRootOnly(
        rewriter, dacppFile->getTranslationUnitDecl());

    const std::string mpiInit = R"(
    int dacpp_mpi_finalize_needed = 0;
    int dacpp_mpi_initialized = 0;
    MPI_Initialized(&dacpp_mpi_initialized);
    if (!dacpp_mpi_initialized) {
        int dacpp_mpi_argc = 0;
        char** dacpp_mpi_argv = nullptr;
        MPI_Init(&dacpp_mpi_argc, &dacpp_mpi_argv);
        dacpp_mpi_finalize_needed = 1;
    }
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
)";

    const std::string mpiFinish = R"(
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
)";

    rewriter->InsertTextAfterToken(body->getLBracLoc(), mpiInit);
    std::vector<const clang::ReturnStmt*> returnStmts;
    mpi_rewriter::collectReturnStmts(body, returnStmts);
    for (const clang::ReturnStmt* returnStmt : returnStmts) {
        rewriter->InsertTextBefore(returnStmt->getBeginLoc(), mpiFinish);
    }

    if (!body->body_empty()) {
        const Stmt* lastStmt = body->body_back();
        if (!llvm::isa<clang::ReturnStmt>(lastStmt)) {
            rewriter->InsertTextBefore(body->getRBracLoc(), mpiFinish);
        }
    } else {
        rewriter->InsertTextBefore(body->getRBracLoc(), mpiFinish);
    }

    dacppFile->setMainAlreadyRewritten(true);
}

}  // namespace dacppTranslator
