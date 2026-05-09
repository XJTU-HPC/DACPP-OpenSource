#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "Rewriter_MPI_Common.h"
#include "mpi/shared/PostRegion_Internal.h"
#include "mpi/shared/LoopLoweredRewrite.h"
#include "Rewriter_MPI_Stencil_Common.h"

namespace {

const clang::CallExpr* getShellCallExpr(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }

    clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    return dacppTranslator::getNode<clang::CallExpr>(shellExpr);
}

std::string getExprSourceText(const clang::Expr* expr,
                              const clang::SourceManager& sourceManager,
                              const clang::LangOptions& langOptions) {
    if (!expr) {
        return "";
    }

    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(expr->getSourceRange()),
               sourceManager,
               langOptions)
        .str();
}

std::string joinShellCallArgs(const clang::BinaryOperator* dacExpr,
                              clang::ASTContext* context) {
    if (!dacExpr || !context) {
        return "";
    }

    const clang::CallExpr* shellCall = getShellCallExpr(dacExpr);
    if (!shellCall) {
        return "";
    }

    const auto& sourceManager = context->getSourceManager();
    const auto& langOptions = context->getLangOpts();
    std::string args;
    for (unsigned argIdx = 0; argIdx < shellCall->getNumArgs(); ++argIdx) {
        if (!args.empty()) {
            args += ", ";
        }
        args += getExprSourceText(shellCall->getArg(argIdx), sourceManager, langOptions);
    }
    return args;
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
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getRHS()
                                                         : dacExpr->getLHS();
    const std::string calcName = getDeclRefName(calcExpr);
    if (calcName.empty()) {
        return "";
    }
    return shellCall->getDirectCallee()->getNameAsString() + "_" + calcName;
}

int exprIndexForDacExpr(dacppTranslator::DacppFile* dacppFile,
                        const clang::BinaryOperator* dacExpr) {
    if (!dacppFile || !dacExpr) {
        return -1;
    }
    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        auto* expr = dacppFile->getExpression(exprIdx);
        if (expr && expr->getDacExpr() == dacExpr) {
            return exprIdx;
        }
    }
    return -1;
}

void insertMainMPISetup(dacppTranslator::DacppFile* dacppFile,
                        clang::Rewriter* rewriter,
                        const clang::FunctionDecl* mainFunc) {
    if (!dacppFile || !rewriter || !mainFunc) {
        return;
    }

    const auto* body = llvm::dyn_cast<clang::CompoundStmt>(mainFunc->getBody());
    if (!body) {
        return;
    }

    dacppTranslator::mpi_rewriter::rewritePrintCallsRootOnly(
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
    dacppTranslator::mpi_rewriter::collectReturnStmts(body, returnStmts);
    for (const clang::ReturnStmt* returnStmt : returnStmts) {
        rewriter->InsertTextBefore(returnStmt->getBeginLoc(), mpiFinish);
    }

    if (!body->body_empty()) {
        const clang::Stmt* lastStmt = body->body_back();
        if (!llvm::isa<clang::ReturnStmt>(lastStmt)) {
            rewriter->InsertTextBefore(body->getRBracLoc(), mpiFinish);
        }
    } else {
        rewriter->InsertTextBefore(body->getRBracLoc(), mpiFinish);
    }
}

}  // namespace

namespace dacppTranslator {
namespace mpi_stencil_rewriter {

void insertMainMPISetup(DacppFile* dacppFile,
                        clang::Rewriter* rewriter,
                        const clang::FunctionDecl* mainFunc) {
    ::insertMainMPISetup(dacppFile, rewriter, mainFunc);
}

void rewriteStencilPhaseCSite(DacppFile* dacppFile,
                              clang::Rewriter* rewriter,
                              const MpiStencilSite& site,
                              int exprIdx,
                              Shell* shell,
                              Calc* calc) {
    if (!dacppFile || !rewriter || !site.dacExpr || !site.outerLoop || !shell ||
        !calc) {
        return;
    }

    const clang::BinaryOperator* dacExpr = site.dacExpr;
    const std::string argText = joinShellCallArgs(dacExpr, dacppFile->getContext());
    const std::string ctxVar =
        "__dacpp_mpi_stencil_ctx_" + std::to_string(exprIdx);
    mpi_rewriter::LoopLoweredRewriteSpec rewriteSpec;
    rewriteSpec.outerLoop = site.outerLoop;
    rewriteSpec.dacExpr = dacExpr;
    rewriteSpec.contextTypeName = contextTypeName(shell, calc, exprIdx);
    rewriteSpec.contextVariableName = ctxVar;
    rewriteSpec.initFunctionName = initFunctionName(shell, calc, exprIdx);
    rewriteSpec.runFunctionName = runFunctionName(shell, calc, exprIdx);
    rewriteSpec.materializeFunctionName =
        materializeFunctionName(shell, calc, exprIdx);
    rewriteSpec.argumentText = argText;
    mpi_rewriter::rewriteLoopLoweredDacExpr(rewriter, rewriteSpec);

    const auto rootRegions =
        mpi_rewriter::collectRootCentricPostRegions(dacppFile, shell, calc,
                                                    exprIdx, dacExpr);
    for (const auto& region : rootRegions) {
        std::string regionCall = region.helperName + "(" + ctxVar;
        if (!argText.empty()) {
            regionCall += ", " + argText;
        }
        regionCall += ");";
        rewriter->ReplaceText(region.stmt->getSourceRange(), regionCall);
    }

    const auto distributedRegions =
        mpi_rewriter::collectDistributedFollowupRegions(dacppFile, shell, calc,
                                                        dacExpr);
    BufferRegionPlan regionPlan;
    const bool hasRegionPlan = mpi_rewriter::buildBufferRegionPlanForDacExpr(
        dacppFile, shell, dacExpr, regionPlan);
    const auto sitePlan =
        mpi_rewriter::analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    const bool useDistributedReaderMaterialize =
        sitePlan.supported && !sitePlan.hasRootBridge &&
        !sitePlan.boundaryLocalStmts.empty() &&
        sitePlan.followupMappings.size() == 1;
    for (const auto& region : distributedRegions) {
        if (!sitePlan.hasRootBridge || useDistributedReaderMaterialize) {
            rewriter->RemoveText(region.stmt->getSourceRange());
            continue;
        }
        std::size_t stmtIdx = 0;
        bool foundStmtIdx = false;
        if (!hasRegionPlan || !regionPlan.enabled || regionPlan.dacExpr != dacExpr) {
            continue;
        }
        for (; stmtIdx < regionPlan.siblingStmts.size(); ++stmtIdx) {
            if (regionPlan.siblingStmts[stmtIdx] == region.stmt) {
                foundStmtIdx = true;
                break;
            }
        }
        if (foundStmtIdx) {
            std::string regionCall =
                mpi_rewriter::detail::helperNameFor(shell, calc, exprIdx,
                                                    stmtIdx) +
                "(" + ctxVar;
            if (!argText.empty()) {
                regionCall += ", " + argText;
            }
            regionCall += ");";
            rewriter->ReplaceText(region.stmt->getSourceRange(), regionCall);
        }
    }
    if (useDistributedReaderMaterialize) {
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            rewriter->RemoveText(stmt->getSourceRange());
        }
    }
}

}  // namespace mpi_stencil_rewriter

void Rewriter::rewriteMPIStencil() {
    std::string generated = mpi_rewriter::buildPrelude(dacppFile);
    std::set<std::string> generatedWrappers;
    std::unordered_map<const clang::BinaryOperator*, MpiStencilSite> siteByExpr;

    for (const auto& site : dacppFile->getMPIStencilSites()) {
        if (!site.dacExpr || !site.outerLoop) {
            continue;
        }
        siteByExpr.emplace(site.dacExpr, site);
    }

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();

        const std::string wrapper =
            mpi_stencil_rewriter::wrapperName(shell, calc, exprIdx);
        if (generatedWrappers.insert(wrapper).second) {
            generated += mpi_rewriter::buildLocalCalcCode(shell, calc);
            generated += "\n";
            generated += mpi_stencil_rewriter::buildStencilWrapperCode(
                dacppFile, shell, calc, exprIdx, expr->getDacExpr());
            generated += "\n";

            rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
            rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
        }
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(), generated);
    mpi_stencil_rewriter::insertMainMPISetup(dacppFile, rewriter,
                                             dacppFile->getMainFunction());

    std::set<const clang::BinaryOperator*> rewrittenDacExprs;
    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        const clang::BinaryOperator* dacExpr = expr->getDacExpr();
        const std::string argText = joinShellCallArgs(dacExpr, dacppFile->getContext());

        auto siteIt = siteByExpr.find(dacExpr);
        if (siteIt != siteByExpr.end()) {
            mpi_stencil_rewriter::rewriteStencilPhaseCSite(
                dacppFile, rewriter, siteIt->second, exprIdx, shell, calc);
            rewrittenDacExprs.insert(dacExpr);
            continue;
        }

        std::string wrapperCall =
            mpi_stencil_rewriter::wrapperName(shell, calc, exprIdx) + "(" +
            argText + ")";
        rewriter->ReplaceText(dacExpr->getSourceRange(), wrapperCall);
        rewrittenDacExprs.insert(dacExpr);
    }

    // Wrapper generation is intentionally deduplicated by shell/calc pair, but
    // every source-level <-> occurrence still needs to be rewritten. Duplicate
    // calls such as odd-even's two ODDEVEN <-> oddeven sites are present in
    // dacExprs even when only one Expression was generated.
    for (const clang::BinaryOperator* dacExpr : dacppFile->dacExprs) {
        if (!dacExpr || rewrittenDacExprs.count(dacExpr) != 0) {
            continue;
        }
        const int exprIdx = exprIndexForDacExpr(dacppFile, dacExpr);
        const std::string wrapper =
            exprIdx >= 0 ? mpi_stencil_rewriter::wrapperName(
                               dacppFile->getExpression(exprIdx)->getShell(),
                               dacppFile->getExpression(exprIdx)->getCalc(),
                               exprIdx)
                         : wrapperNameForDacExpr(dacExpr);
        if (wrapper.empty()) {
            continue;
        }
        const std::string argText = joinShellCallArgs(dacExpr, dacppFile->getContext());
        rewriter->ReplaceText(dacExpr->getSourceRange(),
                              wrapper + "(" + argText + ")");
    }

    dacppFile->setMainAlreadyRewritten(true);
}

}  // namespace dacppTranslator
