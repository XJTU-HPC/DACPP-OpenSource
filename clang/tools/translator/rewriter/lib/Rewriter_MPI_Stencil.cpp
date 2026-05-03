#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "Rewriter_MPI_Common.h"
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

        const std::string wrapper = mpi_stencil_rewriter::wrapperName(shell, calc);
        if (generatedWrappers.insert(wrapper).second) {
            generated += mpi_rewriter::buildLocalCalcCode(shell, calc);
            generated += "\n";
            generated += mpi_stencil_rewriter::buildStencilWrapperCode(
                dacppFile, shell, calc, expr->getDacExpr());
            generated += "\n";

            rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
            rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
        }
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(), generated);
    insertMainMPISetup(dacppFile, rewriter, dacppFile->getMainFunction());

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        const clang::BinaryOperator* dacExpr = expr->getDacExpr();
        const std::string argText = joinShellCallArgs(dacExpr, dacppFile->getContext());

        auto siteIt = siteByExpr.find(dacExpr);
        if (siteIt != siteByExpr.end()) {
            const std::string ctxVar = "__dacpp_mpi_stencil_ctx_" + std::to_string(exprIdx);
            std::string initCode = "    " +
                mpi_stencil_rewriter::contextTypeName(shell, calc) + " " + ctxVar + ";\n";
            initCode += "    " + mpi_stencil_rewriter::initFunctionName(shell, calc) +
                        "(" + ctxVar;
            if (!argText.empty()) {
                initCode += ", " + argText;
            }
            initCode += ");\n";
            rewriter->InsertTextBefore(siteIt->second.outerLoop->getBeginLoc(), initCode);

            std::string runCall = mpi_stencil_rewriter::runFunctionName(shell, calc) +
                                  "(" + ctxVar;
            if (!argText.empty()) {
                runCall += ", " + argText;
            }
            runCall += ")";
            rewriter->ReplaceText(dacExpr->getSourceRange(), runCall);

            const auto rootRegions = mpi_rewriter::collectRootCentricPostRegions(
                dacppFile, shell, calc, dacExpr);
            for (const auto& region : rootRegions) {
                std::string regionCall = region.helperName + "(" + ctxVar;
                if (!argText.empty()) {
                    regionCall += ", " + argText;
                }
                regionCall += ");";
                rewriter->ReplaceText(region.stmt->getSourceRange(), regionCall);
            }
            continue;
        }

        std::string wrapperCall = mpi_stencil_rewriter::wrapperName(shell, calc) + "(" + argText + ")";
        rewriter->ReplaceText(dacExpr->getSourceRange(), wrapperCall);
    }

    dacppFile->setMainAlreadyRewritten(true);
}

}  // namespace dacppTranslator
