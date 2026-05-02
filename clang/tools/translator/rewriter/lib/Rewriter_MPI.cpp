#include <set>
#include <string>
#include <vector>

#include "clang/AST/Stmt.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {

void Rewriter::rewriteMPI() {
    if (dacppFile && dacppFile->hasMPIStencilSites()) {
        rewriteMPIStencil();
        return;
    }

    std::string generated = mpi_rewriter::buildPrelude(dacppFile);
    std::set<std::string> generatedWrappers;

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();

        const std::string wrapperName = shell->getName() + "_" + calc->getName();
        if (generatedWrappers.insert(wrapperName).second) {
            generated += mpi_rewriter::buildLocalCalcCode(shell, calc);
            generated += "\n";
            generated += mpi_rewriter::buildWrapperCode(
                dacppFile, shell, calc, expr->getDacExpr());
            generated += "\n";

            rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
            rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
        }
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(), generated);

    const FunctionDecl* mainFunc = dacppFile->getMainFunction();
    if (!mainFunc) {
        return;
    }

    const auto* body = llvm::dyn_cast<CompoundStmt>(mainFunc->getBody());
    if (!body) {
        return;
    }

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
    if (mpi_rank != 0) {
        std::freopen("/dev/null", "w", stdout);
    }
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
}

}  // namespace dacppTranslator
