#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

const clang::CallExpr* getShellCallExprMPI(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }

    const clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    return dacppTranslator::getNode<clang::CallExpr>(
        const_cast<clang::Expr*>(shellExpr));
}

const clang::ValueDecl* getExprValueDeclMPI(const clang::Expr* expr) {
    if (!expr) {
        return nullptr;
    }

    const clang::Expr* stripped = expr->IgnoreParenImpCasts();
    if (const auto* DRE = llvm::dyn_cast<clang::DeclRefExpr>(stripped)) {
        return DRE->getDecl();
    }
    return nullptr;
}

std::vector<const clang::ValueDecl*> collectShellCallArgDeclsMPI(
    const clang::BinaryOperator* dacExpr,
    int expectedArgCount) {
    std::vector<const clang::ValueDecl*> argDecls(
        static_cast<std::size_t>(expectedArgCount), nullptr);
    const clang::CallExpr* shellCall = getShellCallExprMPI(dacExpr);
    if (!shellCall) {
        return argDecls;
    }

    const int argCount = std::min<int>(expectedArgCount, shellCall->getNumArgs());
    for (int argIdx = 0; argIdx < argCount; ++argIdx) {
        argDecls[static_cast<std::size_t>(argIdx)] =
            getExprValueDeclMPI(shellCall->getArg(static_cast<unsigned>(argIdx)));
    }
    return argDecls;
}

}  // namespace

MPIRegionTransferPolicy analyzeMPIRegionTransferPolicy(
    DacppFile* dacppFile,
    Expression* expr,
    const std::vector<IOTYPE>& paramModes) {
    MPIRegionTransferPolicy policy;
    const int n = static_cast<int>(paramModes.size());
    policy.needsInitScatter.assign(static_cast<std::size_t>(n), false);
    policy.needsSyncGather.assign(static_cast<std::size_t>(n), false);
    for (int i = 0; i < n; ++i) {
        if (paramModes[i] != IOTYPE::WRITE) {
            policy.needsInitScatter[static_cast<std::size_t>(i)] = true;
        }
        if (paramModes[i] != IOTYPE::READ) {
            policy.needsSyncGather[static_cast<std::size_t>(i)] = true;
        }
    }

    if (!dacppFile || !expr) {
        return policy;
    }

    Shell* shell = expr->getShell();
    const auto& plan = dacppFile->getBufferRegionPlan();
    if (!shell || !plan.enabled) {
        return policy;
    }

    const auto argDecls = collectShellCallArgDeclsMPI(plan.dacExpr, n);
    std::unordered_map<const clang::ValueDecl*, int> argDeclIndices;
    for (int paramIdx = 0; paramIdx < n; ++paramIdx) {
        const clang::ValueDecl* argDecl = argDecls[static_cast<std::size_t>(paramIdx)];
        if (argDecl) {
            argDeclIndices.emplace(argDecl, paramIdx);
        }
    }

    for (const clang::ForStmt* siblingFor : plan.siblingForStmts) {
        const auto siblingSummary =
            summarizeStmtAccess(siblingFor, argDeclIndices, n);
        for (int paramIdx = 0; paramIdx < n; ++paramIdx) {
            const AccessSummary& access =
                siblingSummary[static_cast<std::size_t>(paramIdx)];
            if (access.reads &&
                paramModes[static_cast<std::size_t>(paramIdx)] != IOTYPE::WRITE) {
                policy.needsInitScatter[static_cast<std::size_t>(paramIdx)] = true;
            }
            if (access.writes) {
                policy.needsSyncGather[static_cast<std::size_t>(paramIdx)] = true;
                if (paramModes[static_cast<std::size_t>(paramIdx)] == IOTYPE::WRITE) {
                    policy.needsInitScatter[static_cast<std::size_t>(paramIdx)] = true;
                }
            }
        }
    }

    return policy;
}

std::string joinShellCallArgsMPI(const clang::BinaryOperator* dacExpr,
                                 clang::ASTContext* context) {
    if (!dacExpr || !context) {
        return "";
    }

    clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    auto* shellCall = dacppTranslator::getNode<clang::CallExpr>(shellExpr);
    if (!shellCall) {
        return "";
    }

    const auto& SM = context->getSourceManager();
    const auto& LO = context->getLangOpts();
    std::string args;
    for (unsigned argIdx = 0; argIdx < shellCall->getNumArgs(); ++argIdx) {
        if (!args.empty()) {
            args += ", ";
        }
        args += clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(
                shellCall->getArg(argIdx)->getSourceRange()),
            SM, LO).str();
    }
    return args;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
