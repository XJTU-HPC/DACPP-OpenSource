#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

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

/// MPI-specific transfer policy: decides which params need Scatterv at init
/// and which need Gatherv at sync.
struct MPIRegionTransferPolicy {
    std::vector<bool> needsInitScatter;   // Scatterv at init (READ / READ_WRITE)
    std::vector<bool> needsSyncGather;    // Gatherv at sync (WRITE / READ_WRITE)
};

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

    for (const clang::Stmt* siblingStmt : plan.siblingStmts) {
        const auto siblingSummary =
            summarizeStmtAccess(siblingStmt, argDeclIndices, n);
        for (int paramIdx = 0; paramIdx < n; ++paramIdx) {
            const AccessSummary& access =
                siblingSummary[static_cast<std::size_t>(paramIdx)];
            if (access.reads && paramModes[static_cast<std::size_t>(paramIdx)] != IOTYPE::WRITE) {
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

struct MPISiblingRewriteResult {
    std::string code;
    std::set<std::string> readVars;
    std::set<std::string> writtenVars;
};

std::string getSourceTextMPI(const clang::Stmt* stmt,
                             clang::ASTContext* context) {
    if (!stmt || !context) {
        return "";
    }
    return clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(stmt->getSourceRange()),
        context->getSourceManager(),
        context->getLangOpts()).str();
}

std::string getSourceTextMPI(const clang::Expr* expr,
                             clang::ASTContext* context) {
    if (!expr || !context) {
        return "";
    }
    return clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(expr->getSourceRange()),
        context->getSourceManager(),
        context->getLangOpts()).str();
}

void collectWrittenParamNamesMPI(
    const clang::Stmt* stmt,
    const std::unordered_map<const clang::ValueDecl*, std::string>& declToName,
    std::set<std::string>& writtenNames) {
    if (!stmt) {
        return;
    }

    class WriteVisitor : public clang::RecursiveASTVisitor<WriteVisitor> {
    public:
        const std::unordered_map<const clang::ValueDecl*, std::string>& DeclToName;
        std::set<std::string>& WrittenNames;
        int WriteDepth = 0;

        WriteVisitor(
            const std::unordered_map<const clang::ValueDecl*, std::string>& declToName,
            std::set<std::string>& writtenNames)
            : DeclToName(declToName), WrittenNames(writtenNames) {}

        bool VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
            if (!DRE || WriteDepth <= 0) {
                return true;
            }
            auto it = DeclToName.find(DRE->getDecl());
            if (it != DeclToName.end()) {
                WrittenNames.insert(it->second);
            }
            return true;
        }

        bool TraverseBinaryOperator(clang::BinaryOperator* BO) {
            if (!BO) {
                return true;
            }
            if (BO->isAssignmentOp()) {
                ++WriteDepth;
                TraverseStmt(BO->getLHS());
                --WriteDepth;
                TraverseStmt(BO->getRHS());
                return true;
            }
            return clang::RecursiveASTVisitor<WriteVisitor>::TraverseBinaryOperator(BO);
        }

        bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* BO) {
            if (!BO) {
                return true;
            }
            ++WriteDepth;
            TraverseStmt(BO->getLHS());
            --WriteDepth;
            TraverseStmt(BO->getRHS());
            return true;
        }

        bool TraverseUnaryOperator(clang::UnaryOperator* UO) {
            if (!UO) {
                return true;
            }
            if (UO->isIncrementDecrementOp()) {
                ++WriteDepth;
                TraverseStmt(UO->getSubExpr());
                --WriteDepth;
                return true;
            }
            return clang::RecursiveASTVisitor<WriteVisitor>::TraverseUnaryOperator(UO);
        }

        bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall) {
            if (!OpCall) {
                return true;
            }
            if (OpCall->isAssignmentOp()) {
                if (OpCall->getNumArgs() > 0) {
                    ++WriteDepth;
                    TraverseStmt(OpCall->getArg(0));
                    --WriteDepth;
                }
                for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs(); ++argIdx) {
                    TraverseStmt(OpCall->getArg(argIdx));
                }
                return true;
            }
            return clang::RecursiveASTVisitor<WriteVisitor>::TraverseCXXOperatorCallExpr(OpCall);
        }
    };

    WriteVisitor visitor(declToName, writtenNames);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
}

std::string rewriteMPIIndexExpr(const clang::Expr* expr,
                                clang::ASTContext* context,
                                const std::string& loopVar) {
    std::string text = getSourceTextMPI(expr, context);
    if (text.empty()) {
        return loopVar;
    }

    std::size_t pos = 0;
    while ((pos = text.find(loopVar, pos)) != std::string::npos) {
        const bool leftOk =
            pos == 0 ||
            (!std::isalnum(static_cast<unsigned char>(text[pos - 1])) &&
             text[pos - 1] != '_');
        const std::size_t end = pos + loopVar.size();
        const bool rightOk =
            end >= text.size() ||
            (!std::isalnum(static_cast<unsigned char>(text[end])) &&
             text[end] != '_');
        if (leftOk && rightOk) {
            text.replace(pos, loopVar.size(), "__dacpp_i");
            pos += std::string("__dacpp_i").size();
        } else {
            pos = end;
        }
    }
    return text;
}

MPISiblingRewriteResult rewriteMPISiblingBody(
    const clang::Stmt* stmt,
    clang::ASTContext* context,
    const std::unordered_map<const clang::ValueDecl*, std::string>& declToName,
    const std::set<std::string>& writtenNames,
    const std::string& loopVar,
    bool topLevel = true) {
    MPISiblingRewriteResult result;
    if (!stmt || !context) {
        return result;
    }

    if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
        result.code += topLevel ? "{\n" : "{\n";
        for (const clang::Stmt* child : compound->body()) {
            auto childResult = rewriteMPISiblingBody(
                child, context, declToName, writtenNames, loopVar, false);
            result.code += childResult.code;
            if (!result.code.empty() && result.code.back() != '\n') {
                result.code += "\n";
            }
            result.readVars.insert(childResult.readVars.begin(), childResult.readVars.end());
            result.writtenVars.insert(childResult.writtenVars.begin(), childResult.writtenVars.end());
        }
        result.code += "}\n";
        return result;
    }

    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->isAssignmentOp() && opCall->getNumArgs() >= 2) {
            const clang::Expr* lhs = opCall->getArg(0);
            const clang::Expr* rhs = opCall->getArg(1);
            const auto* subscript =
                llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(
                    lhs ? lhs->IgnoreParenImpCasts() : nullptr);
            if (subscript && subscript->getOperator() == clang::OO_Subscript &&
                subscript->getNumArgs() >= 2) {
                const clang::Expr* baseExpr = subscript->getArg(0);
                const clang::Expr* indexExpr = subscript->getArg(1);
                const auto* baseDecl =
                    llvm::dyn_cast_or_null<clang::DeclRefExpr>(
                        baseExpr ? baseExpr->IgnoreParenImpCasts() : nullptr);
                if (baseDecl) {
                    auto nameIt = declToName.find(baseDecl->getDecl());
                    if (nameIt != declToName.end()) {
                        const std::string& paramName = nameIt->second;
                        auto rhsResult = rewriteMPISiblingBody(
                            rhs, context, declToName, writtenNames, loopVar, false);
                        result.readVars = rhsResult.readVars;
                        result.writtenVars = rhsResult.writtenVars;
                        result.writtenVars.insert(paramName);
                        result.code += "__dacpp_mpi_write_" + paramName + "(" +
                                       rewriteMPIIndexExpr(indexExpr, context, loopVar) +
                                       ", " + rhsResult.code + ");\n";
                        return result;
                    }
                }
            }
        }
    }

    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(
            llvm::dyn_cast<clang::Expr>(stmt)
                ? llvm::dyn_cast<clang::Expr>(stmt)->IgnoreParenImpCasts()
                : stmt)) {
        auto it = declToName.find(declRef->getDecl());
        if (it != declToName.end()) {
            result.readVars.insert(it->second);
            result.code = "__dacpp_mpi_read_" + it->second + "(0)";
            return result;
        }
    }

    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() >= 2) {
            const clang::Expr* baseExpr = opCall->getArg(0);
            const clang::Expr* indexExpr = opCall->getArg(1);
            const auto* baseDecl =
                llvm::dyn_cast_or_null<clang::DeclRefExpr>(
                    baseExpr ? baseExpr->IgnoreParenImpCasts() : nullptr);
            if (baseDecl) {
                auto nameIt = declToName.find(baseDecl->getDecl());
                if (nameIt != declToName.end()) {
                    const std::string& paramName = nameIt->second;
                    result.readVars.insert(paramName);
                    result.code = "__dacpp_mpi_read_" + paramName + "(" +
                                  rewriteMPIIndexExpr(indexExpr, context, loopVar) +
                                  ")";
                    return result;
                }
            }
        }
    }

    if (const auto* expr = llvm::dyn_cast<clang::Expr>(stmt)) {
        std::string text = getSourceTextMPI(expr, context);

        struct Replacement {
            unsigned beginOffset = 0;
            unsigned endOffset = 0;
            std::string text;
        };

        class ExprRewriteVisitor
            : public clang::RecursiveASTVisitor<ExprRewriteVisitor> {
        public:
            clang::ASTContext* Context;
            const std::unordered_map<const clang::ValueDecl*, std::string>& DeclToName;
            const std::set<std::string>& WrittenNames;
            const std::string& LoopVar;
            const clang::SourceLocation BaseLoc;
            std::vector<Replacement> Replacements;
            std::set<std::string>& ReadVars;

            ExprRewriteVisitor(
                clang::ASTContext* context,
                const std::unordered_map<const clang::ValueDecl*, std::string>& declToName,
                const std::set<std::string>& writtenNames,
                const std::string& loopVar,
                clang::SourceLocation baseLoc,
                std::set<std::string>& readVars)
                : Context(context), DeclToName(declToName), WrittenNames(writtenNames),
                  LoopVar(loopVar), BaseLoc(baseLoc), ReadVars(readVars) {}

            bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall) {
                if (!OpCall) {
                    return true;
                }
                if (OpCall->isAssignmentOp()) {
                    return true;
                }
                return clang::RecursiveASTVisitor<ExprRewriteVisitor>::TraverseCXXOperatorCallExpr(OpCall);
            }

            bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall) {
                if (!OpCall || OpCall->getOperator() != clang::OO_Subscript ||
                    OpCall->getNumArgs() < 2) {
                    return true;
                }

                const clang::Expr* baseExpr = OpCall->getArg(0);
                const clang::Expr* indexExpr = OpCall->getArg(1);
                const auto* baseDecl =
                    llvm::dyn_cast_or_null<clang::DeclRefExpr>(
                        baseExpr ? baseExpr->IgnoreParenImpCasts() : nullptr);
                if (!baseDecl) {
                    return true;
                }

                auto nameIt = DeclToName.find(baseDecl->getDecl());
                if (nameIt == DeclToName.end()) {
                    return true;
                }

                const std::string& paramName = nameIt->second;
                ReadVars.insert(paramName);
                const clang::SourceManager& SM = Context->getSourceManager();
                const unsigned beginOffset = SM.getFileOffset(OpCall->getBeginLoc()) -
                                             SM.getFileOffset(BaseLoc);
                const clang::SourceLocation endLoc =
                    clang::Lexer::getLocForEndOfToken(OpCall->getEndLoc(), 0, SM,
                                                      Context->getLangOpts());
                const unsigned endOffset = SM.getFileOffset(endLoc) -
                                           SM.getFileOffset(BaseLoc);
                Replacements.push_back(Replacement{
                    beginOffset,
                    endOffset,
                    "__dacpp_mpi_read_" + paramName + "(" +
                        rewriteMPIIndexExpr(indexExpr, Context, LoopVar) + ")"});
                return true;
            }
        };

        ExprRewriteVisitor visitor(context, declToName, writtenNames, loopVar,
                                   expr->getBeginLoc(), result.readVars);
        visitor.TraverseStmt(const_cast<clang::Expr*>(expr));
        std::sort(visitor.Replacements.begin(), visitor.Replacements.end(),
                  [](const Replacement& lhs, const Replacement& rhs) {
                      return lhs.beginOffset > rhs.beginOffset;
                  });
        for (const Replacement& replacement : visitor.Replacements) {
            if (replacement.endOffset <= text.size() &&
                replacement.beginOffset <= replacement.endOffset) {
                text.replace(replacement.beginOffset,
                             replacement.endOffset - replacement.beginOffset,
                             replacement.text);
            }
        }
        result.code = text;
        return result;
    }

    result.code = getSourceTextMPI(stmt, context);
    return result;
}

std::string getLoopBoundTextMPI(const clang::Expr* expr,
                                clang::ASTContext* context) {
    std::string text = getSourceTextMPI(expr, context);
    return text.empty() ? "0" : text;
}

bool parseForLoopBoundsMPI(const clang::ForStmt* forStmt,
                           clang::ASTContext* context,
                           std::string& loopVar,
                           std::string& beginExpr,
                           std::string& endExpr,
                           bool& inclusiveEnd) {
    if (!forStmt || !context) {
        return false;
    }

    if (const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(
            forStmt->getInit())) {
        if (const auto* varDecl =
                llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl())) {
            loopVar = varDecl->getNameAsString();
            beginExpr = varDecl->getInit()
                            ? getLoopBoundTextMPI(varDecl->getInit(), context)
                            : "0";
        }
    }

    const auto* cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(
        forStmt->getCond());
    if (!cond || !cond->isComparisonOp()) {
        return false;
    }

    const clang::Expr* lhs = cond->getLHS();
    const clang::Expr* rhs = cond->getRHS();
    const std::string lhsText = getLoopBoundTextMPI(lhs, context);
    const std::string rhsText = getLoopBoundTextMPI(rhs, context);
    if (loopVar.empty()) {
        loopVar = lhsText;
    }
    if (lhsText == loopVar) {
        endExpr = rhsText;
        inclusiveEnd = cond->getOpcode() == clang::BO_LE ||
                       cond->getOpcode() == clang::BO_GE;
    } else if (rhsText == loopVar) {
        endExpr = lhsText;
        inclusiveEnd = cond->getOpcode() == clang::BO_GE ||
                       cond->getOpcode() == clang::BO_LE;
    } else {
        return false;
    }

    return !loopVar.empty() && !beginExpr.empty() && !endExpr.empty();
}

std::string buildMPISiblingDenseSyncCode(Shell* shell,
                                         Calc* calc,
                                         const std::string& writtenName,
                                         const std::string& indent) {
    std::string code;
    int paramIdx = -1;
    for (int idx = 0; idx < shell->getNumShellParams(); ++idx) {
        if (calc->getParam(idx)->getName() == writtenName ||
            shell->getParam(idx)->getName() == writtenName) {
            paramIdx = idx;
            break;
        }
    }
    if (paramIdx < 0) {
        return code;
    }

    Param* calcParam = calc->getParam(paramIdx);
    const std::string baseType = calcParam->getBasicType();
    const std::string mpiType = mpiDatatypeFor(baseType);
    const std::string payloadLocalDense =
        mpiPayloadCountExpr("__dacpp_dense_count", baseType);
    const std::string payloadGlobalDense =
        mpiPayloadCountExpr("__dacpp_global_count", baseType);

    code += indent + "{\n";
    code += indent + "    std::vector<" + baseType + "> __dacpp_dense_" + writtenName +
            "(ctx.pattern_" + writtenName + ".data_info.dimLength.empty() ? 0 : 1);\n";
    code += indent + "    std::size_t __dacpp_dense_count_" + writtenName + " = 1;\n";
    code += indent + "    for (int __dacpp_dim : ctx.pattern_" + writtenName +
            ".data_info.dimLength) __dacpp_dense_count_" + writtenName +
            " *= static_cast<std::size_t>(__dacpp_dim);\n";
    code += indent + "    __dacpp_dense_" + writtenName + ".assign(__dacpp_dense_count_" +
            writtenName + ", " + baseType + "{});\n";
    code += indent + "    std::vector<unsigned char> __dacpp_mask_" + writtenName +
            "(__dacpp_dense_count_" + writtenName + ", 0);\n";
    code += indent + "    for (std::size_t __dacpp_l = 0; __dacpp_l < ctx.pack_" +
            writtenName + ".globals.size(); ++__dacpp_l) {\n";
    code += indent + "        const std::size_t __dacpp_g = static_cast<std::size_t>(ctx.pack_" +
            writtenName + ".globals[__dacpp_l]);\n";
    code += indent + "        if (__dacpp_g < __dacpp_dense_" + writtenName +
            ".size()) __dacpp_dense_" + writtenName + "[__dacpp_g] = ctx.local_" +
            writtenName + "[__dacpp_l];\n";
    code += indent + "    }\n";
    code += indent + "    for (int64_t __dacpp_g : ctx.mpi_sibling_written_" +
            writtenName + ") {\n";
    code += indent + "        if (__dacpp_g >= 0 && static_cast<std::size_t>(__dacpp_g) < __dacpp_mask_" +
            writtenName + ".size()) __dacpp_mask_" + writtenName +
            "[static_cast<std::size_t>(__dacpp_g)] = 1;\n";
    code += indent + "    }\n";
    code += indent + "    std::size_t __dacpp_global_count_" + writtenName + " = __dacpp_dense_count_" +
            writtenName + ";\n";
    code += indent + "    MPI_Datatype __dacpp_mpi_dt_" + writtenName + " = " + mpiType + ";\n";
    code += indent + "    MPI_Allreduce(MPI_IN_PLACE, __dacpp_dense_" + writtenName +
            ".data(), static_cast<int>(" +
            mpiPayloadCountExpr("__dacpp_global_count_" + writtenName, baseType) +
            "), __dacpp_mpi_dt_" + writtenName + ", MPI_SUM, MPI_COMM_WORLD);\n";
    code += indent + "    MPI_Allreduce(MPI_IN_PLACE, __dacpp_mask_" + writtenName +
            ".data(), static_cast<int>(__dacpp_global_count_" + writtenName +
            "), MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);\n";
    code += indent + "    for (std::size_t __dacpp_l = 0; __dacpp_l < ctx.pack_" +
            writtenName + ".globals.size(); ++__dacpp_l) {\n";
    code += indent + "        const std::size_t __dacpp_g = static_cast<std::size_t>(ctx.pack_" +
            writtenName + ".globals[__dacpp_l]);\n";
    code += indent + "        if (__dacpp_g < __dacpp_mask_" + writtenName +
            ".size() && __dacpp_mask_" + writtenName + "[__dacpp_g]) ctx.local_" +
            writtenName + "[__dacpp_l] = __dacpp_dense_" + writtenName +
            "[__dacpp_g];\n";
    code += indent + "    }\n";
    code += indent + "    ctx.mpi_sibling_written_" + writtenName + ".clear();\n";
    code += indent + "}\n";

    (void)payloadLocalDense;
    (void)payloadGlobalDense;
    return code;
}

MPIRegionGeneratedCode buildMPIRegionCode(
    DacppFile* dacppFile,
    Expression* expr) {
    MPIRegionGeneratedCode generated;

    Shell* shell = expr->getShell();
    Calc* calc = expr->getCalc();
    const std::string baseName = shell->getName() + "_" + calc->getName();
    const auto paramModes = inferEffectiveParamModes(shell, calc);
    const auto splitMeta = collectSplitBindMeta(shell);
    const auto transferPolicy =
        analyzeMPIRegionTransferPolicy(dacppFile, expr, paramModes);

    generated.ctxTypeName = "__dacpp_mpi_ctx_" + baseName;
    generated.ctxVarName = generated.ctxTypeName + "_0";
    generated.initName = "__dacpp_mpi_init_" + baseName;
    generated.submitName = "__dacpp_mpi_submit_" + baseName;
    generated.haloName = "__dacpp_mpi_halo_" + baseName;
    generated.syncName = "__dacpp_mpi_sync_" + baseName;

    std::string code;

    // === ctx struct ===
    code += "struct " + generated.ctxTypeName + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    sycl::queue q{};\n";
    code += "    int64_t total_items = 1;\n";
    code += "    int64_t local_item_count = 0;\n";
    code += "    std::vector<int64_t> binding_split_sizes;\n";
    code += "    bool has_halo = false;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& baseType = calcParam->getBasicType();
        code += "    dacpp::mpi::AccessPattern pattern_" + name + ";\n";
        code += "    dacpp::mpi::PackMap pack_" + name + ";\n";
        code += "    std::vector<int32_t> slots_" + name + ";\n";
        code += "    std::vector<" + baseType + "> local_" + name + ";\n";
        code += "    std::unique_ptr<sycl::buffer<" + baseType + ", 1>> buf_" + name + ";\n";
        code += "    std::unique_ptr<sycl::buffer<int32_t, 1>> slots_buf_" + name + ";\n";
        code += "    int " + name + "_partition_size = 0;\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    int " + name + "_cols = 0;\n";
        }
        code += "    dacpp::mpi::ParamHalo halo_" + name + ";\n";
    }
    code += "};\n\n";

    // === Shell parameter signature for init/sync ===
    std::string shellParamSig;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (paramType.size() > 0 && paramType.back() != '&' && paramType.back() != '*') {
            paramType += "&";
        }
        if (!shellParamSig.empty()) shellParamSig += ", ";
        shellParamSig += paramType + " " + param->getName();
    }

    // === init function ===
    code += "void " + generated.initName + "(" + generated.ctxTypeName + "& ctx";
    if (!shellParamSig.empty()) code += ", " + shellParamSig;
    code += ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.q = sycl::queue(sycl::default_selector_v);\n";
    // Pattern init (reuse buildPatternInitCode logic inline)
    // We reference the tensor params directly by name (they are function args)
    code += "    ctx.binding_split_sizes.clear();\n";
    // Build patterns using the same logic as buildPatternInitCode but referencing ctx members
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "ctx.pattern_" + name;
        const IOTYPE mode = paramModes[paramIdx];

        code += "    ctx.pattern_" + name + " = dacpp::mpi::AccessPattern();\n";
        code += "    ctx.pattern_" + name + ".param_id = " + std::to_string(paramIdx) + ";\n";
        code += "    ctx.pattern_" + name + ".name = \"" + name + "\";\n";
        code += "    ctx.pattern_" + name + ".mode = " + toPlannerMode(mode) + ";\n";
        code += "    ctx.pattern_" + name + ".data_info.dim = " + tensorName + ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName +
                ".getDim(); ++dim) ctx.pattern_" + name +
                ".data_info.dimLength.push_back(" + tensorName + ".getShape(dim));\n";

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") continue;

            auto metaIt = splitMeta.find(split->getId());
            SplitBindMeta bindMeta;
            if (metaIt != splitMeta.end()) bindMeta = metaIt->second;

            const bool isIndex = split->type == "IndexSplit";
            const std::string opName = "__dacpp_op_" + name + "_" + std::to_string(splitIdx);

            code += "    { Dac_Op " + opName + ";\n";
            code += "    " + opName + ".setDimId(" + std::to_string(split->getDimIdx()) + ");\n";
            code += "    " + opName + ".size = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitSize()) + ";\n";
            code += "    " + opName + ".stride = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitStride()) + ";\n";
            if (isIndex) {
                code += "    " + opName + ".SetSplitSize(" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) + "));\n";
            } else {
                code += "    " + opName + ".SetSplitSize((" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) + ") - " +
                        std::to_string(static_cast<RegularSplit*>(split)->getSplitSize()) + ") / " +
                        std::to_string(static_cast<RegularSplit*>(split)->getSplitStride()) + " + 1);\n";
            }
            code += "    ctx.pattern_" + name + ".param_ops.push_back(" + opName + ");\n";
            code += "    ctx.pattern_" + name + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    ctx.pattern_" + name + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    ctx.pattern_" + name + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + "); }\n";
        }

        code += "    ctx.pattern_" + name +
                ".partition_shape = dacpp::mpi::init_partition_shape(ctx.pattern_" + name + ");\n";
        code += "    ctx.pattern_" + name +
                ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(ctx.pattern_" + name + ");\n";
        code += "    if (ctx.binding_split_sizes.size() < ctx.pattern_" + name +
                ".bind_split_sizes.size()) ctx.binding_split_sizes.resize(ctx.pattern_" + name +
                ".bind_split_sizes.size(), 1);\n";
        code += "    for (std::size_t bind_i = 0; bind_i < ctx.pattern_" + name +
                ".bind_split_sizes.size(); ++bind_i) {\n";
        code += "        ctx.binding_split_sizes[bind_i] = std::max<int64_t>(ctx.binding_split_sizes[bind_i], ctx.pattern_" +
                name + ".bind_split_sizes[bind_i]);\n";
        code += "    }\n";
    }

    // total_items and item range
    code += "    ctx.total_items = 1;\n";
    code += "    for (int64_t split_size : ctx.binding_split_sizes) ctx.total_items *= split_size;\n";
    code += "    auto item_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.local_item_count = item_range.size();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& name = calc->getParam(paramIdx)->getName();
        code += "    ctx.pattern_" + name + ".bind_split_sizes = ctx.binding_split_sizes;\n";
    }

    // Build packs, slots, Scatterv for params that need init scatter
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const IOTYPE mode = paramModes[paramIdx];
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "ctx.pattern_" + calcName;
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());

        code += "    ctx.pack_" + calcName + " = " +
                buildPackBuilderExpr(mode, patternName) + ";\n";
        code += "    ctx.slots_" + calcName + " = dacpp::mpi::build_item_slots(item_range, " +
                patternName + ", ctx.pack_" + calcName + ");\n";
        code += "    ctx.local_" + calcName + ".resize(ctx.pack_" + calcName + ".globals.size());\n";

        // Scatterv for params that need data from root
        if (transferPolicy.needsInitScatter[static_cast<std::size_t>(paramIdx)]) {
            const std::string payloadRecvCount =
                mpiPayloadCountExpr("recv_count_" + calcName, calcParam->getBasicType());
            code += "    {\n";
            code += "    int recv_count_" + calcName + " = 0;\n";
            code += "    std::vector<int> sendcounts_" + calcName + ";\n";
            code += "    std::vector<int> displs_" + calcName + ";\n";
            code += "    std::vector<" + calcParam->getBasicType() + "> sendbuf_" + calcName + ";\n";
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        sendcounts_" + calcName + ".resize(ctx.mpi_size);\n";
            code += "        displs_" + calcName + ".resize(ctx.mpi_size);\n";
            code += "        int current_displ = 0;\n";
            code += "        std::vector<" + calcParam->getBasicType() + "> global_" + calcName + ";\n";
            code += "        " + tensorName + ".tensor2Array(global_" + calcName + ");\n";
            code += "        for (int r = 0; r < ctx.mpi_size; ++r) {\n";
            code += "            auto r_range = dacpp::mpi::get_rank_item_range(ctx.total_items, r, ctx.mpi_size);\n";
            code += "            auto r_pack = " + buildRemotePackBuilderExpr(mode, "r_range", patternName) + ";\n";
            code += "            auto r_values = dacpp::mpi::pack_values_by_globals(global_" + calcName + ", r_pack.globals);\n";
            code += "            int r_count = static_cast<int>(r_values.size());\n";
            code += "            sendcounts_" + calcName + "[r] = r_count;\n";
            code += "            displs_" + calcName + "[r] = current_displ;\n";
            code += "            current_displ += r_count;\n";
            code += "            sendbuf_" + calcName + ".insert(sendbuf_" + calcName + ".end(), r_values.begin(), r_values.end());\n";
            code += "        }\n";
            code += "    }\n";
            code += "    MPI_Scatter(ctx.mpi_rank == 0 ? sendcounts_" + calcName + ".data() : nullptr, 1, MPI_INT, &recv_count_" + calcName + ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
            code += "    ctx.local_" + calcName + ".resize(recv_count_" + calcName + ");\n";
            // Byte transport handling
            std::string payloadLocalRecv = mpiPayloadCountExpr("recv_count_" + calcName, calcParam->getBasicType());
            if (usesByteTransport(calcParam->getBasicType())) {
                code += "    { std::vector<int> sc_bytes = sendcounts_" + calcName + ";\n";
                code += "      std::vector<int> ds_bytes = displs_" + calcName + ";\n";
                code += "      if (ctx.mpi_rank == 0) { for (int r = 0; r < ctx.mpi_size; ++r) { sc_bytes[r] *= sizeof(" + calcParam->getBasicType() + "); ds_bytes[r] *= sizeof(" + calcParam->getBasicType() + "); } }\n";
                code += "      MPI_Scatterv(ctx.mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? sc_bytes.data() : nullptr, ctx.mpi_rank == 0 ? ds_bytes.data() : nullptr, " + mpiType + ", ctx.local_" + calcName + ".data(), " + payloadLocalRecv + ", " + mpiType + ", 0, MPI_COMM_WORLD); }\n";
            } else {
                code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? sendcounts_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? displs_" + calcName + ".data() : nullptr, " + mpiType + ", ctx.local_" + calcName + ".data(), " + payloadLocalRecv + ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
            code += "    }\n";
        }

        // Partition size
        code += "    ctx." + calcName + "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_" + calcName + "));\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    ctx." + calcName + "_cols = ctx.pattern_" + calcName + ".partition_shape[1];\n";
        }

        // Construct SYCL buffers
        code += "    if (ctx.local_item_count > 0) {\n";
        code += "        ctx.buf_" + calcName + " = std::make_unique<sycl::buffer<" + calcParam->getBasicType() + ", 1>>(ctx.local_" + calcName + ".data(), sycl::range<1>(ctx.local_" + calcName + ".size()));\n";
        code += "        ctx.slots_buf_" + calcName + " = std::make_unique<sycl::buffer<int32_t, 1>>(ctx.slots_" + calcName + ".data(), sycl::range<1>(ctx.slots_" + calcName + ".size()));\n";
        code += "    }\n";
    }

    // === Compute halo regions ===
    code += "    ctx.has_halo = (ctx.mpi_size > 1 && ctx.local_item_count > 0);\n";
    code += "    if (ctx.has_halo) {\n";
    code += "        auto my_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& name = calc->getParam(paramIdx)->getName();
        code += "        ctx.halo_" + name + " = dacpp::mpi::computeParamHalo(\n";
        code += "            ctx.pattern_" + name + ", ctx.pattern_" + name + ".mode,\n";
        code += "            my_range, ctx.total_items, ctx.mpi_rank, ctx.mpi_size,\n";
        code += "            ctx.pack_" + name + ");\n";
    }
    code += "    }\n";

    code += "}\n\n";

    // === submit function (kernel only, no Scatter/Gather, no wait) ===
    code += "void " + generated.submitName + "(" + generated.ctxTypeName + "& ctx) {\n";
    code += "    if (ctx.local_item_count <= 0) return;\n";
    code += "    sycl::queue& q = ctx.q;\n";
    code += "    const std::size_t local_item_count = static_cast<std::size_t>(ctx.local_item_count);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "    const int " + name + "_partition_size = ctx." + name + "_partition_size;\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + name + "_cols = ctx." + name + "_cols;\n";
        }
    }
    code += "    {\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "            auto acc_" + name + " = ctx.buf_" + name + "->get_access<" +
                toAccessorMode(paramModes[paramIdx]) + ">(h);\n";
        code += "        auto slots_acc_" + name +
                " = ctx.slots_buf_" + name + "->get_access<sycl::access::mode::read>(h);\n";
    }
    code += "            h.parallel_for(sycl::range<1>(local_item_count), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto* data_" + name +
                " = acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                auto* slots_" + name +
                " = slots_acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (inferViewRank(shellParam, calcParam) <= 1) {
            code += "                " + toViewType(shellParam, calcParam, paramModes[paramIdx]) + " view_" + name +
                    "{data_" + name + ", slots_" + name + ", item_linear * " + name +
                    "_partition_size};\n";
        } else {
            code += "                " + toViewType(shellParam, calcParam, paramModes[paramIdx]) + " view_" + name +
                    "{data_" + name + ", slots_" + name + ", item_linear * " + name +
                    "_partition_size, " + name + "_cols};\n";
        }
    }
    code += "                " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += "view_" + calc->getParam(paramIdx)->getName();
        if (paramIdx + 1 != calc->getNumParams()) code += ", ";
    }
    code += ");\n";
    code += "            });\n";
    code += "        });\n";
    code += "    }\n";
    code += "}\n\n";

    // === halo exchange function ===
    code += "void " + generated.haloName + "(" + generated.ctxTypeName + "& ctx) {\n";
    code += "    ctx.q.wait();\n";
    code += "    if (!ctx.has_halo) return;\n";
    code += "    if (ctx.local_item_count <= 0) return;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const IOTYPE mode = paramModes[paramIdx];
        if (mode == IOTYPE::READ) continue;  // READ-only params don't need halo exchange
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& baseType = calcParam->getBasicType();
        const std::string mpiType = mpiDatatypeFor(baseType);

        // D2H: read from SYCL buffer into local array using host accessor
        code += "    {\n";
        code += "        sycl::host_accessor ha_" + name + "(*ctx.buf_" + name +
                ", sycl::read_only);\n";
        code += "        for (std::size_t i = 0; i < ctx.local_" + name + ".size(); ++i)\n";
        code += "            ctx.local_" + name + "[i] = ha_" + name + "[i];\n";
        // Halo exchange: per-param, per-neighbor
        code += "        MPI_Datatype mpi_dt_" + name + " = " + mpiType + ";\n";
        code += "        dacpp::mpi::exchangeHalo(ctx.local_" + name +
                ", ctx.halo_" + name + ", &mpi_dt_" + name + ");\n";
        // H2D: write updated halo data back to SYCL buffer
        code += "        sycl::host_accessor ha_w_" + name + "(*ctx.buf_" + name +
                ", sycl::write_only);\n";
        code += "        for (std::size_t i = 0; i < ctx.local_" + name + ".size(); ++i)\n";
        code += "            ha_w_" + name + "[i] = ctx.local_" + name + "[i];\n";
        code += "    }\n";
    }
    code += "}\n\n";
    code += "void " + generated.syncName + "(" + generated.ctxTypeName + "& ctx";
    if (!shellParamSig.empty()) code += ", " + shellParamSig;
    code += ") {\n";
    code += "    ctx.q.wait();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (!transferPolicy.needsSyncGather[static_cast<std::size_t>(paramIdx)]) continue;

        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());
        bool needsBcast = tensorNeedsBroadcast(dacppFile, tensorName);

        code += "    {\n";
        code += "    auto wb_" + calcName + " = dacpp::mpi::build_writeback_values(ctx.local_" + calcName + ", ctx.pack_" + calcName + ");\n";
        code += "    const auto& wb_globals_" + calcName + " = ctx.pack_" + calcName + ".writeback_globals.empty() ? ctx.pack_" + calcName + ".globals : ctx.pack_" + calcName + ".writeback_globals;\n";
        if (needsBcast) {
            code += "    std::vector<" + calcParam->getBasicType() + "> synced_" + calcName + ";\n";
        }
        code += "    int send_count_" + calcName + " = static_cast<int>(wb_globals_" + calcName + ".size());\n";
        code += "    std::vector<int> recvcounts_" + calcName + ";\n";
        code += "    std::vector<int> recvdispls_" + calcName + ";\n";
        code += "    std::vector<int64_t> global_recv_globals_" + calcName + ";\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> global_recv_values_" + calcName + ";\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        recvcounts_" + calcName + ".resize(ctx.mpi_size);\n";
        code += "        recvdispls_" + calcName + ".resize(ctx.mpi_size);\n";
        code += "    }\n";
        code += "    MPI_Gather(&send_count_" + calcName + ", 1, MPI_INT, ctx.mpi_rank == 0 ? recvcounts_" + calcName + ".data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        int current_displ = 0;\n";
        code += "        for (int r = 0; r < ctx.mpi_size; ++r) {\n";
        code += "            recvdispls_" + calcName + "[r] = current_displ;\n";
        code += "            current_displ += recvcounts_" + calcName + "[r];\n";
        code += "        }\n";
        code += "        global_recv_globals_" + calcName + ".resize(current_displ);\n";
        code += "        global_recv_values_" + calcName + ".resize(current_displ);\n";
        code += "    }\n";
        code += "    MPI_Gatherv(const_cast<int64_t*>(wb_globals_" + calcName + ".data()), send_count_" + calcName + ", MPI_LONG_LONG, ctx.mpi_rank == 0 ? global_recv_globals_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? recvcounts_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? recvdispls_" + calcName + ".data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);\n";
        // Byte transport handling for gatherv
        {
            const std::string payloadSendCount =
                mpiPayloadCountExpr("send_count_" + calcName, calcParam->getBasicType());
            if (usesByteTransport(calcParam->getBasicType())) {
                code += "    { std::vector<int> rc_bytes = recvcounts_" + calcName + ";\n";
                code += "      std::vector<int> rd_bytes = recvdispls_" + calcName + ";\n";
                code += "      if (ctx.mpi_rank == 0) { for (int r = 0; r < ctx.mpi_size; ++r) { rc_bytes[r] *= sizeof(" + calcParam->getBasicType() + "); rd_bytes[r] *= sizeof(" + calcParam->getBasicType() + "); } }\n";
                code += "      MPI_Gatherv(wb_" + calcName + ".data(), " + payloadSendCount + ", " + mpiType + ", ctx.mpi_rank == 0 ? global_recv_values_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? rc_bytes.data() : nullptr, ctx.mpi_rank == 0 ? rd_bytes.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD); }\n";
            } else {
                code += "    MPI_Gatherv(wb_" + calcName + ".data(), " + payloadSendCount + ", " + mpiType + ", ctx.mpi_rank == 0 ? global_recv_values_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? recvcounts_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? recvdispls_" + calcName + ".data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
        }
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        std::vector<" + calcParam->getBasicType() + "> global_out_" + calcName + ";\n";
        code += "        " + tensorName + ".tensor2Array(global_out_" + calcName + ");\n";
        code += "        dacpp::mpi::apply_writeback_by_globals(global_recv_values_" + calcName + ", global_recv_globals_" + calcName + ", global_out_" + calcName + ");\n";
        code += "        " + tensorName + ".array2Tensor(global_out_" + calcName + ");\n";
        if (needsBcast) {
            code += "        synced_" + calcName + " = global_out_" + calcName + ";\n";
        }
        code += "    }\n";
        if (needsBcast) {
            const std::string payloadSyncedCount =
                mpiPayloadCountExpr("synced_count_" + calcName, calcParam->getBasicType());
            code += "    int synced_count_" + calcName + " = 0;\n";
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        synced_count_" + calcName + " = static_cast<int>(synced_" + calcName + ".size());\n";
            code += "    }\n";
            code += "    MPI_Bcast(&synced_count_" + calcName + ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
            code += "    if (ctx.mpi_rank != 0) {\n";
            code += "        synced_" + calcName + ".resize(synced_count_" + calcName + ");\n";
            code += "    }\n";
            code += "    if (synced_count_" + calcName + " > 0) {\n";
            code += "        MPI_Bcast(synced_" + calcName + ".data(), " + payloadSyncedCount + ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            code += "    }\n";
            code += "    if (ctx.mpi_rank != 0) {\n";
            code += "        " + tensorName + ".array2Tensor(synced_" + calcName + ");\n";
            code += "    }\n";
        }
        code += "    }\n";
    }
    code += "}\n\n";

    generated.definitions = code;
    return generated;
}

std::string joinShellCallArgsMPI(const clang::BinaryOperator* dacExpr,
                                  clang::ASTContext* context) {
    if (!dacExpr || !context) return "";
    clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    auto* shellCall = dacppTranslator::getNode<clang::CallExpr>(shellExpr);
    if (!shellCall) return "";
    const auto& SM = context->getSourceManager();
    const auto& LO = context->getLangOpts();
    std::string args;
    for (unsigned argIdx = 0; argIdx < shellCall->getNumArgs(); ++argIdx) {
        if (!args.empty()) args += ", ";
        args += clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(shellCall->getArg(argIdx)->getSourceRange()),
            SM, LO).str();
    }
    return args;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
