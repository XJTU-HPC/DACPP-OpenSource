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

// WIP helpers for future sibling-loop lowering on top of the MPI region state.
// They are compiled so the split stays build-checked, but the current rewrite
// path still conservatively falls back before these helpers are used.

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
               context->getSourceManager(), context->getLangOpts())
        .str();
}

std::string getSourceTextMPI(const clang::Expr* expr,
                             clang::ASTContext* context) {
    if (!expr || !context) {
        return "";
    }
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(expr->getSourceRange()),
               context->getSourceManager(), context->getLangOpts())
        .str();
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
            return clang::RecursiveASTVisitor<WriteVisitor>::TraverseBinaryOperator(
                BO);
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
            return clang::RecursiveASTVisitor<WriteVisitor>::TraverseUnaryOperator(
                UO);
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
                for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs();
                     ++argIdx) {
                    TraverseStmt(OpCall->getArg(argIdx));
                }
                return true;
            }
            return clang::RecursiveASTVisitor<WriteVisitor>::TraverseCXXOperatorCallExpr(
                OpCall);
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
            result.readVars.insert(childResult.readVars.begin(),
                                   childResult.readVars.end());
            result.writtenVars.insert(childResult.writtenVars.begin(),
                                      childResult.writtenVars.end());
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
                            rhs, context, declToName, writtenNames, loopVar,
                            false);
                        result.readVars = rhsResult.readVars;
                        result.writtenVars = rhsResult.writtenVars;
                        result.writtenVars.insert(paramName);
                        result.code += "__dacpp_mpi_write_" + paramName + "(" +
                                       rewriteMPIIndexExpr(indexExpr, context,
                                                           loopVar) +
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
                                  rewriteMPIIndexExpr(indexExpr, context,
                                                      loopVar) +
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
            const std::unordered_map<const clang::ValueDecl*, std::string>&
                DeclToName;
            const std::set<std::string>& WrittenNames;
            const std::string& LoopVar;
            const clang::SourceLocation BaseLoc;
            std::vector<Replacement> Replacements;
            std::set<std::string>& ReadVars;

            ExprRewriteVisitor(
                clang::ASTContext* context,
                const std::unordered_map<const clang::ValueDecl*, std::string>&
                    declToName,
                const std::set<std::string>& writtenNames,
                const std::string& loopVar,
                clang::SourceLocation baseLoc,
                std::set<std::string>& readVars)
                : Context(context),
                  DeclToName(declToName),
                  WrittenNames(writtenNames),
                  LoopVar(loopVar),
                  BaseLoc(baseLoc),
                  ReadVars(readVars) {}

            bool TraverseCXXOperatorCallExpr(
                clang::CXXOperatorCallExpr* OpCall) {
                if (!OpCall) {
                    return true;
                }
                if (OpCall->isAssignmentOp()) {
                    return true;
                }
                return clang::RecursiveASTVisitor<
                    ExprRewriteVisitor>::TraverseCXXOperatorCallExpr(OpCall);
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
                const unsigned beginOffset =
                    SM.getFileOffset(OpCall->getBeginLoc()) -
                    SM.getFileOffset(BaseLoc);
                const clang::SourceLocation endLoc =
                    clang::Lexer::getLocForEndOfToken(OpCall->getEndLoc(), 0,
                                                      SM, Context->getLangOpts());
                const unsigned endOffset =
                    SM.getFileOffset(endLoc) - SM.getFileOffset(BaseLoc);
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

    if (const auto* declStmt =
            llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit())) {
        if (const auto* varDecl =
                llvm::dyn_cast_or_null<clang::VarDecl>(
                    declStmt->getSingleDecl())) {
            loopVar = varDecl->getNameAsString();
            beginExpr =
                varDecl->getInit()
                    ? getLoopBoundTextMPI(varDecl->getInit(), context)
                    : "0";
        }
    }

    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
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

    code += indent + "{\n";
    code += indent + "    std::vector<" + baseType + "> __dacpp_dense_" +
            writtenName +
            "(ctx.pattern_" + writtenName + ".data_info.dimLength.empty() ? 0 : 1);\n";
    code += indent + "    std::size_t __dacpp_dense_count_" + writtenName +
            " = 1;\n";
    code += indent + "    for (int __dacpp_dim : ctx.pattern_" + writtenName +
            ".data_info.dimLength) __dacpp_dense_count_" + writtenName +
            " *= static_cast<std::size_t>(__dacpp_dim);\n";
    code += indent + "    __dacpp_dense_" + writtenName +
            ".assign(__dacpp_dense_count_" + writtenName + ", " + baseType +
            "{});\n";
    code += indent + "    std::vector<unsigned char> __dacpp_mask_" +
            writtenName + "(__dacpp_dense_count_" + writtenName + ", 0);\n";
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
    code += indent + "    std::size_t __dacpp_global_count_" + writtenName +
            " = __dacpp_dense_count_" + writtenName + ";\n";
    code += indent + "    MPI_Datatype __dacpp_mpi_dt_" + writtenName +
            " = " + mpiType + ";\n";
    code += indent + "    MPI_Allreduce(MPI_IN_PLACE, __dacpp_dense_" +
            writtenName + ".data(), static_cast<int>(" +
            mpiPayloadCountExpr("__dacpp_global_count_" + writtenName, baseType) +
            "), __dacpp_mpi_dt_" + writtenName +
            ", MPI_SUM, MPI_COMM_WORLD);\n";
    code += indent + "    MPI_Allreduce(MPI_IN_PLACE, __dacpp_mask_" +
            writtenName + ".data(), static_cast<int>(__dacpp_global_count_" +
            writtenName + "), MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);\n";
    code += indent + "    for (std::size_t __dacpp_l = 0; __dacpp_l < ctx.pack_" +
            writtenName + ".globals.size(); ++__dacpp_l) {\n";
    code += indent + "        const std::size_t __dacpp_g = static_cast<std::size_t>(ctx.pack_" +
            writtenName + ".globals[__dacpp_l]);\n";
    code += indent + "        if (__dacpp_g < __dacpp_mask_" + writtenName +
            ".size() && __dacpp_mask_" + writtenName + "[__dacpp_g]) ctx.local_" +
            writtenName + "[__dacpp_l] = __dacpp_dense_" + writtenName +
            "[__dacpp_g];\n";
    code += indent + "    }\n";
    code += indent + "    ctx.mpi_sibling_written_" + writtenName +
            ".clear();\n";
    code += indent + "}\n";
    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
