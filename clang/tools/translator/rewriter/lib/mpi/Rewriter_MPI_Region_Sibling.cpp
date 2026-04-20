#include <algorithm>
#include <cctype>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Casting.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

struct MPISiblingRewriteResult {
    std::string code;
    std::set<std::string> readVars;
    std::set<std::string> writtenVars;
};

std::string getSourceTextMPI(const clang::Stmt* stmt,
                             clang::ASTContext* context);
std::string getSourceTextMPI(const clang::Expr* expr,
                             clang::ASTContext* context);
bool parseForLoopBoundsMPI(const clang::ForStmt* forStmt,
                           clang::ASTContext* context,
                           std::string& loopVar,
                           std::string& beginExpr,
                           std::string& endExpr,
                           bool& inclusiveEnd);

namespace {

std::string replaceWordMPI(std::string text,
                           const std::string& word,
                           const std::string& replacement) {
    if (word.empty()) {
        return text;
    }
    return std::regex_replace(text, std::regex("\\b" + word + "\\b"),
                              replacement);
}

int inferSiblingViewRank(ShellParam* shellParam, Param* calcParam) {
    if (shellParam && shellParam->getDimension() > 0) {
        return shellParam->getDimension();
    }
    return inferViewRank(shellParam, calcParam);
}

std::string buildDenseViewType(ShellParam* shellParam,
                               Param* calcParam,
                               bool isConst) {
    const std::string baseType = calcParam->getBasicType();
    const std::string qualifiedType = isConst ? ("const " + baseType) : baseType;
    const int shellRank = inferSiblingViewRank(shellParam, calcParam);
    if (shellRank <= 1) {
        return "dacpp::mpi::DenseVectorView<" + qualifiedType + ">";
    }
    return "dacpp::mpi::DenseMatrixView<" + qualifiedType + ">";
}

std::string buildDenseToLocalCopy(const std::string& calcName,
                                  const std::string& indent,
                                  bool onlyDirty) {
    std::string code;
    code += indent +
            "for (std::size_t __dacpp_l = 0; __dacpp_l < ctx.state_" +
            calcName + ".runtime_pack.globals.size(); ++__dacpp_l) {\n";
    code += indent +
            "    const std::size_t __dacpp_g = static_cast<std::size_t>(ctx.state_" +
            calcName + ".runtime_pack.globals[__dacpp_l]);\n";
    code += indent + "    if (__dacpp_g < __dacpp_dense_" + calcName +
            ".size()";
    if (onlyDirty) {
        code += " && __dacpp_dirty_" + calcName + "[__dacpp_g]";
    }
    code += ") ctx.state_" + calcName + ".local[__dacpp_l] = __dacpp_dense_" +
            calcName + "[__dacpp_g];\n";
    code += indent + "}\n";
    return code;
}

std::string buildLocalToDenseCopy(const std::string& calcName,
                                  const std::string& indent) {
    std::string code;
    code += indent +
            "for (std::size_t __dacpp_l = 0; __dacpp_l < ctx.state_" +
            calcName + ".runtime_pack.globals.size(); ++__dacpp_l) {\n";
    code += indent +
            "    const std::size_t __dacpp_g = static_cast<std::size_t>(ctx.state_" +
            calcName + ".runtime_pack.globals[__dacpp_l]);\n";
    code += indent + "    if (__dacpp_g < __dacpp_dense_" + calcName +
            ".size()) __dacpp_dense_" + calcName +
            "[__dacpp_g] = ctx.state_" + calcName + ".local[__dacpp_l];\n";
    code += indent + "}\n";
    return code;
}

std::string buildBufferToLocalCopy(Param* calcParam,
                                   const std::string& calcName,
                                   const std::string& indent) {
    std::string code;
    code += indent + "dacpp::mpi::sync_buffer_to_local(ctx.state_" + calcName +
            ");\n";
    (void)calcParam;
    return code;
}

std::string buildLocalToBufferCopy(Param* calcParam,
                                   const std::string& calcName,
                                   const std::string& indent) {
    std::string code;
    code += indent + "dacpp::mpi::sync_local_to_buffer(ctx.state_" + calcName +
            ");\n";
    (void)calcParam;
    return code;
}

std::string buildPackedViewType(ShellParam* shellParam,
                                Param* calcParam,
                                bool isConst) {
    const std::string baseType = calcParam->getBasicType();
    const std::string qualifiedType = isConst ? ("const " + baseType) : baseType;
    const int shellRank = inferSiblingViewRank(shellParam, calcParam);
    if (shellRank <= 1) {
        return "dacpp::mpi::PackedVectorView<" + qualifiedType + ">";
    }
    return "dacpp::mpi::PackedMatrixView<" + qualifiedType + ">";
}

std::string buildSiblingLoopDeviceCode(
    const clang::ForStmt* siblingFor,
    clang::ASTContext* context,
    const std::vector<std::string>& sourceNames,
    Shell* shell,
    Calc* calc,
    const std::vector<AccessSummary>& siblingSummary) {
    if (!siblingFor || !context || !shell || !calc) {
        return "";
    }

    std::string loopVar;
    std::string beginExpr;
    std::string endExpr;
    bool inclusiveEnd = false;
    if (!parseForLoopBoundsMPI(siblingFor, context, loopVar, beginExpr, endExpr,
                               inclusiveEnd)) {
        return "";
    }

    const clang::Stmt* loopBody = siblingFor->getBody();
    if (!loopBody) {
        return "";
    }
    std::string bodyText = getSourceTextMPI(loopBody, context);
    if (bodyText.empty()) {
        return "";
    }
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string shellName = shell->getParam(paramIdx)->getName();
        const std::string calcName = calc->getParam(paramIdx)->getName();
        const std::string& sourceName =
            sourceNames[static_cast<std::size_t>(paramIdx)];
        if (shellName != sourceName) {
            bodyText = replaceWordMPI(bodyText, shellName, sourceName);
        }
        if (calcName != sourceName) {
            bodyText = replaceWordMPI(bodyText, calcName, sourceName);
        }
    }
    bodyText = replaceWordMPI(bodyText, loopVar, "__dacpp_i");

    std::string code;
    code += "        const int __dacpp_begin = static_cast<int>(" + beginExpr + ");\n";
    code += "        const int __dacpp_end = static_cast<int>(" + endExpr + ");\n";
    code += "        const int __dacpp_extent = (__dacpp_end - __dacpp_begin) + " +
            std::string(inclusiveEnd ? "1" : "0") + ";\n";
    code += "        ctx.q.wait();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const AccessSummary& access =
            siblingSummary[static_cast<std::size_t>(paramIdx)];
        if (!access.reads && !access.writes) {
            continue;
        }
        code += "        dacpp::mpi::sync_buffer_to_local(ctx.state_" + calcName +
                ");\n";
    }
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const AccessSummary& access =
            siblingSummary[static_cast<std::size_t>(paramIdx)];
        if (!access.reads && !access.writes) {
            continue;
        }
        code += "        const std::size_t __dacpp_lookup_size_" + calcName +
                " = ctx.state_" + calcName + ".global_to_local.size();\n";
        code += "        std::vector<" + calcParam->getBasicType() + "> __dacpp_dense_read_" +
                calcName + "(__dacpp_lookup_size_" + calcName + ", " +
                calcParam->getBasicType() + "{});\n";
        if (access.writes) {
            code += "        std::vector<" + calcParam->getBasicType() + "> __dacpp_dense_shadow_" +
                    calcName + "(__dacpp_lookup_size_" + calcName + ", " +
                    calcParam->getBasicType() + "{});\n";
        }
        code += "        for (int64_t __dacpp_g64 : ctx.state_" + calcName +
                ".runtime_pack.globals) {\n";
        code += "            const std::size_t __dacpp_g = static_cast<std::size_t>(__dacpp_g64);\n";
        code += "            const int32_t __dacpp_slot = (__dacpp_g < __dacpp_lookup_size_" +
                calcName + ") ? ctx.state_" + calcName + ".global_to_local" +
                "[__dacpp_g] : -1;\n";
        code += "            if (__dacpp_g < __dacpp_lookup_size_" + calcName +
                " && __dacpp_slot >= 0) {\n";
        code += "                __dacpp_dense_read_" + calcName +
                "[__dacpp_g] = ctx.state_" + calcName +
                ".local[static_cast<std::size_t>(__dacpp_slot)];\n";
        if (access.writes) {
            code += "                __dacpp_dense_shadow_" + calcName +
                    "[__dacpp_g] = __dacpp_dense_read_" + calcName +
                    "[__dacpp_g];\n";
        }
        code += "            }\n";
        code += "        }\n";
        code += "        MPI_Datatype __dacpp_mpi_dt_read_" + calcName +
                " = " + mpiDatatypeFor(calcParam->getBasicType()) + ";\n";
        code += "        std::vector<int> __dacpp_present_read_" + calcName +
                "(__dacpp_lookup_size_" + calcName + ", 0);\n";
        code += "        for (int64_t __dacpp_g64 : ctx.state_" + calcName +
                ".runtime_pack.globals) {\n";
        code += "            const std::size_t __dacpp_g = static_cast<std::size_t>(__dacpp_g64);\n";
        code += "            if (__dacpp_g < __dacpp_lookup_size_" + calcName +
                ") __dacpp_present_read_" + calcName + "[__dacpp_g] = 1;\n";
        code += "        }\n";
        code += "        if (__dacpp_lookup_size_" + calcName + " > 0) {\n";
        code += "            MPI_Allreduce(MPI_IN_PLACE, __dacpp_dense_read_" + calcName +
                ".data(), " +
                mpiPayloadCountExpr("__dacpp_lookup_size_" + calcName,
                                    calcParam->getBasicType()) +
                ", __dacpp_mpi_dt_read_" + calcName +
                ", MPI_SUM, MPI_COMM_WORLD);\n";
        code += "            MPI_Allreduce(MPI_IN_PLACE, __dacpp_present_read_" + calcName +
                ".data(), static_cast<int>(__dacpp_lookup_size_" + calcName +
                "), MPI_INT, MPI_SUM, MPI_COMM_WORLD);\n";
        code += "            for (std::size_t __dacpp_g = 0; __dacpp_g < __dacpp_lookup_size_" +
                calcName + "; ++__dacpp_g) {\n";
        code += "                if (__dacpp_present_read_" + calcName +
                "[__dacpp_g] > 1) {\n";
        code += "                    __dacpp_dense_read_" + calcName +
                "[__dacpp_g] = __dacpp_dense_read_" + calcName +
                "[__dacpp_g] / static_cast<" + calcParam->getBasicType() +
                ">(__dacpp_present_read_" + calcName + "[__dacpp_g]);\n";
        code += "                }\n";
        code += "            }\n";
        code += "        }\n";
        if (access.writes) {
            code += "        std::vector<unsigned char> __dacpp_dirty_" + calcName +
                    "(__dacpp_lookup_size_" + calcName + ", 0);\n";
            code += "        sycl::buffer<unsigned char, 1> __dacpp_dirty_buf_" +
                    calcName + "(__dacpp_dirty_" + calcName +
                    ".data(), sycl::range<1>(__dacpp_lookup_size_" + calcName +
                    "));\n";
        }
        code += "        sycl::buffer<" + calcParam->getBasicType() +
                ", 1> __dacpp_dense_read_buf_" + calcName + "(__dacpp_dense_read_" +
                calcName + ".data(), sycl::range<1>(__dacpp_lookup_size_" +
                calcName + "));\n";
        if (access.writes) {
            code += "        sycl::buffer<" + calcParam->getBasicType() +
                    ", 1> __dacpp_dense_shadow_buf_" + calcName +
                    "(__dacpp_dense_shadow_" + calcName +
                    ".data(), sycl::range<1>(__dacpp_lookup_size_" + calcName +
                    "));\n";
        }
        if (inferSiblingViewRank(shell->getShellParam(paramIdx), calcParam) > 1) {
            code += "        const int __dacpp_dense_cols_" + calcName +
                    " = ctx.state_" + calcName +
                    ".pattern.data_info.dimLength.size() > 1 ? ctx.state_" +
                    calcName + ".pattern.data_info.dimLength[1] : 1;\n";
        }
    }
    code += "        if (__dacpp_extent > 0) {\n";
    code += "            ctx.q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const AccessSummary& access =
            siblingSummary[static_cast<std::size_t>(paramIdx)];
        if (!access.reads && !access.writes) {
            continue;
        }
        if (access.writes) {
            code += "                auto acc_" + calcName + " = ctx.state_" +
                    calcName +
                    ".buf->get_access<sycl::access::mode::read_write>(h);\n";
        }
        code += "                auto lookup_acc_" + calcName +
                " = ctx.state_" + calcName + ".global_to_local_buf" +
                "->get_access<sycl::access::mode::read>(h);\n";
        code += "                auto dense_read_acc_" + calcName +
                " = __dacpp_dense_read_buf_" + calcName +
                ".template get_access<sycl::access::mode::read>(h);\n";
        if (access.writes) {
            code += "                auto dense_shadow_acc_" + calcName +
                    " = __dacpp_dense_shadow_buf_" + calcName +
                    ".template get_access<sycl::access::mode::read_write>(h);\n";
        }
        if (access.writes) {
            code += "                auto dirty_acc_" + calcName +
                    " = __dacpp_dirty_buf_" + calcName +
                    ".template get_access<sycl::access::mode::read_write>(h);\n";
        }
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__dacpp_extent)), [=](sycl::id<1> idx) {\n";
    code += "                    const int __dacpp_i = __dacpp_begin + static_cast<int>(idx[0]);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& sourceName = sourceNames[static_cast<std::size_t>(paramIdx)];
        const std::string shellName = shell->getParam(paramIdx)->getName();
        const AccessSummary& access =
            siblingSummary[static_cast<std::size_t>(paramIdx)];
        if (!access.reads && !access.writes) {
            continue;
        }

        const bool isConstView = !access.writes;
        if (access.writes) {
            code += "                    auto* data_" + calcName + " = acc_" + calcName +
                    ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        } else {
            code += "                    " + calcParam->getBasicType() +
                    " const* data_" + calcName + " = nullptr;\n";
        }
        code += "                    auto* lookup_" + calcName +
                " = lookup_acc_" + calcName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* dense_read_" + calcName +
                " = dense_read_acc_" + calcName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (access.writes) {
            code += "                    auto* dense_shadow_" + calcName +
                    " = dense_shadow_acc_" + calcName +
                    ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        }
        if (access.writes) {
            code += "                    auto* dirty_" + calcName +
                    " = dirty_acc_" + calcName +
                    ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        }
        if (inferSiblingViewRank(shellParam, calcParam) <= 1) {
            code += "                    " +
                    buildPackedViewType(shellParam, calcParam, isConstView) +
                    " " + sourceName + "(data_" + calcName + ", lookup_" +
                    calcName + ", __dacpp_lookup_size_" + calcName;
            if (access.writes) {
                code += ", dirty_" + calcName + ", dense_read_" + calcName +
                        ", dense_shadow_" + calcName;
            } else {
                code += ", nullptr, dense_read_" + calcName;
            }
            code += ");\n";
        } else {
            code += "                    " +
                    buildPackedViewType(shellParam, calcParam, isConstView) +
                    " " + sourceName + "(data_" + calcName + ", lookup_" +
                    calcName + ", __dacpp_lookup_size_" + calcName +
                    ", __dacpp_dense_cols_" + calcName;
            if (access.writes) {
                code += ", dirty_" + calcName + ", dense_read_" + calcName +
                        ", dense_shadow_" + calcName;
            } else {
                code += ", nullptr, dense_read_" + calcName;
            }
            code += ");\n";
        }
        if (shellName != sourceName) {
            code += "                    auto& " + shellName + " = " +
                    sourceName + ";\n";
        }
        if (calcName != sourceName && calcName != shellName) {
            code += "                    auto& " + calcName + " = " +
                    sourceName + ";\n";
        }
    }
    code += "                    " + bodyText + "\n";
    code += "                });\n";
    code += "            });\n";
    code += "        }\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const AccessSummary& access =
            siblingSummary[static_cast<std::size_t>(paramIdx)];
        if (!access.writes) {
            continue;
        }
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        code += "        {\n";
        code += "            sycl::host_accessor __dacpp_dirty_sync_" + calcName +
                "(__dacpp_dirty_buf_" + calcName + ", sycl::read_only);\n";
        code += "            (void)__dacpp_dirty_sync_" + calcName + ";\n";
        code += "        }\n";
        code += "        {\n";
        code += "            sycl::host_accessor __dacpp_shadow_sync_" + calcName +
                "(__dacpp_dense_shadow_buf_" + calcName + ", sycl::read_only);\n";
        code += "            (void)__dacpp_shadow_sync_" + calcName + ";\n";
        code += "        }\n";
    }
    return code;
}

}  // namespace

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

    if (const auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
        result.code += "if (";
        auto condResult = rewriteMPISiblingBody(
            ifStmt->getCond(), context, declToName, writtenNames, loopVar, false);
        result.code += condResult.code;
        result.code += ") ";
        result.readVars = condResult.readVars;
        result.writtenVars = condResult.writtenVars;

        auto thenResult = rewriteMPISiblingBody(
            ifStmt->getThen(), context, declToName, writtenNames, loopVar, false);
        result.code += thenResult.code;
        result.readVars.insert(thenResult.readVars.begin(),
                               thenResult.readVars.end());
        result.writtenVars.insert(thenResult.writtenVars.begin(),
                                  thenResult.writtenVars.end());

        if (ifStmt->getElse()) {
            result.code += " else ";
            auto elseResult = rewriteMPISiblingBody(
                ifStmt->getElse(), context, declToName, writtenNames, loopVar, false);
            result.code += elseResult.code;
            result.readVars.insert(elseResult.readVars.begin(),
                                   elseResult.readVars.end());
            result.writtenVars.insert(elseResult.writtenVars.begin(),
                                      elseResult.writtenVars.end());
        }
        result.code += "\n";
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

std::string buildMPISiblingSparseSyncCode(Shell* shell,
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
    code += indent + "    std::vector<int32_t> __dacpp_written_idx_" +
            writtenName + ";\n";
    code += indent + "    __dacpp_written_idx_" + writtenName +
            ".reserve(__dacpp_dirty_" + writtenName + ".size());\n";
    code += indent + "    for (std::size_t __dacpp_g = 0; __dacpp_g < __dacpp_dirty_" +
            writtenName + ".size(); ++__dacpp_g) {\n";
    code += indent + "        if (__dacpp_dirty_" + writtenName + "[__dacpp_g]) {\n";
    code += indent + "            __dacpp_written_idx_" + writtenName +
            ".push_back(static_cast<int32_t>(__dacpp_g));\n";
    code += indent + "        }\n";
    code += indent + "    }\n";
    code += indent + "    std::sort(__dacpp_written_idx_" + writtenName +
            ".begin(), __dacpp_written_idx_" + writtenName + ".end());\n";
    code += indent + "    __dacpp_written_idx_" + writtenName +
            ".erase(std::unique(__dacpp_written_idx_" + writtenName +
            ".begin(), __dacpp_written_idx_" + writtenName +
            ".end()), __dacpp_written_idx_" + writtenName + ".end());\n";
    code += indent + "    const int __dacpp_send_count_" + writtenName +
            " = static_cast<int>(__dacpp_written_idx_" + writtenName +
            ".size());\n";
    code += indent + "    std::vector<int> __dacpp_all_counts_" + writtenName +
            "(ctx.mpi_size, 0);\n";
    code += indent + "    MPI_Allgather(&__dacpp_send_count_" + writtenName +
            ", 1, MPI_INT, __dacpp_all_counts_" + writtenName +
            ".data(), 1, MPI_INT, MPI_COMM_WORLD);\n";
    code += indent + "    std::vector<int> __dacpp_all_displs_" + writtenName +
            "(ctx.mpi_size, 0);\n";
    code += indent + "    int __dacpp_total_written_" + writtenName +
            " = 0;\n";
    code += indent + "    for (int __dacpp_r = 0; __dacpp_r < ctx.mpi_size; ++__dacpp_r) {\n";
    code += indent + "        __dacpp_all_displs_" + writtenName +
            "[__dacpp_r] = __dacpp_total_written_" + writtenName + ";\n";
    code += indent + "        __dacpp_total_written_" + writtenName +
            " += __dacpp_all_counts_" + writtenName + "[__dacpp_r];\n";
    code += indent + "    }\n";
    code += indent + "    std::vector<int32_t> __dacpp_all_written_idx_" +
            writtenName + "(static_cast<std::size_t>(__dacpp_total_written_" +
            writtenName + "));\n";
    code += indent + "    if (__dacpp_total_written_" + writtenName +
            " > 0) {\n";
    code += indent + "        MPI_Allgatherv(__dacpp_written_idx_" + writtenName +
            ".data(), __dacpp_send_count_" + writtenName +
            ", MPI_INT32_T, __dacpp_all_written_idx_" + writtenName +
            ".data(), __dacpp_all_counts_" + writtenName +
            ".data(), __dacpp_all_displs_" + writtenName +
            ".data(), MPI_INT32_T, MPI_COMM_WORLD);\n";
    code += indent + "    }\n";
    code += indent + "    std::sort(__dacpp_all_written_idx_" + writtenName +
            ".begin(), __dacpp_all_written_idx_" + writtenName + ".end());\n";
    code += indent + "    __dacpp_all_written_idx_" + writtenName +
            ".erase(std::unique(__dacpp_all_written_idx_" + writtenName +
            ".begin(), __dacpp_all_written_idx_" + writtenName +
            ".end()), __dacpp_all_written_idx_" + writtenName + ".end());\n";
    code += indent + "    const std::size_t __dacpp_sync_count_" + writtenName +
            " = __dacpp_all_written_idx_" + writtenName + ".size();\n";
    code += indent + "    std::vector<" + baseType + "> __dacpp_sparse_values_" +
            writtenName + "(__dacpp_sync_count_" + writtenName + ", " +
            baseType + "{});\n";
    code += indent + "    std::vector<int> __dacpp_sparse_present_" + writtenName +
            "(__dacpp_sync_count_" + writtenName + ", 0);\n";
    code += indent + "    for (std::size_t __dacpp_i = 0; __dacpp_i < __dacpp_sync_count_" +
            writtenName + "; ++__dacpp_i) {\n";
    code += indent + "        const std::size_t __dacpp_g = static_cast<std::size_t>(__dacpp_all_written_idx_" +
            writtenName + "[__dacpp_i]);\n";
    code += indent + "        if (__dacpp_g < __dacpp_dense_shadow_" + writtenName +
            ".size()) {\n";
    code += indent + "            __dacpp_sparse_values_" + writtenName +
            "[__dacpp_i] = __dacpp_dense_shadow_" + writtenName +
            "[__dacpp_g];\n";
    code += indent + "            __dacpp_sparse_present_" + writtenName +
            "[__dacpp_i] = 1;\n";
    code += indent + "        }\n";
    code += indent + "    }\n";
    code += indent + "    MPI_Datatype __dacpp_mpi_dt_" + writtenName +
            " = " + mpiType + ";\n";
    code += indent + "    if (__dacpp_sync_count_" + writtenName + " > 0) {\n";
    code += indent + "        MPI_Allreduce(MPI_IN_PLACE, __dacpp_sparse_values_" +
            writtenName + ".data(), " +
            mpiPayloadCountExpr("__dacpp_sync_count_" + writtenName, baseType) +
            ", __dacpp_mpi_dt_" + writtenName +
            ", MPI_SUM, MPI_COMM_WORLD);\n";
    code += indent + "        MPI_Allreduce(MPI_IN_PLACE, __dacpp_sparse_present_" +
            writtenName + ".data(), static_cast<int>(__dacpp_sync_count_" +
            writtenName + "), MPI_INT, MPI_SUM, MPI_COMM_WORLD);\n";
    code += indent + "        for (std::size_t __dacpp_i = 0; __dacpp_i < __dacpp_sync_count_" +
            writtenName + "; ++__dacpp_i) {\n";
    code += indent + "            if (__dacpp_sparse_present_" + writtenName +
            "[__dacpp_i] > 1) {\n";
    code += indent + "                __dacpp_sparse_values_" + writtenName +
            "[__dacpp_i] = __dacpp_sparse_values_" + writtenName +
            "[__dacpp_i] / static_cast<" + baseType + ">(__dacpp_sparse_present_" +
            writtenName + "[__dacpp_i]);\n";
    code += indent + "            }\n";
    code += indent + "        }\n";
    code += indent + "        for (std::size_t __dacpp_i = 0; __dacpp_i < __dacpp_sync_count_" +
            writtenName + "; ++__dacpp_i) {\n";
    code += indent + "            const std::size_t __dacpp_g = static_cast<std::size_t>(__dacpp_all_written_idx_" +
            writtenName + "[__dacpp_i]);\n";
    code += indent + "            const int32_t __dacpp_slot = (__dacpp_g < ctx.state_" +
            writtenName + ".global_to_local.size()) ? ctx.state_" +
            writtenName + ".global_to_local[__dacpp_g] : -1;\n";
    code += indent + "            if (__dacpp_slot >= 0) {\n";
    code += indent + "                ctx.state_" + writtenName +
            ".local[static_cast<std::size_t>(__dacpp_slot)] = __dacpp_sparse_values_" +
            writtenName + "[__dacpp_i];\n";
    code += indent + "            }\n";
    code += indent + "        }\n";
    code += indent + "    }\n";
    code += indent + "}\n";
    return code;
}

std::string buildMPIRegionSiblingCode(DacppFile* dacppFile,
                                      Expression* expr,
                                      MPIRegionGeneratedCode& generated) {
    if (!dacppFile || !expr || !expr->getShell() || !expr->getCalc()) {
        return "";
    }

    const auto& plan = dacppFile->getBufferRegionPlan();
    if (!plan.enabled || plan.siblingStmts.empty()) {
        return "";
    }

    Shell* shell = expr->getShell();
    Calc* calc = expr->getCalc();
    clang::ASTContext* context = dacppFile->getContext();
    if (!context) {
        return "";
    }

    const std::string baseName = shell->getName() + "_" + calc->getName();
    std::vector<std::string> sourceNames(
        static_cast<std::size_t>(shell->getNumShellParams()));

    const clang::CallExpr* shellCall = nullptr;
    if (plan.dacExpr) {
        clang::Expr* shellExpr =
            dacppTranslator::Expression::shellLHS_p(plan.dacExpr)
                ? plan.dacExpr->getLHS()
                : plan.dacExpr->getRHS();
        shellCall = dacppTranslator::getNode<clang::CallExpr>(shellExpr);
    }

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        std::string sourceName = shell->getParam(paramIdx)->getName();
        if (shellCall && paramIdx < static_cast<int>(shellCall->getNumArgs())) {
            const clang::Expr* arg = shellCall->getArg(static_cast<unsigned>(paramIdx));
            if (const auto* declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
                    arg ? arg->IgnoreParenImpCasts() : nullptr)) {
                sourceName = declRef->getDecl()->getNameAsString();
            } else {
                const std::string argText = getSourceTextMPI(arg, context);
                if (!argText.empty() &&
                    std::regex_match(argText,
                                     std::regex("[A-Za-z_][A-Za-z0-9_]*"))) {
                    sourceName = argText;
                }
            }
        }
        sourceNames[static_cast<std::size_t>(paramIdx)] = sourceName;
    }

    std::string code;
    std::unordered_map<const clang::ValueDecl*, int> argDeclIndices;
    if (shellCall) {
        for (int paramIdx = 0;
             paramIdx < std::min<int>(shell->getNumShellParams(),
                                      static_cast<int>(shellCall->getNumArgs()));
             ++paramIdx) {
            const clang::Expr* arg = shellCall->getArg(static_cast<unsigned>(paramIdx));
            if (const auto* declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
                    arg ? arg->IgnoreParenImpCasts() : nullptr)) {
                argDeclIndices.emplace(declRef->getDecl(), paramIdx);
            }
        }
    }
    for (std::size_t helperIdx = 0; helperIdx < plan.siblingStmts.size();
         ++helperIdx) {
        const clang::Stmt* siblingStmt = plan.siblingStmts[helperIdx];
        if (!siblingStmt) {
            continue;
        }

        const clang::ForStmt* siblingFor = llvm::dyn_cast_or_null<clang::ForStmt>(siblingStmt);

        const std::string helperName = "__dacpp_mpi_submit_region_" +
                                       baseName + "_stmt_" +
                                       std::to_string(helperIdx);
        generated.siblingHelpers.emplace_back(siblingStmt, helperName);

        code += "void " + helperName + "(" + generated.ctxTypeName +
                "& ctx) {\n";
        const auto siblingSummary =
            summarizeStmtAccess(siblingStmt, argDeclIndices,
                                shell->getNumShellParams());
        // Create local aliases for non-shell captured variables from ctx
        for (const auto& var : plan.capturedNonShellVars) {
            code += "    " + var.second + " " + var.first + " = ctx." +
                    var.first + ";\n";
        }

        bool emittedDeviceLoop = false;
        if (siblingFor) {
            const std::string deviceLoopCode = buildSiblingLoopDeviceCode(
                siblingFor, context, sourceNames, shell, calc, siblingSummary);
            if (!deviceLoopCode.empty()) {
                code += deviceLoopCode;
                emittedDeviceLoop = true;
            }
        }

        if (!emittedDeviceLoop) {
            code += "    ctx.q.wait();\n";
            for (int paramIdx = 0; paramIdx < shell->getNumShellParams();
                 ++paramIdx) {
                ShellParam* shellParam = shell->getShellParam(paramIdx);
                Param* calcParam = calc->getParam(paramIdx);
                const std::string calcName = calcParam->getName();
                const std::string sourceName =
                    sourceNames[static_cast<std::size_t>(paramIdx)];
                const std::string shellName = shell->getParam(paramIdx)->getName();
                const std::string baseType = calcParam->getBasicType();
                const std::string mpiType = mpiDatatypeFor(baseType);

                code += buildBufferToLocalCopy(calcParam, calcName, "    ");
                code += "    std::size_t __dacpp_dense_count_" + calcName +
                        " = 1;\n";
                code += "    for (int __dacpp_dim : ctx.state_" + calcName +
                        ".pattern.data_info.dimLength) __dacpp_dense_count_" +
                        calcName +
                        " *= static_cast<std::size_t>(__dacpp_dim);\n";
                code += "    std::vector<" + baseType + "> __dacpp_dense_" +
                        calcName + "(__dacpp_dense_count_" + calcName + ", " +
                        baseType + "{});\n";
                code += "    std::vector<int> __dacpp_present_" + calcName +
                        "(__dacpp_dense_count_" + calcName + ", 0);\n";
                code += buildLocalToDenseCopy(calcName, "    ");
                code += "    for (std::size_t __dacpp_l = 0; __dacpp_l < ctx.state_" +
                        calcName + ".runtime_pack.globals.size(); ++__dacpp_l) {\n";
                code += "        const std::size_t __dacpp_g = static_cast<std::size_t>(ctx.state_" +
                        calcName + ".runtime_pack.globals[__dacpp_l]);\n";
                code += "        if (__dacpp_g < __dacpp_present_" + calcName +
                        ".size()) __dacpp_present_" + calcName +
                        "[__dacpp_g] = 1;\n";
                code += "    }\n";
                code += "    MPI_Datatype __dacpp_mpi_dt_" + calcName + " = " +
                        mpiType + ";\n";
                code += "    if (__dacpp_dense_count_" + calcName + " > 0) {\n";
                code += "        MPI_Allreduce(MPI_IN_PLACE, __dacpp_dense_" +
                        calcName + ".data(), " +
                        mpiPayloadCountExpr("__dacpp_dense_count_" + calcName,
                                            baseType) +
                        ", __dacpp_mpi_dt_" + calcName +
                        ", MPI_SUM, MPI_COMM_WORLD);\n";
                code += "        MPI_Allreduce(MPI_IN_PLACE, __dacpp_present_" +
                        calcName + ".data(), static_cast<int>(__dacpp_dense_count_" +
                        calcName + "), MPI_INT, MPI_SUM, MPI_COMM_WORLD);\n";
                code += "        for (std::size_t __dacpp_g = 0; __dacpp_g < __dacpp_dense_" +
                        calcName + ".size(); ++__dacpp_g) {\n";
                code += "            if (__dacpp_present_" + calcName +
                        "[__dacpp_g] > 1) __dacpp_dense_" + calcName +
                        "[__dacpp_g] = __dacpp_dense_" + calcName +
                        "[__dacpp_g] / static_cast<" + baseType +
                        ">(__dacpp_present_" + calcName + "[__dacpp_g]);\n";
                code += "        }\n";
                code += "    }\n";
                code += "    " + buildDenseViewType(shellParam, calcParam, false) +
                        " " + sourceName + "(__dacpp_dense_" + calcName +
                        ", ctx.state_" + calcName +
                        ".pattern.data_info.dimLength);\n";
                if (shellName != sourceName) {
                    code += "    auto& " + shellName + " = " + sourceName +
                            ";\n";
                }
                if (calcName != sourceName && calcName != shellName) {
                    code += "    auto& " + calcName + " = " + sourceName +
                            ";\n";
                }
            }

            std::string stmtText = getSourceTextMPI(siblingStmt, context);
            for (int paramIdx = 0; paramIdx < shell->getNumShellParams();
                 ++paramIdx) {
                const std::string shellName = shell->getParam(paramIdx)->getName();
                const std::string calcName = calc->getParam(paramIdx)->getName();
                const std::string sourceName =
                    sourceNames[static_cast<std::size_t>(paramIdx)];
                if (shellName != sourceName) {
                    stmtText = replaceWordMPI(stmtText, shellName, sourceName);
                }
                if (calcName != sourceName) {
                    stmtText = replaceWordMPI(stmtText, calcName, sourceName);
                }
            }
            code += "    " + stmtText + "\n";

            for (int paramIdx = 0; paramIdx < shell->getNumShellParams();
                 ++paramIdx) {
                Param* calcParam = calc->getParam(paramIdx);
                const std::string calcName = calcParam->getName();
                code += buildDenseToLocalCopy(calcName, "    ", false);
                code += buildLocalToBufferCopy(calcParam, calcName, "    ");
            }
        } else {
            code += "    ctx.q.wait();\n";
            for (int paramIdx = 0; paramIdx < shell->getNumShellParams();
                 ++paramIdx) {
                const AccessSummary& access =
                    siblingSummary[static_cast<std::size_t>(paramIdx)];
                if (!access.reads && !access.writes) {
                    continue;
                }
                Param* calcParam = calc->getParam(paramIdx);
                const std::string calcName = calcParam->getName();
                code += "    dacpp::mpi::sync_buffer_to_local(ctx.state_" +
                        calcName + ");\n";
                if (!access.writes) {
                    continue;
                }
                code += buildMPISiblingSparseSyncCode(shell, calc, calcName,
                                                      "    ");
                code += "    dacpp::mpi::sync_local_to_buffer(ctx.state_" +
                        calcName + ");\n";
            }
        }

        code += "}\n\n";
    }

    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
