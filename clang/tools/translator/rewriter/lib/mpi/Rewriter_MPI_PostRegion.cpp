#include <regex>
#include <set>
#include <string>
#include <vector>

#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

std::string getStmtSourceText(const clang::Stmt* stmt,
                              clang::ASTContext* context) {
    if (!stmt || !context) {
        return "";
    }
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(stmt->getSourceRange()),
               context->getSourceManager(),
               context->getLangOpts())
        .str();
}

std::string trim(std::string text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string stripOuterBraces(std::string text) {
    text = trim(std::move(text));
    if (text.size() >= 2 && text.front() == '{' && text.back() == '}') {
        text = text.substr(1, text.size() - 2);
    }
    return trim(std::move(text));
}

std::string stripComments(const std::string& text) {
    std::string withoutBlock;
    withoutBlock.reserve(text.size());
    bool inBlockComment = false;
    for (std::size_t idx = 0; idx < text.size(); ++idx) {
        if (!inBlockComment && idx + 1 < text.size() &&
            text[idx] == '/' && text[idx + 1] == '*') {
            inBlockComment = true;
            ++idx;
            continue;
        }
        if (inBlockComment && idx + 1 < text.size() &&
            text[idx] == '*' && text[idx + 1] == '/') {
            inBlockComment = false;
            ++idx;
            continue;
        }
        if (!inBlockComment) {
            withoutBlock.push_back(text[idx]);
        }
    }

    std::string result;
    result.reserve(withoutBlock.size());
    for (std::size_t idx = 0; idx < withoutBlock.size(); ++idx) {
        if (idx + 1 < withoutBlock.size() &&
            withoutBlock[idx] == '/' && withoutBlock[idx + 1] == '/') {
            while (idx < withoutBlock.size() && withoutBlock[idx] != '\n') {
                ++idx;
            }
            if (idx < withoutBlock.size()) {
                result.push_back(withoutBlock[idx]);
            }
            continue;
        }
        result.push_back(withoutBlock[idx]);
    }
    return result;
}

bool containsWord(const std::string& text, const std::string& word) {
    if (word.empty()) {
        return false;
    }
    return std::regex_search(text, std::regex("\\b" + word + "\\b"));
}

std::string replaceWord(std::string text,
                        const std::string& word,
                        const std::string& replacement) {
    if (word.empty()) {
        return text;
    }
    return std::regex_replace(text, std::regex("\\b" + word + "\\b"), replacement);
}

bool isVectorParam(Param* param) {
    return param && param->getType().find("Vector<") != std::string::npos;
}

struct LoopRegionInfo {
    std::string loopVar;
    std::string lowerExpr;
    std::string upperExpr;
    bool upperInclusive = true;
    std::string bodyText;
    std::set<std::string> readTensors;
    std::set<std::string> writtenTensors;
};

bool isSupportedIncrement(const clang::ForStmt* forStmt) {
    const auto* inc = forStmt ? forStmt->getInc() : nullptr;
    if (!inc) {
        return false;
    }

    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(inc)) {
        return unary->isIncrementOp();
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(inc)) {
        return opCall->getOperator() == clang::OO_PlusPlus;
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(inc)) {
        if (!binary->isAssignmentOp()) {
            return false;
        }
        const std::string text =
            binary->getOpcodeStr().str();
        return text == "+=" || text == "=";
    }
    return false;
}

bool extractLoopRegionInfo(const clang::ForStmt* forStmt,
                           clang::ASTContext* context,
                           Shell* shell,
                           const BufferRegionPlan& plan,
                           LoopRegionInfo& info) {
    if (!forStmt || !context || !shell || !plan.capturedNonShellVars.empty()) {
        return false;
    }

    const auto& SM = context->getSourceManager();
    const auto& LO = context->getLangOpts();

    const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
    if (!declStmt || !declStmt->isSingleDecl()) {
        return false;
    }
    const auto* loopVarDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    if (!loopVarDecl || !loopVarDecl->getInit()) {
        return false;
    }
    info.loopVar = loopVarDecl->getNameAsString();
    info.lowerExpr =
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(loopVarDecl->getInit()->getSourceRange()),
            SM, LO)
            .str();

    const auto* cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
    if (!cond) {
        return false;
    }
    const std::string lhsText =
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(cond->getLHS()->getSourceRange()),
            SM, LO)
            .str();
    if (trim(lhsText) != info.loopVar) {
        return false;
    }
    if (cond->getOpcode() == clang::BO_LE) {
        info.upperInclusive = true;
    } else if (cond->getOpcode() == clang::BO_LT) {
        info.upperInclusive = false;
    } else {
        return false;
    }
    info.upperExpr =
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(cond->getRHS()->getSourceRange()),
            SM, LO)
            .str();

    if (!isSupportedIncrement(forStmt)) {
        return false;
    }

    info.bodyText = stripOuterBraces(
        stripComments(getStmtSourceText(forStmt->getBody(), context)));
    if (info.bodyText.empty()) {
        return false;
    }

    const std::vector<std::string> rejected = {
        "for", "while", "switch", "return", "break", "continue", "goto",
        "MPI_", "std::", "cout", "cerr", "printf"
    };
    for (const auto& token : rejected) {
        if (containsWord(info.bodyText, token)) {
            return false;
        }
    }

    const std::size_t eqPos = info.bodyText.find('=');
    if (eqPos == std::string::npos ||
        info.bodyText.find('=', eqPos + 1) != std::string::npos) {
        return false;
    }
    if ((eqPos > 0 &&
         (info.bodyText[eqPos - 1] == '+' || info.bodyText[eqPos - 1] == '-' ||
          info.bodyText[eqPos - 1] == '*' || info.bodyText[eqPos - 1] == '/' ||
          info.bodyText[eqPos - 1] == '%' || info.bodyText[eqPos - 1] == '=' ||
          info.bodyText[eqPos - 1] == '!' || info.bodyText[eqPos - 1] == '<' ||
          info.bodyText[eqPos - 1] == '>')) ||
        (eqPos + 1 < info.bodyText.size() && info.bodyText[eqPos + 1] == '=')) {
        return false;
    }

    const std::string lhs = info.bodyText.substr(0, eqPos);
    const std::string rhs = info.bodyText.substr(eqPos + 1);

    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        const std::string name = shell->getParam(paramIdx)->getName();
        if (containsWord(lhs, name)) {
            if (!isVectorParam(shell->getParam(paramIdx))) {
                return false;
            }
            info.writtenTensors.insert(name);
        }
        if (containsWord(rhs, name)) {
            if (!isVectorParam(shell->getParam(paramIdx))) {
                return false;
            }
            info.readTensors.insert(name);
        }
    }

    return !info.writtenTensors.empty() &&
           (!info.readTensors.empty() ||
            info.bodyText.find('[') != std::string::npos);
}

bool isRootCentricRegionSupported(DacppFile* dacppFile,
                                  Shell* shell,
                                  const clang::Stmt* stmt) {
    if (!dacppFile || !shell || !stmt) {
        return false;
    }
    const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(stmt);
    if (!forStmt) {
        return false;
    }
    LoopRegionInfo info;
    return extractLoopRegionInfo(
        forStmt, dacppFile->getContext(), shell,
        dacppFile->getBufferRegionPlan(), info);
}

std::string helperBaseName(Shell* shell, Calc* calc) {
    return shell->getName() + "_" + calc->getName();
}

std::string helperNameFor(Shell* shell, Calc* calc, std::size_t stmtIdx) {
    return "__dacpp_mpi_region_" + helperBaseName(shell, calc) +
           "_stmt_" + std::to_string(stmtIdx);
}

std::string accessModeFor(const std::string& name,
                          const LoopRegionInfo& info) {
    return info.writtenTensors.count(name) != 0
               ? "sycl::access::mode::read_write"
               : "sycl::access::mode::read";
}

std::string buildLoopRegionHelper(DacppFile* dacppFile,
                                  Shell* shell,
                                  Calc* calc,
                                  const clang::ForStmt* forStmt,
                                  std::size_t stmtIdx,
                                  const std::string& ctxTypeName,
                                  const std::string& shellSignature) {
    LoopRegionInfo info;
    if (!extractLoopRegionInfo(
            forStmt, dacppFile->getContext(), shell,
            dacppFile->getBufferRegionPlan(), info)) {
        return "";
    }

    std::set<std::string> usedTensors = info.readTensors;
    usedTensors.insert(info.writtenTensors.begin(), info.writtenTensors.end());

    std::string kernelBody = info.bodyText;
    for (const std::string& name : usedTensors) {
        kernelBody = replaceWord(kernelBody, name, "acc_" + name);
    }

    std::string code;
    code += "void " + helperNameFor(shell, calc, stmtIdx) + "(" + ctxTypeName +
            "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    if (ctx.mpi_rank != 0) {\n";
    code += "        return;\n";
    code += "    }\n";
    code += "    auto& q = *ctx.q;\n";

    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        const std::string& name = param->getName();
        if (usedTensors.count(name) == 0) {
            continue;
        }
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        code += "    std::vector<" + shellParam->getBasicType() + "> h_" + name +
                ";\n";
        code += "    " + name + ".tensor2Array(h_" + name + ");\n";
    }

    code += "    {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        const std::string& name = param->getName();
        if (usedTensors.count(name) == 0) {
            continue;
        }
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        code += "        sycl::buffer<" + shellParam->getBasicType() +
                ", 1> b_" + name + "(h_" + name +
                ".data(), sycl::range<1>(h_" + name + ".size()));\n";
    }
    code += "        int __L = (" + info.lowerExpr + ");\n";
    code += "        int __R = (" + info.upperExpr + ");\n";
    code += "        int __N = " +
            std::string(info.upperInclusive ? "(__R - __L + 1)" : "(__R - __L)") +
            ";\n";
    code += "        if (__N > 0) {\n";
    code += "            q.submit([&](sycl::handler& h) {\n";
    for (const std::string& name : usedTensors) {
        code += "                auto acc_" + name + " = b_" + name +
                ".get_access<" + accessModeFor(name, info) + ">(h);\n";
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__N)), [=](sycl::id<1> idx) {\n";
    code += "                    int " + info.loopVar +
            " = __L + static_cast<int>(idx[0]);\n";
    code += "                    " + kernelBody + "\n";
    code += "                });\n";
    code += "            });\n";
    code += "            q.wait();\n";
    code += "        }\n";
    code += "    }\n";

    for (const std::string& name : info.writtenTensors) {
        code += "    " + name + ".array2Tensor(h_" + name + ");\n";
    }
    code += "}\n\n";
    return code;
}

}  // namespace

std::vector<RootCentricPostRegion> collectRootCentricPostRegions(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    std::vector<RootCentricPostRegion> result;
    if (!dacppFile || !shell || !calc || !dacExpr) {
        return result;
    }

    const auto& plan = dacppFile->getBufferRegionPlan();
    if (!plan.enabled || plan.dacExpr != dacExpr) {
        return result;
    }

    for (std::size_t stmtIdx = 0; stmtIdx < plan.siblingStmts.size(); ++stmtIdx) {
        const clang::Stmt* stmt = plan.siblingStmts[stmtIdx];
        if (!isRootCentricRegionSupported(dacppFile, shell, stmt)) {
            continue;
        }
        result.push_back({stmt, helperNameFor(shell, calc, stmtIdx)});
    }
    return result;
}

std::vector<const clang::Stmt*> collectRootCentricPostRegionStmts(
    DacppFile* dacppFile,
    const clang::BinaryOperator* dacExpr) {
    std::vector<const clang::Stmt*> result;
    if (!dacppFile || !dacExpr || !dacppFile->getContext()) {
        return result;
    }

    const auto& plan = dacppFile->getBufferRegionPlan();
    if (!plan.enabled || plan.dacExpr != dacExpr) {
        return result;
    }

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        if (!expr || expr->getDacExpr() != dacExpr) {
            continue;
        }
        Shell* shell = expr->getShell();
        for (const clang::Stmt* stmt : plan.siblingStmts) {
            if (isRootCentricRegionSupported(dacppFile, shell, stmt)) {
                result.push_back(stmt);
            }
        }
        break;
    }

    return result;
}

std::string buildRootCentricPostRegionHelpers(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr,
    const std::string& ctxTypeName,
    const std::string& shellSignature) {
    if (!dacppFile || !shell || !calc || !dacExpr) {
        return "";
    }

    const auto& plan = dacppFile->getBufferRegionPlan();
    if (!plan.enabled || plan.dacExpr != dacExpr) {
        return "";
    }

    std::string code;
    for (std::size_t stmtIdx = 0; stmtIdx < plan.siblingStmts.size(); ++stmtIdx) {
        const auto* forStmt =
            llvm::dyn_cast_or_null<clang::ForStmt>(plan.siblingStmts[stmtIdx]);
        if (!forStmt) {
            continue;
        }
        code += buildLoopRegionHelper(
            dacppFile, shell, calc, forStmt, stmtIdx, ctxTypeName,
            shellSignature);
    }
    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
