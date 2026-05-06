#include <string>
#include <unordered_map>
#include <vector>

#include "clang/Lex/Lexer.h"

#include "StencilAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace detail {

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

bool isWordBoundary(char c) {
    return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_');
}

bool containsWord(const std::string& text, const std::string& word) {
    if (word.empty() || text.empty() || word.size() > text.size()) {
        return false;
    }
    std::size_t pos = 0;
    while ((pos = text.find(word, pos)) != std::string::npos) {
        const bool leftOk = pos == 0 || isWordBoundary(text[pos - 1]);
        const std::size_t rightIdx = pos + word.size();
        const bool rightOk =
            rightIdx >= text.size() || isWordBoundary(text[rightIdx]);
        if (leftOk && rightOk) {
            return true;
        }
        ++pos;
    }
    return false;
}

Split* findSingleNonVoidSplit(ShellParam* shellParam) {
    if (!shellParam) {
        return nullptr;
    }

    Split* result = nullptr;
    for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
        Split* split = shellParam->getSplit(splitIdx);
        if (!split || split->getId() == "void") {
            continue;
        }
        if (result) {
            return nullptr;
        }
        result = split;
    }
    return result;
}

bool isStrideOneInputDomainSplit(Split* split) {
    if (!split || split->getId() == "void") {
        return false;
    }
    if (split->type == "IndexSplit") {
        return true;
    }
    auto* regular = static_cast<RegularSplit*>(split);
    return regular && regular->getSplitStride() == 1;
}

}  // namespace

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

std::string resolveActualTensorName(const std::string& shellParamName,
                                    const clang::BinaryOperator* dacExpr) {
    const clang::CallExpr* shellCall = getShellCallExpr(dacExpr);
    if (!shellCall) {
        return shellParamName;
    }

    const clang::FunctionDecl* callee = shellCall->getDirectCallee();
    if (!callee) {
        return shellParamName;
    }

    for (unsigned paramIdx = 0;
         paramIdx < callee->getNumParams() && paramIdx < shellCall->getNumArgs();
         ++paramIdx) {
        const clang::ParmVarDecl* param = callee->getParamDecl(paramIdx);
        if (!param || param->getNameAsString() != shellParamName) {
            continue;
        }

        const auto* dre = dacppTranslator::getNode<clang::DeclRefExpr>(
            const_cast<clang::Expr*>(shellCall->getArg(paramIdx)));
        if (dre && dre->getDecl()) {
            return dre->getDecl()->getNameAsString();
        }
    }
    return shellParamName;
}

bool isVectorParam(Shell* shell, Calc* calc, int paramIdx) {
    if (!shell || !calc || paramIdx < 0 ||
        paramIdx >= shell->getNumShellParams() ||
        paramIdx >= calc->getNumParams()) {
        return false;
    }
    const std::string shellType = shell->getParam(paramIdx)->getType();
    const std::string calcType = calc->getParam(paramIdx)->getType();
    return shellType.find("Vector<") != std::string::npos ||
           calcType.find("Vector<") != std::string::npos;
}

bool isMatrixParam(Shell* shell, Calc* calc, int paramIdx) {
    if (!shell || !calc || paramIdx < 0 ||
        paramIdx >= shell->getNumShellParams() ||
        paramIdx >= calc->getNumParams()) {
        return false;
    }
    const std::string shellType = shell->getParam(paramIdx)->getType();
    const std::string calcType = calc->getParam(paramIdx)->getType();
    return shellType.find("Matrix<") != std::string::npos ||
           calcType.find("Matrix<") != std::string::npos;
}

void collectRootBridgeTensors(DistributedStencilSitePlan& plan,
                              DacppFile* dacppFile,
                              Shell* shell) {
    if (!dacppFile || !shell || !dacppFile->getContext()) {
        return;
    }
    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
        const std::string stmtText = getStmtSourceText(stmt, dacppFile->getContext());
        for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
            const std::string paramName = shell->getParam(paramIdx)->getName();
            if (!containsWord(stmtText, paramName)) {
                continue;
            }
            plan.rootBridgeTensors.insert(paramName);
        }
    }
}

bool isEffectiveWriter(const std::string& tensorName,
                       Shell* shell,
                       const std::vector<IOTYPE>& transportModes) {
    if (!shell) {
        return false;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams() &&
                           paramIdx < static_cast<int>(transportModes.size());
         ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return transportModes[paramIdx] == IOTYPE::WRITE;
        }
    }
    return false;
}

bool isEffectiveReader(const std::string& tensorName,
                       Shell* shell,
                       const std::vector<IOTYPE>& transportModes) {
    if (!shell) {
        return false;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams() &&
                           paramIdx < static_cast<int>(transportModes.size());
         ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return transportModes[paramIdx] == IOTYPE::READ;
        }
    }
    return false;
}

int findShellParamIndex(Shell* shell, const std::string& tensorName) {
    if (!shell) {
        return -1;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return paramIdx;
        }
    }
    return -1;
}

bool writerRouteCoveredByInputDomain(
    Shell* shell,
    const std::vector<IOTYPE>& effectiveModes,
    int writerParamIndex,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta) {
    if (!shell || writerParamIndex < 0 ||
        writerParamIndex >= shell->getNumShellParams()) {
        return false;
    }

    Split* writerSplit =
        findSingleNonVoidSplit(shell->getShellParam(writerParamIndex));
    if (!writerSplit || writerSplit->type != "IndexSplit") {
        return false;
    }

    const auto writerMeta = splitMeta.find(writerSplit->getId());
    if (writerMeta == splitMeta.end()) {
        return false;
    }

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams() &&
                           paramIdx < static_cast<int>(effectiveModes.size());
         ++paramIdx) {
        if (effectiveModes[paramIdx] == IOTYPE::WRITE) {
            continue;
        }
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        if (!shellParam) {
            continue;
        }
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }
            const auto inputMeta = splitMeta.find(split->getId());
            if (inputMeta == splitMeta.end() ||
                inputMeta->second.bindId != writerMeta->second.bindId) {
                continue;
            }
            if (isStrideOneInputDomainSplit(split)) {
                return true;
            }
        }
    }
    return false;
}

bool writerRouteCoveredByInputDomain2D(
    Shell* shell,
    const std::vector<IOTYPE>& effectiveModes,
    int writerParamIndex,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta) {
    if (!shell || writerParamIndex < 0 ||
        writerParamIndex >= shell->getNumShellParams()) {
        return false;
    }

    ShellParam* writerParam = shell->getShellParam(writerParamIndex);
    if (!writerParam || writerParam->getNumSplit() != 2) {
        return false;
    }

    for (int writerSplitIdx = 0; writerSplitIdx < writerParam->getNumSplit();
         ++writerSplitIdx) {
        Split* writerSplit = writerParam->getSplit(writerSplitIdx);
        if (!writerSplit || writerSplit->getId() == "void" ||
            writerSplit->type != "IndexSplit") {
            return false;
        }
        const auto writerMeta = splitMeta.find(writerSplit->getId());
        if (writerMeta == splitMeta.end()) {
            return false;
        }

        bool covered = false;
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams() &&
                               paramIdx < static_cast<int>(effectiveModes.size());
             ++paramIdx) {
            if (effectiveModes[paramIdx] == IOTYPE::WRITE) {
                continue;
            }
            ShellParam* shellParam = shell->getShellParam(paramIdx);
            if (!shellParam) {
                continue;
            }
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
                Split* split = shellParam->getSplit(splitIdx);
                if (!split || split->getId() == "void") {
                    continue;
                }
                const auto inputMeta = splitMeta.find(split->getId());
                if (inputMeta == splitMeta.end() ||
                    inputMeta->second.bindId != writerMeta->second.bindId) {
                    continue;
                }
                if (isStrideOneInputDomainSplit(split)) {
                    covered = true;
                    break;
                }
            }
            if (covered) {
                break;
            }
        }
        if (!covered) {
            return false;
        }
    }
    return true;
}

}  // namespace detail
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
