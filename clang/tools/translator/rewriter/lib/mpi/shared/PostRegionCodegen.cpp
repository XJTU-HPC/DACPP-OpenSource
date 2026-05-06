#include <set>
#include <string>
#include <functional>

#include "clang/AST/Expr.h"
#include "clang/Lex/Lexer.h"

#include "PostRegion_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

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

std::string replaceWord(std::string text,
                        const std::string& word,
                        const std::string& replacement) {
    if (word.empty() || text.empty() || word.size() > text.size()) {
        return text;
    }
    std::string result;
    result.reserve(text.size());
    std::size_t pos = 0;
    std::size_t lastPos = 0;
    while ((pos = text.find(word, pos)) != std::string::npos) {
        const bool leftOk = pos == 0 || isWordBoundary(text[pos - 1]);
        const std::size_t rightIdx = pos + word.size();
        const bool rightOk =
            rightIdx >= text.size() || isWordBoundary(text[rightIdx]);
        if (leftOk && rightOk) {
            result.append(text, lastPos, pos - lastPos);
            result.append(replacement);
            pos += word.size();
            lastPos = pos;
        } else {
            ++pos;
        }
    }
    result.append(text, lastPos, text.size() - lastPos);
    return result;
}

std::string accessModeFor(const std::string& name,
                          const detail::LoopRegionInfo& info) {
    return info.writtenTensors.count(name) != 0
               ? "sycl::access::mode::read_write"
               : "sycl::access::mode::read";
}

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

std::set<std::string> collectWrittenShellTensorsByText(DacppFile* dacppFile,
                                                       Shell* shell,
                                                       const clang::Stmt* stmt) {
    std::set<std::string> written;
    if (!dacppFile || !shell || !stmt || !dacppFile->getContext()) {
        return written;
    }

    std::function<void(const clang::Expr*)> recordLhs =
        [&](const clang::Expr* lhs) {
            const std::string lhsText = getStmtSourceText(lhs, dacppFile->getContext());
            for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
                const std::string name = shell->getParam(paramIdx)->getName();
                if (containsWord(lhsText, name)) {
                    written.insert(name);
                }
            }
        };

    std::function<void(const clang::Stmt*)> scan = [&](const clang::Stmt* node) {
        if (!node) {
            return;
        }
        if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(node)) {
            if (binary->isAssignmentOp()) {
                recordLhs(binary->getLHS());
            }
        } else if (const auto* opCall =
                       llvm::dyn_cast<clang::CXXOperatorCallExpr>(node)) {
            if (opCall->getOperator() == clang::OO_Equal &&
                opCall->getNumArgs() >= 1) {
                recordLhs(opCall->getArg(0));
            }
        }
        for (const clang::Stmt* child : node->children()) {
            scan(child);
        }
    };

    scan(stmt);
    return written;
}

std::string buildRootBridgeRefreshCode(Shell* shell,
                                       Calc* calc,
                                       const std::set<std::string>& writtenTensors,
                                       const std::string& rootFlagName) {
    std::string code;
    for (const std::string& name : writtenTensors) {
        std::string calcName = name;
        std::string elemType = "double";
        for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
            if (shell->getParam(paramIdx)->getName() == name) {
                calcName = calc->getParam(paramIdx)->getName();
                elemType = shell->getShellParam(paramIdx)->getBasicType();
                break;
            }
        }
        code += "    if (ctx.use_partial_exchange && ctx.dist_" + calcName +
                ".enabled && ctx.dist_" + calcName +
                ".root_bridge_plan.supported) {\n";
        code += "        std::vector<" + elemType + "> bridge_dense_" + name +
                ";\n";
        code += "        if (" + rootFlagName + ") {\n";
        code += "            " + name + ".tensor2Array(bridge_dense_" + name +
                ");\n";
        code += "            dacpp::mpi::pack_values_by_globals_parallel_range_into(bridge_dense_" +
                name + ", ctx.plan_" + calcName + ".pack.globals.data(), ctx.plan_" +
                calcName + ".pack.globals.size(), ctx.dist_" + calcName +
                ".local_cache);\n";
        code += "        }\n";
        code += "        dacpp::mpi::exchange_values_by_slots(bridge_dense_" +
                name + ", ctx.dist_" + calcName +
                ".root_bridge_plan, ctx.dist_" + calcName +
                ".local_cache);\n";
        code += "    }\n";
    }
    return code;
}

std::string buildRootSerialStmtHelper(DacppFile* dacppFile,
                                      Shell* shell,
                                      Calc* calc,
                                      const clang::Stmt* stmt,
                                      std::size_t stmtIdx,
                                      const std::string& ctxTypeName,
                                      const std::string& shellSignature) {
    const std::string stmtText = getStmtSourceText(stmt, dacppFile->getContext());
    if (stmtText.empty()) {
        return "";
    }
    const std::set<std::string> writtenTensors =
        collectWrittenShellTensorsByText(dacppFile, shell, stmt);
    std::string code;
    code += "void " + detail::helperNameFor(shell, calc, stmtIdx) + "(" +
            ctxTypeName + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    bool __dacpp_root_rank = (ctx.mpi_rank == 0);\n";
    code += "    if (__dacpp_root_rank) {\n";
    code += "        " + stmtText + "\n";
    code += "    }\n";
    code += buildRootBridgeRefreshCode(shell, calc, writtenTensors,
                                       "__dacpp_root_rank");
    code += "}\n\n";
    return code;
}

std::string buildLoopRegionHelper(DacppFile* dacppFile,
                                  Shell* shell,
                                  Calc* calc,
                                  const clang::ForStmt* forStmt,
                                  std::size_t stmtIdx,
                                  const std::string& ctxTypeName,
                                  const std::string& shellSignature) {
    detail::LoopRegionInfo info;
    if (!detail::extractLoopRegionInfo(
            forStmt, dacppFile->getContext(), shell,
            dacppFile->getBufferRegionPlan(), info)) {
        return "";
    }
    const auto distributedSitePlan =
        analyzeDistributedStencilSite(dacppFile, shell, calc,
                                      dacppFile->getBufferRegionPlan().dacExpr);

    std::set<std::string> usedTensors = info.readTensors;
    usedTensors.insert(info.writtenTensors.begin(), info.writtenTensors.end());

    std::string kernelBody = info.bodyText;
    for (const std::string& name : usedTensors) {
        kernelBody = replaceWord(kernelBody, name, "acc_" + name);
    }

    std::string code;
    code += "void " + detail::helperNameFor(shell, calc, stmtIdx) + "(" +
            ctxTypeName + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    bool __dacpp_root_rank = (ctx.mpi_rank == 0);\n";
    code += "    auto& q = *ctx.q;\n";

    code += "    if (__dacpp_root_rank) {\n";
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

    code += "        {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        const std::string& name = param->getName();
        if (usedTensors.count(name) == 0) {
            continue;
        }
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        code += "            sycl::buffer<" + shellParam->getBasicType() +
                ", 1> b_" + name + "(h_" + name +
                ".data(), sycl::range<1>(h_" + name + ".size()));\n";
    }
    code += "            int __L = (" + info.lowerExpr + ");\n";
    code += "            int __R = (" + info.upperExpr + ");\n";
    code += "            int __N = " +
            std::string(info.upperInclusive ? "(__R - __L + 1)" : "(__R - __L)") +
            ";\n";
    code += "            if (__N > 0) {\n";
    code += "                q.submit([&](sycl::handler& h) {\n";
    for (const std::string& name : usedTensors) {
        code += "                    auto acc_" + name + " = b_" + name +
                ".get_access<" + accessModeFor(name, info) + ">(h);\n";
    }
    code += "                    h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__N)), [=](sycl::id<1> idx) {\n";
    code += "                        int " + info.loopVar +
            " = __L + static_cast<int>(idx[0]);\n";
    code += "                        " + kernelBody + "\n";
    code += "                    });\n";
    code += "                });\n";
    code += "                q.wait();\n";
    code += "            }\n";
    code += "        }\n";

    for (const std::string& name : info.writtenTensors) {
        code += "        " + name + ".array2Tensor(h_" + name + ");\n";
    }
    code += "    }\n";

    if (distributedSitePlan.supported && distributedSitePlan.hasRootBridge) {
        for (const std::string& name : info.writtenTensors) {
            if (distributedSitePlan.rootBridgeTensors.count(name) == 0) {
                continue;
            }
        }
        code += buildRootBridgeRefreshCode(shell, calc, info.writtenTensors,
                                           "__dacpp_root_rank");
    }
    code += "}\n\n";
    return code;
}

}  // namespace

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
    const auto sitePlan = analyzeDistributedStencilSite(
        dacppFile, shell, calc, dacExpr);
    if (!sitePlan.supported || !sitePlan.hasRootBridge) {
        return "";
    }

    std::set<const clang::Stmt*> distributedFollowupStmts;
    const auto distributedRegions =
        collectDistributedFollowupRegions(dacppFile, shell, calc, dacExpr);
    for (const auto& region : distributedRegions) {
        distributedFollowupStmts.insert(region.stmt);
    }
    if (!sitePlan.hasRootBridge && sitePlan.followupMappings.size() == 1) {
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            distributedFollowupStmts.insert(stmt);
        }
    }

    std::string code;
    for (std::size_t stmtIdx = 0; stmtIdx < plan.siblingStmts.size(); ++stmtIdx) {
        const auto* forStmt =
            llvm::dyn_cast_or_null<clang::ForStmt>(plan.siblingStmts[stmtIdx]);
        if (!forStmt) {
            continue;
        }
        std::string helper = buildLoopRegionHelper(
            dacppFile, shell, calc, forStmt, stmtIdx, ctxTypeName,
            shellSignature);
        if (helper.empty() && distributedFollowupStmts.count(plan.siblingStmts[stmtIdx]) == 0) {
            helper = buildRootSerialStmtHelper(
                dacppFile, shell, calc, plan.siblingStmts[stmtIdx], stmtIdx,
                ctxTypeName, shellSignature);
        } else if (helper.empty() && distributedFollowupStmts.count(plan.siblingStmts[stmtIdx]) != 0) {
            if (sitePlan.hasRootBridge) {
                helper = buildRootSerialStmtHelper(
                    dacppFile, shell, calc, plan.siblingStmts[stmtIdx], stmtIdx,
                    ctxTypeName, shellSignature);
            }
        }
        code += helper;
    }
    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
