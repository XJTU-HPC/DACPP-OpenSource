#include <set>
#include <string>

#include "Rewriter_MPI_PostRegion_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

bool isWordBoundary(char c) {
    return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_');
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
            std::string calcName = name;
            for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
                if (shell->getParam(paramIdx)->getName() == name) {
                    calcName = calc->getParam(paramIdx)->getName();
                    break;
                }
            }
            std::string elemType = "double";
            for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
                if (shell->getParam(paramIdx)->getName() == name) {
                    elemType = shell->getShellParam(paramIdx)->getBasicType();
                    break;
                }
            }
            code += "    if (ctx.use_partial_exchange && ctx.dist_" + calcName +
                    ".enabled && ctx.dist_" + calcName +
                    ".root_bridge_plan.supported) {\n";
            code += "        std::vector<" + elemType + "> bridge_dense_" + name +
                    ";\n";
            code += "        if (__dacpp_root_rank) {\n";
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

    std::set<const clang::Stmt*> distributedFollowupStmts;
    const auto distributedRegions =
        collectDistributedFollowupRegions(dacppFile, shell, calc, dacExpr);
    for (const auto& region : distributedRegions) {
        distributedFollowupStmts.insert(region.stmt);
    }

    std::string code;
    for (std::size_t stmtIdx = 0; stmtIdx < plan.siblingStmts.size(); ++stmtIdx) {
        if (distributedFollowupStmts.count(plan.siblingStmts[stmtIdx]) != 0) {
            continue;
        }
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
