#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

std::string joinCalcBody(const Calc* calc) {
    std::string body;
    for (const auto& block : calc->body) {
        body += block;
        if (!body.empty() && body.back() != '\n') {
            body += "\n";
        }
    }
    return body;
}

}  // namespace

std::unordered_map<std::string, SplitBindMeta> collectSplitBindMeta(Shell* shell) {
    std::unordered_map<std::string, SplitBindMeta> splitMeta;
    std::unordered_map<int, int> componentBindId;

    std::vector<BINDINFO> bindInfo;
    shell->GetBindInfo(&bindInfo);

    for (const auto& info : bindInfo) {
        Split* split = shell->search_symbol(info.v);
        if (!split || split->getId() == "void") {
            continue;
        }

        auto [it, inserted] =
            componentBindId.emplace(info.icls, static_cast<int>(componentBindId.size()));
        const std::string offset = info.offset.empty() ? "0" : info.offset;
        splitMeta[split->getId()] = SplitBindMeta{it->second, offset};
    }

    std::unordered_map<std::string, int> fallbackBindId;
    const int fallbackBase = static_cast<int>(componentBindId.size());

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            if (splitMeta.find(split->getId()) != splitMeta.end()) {
                continue;
            }

            auto [it, inserted] = fallbackBindId.emplace(
                split->getId(), fallbackBase + static_cast<int>(fallbackBindId.size()));
            splitMeta[split->getId()] = SplitBindMeta{it->second, "0"};
        }
    }

    return splitMeta;
}

std::string buildPatternInitCode(
    Shell* shell,
    Calc* calc,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta,
    const std::vector<IOTYPE>& paramModes) {
    std::string code;
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "pattern_" + name;
        const IOTYPE mode = paramModes[paramIdx];

        code += "    dacpp::mpi::AccessPattern " + patternName + ";\n";
        code += "    " + patternName + ".param_id = " + std::to_string(paramIdx) + ";\n";
        code += "    " + patternName + ".name = \"" + name + "\";\n";
        code += "    " + patternName + ".mode = " + toPlannerMode(mode) + ";\n";
        code += "    " + patternName + ".data_info.dim = " + tensorName + ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName +
                ".getDim(); ++dim) " + patternName +
                ".data_info.dimLength.push_back(" + tensorName + ".getShape(dim));\n";

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            auto metaIt = splitMeta.find(split->getId());
            SplitBindMeta bindMeta;
            if (metaIt != splitMeta.end()) {
                bindMeta = metaIt->second;
            }

            const bool isIndex = split->type == "IndexSplit";
            const std::string opName = patternName + "_op_" + std::to_string(splitIdx);

            code += "    Dac_Op " + opName + ";\n";
            code += "    " + opName + ".setDimId(" + std::to_string(split->getDimIdx()) + ");\n";
            code += "    " + opName + ".size = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitSize()) +
                    ";\n";
            code += "    " + opName + ".stride = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitStride()) +
                    ";\n";
            if (isIndex) {
                code += "    " + opName + ".SetSplitSize(" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) + "));\n";
            } else {
                code += "    " + opName + ".SetSplitSize((" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) + ") - " +
                        std::to_string(static_cast<RegularSplit*>(split)->getSplitSize()) + ") / " +
                        std::to_string(static_cast<RegularSplit*>(split)->getSplitStride()) + " + 1);\n";
            }
            code += "    " + patternName + ".param_ops.push_back(" + opName + ");\n";
            code += "    " + patternName + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    " + patternName + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    " + patternName + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + ");\n";
        }

        code += "    " + patternName +
                ".partition_shape = dacpp::mpi::init_partition_shape(" + patternName + ");\n";
        code += "    " + patternName +
                ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(" + patternName + ");\n";
        code += "    if (binding_split_sizes.size() < " + patternName +
                ".bind_split_sizes.size()) binding_split_sizes.resize(" + patternName +
                ".bind_split_sizes.size(), 1);\n";
        code += "    for (std::size_t bind_i = 0; bind_i < " + patternName +
                ".bind_split_sizes.size(); ++bind_i) {\n";
        code += "        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], " +
                patternName + ".bind_split_sizes[bind_i]);\n";
        code += "    }\n";
    }

    return code;
}

std::string buildLocalCalcCode(Shell* shell, Calc* calc) {
    (void)shell;
    std::string code;
    if (calc->getNumParams() > 0) {
        code += "template <";
        for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
            code += "typename __dacpp_view_t" + std::to_string(paramIdx);
            if (paramIdx + 1 != calc->getNumParams()) {
                code += ", ";
            }
        }
        code += ">\n";
    }
    code += "inline void " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += "__dacpp_view_t" + std::to_string(paramIdx) + " " +
                calc->getParam(paramIdx)->getName();
        if (paramIdx + 1 != calc->getNumParams()) {
            code += ", ";
        }
    }
    code += ") ";
    code += joinCalcBody(calc);
    code += "\n";
    return code;
}

std::string buildPackPlanBuilderExpr(IOTYPE mode,
                                     const std::string& rangeName,
                                     const std::string& patternName) {
    if (mode == IOTYPE::READ) {
        return "dacpp::mpi::build_input_pack_plan(" + rangeName + ", " +
               patternName + ")";
    }
    if (mode == IOTYPE::WRITE) {
        return "dacpp::mpi::build_output_pack_plan(" + rangeName + ", " +
               patternName + ")";
    }
    return "dacpp::mpi::build_rw_pack_plan(" + rangeName + ", " +
           patternName + ")";
}

std::string profileSegmentStartCode(const std::string& varName) {
    return "    auto " + varName + " = dacpp::mpi::profileSegmentStart();\n";
}

std::string profileRecordCode(const std::string& profileName,
                              const std::string& segmentName,
                              const std::string& startVarName) {
    return "    dacpp::mpi::recordProfileSegment(" + profileName +
           ", dacpp::mpi::ProfileSegment::" + segmentName + ", " +
           startVarName + ");\n";
}

std::string buildPrelude(DacppFile* dacppFile) {
    std::string code;
    std::set<std::string> seenHeaders;
    for (int idx = 0; idx < dacppFile->getNumHeaderFile(); ++idx) {
        const std::string header = dacppFile->getHeaderFile(idx)->getName();
    if (seenHeaders.insert(header).second) {
        code += "#include " + header + "\n";
    }
}
    if (seenHeaders.insert("<chrono>").second) {
        code += "#include <chrono>\n";
    }
    if (seenHeaders.insert("<cstdio>").second) {
        code += "#include <cstdio>\n";
    }
    if (seenHeaders.insert("<utility>").second) {
        code += "#include <utility>\n";
    }
    code += "\n";
    code += R"(static inline bool __dacpp_mpi_is_root_rank() {
    int __dacpp_mpi_initialized = 0;
    MPI_Initialized(&__dacpp_mpi_initialized);
    if (!__dacpp_mpi_initialized) {
        return true;
    }
    int __dacpp_mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &__dacpp_mpi_rank);
    return __dacpp_mpi_rank == 0;
}

)";

    std::set<std::string> seenNamespaces;
    for (int idx = 0; idx < dacppFile->getNumNameSpace(); ++idx) {
        const std::string ns = dacppFile->getNameSpace(idx)->getName();
        if (seenNamespaces.insert(ns).second) {
            code += "using namespace " + ns + ";\n";
        }
    }
    code += "\n";
    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
