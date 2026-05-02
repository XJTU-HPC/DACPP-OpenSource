#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "Rewriter_MPI_Stencil_Common.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {

namespace {

std::string buildParamNameList(Shell* shell) {
    std::string args;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        if (!args.empty()) {
            args += ", ";
        }
        args += shell->getParam(paramIdx)->getName();
    }
    return args;
}

void appendPatternInitForContext(std::string& code,
                                 Shell* shell,
                                 Calc* calc,
                                 const std::unordered_map<std::string, mpi_rewriter::SplitBindMeta>& splitMeta,
                                 const std::vector<IOTYPE>& paramModes,
                                 const std::string& ctxVar) {
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = ctxVar + ".pattern_" + name;
        const IOTYPE mode = paramModes[paramIdx];

        code += "    " + patternName + " = dacpp::mpi::AccessPattern{};\n";
        code += "    " + patternName + ".param_id = " + std::to_string(paramIdx) + ";\n";
        code += "    " + patternName + ".name = \"" + name + "\";\n";
        code += "    " + patternName + ".mode = " + mpi_rewriter::toPlannerMode(mode) + ";\n";
        code += "    " + patternName + ".data_info.dim = " + tensorName + ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName + ".getDim(); ++dim) " +
                patternName + ".data_info.dimLength.push_back(" + tensorName + ".getShape(dim));\n";

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            auto metaIt = splitMeta.find(split->getId());
            mpi_rewriter::SplitBindMeta bindMeta;
            if (metaIt != splitMeta.end()) {
                bindMeta = metaIt->second;
            }

            const bool isIndex = split->type == "IndexSplit";
            const std::string opName = "pattern_" + name + "_op_" + std::to_string(splitIdx);

            code += "    Dac_Op " + opName + ";\n";
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
            code += "    " + patternName + ".param_ops.push_back(" + opName + ");\n";
            code += "    " + patternName + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    " + patternName + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    " + patternName + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + ");\n";
        }

        code += "    " + patternName + ".partition_shape = dacpp::mpi::init_partition_shape(" +
                patternName + ");\n";
        code += "    " + patternName + ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(" +
                patternName + ");\n";
        code += "    if (" + ctxVar + ".binding_split_sizes.size() < " + patternName +
                ".bind_split_sizes.size()) " + ctxVar + ".binding_split_sizes.resize(" + patternName +
                ".bind_split_sizes.size(), 1);\n";
        code += "    for (std::size_t bind_i = 0; bind_i < " + patternName + ".bind_split_sizes.size(); ++bind_i) {\n";
        code += "        " + ctxVar + ".binding_split_sizes[bind_i] = std::max<int64_t>(" +
                ctxVar + ".binding_split_sizes[bind_i], " + patternName + ".bind_split_sizes[bind_i]);\n";
        code += "    }\n";
    }
}

}  // namespace

std::string wrapperName(Shell* shell, Calc* calc) {
    return shell->getName() + "_" + calc->getName();
}

std::string contextTypeName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_ctx_" + wrapperName(shell, calc);
}

std::string initFunctionName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_init_" + wrapperName(shell, calc);
}

std::string runFunctionName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_run_" + wrapperName(shell, calc);
}

std::string buildShellSignature(Shell* shell) {
    std::string signature;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (!paramType.empty() && paramType.back() != '&' && paramType.back() != '*') {
            paramType += "&";
        }
        signature += paramType + " " + param->getName();
        if (paramIdx + 1 != shell->getNumParams()) {
            signature += ", ";
        }
    }
    return signature;
}

std::string buildStencilWrapperCode(DacppFile* dacppFile,
                                    Shell* shell,
                                    Calc* calc,
                                    const clang::BinaryOperator* dacExpr) {
    const std::string wrapper = wrapperName(shell, calc);
    const std::string ctxType = contextTypeName(shell, calc);
    const std::string initName = initFunctionName(shell, calc);
    const std::string runName = runFunctionName(shell, calc);
    const std::string shellSignature = buildShellSignature(shell);
    const std::string shellArgNames = buildParamNameList(shell);
    const auto paramModes = mpi_rewriter::inferEffectiveParamModes(shell, calc);
    const auto splitMeta = mpi_rewriter::collectSplitBindMeta(shell);

    std::string code;
    code += "struct " + ctxType + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    std::unique_ptr<sycl::queue> q;\n";
    code += "    std::vector<int64_t> binding_split_sizes;\n";
    code += "    int64_t total_items = 1;\n";
    code += "    dacpp::mpi::ItemRange item_range{};\n";
    code += "    int64_t local_item_count = 0;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        code += "    dacpp::mpi::AccessPattern pattern_" + calcName + ";\n";
        code += "    dacpp::mpi::PackPlan plan_" + calcName + ";\n";
    }
    code += "};\n\n";

    code += "void " + initName + "(" + ctxType + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.q = std::make_unique<sycl::queue>(sycl::default_selector_v);\n";
    code += "    ctx.binding_split_sizes.clear();\n";
    code += "    ctx.total_items = 1;\n";
    appendPatternInitForContext(code, shell, calc, splitMeta, paramModes, "ctx");
    code += "    for (int64_t split_size : ctx.binding_split_sizes) ctx.total_items *= split_size;\n";
    code += "    ctx.item_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.local_item_count = ctx.item_range.size();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        code += "    ctx.pattern_" + calcName + ".bind_split_sizes = ctx.binding_split_sizes;\n";
        code += "    ctx.plan_" + calcName + " = " +
                mpi_rewriter::buildPackPlanBuilderExpr(paramModes[paramIdx],
                                                       "ctx.item_range",
                                                       "ctx.pattern_" + calcName) + ";\n";
    }
    code += "}\n\n";

    code += "void " + runName + "(" + ctxType + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    int mpi_size = ctx.mpi_size;\n";
    code += "    dacpp::mpi::resetCollectPositionsProfile();\n";
    code += "    auto dacpp_wrapper_start = std::chrono::steady_clock::now();\n";
    code += "    auto& q = *ctx.q;\n";
    code += "    const int64_t local_item_count = ctx.local_item_count;\n";

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const IOTYPE mode = paramModes[paramIdx];
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string packName = "ctx.plan_" + calcName + ".pack";
        const std::string slotsName = "ctx.plan_" + calcName + ".compact_slots";
        const std::string keyOffsetsName = "ctx.plan_" + calcName + ".item_key_offsets";
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_" + calcName;
        const std::string mpiType = mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());

        code += "    std::vector<" + calcParam->getBasicType() + "> " + localName + "(" +
                packName + ".globals.size());\n";
        if (mode != IOTYPE::WRITE) {
            code += "    std::vector<int> sendcounts_" + calcName + ";\n";
            code += "    std::vector<int> displs_" + calcName + ";\n";
            code += "    int local_global_count_" + calcName + " = static_cast<int>(" + packName + ".globals.size());\n";
            code += "    std::vector<int> global_counts_" + calcName + ";\n";
            code += "    std::vector<int> global_displs_" + calcName + ";\n";
            code += "    std::vector<int64_t> gathered_globals_" + calcName + ";\n";
            code += "    if (mpi_rank == 0) {\n";
            code += "        global_counts_" + calcName + ".resize(mpi_size);\n";
            code += "        global_displs_" + calcName + ".resize(mpi_size);\n";
            code += "    }\n";
            code += "    MPI_Gather(&local_global_count_" + calcName + ", 1, MPI_INT, mpi_rank == 0 ? global_counts_" + calcName + ".data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
            code += "    if (mpi_rank == 0) {\n";
            code += "        int current_global_displ = 0;\n";
            code += "        for (int r = 0; r < mpi_size; ++r) {\n";
            code += "            global_displs_" + calcName + "[r] = current_global_displ;\n";
            code += "            current_global_displ += global_counts_" + calcName + "[r];\n";
            code += "        }\n";
            code += "        gathered_globals_" + calcName + ".resize(current_global_displ);\n";
            code += "    }\n";
            code += "    MPI_Gatherv(const_cast<int64_t*>(" + packName + ".globals.data()), local_global_count_" + calcName + ", MPI_LONG_LONG, mpi_rank == 0 ? gathered_globals_" + calcName + ".data() : nullptr, mpi_rank == 0 ? global_counts_" + calcName + ".data() : nullptr, mpi_rank == 0 ? global_displs_" + calcName + ".data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);\n";
            code += "    std::vector<" + calcParam->getBasicType() + "> sendbuf_" + calcName + ";\n";
            code += "    if (mpi_rank == 0) {\n";
            code += "        sendcounts_" + calcName + ".resize(mpi_size);\n";
            code += "        displs_" + calcName + ".resize(mpi_size);\n";
            code += "        int current_displ = 0;\n";
            code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName + ";\n";
            code += "        " + tensorName + ".tensor2Array(" + globalName + ");\n";
            code += "        for (int r = 0; r < mpi_size; ++r) {\n";
            code += "            sendcounts_" + calcName + "[r] = global_counts_" + calcName + "[r];\n";
            code += "            displs_" + calcName + "[r] = current_displ;\n";
            code += "            current_displ += sendcounts_" + calcName + "[r];\n";
            code += "        }\n";
            code += "        sendbuf_" + calcName + " = dacpp::mpi::pack_values_by_globals_parallel_range(" +
                    globalName + ", gathered_globals_" + calcName + ".data(), gathered_globals_" +
                    calcName + ".size());\n";
            code += "    }\n";
            code += "    " + localName + ".resize(static_cast<std::size_t>(local_global_count_" + calcName + "));\n";
            if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
                code += "    std::vector<int> sendcounts_bytes_" + calcName + " = sendcounts_" + calcName + ";\n";
                code += "    std::vector<int> displs_bytes_" + calcName + " = displs_" + calcName + ";\n";
                code += "    if (mpi_rank == 0) {\n";
                code += "        for (int r = 0; r < mpi_size; ++r) {\n";
                code += "            sendcounts_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
                code += "            displs_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
                code += "        }\n";
                code += "    }\n";
                code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_" + calcName + ".data() : nullptr, mpi_rank == 0 ? displs_bytes_" + calcName + ".data() : nullptr, " + mpiType + ", " + localName + ".data(), " +
                        mpi_rewriter::mpiPayloadCountExpr("local_global_count_" + calcName,
                                                          calcParam->getBasicType()) +
                        ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            } else {
                code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? sendcounts_" + calcName + ".data() : nullptr, mpi_rank == 0 ? displs_" + calcName + ".data() : nullptr, " + mpiType + ", " + localName + ".data(), local_global_count_" + calcName + ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
        }

        code += "    const int " + calcName + "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_" + calcName + "));\n";
        if (mpi_rewriter::inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + calcName + "_cols = ctx.pattern_" + calcName + ".partition_shape[1];\n";
        }
    }

    code += "    if (local_item_count > 0) {\n";
    code += "        {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "            sycl::buffer<" + calcParam->getBasicType() + ", 1> buffer_" + name +
                "(local_" + name + ".data(), sycl::range<1>(local_" + name + ".size()));\n";
        code += "            sycl::buffer<int32_t, 1> slots_buffer_" + name +
                "(ctx.plan_" + name + ".compact_slots.data(), sycl::range<1>(ctx.plan_" + name +
                ".compact_slots.size()));\n";
        code += "            sycl::buffer<int32_t, 1> key_offsets_buffer_" + name +
                "(ctx.plan_" + name + ".item_key_offsets.data(), sycl::range<1>(ctx.plan_" + name +
                ".item_key_offsets.size()));\n";
    }
    code += "            q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto acc_" + name + " = buffer_" + name + ".get_access<" +
                mpi_rewriter::toAccessorMode(paramModes[paramIdx]) + ">(h);\n";
        code += "                auto slots_acc_" + name +
                " = slots_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
        code += "                auto key_offsets_acc_" + name +
                " = key_offsets_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                    auto* data_" + name +
                " = acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* slots_" + name +
                " = slots_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* key_offsets_" + name +
                " = key_offsets_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (mpi_rewriter::inferViewRank(shellParam, calcParam) <= 1) {
            code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, paramModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                    "[item_linear]};\n";
        } else {
            code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, paramModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                    "[item_linear], " + name + "_cols};\n";
        }
    }
    code += "                    " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += "view_" + calc->getParam(paramIdx)->getName();
        if (paramIdx + 1 != calc->getNumParams()) {
            code += ", ";
        }
    }
    code += ");\n";
    code += "                });\n";
    code += "            });\n";
    code += "            q.wait();\n";
    code += "        }\n";
    code += "    }\n";

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (paramModes[paramIdx] == IOTYPE::READ) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string packName = "ctx.plan_" + calcName + ".pack";
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_out_" + calcName;
        const std::string mpiType = mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());
        const bool needsBcast = mpi_rewriter::tensorNeedsBroadcast(dacppFile, tensorName, dacExpr);

        code += "    const auto& writeback_globals_" + calcName + " = " + packName +
                ".writeback_globals.empty() ? " + packName + ".globals : " + packName +
                ".writeback_globals;\n";
        code += "    int send_count_" + calcName + " = static_cast<int>(writeback_globals_" +
                calcName + ".size());\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> writeback_values_" + calcName +
                " = dacpp::mpi::build_writeback_values_parallel(" + localName + ", " +
                packName + ");\n";
        code += "    std::vector<int> recvcounts_" + calcName + ";\n";
        code += "    std::vector<int> recvdispls_" + calcName + ";\n";
        code += "    std::vector<int64_t> global_recv_globals_" + calcName + ";\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> global_recv_values_" + calcName + ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        recvcounts_" + calcName + ".resize(mpi_size);\n";
        code += "        recvdispls_" + calcName + ".resize(mpi_size);\n";
        code += "    }\n";
        code += "    MPI_Gather(&send_count_" + calcName + ", 1, MPI_INT, mpi_rank == 0 ? recvcounts_" +
                calcName + ".data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        int current_displ = 0;\n";
        code += "        for (int r = 0; r < mpi_size; ++r) {\n";
        code += "            recvdispls_" + calcName + "[r] = current_displ;\n";
        code += "            current_displ += recvcounts_" + calcName + "[r];\n";
        code += "        }\n";
        code += "        global_recv_globals_" + calcName + ".resize(current_displ);\n";
        code += "        global_recv_values_" + calcName + ".resize(current_displ);\n";
        code += "    }\n";
        code += "    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_" + calcName + ".data()), send_count_" +
                calcName + ", MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_" + calcName +
                ".data() : nullptr, mpi_rank == 0 ? recvcounts_" + calcName +
                ".data() : nullptr, mpi_rank == 0 ? recvdispls_" + calcName +
                ".data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);\n";
        if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
            code += "    std::vector<int> recvcounts_bytes_" + calcName + " = recvcounts_" + calcName + ";\n";
            code += "    std::vector<int> recvdispls_bytes_" + calcName + " = recvdispls_" + calcName + ";\n";
            code += "    if (mpi_rank == 0) {\n";
            code += "        for (int r = 0; r < mpi_size; ++r) {\n";
            code += "            recvcounts_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
            code += "            recvdispls_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
            code += "        }\n";
            code += "    }\n";
            code += "    MPI_Gatherv(writeback_values_" + calcName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr("send_count_" + calcName,
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", mpi_rank == 0 ? global_recv_values_" + calcName +
                    ".data() : nullptr, mpi_rank == 0 ? recvcounts_bytes_" + calcName +
                    ".data() : nullptr, mpi_rank == 0 ? recvdispls_bytes_" + calcName +
                    ".data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Gatherv(writeback_values_" + calcName + ".data(), send_count_" + calcName + ", " + mpiType +
                    ", mpi_rank == 0 ? global_recv_values_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvcounts_" +
                    calcName + ".data() : nullptr, mpi_rank == 0 ? recvdispls_" + calcName +
                    ".data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    if (mpi_rank == 0) {\n";
        code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName + ";\n";
        code += "        " + tensorName + ".tensor2Array(" + globalName + ");\n";
        code += "        dacpp::mpi::apply_writeback_by_globals(global_recv_values_" + calcName +
                ", global_recv_globals_" + calcName + ", " + globalName + ");\n";
        code += "        " + tensorName + ".array2Tensor(" + globalName + ");\n";
        if (needsBcast) {
            code += "        if (!" + globalName + ".empty()) {\n";
            code += "            MPI_Bcast(" + globalName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr(tensorName + ".getSize()",
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            code += "        }\n";
        }
        code += "    } else ";
        if (needsBcast) {
            code += "{\n";
            code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName +
                    "(static_cast<std::size_t>(" + tensorName + ".getSize()));\n";
            code += "        if (!" + globalName + ".empty()) {\n";
            code += "            MPI_Bcast(" + globalName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr(tensorName + ".getSize()",
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            code += "        }\n";
            code += "        " + tensorName + ".array2Tensor(" + globalName + ");\n";
            code += "    }\n";
        } else {
            code += "{\n";
            code += "    }\n";
        }
    }

    code += "    auto dacpp_wrapper_end = std::chrono::steady_clock::now();\n";
    code += "    double dacpp_wrapper_local_ms = std::chrono::duration<double, std::milli>(dacpp_wrapper_end - dacpp_wrapper_start).count();\n";
    code += "    double dacpp_wrapper_max_ms = 0.0;\n";
    code += "    MPI_Reduce(&dacpp_wrapper_local_ms, &dacpp_wrapper_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);\n";
    code += "    if (mpi_rank == 0 && dacpp::mpi::profilingEnabled()) {\n";
    code += "        std::fprintf(stderr, \"[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\\n\", \"" + wrapper + "\", dacpp_wrapper_max_ms);\n";
    code += "    }\n";
    code += "    dacpp::mpi::reportCollectPositionsProfile(\"" + wrapper + "\", MPI_COMM_WORLD);\n";
    code += "}\n\n";

    code += "void " + wrapper + "(" + shellSignature + ") {\n";
    code += "    " + ctxType + " ctx;\n";
    code += "    " + initName + "(ctx";
    if (!shellArgNames.empty()) {
        code += ", " + shellArgNames;
    }
    code += ");\n";
    code += "    " + runName + "(ctx";
    if (!shellArgNames.empty()) {
        code += ", " + shellArgNames;
    }
    code += ");\n";
    code += "}\n";

    return code;
}

}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator
