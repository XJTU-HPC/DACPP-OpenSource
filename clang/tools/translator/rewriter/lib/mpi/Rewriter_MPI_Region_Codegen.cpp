#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

MPIRegionGeneratedCode buildMPIRegionNames(Expression* expr) {
    MPIRegionGeneratedCode generated;
    if (!expr || !expr->getShell() || !expr->getCalc()) {
        return generated;
    }

    Shell* shell = expr->getShell();
    Calc* calc = expr->getCalc();
    const std::string baseName = shell->getName() + "_" + calc->getName();
    generated.ctxTypeName = "__dacpp_mpi_ctx_" + baseName;
    generated.ctxVarName = generated.ctxTypeName + "_0";
    generated.initName = "__dacpp_mpi_init_" + baseName;
    generated.submitName = "__dacpp_mpi_submit_" + baseName;
    generated.haloName = "__dacpp_mpi_halo_" + baseName;
    generated.syncName = "__dacpp_mpi_sync_" + baseName;
    return generated;
}

std::string buildShellParamSignature(Shell* shell) {
    if (!shell) {
        return "";
    }

    std::string shellParamSig;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (!paramType.empty() && paramType.back() != '&' &&
            paramType.back() != '*') {
            paramType += "&";
        }
        if (!shellParamSig.empty()) {
            shellParamSig += ", ";
        }
        shellParamSig += paramType + " " + param->getName();
    }
    return shellParamSig;
}

}  // namespace

std::string buildMPIRegionCodegen(
    DacppFile* dacppFile,
    Expression* expr,
    const MPIRegionTransferPolicy& transferPolicy) {
    if (!expr || !expr->getShell() || !expr->getCalc()) {
        return "";
    }

    Shell* shell = expr->getShell();
    Calc* calc = expr->getCalc();
    const auto generated = buildMPIRegionNames(expr);
    const auto paramModes = inferEffectiveParamModes(shell, calc);
    const auto splitMeta = collectSplitBindMeta(shell);
    const std::string shellParamSig = buildShellParamSignature(shell);

    std::string code;

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
        code += "    std::unique_ptr<sycl::buffer<" + baseType +
                ", 1>> buf_" + name + ";\n";
        code += "    std::unique_ptr<sycl::buffer<int32_t, 1>> slots_buf_" +
                name + ";\n";
        code += "    int " + name + "_partition_size = 0;\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    int " + name + "_cols = 0;\n";
        }
        code += "    dacpp::mpi::ParamHalo halo_" + name + ";\n";
    }
    code += "};\n\n";

    code += "void " + generated.initName + "(" + generated.ctxTypeName +
            "& ctx";
    if (!shellParamSig.empty()) {
        code += ", " + shellParamSig;
    }
    code += ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.q = sycl::queue(sycl::default_selector_v);\n";
    code += "    ctx.binding_split_sizes.clear();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "ctx.pattern_" + name;
        const IOTYPE mode = paramModes[paramIdx];

        code += "    ctx.pattern_" + name + " = dacpp::mpi::AccessPattern();\n";
        code += "    ctx.pattern_" + name + ".param_id = " +
                std::to_string(paramIdx) + ";\n";
        code += "    ctx.pattern_" + name + ".name = \"" + name + "\";\n";
        code += "    ctx.pattern_" + name + ".mode = " + toPlannerMode(mode) +
                ";\n";
        code += "    ctx.pattern_" + name + ".data_info.dim = " + tensorName +
                ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName +
                ".getDim(); ++dim) ctx.pattern_" + name +
                ".data_info.dimLength.push_back(" + tensorName +
                ".getShape(dim));\n";

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
            const std::string opName =
                "__dacpp_op_" + name + "_" + std::to_string(splitIdx);

            code += "    { Dac_Op " + opName + ";\n";
            code += "    " + opName + ".setDimId(" +
                    std::to_string(split->getDimIdx()) + ");\n";
            code += "    " + opName + ".size = " +
                    std::to_string(isIndex ? 1
                                           : static_cast<RegularSplit*>(split)
                                                 ->getSplitSize()) +
                    ";\n";
            code += "    " + opName + ".stride = " +
                    std::to_string(isIndex ? 1
                                           : static_cast<RegularSplit*>(split)
                                                 ->getSplitStride()) +
                    ";\n";
            if (isIndex) {
                code += "    " + opName + ".SetSplitSize(" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) +
                        "));\n";
            } else {
                code += "    " + opName + ".SetSplitSize((" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) +
                        ") - " +
                        std::to_string(
                            static_cast<RegularSplit*>(split)->getSplitSize()) +
                        ") / " +
                        std::to_string(static_cast<RegularSplit*>(split)
                                           ->getSplitStride()) +
                        " + 1);\n";
            }
            code += "    ctx.pattern_" + name + ".param_ops.push_back(" + opName +
                    ");\n";
            code += "    ctx.pattern_" + name + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    ctx.pattern_" + name + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    ctx.pattern_" + name + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + "); }\n";
        }

        code += "    ctx.pattern_" + name +
                ".partition_shape = dacpp::mpi::init_partition_shape(ctx.pattern_" +
                name + ");\n";
        code += "    ctx.pattern_" + name +
                ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(ctx.pattern_" +
                name + ");\n";
        code += "    if (ctx.binding_split_sizes.size() < ctx.pattern_" + name +
                ".bind_split_sizes.size()) ctx.binding_split_sizes.resize(ctx.pattern_" +
                name + ".bind_split_sizes.size(), 1);\n";
        code += "    for (std::size_t bind_i = 0; bind_i < ctx.pattern_" + name +
                ".bind_split_sizes.size(); ++bind_i) {\n";
        code +=
            "        ctx.binding_split_sizes[bind_i] = std::max<int64_t>(ctx.binding_split_sizes[bind_i], ctx.pattern_" +
            name + ".bind_split_sizes[bind_i]);\n";
        code += "    }\n";
    }

    code += "    ctx.total_items = 1;\n";
    code +=
        "    for (int64_t split_size : ctx.binding_split_sizes) ctx.total_items *= split_size;\n";
    code +=
        "    auto item_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.local_item_count = item_range.size();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& name = calc->getParam(paramIdx)->getName();
        code += "    ctx.pattern_" + name +
                ".bind_split_sizes = ctx.binding_split_sizes;\n";
    }

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
        code += "    ctx.slots_" + calcName +
                " = dacpp::mpi::build_item_slots(item_range, " + patternName +
                ", ctx.pack_" + calcName + ");\n";
        code += "    ctx.local_" + calcName +
                ".resize(ctx.pack_" + calcName + ".globals.size());\n";

        if (transferPolicy.needsInitScatter[static_cast<std::size_t>(paramIdx)]) {
            const std::string payloadRecvCount = mpiPayloadCountExpr(
                "recv_count_" + calcName, calcParam->getBasicType());
            code += "    {\n";
            code += "    int recv_count_" + calcName + " = 0;\n";
            code += "    std::vector<int> sendcounts_" + calcName + ";\n";
            code += "    std::vector<int> displs_" + calcName + ";\n";
            code += "    std::vector<" + calcParam->getBasicType() +
                    "> sendbuf_" + calcName + ";\n";
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        sendcounts_" + calcName +
                    ".resize(ctx.mpi_size);\n";
            code += "        displs_" + calcName + ".resize(ctx.mpi_size);\n";
            code += "        int current_displ = 0;\n";
            code += "        std::vector<" + calcParam->getBasicType() +
                    "> global_" + calcName + ";\n";
            code += "        " + tensorName + ".tensor2Array(global_" + calcName +
                    ");\n";
            code += "        for (int r = 0; r < ctx.mpi_size; ++r) {\n";
            code +=
                "            auto r_range = dacpp::mpi::get_rank_item_range(ctx.total_items, r, ctx.mpi_size);\n";
            code += "            auto r_pack = " +
                    buildRemotePackBuilderExpr(mode, "r_range", patternName) +
                    ";\n";
            code += "            auto r_values = dacpp::mpi::pack_values_by_globals(global_" +
                    calcName + ", r_pack.globals);\n";
            code += "            int r_count = static_cast<int>(r_values.size());\n";
            code += "            sendcounts_" + calcName + "[r] = r_count;\n";
            code += "            displs_" + calcName + "[r] = current_displ;\n";
            code += "            current_displ += r_count;\n";
            code += "            sendbuf_" + calcName +
                    ".insert(sendbuf_" + calcName +
                    ".end(), r_values.begin(), r_values.end());\n";
            code += "        }\n";
            code += "    }\n";
            code += "    MPI_Scatter(ctx.mpi_rank == 0 ? sendcounts_" + calcName +
                    ".data() : nullptr, 1, MPI_INT, &recv_count_" + calcName +
                    ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
            code += "    ctx.local_" + calcName +
                    ".resize(recv_count_" + calcName + ");\n";
            if (usesByteTransport(calcParam->getBasicType())) {
                code += "    { std::vector<int> sc_bytes = sendcounts_" +
                        calcName + ";\n";
                code += "      std::vector<int> ds_bytes = displs_" + calcName +
                        ";\n";
                code += "      if (ctx.mpi_rank == 0) { for (int r = 0; r < ctx.mpi_size; ++r) { sc_bytes[r] *= sizeof(" +
                        calcParam->getBasicType() + "); ds_bytes[r] *= sizeof(" +
                        calcParam->getBasicType() + "); } }\n";
                code += "      MPI_Scatterv(ctx.mpi_rank == 0 ? sendbuf_" +
                        calcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? sc_bytes.data() : nullptr, ctx.mpi_rank == 0 ? ds_bytes.data() : nullptr, " +
                        mpiType + ", ctx.local_" + calcName + ".data(), " +
                        payloadRecvCount + ", " + mpiType +
                        ", 0, MPI_COMM_WORLD); }\n";
            } else {
                code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? sendbuf_" +
                        calcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? sendcounts_" +
                        calcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? displs_" +
                        calcName + ".data() : nullptr, " + mpiType +
                        ", ctx.local_" + calcName + ".data(), " +
                        payloadRecvCount + ", " + mpiType +
                        ", 0, MPI_COMM_WORLD);\n";
            }
            code += "    }\n";
        }

        code += "    ctx." + calcName +
                "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_" +
                calcName + "));\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    ctx." + calcName + "_cols = ctx.pattern_" + calcName +
                    ".partition_shape[1];\n";
        }

        code += "    if (ctx.local_item_count > 0) {\n";
        code += "        ctx.buf_" + calcName +
                " = std::make_unique<sycl::buffer<" +
                calcParam->getBasicType() + ", 1>>(ctx.local_" + calcName +
                ".data(), sycl::range<1>(ctx.local_" + calcName + ".size()));\n";
        code += "        ctx.slots_buf_" + calcName +
                " = std::make_unique<sycl::buffer<int32_t, 1>>(ctx.slots_" +
                calcName + ".data(), sycl::range<1>(ctx.slots_" + calcName +
                ".size()));\n";
        code += "    }\n";
    }

    code += "    ctx.has_halo = (ctx.mpi_size > 1 && ctx.local_item_count > 0);\n";
    code += "    if (ctx.has_halo) {\n";
    code +=
        "        auto my_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& name = calc->getParam(paramIdx)->getName();
        code += "        ctx.halo_" + name +
                " = dacpp::mpi::computeParamHalo(\n";
        code += "            ctx.pattern_" + name + ", ctx.pattern_" + name +
                ".mode,\n";
        code += "            my_range, ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    }
    code += "    }\n";
    code += "}\n\n";

    code += "void " + generated.submitName + "(" + generated.ctxTypeName +
            "& ctx) {\n";
    code += "    if (ctx.local_item_count <= 0) return;\n";
    code += "    sycl::queue& q = ctx.q;\n";
    code +=
        "    const std::size_t local_item_count = static_cast<std::size_t>(ctx.local_item_count);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "    const int " + name + "_partition_size = ctx." + name +
                "_partition_size;\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + name + "_cols = ctx." + name +
                    "_cols;\n";
        }
    }
    code += "    {\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "            auto acc_" + name + " = ctx.buf_" + name +
                "->get_access<" + toAccessorMode(paramModes[paramIdx]) +
                ">(h);\n";
        code += "            auto slots_acc_" + name +
                " = ctx.slots_buf_" + name +
                "->get_access<sycl::access::mode::read>(h);\n";
    }
    code +=
        "            h.parallel_for(sycl::range<1>(local_item_count), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto* data_" + name + " = acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                auto* slots_" + name + " = slots_acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (inferViewRank(shellParam, calcParam) <= 1) {
            code += "                " +
                    toViewType(shellParam, calcParam, paramModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name +
                    ", item_linear * " + name + "_partition_size};\n";
        } else {
            code += "                " +
                    toViewType(shellParam, calcParam, paramModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name +
                    ", item_linear * " + name + "_partition_size, " + name +
                    "_cols};\n";
        }
    }
    code += "                " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += "view_" + calc->getParam(paramIdx)->getName();
        if (paramIdx + 1 != calc->getNumParams()) {
            code += ", ";
        }
    }
    code += ");\n";
    code += "            });\n";
    code += "        });\n";
    code += "    }\n";
    code += "}\n\n";

    code += "void " + generated.haloName + "(" + generated.ctxTypeName +
            "& ctx) {\n";
    code += "    ctx.q.wait();\n";
    code += "    if (!ctx.has_halo) return;\n";
    code += "    if (ctx.local_item_count <= 0) return;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const IOTYPE mode = paramModes[paramIdx];
        if (mode == IOTYPE::READ) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& baseType = calcParam->getBasicType();
        const std::string mpiType = mpiDatatypeFor(baseType);

        code += "    {\n";
        code += "        sycl::host_accessor ha_" + name + "(*ctx.buf_" + name +
                ", sycl::read_only);\n";
        code += "        for (std::size_t i = 0; i < ctx.local_" + name +
                ".size(); ++i)\n";
        code += "            ctx.local_" + name + "[i] = ha_" + name + "[i];\n";
        code += "        MPI_Datatype mpi_dt_" + name + " = " + mpiType + ";\n";
        code += "        dacpp::mpi::exchangeHalo(ctx.local_" + name +
                ", ctx.halo_" + name + ", &mpi_dt_" + name + ");\n";
        code += "        sycl::host_accessor ha_w_" + name + "(*ctx.buf_" +
                name + ", sycl::write_only);\n";
        code += "        for (std::size_t i = 0; i < ctx.local_" + name +
                ".size(); ++i)\n";
        code += "            ha_w_" + name + "[i] = ctx.local_" + name +
                "[i];\n";
        code += "    }\n";
    }
    code += "}\n\n";

    code += "void " + generated.syncName + "(" + generated.ctxTypeName +
            "& ctx";
    if (!shellParamSig.empty()) {
        code += ", " + shellParamSig;
    }
    code += ") {\n";
    code += "    ctx.q.wait();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (!transferPolicy.needsSyncGather[static_cast<std::size_t>(paramIdx)]) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());
        const bool needsBcast = tensorNeedsBroadcast(dacppFile, tensorName);

        code += "    {\n";
        code += "    auto wb_" + calcName +
                " = dacpp::mpi::build_writeback_values(ctx.local_" + calcName +
                ", ctx.pack_" + calcName + ");\n";
        code += "    const auto& wb_globals_" + calcName +
                " = ctx.pack_" + calcName +
                ".writeback_globals.empty() ? ctx.pack_" + calcName +
                ".globals : ctx.pack_" + calcName + ".writeback_globals;\n";
        if (needsBcast) {
            code += "    std::vector<" + calcParam->getBasicType() +
                    "> synced_" + calcName + ";\n";
        }
        code += "    int send_count_" + calcName +
                " = static_cast<int>(wb_globals_" + calcName + ".size());\n";
        code += "    std::vector<int> recvcounts_" + calcName + ";\n";
        code += "    std::vector<int> recvdispls_" + calcName + ";\n";
        code += "    std::vector<int64_t> global_recv_globals_" + calcName +
                ";\n";
        code += "    std::vector<" + calcParam->getBasicType() +
                "> global_recv_values_" + calcName + ";\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        recvcounts_" + calcName +
                ".resize(ctx.mpi_size);\n";
        code += "        recvdispls_" + calcName +
                ".resize(ctx.mpi_size);\n";
        code += "    }\n";
        code += "    MPI_Gather(&send_count_" + calcName +
                ", 1, MPI_INT, ctx.mpi_rank == 0 ? recvcounts_" + calcName +
                ".data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        int current_displ = 0;\n";
        code += "        for (int r = 0; r < ctx.mpi_size; ++r) {\n";
        code += "            recvdispls_" + calcName +
                "[r] = current_displ;\n";
        code += "            current_displ += recvcounts_" + calcName +
                "[r];\n";
        code += "        }\n";
        code += "        global_recv_globals_" + calcName +
                ".resize(current_displ);\n";
        code += "        global_recv_values_" + calcName +
                ".resize(current_displ);\n";
        code += "    }\n";
        code += "    MPI_Gatherv(const_cast<int64_t*>(wb_globals_" + calcName +
                ".data()), send_count_" + calcName +
                ", MPI_LONG_LONG, ctx.mpi_rank == 0 ? global_recv_globals_" +
                calcName +
                ".data() : nullptr, ctx.mpi_rank == 0 ? recvcounts_" + calcName +
                ".data() : nullptr, ctx.mpi_rank == 0 ? recvdispls_" + calcName +
                ".data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);\n";
        {
            const std::string payloadSendCount = mpiPayloadCountExpr(
                "send_count_" + calcName, calcParam->getBasicType());
            if (usesByteTransport(calcParam->getBasicType())) {
                code += "    { std::vector<int> rc_bytes = recvcounts_" +
                        calcName + ";\n";
                code += "      std::vector<int> rd_bytes = recvdispls_" +
                        calcName + ";\n";
                code += "      if (ctx.mpi_rank == 0) { for (int r = 0; r < ctx.mpi_size; ++r) { rc_bytes[r] *= sizeof(" +
                        calcParam->getBasicType() + "); rd_bytes[r] *= sizeof(" +
                        calcParam->getBasicType() + "); } }\n";
                code += "      MPI_Gatherv(wb_" + calcName + ".data(), " +
                        payloadSendCount + ", " + mpiType +
                        ", ctx.mpi_rank == 0 ? global_recv_values_" + calcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? rc_bytes.data() : nullptr, ctx.mpi_rank == 0 ? rd_bytes.data() : nullptr, " +
                        mpiType + ", 0, MPI_COMM_WORLD); }\n";
            } else {
                code += "    MPI_Gatherv(wb_" + calcName + ".data(), " +
                        payloadSendCount + ", " + mpiType +
                        ", ctx.mpi_rank == 0 ? global_recv_values_" + calcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? recvcounts_" +
                        calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? recvdispls_" +
                        calcName + ".data() : nullptr, " + mpiType +
                        ", 0, MPI_COMM_WORLD);\n";
            }
        }
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        std::vector<" + calcParam->getBasicType() +
                "> global_out_" + calcName + ";\n";
        code += "        " + tensorName + ".tensor2Array(global_out_" +
                calcName + ");\n";
        code += "        dacpp::mpi::apply_writeback_by_globals(global_recv_values_" +
                calcName + ", global_recv_globals_" + calcName +
                ", global_out_" + calcName + ");\n";
        code += "        " + tensorName + ".array2Tensor(global_out_" + calcName +
                ");\n";
        if (needsBcast) {
            code += "        synced_" + calcName + " = global_out_" + calcName +
                    ";\n";
        }
        code += "    }\n";
        if (needsBcast) {
            const std::string payloadSyncedCount = mpiPayloadCountExpr(
                "synced_count_" + calcName, calcParam->getBasicType());
            code += "    int synced_count_" + calcName + " = 0;\n";
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        synced_count_" + calcName +
                    " = static_cast<int>(synced_" + calcName + ".size());\n";
            code += "    }\n";
            code += "    MPI_Bcast(&synced_count_" + calcName +
                    ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
            code += "    if (ctx.mpi_rank != 0) {\n";
            code += "        synced_" + calcName + ".resize(synced_count_" +
                    calcName + ");\n";
            code += "    }\n";
            code += "    if (synced_count_" + calcName + " > 0) {\n";
            code += "        MPI_Bcast(synced_" + calcName + ".data(), " +
                    payloadSyncedCount + ", " + mpiType +
                    ", 0, MPI_COMM_WORLD);\n";
            code += "    }\n";
            code += "    if (ctx.mpi_rank != 0) {\n";
            code += "        " + tensorName + ".array2Tensor(synced_" + calcName +
                    ");\n";
            code += "    }\n";
        }
        code += "    }\n";
    }
    code += "}\n\n";

    return code;
}

MPIRegionGeneratedCode buildMPIRegionCode(DacppFile* dacppFile,
                                          Expression* expr) {
    MPIRegionGeneratedCode generated = buildMPIRegionNames(expr);
    if (!expr || !expr->getShell() || !expr->getCalc()) {
        return generated;
    }

    Shell* shell = expr->getShell();
    Calc* calc = expr->getCalc();
    const auto paramModes = inferEffectiveParamModes(shell, calc);
    const auto transferPolicy =
        analyzeMPIRegionTransferPolicy(dacppFile, expr, paramModes);
    generated.definitions =
        buildMPIRegionCodegen(dacppFile, expr, transferPolicy);
    return generated;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
