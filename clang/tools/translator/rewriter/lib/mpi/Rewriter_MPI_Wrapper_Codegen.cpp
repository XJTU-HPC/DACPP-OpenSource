#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

std::string buildWrapperCode(DacppFile* dacppFile,
                             Shell* shell,
                             Calc* calc,
                             const clang::BinaryOperator* dacExpr) {
    const std::string wrapperName = shell->getName() + "_" + calc->getName();
    const auto paramModes = inferEffectiveParamModes(shell, calc);

    std::string signature;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (paramType.size() > 0 && paramType.back() != '&' && paramType.back() != '*') {
            paramType += "&";
        }
        signature += paramType + " " + param->getName();
        if (paramIdx + 1 != shell->getNumParams()) {
            signature += ", ";
        }
    }

    const auto splitMeta = collectSplitBindMeta(shell);

    std::string code;
    code += "void " + wrapperName + "(" + signature + ") {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);\n";
    code += "    dacpp::mpi::resetCollectPositionsProfile();\n";
    code += "    auto dacpp_wrapper_start = std::chrono::steady_clock::now();\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    code += "    std::vector<int64_t> binding_split_sizes;\n";
    code += buildPatternInitCode(shell, calc, splitMeta, paramModes);
    code += "    int64_t total_items = 1;\n";
    code += "    for (int64_t split_size : binding_split_sizes) total_items *= split_size;\n";
    code += "    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);\n";
    code += "    const int64_t local_item_count = item_range.size();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        code += "    pattern_" + calcName + ".bind_split_sizes = binding_split_sizes;\n";
    }

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const IOTYPE mode = paramModes[paramIdx];
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "pattern_" + calcName;
        const std::string packName = "pack_" + calcName;
        const std::string slotsName = "slots_" + calcName;
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_" + calcName;
        const std::string wbName = "writeback_" + calcName;
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());

        code += "    auto plan_" + calcName + " = " +
                buildPackPlanBuilderExpr(mode, "item_range", patternName) + ";\n";
        code += "    auto& " + packName + " = plan_" + calcName + ".pack;\n";
        code += "    auto& " + slotsName + " = plan_" + calcName + ".compact_slots;\n";
        code += "    auto& key_offsets_" + calcName + " = plan_" + calcName + ".item_key_offsets;\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> " + localName + "(" +
                packName + ".globals.size());\n";

        if (mode != IOTYPE::WRITE) {
            code += "    std::vector<int> sendcounts_" + calcName + ";\n";
            code += "    std::vector<int> displs_" + calcName + ";\n";
            code += "    int local_global_count_" + calcName + " = static_cast<int>(" +
                    packName + ".globals.size());\n";
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
            code += "            const int64_t* r_globals_ptr = gathered_globals_" + calcName + ".data() + global_displs_" + calcName + "[r];\n";
            code += "            auto r_values = dacpp::mpi::pack_values_by_globals_parallel_range(" + globalName + ", r_globals_ptr, static_cast<std::size_t>(global_counts_" + calcName + "[r]));\n";
            code += "            int r_count = static_cast<int>(r_values.size());\n";
            code += "            sendcounts_" + calcName + "[r] = r_count;\n";
            code += "            displs_" + calcName + "[r] = current_displ;\n";
            code += "            current_displ += r_count;\n";
            code += "            sendbuf_" + calcName + ".insert(sendbuf_" + calcName + ".end(), r_values.begin(), r_values.end());\n";
            code += "        }\n";
            code += "    }\n";
            code += "    " + localName + ".resize(static_cast<std::size_t>(local_global_count_" + calcName + "));\n";
            if (usesByteTransport(calcParam->getBasicType())) {
                code += "    std::vector<int> sendcounts_bytes_" + calcName + " = sendcounts_" + calcName + ";\n";
                code += "    std::vector<int> displs_bytes_" + calcName + " = displs_" + calcName + ";\n";
                code += "    if (mpi_rank == 0) {\n";
                code += "        for (int r = 0; r < mpi_size; ++r) {\n";
                code += "            sendcounts_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
                code += "            displs_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
                code += "        }\n";
                code += "    }\n";
                std::string payloadRecvCount =
                    mpiPayloadCountExpr("local_global_count_" + calcName, calcParam->getBasicType());
                code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_" + calcName + ".data() : nullptr, mpi_rank == 0 ? displs_bytes_" + calcName + ".data() : nullptr, " + mpiType + ", " + localName + ".data(), " + payloadRecvCount + ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            } else {
                code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? sendcounts_" + calcName + ".data() : nullptr, mpi_rank == 0 ? displs_" + calcName + ".data() : nullptr, " + mpiType + ", " + localName + ".data(), local_global_count_" + calcName + ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
        }

        code += "    const int " + calcName + "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(" +
                patternName + "));\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + calcName + "_cols = " + patternName + ".partition_shape[1];\n";
        }
    }

    code += "    if (local_item_count > 0) {\n";
    code += "        {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string localName = "local_" + name;
        const std::string slotsName = "slots_" + name;
        code += "            sycl::buffer<" + calcParam->getBasicType() + ", 1> buffer_" + name +
                "(" + localName + ".data(), sycl::range<1>(" + localName + ".size()));\n";
        code += "            sycl::buffer<int32_t, 1> slots_buffer_" + name + "(" + slotsName +
                ".data(), sycl::range<1>(" + slotsName + ".size()));\n";
        code += "            sycl::buffer<int32_t, 1> key_offsets_buffer_" + name +
                "(key_offsets_" + name + ".data(), sycl::range<1>(key_offsets_" + name + ".size()));\n";
    }
    code += "            q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto acc_" + name + " = buffer_" + name + ".get_access<" +
                toAccessorMode(paramModes[paramIdx]) + ">(h);\n";
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
                " = acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* slots_" + name +
                " = slots_acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* key_offsets_" + name +
                " = key_offsets_acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (inferViewRank(shellParam, calcParam) <= 1) {
            code += "                    " + toViewType(shellParam, calcParam, paramModes[paramIdx]) + " view_" + name +
                    "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                    "[item_linear]};\n";
        } else {
            code += "                    " + toViewType(shellParam, calcParam, paramModes[paramIdx]) + " view_" + name +
                    "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
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
        const std::string packName = "pack_" + calcName;
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_out_" + calcName;
        const std::string wbName = "writeback_" + calcName;
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());
        bool needsBcast = tensorNeedsBroadcast(dacppFile, tensorName, dacExpr);

        code += "    auto " + wbName + " = dacpp::mpi::build_writeback_values_parallel(" + localName + ", " + packName + ");\n";
        code += "    const auto& writeback_globals_" + calcName + " = " + packName + ".writeback_globals.empty() ? " + packName + ".globals : " + packName + ".writeback_globals;\n";
        code += "    int send_count_" + calcName + " = static_cast<int>(writeback_globals_" + calcName + ".size());\n";
        code += "    std::vector<int> recvcounts_" + calcName + ";\n";
        code += "    std::vector<int> recvdispls_" + calcName + ";\n";
        code += "    std::vector<int64_t> global_recv_globals_" + calcName + ";\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> global_recv_values_" + calcName + ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        recvcounts_" + calcName + ".resize(mpi_size);\n";
        code += "        recvdispls_" + calcName + ".resize(mpi_size);\n";
        code += "    }\n";
        code += "    MPI_Gather(&send_count_" + calcName + ", 1, MPI_INT, mpi_rank == 0 ? recvcounts_" + calcName + ".data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        int current_displ = 0;\n";
        code += "        for (int r = 0; r < mpi_size; ++r) {\n";
        code += "            recvdispls_" + calcName + "[r] = current_displ;\n";
        code += "            current_displ += recvcounts_" + calcName + "[r];\n";
        code += "        }\n";
        code += "        global_recv_globals_" + calcName + ".resize(current_displ);\n";
        code += "        global_recv_values_" + calcName + ".resize(current_displ);\n";
        code += "    }\n";
        code += "    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_" + calcName + ".data()), send_count_" + calcName + ", MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvcounts_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvdispls_" + calcName + ".data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);\n";
        if (usesByteTransport(calcParam->getBasicType())) {
            code += "    std::vector<int> recvcounts_bytes_" + calcName + " = recvcounts_" + calcName + ";\n";
            code += "    std::vector<int> recvdispls_bytes_" + calcName + " = recvdispls_" + calcName + ";\n";
            code += "    if (mpi_rank == 0) {\n";
            code += "        for (int r = 0; r < mpi_size; ++r) {\n";
            code += "            recvcounts_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
            code += "            recvdispls_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
            code += "        }\n";
            code += "    }\n";
            std::string payloadSendCount2 =
                mpiPayloadCountExpr("send_count_" + calcName, calcParam->getBasicType());
            code += "    MPI_Gatherv(" + wbName + ".data(), " + payloadSendCount2 + ", " + mpiType + ", mpi_rank == 0 ? global_recv_values_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvcounts_bytes_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvdispls_bytes_" + calcName + ".data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Gatherv(" + wbName + ".data(), send_count_" + calcName + ", " + mpiType + ", mpi_rank == 0 ? global_recv_values_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvcounts_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvdispls_" + calcName + ".data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    if (mpi_rank == 0) {\n";
        code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName + ";\n";
        code += "        " + tensorName + ".tensor2Array(" + globalName + ");\n";
        code += "        dacpp::mpi::apply_writeback_by_globals(global_recv_values_" + calcName + ", global_recv_globals_" + calcName + ", " + globalName + ");\n";
        code += "        " + tensorName + ".array2Tensor(" + globalName + ");\n";
            if (needsBcast) {
                const std::string bcastCountExpr =
                    mpiPayloadCountExpr(tensorName + ".getSize()", calcParam->getBasicType());
                code += "        if (!" + globalName + ".empty()) {\n";
                code += "            MPI_Bcast(" + globalName + ".data(), " + bcastCountExpr + ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                code += "        }\n";
            }
        code += "    } else ";
        if (needsBcast) {
            code += "{\n";
            code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName + "(static_cast<std::size_t>(" + tensorName + ".getSize()));\n";
            code += "        if (!" + globalName + ".empty()) {\n";
            code += "            MPI_Bcast(" + globalName + ".data(), " + mpiPayloadCountExpr(tensorName + ".getSize()", calcParam->getBasicType()) + ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
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
    code += "        std::fprintf(stderr, \"[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\\n\", \"" + wrapperName + "\", dacpp_wrapper_max_ms);\n";
    code += "    }\n";
    code += "    dacpp::mpi::reportCollectPositionsProfile(\"" + wrapperName + "\", MPI_COMM_WORLD);\n";
    code += "}\n";
    return code;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
