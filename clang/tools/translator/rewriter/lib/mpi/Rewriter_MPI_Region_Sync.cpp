#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

std::string buildMPIRegionSyncCode(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const MPIRegionGeneratedCode& generated,
    const std::vector<IOTYPE>& storageModes,
    const MPIRegionTransferPolicy& transferPolicy,
    const std::string& shellParamSig) {
    if (!shell || !calc) {
        return "";
    }

    std::string code;

    code += "void " + generated.syncName + "(" + generated.ctxTypeName +
            "& ctx";
    if (!shellParamSig.empty()) {
        code += ", " + shellParamSig;
    }
    code += ") {\n";
    code += "    ctx.q.wait();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const IOTYPE mode = storageModes[paramIdx];
        if (mode == IOTYPE::WRITE || mode == IOTYPE::READ_WRITE) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "    if (ctx.local_item_count > 0) {\n";
            code += "        sycl::host_accessor ha_sync_" + name +
                    "(*ctx.buf_" + name + ", sycl::read_only);\n";
            code += "        for (std::size_t i = 0; i < ctx.local_" + name +
                    ".size(); ++i)\n";
            code += "            ctx.local_" + name + "[i] = ha_sync_" + name +
                    "[i];\n";
            code += "    }\n";
        }
    }
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

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
