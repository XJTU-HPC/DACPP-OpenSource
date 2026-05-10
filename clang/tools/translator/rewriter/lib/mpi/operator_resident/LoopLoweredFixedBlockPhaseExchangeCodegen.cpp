#include <string>

#include "DacppStructure.h"
#include "OperatorResidentCodegen_Internal.h"
#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

const ParamAccessPlan* fixedBlockReader(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::FixedBlock && param.reads &&
            !param.writes) {
            return &param;
        }
    }
    return nullptr;
}

const ParamAccessPlan* fixedBlockWriter(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::FixedBlock && param.writes &&
            !param.reads) {
            return &param;
        }
    }
    return nullptr;
}

std::string narrowMpi(const std::string& expr, const std::string& what) {
    return "dacpp::mpi::operator_resident::narrow_mpi_count_or_abort("
           "static_cast<int64_t>(" +
           expr + "), \"" + what + "\")";
}

}  // namespace

std::string buildLoopLoweredFixedBlockPhaseExchangeFamilyCode(
    const std::string& baseName,
    DacppFile*,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = fixedBlockReader(plan);
    const ParamAccessPlan* writer = fixedBlockWriter(plan);
    if (!reader || !writer || !plan.exprNode.shell || !plan.exprNode.calc ||
        !plan.orLoopLower.fixedBlockPhaseExchange.enabled) {
        return {};
    }

    Shell* shell = plan.exprNode.shell;
    Calc* calc = plan.exprNode.calc;
    const int exprIndex = plan.exprIndex;
    const std::string ctxName =
        operatorResidentContextTypeName(shell, calc, exprIndex);
    const std::string initName =
        operatorResidentInitFunctionName(shell, calc, exprIndex);
    const std::string runName =
        operatorResidentRunFunctionName(shell, calc, exprIndex);
    const std::string materializeName =
        operatorResidentMaterializeFunctionName(shell, calc, exprIndex);

    const std::string elemType = ::dacppTranslator::mpi_rewriter::
        operator_resident::elemType(plan, *reader);
    const std::string mpiType = mpiDatatypeFor(elemType);
    const std::string sig = wrapperSignature(plan);
    const std::string readerVar = paramVarName(*reader);
    const std::string writerVar = paramVarName(*writer);
    const int blockSize = plan.orLoopLower.fixedBlockPhaseExchange.blockSize;
    const int blockStride =
        plan.orLoopLower.fixedBlockPhaseExchange.blockStride;
    const int phaseShiftOffset =
        plan.orLoopLower.fixedBlockPhaseExchange.phaseShiftOffset;
    const int64_t provenEvenTotal =
        plan.orLoopLower.fixedBlockPhaseExchange.provenEvenTotal;
    (void)blockStride;

    std::string code;

    // Context type
    code += "struct " + ctxName + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    int64_t total = 0;\n";
    code += "    int block_size = " + std::to_string(blockSize) + ";\n";
    code += "    int phase_shift_offset = " +
            std::to_string(phaseShiftOffset) + ";\n";
    code +=
        "    dacpp::mpi::operator_resident::FixedBlockPhaseExchangeLayout layout{};\n";
    code += "    std::vector<int> elem_counts;\n";
    code += "    std::vector<int> elem_displs;\n";
    code += "    std::vector<" + elemType + "> local;\n";
    code += "};\n";

    // init: scatter once
    code += "void " + initName + "(" + ctxName + "& ctx, " + sig + ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.total = static_cast<int64_t>(" + readerVar +
            ".getSize());\n";
    // Guard: reject at runtime if the reader total does not match the writer's
    // statically proven even total. The static gate in OperatorChainAnalysis
    // only proves the writer (locally declared phase-A output). A runtime
    // reader with odd length would otherwise silently break phase-B semantics.
    code += "    dacpp::mpi::operator_resident::"
            "check_fixed_block_phase_exchange_total("
            "ctx.total, static_cast<int64_t>(" +
            writerVar + ".getSize()));\n";
    code += "    dacpp::mpi::operator_resident::"
            "check_fixed_block_phase_exchange_total("
            "ctx.total, static_cast<int64_t>(" +
            std::to_string(provenEvenTotal) + "));\n";
    code += "    ctx.layout = dacpp::mpi::operator_resident::"
            "fixed_block_phase_exchange_layout(ctx.total, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d("
            "ctx.total, ctx.mpi_size, ctx.elem_counts, ctx.elem_displs);\n";
    code += "    ctx.local.assign(static_cast<std::size_t>(ctx.layout.local_count), " +
            elemType + "{});\n";
    code += "    std::vector<" + elemType + "> __or_global;\n";
    code += "    if (ctx.mpi_rank == 0) {\n";
    code += "        " + readerVar + ".tensor2Array(__or_global);\n";
    code += "    }\n";
    code += "    const int __or_local_count = " +
            narrowMpi("ctx.layout.local_count",
                      "[DACPP][MPI][OR][P5][PhaseExchange] scatter count exceeds MPI int range") +
            ";\n";
    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? __or_global.data() : nullptr,"
            " ctx.mpi_rank == 0 ? ctx.elem_counts.data() : nullptr,"
            " ctx.mpi_rank == 0 ? ctx.elem_displs.data() : nullptr, " +
            mpiType + ", ctx.local.data(), __or_local_count, " + mpiType +
            ", 0, MPI_COMM_WORLD);\n";
    // Suppress unused warnings on writer
    code += "    (void)" + writerVar + ";\n";
    code += "}\n";

    // run: phase A + phase B compare-swap
    code += "void " + runName + "(" + ctxName + "& ctx, " + sig + ") {\n";
    code += "    auto __or_compare_swap = [](" + elemType + "* pair) {\n";
    code += "        " + elemType + " __or_in[2] = {pair[0], pair[1]};\n";
    code += "        dacpp::mpi::ContiguousView1D<const " + elemType +
            "> view_" + reader->calcParamName + "{__or_in, 0};\n";
    code += "        dacpp::mpi::ContiguousView1D<" + elemType + "> view_" +
            writer->calcParamName + "{pair, 0};\n";
    code += "        " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + calc->getParam(paramIdx)->getName();
    }
    code += ");\n";
    code += "    };\n";
    code += "    dacpp::mpi::operator_resident::fixed_block_phase_exchange_step("
            "ctx.local, ctx.layout.local_begin, ctx.total, ctx.mpi_rank, ctx.mpi_size, "
            "ctx.block_size, ctx.phase_shift_offset, " +
            mpiType + ", __or_compare_swap);\n";
    code += "    (void)" + readerVar + ";\n";
    code += "    (void)" + writerVar + ";\n";
    code += "}\n";

    // materialize: gather to root, broadcast for host-visible source semantics.
    // The reader parameter is declared `const` in the shell signature but the
    // original loop body wrote to it via post-stmts; phase-exchange absorbs
    // those writes, so we need a non-const alias here.
    code += "void " + materializeName + "(" + ctxName + "& ctx, " + sig +
            ") {\n";
    code += "    auto& __or_writable_reader = const_cast<std::remove_const_t<"
            "std::remove_reference_t<decltype(" +
            readerVar + ")>>&>(" + readerVar + ");\n";
    code += "    std::vector<" + elemType + "> __or_global(ctx.mpi_rank == 0 ? "
            "static_cast<std::size_t>(ctx.total) : 0);\n";
    code += "    const int __or_local_count = " +
            narrowMpi("ctx.layout.local_count",
                      "[DACPP][MPI][OR][P5][PhaseExchange] gather count exceeds MPI int range") +
            ";\n";
    code += "    MPI_Gatherv(ctx.local.data(), __or_local_count, " + mpiType +
            ", ctx.mpi_rank == 0 ? __or_global.data() : nullptr,"
            " ctx.mpi_rank == 0 ? ctx.elem_counts.data() : nullptr,"
            " ctx.mpi_rank == 0 ? ctx.elem_displs.data() : nullptr, " +
            mpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    if (ctx.mpi_rank == 0) {\n";
    code += "        __or_writable_reader.array2Tensor(__or_global);\n";
    code += "    } else {\n";
    code += "        __or_global.assign(static_cast<std::size_t>(ctx.total), " +
            elemType + "{});\n";
    code += "    }\n";
    code += "    if (ctx.total > 0) {\n";
    code += "        MPI_Bcast(__or_global.data(), " +
            narrowMpi("ctx.total",
                      "[DACPP][MPI][OR][P5][PhaseExchange] broadcast count exceeds MPI int range") +
            ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "        if (ctx.mpi_rank != 0) {\n";
    code += "            __or_writable_reader.array2Tensor(__or_global);\n";
    code += "        }\n";
    code += "    }\n";
    code += "    (void)" + writerVar + ";\n";
    code += "}\n";

    // Wrapper that simply runs the standalone init/run/materialize sequence,
    // used only when this site is invoked outside a recognized loop.
    code += "void " + baseName + "(" + sig + ") {\n";
    code += "    " + ctxName + " ctx;\n";
    code += "    " + initName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "    " + runName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "    " + materializeName + "(ctx";
    for (const auto& param : plan.params) {
        code += ", " + paramVarName(param);
    }
    code += ");\n";
    code += "}\n";

    return code;
}

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
