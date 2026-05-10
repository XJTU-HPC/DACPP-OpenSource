#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "OperatorResidentCodegen_Internal.h"

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

std::string fixedBlockCountExpr(const std::string& tensorExpr,
                                int blockSize,
                                int blockStride) {
    return "dacpp::mpi::operator_resident::fixed_block_count_1d(" +
           tensorExpr + ".getShape(0), " + std::to_string(blockSize) + ", " +
           std::to_string(blockStride) + ")";
}

std::string checkedMulExpr(const std::string& lhs,
                           const std::string& rhs,
                           const std::string& what) {
    return "dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(" +
           lhs + "), static_cast<int64_t>(" + rhs + "), \"" + what + "\")";
}

std::string checkedMpiCountExpr(const std::string& expr,
                                const std::string& what) {
    return "dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(" +
           expr + "), \"" + what + "\")";
}

std::string checkedMpiPayloadCountExpr(const std::string& elemCountExpr,
                                       const std::string& type,
                                       const std::string& what) {
    if (usesByteTransport(type)) {
        return checkedMpiCountExpr(
            checkedMulExpr(elemCountExpr, "sizeof(" + type + ")", what),
            what);
    }
    return checkedMpiCountExpr(elemCountExpr, what);
}

void emitBlockCountsDispls(std::string& code,
                           const std::string& totalBlocks,
                           const std::string& blockSize) {
    code += "    std::vector<int> __or_block_counts(ctx_mpi_size);\n";
    code += "    std::vector<int> __or_block_displs(ctx_mpi_size);\n";
    code += "    std::vector<int> __or_elem_counts(ctx_mpi_size);\n";
    code += "    std::vector<int> __or_elem_displs(ctx_mpi_size);\n";
    code += "    for (int r = 0; r < ctx_mpi_size; ++r) {\n";
    code += "        const auto __or_r_range = dacpp::mpi::operator_resident::rank_range_1d(" +
            totalBlocks + ", r, ctx_mpi_size);\n";
    code += "        __or_block_counts[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_r_range.count, \"[DACPP][MPI][OR][P5] fixed block count exceeds MPI int range\");\n";
    code += "        __or_block_displs[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_r_range.begin, \"[DACPP][MPI][OR][P5] fixed block displacement exceeds MPI int range\");\n";
    code += "        __or_elem_counts[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_r_range.count, " +
            blockSize + ", \"[DACPP][MPI][OR][P5] fixed block elem count overflow\"), \"[DACPP][MPI][OR][P5] fixed block elem count exceeds MPI int range\");\n";
    code += "        __or_elem_displs[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_r_range.begin, " +
            blockSize + ", \"[DACPP][MPI][OR][P5] fixed block elem displacement overflow\"), \"[DACPP][MPI][OR][P5] fixed block elem displacement exceeds MPI int range\");\n";
    code += "    }\n";
}

} // namespace

std::string buildFixedBlockWrapperCode(const std::string& wrapperName,
                                       DacppFile*,
                                       const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = fixedBlockReader(plan);
    const ParamAccessPlan* writer = fixedBlockWriter(plan);
    if (!reader || !writer) {
        return "";
    }

    const int blockSize = writer->fixedBlockSize;
    const int blockStride = writer->fixedBlockStride;
    const std::string readerType = elemType(plan, *reader);
    const std::string writerType = elemType(plan, *writer);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    const std::string writerMpiType = mpiDatatypeFor(writerType);
    const std::string blockSizeExpr = "static_cast<int64_t>(" +
                                      std::to_string(blockSize) + ")";
    std::string code;

    code += "void " + wrapperName + "(" + wrapperSignature(plan) + ") {\n";
    code += "    int ctx_mpi_rank = 0;\n";
    code += "    int ctx_mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx_mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx_mpi_size);\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    code += "    const int64_t __or_fixed_block_size = " +
            std::to_string(blockSize) + ";\n";
    code += "    const int64_t __or_fixed_block_stride = " +
            std::to_string(blockStride) + ";\n";
    code += "    const int64_t __or_reader_blocks = " +
            fixedBlockCountExpr(paramVarName(*reader), blockSize, blockStride) +
            ";\n";
    code += "    const int64_t __or_writer_blocks = " +
            fixedBlockCountExpr(paramVarName(*writer), blockSize, blockStride) +
            ";\n";
    code += "    if (__or_reader_blocks != __or_writer_blocks) {\n";
    code += "        if (ctx_mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR][P5] fixed block reader/writer block count mismatch\\n\");\n";
    code += "        MPI_Abort(MPI_COMM_WORLD, 6);\n";
    code += "    }\n";
    code += "    const int64_t __or_total_blocks = __or_writer_blocks;\n";
    code += "    const auto __or_block_range = dacpp::mpi::operator_resident::rank_range_1d(__or_total_blocks, ctx_mpi_rank, ctx_mpi_size);\n";
    code += "    const int64_t __or_local_block_count = __or_block_range.count;\n";
    code += "    const int64_t __or_local_elem_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_local_block_count, __or_fixed_block_size, \"[DACPP][MPI][OR][P5] fixed block local element count overflow\");\n";
    emitBlockCountsDispls(code, "__or_total_blocks", blockSizeExpr);

    code += "    std::vector<" + readerType + "> " + localName(*reader) +
            "(static_cast<std::size_t>(__or_local_elem_count));\n";
    code += "    std::vector<" + readerType + "> " + globalName(*reader) +
            ";\n";
    code += "    if (ctx_mpi_rank == 0) {\n";
    code += "        " + paramVarName(*reader) + ".tensor2Array(" +
            globalName(*reader) + ");\n";
    code += "    }\n";
    code += "    MPI_Scatterv(ctx_mpi_rank == 0 ? " + globalName(*reader) +
            ".data() : nullptr, ctx_mpi_rank == 0 ? __or_elem_counts.data() : nullptr, ctx_mpi_rank == 0 ? __or_elem_displs.data() : nullptr, " +
            readerMpiType + ", " + localName(*reader) +
            ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_local_elem_count, \"[DACPP][MPI][OR][P5] fixed block scatter count exceeds MPI int range\"), " +
            readerMpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    std::vector<" + writerType + "> " + localName(*writer) +
            "(static_cast<std::size_t>(__or_local_elem_count), " +
            writerType + "{});\n";
    code += "    if (__or_local_block_count > 0) {\n";
    code += "        sycl::buffer<" + readerType + ", 1> __or_reader_buf(" +
            localName(*reader) + ".data(), sycl::range<1>(" +
            localName(*reader) + ".size()));\n";
    code += "        sycl::buffer<" + writerType + ", 1> __or_writer_buf(" +
            localName(*writer) + ".data(), sycl::range<1>(" +
            localName(*writer) + ".size()));\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    code += "            auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    code += "            auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_block_count)), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                const int __or_block_offset = item_linear * " +
            std::to_string(blockSize) + ";\n";
    code += "                auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    code += "                auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.paramIndex == reader->paramIndex) {
            code += "                dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_reader_data, __or_block_offset};\n";
        } else if (param.paramIndex == writer->paramIndex) {
            code += "                dacpp::mpi::ContiguousView1D<" +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_writer_data, __or_block_offset};\n";
        }
    }
    code += "                " + plan.exprNode.calc->getName() +
            "_mpi_local(";
    for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
         ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + plan.exprNode.calc->getParam(paramIdx)->getName();
    }
    code += ");\n";
    code += "            });\n";
    code += "        });\n";
    code += "        q.wait();\n";
    code += "    }\n";

    code += "    const int64_t __or_total_output_elems = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_total_blocks, __or_fixed_block_size, \"[DACPP][MPI][OR][P5] fixed block total output element overflow\");\n";
    code += "    std::vector<" + writerType + "> __or_gathered_" +
            writer->calcParamName + ";\n";
    code += "    if (ctx_mpi_rank == 0) {\n";
    code += "        __or_gathered_" + writer->calcParamName +
            ".resize(static_cast<std::size_t>(__or_total_output_elems));\n";
    code += "    }\n";
    code += "    MPI_Gatherv(" + localName(*writer) +
            ".data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_local_elem_count, \"[DACPP][MPI][OR][P5] fixed block gather count exceeds MPI int range\"), " +
            writerMpiType + ", ctx_mpi_rank == 0 ? __or_gathered_" +
            writer->calcParamName +
            ".data() : nullptr, ctx_mpi_rank == 0 ? __or_elem_counts.data() : nullptr, ctx_mpi_rank == 0 ? __or_elem_displs.data() : nullptr, " +
            writerMpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    std::vector<" + writerType + "> __or_materialized_" +
            writer->calcParamName + ";\n";
    code += "    if (ctx_mpi_rank == 0) {\n";
    code += "        " + paramVarName(*writer) + ".tensor2Array(__or_materialized_" +
            writer->calcParamName + ");\n";
    code += "        for (int64_t __or_idx = 0; __or_idx < __or_total_output_elems && __or_idx < static_cast<int64_t>(__or_materialized_" +
            writer->calcParamName + ".size()); ++__or_idx) {\n";
    code += "            __or_materialized_" + writer->calcParamName +
            "[static_cast<std::size_t>(__or_idx)] = __or_gathered_" +
            writer->calcParamName + "[static_cast<std::size_t>(__or_idx)];\n";
    code += "        }\n";
    code += "        " + paramVarName(*writer) + ".array2Tensor(__or_materialized_" +
            writer->calcParamName + ");\n";
    code += "        if (!__or_materialized_" + writer->calcParamName +
            ".empty()) {\n";
    code += "            MPI_Bcast(__or_materialized_" +
            writer->calcParamName + ".data(), " +
            checkedMpiPayloadCountExpr(
                paramVarName(*writer) + ".getSize()", writerType,
                "[DACPP][MPI][OR][P5] fixed block materialized broadcast count exceeds MPI int range") +
            ", " + writerMpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "        }\n";
    code += "    } else {\n";
    code += "        __or_materialized_" + writer->calcParamName +
            ".resize(static_cast<std::size_t>(" + paramVarName(*writer) +
            ".getSize()));\n";
    code += "        if (!__or_materialized_" + writer->calcParamName +
            ".empty()) {\n";
    code += "            MPI_Bcast(__or_materialized_" +
            writer->calcParamName + ".data(), " +
            checkedMpiPayloadCountExpr(
                paramVarName(*writer) + ".getSize()", writerType,
                "[DACPP][MPI][OR][P5] fixed block materialized broadcast count exceeds MPI int range") +
            ", " + writerMpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "        }\n";
    code += "        " + paramVarName(*writer) +
            ".array2Tensor(__or_materialized_" +
            writer->calcParamName + ");\n";
    code += "    }\n";
    code += "}\n";
    return code;
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
