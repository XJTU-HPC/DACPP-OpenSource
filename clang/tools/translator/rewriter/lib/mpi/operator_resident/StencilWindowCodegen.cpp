#include <string>

#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "OperatorResidentCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

const ParamAccessPlan* findStencilReader(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::StencilWindow) {
            return &param;
        }
    }
    return nullptr;
}

const ParamAccessPlan* findStencilWriter(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::OutputDirect && param.writes) {
            return &param;
        }
    }
    return nullptr;
}

}  // namespace

std::string buildStencilWindow2DWrapperCode(
    const std::string& wrapperName,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = findStencilReader(plan);
    const ParamAccessPlan* writer = findStencilWriter(plan);
    if (!reader || !writer) {
        return {};
    }

    const std::string readerType = elemType(plan, *reader);
    const std::string writerType = elemType(plan, *writer);
    const std::string readerMpiType = mpi_rewriter::mpiDatatypeFor(readerType);
    const std::string writerMpiType = mpi_rewriter::mpiDatatypeFor(writerType);
    const std::string readerArg = paramVarName(*reader);
    const std::string writerArg = paramVarName(*writer);
    const std::string calcName = plan.exprNode.calc->getName();

    std::string code;
    code += "void " + wrapperName + "(" + wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    code += "    const int64_t __or_input_rows = " + readerArg + ".getShape(0);\n";
    code += "    const int64_t __or_input_cols = " + readerArg + ".getShape(1);\n";
    code += "    const int64_t __or_output_rows = " + writerArg + ".getShape(0);\n";
    code += "    const int64_t __or_output_cols = " + writerArg + ".getShape(1);\n";
    code += "    const auto __or_output_row_range = dacpp::mpi::operator_resident::rank_range_1d(__or_output_rows, mpi_rank, mpi_size);\n";
    code += "    const int64_t __or_local_output_rows = __or_output_row_range.count;\n";
    code += "    const int64_t __or_output_row_begin = __or_output_row_range.begin;\n";
    code += "    const int64_t __or_local_item_count = __or_local_output_rows * __or_output_cols;\n";
    code += "    std::vector<int> __or_row_counts;\n";
    code += "    std::vector<int> __or_row_displs;\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(__or_output_rows, mpi_size, __or_row_counts, __or_row_displs);\n";
    code += "    std::vector<int> __or_counts(mpi_size);\n";
    code += "    std::vector<int> __or_displs(mpi_size);\n";
    code += "    for (int r = 0; r < mpi_size; ++r) {\n";
    code += "        __or_counts[r] = static_cast<int>(__or_row_counts[r] * __or_output_cols);\n";
    code += "        __or_displs[r] = static_cast<int>(__or_row_displs[r] * __or_output_cols);\n";
    code += "    }\n";
    code += "    std::vector<" + readerType + "> __or_global_" + reader->calcParamName + ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + readerArg + ".tensor2Array(__or_global_" + reader->calcParamName + ");\n";
    code += "    }\n";
    code += "    __or_global_" + reader->calcParamName + ".resize(static_cast<std::size_t>(__or_input_rows * __or_input_cols));\n";
    if (usesByte(plan, *reader)) {
        code += "    MPI_Bcast(__or_global_" + reader->calcParamName + ".data(), static_cast<int>(__or_input_rows * __or_input_cols * sizeof(" + readerType + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Bcast(__or_global_" + reader->calcParamName + ".data(), static_cast<int>(__or_input_rows * __or_input_cols), " + readerMpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    code += "    std::vector<" + writerType + "> __or_local_" + writer->calcParamName + "(static_cast<std::size_t>(__or_local_item_count));\n";
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        sycl::buffer<" + readerType + ", 1> __or_reader_buf(__or_global_" + reader->calcParamName + ".data(), sycl::range<1>(__or_global_" + reader->calcParamName + ".size()));\n";
    code += "        sycl::buffer<" + writerType + ", 1> __or_writer_buf(__or_local_" + writer->calcParamName + ".data(), sycl::range<1>(__or_local_" + writer->calcParamName + ".size()));\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    code += "            auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    code += "            auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                const int local_row = item_linear / static_cast<int>(__or_output_cols);\n";
    code += "                const int local_col = item_linear % static_cast<int>(__or_output_cols);\n";
    code += "                const int input_row = static_cast<int>(__or_output_row_begin) + local_row;\n";
    code += "                const int input_col = local_col;\n";
    code += "                auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    code += "                auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    code += "                dacpp::mpi::ContiguousView2D<const " + readerType + "> view_" + reader->calcParamName + "{__or_reader_data, input_row * static_cast<int>(__or_input_cols) + input_col, static_cast<int>(__or_input_cols)};\n";
    code += "                dacpp::mpi::ContiguousView1D<" + writerType + "> view_" + writer->calcParamName + "{__or_writer_data, item_linear};\n";
    code += "                " + calcName + "_mpi_local(view_" + reader->calcParamName + ", view_" + writer->calcParamName + ");\n";
    code += "            });\n";
    code += "        });\n";
    code += "        q.wait();\n";
    code += "    }\n";
    code += "    std::vector<" + writerType + "> __or_global_" + writer->calcParamName + ";\n";
    code += "    __or_global_" + writer->calcParamName + ".resize(static_cast<std::size_t>(__or_output_rows * __or_output_cols));\n";
    if (usesByte(plan, *writer)) {
        code += "    std::vector<int> __or_counts_bytes;\n";
        code += "    std::vector<int> __or_displs_bytes;\n";
        code += "    dacpp::mpi::operator_resident::byte_counts_displs(__or_counts, __or_displs, sizeof(" + writerType + "), __or_counts_bytes, __or_displs_bytes);\n";
        code += "    MPI_Gatherv(__or_local_" + writer->calcParamName + ".data(), static_cast<int>(__or_local_item_count * sizeof(" + writerType + ")), MPI_BYTE, mpi_rank == 0 ? __or_global_" + writer->calcParamName + ".data() : nullptr, mpi_rank == 0 ? __or_counts_bytes.data() : nullptr, mpi_rank == 0 ? __or_displs_bytes.data() : nullptr, MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        code += "    MPI_Bcast(__or_global_" + writer->calcParamName + ".data(), static_cast<int>(__or_output_rows * __or_output_cols * sizeof(" + writerType + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Gatherv(__or_local_" + writer->calcParamName + ".data(), static_cast<int>(__or_local_item_count), " + writerMpiType + ", mpi_rank == 0 ? __or_global_" + writer->calcParamName + ".data() : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, " + writerMpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "    MPI_Bcast(__or_global_" + writer->calcParamName + ".data(), static_cast<int>(__or_output_rows * __or_output_cols), " + writerMpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    code += "    " + writerArg + ".array2Tensor(__or_global_" + writer->calcParamName + ");\n";
    code += "}\n";
    return code;
}

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
