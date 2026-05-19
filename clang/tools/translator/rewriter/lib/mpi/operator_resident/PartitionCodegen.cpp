#include "DacppStructure.h"
#include "OperatorResidentCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

void emitPartitionCode(std::string& code, const ShellPartitionPlan& plan) {
    const auto& firstDomain = plan.bindDomains.front();
    const std::string firstTensor =
        paramVarName(plan.params[static_cast<std::size_t>(
            firstDomain.runtimeSizeParam)]);
    const bool uses1DPartition =
        plan.signature.layout == LocalLayoutKind::Contiguous1D ||
        plan.signature.layout == LocalLayoutKind::ReplicatedFullTensor ||
        (plan.signature.layout == LocalLayoutKind::RowPartitionFullRow &&
         plan.signature.bindOrder.size() == 1);
    if (uses1DPartition) {
        code += "    const int64_t __or_total_items = " + firstTensor +
                ".getShape(" + std::to_string(firstDomain.dimId) + ");\n";
        code += "    const auto __or_range = dacpp::mpi::operator_resident::rank_range_1d(__or_total_items, mpi_rank, mpi_size);\n";
        code += "    const int64_t __or_local_item_count = __or_range.count;\n";
        code += "    std::vector<int> __or_counts;\n";
        code += "    std::vector<int> __or_displs;\n";
        code += "    dacpp::mpi::operator_resident::counts_displs_1d(__or_total_items, mpi_size, __or_counts, __or_displs);\n";
        return;
    }

    const auto& rowDomain = plan.bindDomains[0];
    const auto& colDomain = plan.bindDomains[1];
    const std::string tensor =
        paramVarName(plan.params[static_cast<std::size_t>(
            rowDomain.runtimeSizeParam)]);
    code += "    const int64_t __or_rows = " + tensor + ".getShape(" +
            std::to_string(rowDomain.dimId) + ");\n";
    code += "    const int64_t __or_cols = " + tensor + ".getShape(" +
            std::to_string(colDomain.dimId) + ");\n";
    code += "    const int64_t __or_total_items = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_rows, __or_cols, \"[DACPP][MPI][OR] row-block total item count overflow\");\n";
    code += "    const auto __or_row_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, mpi_rank, mpi_size);\n";
    code += "    const int64_t __or_local_item_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_row_range.count, __or_cols, \"[DACPP][MPI][OR] row-block local item count overflow\");\n";
    code += "    std::vector<int> __or_row_counts;\n";
    code += "    std::vector<int> __or_row_displs;\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(__or_rows, mpi_size, __or_row_counts, __or_row_displs);\n";
    code += "    std::vector<int> __or_counts(mpi_size);\n";
    code += "    std::vector<int> __or_displs(mpi_size);\n";
    code += "    for (int r = 0; r < mpi_size; ++r) {\n";
    code += "        __or_counts[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_row_counts[r]), __or_cols, \"[DACPP][MPI][OR] row-block scatter count overflow\"), \"[DACPP][MPI][OR] row-block scatter count exceeds MPI int range\");\n";
    code += "        __or_displs[r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(__or_row_displs[r]), __or_cols, \"[DACPP][MPI][OR] row-block scatter displacement overflow\"), \"[DACPP][MPI][OR] row-block scatter displacement exceeds MPI int range\");\n";
    code += "    }\n";
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
