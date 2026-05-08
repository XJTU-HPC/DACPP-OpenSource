#include <string>
#include <vector>

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

std::vector<const ParamAccessPlan*> findStencilDirectReaders(
    const ShellPartitionPlan& plan) {
    std::vector<const ParamAccessPlan*> readers;
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::DirectMapped &&
            param.reads &&
            !param.writes) {
            readers.push_back(&param);
        }
    }
    return readers;
}

const ParamAccessPlan* findParamByIndex(const ShellPartitionPlan& plan,
                                        int paramIndex) {
    for (const auto& param : plan.params) {
        if (param.paramIndex == paramIndex) {
            return &param;
        }
    }
    return nullptr;
}

DistributedStencilSitePlan stencilSitePlanFor(
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan) {
    if (!dacppFile || !plan.exprNode.shell || !plan.exprNode.calc ||
        !plan.exprNode.dacExpr) {
        return {};
    }
    return analyzeDistributedStencilSite(dacppFile, plan.exprNode.shell,
                                        plan.exprNode.calc,
                                        plan.exprNode.dacExpr);
}

std::vector<DistributedFollowupMapping> followupsForWriter(
    const DistributedStencilSitePlan& sitePlan,
    const ParamAccessPlan& writer) {
    std::vector<DistributedFollowupMapping> result;
    if (!sitePlan.supported || sitePlan.hasRootBridge) {
        return result;
    }
    for (const auto& mapping : sitePlan.followupMappings) {
        if (mapping.writerParamIndex == writer.paramIndex ||
            mapping.writerTensor == writer.actualTensorName ||
            mapping.writerTensor == writer.shellParamName) {
            result.push_back(mapping);
        }
    }
    return result;
}

void emitReadCacheTransitions2D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const std::vector<ReadCacheStateTransition>& transitions) {
    for (const auto& transition : transitions) {
        if (transition.rank != 2) {
            continue;
        }
        const ParamAccessPlan* source =
            findParamByIndex(plan, transition.writerParamIndex);
        const ParamAccessPlan* target =
            findParamByIndex(plan, transition.readerParamIndex);
        if (!source || !target) {
            continue;
        }
        const std::string targetType = elemType(plan, *target);
        const std::string targetMpiType = mpiDatatypeFor(targetType);
        const std::string transitionGlobal =
            "__or_transition_global_" + target->calcParamName;
        code += "    {\n";
        code += "        std::vector<" + targetType + "> " +
                transitionGlobal + ";\n";
        code += "        if (mpi_rank == 0) {\n";
        code += "            " + paramVarName(*target) +
                ".tensor2Array(" + transitionGlobal + ");\n";
        code += "            const int64_t __or_transition_source_cols_" +
                source->calcParamName + " = static_cast<int64_t>(" +
                paramVarName(*source) + ".getShape(1));\n";
        code += "            const int64_t __or_transition_target_cols_" +
                target->calcParamName + " = static_cast<int64_t>(" +
                paramVarName(*target) + ".getShape(1));\n";
        code += "            for (std::size_t __or_idx = 0; __or_idx < __or_global_" +
                source->calcParamName + ".size(); ++__or_idx) {\n";
        code += "                const int64_t __or_target = dacpp::mpi::map_2d_global_with_offset(static_cast<int64_t>(__or_idx), __or_transition_source_cols_" +
                source->calcParamName + ", __or_transition_target_cols_" +
                target->calcParamName + ", " +
                std::to_string(transition.targetRowOffset) + ", " +
                std::to_string(transition.targetColOffset) + ");\n";
        code += "                if (__or_target >= 0 && static_cast<std::size_t>(__or_target) < " +
                transitionGlobal + ".size()) {\n";
        code += "                    " + transitionGlobal +
                "[static_cast<std::size_t>(__or_target)] = static_cast<" +
                targetType + ">(__or_global_" + source->calcParamName +
                "[__or_idx]);\n";
        code += "                }\n";
        code += "            }\n";
        code += "        } else {\n";
        code += "            " + transitionGlobal +
                ".resize(static_cast<std::size_t>(" + paramVarName(*target) +
                ".getSize()));\n";
        code += "        }\n";
        code += "        if (!" + transitionGlobal + ".empty()) {\n";
        code += "            MPI_Bcast(" + transitionGlobal + ".data(), " +
                mpiPayloadCountExpr(paramVarName(*target) + ".getSize()",
                                    targetType) +
                ", " + targetMpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "        }\n";
        code += "        " + paramVarName(*target) + ".array2Tensor(" +
                transitionGlobal + ");\n";
        code += "    }\n";
    }
}

void emitBoundaryLocalMaterialize2D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const std::vector<BoundaryLocalUpdate>& updates,
    const ParamAccessPlan& reader,
    const ParamAccessPlan& writer,
    const std::string& materializedWriterName,
    const std::string& readerGlobalName,
    const std::string& readerRowsExpr,
    const std::string& readerColsExpr,
    const std::string& outputRowsExpr,
    const std::string& outputColsExpr) {
    for (const auto& update : updates) {
        if (update.rank != 2 || update.paramIndex != reader.paramIndex) {
            continue;
        }
        const std::string loopLower =
            update.loopLowerExpr.empty() ? "0" : update.loopLowerExpr;
        const std::string loopUpper =
            update.loopUpperExpr.empty()
                ? ((update.targetRowUsesLoop || update.sourceRowUsesLoop)
                       ? readerRowsExpr
                       : readerColsExpr)
                : update.loopUpperExpr;
        const std::string loopCmp = update.loopUpperInclusive ? "<=" : "<";
        code += "            {\n";
        code += "                const int64_t __or_boundary_begin = static_cast<int64_t>(" +
                loopLower + ");\n";
        code += "                const int64_t __or_boundary_end = static_cast<int64_t>(" +
                loopUpper + ");\n";
        code += "                for (int64_t __or_boundary_idx = __or_boundary_begin; __or_boundary_idx " +
                loopCmp +
                " __or_boundary_end; ++__or_boundary_idx) {\n";
        code += "                    const int64_t __or_boundary_target_row = " +
                (update.targetRowUsesLoop ? "__or_boundary_idx"
                                          : update.targetRowExpr) +
                ";\n";
        code += "                    const int64_t __or_boundary_target_col = " +
                (update.targetColUsesLoop ? "__or_boundary_idx"
                                          : update.targetColExpr) +
                ";\n";
        code += "                    if (__or_boundary_target_row < 0 || __or_boundary_target_col < 0 || __or_boundary_target_row >= " +
                readerRowsExpr + " || __or_boundary_target_col >= " +
                readerColsExpr + ") continue;\n";
        code += "                    const int64_t __or_boundary_target = __or_boundary_target_row * " +
                readerColsExpr + " + __or_boundary_target_col;\n";
        code += "                    if (__or_boundary_target < 0 || static_cast<std::size_t>(__or_boundary_target) >= " +
                readerGlobalName + ".size()) continue;\n";
        if (update.constantRhs) {
            code += "                    " + readerGlobalName +
                    "[static_cast<std::size_t>(__or_boundary_target)] = static_cast<" +
                    elemType(plan, reader) + ">(" +
                    (update.constantValue.empty() ? "0"
                                                  : update.constantValue) +
                    ");\n";
        } else {
            code += "                    const int64_t __or_boundary_source_row = " +
                    (update.sourceRowUsesLoop ? "__or_boundary_idx"
                                              : update.sourceRowExpr) +
                    ";\n";
            code += "                    const int64_t __or_boundary_source_col = " +
                    (update.sourceColUsesLoop ? "__or_boundary_idx"
                                              : update.sourceColExpr) +
                    ";\n";
            if (update.sourceParamIndex == writer.paramIndex) {
                code += "                    if (__or_boundary_source_row < 0 || __or_boundary_source_col < 0 || __or_boundary_source_row >= " +
                        outputRowsExpr + " || __or_boundary_source_col >= " +
                        outputColsExpr + ") continue;\n";
                code += "                    const int64_t __or_boundary_source = __or_boundary_source_row * " +
                        outputColsExpr + " + __or_boundary_source_col;\n";
                code += "                    if (__or_boundary_source >= 0 && static_cast<std::size_t>(__or_boundary_source) < " +
                        materializedWriterName + ".size()) {\n";
                code += "                        " + readerGlobalName +
                        "[static_cast<std::size_t>(__or_boundary_target)] = static_cast<" +
                        elemType(plan, reader) + ">(" +
                        materializedWriterName +
                        "[static_cast<std::size_t>(__or_boundary_source)]);\n";
                code += "                    }\n";
            } else {
                code += "                    if (__or_boundary_source_row < 0 || __or_boundary_source_col < 0 || __or_boundary_source_row >= " +
                        readerRowsExpr + " || __or_boundary_source_col >= " +
                        readerColsExpr + ") continue;\n";
                code += "                    const int64_t __or_boundary_source = __or_boundary_source_row * " +
                        readerColsExpr + " + __or_boundary_source_col;\n";
                code += "                    if (__or_boundary_source >= 0 && static_cast<std::size_t>(__or_boundary_source) < " +
                        readerGlobalName + ".size()) {\n";
                code += "                        " + readerGlobalName +
                        "[static_cast<std::size_t>(__or_boundary_target)] = " +
                        readerGlobalName +
                        "[static_cast<std::size_t>(__or_boundary_source)];\n";
                code += "                    }\n";
            }
        }
        code += "                }\n";
        code += "            }\n";
    }
}

void emitFollowupMaterialize2D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& writer,
    const std::vector<DistributedFollowupMapping>& followups,
    const std::vector<BoundaryLocalUpdate>& boundaryUpdates) {
    if (followups.empty()) {
        return;
    }
    const std::string materialized =
        "__or_materialized_" + writer.calcParamName;
    for (const auto& mapping : followups) {
        if (mapping.rank != 2) {
            continue;
        }
        const ParamAccessPlan* reader =
            findParamByIndex(plan, mapping.readerParamIndex);
        if (!reader) {
            continue;
        }
        const std::string readerType = elemType(plan, *reader);
        const std::string readerMpiType = mpiDatatypeFor(readerType);
        const std::string followupGlobal =
            "__or_followup_global_" + reader->calcParamName;
        code += "    {\n";
        code += "        std::vector<" + readerType + "> " + followupGlobal +
                ";\n";
        code += "        if (mpi_rank == 0) {\n";
        code += "            " + paramVarName(*reader) +
                ".tensor2Array(" + followupGlobal + ");\n";
        code += "            const int64_t __or_followup_reader_cols_" +
                reader->calcParamName + " = static_cast<int64_t>(" +
                paramVarName(*reader) + ".getShape(1));\n";
        code += "            for (std::size_t __or_idx = 0; __or_idx < " +
                materialized + ".size(); ++__or_idx) {\n";
        code += "                const int64_t __or_target = dacpp::mpi::map_2d_global_with_offset(static_cast<int64_t>(__or_idx), __or_output_cols, __or_followup_reader_cols_" +
                reader->calcParamName + ", " +
                std::to_string(mapping.targetRowOffset) + ", " +
                std::to_string(mapping.targetColOffset) + ");\n";
        code += "                if (__or_target >= 0 && static_cast<std::size_t>(__or_target) < " +
                followupGlobal + ".size()) {\n";
        code += "                    " + followupGlobal +
                "[static_cast<std::size_t>(__or_target)] = static_cast<" +
                readerType + ">(" + materialized + "[__or_idx]);\n";
        code += "                }\n";
        code += "            }\n";
        emitBoundaryLocalMaterialize2D(
            code, plan, boundaryUpdates, *reader, writer, materialized,
            followupGlobal, paramVarName(*reader) + ".getShape(0)",
            "__or_followup_reader_cols_" + reader->calcParamName,
            "__or_output_rows", "__or_output_cols");
        code += "        } else {\n";
        code += "            " + followupGlobal +
                ".resize(static_cast<std::size_t>(" + paramVarName(*reader) +
                ".getSize()));\n";
        code += "        }\n";
        code += "        if (!" + followupGlobal + ".empty()) {\n";
        code += "            MPI_Bcast(" + followupGlobal + ".data(), " +
                mpiPayloadCountExpr(paramVarName(*reader) + ".getSize()",
                                    readerType) +
                ", " + readerMpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "        }\n";
        code += "        " + paramVarName(*reader) + ".array2Tensor(" +
                followupGlobal + ");\n";
        code += "    }\n";
    }
}

}  // namespace

std::string buildStencilWindow2DWrapperCode(
    const std::string& wrapperName,
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = findStencilReader(plan);
    const ParamAccessPlan* writer = findStencilWriter(plan);
    const auto directReaders = findStencilDirectReaders(plan);
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
    const DistributedStencilSitePlan sitePlan =
        stencilSitePlanFor(dacppFile, plan);
    const auto followups = followupsForWriter(sitePlan, *writer);

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
    for (const auto* directReader : directReaders) {
        const std::string directType = elemType(plan, *directReader);
        const std::string directMpiType =
            mpi_rewriter::mpiDatatypeFor(directType);
        const std::string directArg = paramVarName(*directReader);
        code += "    const int64_t __or_direct_rows_" + directReader->calcParamName +
                " = " + directArg + ".getShape(0);\n";
        code += "    const int64_t __or_direct_cols_" + directReader->calcParamName +
                " = " + directArg + ".getShape(1);\n";
        code += "    if (__or_direct_rows_" + directReader->calcParamName +
                " != __or_output_rows || __or_direct_cols_" +
                directReader->calcParamName + " != __or_output_cols) {\n";
        code += "        if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] stencil direct reader " +
                directReader->shellParamName +
                " shape mismatch with output\\n\");\n";
        code += "        MPI_Abort(MPI_COMM_WORLD, 4);\n";
        code += "    }\n";
        code += "    std::vector<" + directType + "> __or_global_" +
                directReader->calcParamName + ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + directArg + ".tensor2Array(__or_global_" +
                directReader->calcParamName + ");\n";
        code += "    }\n";
        code += "    __or_global_" + directReader->calcParamName +
                ".resize(static_cast<std::size_t>(__or_output_rows * __or_output_cols));\n";
        if (usesByte(plan, *directReader)) {
            code += "    MPI_Bcast(__or_global_" + directReader->calcParamName +
                    ".data(), static_cast<int>(__or_output_rows * __or_output_cols * sizeof(" +
                    directType + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Bcast(__or_global_" + directReader->calcParamName +
                    ".data(), static_cast<int>(__or_output_rows * __or_output_cols), " +
                    directMpiType + ", 0, MPI_COMM_WORLD);\n";
        }
    }
    code += "    std::vector<" + writerType + "> __or_local_" + writer->calcParamName + "(static_cast<std::size_t>(__or_local_item_count));\n";
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        sycl::buffer<" + readerType + ", 1> __or_reader_buf(__or_global_" + reader->calcParamName + ".data(), sycl::range<1>(__or_global_" + reader->calcParamName + ".size()));\n";
    for (const auto* directReader : directReaders) {
        const std::string directType = elemType(plan, *directReader);
        code += "        sycl::buffer<" + directType + ", 1> __or_buffer_" +
                directReader->calcParamName + "(__or_global_" +
                directReader->calcParamName + ".data(), sycl::range<1>(__or_global_" +
                directReader->calcParamName + ".size()));\n";
    }
    code += "        sycl::buffer<" + writerType + ", 1> __or_writer_buf(__or_local_" + writer->calcParamName + ".data(), sycl::range<1>(__or_local_" + writer->calcParamName + ".size()));\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    code += "            auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    for (const auto* directReader : directReaders) {
        code += "            auto __or_acc_" + directReader->calcParamName +
                " = __or_buffer_" + directReader->calcParamName +
                ".get_access<sycl::access::mode::read>(h);\n";
    }
    code += "            auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                const int local_row = item_linear / static_cast<int>(__or_output_cols);\n";
    code += "                const int local_col = item_linear % static_cast<int>(__or_output_cols);\n";
    code += "                const int output_row = static_cast<int>(__or_output_row_begin) + local_row;\n";
    code += "                const int output_col = local_col;\n";
    code += "                const int input_row = output_row;\n";
    code += "                const int input_col = output_col;\n";
    code += "                const int global_output_linear = output_row * static_cast<int>(__or_output_cols) + output_col;\n";
    code += "                auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto* directReader : directReaders) {
        code += "                auto* __or_data_" + directReader->calcParamName +
                " = __or_acc_" + directReader->calcParamName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    }
    code += "                auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.access == ParamAccessKind::StencilWindow) {
            code += "                dacpp::mpi::ContiguousView2D<const " + paramType + "> view_" + param.calcParamName + "{__or_reader_data, input_row * static_cast<int>(__or_input_cols) + input_col, static_cast<int>(__or_input_cols)};\n";
            continue;
        }
        if (param.access == ParamAccessKind::DirectMapped &&
            param.reads &&
            !param.writes) {
            code += "                dacpp::mpi::ContiguousView1D<const " + paramType + "> view_" + param.calcParamName + "{__or_data_" + param.calcParamName + ", global_output_linear};\n";
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            !param.reads) {
            code += "                dacpp::mpi::ContiguousView1D<" + paramType + "> view_" + param.calcParamName + "{__or_writer_data, item_linear};\n";
            continue;
        }
        return {};
    }
    code += "                " + calcName + "_mpi_local(";
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
    code += "    auto& __or_resident_out_" + writer->calcParamName +
            " = dacpp::mpi::operator_resident::ensure_resident<" + writerType +
            ">(" + writerArg + ", __or_local_" + writer->calcParamName +
            ".size());\n";
    code += "    __or_resident_out_" + writer->calcParamName + " = __or_local_" +
            writer->calcParamName + ";\n";
    emitGatherMaterializeFromLocalBuffer(
        code, plan, *writer, "__or_local_" + writer->calcParamName,
        "__or_output_rows * __or_output_cols");
    emitReadCacheTransitions2D(code, plan, sitePlan.readCacheTransitions);
    emitFollowupMaterialize2D(code, plan, *writer, followups,
                              sitePlan.boundaryLocalUpdates);
    code += "}\n";
    return code;
}

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
