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

const ParamAccessPlan* findParamByIndex(const ShellPartitionPlan& plan,
                                        int paramIndex) {
    for (const auto& param : plan.params) {
        if (param.paramIndex == paramIndex) {
            return &param;
        }
    }
    return nullptr;
}

std::vector<const ParamAccessPlan*> scalarReaders(
    const ShellPartitionPlan& plan) {
    std::vector<const ParamAccessPlan*> result;
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar &&
            param.reads && !param.writes) {
            result.push_back(&param);
        }
    }
    return result;
}

DistributedStencilSitePlan stencilSitePlanFor(DacppFile* dacppFile,
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
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& writer) {
    std::vector<DistributedFollowupMapping> result;
    const DistributedStencilSitePlan sitePlan = stencilSitePlanFor(dacppFile,
                                                                   plan);
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

void emitBoundaryLocalMaterialize1D(
    std::string& code,
    const ShellPartitionPlan& plan,
    const std::vector<BoundaryLocalUpdate>& updates,
    const ParamAccessPlan& reader,
    const ParamAccessPlan& writer,
    const std::string& materializedWriterName,
    const std::string& readerGlobalName) {
    for (const auto& update : updates) {
        if (update.rank != 1 ||
            update.paramIndex != reader.paramIndex ||
            update.targetRowUsesLoop ||
            update.sourceRowUsesLoop) {
            continue;
        }
        code += "            {\n";
        code += "                const int64_t __or_boundary_target = " +
                update.targetRowExpr + ";\n";
        code += "                if (__or_boundary_target >= 0 && static_cast<std::size_t>(__or_boundary_target) < " +
                readerGlobalName + ".size()) {\n";
        if (update.constantRhs) {
            code += "                    " + readerGlobalName +
                    "[static_cast<std::size_t>(__or_boundary_target)] = static_cast<" +
                    elemType(plan, reader) + ">(" +
                    (update.constantValue.empty() ? "0"
                                                  : update.constantValue) +
                    ");\n";
        } else {
            code += "                    const int64_t __or_boundary_source = " +
                    update.sourceRowExpr + ";\n";
            if (update.sourceParamIndex == writer.paramIndex) {
                code += "                    if (__or_boundary_source >= 0 && static_cast<std::size_t>(__or_boundary_source) < " +
                        materializedWriterName + ".size()) {\n";
                code += "                        " + readerGlobalName +
                        "[static_cast<std::size_t>(__or_boundary_target)] = static_cast<" +
                        elemType(plan, reader) + ">(" +
                        materializedWriterName +
                        "[static_cast<std::size_t>(__or_boundary_source)]);\n";
                code += "                    }\n";
            } else {
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

void emitFollowupMaterialize1D(
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
        const ParamAccessPlan* reader = findParamByIndex(
            plan, mapping.readerParamIndex);
        if (!reader || mapping.rank != 1) {
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
        code += "            for (std::size_t __or_idx = 0; __or_idx < " +
                materialized + ".size(); ++__or_idx) {\n";
        code += "                const int64_t __or_target = static_cast<int64_t>(__or_idx) + static_cast<int64_t>(" +
                std::to_string(mapping.targetOffset) + ");\n";
        code += "                if (__or_target >= 0 && static_cast<std::size_t>(__or_target) < " +
                followupGlobal + ".size()) {\n";
        code += "                    " + followupGlobal +
                "[static_cast<std::size_t>(__or_target)] = static_cast<" +
                readerType + ">(" + materialized + "[__or_idx]);\n";
        code += "                }\n";
        code += "            }\n";
        emitBoundaryLocalMaterialize1D(code, plan, boundaryUpdates, *reader,
                                       writer, materialized, followupGlobal);
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

void emitContextType(std::string& code,
                     const std::string& ctxName,
                     const ShellPartitionPlan& plan,
                     const ParamAccessPlan& reader,
                     const ParamAccessPlan& writer) {
    code += "struct " + ctxName + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    int64_t __or_input_size = 0;\n";
    code += "    int64_t __or_output_size = 0;\n";
    code += "    int64_t __or_local_item_count = 0;\n";
    code += "    int64_t __or_output_begin = 0;\n";
    code += "    dacpp::mpi::operator_resident::RankRange1D __or_range{};\n";
    code += "    std::vector<int> __or_counts;\n";
    code += "    std::vector<int> __or_displs;\n";
    code += "    dacpp::mpi::SegmentedProfile __or_profile;\n";
    code += "    sycl::queue& q = dacpp::mpi::operator_resident::default_queue();\n";
    code += "    std::vector<" + elemType(plan, reader) + "> " +
            globalName(reader) + ";\n";
    code += "    std::vector<" + elemType(plan, writer) + "> " +
            localName(writer) + ";\n";
    for (const auto* scalar : scalarReaders(plan)) {
        const std::string type = elemType(plan, *scalar);
        code += "    " + type + " " + scalarName(*scalar) + "{};\n";
        code += "    std::vector<" + type + "> " + localName(*scalar) +
                ";\n";
    }
    code += "};\n";
}

void emitInitFunction(std::string& code,
                      const std::string& ctxName,
                      const std::string& initName,
                      const ShellPartitionPlan& plan,
                      const ParamAccessPlan& reader,
                      const ParamAccessPlan& writer) {
    const std::string readerType = elemType(plan, reader);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    code += "void " + initName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.__or_input_size = " + paramVarName(reader) +
            ".getShape(0);\n";
    code += "    ctx.__or_output_size = " + paramVarName(writer) +
            ".getShape(0);\n";
    code += "    ctx.__or_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_output_size, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.__or_local_item_count = ctx.__or_range.count;\n";
    code += "    ctx.__or_output_begin = ctx.__or_range.begin;\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(ctx.__or_output_size, ctx.mpi_size, ctx.__or_counts, ctx.__or_displs);\n";
    code += "    ctx." + globalName(reader) +
            ".resize(static_cast<std::size_t>(ctx.__or_input_size));\n";
    code += "    ctx." + localName(writer) +
            ".assign(static_cast<std::size_t>(ctx.__or_local_item_count), " +
            elemType(plan, writer) + "{});\n";
    for (const auto* scalar : scalarReaders(plan)) {
        code += "    ctx." + localName(*scalar) + ".assign(1, " +
                elemType(plan, *scalar) + "{});\n";
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);\n";
    if (plan.orLoopLower.hoistReaderSync) {
        code += "    auto dacpp_profile_bcast_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        " + paramVarName(reader) + ".tensor2Array(ctx." +
                globalName(reader) + ");\n";
        code += "    }\n";
        if (usesByte(plan, reader)) {
            code += "    MPI_Bcast(ctx." + globalName(reader) +
                    ".data(), static_cast<int>(ctx.__or_input_size * sizeof(" +
                    readerType + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Bcast(ctx." + globalName(reader) +
                    ".data(), static_cast<int>(ctx.__or_input_size), " +
                    readerMpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start);\n";
    }
    code += "}\n";
}

void emitScalarRefreshes(std::string& code, const ShellPartitionPlan& plan) {
    for (const auto* scalar : scalarReaders(plan)) {
        const std::string type = elemType(plan, *scalar);
        const std::string scalarVar = scalarName(*scalar);
        const std::string local = localName(*scalar);
        code += "    if (" + paramVarName(*scalar) + ".getSize() != 1) {\n";
        code += "        if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR][P4.6] scalar parameter " +
                scalar->shellParamName + " expected size 1\\n\");\n";
        code += "        MPI_Abort(MPI_COMM_WORLD, 2);\n";
        code += "    }\n";
        code += "    " + scalarVar + " = " + type + "{};\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        std::vector<" + type + "> __or_scalar_vec_" +
                scalar->calcParamName + ";\n";
        code += "        " + paramVarName(*scalar) +
                ".tensor2Array(__or_scalar_vec_" + scalar->calcParamName +
                ");\n";
        code += "        if (!__or_scalar_vec_" + scalar->calcParamName +
                ".empty()) " + scalarVar + " = __or_scalar_vec_" +
                scalar->calcParamName + "[0];\n";
        code += "    }\n";
        if (usesByte(plan, *scalar)) {
            code += "    MPI_Bcast(&" + scalarVar +
                    ", static_cast<int>(sizeof(" + type +
                    ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Bcast(&" + scalarVar + ", 1, " +
                    mpiDatatypeFor(type) + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    " + local + ".assign(1, " + scalarVar + ");\n";
    }
}

void emitRunFunction(std::string& code,
                     const std::string& ctxName,
                     const std::string& runName,
                     DacppFile* dacppFile,
                     const ShellPartitionPlan& plan,
                     const ParamAccessPlan& reader,
                     const ParamAccessPlan& writer) {
    const std::string readerType = elemType(plan, reader);
    const std::string writerType = elemType(plan, writer);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    const std::string calcName = plan.exprNode.calc->getName();
    const DistributedStencilSitePlan sitePlan = stencilSitePlanFor(dacppFile,
                                                                   plan);
    const auto followups = followupsForWriter(dacppFile, plan, writer);

    code += "void " + runName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    auto& q = ctx.q;\n";
    code += "    auto& __or_counts = ctx.__or_counts;\n";
    code += "    auto& __or_displs = ctx.__or_displs;\n";
    code += "    const int64_t __or_input_size = ctx.__or_input_size;\n";
    code += "    const int64_t __or_output_size = ctx.__or_output_size;\n";
    code += "    const int64_t __or_local_item_count = ctx.__or_local_item_count;\n";
    code += "    const int64_t __or_output_begin = ctx.__or_output_begin;\n";
    code += "    auto& " + globalName(reader) + " = ctx." +
            globalName(reader) + ";\n";
    code += "    auto& " + localName(writer) + " = ctx." +
            localName(writer) + ";\n";
    for (const auto* scalar : scalarReaders(plan)) {
        code += "    auto& " + scalarName(*scalar) + " = ctx." +
                scalarName(*scalar) + ";\n";
        code += "    auto& " + localName(*scalar) + " = ctx." +
                localName(*scalar) + ";\n";
    }
    if (!plan.orLoopLower.hoistReaderSync) {
        code += "    auto dacpp_profile_bcast_start_reader = dacpp::mpi::profileSegmentStart();\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + paramVarName(reader) + ".tensor2Array(" +
                globalName(reader) + ");\n";
        code += "    }\n";
        code += "    " + globalName(reader) +
                ".resize(static_cast<std::size_t>(__or_input_size));\n";
        if (usesByte(plan, reader)) {
            code += "    MPI_Bcast(" + globalName(reader) +
                    ".data(), static_cast<int>(__or_input_size * sizeof(" +
                    readerType + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Bcast(" + globalName(reader) +
                    ".data(), static_cast<int>(__or_input_size), " +
                    readerMpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_reader);\n";
    }
    if (!scalarReaders(plan).empty()) {
        code += "    auto dacpp_profile_bcast_start_scalar = dacpp::mpi::profileSegmentStart();\n";
    }
    emitScalarRefreshes(code, plan);
    if (!scalarReaders(plan).empty()) {
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_scalar);\n";
    }
    code += "    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        sycl::buffer<" + readerType + ", 1> __or_reader_buf(" +
            globalName(reader) + ".data(), sycl::range<1>(" +
            globalName(reader) + ".size()));\n";
    for (const auto* scalar : scalarReaders(plan)) {
        const std::string scalarType = elemType(plan, *scalar);
        code += "        sycl::buffer<" + scalarType + ", 1> __or_scalar_buf_" +
                scalar->calcParamName + "(" + localName(*scalar) +
                ".data(), sycl::range<1>(" + localName(*scalar) +
                ".size()));\n";
    }
    code += "        sycl::buffer<" + writerType + ", 1> __or_writer_buf(" +
            localName(writer) + ".data(), sycl::range<1>(" +
            localName(writer) + ".size()));\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    code += "            auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    for (const auto* scalar : scalarReaders(plan)) {
        code += "            auto __or_scalar_acc_" + scalar->calcParamName +
                " = __or_scalar_buf_" + scalar->calcParamName +
                ".get_access<sycl::access::mode::read>(h);\n";
    }
    code += "            auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                const int output_idx = static_cast<int>(__or_output_begin) + item_linear;\n";
    code += "                auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto* scalar : scalarReaders(plan)) {
        code += "                auto* __or_scalar_data_" +
                scalar->calcParamName + " = __or_scalar_acc_" +
                scalar->calcParamName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    }
    code += "                auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.access == ParamAccessKind::StencilWindow) {
            code += "                dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_reader_data, output_idx};\n";
            continue;
        }
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            code += "                dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_scalar_data_" + param.calcParamName + ", 0};\n";
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            !param.reads) {
            code += "                dacpp::mpi::ContiguousView1D<" +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_writer_data, item_linear};\n";
            continue;
        }
        return;
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
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
    code += "    auto& __or_resident_out_" + writer.calcParamName +
            " = dacpp::mpi::operator_resident::ensure_resident<" +
            writerType + ">(" + paramVarName(writer) + ", " +
            localName(writer) + ".size());\n";
    code += "    __or_resident_out_" + writer.calcParamName + " = " +
            localName(writer) + ";\n";
    emitGatherMaterializeFromLocalBuffer(code, plan, writer,
                                         localName(writer),
                                         "__or_output_size",
                                         "ctx.__or_profile");
    emitFollowupMaterialize1D(code, plan, writer, followups,
                              sitePlan.boundaryLocalUpdates);
    code += "}\n";
}

void emitMaterializeFunction(std::string& code,
                             const std::string& ctxName,
                             const std::string& materializeName,
                             const ShellPartitionPlan& plan) {
    code += "void " + materializeName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    (void)ctx;\n";
    for (const auto& param : plan.params) {
        code += "    (void)" + paramVarName(param) + ";\n";
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    code += "    dacpp::mpi::reportSegmentedProfile(\"" + materializeName +
            "\", ctx.__or_profile, MPI_COMM_WORLD);\n";
    code += "}\n";
}

void emitResidentHaloContextType(std::string& code,
                                 const std::string& ctxName,
                                 const ShellPartitionPlan& plan,
                                 const ParamAccessPlan& reader,
                                 const ParamAccessPlan& writer) {
    code += "struct " + ctxName + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    int64_t __or_input_size = 0;\n";
    code += "    int64_t __or_output_size = 0;\n";
    code += "    int64_t __or_local_item_count = 0;\n";
    code += "    int64_t __or_output_begin = 0;\n";
    code += "    int __or_window_size = " +
            std::to_string(plan.orLoopLower.stencilResidentHalo.windowSize) +
            ";\n";
    code += "    int __or_followup_offset = " +
            std::to_string(plan.orLoopLower.stencilResidentHalo.followupTargetOffset) +
            ";\n";
    if (plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
        code += "    int __or_temporal_block_size = " +
                std::to_string(
                    plan.orLoopLower.stencilResidentHalo.temporalBlockSize) +
                ";\n";
    }
    code += "    dacpp::mpi::operator_resident::RankRange1D __or_range{};\n";
    code += "    dacpp::mpi::operator_resident::ResidentHalo1DLayout __or_halo_layout{};\n";
    code += "    std::vector<int> __or_counts;\n";
    code += "    std::vector<int> __or_displs;\n";
    code += "    dacpp::mpi::SegmentedProfile __or_profile;\n";
    code += "    sycl::queue& q = dacpp::mpi::operator_resident::default_queue();\n";
    code += "    std::vector<" + elemType(plan, reader) + "> " +
            localName(reader) + ";\n";
    code += "    std::vector<" + elemType(plan, writer) + "> " +
            localName(writer) + ";\n";
    for (const auto* scalar : scalarReaders(plan)) {
        const std::string type = elemType(plan, *scalar);
        code += "    " + type + " " + scalarName(*scalar) + "{};\n";
        code += "    std::vector<" + type + "> " + localName(*scalar) +
                ";\n";
    }
    code += "};\n";
}

void emitResidentHaloInitFunction(std::string& code,
                                  const std::string& ctxName,
                                  const std::string& initName,
                                  const ShellPartitionPlan& plan,
                                  const ParamAccessPlan& reader,
                                  const ParamAccessPlan& writer) {
    const std::string readerType = elemType(plan, reader);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    code += "void " + initName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.__or_input_size = " + paramVarName(reader) +
            ".getShape(0);\n";
    code += "    ctx.__or_output_size = " + paramVarName(writer) +
            ".getShape(0);\n";
    code += "    ctx.__or_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_output_size, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.__or_local_item_count = ctx.__or_range.count;\n";
    code += "    ctx.__or_output_begin = ctx.__or_range.begin;\n";
    if (plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
        code += "    ctx.__or_halo_layout = dacpp::mpi::operator_resident::resident_halo_1d_layout_temporal(ctx.__or_output_size, ctx.mpi_rank, ctx.mpi_size, ctx.__or_temporal_block_size, ctx.__or_window_size);\n";
    } else {
        code += "    ctx.__or_halo_layout = dacpp::mpi::operator_resident::resident_halo_1d_layout(ctx.__or_output_size, ctx.mpi_rank, ctx.mpi_size, ctx.__or_window_size);\n";
    }
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(ctx.__or_output_size, ctx.mpi_size, ctx.__or_counts, ctx.__or_displs);\n";
    code += "    ctx." + localName(reader) +
            ".assign(static_cast<std::size_t>(ctx.__or_halo_layout.local_size), " +
            readerType + "{});\n";
    code += "    ctx." + localName(writer) +
            ".assign(static_cast<std::size_t>(ctx.__or_halo_layout.local_size), " +
            elemType(plan, writer) + "{});\n";
    for (const auto* scalar : scalarReaders(plan)) {
        code += "    ctx." + localName(*scalar) + ".assign(1, " +
                elemType(plan, *scalar) + "{});\n";
    }
    code += "    std::vector<" + readerType + "> __or_initial_global_" +
            reader.calcParamName + ";\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);\n";
    code += "    auto dacpp_profile_scatter_start = dacpp::mpi::profileSegmentStart();\n";
    if (reader.constantInit.supported) {
        code += "    std::fill(ctx." + localName(reader) + ".begin(), ctx." +
                localName(reader) + ".end(), " +
                reader.constantInit.valueExpr + ");\n";
        code += "    // Constant-initialized resident halo reader " +
                reader.calcParamName +
                " is filled locally; skip root tensor2Array/scatter_window_1d.\n";
    } else {
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        " + paramVarName(reader) +
                ".tensor2Array(__or_initial_global_" + reader.calcParamName +
                ");\n";
        code += "    }\n";
        if (plan.orLoopLower.stencilResidentHalo.temporalBlockSize > 1) {
            code += "    dacpp::mpi::operator_resident::scatter_window_1d_temporal(__or_initial_global_" +
                    reader.calcParamName + ", ctx." + localName(reader) +
                    ", ctx.__or_output_size, ctx.__or_input_size, ctx.__or_temporal_block_size, ctx.__or_window_size, ctx.__or_halo_layout, ctx.mpi_rank, ctx.mpi_size, " +
                    readerMpiType + ");\n";
        } else {
            code += "    dacpp::mpi::operator_resident::scatter_window_1d(__or_initial_global_" +
                    reader.calcParamName + ", ctx." + localName(reader) +
                    ", ctx.__or_output_size, ctx.__or_input_size, ctx.__or_window_size, ctx.__or_halo_layout, ctx.mpi_rank, ctx.mpi_size, " +
                    readerMpiType + ");\n";
        }
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start);\n";
    code += "    ctx." + localName(writer) + " = ctx." +
            localName(reader) + ";\n";
    code += "}\n";
}

void emitResidentHaloScalarRefreshes(std::string& code,
                                     const ShellPartitionPlan& plan) {
    emitScalarRefreshes(code, plan);
}

void emitResidentHaloRunFunction(std::string& code,
                                 const std::string& ctxName,
                                 const std::string& runName,
                                 const ShellPartitionPlan& plan,
                                 const ParamAccessPlan& reader,
                                 const ParamAccessPlan& writer) {
    const std::string readerType = elemType(plan, reader);
    const std::string writerType = elemType(plan, writer);
    const std::string writerMpiType = mpiDatatypeFor(writerType);
    const std::string calcName = plan.exprNode.calc->getName();
    const auto& halo = plan.orLoopLower.stencilResidentHalo;
    const int temporalBlockSize = halo.temporalBlockSize;
    const bool temporalBlocked = temporalBlockSize > 1;
    code += "void " + runName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    auto& q = ctx.q;\n";
    code += "    const int64_t __or_local_item_count = ctx.__or_local_item_count;\n";
    code += "    const int __or_writer_offset = ctx.__or_followup_offset;\n";
    code += "    int64_t __or_kernel_item_count = __or_local_item_count;\n";
    if (temporalBlocked) {
        code += "    const int __or_temporal_block_size = ctx.__or_temporal_block_size;\n";
        code += "    const bool __or_temporal_block_safe = ctx.__or_output_size >= static_cast<int64_t>(ctx.mpi_size) * static_cast<int64_t>(__or_temporal_block_size);\n";
        code += "    const int __or_runtime_temporal_block_size = __or_temporal_block_safe ? __or_temporal_block_size : 1;\n";
    }
    code += "    auto& " + localName(reader) + " = ctx." +
            localName(reader) + ";\n";
    code += "    auto& " + localName(writer) + " = ctx." +
            localName(writer) + ";\n";
    for (const auto* scalar : scalarReaders(plan)) {
        code += "    auto& " + scalarName(*scalar) + " = ctx." +
                scalarName(*scalar) + ";\n";
        code += "    auto& " + localName(*scalar) + " = ctx." +
                localName(*scalar) + ";\n";
    }
    if (!scalarReaders(plan).empty()) {
        code += "    auto dacpp_profile_bcast_start = dacpp::mpi::profileSegmentStart();\n";
    }
    emitResidentHaloScalarRefreshes(code, plan);
    if (!scalarReaders(plan).empty()) {
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start);\n";
    }
    if (temporalBlocked) {
        code += "    int64_t __or_time_step = 0;\n";
        code += "    const int64_t __or_time_limit = static_cast<int64_t>(" +
                halo.temporalLoopLimitExpr + ");\n";
        code += "    const int64_t __or_time_end = __or_time_limit" +
                std::string(halo.temporalLoopLimitInclusive ? " + 1" : "") +
                ";\n";
        code += "    while (__or_time_step < __or_time_end) {\n";
        code += "    const int __or_inner_steps = static_cast<int>(std::min<int64_t>(__or_runtime_temporal_block_size, __or_time_end - __or_time_step));\n";
        code += "    auto dacpp_profile_halo_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    dacpp::mpi::operator_resident::exchange_halo_1d_temporal_inplace(" +
                localName(reader) +
                ", ctx.__or_halo_layout, ctx.__or_output_size, ctx.__or_window_size, ctx.__or_followup_offset, __or_runtime_temporal_block_size, mpi_rank, ctx.mpi_size, " +
                writerMpiType + ");\n";
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Halo, dacpp_profile_halo_start);\n";
    }
    auto emitOneKernel = [&](const std::string& indent,
                             const std::string& readerBufferName,
                             const std::string& writerBufferName,
                             const std::string& readerIndexExpr,
                             const std::string& writerIndexExpr,
                             bool recordKernelProfile) {
        if (recordKernelProfile) {
            code += indent +
                    "auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
        }
        code += indent + "if (__or_kernel_item_count > 0) {\n";
        code += indent + "    sycl::buffer<" + readerType +
                ", 1> __or_reader_buf(" + readerBufferName +
                ".data(), sycl::range<1>(" + readerBufferName +
                ".size()));\n";
    for (const auto* scalar : scalarReaders(plan)) {
        const std::string scalarType = elemType(plan, *scalar);
        code += indent + "    sycl::buffer<" + scalarType + ", 1> __or_scalar_buf_" +
                scalar->calcParamName + "(" + localName(*scalar) +
                ".data(), sycl::range<1>(" + localName(*scalar) +
                ".size()));\n";
    }
        code += indent + "    sycl::buffer<" + writerType +
                ", 1> __or_writer_buf(" + writerBufferName +
                ".data(), sycl::range<1>(" + writerBufferName +
                ".size()));\n";
        code += indent + "    q.submit([&](sycl::handler& h) {\n";
        code += indent + "        auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    for (const auto* scalar : scalarReaders(plan)) {
        code += indent + "        auto __or_scalar_acc_" + scalar->calcParamName +
                " = __or_scalar_buf_" + scalar->calcParamName +
                ".get_access<sycl::access::mode::read>(h);\n";
    }
        code += indent + "        auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
        code += indent + "        h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_kernel_item_count)), [=](sycl::id<1> idx) {\n";
        code += indent + "            const int item_linear = static_cast<int>(idx[0]);\n";
        code += indent + "            auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto* scalar : scalarReaders(plan)) {
        code += indent + "            auto* __or_scalar_data_" +
                scalar->calcParamName + " = __or_scalar_acc_" +
                scalar->calcParamName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    }
        code += indent + "            auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.access == ParamAccessKind::StencilWindow) {
            code += indent + "            dacpp::mpi::ResidentHaloView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_reader_data, " + readerIndexExpr + "};\n";
            continue;
        }
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            code += indent + "            dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_scalar_data_" + param.calcParamName + ", 0};\n";
            continue;
        }
        if (param.access == ParamAccessKind::OutputDirect &&
            param.writes &&
            !param.reads) {
            code += indent + "            dacpp::mpi::ContiguousView1D<" +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_writer_data, " + writerIndexExpr + "};\n";
            continue;
        }
        return;
    }
        code += indent + "            " + calcName + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
         ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + plan.exprNode.calc->getParam(paramIdx)->getName();
    }
        code += ");\n";
        code += indent + "        });\n";
        code += indent + "    });\n";
        code += indent + "    q.wait();\n";
        code += indent + "}\n";
        if (recordKernelProfile) {
            code += indent +
                    "dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
        }
    };
    if (temporalBlocked) {
        code += "    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
        code += "    for (int __or_block_step = 0; __or_block_step < __or_inner_steps; ++__or_block_step) {\n";
        code += "        const int64_t __or_compute_begin = std::max<int64_t>(0, ctx.__or_halo_layout.owned_offset - (__or_inner_steps - __or_block_step - 1));\n";
        code += "        const int64_t __or_compute_end = std::min<int64_t>(ctx.__or_halo_layout.local_size - 2, ctx.__or_halo_layout.owned_offset + __or_local_item_count + (__or_inner_steps - __or_block_step - 1));\n";
        code += "        __or_kernel_item_count = std::max<int64_t>(0, __or_compute_end - __or_compute_begin);\n";
        emitOneKernel("        ", localName(reader), localName(writer),
                      "item_linear + static_cast<int>(__or_compute_begin)",
                      "item_linear + static_cast<int>(__or_compute_begin) + __or_writer_offset",
                      false);
        code += "        std::swap(" + localName(reader) + ", " +
                localName(writer) + ");\n";
        code += "    }\n";
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
        code += "    __or_time_step += __or_inner_steps;\n";
        code += "    }\n";
        code += "    // P4.6 1D temporal-block=2 leaves the latest state in the resident reader buffer after the block loop.\n";
        code += "}\n";
        return;
    }
    emitOneKernel("    ", localName(reader), localName(writer), "item_linear",
                  "item_linear + __or_writer_offset", true);
    code += "    auto dacpp_profile_halo_start = dacpp::mpi::profileSegmentStart();\n";
    if (halo.hasBoundaryLocalUpdate) {
        if (halo.boundaryCopiesWriter) {
            code += "    if (mpi_rank == 0 && !" + localName(writer) +
                    ".empty()) {\n";
            code += "        const int64_t __or_boundary_target = " +
                    std::to_string(halo.boundaryTargetIndex) + ";\n";
            code += "        const int64_t __or_boundary_source = static_cast<int64_t>(__or_writer_offset) + " +
                    std::to_string(halo.boundarySourceIndex) + ";\n";
            code += "        if (__or_boundary_target >= 0 && __or_boundary_target < static_cast<int64_t>(" +
                    localName(writer) + ".size()) && __or_boundary_source >= 0 && __or_boundary_source < static_cast<int64_t>(" +
                    localName(writer) + ".size())) {\n";
            code += "            " + localName(writer) +
                    "[static_cast<std::size_t>(__or_boundary_target)] = " +
                    localName(writer) +
                    "[static_cast<std::size_t>(__or_boundary_source)];\n";
            code += "        }\n";
            code += "    }\n";
        } else if (!halo.boundaryConstantValue.empty()) {
            code += "    if (mpi_rank == 0) {\n";
            code += "        const int64_t __or_boundary_target = " +
                    std::to_string(halo.boundaryTargetIndex) + ";\n";
            code += "        if (__or_boundary_target >= 0 && __or_boundary_target < static_cast<int64_t>(" +
                    localName(writer) + ".size())) {\n";
            code += "            " + localName(writer) +
                    "[static_cast<std::size_t>(__or_boundary_target)] = static_cast<" +
                    readerType + ">(" + halo.boundaryConstantValue + ");\n";
            code += "        }\n";
            code += "    }\n";
        }
    }
    code += "    dacpp::mpi::operator_resident::exchange_halo_1d_inplace(" +
            localName(writer) +
            ", ctx.__or_halo_layout, ctx.__or_output_size, ctx.__or_window_size, ctx.__or_followup_offset, mpi_rank, ctx.mpi_size, " +
            writerMpiType + ");\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Halo, dacpp_profile_halo_start);\n";
    code += "    ctx." + localName(reader) + ".swap(ctx." +
            localName(writer) + ");\n";
    code += "}\n";
}

void emitResidentHaloMaterializeFunction(std::string& code,
                                         const std::string& ctxName,
                                         const std::string& materializeName,
                                         const ShellPartitionPlan& plan,
                                         const ParamAccessPlan& reader,
                                         const ParamAccessPlan& writer) {
    const std::string readerType = elemType(plan, reader);
    const std::string readerMpiType = mpiDatatypeFor(readerType);
    const std::string writerType = elemType(plan, writer);
    const std::string writerMpiType = mpiDatatypeFor(writerType);
    const auto& halo = plan.orLoopLower.stencilResidentHalo;
    code += "void " + materializeName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    auto dacpp_profile_gather_start_writer = dacpp::mpi::profileSegmentStart();\n";
    code += "    std::vector<" + writerType + "> __or_materialized_" +
            writer.calcParamName + ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        __or_materialized_" + writer.calcParamName +
            ".resize(static_cast<std::size_t>(ctx.__or_output_size));\n";
    code += "    }\n";
    code += "    const auto __or_owned_" + writer.calcParamName +
            " = dacpp::mpi::operator_resident::owned_slice_1d(ctx." +
            localName(reader) + ", ctx.__or_halo_layout, ctx.__or_followup_offset);\n";
    code += "    MPI_Gatherv(__or_owned_" + writer.calcParamName +
            ".data(), static_cast<int>(ctx.__or_local_item_count), " +
            writerMpiType + ", mpi_rank == 0 ? __or_materialized_" +
            writer.calcParamName +
            ".data() : nullptr, mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
            writerMpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_writer);\n";
    code += "    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + paramVarName(writer) + ".array2Tensor(__or_materialized_" +
            writer.calcParamName + ");\n";
    code += "    }\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    code += "    std::vector<" + readerType + "> __or_materialized_" +
            reader.calcParamName + ";\n";
    code += "    dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + paramVarName(reader) + ".tensor2Array(__or_materialized_" +
            reader.calcParamName + ");\n";
    code += "    }\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    code += "    auto dacpp_profile_gather_start_reader = dacpp::mpi::profileSegmentStart();\n";
    code += "    MPI_Gatherv(__or_owned_" + writer.calcParamName +
            ".data(), static_cast<int>(ctx.__or_local_item_count), " +
            readerMpiType + ", mpi_rank == 0 ? __or_materialized_" +
            reader.calcParamName +
            ".data() + 1 : nullptr, mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
            readerMpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_reader);\n";
    code += "    dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (mpi_rank == 0) {\n";
    if (halo.hasBoundaryLocalUpdate) {
        code += "        const int64_t __or_boundary_target = " +
                std::to_string(halo.boundaryTargetIndex) + ";\n";
        code += "        if (__or_boundary_target >= 0 && __or_boundary_target < static_cast<int64_t>(__or_materialized_" +
                reader.calcParamName + ".size())) {\n";
        code += "            __or_materialized_" + reader.calcParamName +
                "[static_cast<std::size_t>(__or_boundary_target)] = ctx." +
                localName(reader) +
                "[static_cast<std::size_t>(__or_boundary_target)];\n";
        code += "        }\n";
    }
    code += "        " + paramVarName(reader) + ".array2Tensor(__or_materialized_" +
            reader.calcParamName + ");\n";
    code += "    }\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    for (const auto& param : plan.params) {
        if (param.paramIndex != reader.paramIndex) {
            code += "    (void)" + paramVarName(param) + ";\n";
        }
    }
    code += "    dacpp::mpi::reportSegmentedProfile(\"" + materializeName +
            "\", ctx.__or_profile, MPI_COMM_WORLD);\n";
    code += "}\n";
}

}  // namespace

std::string buildLoopLoweredStencil1DFullSyncFamilyCode(
    const std::string& baseName,
    DacppFile* dacppFile,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = findStencilReader(plan);
    const ParamAccessPlan* writer = findStencilWriter(plan);
    if (!reader || !writer || !plan.exprNode.shell || !plan.exprNode.calc) {
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

    std::string code;
    emitContextType(code, ctxName, plan, *reader, *writer);
    emitInitFunction(code, ctxName, initName, plan, *reader, *writer);
    emitRunFunction(code, ctxName, runName, dacppFile, plan, *reader, *writer);
    emitMaterializeFunction(code, ctxName, materializeName, plan);
    code += "void " + baseName + "(" + wrapperSignature(plan) + ") {\n";
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

std::string buildLoopLoweredStencil1DResidentHaloFamilyCode(
    const std::string& baseName,
    DacppFile*,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = findStencilReader(plan);
    const ParamAccessPlan* writer = findStencilWriter(plan);
    if (!reader || !writer || !plan.exprNode.shell || !plan.exprNode.calc ||
        !plan.orLoopLower.stencilResidentHalo.enabled) {
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

    std::string code;
    emitResidentHaloContextType(code, ctxName, plan, *reader, *writer);
    emitResidentHaloInitFunction(code, ctxName, initName, plan, *reader,
                                 *writer);
    emitResidentHaloRunFunction(code, ctxName, runName, plan, *reader,
                                *writer);
    emitResidentHaloMaterializeFunction(code, ctxName, materializeName, plan,
                                        *reader, *writer);
    code += "void " + baseName + "(" + wrapperSignature(plan) + ") {\n";
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
