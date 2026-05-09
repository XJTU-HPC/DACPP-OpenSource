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
    code += "    sycl::queue q{sycl::default_selector_v};\n";
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
    if (plan.orLoopLower.hoistReaderSync) {
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
    }
    emitScalarRefreshes(code, plan);
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
    code += "    auto& __or_resident_out_" + writer.calcParamName +
            " = dacpp::mpi::operator_resident::ensure_resident<" +
            writerType + ">(" + paramVarName(writer) + ", " +
            localName(writer) + ".size());\n";
    code += "    __or_resident_out_" + writer.calcParamName + " = " +
            localName(writer) + ";\n";
    emitGatherMaterializeFromLocalBuffer(code, plan, writer,
                                         localName(writer),
                                         "__or_output_size");
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
    code += "    (void)ctx;\n";
    for (const auto& param : plan.params) {
        code += "    (void)" + paramVarName(param) + ";\n";
    }
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

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
