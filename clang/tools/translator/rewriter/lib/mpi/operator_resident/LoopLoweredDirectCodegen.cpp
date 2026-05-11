#include "DacppStructure.h"
#include "Rewriter_MPI_Common.h"
#include "OperatorResidentCodegen_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

namespace {

bool isLoopInvariantDirectRead(const ParamAccessPlan& param) {
    return param.reads && !param.writes &&
           param.access == ParamAccessKind::DirectMapped;
}

std::string ctxLocalName(const ParamAccessPlan& param) {
    return "ctx." + localName(param);
}

std::string ctxScalarName(const ParamAccessPlan& param) {
    return "ctx." + scalarName(param);
}

void emitContextType(std::string& code,
                     const std::string& ctxName,
                     const ShellPartitionPlan& plan) {
    code += "struct " + ctxName + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    int64_t __or_total_items = 0;\n";
    code += "    int64_t __or_local_item_count = 0;\n";
    code += "    dacpp::mpi::operator_resident::RankRange1D __or_range{};\n";
    code += "    std::vector<int> __or_counts;\n";
    code += "    std::vector<int> __or_displs;\n";
    code += "    dacpp::mpi::SegmentedProfile __or_profile;\n";
    code += "    sycl::queue q{sycl::default_selector_v};\n";
    for (const auto& param : plan.params) {
        const std::string type = elemType(plan, param);
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            code += "    " + type + " " + scalarName(param) + "{};\n";
            code += "    std::vector<" + type + "> " + localName(param) +
                    ";\n";
        } else if (isLoopInvariantDirectRead(param) || param.writes) {
            code += "    std::vector<" + type + "> " + localName(param) +
                    ";\n";
        }
    }
    code += "};\n";
}

void emitInitFunction(std::string& code,
                      const std::string& ctxName,
                      const std::string& initName,
                      const ShellPartitionPlan& plan) {
    const auto& firstDomain = plan.bindDomains.front();
    const std::string firstTensor =
        paramVarName(plan.params[static_cast<std::size_t>(
            firstDomain.runtimeSizeParam)]);
    code += "void " + initName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    code += "    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.__or_total_items = " + firstTensor + ".getShape(" +
            std::to_string(firstDomain.dimId) + ");\n";
    code += "    ctx.__or_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.__or_local_item_count = ctx.__or_range.count;\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(ctx.__or_total_items, ctx.mpi_size, ctx.__or_counts, ctx.__or_displs);\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);\n";
    for (const auto& param : plan.params) {
        if (!isLoopInvariantDirectRead(param)) {
            continue;
        }
        const std::string type = elemType(plan, param);
        const std::string mpiType = mpiDatatypeFor(type);
        const std::string local = ctxLocalName(param);
        const std::string global = "__or_global_" + param.calcParamName;
        code += "    auto dacpp_profile_scatter_start_" +
                param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        code += "    " + local +
                ".resize(static_cast<std::size_t>(ctx.__or_local_item_count));\n";
        code += "    std::vector<" + type + "> " + global + ";\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        " + paramVarName(param) + ".tensor2Array(" + global +
                ");\n";
        code += "    }\n";
        if (usesByte(plan, param)) {
            code += "    std::vector<int> __or_counts_bytes_" +
                    param.calcParamName + ";\n";
            code += "    std::vector<int> __or_displs_bytes_" +
                    param.calcParamName + ";\n";
            code += "    dacpp::mpi::operator_resident::byte_counts_displs(ctx.__or_counts, ctx.__or_displs, sizeof(" +
                    type + "), __or_counts_bytes_" + param.calcParamName +
                    ", __or_displs_bytes_" + param.calcParamName + ");\n";
            code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? " + global +
                    ".data() : nullptr, ctx.mpi_rank == 0 ? __or_counts_bytes_" +
                    param.calcParamName +
                    ".data() : nullptr, ctx.mpi_rank == 0 ? __or_displs_bytes_" +
                    param.calcParamName + ".data() : nullptr, MPI_BYTE, " +
                    local +
                    ".data(), static_cast<int>(ctx.__or_local_item_count * sizeof(" +
                    type + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? " + global +
                    ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
                    mpiType + ", " + local +
                    ".data(), static_cast<int>(ctx.__or_local_item_count), " +
                    mpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_" +
                param.calcParamName + ");\n";
    }
    for (const auto& param : plan.params) {
        if (!param.writes || param.access != ParamAccessKind::OutputDirect) {
            continue;
        }
        code += "    " + ctxLocalName(param) +
                ".assign(static_cast<std::size_t>(ctx.__or_local_item_count), " +
                elemType(plan, param) + "{});\n";
    }
    code += "}\n";
}

void emitScalarRefreshes(std::string& code, const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access != ParamAccessKind::ReplicatedScalar) {
            continue;
        }
        const std::string type = elemType(plan, param);
        const std::string scalar = ctxScalarName(param);
        const std::string local = ctxLocalName(param);
        code += "    if (" + paramVarName(param) + ".getSize() != 1) {\n";
        code += "        if (ctx.mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR][P4.5] scalar parameter " +
                param.shellParamName + " expected size 1\\n\");\n";
        code += "        MPI_Abort(MPI_COMM_WORLD, 2);\n";
        code += "    }\n";
        code += "    " + scalar + " = " + type + "{};\n";
        code += "    if (ctx.mpi_rank == 0) {\n";
        code += "        std::vector<" + type + "> __or_scalar_vec_" +
                param.calcParamName + ";\n";
        code += "        " + paramVarName(param) + ".tensor2Array(__or_scalar_vec_" +
                param.calcParamName + ");\n";
        code += "        if (!__or_scalar_vec_" + param.calcParamName +
                ".empty()) " + scalar + " = __or_scalar_vec_" +
                param.calcParamName + "[0];\n";
        code += "    }\n";
        if (usesByte(plan, param)) {
            code += "    MPI_Bcast(&" + scalar +
                    ", static_cast<int>(sizeof(" + type +
                    ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Bcast(&" + scalar + ", 1, " +
                    mpiDatatypeFor(type) + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    " + local + ".assign(1, " + scalar + ");\n";
    }
}

void emitRunKernel(std::string& code, const ShellPartitionPlan& plan) {
    code += "    if (ctx.__or_local_item_count > 0) {\n";
    code += "        {\n";
    for (const auto& param : plan.params) {
        const std::string type = elemType(plan, param);
        const std::string name = param.calcParamName;
        code += "            sycl::buffer<" + type + ", 1> __or_buffer_" +
                name + "(" + ctxLocalName(param) +
                ".data(), sycl::range<1>(" + ctxLocalName(param) +
                ".size()));\n";
    }
    code += "            ctx.q.submit([&](sycl::handler& h) {\n";
    for (const auto& param : plan.params) {
        const std::string mode =
            param.reads && !param.writes ? "sycl::access::mode::read"
                                         : "sycl::access::mode::read_write";
        code += "                auto __or_acc_" + param.calcParamName +
                " = __or_buffer_" + param.calcParamName +
                ".get_access<" + mode + ">(h);\n";
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(ctx.__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    for (const auto& param : plan.params) {
        code += "                    auto* __or_data_" + param.calcParamName +
                " = __or_acc_" + param.calcParamName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        std::string offset = "item_linear";
        if (param.access == ParamAccessKind::ReplicatedScalar ||
            param.access == ParamAccessKind::ReplicatedFullTensor) {
            offset = "0";
        }
        code += "                    " + viewType(plan, param) + " view_" +
                param.calcParamName + "{__or_data_" + param.calcParamName +
                ", " + offset + "};\n";
    }
    code += "                    " + plan.exprNode.calc->getName() +
            "_mpi_local(";
    for (int paramIdx = 0; paramIdx < plan.exprNode.calc->getNumParams();
         ++paramIdx) {
        if (paramIdx != 0) {
            code += ", ";
        }
        code += "view_" + plan.exprNode.calc->getParam(paramIdx)->getName();
    }
    code += ");\n";
    code += "                });\n";
    code += "            });\n";
    code += "            ctx.q.wait();\n";
    code += "        }\n";
    code += "    }\n";
}

void emitMaterializeOutput(std::string& code,
                           const ShellPartitionPlan& plan,
                           const ParamAccessPlan& param,
                           const std::string& localBufferName) {
    const std::string type = elemType(plan, param);
    const std::string mpiType = mpiDatatypeFor(type);
    const std::string global = "__or_materialized_" + param.calcParamName;
    code += "    std::vector<" + type + "> " + global + ";\n";
    code += "    if (ctx.mpi_rank == 0) {\n";
    code += "        " + global +
            ".resize(static_cast<std::size_t>(ctx.__or_total_items));\n";
    code += "    }\n";
    code += "    auto dacpp_profile_gather_start_" + param.calcParamName +
            " = dacpp::mpi::profileSegmentStart();\n";
    if (usesByte(plan, param)) {
        code += "    std::vector<int> __or_counts_bytes_" +
                param.calcParamName + "_gather;\n";
        code += "    std::vector<int> __or_displs_bytes_" +
                param.calcParamName + "_gather;\n";
        code += "    dacpp::mpi::operator_resident::byte_counts_displs(ctx.__or_counts, ctx.__or_displs, sizeof(" +
                type + "), __or_counts_bytes_" + param.calcParamName +
                "_gather, __or_displs_bytes_" + param.calcParamName +
                "_gather);\n";
        code += "    MPI_Gatherv(" + localBufferName +
                ".data(), static_cast<int>(ctx.__or_local_item_count * sizeof(" +
                type + ")), MPI_BYTE, ctx.mpi_rank == 0 ? " + global +
                ".data() : nullptr, ctx.mpi_rank == 0 ? __or_counts_bytes_" +
                param.calcParamName +
                "_gather.data() : nullptr, ctx.mpi_rank == 0 ? __or_displs_bytes_" +
                param.calcParamName +
                "_gather.data() : nullptr, MPI_BYTE, 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Gatherv(" + localBufferName +
                ".data(), static_cast<int>(ctx.__or_local_item_count), " +
                mpiType + ", ctx.mpi_rank == 0 ? " + global +
                ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, " +
                mpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_" +
            param.calcParamName + ");\n";
    code += "    auto dacpp_profile_materialize_start_" + param.calcParamName +
            " = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (ctx.mpi_rank == 0) {\n";
    code += "        " + paramVarName(param) + ".array2Tensor(" + global +
            ");\n";
    if (param.broadcastMaterializedOutput) {
        code += "        if (!" + global + ".empty()) {\n";
        code += "            auto dacpp_profile_bcast_start_" +
                param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        code += "            MPI_Bcast(" + global + ".data(), " +
                mpiPayloadCountExpr(paramVarName(param) + ".getSize()", type) +
                ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "            dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_" +
                param.calcParamName + ");\n";
        code += "        }\n";
    }
    if (param.broadcastMaterializedOutput) {
        code += "    } else {\n";
        code += "        " + global + ".resize(static_cast<std::size_t>(" +
                paramVarName(param) + ".getSize()));\n";
        code += "        if (!" + global + ".empty()) {\n";
        code += "            auto dacpp_profile_bcast_start_" +
                param.calcParamName +
                " = dacpp::mpi::profileSegmentStart();\n";
        code += "            MPI_Bcast(" + global + ".data(), " +
                mpiPayloadCountExpr(paramVarName(param) + ".getSize()", type) +
                ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "            dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_" +
                param.calcParamName + ");\n";
        code += "        }\n";
        code += "        " + paramVarName(param) + ".array2Tensor(" + global +
                ");\n";
        code += "    }\n";
    } else {
        code += "    }\n";
    }
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start_" +
            param.calcParamName + ");\n";
}

void emitRunFunction(std::string& code,
                     const std::string& ctxName,
                     const std::string& runName,
                     const ShellPartitionPlan& plan) {
    code += "void " + runName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    const bool hasScalarRefresh = [&]() {
        for (const auto& param : plan.params) {
            if (param.access == ParamAccessKind::ReplicatedScalar) {
                return true;
            }
        }
        return false;
    }();
    if (hasScalarRefresh) {
        code += "    auto dacpp_profile_bcast_start = dacpp::mpi::profileSegmentStart();\n";
    }
    emitScalarRefreshes(code, plan);
    if (hasScalarRefresh) {
        code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start);\n";
    }
    code += "    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
    emitRunKernel(code, plan);
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
    for (const auto& param : plan.params) {
        if (!param.writes || param.access != ParamAccessKind::OutputDirect) {
            continue;
        }
        const std::string type = elemType(plan, param);
        if (!param.materializeAfterWrite || param.retainResidentAfterWrite) {
            code += "    auto& __or_resident_out_" + param.calcParamName +
                    " = dacpp::mpi::operator_resident::ensure_resident<" +
                    type + ">(" + paramVarName(param) + ", " +
                    ctxLocalName(param) + ".size());\n";
            code += "    __or_resident_out_" + param.calcParamName + " = " +
                    ctxLocalName(param) + ";\n";
        } else {
            code += "    // No downstream resident reader for " +
                    param.calcParamName +
                    "; host materialization below preserves visibility.\n";
        }
        if (plan.loopLowerMaterializeEveryRun || param.materializeAfterWrite) {
            emitMaterializeOutput(code, plan, param, ctxLocalName(param));
        }
    }
    code += "}\n";
}

void emitMaterializeFunction(std::string& code,
                             const std::string& ctxName,
                             const std::string& materializeName,
                             const ShellPartitionPlan& plan) {
    code += "void " + materializeName + "(" + ctxName + "& ctx, " +
            wrapperSignature(plan) + ") {\n";
    for (const auto& param : plan.params) {
        if (!param.writes || param.access != ParamAccessKind::OutputDirect) {
            continue;
        }
        if (plan.loopLowerMaterializeEveryRun) {
            code += "    (void)ctx;\n";
            code += "    dacpp::mpi::reportSegmentedProfile(\"" + materializeName +
                    "\", ctx.__or_profile, MPI_COMM_WORLD);\n";
            code += "}\n";
            return;
        }
        emitMaterializeOutput(code, plan, param, ctxLocalName(param));
    }
    code += "    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    code += "    dacpp::mpi::reportSegmentedProfile(\"" + materializeName +
            "\", ctx.__or_profile, MPI_COMM_WORLD);\n";
    code += "}\n";
}

} // namespace

std::string buildLoopLoweredDirect1DFamilyCode(
    const std::string& baseName,
    DacppFile*,
    const ShellPartitionPlan& plan) {
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
    emitContextType(code, ctxName, plan);
    emitInitFunction(code, ctxName, initName, plan);
    emitRunFunction(code, ctxName, runName, plan);
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
    if (!plan.loopLowerMaterializeEveryRun) {
        code += "    " + materializeName + "(ctx";
        for (const auto& param : plan.params) {
            code += ", " + paramVarName(param);
        }
        code += ");\n";
    }
    code += "}\n";
    return code;
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
