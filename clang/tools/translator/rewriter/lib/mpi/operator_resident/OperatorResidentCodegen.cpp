#include <string>
#include <vector>

#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

std::string paramVarName(const ParamAccessPlan& param) {
    return "__or_arg" + std::to_string(param.paramIndex);
}

std::string wrapperSignature(const ShellPartitionPlan& plan) {
    std::string signature;
    Shell* shell = plan.exprNode.shell;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (!paramType.empty() && paramType.back() != '&' &&
            paramType.back() != '*') {
            paramType += "&";
        }
        signature += paramType + " " + paramVarName(plan.params[paramIdx]);
        if (paramIdx + 1 != shell->getNumParams()) {
            signature += ", ";
        }
    }
    return signature;
}

std::string elemType(const ShellPartitionPlan& plan,
                     const ParamAccessPlan& param) {
    return plan.exprNode.calc->getParam(param.paramIndex)->getBasicType();
}

std::string viewType(const ShellPartitionPlan& plan,
                     const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    if (param.reads && !param.writes) {
        return "dacpp::mpi::ContiguousView1D<const " + type + ">";
    }
    return "dacpp::mpi::ContiguousView1D<" + type + ">";
}

std::string localName(const ParamAccessPlan& param) {
    return "__or_local_" + param.calcParamName;
}

std::string globalName(const ParamAccessPlan& param) {
    return "__or_global_" + param.calcParamName;
}

std::string scalarName(const ParamAccessPlan& param) {
    return "__or_scalar_" + param.calcParamName;
}

bool usesByte(const ShellPartitionPlan& plan, const ParamAccessPlan& param) {
    return usesByteTransport(elemType(plan, param));
}

void emitByteCounts(std::string& code, const std::string& suffix,
                    const std::string& type) {
    code += "    std::vector<int> __or_counts_bytes_" + suffix + ";\n";
    code += "    std::vector<int> __or_displs_bytes_" + suffix + ";\n";
    code += "    dacpp::mpi::operator_resident::byte_counts_displs(__or_counts, __or_displs, sizeof(" +
            type + "), __or_counts_bytes_" + suffix + ", __or_displs_bytes_" +
            suffix + ");\n";
}

void emitScatter(std::string& code,
                 const ShellPartitionPlan& plan,
                 const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    const std::string mpiType = mpiDatatypeFor(type);
    const std::string local = localName(param);
    const std::string global = globalName(param);

    code += "    std::vector<" + type + "> " + local +
            "(static_cast<std::size_t>(__or_local_item_count));\n";
    code += "    std::vector<" + type + "> " + global + ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + paramVarName(param) + ".tensor2Array(" + global +
            ");\n";
    code += "    }\n";
    if (usesByte(plan, param)) {
        emitByteCounts(code, param.calcParamName, type);
        code += "    MPI_Scatterv(mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts_bytes_" +
                param.calcParamName +
                ".data() : nullptr, mpi_rank == 0 ? __or_displs_bytes_" +
                param.calcParamName +
                ".data() : nullptr, " + mpiType + ", " + local +
                ".data(), static_cast<int>(__or_local_item_count * sizeof(" +
                type + ")), " + mpiType + ", 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Scatterv(mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, " +
                mpiType + ", " + local +
                ".data(), static_cast<int>(__or_local_item_count), " +
                mpiType + ", 0, MPI_COMM_WORLD);\n";
    }
}

void emitResidentOrScatter(std::string& code,
                           const ShellPartitionPlan& plan,
                           const ParamAccessPlan& param) {
    if (!param.readFromResident) {
        emitScatter(code, plan, param);
        return;
    }

    const std::string type = elemType(plan, param);
    const std::string local = localName(param);
    code += "    std::vector<" + type + "> " + local + ";\n";
    code += "    if (auto* __or_resident_" + param.calcParamName +
            " = dacpp::mpi::operator_resident::find_resident<" + type +
            ">(" + paramVarName(param) + ")) {\n";
    code += "        " + local + " = *__or_resident_" + param.calcParamName +
            ";\n";
    code += "    } else {\n";
    code += "        if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] missing resident tensor " +
            param.shellParamName + "\\n\");\n";
    code += "        MPI_Abort(MPI_COMM_WORLD, 3);\n";
    code += "    }\n";
}

void emitScalarBroadcast(std::string& code,
                         const ShellPartitionPlan& plan,
                         const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    const std::string scalar = scalarName(param);
    const std::string local = localName(param);
    code += "    if (" + paramVarName(param) + ".getSize() != 1) {\n";
    code += "        if (mpi_rank == 0) std::fprintf(stderr, \"[DACPP][MPI][OR] scalar parameter " +
            param.shellParamName + " expected size 1\\n\");\n";
    code += "        MPI_Abort(MPI_COMM_WORLD, 2);\n";
    code += "    }\n";
    code += "    " + type + " " + scalar + "{};\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        std::vector<" + type + "> __or_scalar_vec_" +
            param.calcParamName + ";\n";
    code += "        " + paramVarName(param) + ".tensor2Array(__or_scalar_vec_" +
            param.calcParamName + ");\n";
    code += "        if (!__or_scalar_vec_" + param.calcParamName +
            ".empty()) " + scalar + " = __or_scalar_vec_" +
            param.calcParamName + "[0];\n";
    code += "    }\n";
    if (usesByte(plan, param)) {
        code += "    MPI_Bcast(&" + scalar + ", static_cast<int>(sizeof(" +
                type + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Bcast(&" + scalar + ", 1, " +
                mpiDatatypeFor(type) + ", 0, MPI_COMM_WORLD);\n";
    }
    code += "    std::vector<" + type + "> " + local + "(1, " + scalar +
            ");\n";
}

void emitOutputBuffer(std::string& code,
                      const ShellPartitionPlan& plan,
                      const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    code += "    std::vector<" + type + "> " + localName(param) +
            "(static_cast<std::size_t>(__or_local_item_count));\n";
}

void emitParamLocalStorage(std::string& code,
                           const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            emitScalarBroadcast(code, plan, param);
        } else if (param.writes) {
            emitOutputBuffer(code, plan, param);
        } else {
            emitResidentOrScatter(code, plan, param);
        }
    }
}

void emitPartitionCode(std::string& code, const ShellPartitionPlan& plan) {
    const auto& firstDomain = plan.bindDomains.front();
    const std::string firstTensor =
        paramVarName(plan.params[static_cast<std::size_t>(
            firstDomain.runtimeSizeParam)]);
    if (plan.signature.layout == LocalLayoutKind::Contiguous1D) {
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
    code += "    const int64_t __or_total_items = __or_rows * __or_cols;\n";
    code += "    const auto __or_row_range = dacpp::mpi::operator_resident::rank_range_1d(__or_rows, mpi_rank, mpi_size);\n";
    code += "    const int64_t __or_local_item_count = __or_row_range.count * __or_cols;\n";
    code += "    std::vector<int> __or_row_counts;\n";
    code += "    std::vector<int> __or_row_displs;\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(__or_rows, mpi_size, __or_row_counts, __or_row_displs);\n";
    code += "    std::vector<int> __or_counts(mpi_size);\n";
    code += "    std::vector<int> __or_displs(mpi_size);\n";
    code += "    for (int r = 0; r < mpi_size; ++r) {\n";
    code += "        __or_counts[r] = static_cast<int>(__or_row_counts[r] * __or_cols);\n";
    code += "        __or_displs[r] = static_cast<int>(__or_row_displs[r] * __or_cols);\n";
    code += "    }\n";
}

void emitKernel(std::string& code,
                const ShellPartitionPlan& plan) {
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        {\n";
    for (const auto& param : plan.params) {
        const std::string type = elemType(plan, param);
        const std::string name = param.calcParamName;
        code += "            sycl::buffer<" + type + ", 1> __or_buffer_" +
                name + "(" + localName(param) +
                ".data(), sycl::range<1>(" + localName(param) +
                ".size()));\n";
    }
    code += "            q.submit([&](sycl::handler& h) {\n";
    for (const auto& param : plan.params) {
        const std::string mode =
            param.reads && !param.writes ? "sycl::access::mode::read"
                                         : "sycl::access::mode::read_write";
        code += "                auto __or_acc_" + param.calcParamName +
                " = __or_buffer_" + param.calcParamName +
                ".get_access<" + mode + ">(h);\n";
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    for (const auto& param : plan.params) {
        code += "                    auto* __or_data_" + param.calcParamName +
                " = __or_acc_" + param.calcParamName +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        const std::string offset =
            param.access == ParamAccessKind::ReplicatedScalar ? "0"
                                                              : "item_linear";
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
    code += "            q.wait();\n";
    code += "        }\n";
    code += "    }\n";
}

void emitGatherMaterialize(std::string& code,
                           const ShellPartitionPlan& plan,
                           const ParamAccessPlan& param) {
    const std::string type = elemType(plan, param);
    const std::string mpiType = mpiDatatypeFor(type);
    const std::string local = localName(param);
    const std::string global = "__or_materialized_" + param.calcParamName;
    code += "    std::vector<" + type + "> " + global + ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + global + ".resize(static_cast<std::size_t>(__or_total_items));\n";
    code += "    }\n";
    if (usesByte(plan, param)) {
        emitByteCounts(code, param.calcParamName + "_gather", type);
        code += "    MPI_Gatherv(" + local +
                ".data(), static_cast<int>(__or_local_item_count * sizeof(" +
                type + ")), " + mpiType + ", mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts_bytes_" +
                param.calcParamName +
                "_gather.data() : nullptr, mpi_rank == 0 ? __or_displs_bytes_" +
                param.calcParamName +
                "_gather.data() : nullptr, " + mpiType +
                ", 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Gatherv(" + local +
                ".data(), static_cast<int>(__or_local_item_count), " +
                mpiType + ", mpi_rank == 0 ? " + global +
                ".data() : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, " +
                mpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + paramVarName(param) + ".array2Tensor(" + global +
            ");\n";
    code += "    }\n";
}

void emitResidencyAndMaterialization(std::string& code,
                                     const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (!param.writes || param.access != ParamAccessKind::OutputDirect) {
            continue;
        }
        const std::string type = elemType(plan, param);
        code += "    auto& __or_resident_out_" + param.calcParamName +
                " = dacpp::mpi::operator_resident::ensure_resident<" + type +
                ">(" + paramVarName(param) + ", " + localName(param) +
                ".size());\n";
        code += "    __or_resident_out_" + param.calcParamName + " = " +
                localName(param) + ";\n";
        if (param.materializeAfterWrite) {
            emitGatherMaterialize(code, plan, param);
        }
    }
}

} // namespace

std::string operatorResidentWrapperName(Shell* shell,
                                        Calc* calc,
                                        int exprIndex) {
    return "__dacpp_mpi_or_" + shell->getName() + "_" + calc->getName() +
           "_" + std::to_string(exprIndex);
}

std::string buildOperatorResidentWrapperCode(
    DacppFile*,
    const OperatorResidentChainPlan&,
    const ShellPartitionPlan& exprPlan) {
    Shell* shell = exprPlan.exprNode.shell;
    Calc* calc = exprPlan.exprNode.calc;
    const std::string wrapper =
        operatorResidentWrapperName(shell, calc, exprPlan.exprIndex);

    std::string code;
    code += "void " + wrapper + "(" + wrapperSignature(exprPlan) + ") {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    emitPartitionCode(code, exprPlan);
    emitParamLocalStorage(code, exprPlan);
    emitKernel(code, exprPlan);
    emitResidencyAndMaterialization(code, exprPlan);
    code += "}\n";
    return code;
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
