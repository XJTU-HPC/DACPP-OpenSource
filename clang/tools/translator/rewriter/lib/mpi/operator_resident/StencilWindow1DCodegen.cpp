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

const ParamAccessPlan* findScalarReader(const ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            return &param;
        }
    }
    return nullptr;
}

}  // namespace

std::string buildStencilWindow1DWrapperCode(
    const std::string& wrapperName,
    const ShellPartitionPlan& plan) {
    const ParamAccessPlan* reader = findStencilReader(plan);
    const ParamAccessPlan* writer = findStencilWriter(plan);
    const ParamAccessPlan* scalar = findScalarReader(plan);
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
    code += "    const int64_t __or_input_size = " + readerArg + ".getShape(0);\n";
    code += "    const int64_t __or_output_size = " + writerArg + ".getShape(0);\n";
    code += "    const auto __or_range = dacpp::mpi::operator_resident::rank_range_1d(__or_output_size, mpi_rank, mpi_size);\n";
    code += "    const int64_t __or_local_item_count = __or_range.count;\n";
    code += "    const int64_t __or_output_begin = __or_range.begin;\n";
    code += "    std::vector<int> __or_counts;\n";
    code += "    std::vector<int> __or_displs;\n";
    code += "    dacpp::mpi::operator_resident::counts_displs_1d(__or_output_size, mpi_size, __or_counts, __or_displs);\n";
    code += "    std::vector<" + readerType + "> __or_global_" + reader->calcParamName + ";\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        " + readerArg + ".tensor2Array(__or_global_" + reader->calcParamName + ");\n";
    code += "    }\n";
    code += "    __or_global_" + reader->calcParamName + ".resize(static_cast<std::size_t>(__or_input_size));\n";
    if (usesByte(plan, *reader)) {
        code += "    MPI_Bcast(__or_global_" + reader->calcParamName + ".data(), static_cast<int>(__or_input_size * sizeof(" + readerType + ")), MPI_BYTE, 0, MPI_COMM_WORLD);\n";
    } else {
        code += "    MPI_Bcast(__or_global_" + reader->calcParamName + ".data(), static_cast<int>(__or_input_size), " + readerMpiType + ", 0, MPI_COMM_WORLD);\n";
    }
    if (scalar) {
        emitScalarBroadcast(code, plan, *scalar);
    }
    code += "    std::vector<" + writerType + "> __or_local_" + writer->calcParamName + "(static_cast<std::size_t>(__or_local_item_count));\n";
    code += "    if (__or_local_item_count > 0) {\n";
    code += "        sycl::buffer<" + readerType + ", 1> __or_reader_buf(__or_global_" + reader->calcParamName + ".data(), sycl::range<1>(__or_global_" + reader->calcParamName + ".size()));\n";
    if (scalar) {
        const std::string scalarType = elemType(plan, *scalar);
        code += "        sycl::buffer<" + scalarType + ", 1> __or_scalar_buf(" +
                localName(*scalar) + ".data(), sycl::range<1>(" +
                localName(*scalar) + ".size()));\n";
    }
    code += "        sycl::buffer<" + writerType + ", 1> __or_writer_buf(__or_local_" + writer->calcParamName + ".data(), sycl::range<1>(__or_local_" + writer->calcParamName + ".size()));\n";
    code += "        q.submit([&](sycl::handler& h) {\n";
    code += "            auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    if (scalar) {
        code += "            auto __or_scalar_acc = __or_scalar_buf.get_access<sycl::access::mode::read>(h);\n";
    }
    code += "            auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                const int output_idx = static_cast<int>(__or_output_begin) + item_linear;\n";
    code += "                auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    if (scalar) {
        code += "                auto* __or_scalar_data = __or_scalar_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    }
    code += "                auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : plan.params) {
        const std::string paramType = elemType(plan, param);
        if (param.access == ParamAccessKind::StencilWindow) {
            code += "                dacpp::mpi::ContiguousView1D<const " + paramType + "> view_" + param.calcParamName + "{__or_reader_data, output_idx};\n";
            continue;
        }
        if (param.access == ParamAccessKind::ReplicatedScalar) {
            code += "                dacpp::mpi::ContiguousView1D<const " + paramType + "> view_" + param.calcParamName + "{__or_scalar_data, 0};\n";
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
    emitGatherMaterializeFromLocalBuffer(code, plan, *writer,
                                         "__or_local_" + writer->calcParamName,
                                         "__or_output_size");
    code += "}\n";
    return code;
}

}  // namespace operator_resident
}  // namespace mpi_rewriter
}  // namespace dacppTranslator
