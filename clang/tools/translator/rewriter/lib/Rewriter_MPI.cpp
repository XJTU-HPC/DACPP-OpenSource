#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/StringExtras.h"

#include "Rewriter.h"

namespace {

using namespace dacppTranslator;

std::string mpiDatatypeFor(const std::string& type) {
    if (type == "float") return "MPI_FLOAT";
    if (type == "double") return "MPI_DOUBLE";
    if (type == "long double") return "MPI_LONG_DOUBLE";
    if (type == "int") return "MPI_INT";
    if (type == "short" || type == "short int") return "MPI_SHORT";
    if (type == "long") return "MPI_LONG";
    if (type == "long long") return "MPI_LONG_LONG";
    if (type == "char") return "MPI_CHAR";
    if (type == "signed char") return "MPI_SIGNED_CHAR";
    if (type == "unsigned") return "MPI_UNSIGNED";
    if (type == "unsigned int") return "MPI_UNSIGNED";
    if (type == "unsigned short" || type == "unsigned short int") return "MPI_UNSIGNED_SHORT";
    if (type == "unsigned long") return "MPI_UNSIGNED_LONG";
    if (type == "unsigned long long") return "MPI_UNSIGNED_LONG_LONG";
    if (type == "unsigned char") return "MPI_UNSIGNED_CHAR";
    if (type == "bool") return "MPI_CXX_BOOL";
    if (type == "std::complex<float>" || type == "complex<float>") {
        return "MPI_C_FLOAT_COMPLEX";
    }
    if (type == "std::complex<double>" || type == "complex<double>") {
        return "MPI_C_DOUBLE_COMPLEX";
    }
    if (type == "std::complex<long double>" || type == "complex<long double>") {
        return "MPI_C_LONG_DOUBLE_COMPLEX";
    }
    return "MPI_BYTE";
}

bool usesByteTransport(const std::string& type) {
    return mpiDatatypeFor(type) == "MPI_BYTE";
}

std::string mpiPayloadCountExpr(const std::string& elemCountExpr,
                                const std::string& type) {
    if (usesByteTransport(type)) {
        return "static_cast<int>((" + elemCountExpr + ") * sizeof(" + type + "))";
    }
    return elemCountExpr;
}

std::string toPlannerMode(IOTYPE mode) {
    switch (mode) {
    case IOTYPE::READ:
        return "dacpp::mpi::AccessMode::Read";
    case IOTYPE::WRITE:
        return "dacpp::mpi::AccessMode::Write";
    case IOTYPE::READ_WRITE:
        return "dacpp::mpi::AccessMode::ReadWrite";
    }
    return "dacpp::mpi::AccessMode::Read";
}

std::string toAccessorMode(IOTYPE mode) {
    return mode == IOTYPE::READ ? "sycl::access::mode::read"
                                : "sycl::access::mode::read_write";
}

int inferViewRank(ShellParam* shellParam, Param* calcParam) {
    const std::string calcType = calcParam->getType();
    if (calcType.find('*') != std::string::npos) {
        return 1;
    }
    if (calcType.find("Vector<") != std::string::npos) {
        return 1;
    }
    if (calcType.find("Matrix<") != std::string::npos) {
        return 2;
    }

    const int calcDim = calcParam->getDimension();
    if (calcDim > 0) {
        return calcDim;
    }

    const int shellDim = shellParam->getDimension();
    return shellDim > 0 ? shellDim : 1;
}

std::string toViewType(ShellParam* shellParam, Param* calcParam) {
    const std::string baseType = calcParam->getBasicType();
    const bool isReadOnly = shellParam->getRw() == IOTYPE::READ;
    const std::string qualifiedType = isReadOnly ? ("const " + baseType) : baseType;
    const int dim = inferViewRank(shellParam, calcParam);

    if (dim <= 1) {
        return "dacpp::mpi::View1D<" + qualifiedType + ">";
    }
    return "dacpp::mpi::View2D<" + qualifiedType + ">";
}

std::string joinCalcBody(const Calc* calc) {
    std::string body;
    for (const auto& block : calc->body) {
        body += block;
        if (!body.empty() && body.back() != '\n') {
            body += "\n";
        }
    }
    return body;
}

void collectReturnStmts(const clang::Stmt* stmt,
                        std::vector<const clang::ReturnStmt*>& returns) {
    if (!stmt) {
        return;
    }
    if (const auto* returnStmt = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
        returns.push_back(returnStmt);
    }
    for (const clang::Stmt* child : stmt->children()) {
        collectReturnStmts(child, returns);
    }
}

struct SplitBindMeta {
    int bindId = 0;
    std::string offset = "0";
};

std::unordered_map<std::string, SplitBindMeta>
collectSplitBindMeta(Shell* shell) {
    std::unordered_map<std::string, int> splitBindId;
    std::unordered_map<std::string, SplitBindMeta> splitMeta;

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            auto [it, inserted] =
                splitBindId.emplace(split->getId(), static_cast<int>(splitBindId.size()));
            splitMeta[split->getId()] = SplitBindMeta{it->second, "0"};
        }
    }

    return splitMeta;
}

std::string buildPatternInitCode(Shell* shell,
                                 Calc* calc,
                                 const std::unordered_map<std::string, SplitBindMeta>& splitMeta) {
    std::string code;
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "pattern_" + name;

        code += "    dacpp::mpi::AccessPattern " + patternName + ";\n";
        code += "    " + patternName + ".param_id = " + std::to_string(paramIdx) + ";\n";
        code += "    " + patternName + ".name = \"" + name + "\";\n";
        code += "    " + patternName + ".mode = " + toPlannerMode(shellParam->getRw()) + ";\n";
        code += "    " + patternName + ".data_info.dim = " + tensorName + ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName +
                ".getDim(); ++dim) " + patternName +
                ".data_info.dimLength.push_back(" + tensorName + ".getShape(dim));\n";

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            auto metaIt = splitMeta.find(split->getId());
            SplitBindMeta bindMeta;
            if (metaIt != splitMeta.end()) {
                bindMeta = metaIt->second;
            }

            const bool isIndex = split->type == "IndexSplit";
            const std::string opName =
                patternName + "_op_" + std::to_string(splitIdx);

            code += "    Dac_Op " + opName + ";\n";
            code += "    " + opName + ".setDimId(" + std::to_string(split->getDimIdx()) + ");\n";
            code += "    " + opName + ".size = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitSize()) +
                    ";\n";
            code += "    " + opName + ".stride = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitStride()) +
                    ";\n";
            if (isIndex) {
                code += "    " + opName + ".SetSplitSize(" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) + "));\n";
            } else {
                code += "    " + opName + ".SetSplitSize((" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) + ") - " +
                        std::to_string(static_cast<RegularSplit*>(split)->getSplitSize()) + ") / " +
                        std::to_string(static_cast<RegularSplit*>(split)->getSplitStride()) + " + 1);\n";
            }
            code += "    " + patternName + ".param_ops.push_back(" + opName + ");\n";
            code += "    " + patternName + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    " + patternName + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    " + patternName + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + ");\n";
        }

        code += "    " + patternName +
                ".partition_shape = dacpp::mpi::init_partition_shape(" + patternName + ");\n";
        code += "    " + patternName +
                ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(" + patternName + ");\n";
        code += "    if (binding_split_sizes.size() < " + patternName +
                ".bind_split_sizes.size()) binding_split_sizes.resize(" + patternName +
                ".bind_split_sizes.size(), 1);\n";
        code += "    for (std::size_t bind_i = 0; bind_i < " + patternName +
                ".bind_split_sizes.size(); ++bind_i) {\n";
        code += "        binding_split_sizes[bind_i] = std::max<int64_t>(binding_split_sizes[bind_i], " +
                patternName + ".bind_split_sizes[bind_i]);\n";
        code += "    }\n";
    }

    return code;
}

std::string buildLocalCalcCode(Shell* shell, Calc* calc) {
    std::string code = "inline void " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += toViewType(shell->getShellParam(paramIdx), calc->getParam(paramIdx)) + " " +
                calc->getParam(paramIdx)->getName();
        if (paramIdx + 1 != calc->getNumParams()) {
            code += ", ";
        }
    }
    code += ") ";
    code += joinCalcBody(calc);
    code += "\n";
    return code;
}

std::string buildPackBuilderExpr(IOTYPE mode, const std::string& patternName) {
    if (mode == IOTYPE::READ) {
        return "dacpp::mpi::build_input_pack_map(item_range, " + patternName + ")";
    }
    if (mode == IOTYPE::WRITE) {
        return "dacpp::mpi::build_output_pack_map(item_range, " + patternName + ")";
    }
    return "dacpp::mpi::build_rw_pack_map(item_range, " + patternName + ")";
}

std::string buildRemotePackBuilderExpr(IOTYPE mode,
                                       const std::string& rangeName,
                                       const std::string& patternName) {
    if (mode == IOTYPE::READ) {
        return "dacpp::mpi::build_input_pack_map(" + rangeName + ", " + patternName + ")";
    }
    if (mode == IOTYPE::WRITE) {
        return "dacpp::mpi::build_output_pack_map(" + rangeName + ", " + patternName + ")";
    }
    return "dacpp::mpi::build_rw_pack_map(" + rangeName + ", " + patternName + ")";
}

std::string buildWrapperCode(Shell* shell, Calc* calc) {
    const std::string wrapperName = shell->getName() + "_" + calc->getName();

    std::string signature;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        signature += param->getType() + " " + param->getName();
        if (paramIdx + 1 != shell->getNumParams()) {
            signature += ", ";
        }
    }

    const auto splitMeta = collectSplitBindMeta(shell);

    std::string code;
    code += "void " + wrapperName + "(" + signature + ") {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    code += "    std::vector<int64_t> binding_split_sizes;\n";
    code += buildPatternInitCode(shell, calc, splitMeta);
    code += "    int64_t total_items = 1;\n";
    code += "    for (int64_t split_size : binding_split_sizes) total_items *= split_size;\n";
    code += "    auto item_range = dacpp::mpi::get_rank_item_range(total_items, mpi_rank, mpi_size);\n";
    code += "    const int64_t local_item_count = item_range.size();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        code += "    pattern_" + calcName + ".bind_split_sizes = binding_split_sizes;\n";
    }

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "pattern_" + calcName;
        const std::string packName = "pack_" + calcName;
        const std::string slotsName = "slots_" + calcName;
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_" + calcName;
        const std::string wbName = "writeback_" + calcName;
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());
        const std::string payloadPeerCount =
            mpiPayloadCountExpr("peer_count", calcParam->getBasicType());
        const std::string payloadRecvCount =
            mpiPayloadCountExpr("recv_count", calcParam->getBasicType());

        code += "    auto " + packName + " = " +
                buildPackBuilderExpr(shellParam->getRw(), patternName) + ";\n";
        code += "    auto " + slotsName + " = dacpp::mpi::build_item_slots(item_range, " +
                patternName + ", " + packName + ");\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> " + localName + "(" +
                packName + ".globals.size());\n";

        if (shellParam->getRw() != IOTYPE::WRITE) {
            code += "    if (mpi_rank == 0) {\n";
            code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName + ";\n";
            code += "        " + tensorName + ".tensor2Array(" + globalName + ");\n";
            code += "        " + localName +
                    " = dacpp::mpi::pack_values_by_globals(" + globalName + ", " + packName +
                    ".globals);\n";
            code += "        for (int peer = 1; peer < mpi_size; ++peer) {\n";
            code += "            auto peer_range = dacpp::mpi::get_rank_item_range(total_items, peer, mpi_size);\n";
            code += "            auto peer_pack = " +
                    buildRemotePackBuilderExpr(shellParam->getRw(), "peer_range", patternName) +
                    ";\n";
            code += "            auto peer_values = dacpp::mpi::pack_values_by_globals(" + globalName +
                    ", peer_pack.globals);\n";
            code += "            int peer_count = static_cast<int>(peer_values.size());\n";
            code += "            MPI_Send(&peer_count, 1, MPI_INT, peer, " +
                    std::to_string(1000 + paramIdx) + ", MPI_COMM_WORLD);\n";
            code += "            if (peer_count > 0) {\n";
            code += "                MPI_Send(peer_values.data(), " + payloadPeerCount + ", " + mpiType +
                    ", peer, " + std::to_string(2000 + paramIdx) + ", MPI_COMM_WORLD);\n";
            code += "            }\n";
            code += "        }\n";
            code += "    } else {\n";
            code += "        int recv_count = 0;\n";
            code += "        MPI_Recv(&recv_count, 1, MPI_INT, 0, " +
                    std::to_string(1000 + paramIdx) + ", MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n";
            code += "        " + localName + ".resize(recv_count);\n";
            code += "        if (recv_count > 0) {\n";
            code += "            MPI_Recv(" + localName + ".data(), " + payloadRecvCount + ", " + mpiType +
                    ", 0, " + std::to_string(2000 + paramIdx) +
                    ", MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n";
            code += "        }\n";
            code += "    }\n";
        }

        code += "    const int " + calcName + "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(" +
                patternName + "));\n";
        if (inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + calcName + "_cols = " + patternName + ".partition_shape[1];\n";
        }
    }

    code += "    if (local_item_count > 0) {\n";
    code += "        {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string localName = "local_" + name;
        const std::string slotsName = "slots_" + name;
        code += "            sycl::buffer<" + calcParam->getBasicType() + ", 1> buffer_" + name +
                "(" + localName + ".data(), sycl::range<1>(" + localName + ".size()));\n";
        code += "            sycl::buffer<int32_t, 1> slots_buffer_" + name + "(" + slotsName +
                ".data(), sycl::range<1>(" + slotsName + ".size()));\n";
    }
    code += "            q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto acc_" + name + " = buffer_" + name + ".get_access<" +
                toAccessorMode(shellParam->getRw()) + ">(h);\n";
        code += "                auto slots_acc_" + name +
                " = slots_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                    auto* data_" + name +
                " = acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* slots_" + name +
                " = slots_acc_" + name +
                ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (inferViewRank(shellParam, calcParam) <= 1) {
            code += "                    " + toViewType(shellParam, calcParam) + " view_" + name +
                    "{data_" + name + ", slots_" + name + ", item_linear * " + name +
                    "_partition_size};\n";
        } else {
            code += "                    " + toViewType(shellParam, calcParam) + " view_" + name +
                    "{data_" + name + ", slots_" + name + ", item_linear * " + name +
                    "_partition_size, " + name + "_cols};\n";
        }
    }
    code += "                    " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += "view_" + calc->getParam(paramIdx)->getName();
        if (paramIdx + 1 != calc->getNumParams()) {
            code += ", ";
        }
    }
    code += ");\n";
    code += "                });\n";
    code += "            });\n";
    code += "            q.wait();\n";
    code += "        }\n";
    code += "    }\n";

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        if (shellParam->getRw() == IOTYPE::READ) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string packName = "pack_" + name;
        const std::string localName = "local_" + name;
        const std::string globalName = "global_out_" + name;
        const std::string wbName = "writeback_" + name;
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());
        const std::string payloadRecvCount =
            mpiPayloadCountExpr("recv_count", calcParam->getBasicType());
        const std::string payloadSendCount =
            mpiPayloadCountExpr("send_count", calcParam->getBasicType());
        const std::string payloadSyncedCount =
            mpiPayloadCountExpr("synced_count_" + name, calcParam->getBasicType());

        code += "    auto " + wbName + " = dacpp::mpi::build_writeback_values(" + localName +
                ", " + packName + ");\n";
        code += "    const auto& writeback_globals_" + name + " = " + packName +
                ".writeback_globals.empty() ? " + packName + ".globals : " + packName +
                ".writeback_globals;\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> synced_" + name + ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName + ";\n";
        code += "        " + tensorName + ".tensor2Array(" + globalName + ");\n";
        code += "        dacpp::mpi::apply_writeback_by_globals(" + wbName +
                ", writeback_globals_" + name + ", " + globalName + ");\n";
        code += "        for (int peer = 1; peer < mpi_size; ++peer) {\n";
        code += "            int recv_count = 0;\n";
        code += "            MPI_Recv(&recv_count, 1, MPI_INT, peer, " +
                std::to_string(3000 + paramIdx) + ", MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n";
        code += "            if (recv_count <= 0) continue;\n";
        code += "            std::vector<int64_t> recv_globals(recv_count);\n";
        code += "            std::vector<" + calcParam->getBasicType() + "> recv_values(recv_count);\n";
        code += "            MPI_Recv(recv_globals.data(), recv_count, MPI_LONG_LONG, peer, " +
                std::to_string(4000 + paramIdx) + ", MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n";
        code += "            MPI_Recv(recv_values.data(), " + payloadRecvCount + ", " + mpiType + ", peer, " +
                std::to_string(5000 + paramIdx) + ", MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n";
        code += "            dacpp::mpi::apply_writeback_by_globals(recv_values, recv_globals, " +
                globalName + ");\n";
        code += "        }\n";
        code += "        " + tensorName + ".array2Tensor(" + globalName + ");\n";
        code += "        synced_" + name + " = " + globalName + ";\n";
        code += "    } else {\n";
        code += "        int send_count = static_cast<int>(writeback_globals_" + name + ".size());\n";
        code += "        MPI_Send(&send_count, 1, MPI_INT, 0, " +
                std::to_string(3000 + paramIdx) + ", MPI_COMM_WORLD);\n";
        code += "        if (send_count > 0) {\n";
        code += "            MPI_Send(const_cast<int64_t*>(writeback_globals_" + name +
                ".data()), send_count, MPI_LONG_LONG, 0, " +
                std::to_string(4000 + paramIdx) + ", MPI_COMM_WORLD);\n";
        code += "            MPI_Send(" + wbName + ".data(), " + payloadSendCount + ", " + mpiType +
                ", 0, " + std::to_string(5000 + paramIdx) + ", MPI_COMM_WORLD);\n";
        code += "        }\n";
        code += "    }\n";
        code += "    int synced_count_" + name + " = 0;\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        synced_count_" + name + " = static_cast<int>(synced_" + name + ".size());\n";
        code += "    }\n";
        code += "    MPI_Bcast(&synced_count_" + name + ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
        code += "    if (mpi_rank != 0) {\n";
        code += "        synced_" + name + ".resize(synced_count_" + name + ");\n";
        code += "    }\n";
        code += "    if (synced_count_" + name + " > 0) {\n";
        code += "        MPI_Bcast(synced_" + name + ".data(), " + payloadSyncedCount + ", " + mpiType +
                ", 0, MPI_COMM_WORLD);\n";
        code += "    }\n";
        code += "    if (mpi_rank != 0) {\n";
        code += "        " + tensorName + ".array2Tensor(synced_" + name + ");\n";
        code += "    }\n";
    }

    code += "}\n";
    return code;
}

std::string buildPrelude(DacppFile* dacppFile) {
    std::string code;
    std::set<std::string> seenHeaders;
    for (int idx = 0; idx < dacppFile->getNumHeaderFile(); ++idx) {
        const std::string header = dacppFile->getHeaderFile(idx)->getName();
        if (seenHeaders.insert(header).second) {
            code += "#include " + header + "\n";
        }
    }
    code += "\n";

    std::set<std::string> seenNamespaces;
    for (int idx = 0; idx < dacppFile->getNumNameSpace(); ++idx) {
        const std::string ns = dacppFile->getNameSpace(idx)->getName();
        if (seenNamespaces.insert(ns).second) {
            code += "using namespace " + ns + ";\n";
        }
    }
    code += "\n";
    return code;
}

}  // namespace

void dacppTranslator::Rewriter::rewriteMPI() {
    std::string generated = buildPrelude(dacppFile);

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();

        generated += buildLocalCalcCode(shell, calc);
        generated += "\n";
        generated += buildWrapperCode(shell, calc);
        generated += "\n";

        rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
        rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(), generated);

    const FunctionDecl* mainFunc = dacppFile->getMainFunction();
    if (!mainFunc) {
        return;
    }

    const auto* body = llvm::dyn_cast<CompoundStmt>(mainFunc->getBody());
    if (!body) {
        return;
    }

    const std::string mpiInit = R"(
    int dacpp_mpi_finalize_needed = 0;
    int dacpp_mpi_initialized = 0;
    MPI_Initialized(&dacpp_mpi_initialized);
    if (!dacpp_mpi_initialized) {
        int dacpp_mpi_argc = 0;
        char** dacpp_mpi_argv = nullptr;
        MPI_Init(&dacpp_mpi_argc, &dacpp_mpi_argv);
        dacpp_mpi_finalize_needed = 1;
    }
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_rank != 0) {
        freopen("/dev/null", "w", stdout);
    }
)";

    const std::string mpiFinish = R"(
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
    }
)";

    rewriter->InsertTextAfterToken(body->getLBracLoc(), mpiInit);
    std::vector<const clang::ReturnStmt*> returnStmts;
    collectReturnStmts(body, returnStmts);
    for (const clang::ReturnStmt* returnStmt : returnStmts) {
        rewriter->InsertTextBefore(returnStmt->getBeginLoc(), mpiFinish);
    }
    rewriter->InsertTextBefore(body->getRBracLoc(), mpiFinish);
}
