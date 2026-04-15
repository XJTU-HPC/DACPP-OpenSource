#include <algorithm>
#include <cctype>
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
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/StringExtras.h"

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

using namespace dacppTranslator;

class TensorUseVisitor : public clang::RecursiveASTVisitor<TensorUseVisitor> {
public:
    std::string TargetName;
    const std::vector<const clang::BinaryOperator*>& DacExprs;
    
    bool NeedsBcast = false;
    int InsideDacExpr = 0;

    TensorUseVisitor(std::string name, const std::vector<const clang::BinaryOperator*>& dacExprs) 
        : TargetName(std::move(name)), DacExprs(dacExprs) {}

    bool TraverseStmt(clang::Stmt* S) {
        if (!S) return true;
        
        bool isDacExpr = false;
        for (auto* expr : DacExprs) {
            if (S == expr) {
                isDacExpr = true;
                break;
            }
        }

        if (isDacExpr) InsideDacExpr++;
        
        bool result = clang::RecursiveASTVisitor<TensorUseVisitor>::TraverseStmt(S);
        
        if (isDacExpr) InsideDacExpr--;
        
        return result;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
        if (NeedsBcast) return true;
        
        if (DRE->getDecl() && DRE->getDecl()->getNameAsString() == TargetName) {
            if (InsideDacExpr == 0) {
                NeedsBcast = true;
            }
        }
        return true;
    }
};

class ParamAccessVisitor : public clang::RecursiveASTVisitor<ParamAccessVisitor> {
public:
    const std::unordered_map<const clang::ValueDecl*, int>& ParamIndices;
    std::vector<bool> Reads;
    std::vector<bool> UpdateReads;
    std::vector<bool> Writes;

    explicit ParamAccessVisitor(
        const std::unordered_map<const clang::ValueDecl*, int>& paramIndices,
        int paramCount)
        : ParamIndices(paramIndices),
          Reads(paramCount, false),
          UpdateReads(paramCount, false),
          Writes(paramCount, false) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
        if (!DRE) {
            return true;
        }

        auto it = ParamIndices.find(DRE->getDecl());
        if (it == ParamIndices.end()) {
            return true;
        }

        if (WriteDepth > 0) {
            Writes[it->second] = true;
        } else if (UpdateReadDepth > 0) {
            UpdateReads[it->second] = true;
        } else {
            Reads[it->second] = true;
        }
        return true;
    }

    bool TraverseBinaryOperator(clang::BinaryOperator* BO) {
        if (!BO) {
            return true;
        }

        if (BO->isAssignmentOp()) {
            ++WriteDepth;
            TraverseStmt(BO->getLHS());
            --WriteDepth;
            TraverseStmt(BO->getRHS());
            return true;
        }

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseBinaryOperator(BO);
    }

    bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* BO) {
        if (!BO) {
            return true;
        }

        ++WriteDepth;
        TraverseStmt(BO->getLHS());
        --WriteDepth;
        ++UpdateReadDepth;
        TraverseStmt(BO->getLHS());
        --UpdateReadDepth;
        TraverseStmt(BO->getRHS());
        return true;
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* UO) {
        if (!UO) {
            return true;
        }

        if (UO->isIncrementDecrementOp()) {
            ++WriteDepth;
            TraverseStmt(UO->getSubExpr());
            --WriteDepth;
            ++UpdateReadDepth;
            TraverseStmt(UO->getSubExpr());
            --UpdateReadDepth;
            return true;
        }

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseUnaryOperator(UO);
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* OpCall) {
        if (!OpCall) {
            return true;
        }

        if (OpCall->isAssignmentOp()) {
            if (OpCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(OpCall->getArg(0));
                --WriteDepth;

                if (OpCall->getOperator() != clang::OO_Equal) {
                    ++UpdateReadDepth;
                    TraverseStmt(OpCall->getArg(0));
                    --UpdateReadDepth;
                }
            }

            for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs(); ++argIdx) {
                TraverseStmt(OpCall->getArg(argIdx));
            }
            return true;
        }

        if (OpCall->getOperator() == clang::OO_PlusPlus ||
            OpCall->getOperator() == clang::OO_MinusMinus) {
            if (OpCall->getNumArgs() > 0) {
                ++WriteDepth;
                TraverseStmt(OpCall->getArg(0));
                --WriteDepth;

                ++UpdateReadDepth;
                TraverseStmt(OpCall->getArg(0));
                --UpdateReadDepth;
            }

            for (unsigned argIdx = 1; argIdx < OpCall->getNumArgs(); ++argIdx) {
                TraverseStmt(OpCall->getArg(argIdx));
            }
            return true;
        }

        return clang::RecursiveASTVisitor<ParamAccessVisitor>::TraverseCXXOperatorCallExpr(OpCall);
    }

private:
    int WriteDepth = 0;
    int UpdateReadDepth = 0;
};

std::vector<AccessSummary> summarizeStmtAccess(
    const clang::Stmt* stmt,
    const std::unordered_map<const clang::ValueDecl*, int>& paramIndices,
    int paramCount) {
    std::vector<AccessSummary> summary(static_cast<std::size_t>(paramCount));
    if (!stmt || paramIndices.empty() || paramCount <= 0) {
        return summary;
    }

    ParamAccessVisitor visitor(paramIndices, paramCount);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    for (int paramIdx = 0; paramIdx < paramCount; ++paramIdx) {
        summary[static_cast<std::size_t>(paramIdx)].reads =
            visitor.Reads[paramIdx] || visitor.UpdateReads[paramIdx];
        summary[static_cast<std::size_t>(paramIdx)].writes =
            visitor.Writes[paramIdx];
    }
    return summary;
}

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

bool tensorNeedsBroadcast(DacppFile* dacppFile, const std::string& tensorName) {
    if (!dacppFile) {
        return false;
    }

    bool needsBcast = dacppFile->getMPIBroadcastOutputs();
    if (dacppFile->getMainBody()) {
        TensorUseVisitor visitor(tensorName, dacppFile->dacExprs);
        visitor.TraverseStmt(const_cast<clang::Stmt*>(dacppFile->getMainBody()));
        needsBcast = visitor.NeedsBcast;
    }
    return needsBcast;
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

std::vector<IOTYPE> inferEffectiveParamModes(Shell* shell, Calc* calc) {
    std::vector<IOTYPE> modes;
    modes.reserve(shell->getNumShellParams());
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        modes.push_back(shell->getShellParam(paramIdx)->getRw());
    }

    clang::FunctionDecl* calcLoc = calc->getCalcLoc();
    if (!calcLoc || !calcLoc->getBody()) {
        return modes;
    }

    std::unordered_map<const clang::ValueDecl*, int> paramIndices;
    for (int paramIdx = 0; paramIdx < calcLoc->getNumParams(); ++paramIdx) {
        paramIndices.emplace(calcLoc->getParamDecl(paramIdx), paramIdx);
    }

    ParamAccessVisitor visitor(paramIndices, calc->getNumParams());
    visitor.TraverseStmt(calcLoc->getBody());

    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        const bool reads = visitor.Reads[paramIdx];
        const bool updateReads = visitor.UpdateReads[paramIdx];
        const bool writes = visitor.Writes[paramIdx];

        if (reads && writes) {
            modes[paramIdx] = IOTYPE::READ_WRITE;
        } else if (writes && updateReads) {
            modes[paramIdx] = modes[paramIdx] == IOTYPE::WRITE
                                  ? IOTYPE::WRITE
                                  : IOTYPE::READ_WRITE;
        } else if (writes) {
            modes[paramIdx] = IOTYPE::WRITE;
        } else if (reads || updateReads) {
            modes[paramIdx] = IOTYPE::READ;
        }
    }

    return modes;
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

std::string toViewType(ShellParam* shellParam, Param* calcParam, IOTYPE mode) {
    const std::string baseType = calcParam->getBasicType();
    const bool isReadOnly = mode == IOTYPE::READ;
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

std::unordered_map<std::string, SplitBindMeta>
collectSplitBindMeta(Shell* shell) {
    std::unordered_map<std::string, SplitBindMeta> splitMeta;
    std::unordered_map<int, int> componentBindId;

    std::vector<BINDINFO> bindInfo;
    shell->GetBindInfo(&bindInfo);

    for (const auto& info : bindInfo) {
        Split* split = shell->search_symbol(info.v);
        if (!split || split->getId() == "void") {
            continue;
        }

        auto [it, inserted] =
            componentBindId.emplace(info.icls, static_cast<int>(componentBindId.size()));
        const std::string offset = info.offset.empty() ? "0" : info.offset;
        splitMeta[split->getId()] = SplitBindMeta{it->second, offset};
    }

    std::unordered_map<std::string, int> fallbackBindId;
    const int fallbackBase = static_cast<int>(componentBindId.size());

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            if (splitMeta.find(split->getId()) != splitMeta.end()) {
                continue;
            }

            auto [it, inserted] =
                fallbackBindId.emplace(split->getId(), fallbackBase + static_cast<int>(fallbackBindId.size()));
            splitMeta[split->getId()] = SplitBindMeta{it->second, "0"};
        }
    }

    return splitMeta;
}

std::string buildPatternInitCode(Shell* shell,
                                 Calc* calc,
                                 const std::unordered_map<std::string, SplitBindMeta>& splitMeta,
                                 const std::vector<IOTYPE>& paramModes) {
    std::string code;
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = "pattern_" + name;
        const IOTYPE mode = paramModes[paramIdx];

        code += "    dacpp::mpi::AccessPattern " + patternName + ";\n";
        code += "    " + patternName + ".param_id = " + std::to_string(paramIdx) + ";\n";
        code += "    " + patternName + ".name = \"" + name + "\";\n";
        code += "    " + patternName + ".mode = " + toPlannerMode(mode) + ";\n";
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
    const auto paramModes = inferEffectiveParamModes(shell, calc);
    std::string code = "inline void " + calc->getName() + "_mpi_local(";
    for (int paramIdx = 0; paramIdx < calc->getNumParams(); ++paramIdx) {
        code += toViewType(shell->getShellParam(paramIdx), calc->getParam(paramIdx), paramModes[paramIdx]) + " " +
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

std::string buildWrapperCode(DacppFile* dacppFile, Shell* shell, Calc* calc) {
    const std::string wrapperName = shell->getName() + "_" + calc->getName();
    const auto paramModes = inferEffectiveParamModes(shell, calc);

    std::string signature;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        // Ensure pass-by-reference so that MPI gather writeback
        // (tensorName.array2Tensor(...)) modifies the caller's tensor,
        // not a local copy.  This is essential for multi-stage pipelines
        // where stage N's output feeds stage N+1.
        if (paramType.size() > 0 && paramType.back() != '&' && paramType.back() != '*') {
            paramType += "&";
        }
        signature += paramType + " " + param->getName();
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
    code += buildPatternInitCode(shell, calc, splitMeta, paramModes);
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
        const IOTYPE mode = paramModes[paramIdx];
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
                buildPackBuilderExpr(mode, patternName) + ";\n";
        code += "    auto " + slotsName + " = dacpp::mpi::build_item_slots(item_range, " +
                patternName + ", " + packName + ");\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> " + localName + "(" +
                packName + ".globals.size());\n";

        if (mode != IOTYPE::WRITE) {
            code += "    int recv_count_" + calcName + " = 0;\n";
            code += "    std::vector<int> sendcounts_" + calcName + ";\n";
            code += "    std::vector<int> displs_" + calcName + ";\n";
            code += "    std::vector<" + calcParam->getBasicType() + "> sendbuf_" + calcName + ";\n";
            code += "    if (mpi_rank == 0) {\n";
            code += "        sendcounts_" + calcName + ".resize(mpi_size);\n";
            code += "        displs_" + calcName + ".resize(mpi_size);\n";
            code += "        int current_displ = 0;\n";
            code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName + ";\n";
            code += "        " + tensorName + ".tensor2Array(" + globalName + ");\n";
            code += "        for (int r = 0; r < mpi_size; ++r) {\n";
            code += "            auto r_range = dacpp::mpi::get_rank_item_range(total_items, r, mpi_size);\n";
            code += "            auto r_pack = " + buildRemotePackBuilderExpr(mode, "r_range", patternName) + ";\n";
            code += "            auto r_values = dacpp::mpi::pack_values_by_globals(" + globalName + ", r_pack.globals);\n";
            code += "            int r_count = static_cast<int>(r_values.size());\n";
            code += "            sendcounts_" + calcName + "[r] = r_count;\n";
            code += "            displs_" + calcName + "[r] = current_displ;\n";
            code += "            current_displ += r_count;\n";
            code += "            sendbuf_" + calcName + ".insert(sendbuf_" + calcName + ".end(), r_values.begin(), r_values.end());\n";
            code += "        }\n";
            code += "    }\n";
            code += "    MPI_Scatter(mpi_rank == 0 ? sendcounts_" + calcName + ".data() : nullptr, 1, MPI_INT, &recv_count_" + calcName + ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
            code += "    " + localName + ".resize(recv_count_" + calcName + ");\n";
            code += "    std::vector<int> sendcounts_bytes_" + calcName + " = sendcounts_" + calcName + ";\n";
            code += "    std::vector<int> displs_bytes_" + calcName + " = displs_" + calcName + ";\n";
            if (usesByteTransport(calcParam->getBasicType())) {
                code += "    if (mpi_rank == 0) {\n";
                code += "        for (int r = 0; r < mpi_size; ++r) {\n";
                code += "            sendcounts_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
                code += "            displs_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
                code += "        }\n";
                code += "    }\n";
            }
            std::string payloadRecvCount = mpiPayloadCountExpr("recv_count_" + calcName, calcParam->getBasicType());
            code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? sendcounts_bytes_" + calcName + ".data() : nullptr, mpi_rank == 0 ? displs_bytes_" + calcName + ".data() : nullptr, " + mpiType + ", " + localName + ".data(), " + payloadRecvCount + ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
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
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto acc_" + name + " = buffer_" + name + ".get_access<" +
                toAccessorMode(paramModes[paramIdx]) + ">(h);\n";
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
            code += "                    " + toViewType(shellParam, calcParam, paramModes[paramIdx]) + " view_" + name +
                    "{data_" + name + ", slots_" + name + ", item_linear * " + name +
                    "_partition_size};\n";
        } else {
            code += "                    " + toViewType(shellParam, calcParam, paramModes[paramIdx]) + " view_" + name +
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
        if (paramModes[paramIdx] == IOTYPE::READ) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string packName = "pack_" + calcName;
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_out_" + calcName;
        const std::string wbName = "writeback_" + calcName;
        const std::string mpiType = mpiDatatypeFor(calcParam->getBasicType());
        const std::string payloadRecvCount =
            mpiPayloadCountExpr("recv_count", calcParam->getBasicType());
        const std::string payloadSendCount =
            mpiPayloadCountExpr("send_count", calcParam->getBasicType());
        bool needsBcast = tensorNeedsBroadcast(dacppFile, tensorName);

        code += "    auto " + wbName + " = dacpp::mpi::build_writeback_values(" + localName + ", " + packName + ");\n";
        code += "    const auto& writeback_globals_" + calcName + " = " + packName + ".writeback_globals.empty() ? " + packName + ".globals : " + packName + ".writeback_globals;\n";
        if (needsBcast) {
            code += "    std::vector<" + calcParam->getBasicType() + "> synced_" + calcName + ";\n";
        }
        code += "    int send_count_" + calcName + " = static_cast<int>(writeback_globals_" + calcName + ".size());\n";
        code += "    std::vector<int> recvcounts_" + calcName + ";\n";
        code += "    std::vector<int> recvdispls_" + calcName + ";\n";
        code += "    std::vector<int64_t> global_recv_globals_" + calcName + ";\n";
        code += "    std::vector<" + calcParam->getBasicType() + "> global_recv_values_" + calcName + ";\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        recvcounts_" + calcName + ".resize(mpi_size);\n";
        code += "        recvdispls_" + calcName + ".resize(mpi_size);\n";
        code += "    }\n";
        code += "    MPI_Gather(&send_count_" + calcName + ", 1, MPI_INT, mpi_rank == 0 ? recvcounts_" + calcName + ".data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        int current_displ = 0;\n";
        code += "        for (int r = 0; r < mpi_size; ++r) {\n";
        code += "            recvdispls_" + calcName + "[r] = current_displ;\n";
        code += "            current_displ += recvcounts_" + calcName + "[r];\n";
        code += "        }\n";
        code += "        global_recv_globals_" + calcName + ".resize(current_displ);\n";
        code += "        global_recv_values_" + calcName + ".resize(current_displ);\n";
        code += "    }\n";
        code += "    MPI_Gatherv(const_cast<int64_t*>(writeback_globals_" + calcName + ".data()), send_count_" + calcName + ", MPI_LONG_LONG, mpi_rank == 0 ? global_recv_globals_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvcounts_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvdispls_" + calcName + ".data() : nullptr, MPI_LONG_LONG, 0, MPI_COMM_WORLD);\n";
        code += "    std::vector<int> recvcounts_bytes_" + calcName + " = recvcounts_" + calcName + ";\n";
        code += "    std::vector<int> recvdispls_bytes_" + calcName + " = recvdispls_" + calcName + ";\n";
        if (usesByteTransport(calcParam->getBasicType())) {
            code += "    if (mpi_rank == 0) {\n";
            code += "        for (int r = 0; r < mpi_size; ++r) {\n";
            code += "            recvcounts_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
            code += "            recvdispls_bytes_" + calcName + "[r] *= sizeof(" + calcParam->getBasicType() + ");\n";
            code += "        }\n";
            code += "    }\n";
        }
        std::string payloadSendCount2 = mpiPayloadCountExpr("send_count_" + calcName, calcParam->getBasicType());
        code += "    MPI_Gatherv(" + wbName + ".data(), " + payloadSendCount2 + ", " + mpiType + ", mpi_rank == 0 ? global_recv_values_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvcounts_bytes_" + calcName + ".data() : nullptr, mpi_rank == 0 ? recvdispls_bytes_" + calcName + ".data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        code += "    if (mpi_rank == 0) {\n";
        code += "        std::vector<" + calcParam->getBasicType() + "> " + globalName + ";\n";
        code += "        " + tensorName + ".tensor2Array(" + globalName + ");\n";
        code += "        dacpp::mpi::apply_writeback_by_globals(global_recv_values_" + calcName + ", global_recv_globals_" + calcName + ", " + globalName + ");\n";
        code += "        " + tensorName + ".array2Tensor(" + globalName + ");\n";
        if (needsBcast) {
            code += "        synced_" + calcName + " = " + globalName + ";\n";
        }
        code += "    }\n";
        if (needsBcast) {
            const std::string payloadSyncedCount =
                mpiPayloadCountExpr("synced_count_" + calcName, calcParam->getBasicType());
            code += "    int synced_count_" + calcName + " = 0;\n";
            code += "    if (mpi_rank == 0) {\n";
            code += "        synced_count_" + calcName + " = static_cast<int>(synced_" + calcName + ".size());\n";
            code += "    }\n";
            code += "    MPI_Bcast(&synced_count_" + calcName + ", 1, MPI_INT, 0, MPI_COMM_WORLD);\n";
            code += "    if (mpi_rank != 0) {\n";
            code += "        synced_" + calcName + ".resize(synced_count_" + calcName + ");\n";
            code += "    }\n";
            code += "    if (synced_count_" + calcName + " > 0) {\n";
            code += "        MPI_Bcast(synced_" + calcName + ".data(), " + payloadSyncedCount + ", " + mpiType +
                    ", 0, MPI_COMM_WORLD);\n";
            code += "    }\n";
            code += "    if (mpi_rank != 0) {\n";
            code += "        " + tensorName + ".array2Tensor(synced_" + calcName + ");\n";
            code += "    }\n";
        }
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

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
