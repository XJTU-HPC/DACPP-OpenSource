#include <set>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#include "Rewriter_MPI_Common.h"
#include "Rewriter_MPI_OperatorResident.h"
#include "Rewriter_MPI_Plan.h"
#include "Rewriter_MPI_Stencil_Common.h"
#include "mpi/shared/LoopLoweredRewrite.h"

namespace dacppTranslator {

namespace {

struct FoulaOwnerLoopSpecialization {
    bool enabled = false;
    int exprIndex = -1;
    const clang::ForStmt* outerLoop = nullptr;
    std::string functionName;
    std::string ownerName;
    std::string scalarExpr;
    std::string elemType;
    std::string mpiType;
    int readerParamIndex = -1;
    int writerParamIndex = -1;
    int scalarParamIndex = -1;
};

std::string getStmtSourceText(const clang::Stmt* stmt, DacppFile* dacppFile) {
    if (!stmt || !dacppFile || !dacppFile->getContext()) {
        return "";
    }
    const auto& context = *dacppFile->getContext();
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(stmt->getSourceRange()),
               context.getSourceManager(),
               context.getLangOpts())
        .str();
}

std::string regexEscape(const std::string& text) {
    static const std::regex special(R"([.^$|()\\[\]{}*+?])");
    return std::regex_replace(text, special, R"(\$&)");
}

const clang::CallExpr* getShellCallExpr(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }
    clang::Expr* shellExpr =
        Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS() : dacExpr->getRHS();
    return dacppTranslator::getNode<clang::CallExpr>(shellExpr);
}

std::string getDeclRefName(const clang::Expr* expr) {
    if (!expr) {
        return "";
    }
    const auto* declRef = dacppTranslator::getNode<clang::DeclRefExpr>(
        const_cast<clang::Expr*>(expr));
    if (!declRef || !declRef->getDecl()) {
        return "";
    }
    return declRef->getDecl()->getNameAsString();
}

std::string wrapperNameForDacExpr(const clang::BinaryOperator* dacExpr) {
    const clang::CallExpr* shellCall = getShellCallExpr(dacExpr);
    if (!shellCall || !shellCall->getDirectCallee()) {
        return "";
    }
    const clang::Expr* calcExpr =
        Expression::shellLHS_p(dacExpr) ? dacExpr->getRHS() : dacExpr->getLHS();
    const std::string calcName = getDeclRefName(calcExpr);
    if (calcName.empty()) {
        return "";
    }
    return shellCall->getDirectCallee()->getNameAsString() + "_" + calcName;
}

const clang::ForStmt* outerForLoopForExpr(DacppFile* dacppFile,
                                          const clang::BinaryOperator* dacExpr) {
    if (!dacppFile || !dacppFile->getContext() || !dacExpr) {
        return nullptr;
    }
    clang::DynTypedNode current = clang::DynTypedNode::create(*dacExpr);
    const clang::ForStmt* outerFor = nullptr;
    while (true) {
        auto parents = dacppFile->getContext()->getParents(current);
        if (parents.empty()) {
            break;
        }
        const auto& parent = parents[0];
        if (const auto* forStmt = parent.get<clang::ForStmt>()) {
            outerFor = forStmt;
        }
        if (parent.get<clang::FunctionDecl>()) {
            break;
        }
        current = parent;
    }
    return outerFor;
}

std::string loopVarName(const clang::ForStmt* forStmt) {
    const auto* declStmt =
        forStmt ? llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit())
                : nullptr;
    if (!declStmt || !declStmt->isSingleDecl()) {
        return "";
    }
    const auto* varDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    return varDecl ? varDecl->getNameAsString() : "";
}

const mpi_rewriter::ParamAccessPlan* stencil1DReaderParam(
    const mpi_rewriter::ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == mpi_rewriter::ParamAccessKind::StencilWindow &&
            param.reads && !param.writes) {
            return &param;
        }
    }
    return nullptr;
}

const mpi_rewriter::ParamAccessPlan* stencil1DWriterParam(
    const mpi_rewriter::ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == mpi_rewriter::ParamAccessKind::OutputDirect &&
            param.writes && !param.reads) {
            return &param;
        }
    }
    return nullptr;
}

const mpi_rewriter::ParamAccessPlan* scalarReaderParam(
    const mpi_rewriter::ShellPartitionPlan& plan) {
    for (const auto& param : plan.params) {
        if (param.access == mpi_rewriter::ParamAccessKind::ReplicatedScalar &&
            param.reads && !param.writes) {
            return &param;
        }
    }
    return nullptr;
}

bool firstRegexMatch(const std::string& text,
                     const std::regex& pattern,
                     std::smatch& match) {
    return std::regex_search(text, match, pattern);
}

FoulaOwnerLoopSpecialization detectFoulaOwnerLoopSpecialization(
    DacppFile* dacppFile,
    const mpi_rewriter::ShellPartitionPlan& exprPlan) {
    FoulaOwnerLoopSpecialization spec;
    if (!dacppFile || !exprPlan.exprNode.dacExpr || !exprPlan.exprNode.calc ||
        exprPlan.signature.layout != mpi_rewriter::LocalLayoutKind::StencilWindow1D) {
        return spec;
    }
    if (exprPlan.exprNode.calc->getNumParams() != 3 ||
        exprPlan.params.size() != 3) {
        return spec;
    }
    const auto* reader = stencil1DReaderParam(exprPlan);
    const auto* writer = stencil1DWriterParam(exprPlan);
    const auto* scalar = scalarReaderParam(exprPlan);
    if (!reader || !writer || !scalar) {
        return spec;
    }
    const std::string elemType =
        exprPlan.exprNode.calc->getParam(reader->paramIndex)->getBasicType();
    const std::string writerType =
        exprPlan.exprNode.calc->getParam(writer->paramIndex)->getBasicType();
    const std::string scalarType =
        exprPlan.exprNode.calc->getParam(scalar->paramIndex)->getBasicType();
    if (elemType != writerType || elemType != scalarType ||
        mpi_rewriter::usesByteTransport(elemType)) {
        return spec;
    }

    const clang::ForStmt* outerLoop =
        outerForLoopForExpr(dacppFile, exprPlan.exprNode.dacExpr);
    const std::string loopVar = loopVarName(outerLoop);
    const std::string loopText = getStmtSourceText(outerLoop, dacppFile);
    if (!outerLoop || loopVar.empty() || loopText.empty()) {
        return spec;
    }
    const auto* loopBody =
        llvm::dyn_cast_or_null<clang::CompoundStmt>(outerLoop->getBody());
    if (!loopBody) {
        return spec;
    }
    int topLevelStmtCount = 0;
    for (const clang::Stmt* bodyStmt : loopBody->body()) {
        (void)bodyStmt;
        ++topLevelStmtCount;
    }
    if (topLevelStmtCount != 7) {
        return spec;
    }

    const std::string readerName = regexEscape(reader->actualTensorName);
    const std::string writerName = regexEscape(writer->actualTensorName);
    const std::string scalarName = regexEscape(scalar->actualTensorName);
    const std::string loopVarPattern = regexEscape(loopVar);

    std::smatch readerMatch;
    const std::regex readerPattern(
        "\\b" + readerName +
        "\\s*=\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*\\{\\s*\\}\\s*\\]\\s*\\[\\s*" +
        loopVarPattern + "\\s*\\]");
    if (!firstRegexMatch(loopText, readerPattern, readerMatch) ||
        readerMatch.size() < 2) {
        return spec;
    }
    const std::string ownerName = readerMatch[1].str();

    std::smatch writerMatch;
    const std::regex writerPattern(
        "\\b" + writerName + "\\s*=\\s*" + regexEscape(ownerName) +
        "\\s*\\[\\s*\\{\\s*1\\s*,[^\\}]+\\}\\s*\\]\\s*\\[\\s*" +
        loopVarPattern + "\\s*\\+\\s*1\\s*\\]");
    if (!firstRegexMatch(loopText, writerPattern, writerMatch)) {
        return spec;
    }

    const std::regex postWritePattern(
        regexEscape(ownerName) +
        "\\s*\\[[^\\]]+\\]\\s*\\[\\s*" + loopVarPattern +
        "\\s*\\+\\s*1\\s*\\]\\s*=\\s*" + writerName +
        "\\s*\\[[^\\]]+\\]");
    if (!std::regex_search(loopText, postWritePattern)) {
        return spec;
    }

    std::smatch scalarMatch;
    const std::regex scalarPattern(
        "\\b([A-Za-z_][A-Za-z0-9_]*)\\s*\\.\\s*push_back\\s*\\(\\s*([^() ;]+)\\s*\\)\\s*;[\\s\\S]*\\bdacpp::Vector\\s*<[^>]+>\\s+" +
        scalarName + "\\s*\\(\\s*\\1\\s*\\)");
    if (!firstRegexMatch(loopText, scalarPattern, scalarMatch) ||
        scalarMatch.size() < 3) {
        return spec;
    }

    spec.enabled = true;
    spec.exprIndex = exprPlan.exprIndex;
    spec.outerLoop = outerLoop;
    spec.functionName =
        mpi_rewriter::operatorResidentWrapperName(
            exprPlan.exprNode.shell, exprPlan.exprNode.calc,
            exprPlan.exprIndex) +
        "_owner_loop";
    spec.ownerName = ownerName;
    spec.scalarExpr = scalarMatch[2].str();
    spec.elemType = elemType;
    spec.mpiType = mpi_rewriter::mpiDatatypeFor(elemType);
    spec.readerParamIndex = reader->paramIndex;
    spec.writerParamIndex = writer->paramIndex;
    spec.scalarParamIndex = scalar->paramIndex;
    return spec;
}

std::string buildFoulaOwnerLoopSpecializationCode(
    const FoulaOwnerLoopSpecialization& spec,
    const mpi_rewriter::ShellPartitionPlan& exprPlan) {
    if (!spec.enabled || !exprPlan.exprNode.calc) {
        return "";
    }
    const std::string& type = spec.elemType;
    const std::string& mpiType = spec.mpiType;
    const std::string calcName = exprPlan.exprNode.calc->getName();
    const auto* reader = stencil1DReaderParam(exprPlan);
    const auto* writer = stencil1DWriterParam(exprPlan);
    const auto* scalar = scalarReaderParam(exprPlan);
    if (!reader || !writer || !scalar) {
        return "";
    }
    std::string code;
    code += "void " + spec.functionName + "(dacpp::Matrix<" + type +
            ">& __or_owner, " + type + " __or_scalar_value) {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);\n";
    code += "    dacpp::mpi::SegmentedProfile dacpp_profile;\n";
    code += "    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    sycl::queue q(sycl::default_selector_v);\n";
    code += "    const int64_t __or_rows = __or_owner.getShape(0);\n";
    code += "    const int64_t __or_cols = __or_owner.getShape(1);\n";
    code += "    if (__or_rows < 3 || __or_cols < 2) {\n";
    code += "        return;\n";
    code += "    }\n";
    code += "    const int64_t __or_output_size = __or_rows - 2;\n";
    code += "    const int64_t __or_steps = __or_cols - 1;\n";
    code += "    const int __or_window_size = 3;\n";
    code += "    const auto __or_range = dacpp::mpi::operator_resident::rank_range_1d(__or_output_size, mpi_rank, mpi_size);\n";
    code += "    const int64_t __or_local_item_count = __or_range.count;\n";
    code += "    const auto __or_halo_layout = dacpp::mpi::operator_resident::resident_halo_1d_layout(__or_output_size, mpi_rank, mpi_size, __or_window_size);\n";
    code += "    std::vector<" + type + "> __or_initial_col;\n";
    code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);\n";
    code += "    auto dacpp_profile_scatter_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        __or_initial_col.resize(static_cast<std::size_t>(__or_rows));\n";
    code += "        for (int64_t __or_row = 0; __or_row < __or_rows; ++__or_row) {\n";
    code += "            __or_initial_col[static_cast<std::size_t>(__or_row)] = __or_owner.getElement({static_cast<int>(__or_row), 0});\n";
    code += "        }\n";
    code += "    }\n";
    code += "    std::vector<" + type + "> __or_curr;\n";
    code += "    dacpp::mpi::operator_resident::scatter_window_1d(__or_initial_col, __or_curr, __or_output_size, __or_rows, __or_window_size, __or_halo_layout, mpi_rank, mpi_size, " +
            mpiType + ");\n";
    code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start);\n";
    code += "    std::vector<" + type + "> __or_next(__or_curr.size(), " +
            type + "{});\n";
    code += "    std::vector<" + type + "> __or_scalar_vec(1, __or_scalar_value);\n";
    code += "    std::vector<" + type + "> __or_local_history(static_cast<std::size_t>(__or_local_item_count * __or_cols), " +
            type + "{});\n";
    code += "    for (int64_t __or_i = 0; __or_i < __or_local_item_count; ++__or_i) {\n";
    code += "        __or_local_history[static_cast<std::size_t>(__or_i * __or_cols)] = __or_curr[static_cast<std::size_t>(__or_i + 1)];\n";
    code += "    }\n";
    code += "    const int __or_last_owner_rank = dacpp::mpi::operator_resident::nearest_nonempty_rank_1d(__or_output_size, mpi_size, mpi_size, -1);\n";
    code += "    for (int64_t __or_step = 0; __or_step < __or_steps; ++__or_step) {\n";
    code += "        __or_scalar_vec[0] = __or_scalar_value;\n";
    code += "        auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();\n";
    code += "        if (__or_local_item_count > 0) {\n";
    code += "            sycl::buffer<" + type + ", 1> __or_reader_buf(__or_curr.data(), sycl::range<1>(__or_curr.size()));\n";
    code += "            sycl::buffer<" + type + ", 1> __or_writer_buf(__or_next.data(), sycl::range<1>(__or_next.size()));\n";
    code += "            sycl::buffer<" + type + ", 1> __or_scalar_buf(__or_scalar_vec.data(), sycl::range<1>(__or_scalar_vec.size()));\n";
    code += "            q.submit([&](sycl::handler& h) {\n";
    code += "                auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);\n";
    code += "                auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);\n";
    code += "                auto __or_scalar_acc = __or_scalar_buf.get_access<sycl::access::mode::read>(h);\n";
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    code += "                    auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    code += "                    auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    code += "                    auto* __or_scalar_data = __or_scalar_acc.template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    for (const auto& param : exprPlan.params) {
        const std::string paramType =
            exprPlan.exprNode.calc->getParam(param.paramIndex)->getBasicType();
        if (param.paramIndex == reader->paramIndex) {
            code += "                    dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_reader_data, item_linear};\n";
            continue;
        }
        if (param.paramIndex == writer->paramIndex) {
            code += "                    dacpp::mpi::ContiguousView1D<" +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_writer_data, item_linear + 1};\n";
            continue;
        }
        if (param.paramIndex == scalar->paramIndex) {
            code += "                    dacpp::mpi::ContiguousView1D<const " +
                    paramType + "> view_" + param.calcParamName +
                    "{__or_scalar_data, 0};\n";
            continue;
        }
        return "";
    }
    code += "                    " + calcName + "_mpi_local(view_" +
            exprPlan.exprNode.calc->getParam(0)->getName() + ", view_" +
            exprPlan.exprNode.calc->getParam(1)->getName() + ", view_" +
            exprPlan.exprNode.calc->getParam(2)->getName() + ");\n";
    code += "                });\n";
    code += "            });\n";
    code += "            q.wait();\n";
    code += "        }\n";
    code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);\n";
    code += "        " + type + " __or_left_boundary{};\n";
    code += "        " + type + " __or_right_boundary{};\n";
    code += "        if (mpi_rank == 0) {\n";
    code += "            __or_left_boundary = __or_owner.getElement({0, static_cast<int>(__or_step + 1)});\n";
    code += "            __or_right_boundary = __or_owner.getElement({static_cast<int>(__or_rows - 1), static_cast<int>(__or_step + 1)});\n";
    code += "        }\n";
    code += "        auto dacpp_profile_bcast_start = dacpp::mpi::profileSegmentStart();\n";
    code += "        MPI_Bcast(&__or_left_boundary, 1, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "        MPI_Bcast(&__or_right_boundary, 1, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start);\n";
    code += "        if (mpi_rank == 0 && !__or_next.empty()) {\n";
    code += "            __or_next[0] = __or_left_boundary;\n";
    code += "        }\n";
    code += "        if (mpi_rank == __or_last_owner_rank && __or_local_item_count > 0 && static_cast<std::size_t>(__or_local_item_count + 1) < __or_next.size()) {\n";
    code += "            __or_next[static_cast<std::size_t>(__or_local_item_count + 1)] = __or_right_boundary;\n";
    code += "        }\n";
    code += "        auto dacpp_profile_halo_start = dacpp::mpi::profileSegmentStart();\n";
    code += "        dacpp::mpi::operator_resident::exchange_halo_1d_inplace(__or_next, __or_halo_layout, __or_output_size, __or_window_size, 1, mpi_rank, mpi_size, " +
            mpiType + ");\n";
    code += "        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Halo, dacpp_profile_halo_start);\n";
    code += "        __or_curr.swap(__or_next);\n";
    code += "        for (int64_t __or_i = 0; __or_i < __or_local_item_count; ++__or_i) {\n";
    code += "            __or_local_history[static_cast<std::size_t>(__or_i * __or_cols + (__or_step + 1))] = __or_curr[static_cast<std::size_t>(__or_i + 1)];\n";
    code += "        }\n";
    code += "    }\n";
    code += "    std::vector<int> __or_hist_counts(mpi_size, 0);\n";
    code += "    std::vector<int> __or_hist_displs(mpi_size, 0);\n";
    code += "    for (int __or_r = 0; __or_r < mpi_size; ++__or_r) {\n";
    code += "        const auto __or_r_range = dacpp::mpi::operator_resident::rank_range_1d(__or_output_size, __or_r, mpi_size);\n";
    code += "        __or_hist_counts[__or_r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_r_range.count * __or_cols, \"[DACPP][MPI][OR][FOuLa] history gather count exceeds MPI int range\");\n";
    code += "        __or_hist_displs[__or_r] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_r_range.begin * __or_cols, \"[DACPP][MPI][OR][FOuLa] history gather displacement exceeds MPI int range\");\n";
    code += "    }\n";
    code += "    std::vector<" + type + "> __or_global_history;\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        __or_global_history.resize(static_cast<std::size_t>(__or_output_size * __or_cols));\n";
    code += "    }\n";
    code += "    auto dacpp_profile_gather_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    MPI_Gatherv(__or_local_history.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(__or_local_item_count * __or_cols, \"[DACPP][MPI][OR][FOuLa] local history gather count exceeds MPI int range\"), " +
            mpiType + ", mpi_rank == 0 ? __or_global_history.data() : nullptr, mpi_rank == 0 ? __or_hist_counts.data() : nullptr, mpi_rank == 0 ? __or_hist_displs.data() : nullptr, " +
            mpiType + ", 0, MPI_COMM_WORLD);\n";
    code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start);\n";
    code += "    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();\n";
    code += "    if (mpi_rank == 0) {\n";
    code += "        for (int64_t __or_out = 0; __or_out < __or_output_size; ++__or_out) {\n";
    code += "            for (int64_t __or_col = 0; __or_col < __or_cols; ++__or_col) {\n";
    code += "                __or_owner.reviseValue(__or_global_history[static_cast<std::size_t>(__or_out * __or_cols + __or_col)], {static_cast<int>(__or_out + 1), static_cast<int>(__or_col)});\n";
    code += "            }\n";
    code += "        }\n";
    code += "    }\n";
    code += "    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);\n";
    code += "    dacpp::mpi::reportSegmentedProfile(\"" + spec.functionName +
            "\", dacpp_profile, MPI_COMM_WORLD);\n";
    code += "}\n";
    return code;
}

const mpi_rewriter::ShellPartitionPlan* findOperatorResidentExprPlan(
    const mpi_rewriter::MpiLoweringPlan& plan,
    int exprIdx) {
    if (exprIdx < 0 ||
        exprIdx >= static_cast<int>(plan.exprResults.size())) {
        return nullptr;
    }
    const int chainId = plan.exprResults[exprIdx].operatorResidentChainId;
    if (chainId < 0 || chainId >= static_cast<int>(plan.residentChains.size())) {
        return nullptr;
    }
    for (const auto& candidate : plan.residentChains[chainId].exprPlans) {
        if (candidate.exprIndex == exprIdx) {
            return &candidate;
        }
    }
    return nullptr;
}

void removeOperatorResidentLoweredPostStmts(
    DacppFile* dacppFile,
    clang::Rewriter* rewriter,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    if (!dacppFile || !rewriter || !shell || !calc || !dacExpr) {
        return;
    }
    std::set<const clang::Stmt*> stmtsToRemove;
    const auto distributedRegions =
        mpi_rewriter::collectDistributedFollowupRegions(dacppFile, shell, calc,
                                                        dacExpr);
    for (const auto& region : distributedRegions) {
        stmtsToRemove.insert(region.stmt);
    }

    const auto sitePlan =
        mpi_rewriter::analyzeDistributedStencilSite(dacppFile, shell, calc,
                                                    dacExpr);
    if (sitePlan.supported && !sitePlan.hasRootBridge) {
        for (const clang::Stmt* stmt : sitePlan.boundaryLocalStmts) {
            stmtsToRemove.insert(stmt);
        }
    }

    for (const clang::Stmt* stmt : stmtsToRemove) {
        if (stmt) {
            rewriter->RemoveText(stmt->getSourceRange());
        }
    }
}

void removeContractStmts(clang::Rewriter* rewriter,
                         const mpi_rewriter::LoopLoweringContract& contract) {
    if (!rewriter) {
        return;
    }
    for (const clang::Stmt* stmt :
         mpi_rewriter::loweringContractRemoveStmtSet(contract)) {
        if (stmt) {
            rewriter->RemoveText(stmt->getSourceRange());
        }
    }
}

bool isLoopLoweredOperatorResidentExpr(
    const mpi_rewriter::MpiLoweringPlan& plan,
    int exprIdx,
    const mpi_rewriter::ShellPartitionPlan** exprPlanOut) {
    const auto* exprPlan = findOperatorResidentExprPlan(plan, exprIdx);
    if (!exprPlan ||
        !mpi_rewriter::isLoopLoweredOperatorResidentPlan(*exprPlan)) {
        return false;
    }
    if (exprPlanOut) {
        *exprPlanOut = exprPlan;
    }
    return true;
}

void rewriteLoopLoweredOperatorResidentSite(
    DacppFile* dacppFile,
    clang::Rewriter* rewriter,
    const mpi_rewriter::ShellPartitionPlan& exprPlan) {
    if (!dacppFile || !rewriter || !exprPlan.exprNode.dacExpr ||
        !exprPlan.exprNode.shell ||
        !exprPlan.exprNode.calc) {
        return;
    }

    Shell* shell = exprPlan.exprNode.shell;
    Calc* calc = exprPlan.exprNode.calc;
    const int exprIdx = exprPlan.exprIndex;
    const clang::BinaryOperator* dacExpr = exprPlan.exprNode.dacExpr;
    const clang::Stmt* outerLoop =
        exprPlan.orLoopLower.outerLoop ? exprPlan.orLoopLower.outerLoop
                                       : exprPlan.loopLowerOuterLoop;
    if (!outerLoop) {
        return;
    }

    // FixedBlockPhaseExchangeFollower: the follower's DAC expression and
    // surrounding helper statements were already removed when plan A was
    // rewritten. Skip to avoid double rewriting.
    if (exprPlan.orLoopLower.kind ==
        mpi_rewriter::OrLoopLowerKind::FixedBlockPhaseExchangeFollower) {
        return;
    }

    const std::string argText =
        mpi_rewriter::joinShellCallArgs(dacExpr, dacppFile);
    const std::string ctxVar =
        "__dacpp_mpi_or_ctx_" + std::to_string(exprIdx);

    mpi_rewriter::LoopLoweredRewriteSpec rewriteSpec;
    rewriteSpec.outerLoop = outerLoop;
    rewriteSpec.dacExpr = dacExpr;
    rewriteSpec.contextTypeName =
        mpi_rewriter::operatorResidentContextTypeName(shell, calc, exprIdx);
    rewriteSpec.contextVariableName = ctxVar;
    rewriteSpec.initFunctionName =
        mpi_rewriter::operatorResidentInitFunctionName(shell, calc, exprIdx);
    rewriteSpec.runFunctionName =
        mpi_rewriter::operatorResidentRunFunctionName(shell, calc, exprIdx);
    rewriteSpec.materializeFunctionName =
        mpi_rewriter::operatorResidentMaterializeFunctionName(shell, calc,
                                                              exprIdx);
    rewriteSpec.argumentText = argText;
    mpi_rewriter::rewriteLoopLoweredDacExpr(rewriter, rewriteSpec);

    if (exprPlan.orLoopLower.kind ==
            mpi_rewriter::OrLoopLowerKind::StencilFullSync ||
        exprPlan.orLoopLower.kind ==
            mpi_rewriter::OrLoopLowerKind::StencilResidentHalo) {
        if (exprPlan.orLoopLower.contractRemovalSetMatchesLegacy) {
            removeContractStmts(rewriter, exprPlan.orLoopLower.contract);
        } else {
            llvm::outs()
                << "[DACPP][MPI][OR][P4.6][ContractRemoval] expr="
                << exprIdx
                << " fallback=legacy reason="
                << (exprPlan.orLoopLower.contractRemovalSetReason.empty()
                        ? "contract removal set mismatch"
                        : exprPlan.orLoopLower.contractRemovalSetReason)
                << "\n";
            removeOperatorResidentLoweredPostStmts(dacppFile, rewriter, shell,
                                                   calc, dacExpr);
        }
    }

    if (exprPlan.orLoopLower.kind ==
        mpi_rewriter::OrLoopLowerKind::FixedBlockPhaseExchange) {
        removeContractStmts(rewriter, exprPlan.orLoopLower.contract);
    }
}

}  // namespace

void Rewriter::rewriteMPI() {
    auto plan = mpi_rewriter::buildMpiLoweringPlan(dacppFile);

    std::string generated = mpi_rewriter::buildPrelude(dacppFile);
    std::set<std::string> generatedWrappers;
    std::set<std::string> generatedLocalCalcs;
    std::set<const clang::FunctionDecl*> removedDecls;
    std::unordered_map<const clang::BinaryOperator*, MpiStencilSite> siteByExpr;
    std::unordered_map<int, FoulaOwnerLoopSpecialization> foulaSpecs;
    std::set<const clang::Stmt*> rewrittenFoulaLoops;

    for (const auto& site : dacppFile->getMPIStencilSites()) {
        if (!site.dacExpr || !site.outerLoop) {
            continue;
        }
        siteByExpr.emplace(site.dacExpr, site);
    }

    for (const auto& chain : plan.residentChains) {
        if (!chain.supported) {
            continue;
        }
        for (const auto& exprPlan : chain.exprPlans) {
            FoulaOwnerLoopSpecialization spec =
                detectFoulaOwnerLoopSpecialization(dacppFile, exprPlan);
            if (spec.enabled) {
                foulaSpecs[exprPlan.exprIndex] = spec;
                llvm::outs() << "[DACPP][MPI][OR][FOuLa] expr="
                             << exprPlan.exprIndex
                             << " owner-loop=candidate owner="
                             << spec.ownerName << "\n";
            }
        }
    }

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        const bool hasPlanResult =
            exprIdx < static_cast<int>(plan.exprResults.size());
        const mpi_rewriter::MpiPlanKind planKind =
            hasPlanResult ? plan.exprResults[exprIdx].kind
                          : mpi_rewriter::MpiPlanKind::Unsupported;
        const bool isOperatorResident =
            planKind == mpi_rewriter::MpiPlanKind::OperatorResident;
        const bool isStencilPhaseC =
            planKind == mpi_rewriter::MpiPlanKind::StencilPhaseC;
        std::string wrapperName;
        if (isOperatorResident) {
            wrapperName =
                mpi_rewriter::operatorResidentWrapperName(shell, calc, exprIdx);
        } else if (isStencilPhaseC) {
            wrapperName =
                mpi_stencil_rewriter::wrapperName(shell, calc, exprIdx);
        } else {
            wrapperName = shell->getName() + "_" + calc->getName();
        }
        if (generatedWrappers.insert(wrapperName).second) {
            const std::string localCalcKey = calc->getName();
            if (generatedLocalCalcs.insert(localCalcKey).second) {
                generated += mpi_rewriter::buildLocalCalcCode(shell, calc);
                generated += "\n";
            }
            if (isOperatorResident) {
                const int chainId =
                    plan.exprResults[exprIdx].operatorResidentChainId;
                const auto& chain = plan.residentChains[chainId];
                const mpi_rewriter::ShellPartitionPlan* exprPlan = nullptr;
                for (const auto& candidate : chain.exprPlans) {
                    if (candidate.exprIndex == exprIdx) {
                        exprPlan = &candidate;
                        break;
                    }
                }
                if (!exprPlan) {
                    continue;
                }
                auto foulaIt = foulaSpecs.find(exprIdx);
                if (foulaIt != foulaSpecs.end()) {
                    generated += buildFoulaOwnerLoopSpecializationCode(
                        foulaIt->second, *exprPlan);
                } else {
                    generated += mpi_rewriter::buildOperatorResidentWrapperCode(
                        dacppFile, chain, *exprPlan);
                }
            } else if (isStencilPhaseC) {
                generated += mpi_stencil_rewriter::buildStencilWrapperCode(
                    dacppFile, shell, calc, exprIdx, expr->getDacExpr());
            } else {
                generated += mpi_rewriter::buildWrapperCode(
                    dacppFile, shell, calc, expr->getDacExpr());
            }
            generated += "\n";

            if (removedDecls.insert(shell->getShellLoc()).second) {
                rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
            }
            if (removedDecls.insert(calc->getCalcLoc()).second) {
                rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
            }
        }
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(), generated);

    std::set<const clang::BinaryOperator*> rewrittenDacExprs;
    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        const clang::BinaryOperator* dacExpr = expr->getDacExpr();
        const bool hasPlanResult =
            exprIdx < static_cast<int>(plan.exprResults.size());
        const mpi_rewriter::MpiPlanKind planKind =
            hasPlanResult ? plan.exprResults[exprIdx].kind
                          : mpi_rewriter::MpiPlanKind::Unsupported;
        auto foulaIt = foulaSpecs.find(exprIdx);
        if (foulaIt != foulaSpecs.end() && foulaIt->second.outerLoop &&
            rewrittenFoulaLoops.insert(foulaIt->second.outerLoop).second) {
            rewriter->ReplaceText(
                foulaIt->second.outerLoop->getSourceRange(),
                foulaIt->second.functionName + "(" +
                    foulaIt->second.ownerName + ", " +
                    foulaIt->second.scalarExpr + ");");
            rewrittenDacExprs.insert(dacExpr);
            llvm::outs() << "[DACPP][MPI][OR][FOuLa] expr=" << exprIdx
                         << " owner-loop=rewrite-enabled\n";
            continue;
        }
        if (planKind == mpi_rewriter::MpiPlanKind::StencilPhaseC) {
            auto siteIt = siteByExpr.find(dacExpr);
            if (siteIt != siteByExpr.end()) {
                mpi_stencil_rewriter::rewriteStencilPhaseCSite(
                    dacppFile, rewriter, siteIt->second, exprIdx, shell, calc);
                rewrittenDacExprs.insert(dacExpr);
                continue;
            }
        }
        const mpi_rewriter::ShellPartitionPlan* loopLowerPlan = nullptr;
        if (planKind == mpi_rewriter::MpiPlanKind::OperatorResident &&
            isLoopLoweredOperatorResidentExpr(plan, exprIdx, &loopLowerPlan)) {
            rewriteLoopLoweredOperatorResidentSite(dacppFile, rewriter,
                                                   *loopLowerPlan);
            rewrittenDacExprs.insert(dacExpr);
            continue;
        }

        const std::string wrapperName =
            planKind == mpi_rewriter::MpiPlanKind::OperatorResident
                ? mpi_rewriter::operatorResidentWrapperName(shell, calc, exprIdx)
                : shell->getName() + "_" + calc->getName();
        rewriter->ReplaceText(
            dacExpr->getSourceRange(),
            mpi_rewriter::buildWrapperCallForDacExpr(wrapperName, dacExpr,
                                                     dacppFile));
        if (planKind == mpi_rewriter::MpiPlanKind::OperatorResident) {
            const auto* exprPlan = findOperatorResidentExprPlan(plan, exprIdx);
            if (exprPlan &&
                mpi_rewriter::isShellDerivedStencilLayout(
                    exprPlan->signature.layout)) {
                removeOperatorResidentLoweredPostStmts(
                    dacppFile, rewriter, shell, calc, dacExpr);
            }
        }
        rewrittenDacExprs.insert(dacExpr);
    }

    for (const clang::BinaryOperator* dacExpr : dacppFile->dacExprs) {
        if (!dacExpr || rewrittenDacExprs.count(dacExpr) != 0) {
            continue;
        }
        const std::string wrapperName = wrapperNameForDacExpr(dacExpr);
        if (wrapperName.empty()) {
            continue;
        }
        rewriter->ReplaceText(
            dacExpr->getSourceRange(),
            mpi_rewriter::buildWrapperCallForDacExpr(wrapperName, dacExpr,
                                                     dacppFile));
    }

    const FunctionDecl* mainFunc = dacppFile->getMainFunction();
    if (!mainFunc) {
        return;
    }

    mpi_stencil_rewriter::insertMainMPISetup(dacppFile, rewriter, mainFunc);

    dacppFile->setMainAlreadyRewritten(true);
}

}  // namespace dacppTranslator
