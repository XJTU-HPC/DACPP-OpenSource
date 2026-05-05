#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "Rewriter_MPI_Stencil_Common.h"
#include "llvm/Support/raw_ostream.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {

namespace {

std::vector<const mpi_rewriter::DistributedFollowupMapping*> findFollowupMappingsForWriter(
    const mpi_rewriter::DistributedStencilSitePlan& plan,
    const std::string& writerTensor) {
    std::vector<const mpi_rewriter::DistributedFollowupMapping*> result;
    for (const auto& mapping : plan.followupMappings) {
        if (mapping.writerTensor == writerTensor) {
            result.push_back(&mapping);
        }
    }
    return result;
}

int findDefaultReaderIndex(const std::vector<IOTYPE>& transportModes) {
    for (int candidateIdx = 0; candidateIdx < static_cast<int>(transportModes.size());
         ++candidateIdx) {
        if (transportModes[candidateIdx] != IOTYPE::WRITE) {
            return candidateIdx;
        }
    }
    return -1;
}

std::string buildParamNameList(Shell* shell) {
    std::string args;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        if (!args.empty()) {
            args += ", ";
        }
        args += shell->getParam(paramIdx)->getName();
    }
    return args;
}

class NamedDeclRefVisitor
    : public clang::RecursiveASTVisitor<NamedDeclRefVisitor> {
public:
    explicit NamedDeclRefVisitor(std::string targetName)
        : targetName_(std::move(targetName)) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        if (dre && dre->getDecl() &&
            dre->getDecl()->getNameAsString() == targetName_) {
            found_ = true;
        }
        return !found_;
    }

    bool found() const { return found_; }

private:
    std::string targetName_;
    bool found_ = false;
};

bool exprReferencesName(const clang::Expr* expr, const std::string& name) {
    if (!expr || name.empty()) {
        return false;
    }
    NamedDeclRefVisitor visitor(name);
    visitor.TraverseStmt(const_cast<clang::Expr*>(expr));
    return visitor.found();
}

const clang::Expr* stripExpr(const clang::Expr* expr) {
    while (expr) {
        expr = expr->IgnoreParenImpCasts();
        if (const auto* materialized =
                llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
            expr = materialized->getSubExpr();
            continue;
        }
        if (const auto* cleanup = llvm::dyn_cast<clang::ExprWithCleanups>(expr)) {
            expr = cleanup->getSubExpr();
            continue;
        }
        if (const auto* bind = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
            expr = bind->getSubExpr();
            continue;
        }
        return expr;
    }
    return nullptr;
}

std::string directBaseDeclName(const clang::Expr* expr) {
    expr = stripExpr(expr);
    if (!expr) {
        return "";
    }
    if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        return dre->getDecl() ? dre->getDecl()->getNameAsString() : "";
    }
    if (const auto* subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        return directBaseDeclName(subscript->getBase());
    }
    if (const auto* member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
        return directBaseDeclName(member->getBase());
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
        if (unary->getOpcode() == clang::UO_Deref ||
            unary->getOpcode() == clang::UO_AddrOf) {
            return directBaseDeclName(unary->getSubExpr());
        }
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if ((opCall->getOperator() == clang::OO_Subscript ||
             opCall->getOperator() == clang::OO_Call) &&
            opCall->getNumArgs() > 0) {
            return directBaseDeclName(opCall->getArg(0));
        }
    }
    if (const auto* memberCall =
            llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) {
        return directBaseDeclName(memberCall->getImplicitObjectArgument());
    }
    return "";
}

bool isWholeDeclExprNamed(const clang::Expr* expr, const std::string& name) {
    expr = stripExpr(expr);
    const auto* dre = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
    return dre && dre->getDecl() && dre->getDecl()->getNameAsString() == name;
}

class TensorWriteVisitor
    : public clang::RecursiveASTVisitor<TensorWriteVisitor> {
public:
    explicit TensorWriteVisitor(std::string targetName)
        : targetName_(std::move(targetName)) {}

    bool TraverseBinaryOperator(clang::BinaryOperator* binary) {
        if (!binary || hasWrite_) {
            return !hasWrite_;
        }
        if (binary->isAssignmentOp()) {
            if (exprReferencesName(binary->getLHS(), targetName_)) {
                hasWrite_ = true;
                return false;
            }
            TraverseStmt(binary->getRHS());
            return !hasWrite_;
        }
        return clang::RecursiveASTVisitor<TensorWriteVisitor>::
            TraverseBinaryOperator(binary);
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* unary) {
        if (!unary || hasWrite_) {
            return !hasWrite_;
        }
        if (unary->isIncrementDecrementOp() &&
            exprReferencesName(unary->getSubExpr(), targetName_)) {
            hasWrite_ = true;
            return false;
        }
        return clang::RecursiveASTVisitor<TensorWriteVisitor>::
            TraverseUnaryOperator(unary);
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall || hasWrite_) {
            return !hasWrite_;
        }
        if (opCall->isAssignmentOp()) {
            if (opCall->getNumArgs() > 0 &&
                exprReferencesName(opCall->getArg(0), targetName_)) {
                hasWrite_ = true;
                return false;
            }
            for (unsigned argIdx = 1; argIdx < opCall->getNumArgs(); ++argIdx) {
                TraverseStmt(opCall->getArg(argIdx));
                if (hasWrite_) {
                    return false;
                }
            }
            return true;
        }
        return clang::RecursiveASTVisitor<TensorWriteVisitor>::
            TraverseCXXOperatorCallExpr(opCall);
    }

    bool TraverseCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call || hasWrite_) {
            return !hasWrite_;
        }
        const clang::CXXMethodDecl* method = call->getMethodDecl();
        if (method && !method->isConst() &&
            exprReferencesName(call->getImplicitObjectArgument(), targetName_)) {
            hasWrite_ = true;
            return false;
        }
        return clang::RecursiveASTVisitor<TensorWriteVisitor>::
            TraverseCXXMemberCallExpr(call);
    }

    bool TraverseCallExpr(clang::CallExpr* call) {
        if (!call || hasWrite_) {
            return !hasWrite_;
        }
        if (!llvm::isa<clang::CXXMemberCallExpr>(call) &&
            !llvm::isa<clang::CXXOperatorCallExpr>(call)) {
            for (const clang::Expr* arg : call->arguments()) {
                if (exprReferencesName(arg, targetName_)) {
                    hasWrite_ = true;
                    return false;
                }
            }
        }
        return clang::RecursiveASTVisitor<TensorWriteVisitor>::
            TraverseCallExpr(call);
    }

    bool hasWrite() const { return hasWrite_; }

private:
    std::string targetName_;
    bool hasWrite_ = false;
};

bool stmtWritesTensorName(const clang::Stmt* stmt, const std::string& name) {
    if (!stmt || name.empty()) {
        return false;
    }
    TensorWriteVisitor visitor(name);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    return visitor.hasWrite();
}

class LoopCarriedInputAssignmentVisitor
    : public clang::RecursiveASTVisitor<LoopCarriedInputAssignmentVisitor> {
public:
    explicit LoopCarriedInputAssignmentVisitor(std::string targetName)
        : targetName_(std::move(targetName)) {}

    bool TraverseBinaryOperator(clang::BinaryOperator* binary) {
        if (!binary || invalid_) {
            return !invalid_;
        }
        if (binary->isAssignmentOp() && exprReferencesName(binary->getLHS(),
                                                           targetName_)) {
            recordTargetWrite(binary->getLHS());
            return !invalid_;
        }
        return clang::RecursiveASTVisitor<LoopCarriedInputAssignmentVisitor>::
            TraverseBinaryOperator(binary);
    }

    bool TraverseUnaryOperator(clang::UnaryOperator* unary) {
        if (!unary || invalid_) {
            return !invalid_;
        }
        if (unary->isIncrementDecrementOp() &&
            directBaseDeclName(unary->getSubExpr()) == targetName_) {
            invalid_ = true;
            return false;
        }
        return clang::RecursiveASTVisitor<LoopCarriedInputAssignmentVisitor>::
            TraverseUnaryOperator(unary);
    }

    bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall || invalid_) {
            return !invalid_;
        }
        if (opCall->isAssignmentOp() && opCall->getNumArgs() > 0 &&
            exprReferencesName(opCall->getArg(0), targetName_)) {
            recordTargetWrite(opCall->getArg(0));
            return !invalid_;
        }
        if ((opCall->getOperator() == clang::OO_PlusPlus ||
             opCall->getOperator() == clang::OO_MinusMinus) &&
            opCall->getNumArgs() > 0 &&
            directBaseDeclName(opCall->getArg(0)) == targetName_) {
            invalid_ = true;
            return false;
        }
        return clang::RecursiveASTVisitor<LoopCarriedInputAssignmentVisitor>::
            TraverseCXXOperatorCallExpr(opCall);
    }

    bool TraverseCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call || invalid_) {
            return !invalid_;
        }
        if (directBaseDeclName(call->getImplicitObjectArgument()) == targetName_) {
            const clang::CXXMethodDecl* method = call->getMethodDecl();
            if (!method || !method->isConst()) {
                invalid_ = true;
                return false;
            }
        }
        return clang::RecursiveASTVisitor<LoopCarriedInputAssignmentVisitor>::
            TraverseCXXMemberCallExpr(call);
    }

    bool TraverseCallExpr(clang::CallExpr* call) {
        if (!call || invalid_) {
            return !invalid_;
        }
        if (!llvm::isa<clang::CXXMemberCallExpr>(call) &&
            !llvm::isa<clang::CXXOperatorCallExpr>(call)) {
            for (const clang::Expr* arg : call->arguments()) {
                if (directBaseDeclName(arg) == targetName_) {
                    invalid_ = true;
                    return false;
                }
            }
        }
        return clang::RecursiveASTVisitor<LoopCarriedInputAssignmentVisitor>::
            TraverseCallExpr(call);
    }

    bool hasAnyTargetWrite() const { return targetWriteCount_ > 0; }
    bool hasUnsafeTargetWrite() const { return invalid_; }
    int targetWriteCount() const { return targetWriteCount_; }

private:
    void recordTargetWrite(const clang::Expr* lhs) {
        ++targetWriteCount_;
        if (!isWholeDeclExprNamed(lhs, targetName_)) {
            invalid_ = true;
        }
    }

    std::string targetName_;
    bool invalid_ = false;
    int targetWriteCount_ = 0;
};

bool stmtIsDirectWholeTensorAssignmentFromAny(
    const clang::Stmt* stmt,
    const std::string& targetName,
    const std::unordered_map<std::string, int>& sourceParamByActualName,
    int& sourceParamIndex) {
    const auto* expr = llvm::dyn_cast_or_null<clang::Expr>(stmt);
    expr = stripExpr(expr);
    if (!expr) {
        return false;
    }

    const clang::Expr* lhs = nullptr;
    const clang::Expr* rhs = nullptr;
    bool simpleAssign = false;
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
        if (binary->isAssignmentOp()) {
            lhs = binary->getLHS();
            rhs = binary->getRHS();
            simpleAssign = binary->getOpcode() == clang::BO_Assign;
        }
    } else if (const auto* opCall =
                   llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->isAssignmentOp() && opCall->getNumArgs() > 1) {
            lhs = opCall->getArg(0);
            rhs = opCall->getArg(1);
            simpleAssign = opCall->getOperator() == clang::OO_Equal;
        }
    }

    if (!simpleAssign || !isWholeDeclExprNamed(lhs, targetName)) {
        return false;
    }
    const std::string sourceName = directBaseDeclName(rhs);
    auto it = sourceParamByActualName.find(sourceName);
    if (it == sourceParamByActualName.end()) {
        return false;
    }
    sourceParamIndex = it->second;
    return true;
}

bool fallbackOutputBroadcasts(DacppFile* dacppFile,
                              const std::string& tensorName,
                              const clang::BinaryOperator* dacExpr) {
    const mpi_rewriter::OutputSyncRequirement syncRequirement =
        mpi_rewriter::classifyOutputSyncRequirement(dacppFile, tensorName, dacExpr);
    return syncRequirement ==
               mpi_rewriter::OutputSyncRequirement::DistributedFollowup
               ? true
               : mpi_rewriter::requiresBroadcast(syncRequirement);
}

int findLoopCarriedInputSourceParam(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr,
    int targetParamIndex,
    const std::vector<std::string>& actualTensorNames,
    const std::vector<IOTYPE>& transportModes,
    const std::vector<bool>& aliasesAnyOtherParam) {
    if (!dacppFile || !shell || !calc || !dacExpr || targetParamIndex < 0 ||
        targetParamIndex >= static_cast<int>(actualTensorNames.size()) ||
        targetParamIndex >= static_cast<int>(transportModes.size()) ||
        transportModes[targetParamIndex] != IOTYPE::READ ||
        aliasesAnyOtherParam[static_cast<std::size_t>(targetParamIndex)]) {
        return -1;
    }

    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    if (!regionPlan.enabled || regionPlan.dacExpr != dacExpr ||
        regionPlan.siblingStmts.empty()) {
        return -1;
    }

    const std::string& targetActualName =
        actualTensorNames[static_cast<std::size_t>(targetParamIndex)];
    if (targetActualName.empty()) {
        return -1;
    }

    std::unordered_map<std::string, int> sourceParamByActualName;
    for (int sourceIdx = 0; sourceIdx < shell->getNumShellParams(); ++sourceIdx) {
        if (sourceIdx == targetParamIndex ||
            sourceIdx >= static_cast<int>(transportModes.size()) ||
            sourceIdx >= static_cast<int>(actualTensorNames.size()) ||
            transportModes[sourceIdx] != IOTYPE::WRITE ||
            aliasesAnyOtherParam[static_cast<std::size_t>(sourceIdx)] ||
            calc->getParam(sourceIdx)->getBasicType() !=
                calc->getParam(targetParamIndex)->getBasicType() ||
            mpi_rewriter::inferViewRank(shell->getShellParam(sourceIdx),
                                        calc->getParam(sourceIdx)) !=
                mpi_rewriter::inferViewRank(shell->getShellParam(targetParamIndex),
                                            calc->getParam(targetParamIndex)) ||
            !fallbackOutputBroadcasts(dacppFile, shell->getParam(sourceIdx)->getName(),
                                      dacExpr)) {
            continue;
        }
        const std::string& sourceActualName =
            actualTensorNames[static_cast<std::size_t>(sourceIdx)];
        if (sourceActualName.empty() || sourceActualName == targetActualName) {
            continue;
        }
        bool sourceWrittenBySibling = false;
        for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
            if (stmtWritesTensorName(stmt, sourceActualName)) {
                sourceWrittenBySibling = true;
                break;
            }
        }
        if (!sourceWrittenBySibling) {
            sourceParamByActualName.emplace(sourceActualName, sourceIdx);
        }
    }
    if (sourceParamByActualName.empty()) {
        return -1;
    }

    int directAssignmentSourceIdx = -1;
    bool sawDirectAssignment = false;
    LoopCarriedInputAssignmentVisitor writeVisitor(targetActualName);
    for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
        int candidateSourceIdx = -1;
        if (stmtIsDirectWholeTensorAssignmentFromAny(
                stmt, targetActualName, sourceParamByActualName,
                candidateSourceIdx)) {
            if (sawDirectAssignment ||
                (directAssignmentSourceIdx >= 0 &&
                 directAssignmentSourceIdx != candidateSourceIdx)) {
                return -1;
            }
            sawDirectAssignment = true;
            directAssignmentSourceIdx = candidateSourceIdx;
        }

        writeVisitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
        if (writeVisitor.hasUnsafeTargetWrite()) {
            return -1;
        }
    }
    if (writeVisitor.targetWriteCount() != 1) {
        return -1;
    }
    return sawDirectAssignment && directAssignmentSourceIdx >= 0
               ? directAssignmentSourceIdx
               : -1;
}

bool isFallbackInputCacheCandidate(DacppFile* dacppFile,
                                   const clang::BinaryOperator* dacExpr,
                                   const std::string& actualTensorName,
                                   IOTYPE transportMode) {
    if (!dacppFile || !dacExpr || actualTensorName.empty() ||
        transportMode != IOTYPE::READ) {
        return false;
    }

    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    if (!regionPlan.enabled || regionPlan.dacExpr != dacExpr) {
        return false;
    }

    for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
        if (stmtWritesTensorName(stmt, actualTensorName)) {
            return false;
        }
    }
    return true;
}

std::string resolveActualTensorName(const std::string& shellParamName,
                                    const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return shellParamName;
    }
    const clang::CallExpr* shellCall =
        dacppTranslator::getNode<clang::CallExpr>(
            dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                             : dacExpr->getRHS());
    if (!shellCall) {
        return shellParamName;
    }
    const clang::FunctionDecl* callee = shellCall->getDirectCallee();
    if (!callee) {
        return shellParamName;
    }
    for (unsigned paramIdx = 0;
         paramIdx < callee->getNumParams() && paramIdx < shellCall->getNumArgs();
         ++paramIdx) {
        const clang::ParmVarDecl* param = callee->getParamDecl(paramIdx);
        if (!param || param->getNameAsString() != shellParamName) {
            continue;
        }
        const auto* dre = dacppTranslator::getNode<clang::DeclRefExpr>(
            const_cast<clang::Expr*>(shellCall->getArg(paramIdx)));
        if (dre && dre->getDecl()) {
            return dre->getDecl()->getNameAsString();
        }
    }
    return shellParamName;
}

void appendPatternInitForContext(std::string& code,
                                 Shell* shell,
                                 Calc* calc,
                                 const std::unordered_map<std::string, mpi_rewriter::SplitBindMeta>& splitMeta,
                                 const std::vector<IOTYPE>& transportModes,
                                 const std::vector<IOTYPE>& itemDomainModes,
                                 const std::string& ctxVar) {
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string patternName = ctxVar + ".pattern_" + name;
        const IOTYPE mode = transportModes[paramIdx];

        code += "    " + patternName + " = dacpp::mpi::AccessPattern{};\n";
        code += "    " + patternName + ".param_id = " + std::to_string(paramIdx) + ";\n";
        code += "    " + patternName + ".name = \"" + name + "\";\n";
        code += "    " + patternName + ".mode = " + mpi_rewriter::toPlannerMode(mode) + ";\n";
        code += "    " + patternName + ".data_info.dim = " + tensorName + ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName + ".getDim(); ++dim) " +
                patternName + ".data_info.dimLength.push_back(" + tensorName + ".getShape(dim));\n";

        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            auto metaIt = splitMeta.find(split->getId());
            mpi_rewriter::SplitBindMeta bindMeta;
            if (metaIt != splitMeta.end()) {
                bindMeta = metaIt->second;
            }

            const bool isIndex = split->type == "IndexSplit";
            const std::string opName = "pattern_" + name + "_op_" + std::to_string(splitIdx);

            code += "    Dac_Op " + opName + ";\n";
            code += "    " + opName + ".setDimId(" + std::to_string(split->getDimIdx()) + ");\n";
            code += "    " + opName + ".size = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitSize()) + ";\n";
            code += "    " + opName + ".stride = " +
                    std::to_string(isIndex ? 1 : static_cast<RegularSplit*>(split)->getSplitStride()) + ";\n";
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

        code += "    " + patternName + ".partition_shape = dacpp::mpi::init_partition_shape(" +
                patternName + ");\n";
        code += "    " + patternName + ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(" +
                patternName + ");\n";
        if (paramIdx < static_cast<int>(itemDomainModes.size()) &&
            itemDomainModes[paramIdx] != IOTYPE::WRITE) {
            code += "    if (" + ctxVar + ".binding_split_sizes.size() < " + patternName +
                    ".bind_split_sizes.size()) " + ctxVar + ".binding_split_sizes.resize(" + patternName +
                    ".bind_split_sizes.size(), 1);\n";
            code += "    for (std::size_t bind_i = 0; bind_i < " + patternName + ".bind_split_sizes.size(); ++bind_i) {\n";
            code += "        " + ctxVar + ".binding_split_sizes[bind_i] = std::max<int64_t>(" +
                    ctxVar + ".binding_split_sizes[bind_i], " + patternName + ".bind_split_sizes[bind_i]);\n";
            code += "    }\n";
        }
    }
}

}  // namespace

std::string wrapperName(Shell* shell, Calc* calc) {
    return shell->getName() + "_" + calc->getName();
}

std::string contextTypeName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_ctx_" + wrapperName(shell, calc);
}

std::string initFunctionName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_init_" + wrapperName(shell, calc);
}

std::string runFunctionName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_run_" + wrapperName(shell, calc);
}

std::string materializeFunctionName(Shell* shell, Calc* calc) {
    return "__dacpp_mpi_stencil_materialize_" + wrapperName(shell, calc);
}

std::string buildShellSignature(Shell* shell) {
    std::string signature;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        Param* param = shell->getParam(paramIdx);
        std::string paramType = param->getType();
        if (!paramType.empty() && paramType.back() != '&' && paramType.back() != '*') {
            paramType += "&";
        }
        signature += paramType + " " + param->getName();
        if (paramIdx + 1 != shell->getNumParams()) {
            signature += ", ";
        }
    }
    return signature;
}

std::string buildStencilWrapperCode(DacppFile* dacppFile,
                                    Shell* shell,
                                    Calc* calc,
                                    const clang::BinaryOperator* dacExpr) {
    const std::string wrapper = wrapperName(shell, calc);
    const std::string ctxType = contextTypeName(shell, calc);
    const std::string initName = initFunctionName(shell, calc);
    const std::string runName = runFunctionName(shell, calc);
    const std::string materializeName = materializeFunctionName(shell, calc);
    const std::string shellSignature = buildShellSignature(shell);
    const std::string shellArgNames = buildParamNameList(shell);
    const auto effectiveModes = mpi_rewriter::inferEffectiveParamModes(shell, calc);
    const auto transportModes = mpi_rewriter::inferPhaseCTransportParamModes(shell, calc);
    const auto splitMeta = mpi_rewriter::collectSplitBindMeta(shell);
    const auto distributedSitePlan =
        mpi_rewriter::analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    std::vector<bool> fallbackInputCacheable(
        static_cast<std::size_t>(shell->getNumShellParams()), false);
    std::vector<int> fallbackInputRefreshSource(
        static_cast<std::size_t>(shell->getNumShellParams()), -1);
    std::vector<std::string> actualTensorNames(
        static_cast<std::size_t>(shell->getNumShellParams()));
    std::vector<bool> aliasesAnyOtherParam(
        static_cast<std::size_t>(shell->getNumShellParams()), false);
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        actualTensorNames[static_cast<std::size_t>(paramIdx)] =
            resolveActualTensorName(shell->getParam(paramIdx)->getName(), dacExpr);
    }
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& actualTensorName =
            actualTensorNames[static_cast<std::size_t>(paramIdx)];
        bool aliasesMutableParam = false;
        for (int otherIdx = 0; otherIdx < shell->getNumShellParams(); ++otherIdx) {
            if (otherIdx == paramIdx) {
                continue;
            }
            if (actualTensorNames[static_cast<std::size_t>(otherIdx)] ==
                    actualTensorName &&
                transportModes[otherIdx] != IOTYPE::READ) {
                aliasesMutableParam = true;
            }
            if (actualTensorNames[static_cast<std::size_t>(otherIdx)] ==
                actualTensorName) {
                aliasesAnyOtherParam[static_cast<std::size_t>(paramIdx)] = true;
            }
        }
        fallbackInputCacheable[static_cast<std::size_t>(paramIdx)] =
            !aliasesMutableParam &&
            !distributedSitePlan.supported &&
            isFallbackInputCacheCandidate(
                dacppFile, dacExpr, actualTensorName, transportModes[paramIdx]);
    }
    if (!distributedSitePlan.supported) {
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            if (fallbackInputCacheable[static_cast<std::size_t>(paramIdx)]) {
                continue;
            }
            const int sourceIdx = findLoopCarriedInputSourceParam(
                dacppFile, shell, calc, dacExpr, paramIdx, actualTensorNames,
                transportModes, aliasesAnyOtherParam);
            if (sourceIdx >= 0) {
                fallbackInputCacheable[static_cast<std::size_t>(paramIdx)] = true;
                fallbackInputRefreshSource[static_cast<std::size_t>(paramIdx)] =
                    sourceIdx;
            }
        }
    }

    if (distributedSitePlan.supported) {
        llvm::outs() << "[DACPP][MPI][PhaseC] site " << wrapper
                     << " partial-exchange enabled";
        if (distributedSitePlan.hasRootBridge) {
            llvm::outs() << " (root-bridge)";
        }
        llvm::outs() << "\n";
    } else {
        llvm::outs() << "[DACPP][MPI][PhaseC] site " << wrapper
                     << " partial-exchange disabled: "
                     << distributedSitePlan.disableReason << "\n";
    }
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (fallbackInputCacheable[static_cast<std::size_t>(paramIdx)]) {
            llvm::outs() << "[DACPP][MPI][FallbackCache] input "
                         << shell->getParam(paramIdx)->getName()
                         << " cached in init";
            const int sourceIdx =
                fallbackInputRefreshSource[static_cast<std::size_t>(paramIdx)];
            if (sourceIdx >= 0) {
                llvm::outs() << " refresh-from "
                             << shell->getParam(sourceIdx)->getName();
            }
            llvm::outs() << "\n";
        }
    }

    std::string code;
    code += "struct " + ctxType + " {\n";
    code += "    int mpi_rank = 0;\n";
    code += "    int mpi_size = 1;\n";
    code += "    bool use_partial_exchange = false;\n";
    code += "    std::string partial_exchange_disable_reason;\n";
    code += "    std::unique_ptr<sycl::queue> q;\n";
    code += "    std::vector<int64_t> binding_split_sizes;\n";
    code += "    int64_t total_items = 1;\n";
    code += "    dacpp::mpi::ItemRange item_range{};\n";
    code += "    int64_t local_item_count = 0;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        const std::string elemType = calc->getParam(paramIdx)->getBasicType();
        code += "    dacpp::mpi::AccessPattern pattern_" + calcName + ";\n";
        code += "    dacpp::mpi::PackPlan plan_" + calcName + ";\n";
        code += "    std::vector<" + elemType + "> local_" + calcName + ";\n";
        code += "    dacpp::mpi::DistributedTensorState<" + elemType + "> dist_" +
                calcName + ";\n";
        if (transportModes[paramIdx] != IOTYPE::WRITE) {
            code += "    dacpp::mpi::GatheredIndexLayout input_layout_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> sendbuf_" + calcName + ";\n";
        }
        if (transportModes[paramIdx] != IOTYPE::READ) {
            code += "    dacpp::mpi::GatheredIndexLayout output_layout_" + calcName + ";\n";
            code += "    std::vector<int32_t> writeback_slots_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> writeback_values_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_recv_values_" + calcName + ";\n";
            code += "    std::vector<" + elemType + "> global_out_" + calcName + ";\n";
        }
    }
    code += "};\n\n";

    code += "void " + initName + "(" + ctxType + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);\n";
    code += "    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);\n";
    code += "    ctx.q = std::make_unique<sycl::queue>(sycl::default_selector_v);\n";
    code += "    ctx.binding_split_sizes.clear();\n";
    code += "    ctx.total_items = 1;\n";
    appendPatternInitForContext(code, shell, calc, splitMeta, transportModes,
                                effectiveModes, "ctx");
    code += "    if (ctx.binding_split_sizes.empty()) ctx.binding_split_sizes.push_back(1);\n";
    code += "    for (int64_t split_size : ctx.binding_split_sizes) ctx.total_items *= split_size;\n";
    code += "    ctx.item_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);\n";
    code += "    ctx.local_item_count = ctx.item_range.size();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        const std::string& calcName = calc->getParam(paramIdx)->getName();
        const std::string actualTensorName =
            resolveActualTensorName(shell->getParam(paramIdx)->getName(), dacExpr);
        code += "    ctx.pattern_" + calcName + ".bind_split_sizes = ctx.binding_split_sizes;\n";
        code += "    ctx.plan_" + calcName + " = " +
                mpi_rewriter::buildPackPlanBuilderExpr(transportModes[paramIdx],
                                                       "ctx.item_range",
                                                       "ctx.pattern_" + calcName) + ";\n";
        code += "    ctx.local_" + calcName + ".resize(ctx.plan_" + calcName + ".pack.globals.size());\n";
        if (transportModes[paramIdx] != IOTYPE::WRITE) {
            code += "    dacpp::mpi::init_gathered_index_layout(ctx.input_layout_" + calcName +
                    ", ctx.plan_" + calcName + ".pack.globals, ctx.mpi_rank, ctx.mpi_size);\n";
            if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                code += "    if (ctx.mpi_rank == 0) dacpp::mpi::init_layout_byte_counts(ctx.input_layout_" +
                        calcName + ", sizeof(" + calc->getParam(paramIdx)->getBasicType() + "));\n";
            }
            if (fallbackInputCacheable[static_cast<std::size_t>(paramIdx)]) {
                const std::string& tensorName = shell->getParam(paramIdx)->getName();
                const std::string mpiType =
                    mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType());
                const int sourceIdx =
                    fallbackInputRefreshSource[static_cast<std::size_t>(paramIdx)];
                if (sourceIdx >= 0) {
                    code += "    // DACPP fallback loop-carried input: " +
                            tensorName + " <- " +
                            shell->getParam(sourceIdx)->getName() + "\n";
                } else {
                    code += "    // DACPP fallback cached input: " + tensorName + "\n";
                }
                code += "    if (ctx.mpi_rank == 0) {\n";
                code += "        " + tensorName + ".tensor2Array(ctx.global_" + calcName + ");\n";
                code += "        dacpp::mpi::pack_values_by_globals_parallel_range_into(ctx.global_" +
                        calcName + ", ctx.input_layout_" + calcName +
                        ".globals.data(), ctx.input_layout_" + calcName +
                        ".globals.size(), ctx.sendbuf_" + calcName + ");\n";
                code += "    }\n";
                if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".byte_displs.data() : nullptr, " + mpiType + ", ctx.local_" + calcName + ".data(), " +
                            mpi_rewriter::mpiPayloadCountExpr("ctx.input_layout_" + calcName + ".local_count",
                                                              calc->getParam(paramIdx)->getBasicType()) +
                            ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                } else {
                    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".displs.data() : nullptr, " + mpiType + ", ctx.local_" + calcName + ".data(), ctx.input_layout_" + calcName + ".local_count, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                }
            }
        }
        if (transportModes[paramIdx] != IOTYPE::READ) {
            code += "    const auto& writeback_globals_" + calcName + " = ctx.plan_" + calcName +
                    ".pack.writeback_globals.empty() ? ctx.plan_" + calcName +
                    ".pack.globals : ctx.plan_" + calcName + ".pack.writeback_globals;\n";
            code += "    dacpp::mpi::init_gathered_index_layout(ctx.output_layout_" + calcName +
                    ", writeback_globals_" + calcName + ", ctx.mpi_rank, ctx.mpi_size);\n";
            if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                code += "    if (ctx.mpi_rank == 0) dacpp::mpi::init_layout_byte_counts(ctx.output_layout_" +
                        calcName + ", sizeof(" + calc->getParam(paramIdx)->getBasicType() + "));\n";
            }
            code += "    dacpp::mpi::build_local_slots_for_globals(ctx.plan_" + calcName +
                    ".pack, ctx.writeback_slots_" + calcName + ");\n";
            code += "    ctx.writeback_values_" + calcName +
                    ".resize(ctx.writeback_slots_" + calcName + ".size());\n";
            code += "    if (ctx.mpi_rank == 0) ctx.global_recv_values_" + calcName +
                    ".resize(ctx.output_layout_" + calcName + ".globals.size());\n";
        }
    }
    if (distributedSitePlan.supported) {
        code += "    ctx.use_partial_exchange = true;\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* shellWrapperParam = shell->getParam(paramIdx);
            Param* calcParam = calc->getParam(paramIdx);
            const IOTYPE mode = transportModes[paramIdx];
            const std::string& calcName = calcParam->getName();
            const std::string& tensorName = shellWrapperParam->getName();
            const std::string actualTensorName =
                resolveActualTensorName(tensorName, dacExpr);

            code += "    ctx.dist_" + calcName + ".enabled = true;\n";
            if (mode != IOTYPE::WRITE) {
                code += "    dacpp::mpi::init_all_rank_index_layout(ctx.dist_" + calcName +
                        ".read_layout, ctx.plan_" + calcName +
                        ".pack.globals, ctx.mpi_rank, ctx.mpi_size);\n";
                code += "    ctx.dist_" + calcName + ".local_cache.resize(ctx.plan_" +
                        calcName + ".pack.globals.size());\n";
                code += "    if (ctx.mpi_rank == 0) {\n";
                code += "        " + tensorName + ".tensor2Array(ctx.global_" + calcName + ");\n";
                code += "        dacpp::mpi::pack_values_by_globals_parallel_range_into(ctx.global_" +
                        calcName + ", ctx.dist_" + calcName +
                        ".read_layout.globals.data(), ctx.dist_" + calcName +
                        ".read_layout.globals.size(), ctx.sendbuf_" + calcName + ");\n";
                code += "    }\n";
                if (mpi_rewriter::usesByteTransport(calc->getParam(paramIdx)->getBasicType())) {
                    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".byte_displs.data() : nullptr, " +
                            mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType()) +
                            ", ctx.dist_" + calcName + ".local_cache.data(), " +
                            mpi_rewriter::mpiPayloadCountExpr("ctx.input_layout_" + calcName + ".local_count",
                                                              calc->getParam(paramIdx)->getBasicType()) +
                            ", " +
                            mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType()) +
                            ", 0, MPI_COMM_WORLD);\n";
                } else {
                    code += "    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_" + calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" + calcName + ".displs.data() : nullptr, " +
                            mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType()) +
                            ", ctx.dist_" + calcName + ".local_cache.data(), ctx.input_layout_" + calcName + ".local_count, " +
                            mpi_rewriter::mpiDatatypeFor(calc->getParam(paramIdx)->getBasicType()) +
                            ", 0, MPI_COMM_WORLD);\n";
                }
                code += "    ctx.dist_" + calcName + ".seeded = true;\n";
                if (distributedSitePlan.rootBridgeTensors.count(tensorName) != 0) {
                    code += "    ctx.dist_" + calcName + ".root_bridge_pack = dacpp::mpi::make_dense_cover_pack(static_cast<std::size_t>(" +
                            tensorName + ".getSize()));\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout = dacpp::mpi::AllRankIndexLayout{};\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.local_count = ctx.mpi_rank == 0 ? static_cast<int>(ctx.dist_" + calcName + ".root_bridge_pack.globals.size()) : 0;\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.counts.assign(static_cast<std::size_t>(ctx.mpi_size), 0);\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.displs.assign(static_cast<std::size_t>(ctx.mpi_size), 0);\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_layout.globals = ctx.dist_" + calcName + ".root_bridge_pack.globals;\n";
                    code += "    if (ctx.mpi_size > 0) {\n";
                    code += "        ctx.dist_" + calcName + ".root_bridge_layout.counts[0] = static_cast<int>(ctx.dist_" + calcName + ".root_bridge_pack.globals.size());\n";
                    code += "    }\n";
                    code += "    ctx.dist_" + calcName + ".root_bridge_plan = dacpp::mpi::build_exchange_plan_from_layouts(ctx.dist_" + calcName + ".root_bridge_pack, ctx.dist_" + calcName + ".root_bridge_layout, ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".read_layout, ctx.mpi_rank, ctx.mpi_size);\n";
                }
            } else {
                code += "    ctx.dist_" + calcName + ".local_cache.assign(ctx.plan_" + calcName + ".pack.globals.size(), " + calcParam->getBasicType() + "{});\n";
                code += "    ctx.dist_" + calcName + ".local_write_slots = ctx.writeback_slots_" + calcName + ";\n";
                code += "    ctx.dist_" + calcName + ".local_write_globals = writeback_globals_" + calcName + ";\n";
                code += "    ctx.dist_" + calcName + ".local_write_values.resize(ctx.dist_" + calcName + ".local_write_slots.size());\n";
                code += "    dacpp::mpi::init_all_rank_index_layout(ctx.dist_" + calcName +
                        ".write_layout, writeback_globals_" + calcName +
                        ", ctx.mpi_rank, ctx.mpi_size);\n";
                code += "    if (!dacpp::mpi::validate_unique_writers(ctx.dist_" + calcName + ".write_layout, ctx.mpi_size, &ctx.partial_exchange_disable_reason)) {\n";
                code += "        ctx.use_partial_exchange = false;\n";
                code += "    }\n";
                const auto followupMappings =
                    findFollowupMappingsForWriter(distributedSitePlan, tensorName);
                if (!followupMappings.empty()) {
                    code += "    ctx.dist_" + calcName + ".local_target_slots_by_route.resize(" +
                            std::to_string(followupMappings.size()) + ");\n";
                    code += "    ctx.dist_" + calcName + ".exchange_plans_by_route.resize(" +
                            std::to_string(followupMappings.size()) + ");\n";
                    code += "    ctx.dist_" + calcName + ".halo_plans_by_route.resize(" +
                            std::to_string(followupMappings.size()) + ");\n";
                    for (std::size_t routeIdx = 0; routeIdx < followupMappings.size(); ++routeIdx) {
                        const auto* followupMapping = followupMappings[routeIdx];
                        const int readerIdx =
                            followupMapping ? followupMapping->readerParamIndex : -1;
                        if (readerIdx < 0) {
                            code += "    ctx.partial_exchange_disable_reason = \"phase-c route missing reader for WRITE tensor " + tensorName + "\";\n";
                            code += "    ctx.use_partial_exchange = false;\n";
                            continue;
                        }
                        const std::string& readerCalcName = calc->getParam(readerIdx)->getName();
                        if (followupMapping->rank == 2) {
                            const std::string writerCols =
                                calcName + "_cols";
                            const std::string readerCols =
                                readerCalcName + "_cols";
                            const std::string rowOffset =
                                std::to_string(followupMapping->targetRowOffset);
                            const std::string colOffset =
                                std::to_string(followupMapping->targetColOffset);
                            code += "    const int " + writerCols + " = " +
                                    shell->getParam(paramIdx)->getName() + ".getShape(1);\n";
                            code += "    const int " + readerCols + " = " +
                                    shell->getParam(readerIdx)->getName() + ".getShape(1);\n";
                            code += "    dacpp::mpi::build_target_slots_for_globals_2d_offset(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, " + writerCols + ", " + readerCols + ", " + rowOffset + ", " + colOffset + ", ctx.dist_" + calcName + ".local_target_slots_by_route[" + std::to_string(routeIdx) + "]);\n";
                            code += "    ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "] = dacpp::mpi::build_exchange_plan_from_layouts_2d_offset(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, " + writerCols + ", " + readerCols + ", " + rowOffset + ", " + colOffset + ", ctx.mpi_rank, ctx.mpi_size);\n";
                        } else {
                            const std::string targetOffset =
                                std::to_string(followupMapping->targetOffset);
                            if (followupMapping->targetOffset != 0) {
                                code += "    dacpp::mpi::build_target_slots_for_globals_with_offset(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, " + targetOffset + ", ctx.dist_" + calcName + ".local_target_slots_by_route[" + std::to_string(routeIdx) + "]);\n";
                                code += "    ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "] = dacpp::mpi::build_exchange_plan_from_layouts_with_target_offset(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, " + targetOffset + ", ctx.mpi_rank, ctx.mpi_size);\n";
                            } else {
                                code += "    dacpp::mpi::build_target_slots_for_globals(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, ctx.dist_" + calcName + ".local_target_slots_by_route[" + std::to_string(routeIdx) + "]);\n";
                                code += "    ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "] = dacpp::mpi::build_exchange_plan_from_layouts(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, ctx.mpi_rank, ctx.mpi_size);\n";
                            }
                        }
                        code += "    if (!ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "].supported) {\n";
                        code += "        ctx.partial_exchange_disable_reason = ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "].unsupported_reason;\n";
                        code += "        ctx.use_partial_exchange = false;\n";
                        code += "    }\n";
                        code += "    ctx.dist_" + calcName + ".halo_plans_by_route[" + std::to_string(routeIdx) + "] = dacpp::mpi::build_halo_plan_from_exchange_plan(ctx.dist_" + calcName + ".exchange_plans_by_route[" + std::to_string(routeIdx) + "]);\n";
                        llvm::outs() << "[DACPP][MPI][PhaseC] halo-exchange enabled route="
                                     << tensorName << "->"
                                     << shell->getParam(readerIdx)->getName()
                                     << "\n";
                    }
                    code += "    if (!ctx.dist_" + calcName + ".local_target_slots_by_route.empty()) ctx.dist_" + calcName + ".local_target_slots = ctx.dist_" + calcName + ".local_target_slots_by_route.front();\n";
                    code += "    if (!ctx.dist_" + calcName + ".exchange_plans_by_route.empty()) ctx.dist_" + calcName + ".exchange_plan = ctx.dist_" + calcName + ".exchange_plans_by_route.front();\n";
                    code += "    if (!ctx.dist_" + calcName + ".halo_plans_by_route.empty()) ctx.dist_" + calcName + ".halo_plan = ctx.dist_" + calcName + ".halo_plans_by_route.front();\n";
                } else if (!distributedSitePlan.hasRootBridge) {
                    const int readerIdx = findDefaultReaderIndex(transportModes);
                    if (readerIdx >= 0) {
                        const std::string& readerCalcName = calc->getParam(readerIdx)->getName();
                        code += "    ctx.dist_" + calcName + ".local_target_slots_by_route.resize(1);\n";
                        code += "    ctx.dist_" + calcName + ".exchange_plans_by_route.resize(1);\n";
                        code += "    ctx.dist_" + calcName + ".halo_plans_by_route.resize(1);\n";
                        code += "    dacpp::mpi::build_target_slots_for_globals(ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + calcName + ".local_write_globals, ctx.dist_" + calcName + ".local_target_slots_by_route[0]);\n";
                        code += "    ctx.dist_" + calcName + ".exchange_plans_by_route[0] = dacpp::mpi::build_exchange_plan_from_layouts(ctx.plan_" + calcName + ".pack, ctx.dist_" + calcName + ".write_layout, ctx.plan_" + readerCalcName + ".pack, ctx.dist_" + readerCalcName + ".read_layout, ctx.mpi_rank, ctx.mpi_size);\n";
                        code += "    if (!ctx.dist_" + calcName + ".exchange_plans_by_route[0].supported) {\n";
                        code += "        ctx.partial_exchange_disable_reason = ctx.dist_" + calcName + ".exchange_plans_by_route[0].unsupported_reason;\n";
                        code += "        ctx.use_partial_exchange = false;\n";
                        code += "    }\n";
                        code += "    ctx.dist_" + calcName + ".halo_plans_by_route[0] = dacpp::mpi::build_halo_plan_from_exchange_plan(ctx.dist_" + calcName + ".exchange_plans_by_route[0]);\n";
                        code += "    ctx.dist_" + calcName + ".local_target_slots = ctx.dist_" + calcName + ".local_target_slots_by_route.front();\n";
                        code += "    ctx.dist_" + calcName + ".exchange_plan = ctx.dist_" + calcName + ".exchange_plans_by_route.front();\n";
                        code += "    ctx.dist_" + calcName + ".halo_plan = ctx.dist_" + calcName + ".halo_plans_by_route.front();\n";
                        llvm::outs() << "[DACPP][MPI][PhaseC] halo-exchange enabled route="
                                     << tensorName << "->"
                                     << shell->getParam(readerIdx)->getName()
                                     << "\n";
                    } else {
                        code += "    ctx.partial_exchange_disable_reason = \"phase-c could not find reader for WRITE tensor " + tensorName + "\";\n";
                        code += "    ctx.use_partial_exchange = false;\n";
                    }
                }
            }
        }
    } else {
        code += "    ctx.use_partial_exchange = false;\n";
        code += "    ctx.partial_exchange_disable_reason = \"" +
                distributedSitePlan.disableReason + "\";\n";
    }
    code += "}\n\n";

    code += "void " + runName + "(" + ctxType + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    int mpi_rank = ctx.mpi_rank;\n";
    code += "    int mpi_size = ctx.mpi_size;\n";
    code += "    dacpp::mpi::resetCollectPositionsProfile();\n";
    code += "    auto dacpp_wrapper_start = std::chrono::steady_clock::now();\n";
    code += "    auto& q = *ctx.q;\n";
    code += "    const int64_t local_item_count = ctx.local_item_count;\n";
    code += "    if (!ctx.use_partial_exchange) {\n";

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const IOTYPE mode = transportModes[paramIdx];
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string packName = "ctx.plan_" + calcName + ".pack";
        const std::string slotsName = "ctx.plan_" + calcName + ".compact_slots";
        const std::string keyOffsetsName = "ctx.plan_" + calcName + ".item_key_offsets";
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_" + calcName;
        const std::string mpiType = mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());

        code += "    auto& " + localName + " = ctx.local_" + calcName + ";\n";
        if (mode != IOTYPE::WRITE) {
            if (fallbackInputCacheable[static_cast<std::size_t>(paramIdx)]) {
                const int sourceIdx =
                    fallbackInputRefreshSource[static_cast<std::size_t>(paramIdx)];
                if (sourceIdx >= 0) {
                    code += "    // DACPP fallback loop-carried input reused: " +
                            tensorName + "\n";
                } else {
                    code += "    // DACPP fallback cached input reused: " + tensorName + "\n";
                }
            } else {
                code += "    auto& input_layout_" + calcName + " = ctx.input_layout_" + calcName + ";\n";
                code += "    auto& global_" + calcName + " = ctx.global_" + calcName + ";\n";
                code += "    auto& sendbuf_" + calcName + " = ctx.sendbuf_" + calcName + ";\n";
                code += "    if (mpi_rank == 0) {\n";
                code += "        " + tensorName + ".tensor2Array(global_" + calcName + ");\n";
                code += "        dacpp::mpi::pack_values_by_globals_parallel_range_into(global_" + calcName +
                        ", input_layout_" + calcName + ".globals.data(), input_layout_" + calcName +
                        ".globals.size(), sendbuf_" + calcName + ");\n";
                code += "    }\n";
                code += "    " + localName + ".resize(static_cast<std::size_t>(input_layout_" + calcName + ".local_count));\n";
                if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
                    code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? input_layout_" + calcName + ".byte_counts.data() : nullptr, mpi_rank == 0 ? input_layout_" + calcName + ".byte_displs.data() : nullptr, " + mpiType + ", " + localName + ".data(), " +
                            mpi_rewriter::mpiPayloadCountExpr("input_layout_" + calcName + ".local_count",
                                                              calcParam->getBasicType()) +
                            ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                } else {
                    code += "    MPI_Scatterv(mpi_rank == 0 ? sendbuf_" + calcName + ".data() : nullptr, mpi_rank == 0 ? input_layout_" + calcName + ".counts.data() : nullptr, mpi_rank == 0 ? input_layout_" + calcName + ".displs.data() : nullptr, " + mpiType + ", " + localName + ".data(), input_layout_" + calcName + ".local_count, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                }
            }
        } else {
            code += "    " + localName + ".assign(" + packName + ".globals.size(), " + calcParam->getBasicType() + "{});\n";
        }

        code += "    const int " + calcName + "_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_" + calcName + "));\n";
        if (mpi_rewriter::inferViewRank(shellParam, calcParam) > 1) {
            code += "    const int " + calcName + "_cols = ctx.pattern_" + calcName + ".partition_shape[1];\n";
        }
    }

    code += "    if (local_item_count > 0) {\n";
    code += "        {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "            sycl::buffer<" + calcParam->getBasicType() + ", 1> buffer_" + name +
                "(local_" + name + ".data(), sycl::range<1>(local_" + name + ".size()));\n";
        code += "            sycl::buffer<int32_t, 1> slots_buffer_" + name +
                "(ctx.plan_" + name + ".compact_slots.data(), sycl::range<1>(ctx.plan_" + name +
                ".compact_slots.size()));\n";
        code += "            sycl::buffer<int32_t, 1> key_offsets_buffer_" + name +
                "(ctx.plan_" + name + ".item_key_offsets.data(), sycl::range<1>(ctx.plan_" + name +
                ".item_key_offsets.size()));\n";
    }
    code += "            q.submit([&](sycl::handler& h) {\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                auto acc_" + name + " = buffer_" + name + ".get_access<" +
                mpi_rewriter::toAccessorMode(effectiveModes[paramIdx]) + ">(h);\n";
        code += "                auto slots_acc_" + name +
                " = slots_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
        code += "                auto key_offsets_acc_" + name +
                " = key_offsets_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
    }
    code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {\n";
    code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        Param* calcParam = calc->getParam(paramIdx);
        const std::string& name = calcParam->getName();
        code += "                    auto* data_" + name +
                " = acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* slots_" + name +
                " = slots_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        code += "                    auto* key_offsets_" + name +
                " = key_offsets_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
        if (mpi_rewriter::inferViewRank(shellParam, calcParam) <= 1) {
            code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, effectiveModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                    "[item_linear]};\n";
        } else {
            code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, effectiveModes[paramIdx]) +
                    " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                    "[item_linear], " + name + "_cols};\n";
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
        if (transportModes[paramIdx] == IOTYPE::READ) {
            continue;
        }

        Param* calcParam = calc->getParam(paramIdx);
        Param* shellWrapperParam = shell->getParam(paramIdx);
        const std::string& calcName = calcParam->getName();
        const std::string& tensorName = shellWrapperParam->getName();
        const std::string packName = "ctx.plan_" + calcName + ".pack";
        const std::string localName = "local_" + calcName;
        const std::string globalName = "global_out_" + calcName;
        const std::string mpiType = mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());
        const mpi_rewriter::OutputSyncRequirement syncRequirement =
            mpi_rewriter::classifyOutputSyncRequirement(dacppFile, tensorName, dacExpr);
        const bool needsBcast =
            syncRequirement == mpi_rewriter::OutputSyncRequirement::DistributedFollowup
                ? true
                : mpi_rewriter::requiresBroadcast(syncRequirement);
        llvm::outs() << "[DACPP][MPI] output " << tensorName
                     << " sync="
                     << mpi_rewriter::outputSyncRequirementName(syncRequirement)
                     << "\n";

        code += "    auto& output_layout_" + calcName + " = ctx.output_layout_" + calcName + ";\n";
        code += "    auto& writeback_values_" + calcName + " = ctx.writeback_values_" + calcName + ";\n";
        code += "    auto& global_recv_values_" + calcName + " = ctx.global_recv_values_" + calcName + ";\n";
        code += "    auto& global_out_" + calcName + " = ctx.global_out_" + calcName + ";\n";
        code += "    dacpp::mpi::pack_values_by_slots_parallel_into(" + localName +
                ", ctx.writeback_slots_" + calcName + ", writeback_values_" + calcName + ");\n";
        if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
            code += "    MPI_Gatherv(writeback_values_" + calcName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr("output_layout_" + calcName + ".local_count",
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", mpi_rank == 0 ? global_recv_values_" + calcName +
                    ".data() : nullptr, mpi_rank == 0 ? output_layout_" + calcName +
                    ".byte_counts.data() : nullptr, mpi_rank == 0 ? output_layout_" + calcName +
                    ".byte_displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        } else {
            code += "    MPI_Gatherv(writeback_values_" + calcName + ".data(), output_layout_" + calcName + ".local_count, " + mpiType +
                    ", mpi_rank == 0 ? global_recv_values_" + calcName + ".data() : nullptr, mpi_rank == 0 ? output_layout_" +
                    calcName + ".counts.data() : nullptr, mpi_rank == 0 ? output_layout_" + calcName +
                    ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
        }
        code += "    if (mpi_rank == 0) {\n";
        code += "        " + tensorName + ".tensor2Array(global_out_" + calcName + ");\n";
        code += "        dacpp::mpi::apply_writeback_by_globals(global_recv_values_" + calcName +
                ", output_layout_" + calcName + ".globals, global_out_" + calcName + ");\n";
        code += "        " + tensorName + ".array2Tensor(global_out_" + calcName + ");\n";
        if (needsBcast) {
            code += "        if (!global_out_" + calcName + ".empty()) {\n";
            code += "            MPI_Bcast(global_out_" + calcName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr(tensorName + ".getSize()",
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            code += "        }\n";
        }
        code += "    } else ";
        if (needsBcast) {
            code += "{\n";
            code += "        global_out_" + calcName + ".resize(static_cast<std::size_t>(" +
                    tensorName + ".getSize()));\n";
            code += "        if (!global_out_" + calcName + ".empty()) {\n";
            code += "            MPI_Bcast(global_out_" + calcName + ".data(), " +
                    mpi_rewriter::mpiPayloadCountExpr(tensorName + ".getSize()",
                                                      calcParam->getBasicType()) +
                    ", " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            code += "        }\n";
            code += "        " + tensorName + ".array2Tensor(global_out_" + calcName + ");\n";
            code += "    }\n";
        } else {
            code += "{\n";
            code += "    }\n";
        }
        for (int readerIdx = 0; readerIdx < shell->getNumShellParams(); ++readerIdx) {
            if (fallbackInputRefreshSource[static_cast<std::size_t>(readerIdx)] !=
                paramIdx) {
                continue;
            }
            const std::string& readerCalcName =
                calc->getParam(readerIdx)->getName();
            const std::string& readerTensorName =
                shell->getParam(readerIdx)->getName();
            code += "    // DACPP fallback loop-carried input refreshed: " +
                    readerTensorName + " <- " + tensorName + "\n";
            code += "    dacpp::mpi::pack_values_by_globals_parallel_range_into(global_out_" +
                    calcName + ", ctx.plan_" + readerCalcName +
                    ".pack.globals.data(), ctx.plan_" + readerCalcName +
                    ".pack.globals.size(), ctx.local_" + readerCalcName + ");\n";
        }
    }
    if (distributedSitePlan.supported) {
        code += "    } else {\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& calcName = calcParam->getName();
            code += "    auto& local_" + calcName + " = ctx.dist_" + calcName +
                    ".local_cache;\n";
            if (transportModes[paramIdx] == IOTYPE::WRITE) {
                code += "    local_" + calcName + ".assign(ctx.plan_" + calcName +
                        ".pack.globals.size(), " + calcParam->getBasicType() + "{});\n";
            }
            if (mpi_rewriter::inferViewRank(shell->getShellParam(paramIdx), calcParam) > 1) {
                code += "    const int " + calcName + "_cols = ctx.pattern_" +
                        calcName + ".partition_shape[1];\n";
            }
        }
        code += "    if (local_item_count > 0) {\n";
        code += "        {\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "            sycl::buffer<" + calcParam->getBasicType() + ", 1> buffer_" + name +
                    "(local_" + name + ".data(), sycl::range<1>(local_" + name + ".size()));\n";
            code += "            sycl::buffer<int32_t, 1> slots_buffer_" + name +
                    "(ctx.plan_" + name + ".compact_slots.data(), sycl::range<1>(ctx.plan_" + name +
                    ".compact_slots.size()));\n";
            code += "            sycl::buffer<int32_t, 1> key_offsets_buffer_" + name +
                    "(ctx.plan_" + name + ".item_key_offsets.data(), sycl::range<1>(ctx.plan_" + name +
                    ".item_key_offsets.size()));\n";
        }
        code += "            q.submit([&](sycl::handler& h) {\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "                auto acc_" + name + " = buffer_" + name + ".get_access<" +
                    mpi_rewriter::toAccessorMode(effectiveModes[paramIdx]) + ">(h);\n";
            code += "                auto slots_acc_" + name +
                    " = slots_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
            code += "                auto key_offsets_acc_" + name +
                    " = key_offsets_buffer_" + name + ".get_access<sycl::access::mode::read>(h);\n";
        }
        code += "                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {\n";
        code += "                    const int item_linear = static_cast<int>(idx[0]);\n";
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            ShellParam* shellParam = shell->getShellParam(paramIdx);
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& name = calcParam->getName();
            code += "                    auto* data_" + name +
                    " = acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
            code += "                    auto* slots_" + name +
                    " = slots_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
            code += "                    auto* key_offsets_" + name +
                    " = key_offsets_acc_" + name + ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
            if (mpi_rewriter::inferViewRank(shellParam, calcParam) <= 1) {
                code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, effectiveModes[paramIdx]) +
                        " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                        "[item_linear]};\n";
            } else {
                code += "                    " + mpi_rewriter::toViewType(shellParam, calcParam, effectiveModes[paramIdx]) +
                        " view_" + name + "{data_" + name + ", slots_" + name + ", key_offsets_" + name +
                        "[item_linear], " + name + "_cols};\n";
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
            if (transportModes[paramIdx] == IOTYPE::READ) {
                continue;
            }
            Param* calcParam = calc->getParam(paramIdx);
            const std::string& calcName = calcParam->getName();
            Param* shellWrapperParam = shell->getParam(paramIdx);
            const std::string& tensorName = shellWrapperParam->getName();
            const auto followupMappings =
                findFollowupMappingsForWriter(distributedSitePlan, tensorName);
            const std::size_t routeCount =
                !followupMappings.empty()
                    ? followupMappings.size()
                    : (distributedSitePlan.hasRootBridge ? 0 : 1);
            for (std::size_t routeIdx = 0; routeIdx < routeCount; ++routeIdx) {
                const auto* followupMapping =
                    !followupMappings.empty() ? followupMappings[routeIdx] : nullptr;
                std::string targetCache = "ctx.dist_" + calcName + ".local_cache";
                if (followupMapping && followupMapping->readerParamIndex >= 0) {
                    targetCache = "ctx.dist_" +
                                  calc->getParam(followupMapping->readerParamIndex)->getName() +
                                  ".local_cache";
                }
                code += "    dacpp::mpi::publish_local_writes_with_halo_or_exchange(local_" + calcName +
                        ", ctx.dist_" + calcName + ".local_write_slots, ctx.dist_" + calcName +
                        ".local_target_slots_by_route[" + std::to_string(routeIdx) + "], " +
                        targetCache + ", ctx.dist_" + calcName +
                        ".local_write_values, ctx.dist_" + calcName +
                        ".exchange_plans_by_route[" + std::to_string(routeIdx) +
                        "], ctx.dist_" + calcName + ".halo_plans_by_route[" +
                        std::to_string(routeIdx) + "]);\n";
            }
        }
        if (distributedSitePlan.hasRootBridge) {
            for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
                if (transportModes[paramIdx] == IOTYPE::READ) {
                    continue;
                }
                Param* calcParam = calc->getParam(paramIdx);
                Param* shellWrapperParam = shell->getParam(paramIdx);
                const std::string& calcName = calcParam->getName();
                const std::string& tensorName = shellWrapperParam->getName();
                const std::string mpiType =
                    mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());

                code += "    dacpp::mpi::pack_values_by_slots_parallel_into(local_" + calcName +
                        ", ctx.writeback_slots_" + calcName + ", ctx.writeback_values_" +
                        calcName + ");\n";
                if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
                    code += "    MPI_Gatherv(ctx.writeback_values_" + calcName + ".data(), " +
                            mpi_rewriter::mpiPayloadCountExpr("ctx.output_layout_" + calcName + ".local_count",
                                                              calcParam->getBasicType()) +
                            ", " + mpiType + ", ctx.mpi_rank == 0 ? ctx.global_recv_values_" +
                            calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" +
                            calcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" +
                            calcName + ".byte_displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                } else {
                    code += "    MPI_Gatherv(ctx.writeback_values_" + calcName +
                            ".data(), ctx.output_layout_" + calcName + ".local_count, " + mpiType +
                            ", ctx.mpi_rank == 0 ? ctx.global_recv_values_" + calcName +
                            ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" + calcName +
                            ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" + calcName +
                            ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
                }
                code += "    if (ctx.mpi_rank == 0) {\n";
                code += "        " + tensorName + ".tensor2Array(ctx.global_out_" + calcName + ");\n";
                code += "        dacpp::mpi::apply_writeback_by_globals(ctx.global_recv_values_" + calcName +
                        ", ctx.output_layout_" + calcName + ".globals, ctx.global_out_" + calcName + ");\n";
                code += "        " + tensorName + ".array2Tensor(ctx.global_out_" + calcName + ");\n";
                code += "    }\n";
            }
        }
        code += "    }\n";
    } else {
        code += "    }\n";
    }

    code += "    if (dacpp::mpi::profilingEnabled()) {\n";
    code += "        auto dacpp_wrapper_end = std::chrono::steady_clock::now();\n";
    code += "        double dacpp_wrapper_local_ms = std::chrono::duration<double, std::milli>(dacpp_wrapper_end - dacpp_wrapper_start).count();\n";
    code += "        double dacpp_wrapper_max_ms = 0.0;\n";
    code += "        MPI_Reduce(&dacpp_wrapper_local_ms, &dacpp_wrapper_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);\n";
    code += "        if (mpi_rank == 0) {\n";
    code += "        std::fprintf(stderr, \"[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\\n\", \"" + wrapper + "\", dacpp_wrapper_max_ms);\n";
    code += "        }\n";
    code += "    }\n";
    code += "    dacpp::mpi::reportCollectPositionsProfile(\"" + wrapper + "\", MPI_COMM_WORLD);\n";
    code += "}\n\n";

    code += "void " + materializeName + "(" + ctxType + "& ctx";
    if (!shellSignature.empty()) {
        code += ", " + shellSignature;
    }
    code += ") {\n";
    code += "    if (!ctx.use_partial_exchange) {\n";
    code += "        return;\n";
    code += "    }\n";
    if (!distributedSitePlan.supported || distributedSitePlan.hasRootBridge) {
        code += "    return;\n";
    } else if (!distributedSitePlan.followupMappings.empty()) {
        for (const auto& mapping : distributedSitePlan.followupMappings) {
            if (mapping.writerParamIndex < 0 || mapping.readerParamIndex < 0) {
                continue;
            }
            Param* writerCalcParam = calc->getParam(mapping.writerParamIndex);
            Param* readerCalcParam = calc->getParam(mapping.readerParamIndex);
            Param* readerShellParam = shell->getParam(mapping.readerParamIndex);
            const std::string& writerCalcName = writerCalcParam->getName();
            const std::string& readerCalcName = readerCalcParam->getName();
            const std::string& readerTensorName = readerShellParam->getName();
            const std::string mpiType =
                mpi_rewriter::mpiDatatypeFor(writerCalcParam->getBasicType());

            if (mpi_rewriter::usesByteTransport(writerCalcParam->getBasicType())) {
                code += "    MPI_Gatherv(ctx.dist_" + writerCalcName +
                        ".local_write_values.data(), " +
                        mpi_rewriter::mpiPayloadCountExpr("ctx.output_layout_" + writerCalcName + ".local_count",
                                                          writerCalcParam->getBasicType()) +
                        ", " + mpiType + ", ctx.mpi_rank == 0 ? ctx.global_recv_values_" +
                        writerCalcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" +
                        writerCalcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" +
                        writerCalcName + ".byte_displs.data() : nullptr, " + mpiType +
                        ", 0, MPI_COMM_WORLD);\n";
            } else {
                code += "    MPI_Gatherv(ctx.dist_" + writerCalcName +
                        ".local_write_values.data(), ctx.output_layout_" + writerCalcName +
                        ".local_count, " + mpiType +
                        ", ctx.mpi_rank == 0 ? ctx.global_recv_values_" + writerCalcName +
                        ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" + writerCalcName +
                        ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.output_layout_" + writerCalcName +
                        ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        " + readerTensorName + ".tensor2Array(ctx.global_" + readerCalcName + ");\n";
            if (mapping.rank == 2) {
                Param* writerShellParam = shell->getParam(mapping.writerParamIndex);
                code += "        const int64_t __dacpp_writer_cols = static_cast<int64_t>(" +
                        writerShellParam->getName() + ".getShape(1));\n";
                code += "        const int64_t __dacpp_reader_cols = static_cast<int64_t>(" +
                        readerTensorName + ".getShape(1));\n";
            }
            code += "        for (std::size_t __dacpp_idx = 0; __dacpp_idx < ctx.output_layout_" +
                    writerCalcName + ".globals.size() && __dacpp_idx < ctx.global_recv_values_" +
                    writerCalcName + ".size(); ++__dacpp_idx) {\n";
            if (mapping.rank == 2) {
                code += "            const int64_t __dacpp_target_global = dacpp::mpi::map_2d_global_with_offset(ctx.output_layout_" +
                        writerCalcName + ".globals[__dacpp_idx], __dacpp_writer_cols, __dacpp_reader_cols, " +
                        std::to_string(mapping.targetRowOffset) + ", " +
                        std::to_string(mapping.targetColOffset) + ");\n";
            } else {
                code += "            const int64_t __dacpp_target_global = ctx.output_layout_" +
                        writerCalcName + ".globals[__dacpp_idx] + static_cast<int64_t>(" +
                        std::to_string(mapping.targetOffset) + ");\n";
            }
            code += "            if (__dacpp_target_global >= 0 && static_cast<std::size_t>(__dacpp_target_global) < ctx.global_" +
                    readerCalcName + ".size()) {\n";
            code += "                ctx.global_" + readerCalcName + "[static_cast<std::size_t>(__dacpp_target_global)] = ctx.global_recv_values_" +
                    writerCalcName + "[__dacpp_idx];\n";
            code += "            }\n";
            code += "        }\n";
            code += "        " + readerTensorName + ".array2Tensor(ctx.global_" + readerCalcName + ");\n";
            code += "    }\n";
        }
    } else {
        for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
            if (transportModes[paramIdx] == IOTYPE::WRITE) {
                continue;
            }
            Param* calcParam = calc->getParam(paramIdx);
            Param* shellWrapperParam = shell->getParam(paramIdx);
            const std::string& calcName = calcParam->getName();
            const std::string& tensorName = shellWrapperParam->getName();
            const std::string mpiType =
                mpi_rewriter::mpiDatatypeFor(calcParam->getBasicType());

            if (mpi_rewriter::usesByteTransport(calcParam->getBasicType())) {
                code += "    MPI_Gatherv(ctx.dist_" + calcName + ".local_cache.data(), " +
                        mpi_rewriter::mpiPayloadCountExpr("ctx.input_layout_" + calcName + ".local_count",
                                                          calcParam->getBasicType()) +
                        ", " + mpiType + ", ctx.mpi_rank == 0 ? ctx.sendbuf_" +
                        calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        calcName + ".byte_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        calcName + ".byte_displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            } else {
                code += "    MPI_Gatherv(ctx.dist_" + calcName + ".local_cache.data(), ctx.input_layout_" +
                        calcName + ".local_count, " + mpiType + ", ctx.mpi_rank == 0 ? ctx.sendbuf_" +
                        calcName + ".data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        calcName + ".counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_" +
                        calcName + ".displs.data() : nullptr, " + mpiType + ", 0, MPI_COMM_WORLD);\n";
            }
            code += "    if (ctx.mpi_rank == 0) {\n";
            code += "        " + tensorName + ".tensor2Array(ctx.global_" + calcName + ");\n";
            code += "        dacpp::mpi::apply_writeback_by_globals(ctx.sendbuf_" + calcName +
                    ", ctx.input_layout_" + calcName + ".globals, ctx.global_" + calcName + ");\n";
            code += "        " + tensorName + ".array2Tensor(ctx.global_" + calcName + ");\n";
            code += "    }\n";
        }
    }
    code += "}\n\n";

    code += "void " + wrapper + "(" + shellSignature + ") {\n";
    code += "    " + ctxType + " ctx;\n";
    code += "    " + initName + "(ctx";
    if (!shellArgNames.empty()) {
        code += ", " + shellArgNames;
    }
    code += ");\n";
    code += "    " + runName + "(ctx";
    if (!shellArgNames.empty()) {
        code += ", " + shellArgNames;
    }
    code += ");\n";
    code += "    " + materializeName + "(ctx";
    if (!shellArgNames.empty()) {
        code += ", " + shellArgNames;
    }
    code += ");\n";
    code += "}\n";

    code += "\n";
    code += mpi_rewriter::buildRootCentricPostRegionHelpers(
        dacppFile, shell, calc, dacExpr, ctxType, shellSignature);

    return code;
}

}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator
