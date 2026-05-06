#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "Rewriter_MPI_Stencil_Codegen_Internal.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {
namespace detail {

std::vector<const mpi_rewriter::DistributedFollowupMapping*>
findFollowupMappingsForWriter(
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
        mpi_rewriter::classifyOutputSyncRequirement(dacppFile, tensorName,
                                                    dacExpr);
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
            !fallbackOutputBroadcasts(dacppFile,
                                      shell->getParam(sourceIdx)->getName(),
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

void appendPatternInitForContext(
    std::string& code,
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
        code += "    " + patternName + ".param_id = " +
                std::to_string(paramIdx) + ";\n";
        code += "    " + patternName + ".name = \"" + name + "\";\n";
        code += "    " + patternName + ".mode = " +
                mpi_rewriter::toPlannerMode(mode) + ";\n";
        code += "    " + patternName + ".data_info.dim = " + tensorName +
                ".getDim();\n";
        code += "    for (int dim = 0; dim < " + tensorName +
                ".getDim(); ++dim) " + patternName +
                ".data_info.dimLength.push_back(" + tensorName +
                ".getShape(dim));\n";

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
            const std::string opName =
                "pattern_" + name + "_op_" + std::to_string(splitIdx);

            code += "    Dac_Op " + opName + ";\n";
            code += "    " + opName + ".setDimId(" +
                    std::to_string(split->getDimIdx()) + ");\n";
            code += "    " + opName + ".size = " +
                    std::to_string(isIndex
                                       ? 1
                                       : static_cast<RegularSplit*>(split)
                                             ->getSplitSize()) +
                    ";\n";
            code += "    " + opName + ".stride = " +
                    std::to_string(isIndex
                                       ? 1
                                       : static_cast<RegularSplit*>(split)
                                             ->getSplitStride()) +
                    ";\n";
            if (isIndex) {
                code += "    " + opName + ".SetSplitSize(" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) +
                        "));\n";
            } else {
                code += "    " + opName + ".SetSplitSize((" + tensorName +
                        ".getShape(" + std::to_string(split->getDimIdx()) +
                        ") - " +
                        std::to_string(static_cast<RegularSplit*>(split)
                                           ->getSplitSize()) +
                        ") / " +
                        std::to_string(static_cast<RegularSplit*>(split)
                                           ->getSplitStride()) +
                        " + 1);\n";
            }
            code += "    " + patternName + ".param_ops.push_back(" + opName +
                    ");\n";
            code += "    " + patternName + ".bind_set_id.push_back(" +
                    std::to_string(bindMeta.bindId) + ");\n";
            code += "    " + patternName + ".bind_offset_expr.push_back(\"" +
                    bindMeta.offset + "\");\n";
            code += "    " + patternName + ".is_index_op.push_back(" +
                    std::string(isIndex ? "true" : "false") + ");\n";
        }

        code += "    " + patternName +
                ".partition_shape = dacpp::mpi::init_partition_shape(" +
                patternName + ");\n";
        code += "    " + patternName +
                ".bind_split_sizes = dacpp::mpi::init_bind_split_sizes(" +
                patternName + ");\n";
        if (paramIdx < static_cast<int>(itemDomainModes.size()) &&
            itemDomainModes[paramIdx] != IOTYPE::WRITE) {
            code += "    if (" + ctxVar + ".binding_split_sizes.size() < " +
                    patternName + ".bind_split_sizes.size()) " + ctxVar +
                    ".binding_split_sizes.resize(" + patternName +
                    ".bind_split_sizes.size(), 1);\n";
            code += "    for (std::size_t bind_i = 0; bind_i < " + patternName +
                    ".bind_split_sizes.size(); ++bind_i) {\n";
            code += "        " + ctxVar +
                    ".binding_split_sizes[bind_i] = std::max<int64_t>(" +
                    ctxVar + ".binding_split_sizes[bind_i], " + patternName +
                    ".bind_split_sizes[bind_i]);\n";
            code += "    }\n";
        }
    }
}

}  // namespace detail
}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator
