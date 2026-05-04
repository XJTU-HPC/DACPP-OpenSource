#include <set>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "Rewriter.h"
#include "Split.h"
#include "Param.h"
#include "dacInfo.h"
//  #include "BUFFER_template.h"
#include "buffer_template_new.h"
#include "usm_template.h"
#include "Calc.h"
#include "ASTParse.h"

namespace BUFFER_TEMPLATE {
std::string parallelizeSingleFor(const clang::ForStmt* FS,
                                 clang::ASTContext* Context,
                                 dacppTranslator::DacppFile* dacFile);
}

namespace {

struct SplitRecord {
    std::string id;
    std::string type;
    int dimId = 0;
    int splitSize = 0;
    int splitStride = 0;
    std::string infoName;
};

struct BufferRegionGeneratedCode {
    std::string definitions;
    std::string ctxTypeName;
    std::string ctxVarName;
    std::string submitName;
    std::string initName;
    std::string syncName;
    std::vector<std::pair<const clang::Stmt*, std::string>> siblingHelpers;
};

struct AccessSummary {
    bool reads = false;
    bool writes = false;
};

struct BufferRegionTransferPolicy {
    std::vector<bool> needsInitCopy;
    std::vector<bool> needsSyncCopy;
    std::vector<bool> writtenInRegion;
    std::vector<bool> hostUsedAfterLoop;
};

class ParamAccessVisitor : public clang::RecursiveASTVisitor<ParamAccessVisitor> {
public:
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

    std::vector<bool> Reads;
    std::vector<bool> UpdateReads;
    std::vector<bool> Writes;

private:
    const std::unordered_map<const clang::ValueDecl*, int>& ParamIndices;
    int WriteDepth = 0;
    int UpdateReadDepth = 0;
};

std::vector<dacppTranslator::IOTYPE> inferEffectiveBufferParamModes(
    dacppTranslator::Shell* shell,
    dacppTranslator::Calc* calc) {
    std::vector<dacppTranslator::IOTYPE> modes;
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
            modes[paramIdx] = dacppTranslator::IOTYPE::READ_WRITE;
        } else if (writes && updateReads) {
            modes[paramIdx] = dacppTranslator::IOTYPE::READ_WRITE;
        } else if (writes) {
            modes[paramIdx] = dacppTranslator::IOTYPE::WRITE;
        } else if (reads || updateReads) {
            modes[paramIdx] = dacppTranslator::IOTYPE::READ;
        }
    }

    return modes;
}

bool shouldTrackValueDecl(const clang::ValueDecl* decl) {
    if (!decl) {
        return false;
    }

    if (!llvm::isa<clang::VarDecl>(decl)) {
        return false;
    }

    const clang::QualType type = decl->getType();
    if (type.isNull()) {
        return false;
    }

    if (type->isArithmeticType() || type->isEnumeralType() ||
        type->isFunctionPointerType()) {
        return false;
    }

    if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl)) {
        if (varDecl->isConstexpr()) {
            return false;
        }
    }

    return true;
}

class ValueDeclRefCollector : public clang::RecursiveASTVisitor<ValueDeclRefCollector> {
public:
    explicit ValueDeclRefCollector(std::set<const clang::ValueDecl*>& decls)
        : Decls(decls) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
        if (DRE && DRE->getDecl() && shouldTrackValueDecl(DRE->getDecl())) {
            Decls.insert(DRE->getDecl());
        }
        return true;
    }

private:
    std::set<const clang::ValueDecl*>& Decls;
};

const clang::CallExpr* getShellCallExpr(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }

    const clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    return dacppTranslator::getNode<clang::CallExpr>(
        const_cast<clang::Expr*>(shellExpr));
}

const clang::ValueDecl* getExprValueDecl(const clang::Expr* expr) {
    if (!expr) {
        return nullptr;
    }

    const clang::Expr* stripped = expr->IgnoreParenImpCasts();
    if (const auto* DRE = llvm::dyn_cast<clang::DeclRefExpr>(stripped)) {
        return DRE->getDecl();
    }
    return nullptr;
}

std::vector<const clang::ValueDecl*> collectShellCallArgDecls(
    const clang::BinaryOperator* dacExpr,
    int expectedArgCount) {
    std::vector<const clang::ValueDecl*> argDecls(
        static_cast<std::size_t>(expectedArgCount), nullptr);
    const clang::CallExpr* shellCall = getShellCallExpr(dacExpr);
    if (!shellCall) {
        return argDecls;
    }

    const int argCount = std::min<int>(expectedArgCount, shellCall->getNumArgs());
    for (int argIdx = 0; argIdx < argCount; ++argIdx) {
        argDecls[static_cast<std::size_t>(argIdx)] =
            getExprValueDecl(shellCall->getArg(static_cast<unsigned>(argIdx)));
    }
    return argDecls;
}

const clang::CompoundStmt* findContainingCompoundStmt(const clang::Stmt* root,
                                                      const clang::Stmt* target) {
    if (!root || !target) {
        return nullptr;
    }

    if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(root)) {
        for (const clang::Stmt* child : compound->body()) {
            if (child == target) {
                return compound;
            }
            if (const auto* nested = findContainingCompoundStmt(child, target)) {
                return nested;
            }
        }
        return nullptr;
    }

    for (const clang::Stmt* child : root->children()) {
        if (const auto* nested = findContainingCompoundStmt(child, target)) {
            return nested;
        }
    }
    return nullptr;
}

std::vector<const clang::Stmt*> collectStatementsAfterTarget(
    const clang::CompoundStmt* compound,
    const clang::Stmt* target) {
    std::vector<const clang::Stmt*> stmts;
    if (!compound || !target) {
        return stmts;
    }

    bool foundTarget = false;
    for (const clang::Stmt* child : compound->body()) {
        if (!foundTarget) {
            foundTarget = (child == target);
            continue;
        }
        if (child) {
            stmts.push_back(child);
        }
    }
    return stmts;
}

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
        summary[static_cast<std::size_t>(paramIdx)].writes = visitor.Writes[paramIdx];
    }
    return summary;
}

void collectTrackedValueDecls(const clang::ValueDecl* decl,
                              std::set<const clang::ValueDecl*>& trackedDecls,
                              std::set<const clang::ValueDecl*>& visitedDecls) {
    if (!decl || !visitedDecls.insert(decl).second) {
        return;
    }

    trackedDecls.insert(decl);
    const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
    if (!varDecl || !varDecl->hasInit()) {
        return;
    }

    std::set<const clang::ValueDecl*> referencedDecls;
    ValueDeclRefCollector collector(referencedDecls);
    collector.TraverseStmt(const_cast<clang::Expr*>(varDecl->getInit()));
    for (const clang::ValueDecl* referenced : referencedDecls) {
        if (referenced != decl) {
            collectTrackedValueDecls(referenced, trackedDecls, visitedDecls);
        }
    }
}

bool isPureLocalTrackedDeclSet(const std::set<const clang::ValueDecl*>& trackedDecls) {
    if (trackedDecls.empty()) {
        return false;
    }

    for (const clang::ValueDecl* decl : trackedDecls) {
        const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
        if (!varDecl || llvm::isa<clang::ParmVarDecl>(varDecl) ||
            !varDecl->isLocalVarDecl() || varDecl->hasGlobalStorage()) {
            return false;
        }
    }
    return true;
}

AccessSummary summarizeMainKernelAccess(dacppTranslator::IOTYPE mode) {
    AccessSummary summary;
    if (mode == dacppTranslator::IOTYPE::READ) {
        summary.reads = true;
    } else if (mode == dacppTranslator::IOTYPE::WRITE) {
        summary.writes = true;
    } else if (mode == dacppTranslator::IOTYPE::READ_WRITE) {
        summary.reads = true;
        summary.writes = true;
    }
    return summary;
}

void applyAccessToTransferState(const AccessSummary& access,
                                bool& needsInitCopy,
                                bool& writtenInRegion) {
    if (access.reads && !writtenInRegion) {
        needsInitCopy = true;
    }
    if (access.writes) {
        writtenInRegion = true;
    }
}

BufferRegionTransferPolicy analyzeBufferRegionTransferPolicy(
    dacppTranslator::DacppFile* dacppFile,
    dacppTranslator::Expression* expr,
    const std::vector<dacppTranslator::IOTYPE>& effectiveParamModes) {
    BufferRegionTransferPolicy policy;
    if (!dacppFile || !expr) {
        return policy;
    }

    const auto& plan = dacppFile->getBufferRegionPlan();
    auto* shell = expr->getShell();
    const int paramCount = shell ? shell->getNumParams() : 0;
    policy.needsInitCopy.assign(static_cast<std::size_t>(paramCount), false);
    policy.needsSyncCopy.assign(static_cast<std::size_t>(paramCount), false);
    policy.writtenInRegion.assign(static_cast<std::size_t>(paramCount), false);
    policy.hostUsedAfterLoop.assign(static_cast<std::size_t>(paramCount), false);
    if (!shell || paramCount == 0 || !plan.outerFor) {
        return policy;
    }

    const auto argDecls = collectShellCallArgDecls(plan.dacExpr, paramCount);
    std::unordered_map<const clang::ValueDecl*, int> argDeclIndices;
    for (int paramIdx = 0; paramIdx < paramCount; ++paramIdx) {
        if (argDecls[static_cast<std::size_t>(paramIdx)]) {
            argDeclIndices.emplace(argDecls[static_cast<std::size_t>(paramIdx)], paramIdx);
        }
    }

    std::vector<std::set<const clang::ValueDecl*>> trackedDecls(
        static_cast<std::size_t>(paramCount));
    std::unordered_map<const clang::ValueDecl*, int> trackedDeclIndices;
    for (int paramIdx = 0; paramIdx < paramCount; ++paramIdx) {
        const clang::ValueDecl* argDecl = argDecls[static_cast<std::size_t>(paramIdx)];
        if (!argDecl) {
            continue;
        }

        std::set<const clang::ValueDecl*> visitedDecls;
        collectTrackedValueDecls(
            argDecl,
            trackedDecls[static_cast<std::size_t>(paramIdx)],
            visitedDecls);
        for (const clang::ValueDecl* tracked : trackedDecls[static_cast<std::size_t>(paramIdx)]) {
            trackedDeclIndices.emplace(tracked, paramIdx);
        }
    }

    for (int paramIdx = 0; paramIdx < paramCount; ++paramIdx) {
        bool needsInitCopy = policy.needsInitCopy[static_cast<std::size_t>(paramIdx)];
        bool writtenInRegion = policy.writtenInRegion[static_cast<std::size_t>(paramIdx)];
        applyAccessToTransferState(
            summarizeMainKernelAccess(
                effectiveParamModes[static_cast<std::size_t>(paramIdx)]),
            needsInitCopy,
            writtenInRegion);
        policy.needsInitCopy[static_cast<std::size_t>(paramIdx)] = needsInitCopy;
        policy.writtenInRegion[static_cast<std::size_t>(paramIdx)] = writtenInRegion;
    }

    for (const clang::Stmt* siblingStmt : plan.siblingStmts) {
        const auto siblingSummary =
            summarizeStmtAccess(siblingStmt, argDeclIndices, paramCount);
        for (int paramIdx = 0; paramIdx < paramCount; ++paramIdx) {
            bool needsInitCopy = policy.needsInitCopy[static_cast<std::size_t>(paramIdx)];
            bool writtenInRegion = policy.writtenInRegion[static_cast<std::size_t>(paramIdx)];
            applyAccessToTransferState(
                siblingSummary[static_cast<std::size_t>(paramIdx)],
                needsInitCopy,
                writtenInRegion);
            policy.needsInitCopy[static_cast<std::size_t>(paramIdx)] = needsInitCopy;
            policy.writtenInRegion[static_cast<std::size_t>(paramIdx)] = writtenInRegion;
        }
    }

    const clang::Stmt* functionBody =
        plan.parentFunction ? plan.parentFunction->getBody() : nullptr;
    const clang::CompoundStmt* containingCompound =
        findContainingCompoundStmt(functionBody, plan.outerFor);
    const auto postRegionStmts =
        collectStatementsAfterTarget(containingCompound, plan.outerFor);
    for (const clang::Stmt* stmt : postRegionStmts) {
        const auto hostSummary =
            summarizeStmtAccess(stmt, trackedDeclIndices, paramCount);
        for (int paramIdx = 0; paramIdx < paramCount; ++paramIdx) {
            if (hostSummary[static_cast<std::size_t>(paramIdx)].reads ||
                hostSummary[static_cast<std::size_t>(paramIdx)].writes) {
                policy.hostUsedAfterLoop[static_cast<std::size_t>(paramIdx)] = true;
            }
        }
    }

    for (int paramIdx = 0; paramIdx < paramCount; ++paramIdx) {
        const bool hasSimpleArgDecl = argDecls[static_cast<std::size_t>(paramIdx)] != nullptr;
        const bool canElideDeadLocalSync =
            hasSimpleArgDecl &&
            isPureLocalTrackedDeclSet(trackedDecls[static_cast<std::size_t>(paramIdx)]);

        if (!hasSimpleArgDecl) {
            policy.needsInitCopy[static_cast<std::size_t>(paramIdx)] = true;
        }

        policy.needsSyncCopy[static_cast<std::size_t>(paramIdx)] =
            policy.writtenInRegion[static_cast<std::size_t>(paramIdx)] &&
            (!canElideDeadLocalSync ||
             policy.hostUsedAfterLoop[static_cast<std::size_t>(paramIdx)]);
    }

    return policy;
}

std::string getExprSourceText(const clang::Expr* expr,
                              const clang::SourceManager& SM,
                              const clang::LangOptions& LO) {
    if (!expr) {
        return "";
    }
    return clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(expr->getSourceRange()),
        SM, LO).str();
}

std::string joinShellCallArgs(const clang::BinaryOperator* dacExpr,
                              clang::ASTContext* context) {
    if (!dacExpr || !context) {
        return "";
    }

    clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    auto* shellCall = dacppTranslator::getNode<clang::CallExpr>(shellExpr);
    if (!shellCall) {
        return "";
    }

    const auto& SM = context->getSourceManager();
    const auto& LO = context->getLangOpts();
    std::string args;
    for (unsigned argIdx = 0; argIdx < shellCall->getNumArgs(); ++argIdx) {
        if (!args.empty()) {
            args += ", ";
        }
        args += getExprSourceText(shellCall->getArg(argIdx), SM, LO);
    }
    return args;
}

std::vector<SplitRecord> collectUniqueSplits(dacppTranslator::Shell* shell) {
    std::vector<SplitRecord> records;
    std::set<std::string> seen;

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        auto* shellParam = shell->getShellParam(paramIdx);
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            auto* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void" || !seen.insert(split->getId()).second) {
                continue;
            }

            SplitRecord record;
            record.id = split->getId();
            record.type = split->type;
            record.dimId = split->getDimIdx();
            record.infoName = shellParam->getName();
            if (split->type == "RegularSplit") {
                auto* regular = static_cast<dacppTranslator::RegularSplit*>(split);
                record.splitSize = regular->getSplitSize();
                record.splitStride = regular->getSplitStride();
            }
            records.push_back(record);
        }
    }

    return records;
}

std::string buildRegionShellParamSignature(dacppTranslator::Shell* shell) {
    std::string params;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        if (!params.empty()) {
            params += ", ";
        }
        params += param->getType() + " " + param->getName();
    }
    return params;
}

std::string buildRegionBufferAliases(dacppTranslator::Shell* shell,
                                     const std::vector<dacppTranslator::IOTYPE>& modes,
                                     bool useEffectiveModes) {
    std::string code;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        dacppTranslator::IOTYPE mode =
            useEffectiveModes ? modes[paramIdx] : param->getRw();
        if (mode == dacppTranslator::IOTYPE::READ) {
            code += "    auto& r_" + param->getName() + " = *ctx.r_" +
                    param->getName() + ";\n";
        } else {
            code += "    auto* r_" + param->getName() + " = ctx.r_" +
                    param->getName() + ".get();\n";
        }
    }
    return code;
}

std::string buildRegionShapeAliases(dacppTranslator::Shell* shell) {
    std::string code;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        code += "    auto* info_" + param->getName() + "_Shape = ctx.info_" +
                param->getName() + "_Shape.data();\n";
    }
    return code;
}

std::string buildRegionPartitionAliases(dacppTranslator::Shell* shell) {
    std::string code;
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        code += "    auto& info_partition_" + param->getName() +
                "_buffer = *ctx.info_partition_" + param->getName() +
                "_buffer;\n";
    }
    return code;
}

std::string buildRegionSplitAliases(const std::vector<SplitRecord>& splitRecords) {
    std::string code;
    for (const auto& split : splitRecords) {
        code += "    auto& " + split.id + " = ctx." + split.id + ";\n";
    }
    return code;
}

BufferRegionGeneratedCode buildOptimizedBufferRegionCode(
    dacppTranslator::DacppFile* dacppFile,
    dacppTranslator::Expression* expr,
    const std::vector<dacppTranslator::IOTYPE>& effectiveParamModes,
    const std::vector<dacppTranslator::BINDINFO>& bindInfo,
    const std::string& bindingInit,
    const std::string& accessorInit,
    const std::string& accessorList,
    const std::string& accessorPointerList,
    const std::string& getpos,
    const std::string& calcEmbed,
    const BufferRegionTransferPolicy& transferPolicy) {
    BufferRegionGeneratedCode generated;

    auto* shell = expr->getShell();
    auto* calc = expr->getCalc();
    const auto& plan = dacppFile->getBufferRegionPlan();
    const std::string baseName = shell->getName() + "_" + calc->getName();
    const auto splitRecords = collectUniqueSplits(shell);

    generated.ctxTypeName = "__dacpp_ctx_" + baseName;
    generated.ctxVarName = generated.ctxTypeName + "_0";
    generated.initName = "__dacpp_init_" + baseName;
    generated.submitName = "__dacpp_submit_" + baseName;
    generated.syncName = "__dacpp_sync_" + baseName;

    std::string code;
    code += "struct " + generated.ctxTypeName + " {\n";
    code += "    sycl::queue dacpp_q{sycl::default_selector_v, {sycl::property::queue::in_order()}};\n";
    code += "    ParameterGeneration para_gene_tool;\n";
    code += "    int Item_Size = 0;\n";
    code += "    int dim_x = 1;\n";
    code += "    int dim_y = 1;\n";
    code += "    int local_x = 1;\n";
    code += "    int local_y = 1;\n";
    code += "    int global_x = 1;\n";
    code += "    int global_y = 1;\n";
    for (const auto& split : splitRecords) {
        if (split.type == "IndexSplit") {
            code += "    Index " + split.id + ";\n";
        } else {
            code += "    RegularSlice " + split.id + ";\n";
        }
    }
    code += "    Dac_Ops In_Ops;\n";
    code += "    Dac_Ops Out_Ops;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        auto* shellParam = shell->getShellParam(paramIdx);
        const std::string& name = param->getName();
        code += "    DataInfo info_" + name + ";\n";
        code += "    std::vector<int> info_" + name + "_Shape;\n";
        code += "    Dac_Ops " + name + "_Ops;\n";
        code += "    std::vector<" + shellParam->getBasicType() + "> h_" + name + ";\n";
        code += "    std::unique_ptr<sycl::buffer<" + shellParam->getBasicType() +
                ", 1>> r_" + name + ";\n";
        code += "    std::vector<int> info_partition_" + name + ";\n";
        code += "    std::unique_ptr<sycl::buffer<int, 1>> info_partition_" + name +
                "_buffer;\n";
    }
    code += "};\n\n";

    const std::string shellParamSignature = buildRegionShellParamSignature(shell);
    code += "void " + generated.initName + "(" + generated.ctxTypeName + "& ctx";
    if (!shellParamSignature.empty()) {
        code += ", " + shellParamSignature;
    }
    code += ") {\n";
    code += "    ctx.dacpp_q = sycl::queue(sycl::default_selector_v, {sycl::property::queue::in_order()});\n";
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        const std::string& name = param->getName();
        code += "    ctx.info_" + name + " = DataInfo{};\n";
        code += "    ctx.info_" + name + ".dim = " + name + ".getDim();\n";
        code += "    ctx.info_" + name + ".dimLength.clear();\n";
        code += "    ctx.info_" + name + "_Shape.assign(ctx.info_" + name +
                ".dim, 0);\n";
        code += "    for (int dimIdx = 0; dimIdx < ctx.info_" + name +
                ".dim; ++dimIdx) {\n";
        code += "        ctx.info_" + name + ".dimLength.push_back(" + name +
                ".getShape(dimIdx));\n";
        code += "        ctx.info_" + name + "_Shape[dimIdx] = " + name +
                ".getShape(dimIdx);\n";
        code += "    }\n";
    }
    for (const auto& split : splitRecords) {
        if (split.type == "IndexSplit") {
            code += "    ctx." + split.id + " = Index(\"" + split.id + "\");\n";
        } else {
            code += "    ctx." + split.id + " = RegularSlice(\"" + split.id + "\", " +
                    std::to_string(split.splitSize) + ", " +
                    std::to_string(split.splitStride) + ");\n";
        }
        code += "    ctx." + split.id + ".setDimId(" + std::to_string(split.dimId) +
                ");\n";
        code += "    ctx." + split.id +
                ".SetSplitSize(ctx.para_gene_tool.init_operetor_splitnumber(ctx." +
                split.id + ", ctx.info_" + split.infoName + "));\n";
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        auto* shellParam = shell->getShellParam(paramIdx);
        const std::string& name = param->getName();
        code += "    ctx." + name + "_Ops.clear();\n";
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            auto* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }
            code += "    ctx." + split->getId() + ".setDimId(" +
                    std::to_string(split->getDimIdx()) + ");\n";
            code += "    ctx." + name + "_Ops.push_back(ctx." + split->getId() + ");\n";
        }
    }
    code += "    ctx.In_Ops.clear();\n";
    code += "    ctx.Out_Ops.clear();\n";
    std::set<std::string> seenIn;
    std::set<std::string> seenOut;
    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        auto* shellParam = shell->getShellParam(paramIdx);
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            auto* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }

            std::string componentSet = "0";
            for (const auto& entry : bindInfo) {
                if (shell->search_symbol(entry.v)->getId() == split->getId()) {
                    componentSet = std::to_string(entry.icls);
                    break;
                }
            }

            if (effectiveParamModes[paramIdx] == dacppTranslator::IOTYPE::WRITE) {
                if (seenOut.insert(componentSet).second) {
                    code += "    ctx." + split->getId() + ".setDimId(" +
                            std::to_string(split->getDimIdx()) + ");\n";
                    code += "    ctx.Out_Ops.push_back(ctx." + split->getId() + ");\n";
                }
            } else if (seenIn.insert(componentSet).second) {
                code += "    ctx." + split->getId() + ".setDimId(" +
                        std::to_string(split->getDimIdx()) + ");\n";
                code += "    ctx.In_Ops.push_back(ctx." + split->getId() + ");\n";
            }
        }
    }
    code += "    ctx.Item_Size = ctx.para_gene_tool.init_work_item_size(ctx.In_Ops);\n";
    code += "    sycl::device device = ctx.dacpp_q.get_device();\n";
    code += "    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();\n";
    code += "    ctx.dim_x = static_cast<int>(sycl::ceil(sycl::sqrt(static_cast<float>(ctx.Item_Size))));\n";
    code += "    ctx.dim_y = static_cast<int>(sycl::ceil(static_cast<float>(ctx.Item_Size) / ctx.dim_x));\n";
    code += "    ctx.local_x = std::min(16, static_cast<int>(max_sizes[0]));\n";
    code += "    ctx.local_y = std::min(16, static_cast<int>(max_sizes[1]));\n";
    code += "    ctx.global_x = ((ctx.dim_x + ctx.local_x - 1) / ctx.local_x) * ctx.local_x;\n";
    code += "    ctx.global_y = ((ctx.dim_y + ctx.local_y - 1) / ctx.local_y) * ctx.local_y;\n";
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        auto* shellParam = shell->getShellParam(paramIdx);
        const std::string& name = param->getName();
        const std::string& type = shellParam->getBasicType();
        code += "    ctx.h_" + name + ".clear();\n";
        if (paramIdx < static_cast<int>(transferPolicy.needsInitCopy.size()) &&
            transferPolicy.needsInitCopy[static_cast<std::size_t>(paramIdx)]) {
            code += "    " + name + ".tensor2Array(ctx.h_" + name + ");\n";
            code += "    ctx.r_" + name + " = std::make_unique<sycl::buffer<" + type +
                    ", 1>>(ctx.h_" + name + ".data(), sycl::range<1>(" + name +
                    ".getSize()));\n";
            if (effectiveParamModes[paramIdx] == dacppTranslator::IOTYPE::READ) {
                code += "    ctx.r_" + name + "->set_final_data(nullptr);\n";
            }
        } else {
            code += "    ctx.r_" + name + " = std::make_unique<sycl::buffer<" + type +
                    ", 1>>(sycl::range<1>(" + name + ".getSize()));\n";
        }
        code += "    ctx.info_partition_" + name +
                " = ctx.para_gene_tool.init_partition_data_shape(ctx.info_" + name +
                ", ctx." + name + "_Ops);\n";
        code += "    ctx.info_partition_" + name +
                "_buffer = std::make_unique<sycl::buffer<int, 1>>(ctx.info_partition_" +
                name + ".data(), sycl::range<1>(ctx.info_partition_" + name +
                ".size()));\n";
    }
    code += "}\n\n";

    code += "void " + generated.submitName + "(" + generated.ctxTypeName + "& ctx) {\n";
    code += "    using namespace sycl;\n";
    code += "    auto& dacpp_q = ctx.dacpp_q;\n";
    code += buildRegionSplitAliases(splitRecords);
    code += buildRegionBufferAliases(shell, effectiveParamModes, true);
    code += buildRegionPartitionAliases(shell);
    code += buildRegionShapeAliases(shell);
    code += "    const int Item_Size = ctx.Item_Size;\n";
    code += "    sycl::range<2> local(ctx.local_x, ctx.local_y);\n";
    code += "    sycl::range<2> global(ctx.global_x, ctx.global_y);\n";
    code += "    dacpp_q.submit([&](handler &h) {\n";
    code += accessorList;
    code += accessorInit;
    code += "        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {\n";
    code += "            int gx = item.get_global_id(0);\n";
    code += "            int gy = item.get_global_id(1);\n";
    code += "            int item_id = gx * global[1] + gy;\n";
    code += "            if(item_id >= Item_Size)\n";
    code += "                return;\n";
    code += "            // 索引初始化\n";
    code += bindingInit;
    code += "            // 获得划分数据单元左上角（第一个元素）的位置\n";
    code += getpos;
    code += "            // 获得accessor指针\n";
    code += accessorPointerList;
    code += "            // 嵌入计算\n";
    code += calcEmbed;
    code += "        });\n";
    code += "    });\n";
    code += "}\n\n";

    for (std::size_t helperIdx = 0; helperIdx < plan.siblingStmts.size(); ++helperIdx) {
        const std::string helperName = "__dacpp_submit_region_" + baseName +
                                       "_stmt_" + std::to_string(helperIdx);
        generated.siblingHelpers.emplace_back(plan.siblingStmts[helperIdx], helperName);
        code += "void " + helperName + "(" + generated.ctxTypeName + "& ctx) {\n";
        code += "    using namespace sycl;\n";
        code += "    auto& dacpp_q = ctx.dacpp_q;\n";
        code += buildRegionBufferAliases(shell, effectiveParamModes, false);
        code += buildRegionShapeAliases(shell);
        const auto* siblingFor = llvm::dyn_cast_or_null<clang::ForStmt>(plan.siblingStmts[helperIdx]);
        if (siblingFor) {
            code += BUFFER_TEMPLATE::parallelizeSingleFor(
                siblingFor, dacppFile->getContext(), dacppFile);
        } else {
            // Non-for-loop sibling: emit source text directly
            const auto* siblingStmt = plan.siblingStmts[helperIdx];
            if (siblingStmt && dacppFile->getContext()) {
                std::string stmtText = clang::Lexer::getSourceText(
                    clang::CharSourceRange::getTokenRange(siblingStmt->getSourceRange()),
                    dacppFile->getContext()->getSourceManager(),
                    dacppFile->getContext()->getLangOpts()).str();
                code += "    " + stmtText + "\n";
            }
        }
        code += "}\n\n";
    }

    code += "void " + generated.syncName + "(" + generated.ctxTypeName + "& ctx";
    if (!shellParamSignature.empty()) {
        code += ", " + shellParamSignature;
    }
    code += ") {\n";
    code += "    using namespace sycl;\n";
    code += "    ctx.dacpp_q.wait();\n";
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        auto* param = shell->getParam(paramIdx);
        const std::string& name = param->getName();
        if (paramIdx >= static_cast<int>(transferPolicy.needsSyncCopy.size()) ||
            !transferPolicy.needsSyncCopy[static_cast<std::size_t>(paramIdx)]) {
            continue;
        }
        code += "    ctx.h_" + name + ".resize(" + name + ".getSize());\n";
        code += "    {\n";
        code += "        host_accessor acc(*ctx.r_" + name + ");\n";
        code += "        for (std::size_t idx = 0; idx < ctx.h_" + name +
                ".size(); ++idx) {\n";
        code += "            ctx.h_" + name + "[idx] = acc[idx];\n";
        code += "        }\n";
        code += "    }\n";
        code += "    " + name + ".array2Tensor(ctx.h_" + name + ");\n";
    }
    code += "}\n\n";

    generated.definitions = code;
    return generated;
}

}  // namespace

void dacppTranslator::Rewriter::rewriteDac_Buffer() {
    // std::cout << "软编码BUFFER版本翻译" << std::endl;

    std::string code = "";
    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    BufferRegionGeneratedCode regionGenerated;
    bool regionApplied = false;
    
    // 添加头文件
    for(int i = 0; i < dacppFile->getNumHeaderFile(); i++) {
        code += "#include " + dacppFile->getHeaderFile(i)->getName() + "\n";
    }
    code += "\n";
    
    // 添加命名空间
    for(int i = 0; i < dacppFile->getNumNameSpace(); i++) {
        code += "using namespace " + dacppFile->getNameSpace(i)->getName() + ";\n";
    }
    code += "\n";
    
    // 添加数据关联表达式对应的划分结构和计算结构
    for(int i = 0; i < dacppFile->getNumExpression(); i++) {
        Expression* expr = dacppFile->getExpression(i);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        const auto effectiveParamModes = inferEffectiveBufferParamModes(shell, calc);
        

        std::vector<BINDINFO> info;
        shell->GetBindInfo(&info);//获取binding信息
         Dac_Ops ops;
        std::vector<std::string> sets;  // 存储每个节点所属的连通分量的ID
        std::vector<std::string> bindoffset;  // 存储每个节点相对于其连通分量代表节点的偏移
        int componentID = 1;                               // 连通分量的编号
        // std::string s;                                     // 用于存储 GetBindInfo 的偏移量字符串
        for(int i = 0; i < info.size(); i++){
            // std::cout << shell->search_symbol(info[i].v)->getId() << std::endl;
            // std::cout << info[i].icls << std::endl;
            if(info[i].offset.empty())
                info[i].offset = "0"; 
            // std::cout << info[i].offset << std::endl;
        }
        for(int i = 0; i < info.size(); i++){
            if(shell->search_symbol(info[i].v)->type.compare("IndexSplit") == 0) {
                dacppTranslator::IndexSplit* index = static_cast<dacppTranslator::IndexSplit*>(shell->search_symbol(info[i].v));
                Index tmp = Index(index->getId());
                tmp.SetSplitSize(index->getSplitNumber());
                tmp.setDimId(index->getDimIdx());
                ops.push_back(tmp);
                sets.push_back("id"+std::to_string(info[i].icls));
                bindoffset.push_back(info[i].offset);
            }else if(shell->search_symbol(info[i].v)->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit* r = static_cast<dacppTranslator::RegularSplit*>(shell->search_symbol(info[i].v));
                RegularSlice tmp = RegularSlice(r->getId(), r->getSplitSize(), r->getSplitStride());
                //std::cout << r->getId() << " " << r->getSplitSize() << " " << r->getSplitStride() << std::endl;
                tmp.SetSplitSize(r->getSplitNumber());
                tmp.setDimId(r->getDimIdx());
                ops.push_back(tmp);
                sets.push_back("id"+std::to_string(info[i].icls));
                bindoffset.push_back(info[i].offset);
            }
        }
        // 计算结构
        code += "void " + calc->getName() + "(";
        for(int count = 0; count < calc->getNumParams(); count++) {
            if(effectiveParamModes[count] != IOTYPE::READ)code += calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName() + ",";
            else code += "const " + calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName() + ",";
        }
          for(int count = 0;count < shell->getNumShellParams(); count ++){
            for(int i = 0;i < shell->getShellParam(count)->getDimension();i++){
               code += "int " + calc->getParam(count)->getName()+"_"+std::to_string(i)+",";
            }
        }
          for(int count = 0;count < shell->getNumShellParams(); count ++){
            for(int i = 0;i < shell->getShellParam(count)->getDimension();i++){
               code += "int " + calc->getParam(count)->getName()+"_"+std::to_string(i)+"_shape,";
            }
        }
        for(int count = 0; count < calc->getNumParams(); count++) {
            code += "sycl::accessor<int, 1, sycl::access::mode::read> info_" + calc->getParam(count)->getName() + "_acc";
            if(count != calc->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ") ";
        std::vector<dacppTranslator::clacparam> calcParams;
        calcParams.reserve(shell->getNumShellParams());
        for (int i = 0; i < shell->getNumShellParams(); i++) {
            clacparam temp;
            temp.name = calc->getParam(i)->getName();
            temp.dimesion = shell->getShellParam(i)->getDimension();

            for (int count = 0; count < shell->getShellParam(i)->getNumSplit(); count++) {
                Split *a = shell->getShellParam(i)->getSplit(count);
                temp.dimid.push_back(a->getDimIdx());
                temp.flag.push_back(a->type == "IndexSplit" ? 1 : 0);
            }

            calcParams.push_back(temp);
        }
        for(int count = 0; count < calc->getNumBody(); count++) {
            code += calc->getBody(count, calcParams) + "\n";
        }


        // std::cout << code;

        std::string dacShellName = shell->getName() + "_" + calc->getName();
        // std::string dacShellParams = "";
        // for (int count = 0; count < shell->getNumParams(); count++) {
        //     Param* param = shell->getParam(count);
        //     dacShellParams += param->getType() + " " + param->getName();
        //     if(count != shell->getNumParams() - 1) { 
        //         dacShellParams += ", ";
        //     }
        // }
        std::string dacShellParams = "";



//------------------------------------
// 1. 原有 shell 参数
//------------------------------------
for (int count = 0; count < shell->getNumParams(); count++) {
    Param* param = shell->getParam(count);
    dacShellParams += param->getType() + " " + param->getName();
    if (count != shell->getNumParams() - 1) {
        dacShellParams += ", ";
    }
}

//------------------------------------
// 2. 追加 forStatementVars（排除 shellVars）
//------------------------------------
if(dacppFile->mode!=1){
bool needComma = (shell->getNumParams() > 0);
auto Vars=dacppFile->getForStatementVars();
for (const auto &var : Vars) {
    const std::string &type = var.second;
    const std::string &name = var.first;

    // 检查是否在 shellVars 中
    bool inShellVars = false;
    for (const auto &shellVar : dacppFile->shellVars) {
        if (shellVar.first == name) {
            inShellVars = true;
            break;
        }
    }
    if (inShellVars) continue;

    if (needComma) {
        dacShellParams += ", ";
    }

    dacShellParams += type + " " + name;
    needComma = true;
}
}

        //算子初始化
        std::string opInit = "";
        std::string infoInit = "";
        std::set<std::string> HadInit;
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){//遍历每个参数
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            //infoInit += BUFFER_TEMPLATE::CodeGen_DataInfoInit(shellParam->getName());
            infoInit += BUFFER_TEMPLATE::CodeGen_DataInfoInit(shellParam->getName(), std::to_string(shellParam->getDimension()));
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){//遍历每个参数的划分
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}//保形算子会被跳过
                Split* split = shellParam->getSplit(NumSplit);
                if(split->type.compare("IndexSplit") == 0){
                    if(HadInit.count(split->getId()) == 1){//已经初始化过了
                        continue;
                    }
                    IndexSplit* indexSplit = static_cast<IndexSplit*>(split);
                    opInit += BUFFER_TEMPLATE::CodeGen_IndexInit2(indexSplit->getId(),std::to_string(NumSplit),"info_"+shellParam->getName());//初始化降维算子
                    HadInit.insert(indexSplit->getId());

                }
                else if(split->type.compare("RegularSplit") == 0){
                    if(HadInit.count(split->getId()) == 1){//已经初始化过了
                        continue;
                    }
                    RegularSplit* regularSplit = static_cast<RegularSplit*>(split);
                    opInit += BUFFER_TEMPLATE::CodeGen_RegularSliceInit2(regularSplit->getId(),std::to_string(regularSplit->getSplitSize()),
                    std::to_string(regularSplit->getSplitStride()),std::to_string(NumSplit),"info_"+shellParam->getName());//初始化规则分区算子
                    HadInit.insert(regularSplit->getId());
                }
            }
        }
        opInit = infoInit + opInit;
        // std::cout << opInit;
        //获取算子与作用数据块的关系
        std::string add2Op = "";
        std::string dataOpsInit = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                add2Op += BUFFER_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(NumSplit),shellParam->getName()+"_Ops");//将算子添加到他作用的参数中去
            }
            dataOpsInit += BUFFER_TEMPLATE::CodeGen_DataOpsInit2(shellParam->getName()+"_Ops",add2Op);
            add2Op = "";
        }

        //划分输入输出算子
        std::string add2Op_inops = "";
        std::string add2Op_outops = "";
        //std::string add2Op_reductions = "";
        std::set<std::string> setIn;
        std::set<std::string> setOut;
        Dac_Ops Inops;
        std::string set;
        int inflag = 0;
        int outflag = 0;
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                if(effectiveParamModes[NumShellParam] == IOTYPE::WRITE){
                    for(int i = 0; i < info.size(); i++){
                        if(shell->search_symbol(info[i].v)->getId() == split->getId())
                                set = std::to_string(info[i].icls);
                    }
                    if(setOut.count(set) == 1) {
                        continue;
                    }
                    add2Op_outops += BUFFER_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(outflag),"Out_Ops");//将算子添加到输出算子组中去
                    //add2Op_reductions += BUFFER_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(outflag),"Reduction_Ops");//将算子添加到归约算子组中去
                    outflag++;
                    setOut.insert(set);
                }
                else{
                    for(int i = 0; i < info.size(); i++){
                        if(shell->search_symbol(info[i].v)->getId() == split->getId())
                            set = std::to_string(info[i].icls);
                    }
                    if(setIn.count(set) == 1) {
                        continue;
                    }
                    add2Op_inops += BUFFER_TEMPLATE::CodeGen_AddOp2Ops(split->getId(),std::to_string(inflag),"In_Ops");//将算子添加到输入算子组中去
                    Dac_Op op = Dac_Op(split->getId(),0,inflag);
                    inflag++;
                    // std::cout<<inflag<<std::endl;
                    Inops.push_back(op);
                    setIn.insert(set);
                }
            }
        }
        std::string dataOpsInit_inops = BUFFER_TEMPLATE::CodeGen_DataOpsInit2("In_Ops",add2Op_inops);
        std::string dataOpsInit_outops = BUFFER_TEMPLATE::CodeGen_DataOpsInit2("Out_Ops",add2Op_outops);
        //std::string dataOpsInit_reductions = BUFFER_TEMPLATE::CodeGen_DataOpsInit2("Reduction_Ops",add2Op_reductions);
        // std::cout << dataOpsInit_inops;
        // std::cout << dataOpsInit_outops;
        // std::cout << dataOpsInit_reductions;

        //生成设备内存分配大小
        std::string divice_memory = "";
   
        // std::cout << divice_memory;

        //生成算子的划分长度
        std::string splitLength = "";

        // std::cout << splitLength;

        //AddDacOps2Vector
        std::string AddDacOps2Vector = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(effectiveParamModes[NumShellParam] != IOTYPE::READ){
                AddDacOps2Vector += BUFFER_TEMPLATE::CodeGen_Add_DacOps2Vector("ops_s","In_Ops");
            }
            else{
                AddDacOps2Vector += BUFFER_TEMPLATE::CodeGen_Add_DacOps2Vector("ops_s",shellParam->getName()+"_Ops");
            }
        }
        std::string DeclareDacOpsVector = BUFFER_TEMPLATE::CodeGen_Declare_DacOps_Vector("ops_s",AddDacOps2Vector);
        // std::cout << std::to_string(shell->getNumShellParams()) << std::to_string(inflag) << std::endl;
        //std::string InitSplitLengthMatrix = BUFFER_TEMPLATE::CodeGen_Init_Split_Length_Matrix(DeclareDacOpsVector,std::to_string(shell->getNumShellParams()),std::to_string(inflag),"ops_s");

        //计算工作分配数量
        std::string item_number = BUFFER_TEMPLATE::CodeGen_Init_Work_Item_Number("Item_Size","In_Ops");
        // std::cout << item_number;

        //std::string InitOPS =  dataOpsInit + dataOpsInit_inops + dataOpsInit_outops + dataOpsInit_reductions;
        std::string InitOPS =  dataOpsInit + dataOpsInit_inops + dataOpsInit_outops;
        //生成归约中Split_size中的大小
        // std::string Init_Reduction_Split_Size = BUFFER_TEMPLATE::CodeGen_Init_Reduction_Split_Size("Reduction_Split_Size","In_Ops","Out_Ops");
        //std::cout << Init_Reduction_Split_Size;

        //生成归约中Split_length大小
        // std::string Init_Reduction_Split_Length = BUFFER_TEMPLATE::CodeGen_Init_Reduction_Split_Length("Reduction_Split_Length","Out_Ops");

        //std::string ParameterGenerate = BUFFER_TEMPLATE::CodeGen_ParameterGenerate(InitOPS,divice_memory,splitLength,InitSplitLengthMatrix,item_number); 
        std::string ParameterGenerate = BUFFER_TEMPLATE::CodeGen_ParameterGenerate(InitOPS,divice_memory,splitLength,item_number); 
        // std::cout << ParameterGenerate;
        //设置内存分配
        std::string deviceMemAlloc = "";


        //主机数据移动至设备
        std::string H2DMemMove = "";
        // for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
        //     ShellParam* shellParam = shell->getShellParam(NumShellParam);
        //     H2DMemMove += BUFFER_TEMPLATE::CodeGen_DeviceDataInit(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");
        //     if(shellParam->getRw() == 0){
        //         H2DMemMove += BUFFER_TEMPLATE::CodeGen_H2DMemMov(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");
        //     }
        // }
        // std::cout << H2DMemMove;

       
        std::string BindingInit = BUFFER_TEMPLATE::CodeGen_IndexInit2(ops,sets,bindoffset);

        // 嵌入计算
        Args args = Args();
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            Dac_Ops ops;
            if(effectiveParamModes[NumShellParam] == IOTYPE::READ){
                for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                    if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                    Dac_Op op = Dac_Op(shellParam->getSplit(NumSplit)->getId(), 0, NumSplit);
                    ops.push_back(op);
          
                }
            }else{
                for(int Countin = 0; Countin < Inops.size; Countin++){
                    ops.push_back(Inops[Countin]);
                }
            }
            DacData data = DacData("d_"+shellParam->getName(), 0, ops);
            args.push_back(data);
        }
        // std::string CalcEmbed = CodeGen_CalcEmbed2(calc->getName(),args);//注意调用了新的嵌入计算模板
        std::vector<std::string> splits;
        std::vector<int> splitNum;
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            splitNum.push_back(shellParam->getNumSplit());
            for (int y = 0; y < shellParam->getNumSplit(); y++) {
                splits.push_back(std::to_string(y));
            }
        }
        std::vector<std::string> accessor_names;
        for (int z = 0; z < shell->getNumParams(); z++) {
            accessor_names.push_back(shell->getParam(z)->getName());
        }
        // std::string CalcEmbed = BUFFER_TEMPLATE::CodeGen_CalcEmbed2(calc->getName(),args, accessor_names);
        std::string CalcEmbed = BUFFER_TEMPLATE::CodeGen_CalcEmbed2(calc->getName(), splits, splitNum, accessor_names);

        
        std::string AccessorInit = "";
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            AccessorInit += BUFFER_TEMPLATE::CodeGen_AccessorInit(shell->getShellParam(argCount)->getName());
        }
        std::string Accessor_List = "";
        // for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
        //     Accessor_List += BUFFER_TEMPLATE::CodeGen_AccessorInit0(shell->getShellParam(argCount)->getName());
        // }
        //2025.12.4：这里要进行一个小判断：对于只读的数据，将buffer的访问模式修改为只读 然后禁止数据写回操作
        //2025.12.4：对于只写的数据，访问模式设置为覆盖写 同时注意 写成这样*r_matC
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            if(effectiveParamModes[argCount] == IOTYPE::READ){
                Accessor_List += BUFFER_TEMPLATE::CodeGen_AccessorInit0_read(shell->getShellParam(argCount)->getName(), shell->getShellParam(argCount)->getBasicType());
            }
            else if(effectiveParamModes[argCount] == IOTYPE::WRITE){
                Accessor_List += BUFFER_TEMPLATE::CodeGen_AccessorInit0_write(shell->getShellParam(argCount)->getName(), shell->getShellParam(argCount)->getBasicType());
            }
            else if(effectiveParamModes[argCount] == IOTYPE::READ_WRITE){
                Accessor_List += BUFFER_TEMPLATE::CodeGen_AccessorInit0_read_write(shell->getShellParam(argCount)->getName(), shell->getShellParam(argCount)->getBasicType());
            }
        }
        std::string Accessor_Pointer_List = "";
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            Accessor_Pointer_List += BUFFER_TEMPLATE::CodeGen_AccessorInit1(shell->getShellParam(argCount)->getName());
        }
        std::string getpos = "";
        for (int argCount = 0; argCount < shell->getNumShellParams(); argCount++) {
            ShellParam* shellParam = shell->getShellParam(argCount);
             for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                Split* split = shellParam->getSplit(NumSplit);
                if(split->type.compare("RegularSplit") == 0 || split->type.compare("IndexSplit") == 0){
                    getpos += BUFFER_TEMPLATE::CodeGen_getpos1(shell->getShellParam(argCount)->getName(), split->getId(), std::to_string(NumSplit));
                }
                else 
                    getpos += BUFFER_TEMPLATE::CodeGen_getpos0(shell->getShellParam(argCount)->getName(), std::to_string(NumSplit));
            }  
        }
		    std::string KernelExecute;
	        // rewriteMain() now preserves the original outer host loop and only
	        // replaces the `<->` expression with the generated shell wrapper call.
	        // Re-emitting the whole outer loop body inside the wrapper would execute
	        // time-stepping logic twice (observed in stencil/waveEquation).
	        KernelExecute = BUFFER_TEMPLATE::CodeGen_KernelExecute("Item_Size",AccessorInit,BindingInit,getpos,Accessor_List,Accessor_Pointer_List,CalcEmbed);//注意这里面填的size的大小需要是前面算出来的大小
	        // std::cout << KernelExecute;

        std::string Reduction;
        std::string D2HMemMove;
        std::string ReductionRule = "sycl::plus<>()";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(effectiveParamModes[NumShellParam] != IOTYPE::READ){
                // Reduction += BUFFER_TEMPLATE::CodeGen_Reduction_Span(shellParam->getName()+"Reduction_Size","Reduction_Split_Size","Reduction_Split_Length",shellParam->getName(),shellParam->getBasicType(),ReductionRule);
	            Reduction += BUFFER_TEMPLATE::CodeGen_Result_B2H_Mov(shellParam->getName(),shellParam->getName()+"_Size");
            }
        }
        // std::cout << Reduction;
        // std::cout << D2HMemMove;
       
        std::string dataRecon = "";
        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            if(effectiveParamModes[NumShellParam] == IOTYPE::WRITE){
                dataRecon += BUFFER_TEMPLATE::CodeGen_Init_Host_Memory(shellParam->getBasicType(),shellParam->getName());
            }else if(effectiveParamModes[NumShellParam] == IOTYPE::READ){
                dataRecon += BUFFER_TEMPLATE::CodeGen_D2B_Mov_Buffer(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");}
            else if(effectiveParamModes[NumShellParam] == IOTYPE::READ_WRITE){
                 dataRecon += BUFFER_TEMPLATE::CodeGen_D2B_Mov_Buffer_read_write(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size");}
            }

        //dataRecon
        //std::string dataRecon = "";
       for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++){
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            std::string opPushBack = "";
            for(int NumSplit = 0; NumSplit < shellParam->getNumSplit(); NumSplit++){
                if(shellParam->getSplit(NumSplit)->getId() == "void") { continue;}
                Split* split = shellParam->getSplit(NumSplit);
                opPushBack += BUFFER_TEMPLATE::CodeGen_OpPushBack2Ops(shellParam->getName(),split->getId(),std::to_string(split->getDimIdx()));
            }
            std::string dataOpsInit = BUFFER_TEMPLATE::CodeGen_DataOpsInit(shellParam->getName(),opPushBack);
            // dataRecon += BUFFER_TEMPLATE::CodeGen_DataReconstruct(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size",dataOpsInit);
            if(effectiveParamModes[NumShellParam] != IOTYPE::READ){
                dataRecon += BUFFER_TEMPLATE::CodeGen_DataReconstruct1(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size",dataOpsInit);
            }
            else{
                dataRecon += BUFFER_TEMPLATE::CodeGen_DataReconstruct(shellParam->getBasicType(),shellParam->getName(),shellParam->getName()+"_Size",dataOpsInit);
            }
       }
        // std::cout << dataRecon;
        // std::cout << "-----------------------------------------------------------------------";
        // 内存释放
        std::string MemFree = "";
        // for(int j = 0; j < shell->getNumParams(); j++) {
        //     ShellParam* shellParam = shell->getShellParam(j);
        //     MemFree += BUFFER_TEMPLATE::CodeGen_MemFree(shellParam->getName());
        // }

        if (regionPlan.enabled && regionPlan.exprIndex == i) {
            const BufferRegionTransferPolicy transferPolicy =
                analyzeBufferRegionTransferPolicy(dacppFile, expr, effectiveParamModes);
            regionGenerated = buildOptimizedBufferRegionCode(
                dacppFile,
                expr,
                effectiveParamModes,
                info,
                BindingInit,
                AccessorInit,
                Accessor_List,
                Accessor_Pointer_List,
                getpos,
                CalcEmbed,
                transferPolicy);
            code += regionGenerated.definitions;
            code += "\n";
            regionApplied = true;
            rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
            rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
            continue;
        }

        std::string dac = BUFFER_TEMPLATE::CodeGen_DataAssocComp(dataRecon, H2DMemMove, KernelExecute, Reduction, D2HMemMove);
	    std::string res = BUFFER_TEMPLATE::CodeGen_DAC2SYCL2(
		dacShellName,
		dacShellParams,
        opInit,
        ParameterGenerate,
		deviceMemAlloc,
		dac);
        code += res;
        code += "\n\n";
        rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
        rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
        // std::cout << code;  
    }

    rewriter->InsertText(dacppFile->node->getBeginLoc(),code);

    if (regionApplied && regionPlan.enabled && regionPlan.outerFor && regionPlan.dacExpr) {
        dacppFile->setMainAlreadyRewritten(true);
        const std::string argText = joinShellCallArgs(regionPlan.dacExpr, dacppFile->getContext());
        std::string initInsert = "    " + regionGenerated.ctxTypeName + " " +
                                 regionGenerated.ctxVarName + ";\n";
        initInsert += "    " + regionGenerated.initName + "(" +
                      regionGenerated.ctxVarName;
        if (!argText.empty()) {
            initInsert += ", " + argText;
        }
        initInsert += ");\n";
        rewriter->InsertTextBefore(regionPlan.outerFor->getBeginLoc(), initInsert);

        rewriter->ReplaceText(
            regionPlan.dacExpr->getSourceRange(),
            regionGenerated.submitName + "(" + regionGenerated.ctxVarName + ")");

        for (const auto& helper : regionGenerated.siblingHelpers) {
            rewriter->ReplaceText(
                helper.first->getSourceRange(),
                helper.second + "(" + regionGenerated.ctxVarName + ");");
        }

        std::string syncInsert = "\n    " + regionGenerated.syncName + "(" +
                                 regionGenerated.ctxVarName;
        if (!argText.empty()) {
            syncInsert += ", " + argText;
        }
        syncInsert += ");\n";
        rewriter->InsertTextAfterToken(regionPlan.outerFor->getEndLoc(), syncInsert);
    }

    //rewriter->InsertText(dacppFile->getMainFuncLoc()->getBeginLoc(), code);
}
