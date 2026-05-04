#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <vector>

#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"

#include "Rewriter_MPI_Stencil_Common.h"
#include "Rewriter_MPI_PostRegion_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

const clang::CallExpr* getShellCallExpr(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr) {
        return nullptr;
    }

    clang::Expr* shellExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS()
                                                         : dacExpr->getRHS();
    return dacppTranslator::getNode<clang::CallExpr>(shellExpr);
}

std::string getStmtSourceText(const clang::Stmt* stmt,
                              clang::ASTContext* context) {
    if (!stmt || !context) {
        return "";
    }
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(stmt->getSourceRange()),
               context->getSourceManager(),
               context->getLangOpts())
        .str();
}

bool isWordBoundary(char c) {
    return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_');
}

bool containsWord(const std::string& text, const std::string& word) {
    if (word.empty() || text.empty() || word.size() > text.size()) {
        return false;
    }
    std::size_t pos = 0;
    while ((pos = text.find(word, pos)) != std::string::npos) {
        const bool leftOk = pos == 0 || isWordBoundary(text[pos - 1]);
        const std::size_t rightIdx = pos + word.size();
        const bool rightOk = rightIdx >= text.size() || isWordBoundary(text[rightIdx]);
        if (leftOk && rightOk) {
            return true;
        }
        ++pos;
    }
    return false;
}

std::string resolveActualTensorName(const std::string& shellParamName,
                                    const clang::BinaryOperator* dacExpr) {
    const clang::CallExpr* shellCall = getShellCallExpr(dacExpr);
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

bool isVectorParam(Shell* shell, Calc* calc, int paramIdx) {
    if (!shell || !calc || paramIdx < 0 ||
        paramIdx >= shell->getNumShellParams() ||
        paramIdx >= calc->getNumParams()) {
        return false;
    }
    const std::string shellType = shell->getParam(paramIdx)->getType();
    const std::string calcType = calc->getParam(paramIdx)->getType();
    return shellType.find("Vector<") != std::string::npos ||
           calcType.find("Vector<") != std::string::npos;
}

void collectRootBridgeTensors(DistributedStencilSitePlan& plan,
                              DacppFile* dacppFile,
                              Shell* shell) {
    if (!dacppFile || !shell || !dacppFile->getContext()) {
        return;
    }
    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
        const std::string stmtText = getStmtSourceText(stmt, dacppFile->getContext());
        for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
            const std::string paramName = shell->getParam(paramIdx)->getName();
            if (!containsWord(stmtText, paramName)) {
                continue;
            }
            plan.rootBridgeTensors.insert(paramName);
        }
    }
}

bool isEffectiveWriter(const std::string& tensorName,
                       Shell* shell,
                       const std::vector<IOTYPE>& transportModes) {
    if (!shell) {
        return false;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams() &&
                           paramIdx < static_cast<int>(transportModes.size());
         ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return transportModes[paramIdx] == IOTYPE::WRITE;
        }
    }
    return false;
}

bool isEffectiveReader(const std::string& tensorName,
                       Shell* shell,
                       const std::vector<IOTYPE>& transportModes) {
    if (!shell) {
        return false;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams() &&
                           paramIdx < static_cast<int>(transportModes.size());
         ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return transportModes[paramIdx] == IOTYPE::READ;
        }
    }
    return false;
}

int findShellParamIndex(Shell* shell, const std::string& tensorName) {
    if (!shell) {
        return -1;
    }
    for (int paramIdx = 0; paramIdx < shell->getNumParams(); ++paramIdx) {
        if (shell->getParam(paramIdx)->getName() == tensorName) {
            return paramIdx;
        }
    }
    return -1;
}

Split* findSingleNonVoidSplit(ShellParam* shellParam) {
    if (!shellParam) {
        return nullptr;
    }

    Split* result = nullptr;
    for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
        Split* split = shellParam->getSplit(splitIdx);
        if (!split || split->getId() == "void") {
            continue;
        }
        if (result) {
            return nullptr;
        }
        result = split;
    }
    return result;
}

bool isStrideOneInputDomainSplit(Split* split) {
    if (!split || split->getId() == "void") {
        return false;
    }
    if (split->type == "IndexSplit") {
        return true;
    }
    auto* regular = static_cast<RegularSplit*>(split);
    return regular && regular->getSplitStride() == 1;
}

bool writerRouteCoveredByInputDomain(
    Shell* shell,
    const std::vector<IOTYPE>& effectiveModes,
    int writerParamIndex,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta) {
    if (!shell || writerParamIndex < 0 ||
        writerParamIndex >= shell->getNumShellParams()) {
        return false;
    }

    Split* writerSplit =
        findSingleNonVoidSplit(shell->getShellParam(writerParamIndex));
    if (!writerSplit || writerSplit->type != "IndexSplit") {
        return false;
    }

    const auto writerMeta = splitMeta.find(writerSplit->getId());
    if (writerMeta == splitMeta.end()) {
        return false;
    }

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams() &&
                           paramIdx < static_cast<int>(effectiveModes.size());
         ++paramIdx) {
        if (effectiveModes[paramIdx] == IOTYPE::WRITE) {
            continue;
        }
        ShellParam* shellParam = shell->getShellParam(paramIdx);
        if (!shellParam) {
            continue;
        }
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); ++splitIdx) {
            Split* split = shellParam->getSplit(splitIdx);
            if (!split || split->getId() == "void") {
                continue;
            }
            const auto inputMeta = splitMeta.find(split->getId());
            if (inputMeta == splitMeta.end() ||
                inputMeta->second.bindId != writerMeta->second.bindId) {
                continue;
            }
            if (isStrideOneInputDomainSplit(split)) {
                return true;
            }
        }
    }
    return false;
}

std::string trim(std::string text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

bool isSupportedIncrement(const clang::ForStmt* forStmt) {
    const auto* inc = forStmt ? forStmt->getInc() : nullptr;
    if (!inc) {
        return false;
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(inc)) {
        return unary->isIncrementOp();
    }
    if (const auto* opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(inc)) {
        return opCall->getOperator() == clang::OO_PlusPlus;
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(inc)) {
        if (!binary->isAssignmentOp()) {
            return false;
        }
        const std::string text = binary->getOpcodeStr().str();
        return text == "+=" || text == "=";
    }
    return false;
}

struct RouteLoopInfo {
    std::string loopVar;
    const clang::VarDecl* loopVarDecl = nullptr;
};

bool extractRouteLoopInfo(const clang::ForStmt* forStmt,
                          clang::ASTContext* context,
                          const BufferRegionPlan& plan,
                          RouteLoopInfo& info) {
    if (!forStmt || !context || !plan.capturedNonShellVars.empty()) {
        return false;
    }

    const auto& sourceManager = context->getSourceManager();
    const auto& langOpts = context->getLangOpts();

    const auto* declStmt =
        llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
    if (!declStmt || !declStmt->isSingleDecl()) {
        return false;
    }
    const auto* loopVarDecl =
        llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl());
    if (!loopVarDecl || !loopVarDecl->getInit()) {
        return false;
    }
    info.loopVar = loopVarDecl->getNameAsString();
    info.loopVarDecl = loopVarDecl;

    const auto* cond =
        llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
    if (!cond) {
        return false;
    }
    const std::string lhsText =
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(cond->getLHS()->getSourceRange()),
            sourceManager, langOpts)
            .str();
    if (trim(lhsText) != info.loopVar ||
        (cond->getOpcode() != clang::BO_LE && cond->getOpcode() != clang::BO_LT)) {
        return false;
    }
    if (!isSupportedIncrement(forStmt)) {
        return false;
    }

    return true;
}

const clang::Expr* ignoreTransparentExpr(const clang::Expr* expr) {
    return expr ? expr->IgnoreParenImpCasts() : nullptr;
}

bool parseIntegerLiteralExpr(const clang::Expr* expr, int& value) {
    expr = ignoreTransparentExpr(expr);
    if (const auto* intLiteral = llvm::dyn_cast_or_null<clang::IntegerLiteral>(expr)) {
        value = static_cast<int>(intLiteral->getValue().getSExtValue());
        return true;
    }
    return false;
}

bool parseLoopAffineExpr(const clang::Expr* expr,
                         const clang::VarDecl* loopVarDecl,
                         const std::string& loopVar,
                         int& offset) {
    expr = ignoreTransparentExpr(expr);
    if (!expr || !loopVarDecl) {
        return false;
    }

    if (const auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        if (dre->getDecl() == loopVarDecl) {
            offset = 0;
            return true;
        }
        return false;
    }

    const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(expr);
    if (!binary ||
        (binary->getOpcode() != clang::BO_Add &&
         binary->getOpcode() != clang::BO_Sub)) {
        return false;
    }

    const clang::Expr* lhs = ignoreTransparentExpr(binary->getLHS());
    const clang::Expr* rhs = ignoreTransparentExpr(binary->getRHS());
    const auto* lhsRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(lhs);
    if (!lhsRef || lhsRef->getDecl() != loopVarDecl) {
        return false;
    }

    int literal = 0;
    if (!parseIntegerLiteralExpr(rhs, literal)) {
        return false;
    }

    offset = binary->getOpcode() == clang::BO_Sub ? -literal : literal;
    (void)loopVar;
    return true;
}

const clang::Expr* getSubscriptBaseExpr(const clang::Expr* expr,
                                        const clang::Expr*& indexExpr) {
    expr = ignoreTransparentExpr(expr);
    indexExpr = nullptr;
    if (const auto* subscript = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(expr)) {
        indexExpr = subscript->getIdx();
        return subscript->getBase();
    }
    if (const auto* opCall = llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OO_Subscript &&
            opCall->getNumArgs() >= 2) {
            indexExpr = opCall->getArg(1);
            return opCall->getArg(0);
        }
    }
    return nullptr;
}

bool parseAffineVectorAccessAST(const clang::Expr* expr,
                                const RouteLoopInfo& info,
                                std::string& tensorName,
                                AffineIndex1D& index) {
    const clang::Expr* indexExpr = nullptr;
    const clang::Expr* baseExpr = getSubscriptBaseExpr(expr, indexExpr);
    if (!baseExpr || !indexExpr) {
        return false;
    }

    baseExpr = ignoreTransparentExpr(baseExpr);
    const auto* baseRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(baseExpr);
    if (!baseRef || !baseRef->getDecl()) {
        return false;
    }

    int offset = 0;
    if (!parseLoopAffineExpr(indexExpr, info.loopVarDecl, info.loopVar, offset)) {
        return false;
    }

    tensorName = baseRef->getDecl()->getNameAsString();
    index.loopVar = info.loopVar;
    index.offset = offset;
    return true;
}

struct RouteAssignment {
    const clang::Expr* lhs = nullptr;
    const clang::Expr* rhs = nullptr;
    const clang::Stmt* stmt = nullptr;
};

bool appendSimpleAssignment(const clang::Stmt* stmt,
                            std::vector<RouteAssignment>& assignments) {
    if (const auto* binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(stmt)) {
        if (binary->getOpcode() != clang::BO_Assign) {
            return false;
        }
        assignments.push_back({binary->getLHS(), binary->getRHS(), binary});
        return true;
    }
    if (const auto* opCall =
            llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(stmt)) {
        if (opCall->getOperator() != clang::OO_Equal ||
            opCall->getNumArgs() < 2) {
            return false;
        }
        assignments.push_back({opCall->getArg(0), opCall->getArg(1), opCall});
        return true;
    }
    return false;
}

bool collectTopLevelAssignments(const clang::Stmt* stmt,
                                std::vector<RouteAssignment>& assignments) {
    if (!stmt) {
        return false;
    }

    if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
        for (const clang::Stmt* child : compound->body()) {
            if (!appendSimpleAssignment(child, assignments)) {
                return false;
            }
        }
        return !assignments.empty();
    }

    return appendSimpleAssignment(stmt, assignments);
}

bool tryCollectDistributedFollowup(DistributedStencilSitePlan& plan,
                                   DacppFile* dacppFile,
                                   Shell* shell,
                                   const std::vector<IOTYPE>& effectiveModes,
                                   const std::vector<IOTYPE>& transportModes,
                                   const clang::Stmt* stmt) {
    const auto* forStmt = llvm::dyn_cast_or_null<clang::ForStmt>(stmt);
    if (!forStmt) {
        return false;
    }

    RouteLoopInfo info;
    if (!extractRouteLoopInfo(
            forStmt, dacppFile->getContext(),
            dacppFile->getBufferRegionPlan(), info)) {
        return false;
    }

    std::vector<RouteAssignment> assignments;
    if (!collectTopLevelAssignments(forStmt->getBody(), assignments)) {
        return false;
    }
    if (assignments.empty()) {
        return false;
    }

    std::vector<DistributedFollowupMapping> routes;
    routes.reserve(assignments.size());
    const auto splitMeta = collectSplitBindMeta(shell);
    for (const RouteAssignment& assignment : assignments) {
        std::string readerTensor;
        std::string writerTensor;
        AffineIndex1D readerIndex;
        AffineIndex1D writerIndex;
        if (!parseAffineVectorAccessAST(assignment.lhs, info,
                                        readerTensor, readerIndex) ||
            !parseAffineVectorAccessAST(assignment.rhs, info,
                                        writerTensor, writerIndex)) {
            return false;
        }
        if (!isEffectiveWriter(writerTensor, shell, transportModes) ||
            !isEffectiveReader(readerTensor, shell, transportModes)) {
            return false;
        }
        const int writerIdx = findShellParamIndex(shell, writerTensor);
        const int readerIdx = findShellParamIndex(shell, readerTensor);
        if (writerIdx < 0 || readerIdx < 0) {
            return false;
        }
        if (!writerRouteCoveredByInputDomain(
                shell, effectiveModes, writerIdx, splitMeta)) {
            return false;
        }

        DistributedFollowupMapping route;
        route.writerTensor = writerTensor;
        route.readerTensor = readerTensor;
        route.writerParamIndex = writerIdx;
        route.readerParamIndex = readerIdx;
        route.writerIndex = writerIndex;
        route.readerIndex = readerIndex;
        route.targetOffset = readerIndex.offset - writerIndex.offset;
        route.stmt = assignment.stmt;
        routes.push_back(route);
        llvm::outs() << "[DACPP][MPI][PhaseC] route detected "
                     << writerTensor << "->" << readerTensor
                     << " offset=" << route.targetOffset << "\n";
    }

    plan.followupMappings.insert(plan.followupMappings.end(),
                                 routes.begin(), routes.end());
    plan.distributedFollowupStmts.push_back(stmt);
    return true;
}

}  // namespace

DistributedStencilSitePlan analyzeDistributedStencilSite(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    DistributedStencilSitePlan plan;
    if (!dacppFile || !shell || !calc || !dacExpr) {
        plan.disableReason = "missing shell/calc site";
        return plan;
    }

    const auto& regionPlan = dacppFile->getBufferRegionPlan();
    if (!regionPlan.enabled || regionPlan.dacExpr != dacExpr) {
        plan.disableReason = "phase-c requires rewriteMPIStencil loop lowering";
        return plan;
    }

    const auto effectiveModes = inferEffectiveParamModes(shell, calc);
    const auto transportModes = inferPhaseCTransportParamModes(shell, calc);
    if (effectiveModes.size() != static_cast<std::size_t>(shell->getNumShellParams()) ||
        transportModes.size() != static_cast<std::size_t>(shell->getNumShellParams())) {
        plan.disableReason = "failed to infer phase-c parameter modes";
        return plan;
    }

    for (int paramIdx = 0; paramIdx < shell->getNumShellParams(); ++paramIdx) {
        if (!isVectorParam(shell, calc, paramIdx)) {
            plan.disableReason = "phase-c only supports 1D dacpp::Vector tensors";
            return plan;
        }
        if (effectiveModes[paramIdx] == IOTYPE::READ_WRITE &&
            transportModes[paramIdx] == IOTYPE::WRITE) {
            llvm::outs() << "[DACPP][MPI][PhaseC] param "
                         << shell->getParam(paramIdx)->getName()
                         << " transport=write after write-before-read analysis\n";
        }
        if (transportModes[paramIdx] == IOTYPE::READ_WRITE) {
            plan.disableReason = "phase-c does not support READ_WRITE kernel params";
            return plan;
        }
        plan.distributedTensors.insert(resolveActualTensorName(
            shell->getParam(paramIdx)->getName(), dacExpr));
    }

    std::vector<RootCentricPostRegion> rootRegions;
    const bool allowDistributedFollowup = regionPlan.siblingStmts.size() == 1;
    for (const clang::Stmt* stmt : regionPlan.siblingStmts) {
        if (allowDistributedFollowup &&
            tryCollectDistributedFollowup(plan, dacppFile, shell, effectiveModes,
                                          transportModes, stmt)) {
            continue;
        }
        if (detail::isRootCentricRegionSupported(dacppFile, shell, stmt)) {
            rootRegions.push_back({stmt, ""});
            continue;
        }
        plan.disableReason =
            "phase-c requires post-shell statements to lower as distributed followup or root-centric helpers";
        return plan;
    }

    if (!rootRegions.empty()) {
        plan.hasRootBridge = true;
        collectRootBridgeTensors(plan, dacppFile, shell);
    }

    plan.supported = true;
    return plan;
}

std::vector<DistributedFollowupRegion> collectDistributedFollowupRegions(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr) {
    std::vector<DistributedFollowupRegion> regions;
    const DistributedStencilSitePlan plan =
        analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    if (!plan.supported) {
        return regions;
    }
    for (const clang::Stmt* stmt : plan.distributedFollowupStmts) {
        regions.push_back({stmt});
    }
    return regions;
}

bool tensorUsesDistributedFollowup(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const std::string& tensorName,
    const clang::BinaryOperator* dacExpr) {
    const DistributedStencilSitePlan plan =
        analyzeDistributedStencilSite(dacppFile, shell, calc, dacExpr);
    if (!plan.supported) {
        return false;
    }
    const std::string actualName = resolveActualTensorName(tensorName, dacExpr);
    return plan.distributedTensors.count(actualName) != 0;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
