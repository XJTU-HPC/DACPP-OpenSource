#ifndef DACPP_TRANSLATOR_REWRITER_MPI_COMMON_H
#define DACPP_TRANSLATOR_REWRITER_MPI_COMMON_H

#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"

#include "Rewriter.h"

namespace dacppTranslator {
namespace mpi_rewriter {

struct AccessSummary {
    bool reads = false;
    bool writes = false;
};

struct SplitBindMeta {
    int bindId = 0;
    std::string offset = "0";
};

struct MPIRegionGeneratedCode {
    std::string definitions;
    std::string ctxTypeName;
    std::string ctxVarName;
    std::string initName;
    std::string submitName;
    std::string haloName;
    std::string syncName;
    std::vector<std::pair<const clang::Stmt*, std::string>> siblingHelpers;
};

struct MPIRegionTransferPolicy {
    std::vector<bool> needsInitScatter;
    std::vector<bool> needsSyncGather;
};

std::vector<AccessSummary> summarizeStmtAccess(
    const clang::Stmt* stmt,
    const std::unordered_map<const clang::ValueDecl*, int>& paramIndices,
    int paramCount);

std::string mpiDatatypeFor(const std::string& type);
bool usesByteTransport(const std::string& type);
std::string mpiPayloadCountExpr(const std::string& elemCountExpr,
                                const std::string& type);
std::string toPlannerMode(IOTYPE mode);
std::string toAccessorMode(IOTYPE mode);
bool tensorNeedsBroadcast(DacppFile* dacppFile,
                          const std::string& tensorName,
                          const clang::BinaryOperator* currentDacExpr = nullptr);

std::vector<IOTYPE> inferEffectiveParamModes(Shell* shell, Calc* calc);
int inferViewRank(ShellParam* shellParam, Param* calcParam);
std::string toViewType(ShellParam* shellParam, Param* calcParam, IOTYPE mode);

void collectReturnStmts(const clang::Stmt* stmt,
                        std::vector<const clang::ReturnStmt*>& returns);

std::unordered_map<std::string, SplitBindMeta> collectSplitBindMeta(Shell* shell);

std::string buildPatternInitCode(
    Shell* shell,
    Calc* calc,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta,
    const std::vector<IOTYPE>& paramModes);

std::string buildLocalCalcCode(Shell* shell, Calc* calc);
std::string buildPackBuilderExpr(IOTYPE mode, const std::string& patternName);
std::string buildPackPlanBuilderExpr(IOTYPE mode,
                                     const std::string& rangeName,
                                     const std::string& patternName);
std::string buildRemotePackBuilderExpr(IOTYPE mode,
                                       const std::string& rangeName,
                                       const std::string& patternName);
std::string buildWrapperCode(DacppFile* dacppFile,
                             Shell* shell,
                             Calc* calc,
                             const clang::BinaryOperator* dacExpr = nullptr);
std::string buildPrelude(DacppFile* dacppFile);
std::string buildMPIRegionCodegen(
    DacppFile* dacppFile,
    Expression* expr,
    const MPIRegionTransferPolicy& transferPolicy,
    const clang::BinaryOperator* dacExpr = nullptr);
std::string buildMPIRegionSiblingCode(DacppFile* dacppFile,
                                      Expression* expr,
                                      MPIRegionGeneratedCode& generated);
std::vector<std::string> buildMPIRegionSiblingLookupInitCode(
    Shell* shell,
    Calc* calc,
    const MPIRegionGeneratedCode& generated);

MPIRegionTransferPolicy analyzeMPIRegionTransferPolicy(
    DacppFile* dacppFile,
    Expression* expr,
    const std::vector<IOTYPE>& paramModes);
std::vector<IOTYPE> inferMPIRegionStorageModes(
    DacppFile* dacppFile,
    Expression* expr,
    const std::vector<IOTYPE>& paramModes);

MPIRegionGeneratedCode buildMPIRegionCode(DacppFile* dacppFile,
                                          Expression* expr);

// --- Fine-grained region codegen (split from buildMPIRegionCodegen) ---
std::string buildMPIRegionCtxCode(DacppFile* dacppFile,
                                  Shell* shell,
                                  Calc* calc,
                                  const MPIRegionGeneratedCode& generated);

std::string buildMPIRegionInitCode(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const MPIRegionGeneratedCode& generated,
    const std::vector<IOTYPE>& storageModes,
    const MPIRegionTransferPolicy& transferPolicy,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta,
    const std::string& shellParamSig);

std::string buildMPIRegionSubmitCode(Shell* shell,
                                     Calc* calc,
                                     const MPIRegionGeneratedCode& generated,
                                     const std::vector<IOTYPE>& paramModes);

std::string buildMPIRegionHaloCode(Shell* shell,
                                   Calc* calc,
                                   const MPIRegionGeneratedCode& generated,
                                   const std::vector<IOTYPE>& paramModes);

std::string buildMPIRegionSyncCode(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const MPIRegionGeneratedCode& generated,
    const std::vector<IOTYPE>& storageModes,
    const MPIRegionTransferPolicy& transferPolicy,
    const std::string& shellParamSig,
    const clang::BinaryOperator* dacExpr = nullptr);

std::string buildShellParamSignature(Shell* shell);

std::string joinShellCallArgsMPI(const clang::BinaryOperator* dacExpr,
                                 clang::ASTContext* context);

}  // namespace mpi_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_TRANSLATOR_REWRITER_MPI_COMMON_H
