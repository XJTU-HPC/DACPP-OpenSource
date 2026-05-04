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

enum class OutputSyncRequirement {
    RootOnly,
    AllRanksNeeded,
    RootCentricFollowup,
    DistributedFollowup
};

struct RootCentricPostRegion {
    const clang::Stmt* stmt = nullptr;
    std::string helperName;
};

struct DistributedFollowupRegion {
    const clang::Stmt* stmt = nullptr;
};

struct AffineIndex1D {
    std::string loopVar;
    int offset = 0;
};

struct DistributedFollowupMapping {
    std::string writerTensor;
    std::string readerTensor;
    int writerParamIndex = -1;
    int readerParamIndex = -1;
    AffineIndex1D writerIndex;
    AffineIndex1D readerIndex;
    int targetOffset = 0;
    const clang::Stmt* stmt = nullptr;
};

struct DistributedStencilSitePlan {
    bool supported = false;
    bool hasRootBridge = false;
    std::string disableReason;
    std::set<std::string> distributedTensors;
    std::set<std::string> rootBridgeTensors;
    std::vector<DistributedFollowupMapping> followupMappings;
    std::vector<const clang::Stmt*> distributedFollowupStmts;
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
OutputSyncRequirement classifyOutputSyncRequirement(
    DacppFile* dacppFile,
    const std::string& tensorName,
    const clang::BinaryOperator* currentDacExpr = nullptr);
bool requiresBroadcast(OutputSyncRequirement requirement);
const char* outputSyncRequirementName(OutputSyncRequirement requirement);
bool tensorNeedsBroadcast(DacppFile* dacppFile,
                          const std::string& tensorName,
                          const clang::BinaryOperator* currentDacExpr = nullptr);

std::vector<IOTYPE> inferEffectiveParamModes(Shell* shell, Calc* calc);
std::vector<IOTYPE> inferPhaseCTransportParamModes(Shell* shell, Calc* calc);
int inferViewRank(ShellParam* shellParam, Param* calcParam);
std::string toViewType(ShellParam* shellParam, Param* calcParam, IOTYPE mode);

void collectReturnStmts(const clang::Stmt* stmt,
                        std::vector<const clang::ReturnStmt*>& returns);
void rewritePrintCallsRootOnly(clang::Rewriter* rewriter,
                               clang::TranslationUnitDecl* tuDecl);

std::unordered_map<std::string, SplitBindMeta> collectSplitBindMeta(Shell* shell);

std::string buildPatternInitCode(
    Shell* shell,
    Calc* calc,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta,
    const std::vector<IOTYPE>& paramModes);

std::string buildLocalCalcCode(Shell* shell, Calc* calc);
std::string buildPackPlanBuilderExpr(IOTYPE mode,
                                     const std::string& rangeName,
                                     const std::string& patternName);
std::string buildWrapperCode(DacppFile* dacppFile,
                             Shell* shell,
                             Calc* calc,
                             const clang::BinaryOperator* dacExpr = nullptr);
std::string buildPrelude(DacppFile* dacppFile);

DistributedStencilSitePlan analyzeDistributedStencilSite(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr);
bool tensorUsesDistributedFollowup(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const std::string& tensorName,
    const clang::BinaryOperator* dacExpr);

std::vector<RootCentricPostRegion> collectRootCentricPostRegions(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr);
std::vector<DistributedFollowupRegion> collectDistributedFollowupRegions(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr);
std::vector<const clang::Stmt*> collectRootCentricPostRegionStmts(
    DacppFile* dacppFile,
    const clang::BinaryOperator* dacExpr);
std::string buildRootCentricPostRegionHelpers(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr,
    const std::string& ctxTypeName,
    const std::string& shellSignature);

}  // namespace mpi_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_TRANSLATOR_REWRITER_MPI_COMMON_H
