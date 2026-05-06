#ifndef DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_CODEGEN_INTERNAL_H
#define DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_CODEGEN_INTERNAL_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "Rewriter_MPI_Stencil_Common.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {
namespace detail {

std::vector<const mpi_rewriter::DistributedFollowupMapping*>
findFollowupMappingsForWriter(
    const mpi_rewriter::DistributedStencilSitePlan& plan,
    const std::string& writerTensor);

int findDefaultReaderIndex(const std::vector<IOTYPE>& transportModes);
std::string buildParamNameList(Shell* shell);

int findLoopCarriedInputSourceParam(
    DacppFile* dacppFile,
    Shell* shell,
    Calc* calc,
    const clang::BinaryOperator* dacExpr,
    int targetParamIndex,
    const std::vector<std::string>& actualTensorNames,
    const std::vector<IOTYPE>& transportModes,
    const std::vector<bool>& aliasesAnyOtherParam);

bool isFallbackInputCacheCandidate(DacppFile* dacppFile,
                                   const clang::BinaryOperator* dacExpr,
                                   const std::string& actualTensorName,
                                   IOTYPE transportMode);

std::string resolveActualTensorName(const std::string& shellParamName,
                                    const clang::BinaryOperator* dacExpr);

void appendPatternInitForContext(
    std::string& code,
    Shell* shell,
    Calc* calc,
    const std::unordered_map<std::string, mpi_rewriter::SplitBindMeta>& splitMeta,
    const std::vector<IOTYPE>& transportModes,
    const std::vector<IOTYPE>& itemDomainModes,
    const std::string& ctxVar);

struct WaveSpecializationCodegenConfig {
    bool enable_span_pairs = false;
    bool enable_direct_kernel = false;
};

std::string waveRouteFastPathExpr(int paramIdx, const std::string& routeIdxExpr);
std::string waveReadTransitionFastPathExpr(const std::string& idxExpr);

WaveSpecializationCodegenConfig buildWaveSpecializationCodegenConfig(
    Shell* shell,
    Calc* calc,
    const mpi_rewriter::DistributedStencilSitePlan& distributedSitePlan);

void appendWaveContextFields(std::string& code);
void appendWaveInitFlags(std::string& code,
                         const WaveSpecializationCodegenConfig& waveConfig,
                         int shellParamCount,
                         std::size_t transitionCount);
void appendWaveDirectKernelMetadataInit(
    std::string& code,
    const WaveSpecializationCodegenConfig& waveConfig);
void appendWaveRouteFastPathInit(std::string& code,
                                 const WaveSpecializationCodegenConfig& waveConfig,
                                 const std::string& routeStateExpr,
                                 const std::string& distExpr,
                                 const std::string& routeIdxExpr);
void appendWaveReadTransitionFastPathInit(
    std::string& code,
    const WaveSpecializationCodegenConfig& waveConfig,
    const std::string& idx,
    const std::string& fastPathExpr);
bool appendWaveDistributedWriteReset(
    std::string& code,
    const WaveSpecializationCodegenConfig& waveConfig,
    const std::string& calcName,
    const std::string& elemType);
void appendWaveDistributedKernelDispatchHead(
    std::string& code,
    const WaveSpecializationCodegenConfig& waveConfig);
void appendWaveReadTransitionRun(std::string& code,
                                 const WaveSpecializationCodegenConfig& waveConfig,
                                 const std::string& idx,
                                 const std::string& fastPathExpr,
                                 const std::string& writerCalcName,
                                 const std::string& readerCalcName);
void appendWavePublishRun(std::string& code,
                          const WaveSpecializationCodegenConfig& waveConfig,
                          bool useDistributedReaderMaterialize,
                          const std::string& calcName,
                          const std::string& routeStateExpr,
                          const std::string& routeIdxExpr,
                          const std::string& targetCacheExpr);

}  // namespace detail
}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_CODEGEN_INTERNAL_H
