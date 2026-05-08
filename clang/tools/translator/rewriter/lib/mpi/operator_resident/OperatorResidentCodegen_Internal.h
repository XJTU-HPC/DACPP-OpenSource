#ifndef DACPP_REWRITER_MPI_OPERATOR_RESIDENT_CODEGEN_INTERNAL_H
#define DACPP_REWRITER_MPI_OPERATOR_RESIDENT_CODEGEN_INTERNAL_H

#include <string>

#include "Rewriter_MPI_OperatorResident.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

std::string paramVarName(const ParamAccessPlan& param);
std::string wrapperSignature(const ShellPartitionPlan& plan);
std::string elemType(const ShellPartitionPlan& plan,
                     const ParamAccessPlan& param);
std::string viewType(const ShellPartitionPlan& plan,
                     const ParamAccessPlan& param);
std::string localName(const ParamAccessPlan& param);
std::string globalName(const ParamAccessPlan& param);
std::string scalarName(const ParamAccessPlan& param);
bool usesByte(const ShellPartitionPlan& plan, const ParamAccessPlan& param);

void emitByteCounts(std::string& code,
                    const std::string& suffix,
                    const std::string& type);
void emitScatter(std::string& code,
                 const ShellPartitionPlan& plan,
                 const ParamAccessPlan& param);
void emitResidentOrScatter(std::string& code,
                           const ShellPartitionPlan& plan,
                           const ParamAccessPlan& param);
void emitScalarBroadcast(std::string& code,
                         const ShellPartitionPlan& plan,
                         const ParamAccessPlan& param);
void emitReplicatedFullTensorBroadcast(std::string& code,
                                       const ShellPartitionPlan& plan,
                                       const ParamAccessPlan& param);
void emitRowPartitionFullRowScatter(std::string& code,
                                    const ShellPartitionPlan& plan,
                                    const ParamAccessPlan& param);
void emitParamLocalStorage(std::string& code,
                           const ShellPartitionPlan& plan);
void emitPartitionCode(std::string& code, const ShellPartitionPlan& plan);
void emitKernel(std::string& code, const ShellPartitionPlan& plan);
void emitGatherMaterialize(std::string& code,
                           const ShellPartitionPlan& plan,
                           const ParamAccessPlan& param);
void emitGatherMaterializeFromLocalBuffer(
    std::string& code,
    const ShellPartitionPlan& plan,
    const ParamAccessPlan& param,
    const std::string& localBufferName,
    const std::string& totalItemCountExpr);
void emitResidencyAndMaterialization(std::string& code,
                                     const ShellPartitionPlan& plan);
std::string buildStencilWindow2DWrapperCode(
    const std::string& wrapperName,
    const ShellPartitionPlan& plan);
std::string buildStencilWindow1DWrapperCode(
    const std::string& wrapperName,
    const ShellPartitionPlan& plan);

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator

#endif
