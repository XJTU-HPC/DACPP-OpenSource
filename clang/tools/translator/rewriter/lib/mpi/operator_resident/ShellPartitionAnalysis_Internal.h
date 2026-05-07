#ifndef DACPP_REWRITER_MPI_OPERATOR_RESIDENT_SHELL_PARTITION_ANALYSIS_INTERNAL_H
#define DACPP_REWRITER_MPI_OPERATOR_RESIDENT_SHELL_PARTITION_ANALYSIS_INTERNAL_H

#include <cstdint>
#include <string>
#include <vector>

#include "Rewriter_MPI_OperatorResident.h"

namespace dacppTranslator {

class Calc;
class Shell;
class Split;

namespace mpi_rewriter {
namespace operator_resident {

const char* shellDimKindName(ShellDimKind kind);
int64_t shapeValueFor(Shell* shell, int paramIdx, int dimIdx);
bool splitIsVoid(Split* split);
bool splitIsIndex(Split* split);
bool splitIsRegular(Split* split);
bool isScalarVoidParam(Shell* shell, int paramIdx);
bool calcUsesParamAsScalar(Calc* calc, int paramIdx);
bool sameOrder(const std::vector<int>& lhs, const std::vector<int>& rhs);
std::vector<int> uniquePreserveOrder(const std::vector<int>& values);
bool assignContiguous1DLayout(ShellPartitionPlan& plan,
                              std::string& rejectReason);
bool assignRowBlock2DLayout(ShellPartitionPlan& plan,
                            bool sawScalarParam,
                            std::string& rejectReason);
bool assignRowPartitionFullRowLayout(ShellPartitionPlan& plan,
                                     bool sawScalarParam,
                                     std::string& rejectReason);
bool assignReplicatedFullTensorLayout(ShellPartitionPlan& plan,
                                     bool sawScalarParam,
                                     std::string& rejectReason);
bool assignPhaseLayout(ShellPartitionPlan& plan,
                       bool sawScalarParam,
                       std::string& rejectReason);

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator

#endif
