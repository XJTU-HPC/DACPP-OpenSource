#ifndef DACPP_REWRITER_MPI_OPERATOR_RESIDENT_H
#define DACPP_REWRITER_MPI_OPERATOR_RESIDENT_H

#include <string>
#include <vector>

#include "mpi/operator_resident/OperatorResidentPlan.h"

namespace clang {
class BinaryOperator;
class CallExpr;
class Expr;
class Rewriter;
} // namespace clang

namespace dacppTranslator {

class DacppFile;
class Shell;
class Calc;

namespace mpi_rewriter {

ShellPartitionPlan analyzeShellPartition(const DacExprNode& node);
ShellPartitionPlan analyzeShellPartition(DacppFile* dacppFile,
                                         const DacExprNode& node);

std::vector<OperatorResidentChainPlan> buildOperatorResidentChains(
    DacppFile* dacppFile,
    const std::vector<DacExprNode>& exprNodes,
    const std::vector<ShellPartitionPlan>& shellPlans);

OperatorResidentChainPlan buildSingleOperatorResidentChain(
    const ShellPartitionPlan& shellPlan,
    int chainId);

void analyzeResidency(OperatorResidentChainPlan& chain);

const char* localLayoutKindName(LocalLayoutKind kind);
const char* paramAccessKindName(ParamAccessKind kind);

std::string operatorResidentWrapperName(Shell* shell,
                                        Calc* calc,
                                        int exprIndex);

std::string buildOperatorResidentWrapperCode(
    DacppFile* dacppFile,
    const OperatorResidentChainPlan& chain,
    const ShellPartitionPlan& exprPlan);

bool isShellDerivedStencilLayout(LocalLayoutKind layout);

std::string joinShellCallArgs(const clang::BinaryOperator* dacExpr,
                              DacppFile* dacppFile);

std::string buildWrapperCallForDacExpr(const std::string& wrapperName,
                                       const clang::BinaryOperator* dacExpr,
                                       DacppFile* dacppFile);

} // namespace mpi_rewriter
} // namespace dacppTranslator

#endif
