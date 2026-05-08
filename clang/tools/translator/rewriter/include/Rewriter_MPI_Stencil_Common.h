#ifndef DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_COMMON_H
#define DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_COMMON_H

#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {

std::string wrapperName(Shell* shell, Calc* calc, int exprIdx);
std::string contextTypeName(Shell* shell, Calc* calc, int exprIdx);
std::string initFunctionName(Shell* shell, Calc* calc, int exprIdx);
std::string runFunctionName(Shell* shell, Calc* calc, int exprIdx);
std::string materializeFunctionName(Shell* shell, Calc* calc, int exprIdx);
std::string buildShellSignature(Shell* shell);

std::string buildStencilWrapperCode(DacppFile* dacppFile,
                                    Shell* shell,
                                    Calc* calc,
                                    int exprIdx,
                                    const clang::BinaryOperator* dacExpr);

void insertMainMPISetup(DacppFile* dacppFile,
                        clang::Rewriter* rewriter,
                        const clang::FunctionDecl* mainFunc);

void rewriteStencilPhaseCSite(DacppFile* dacppFile,
                              clang::Rewriter* rewriter,
                              const MpiStencilSite& site,
                              int exprIdx,
                              Shell* shell,
                              Calc* calc);

}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_COMMON_H
