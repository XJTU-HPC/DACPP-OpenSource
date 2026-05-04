#ifndef DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_COMMON_H
#define DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_COMMON_H

#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_stencil_rewriter {

std::string wrapperName(Shell* shell, Calc* calc);
std::string contextTypeName(Shell* shell, Calc* calc);
std::string initFunctionName(Shell* shell, Calc* calc);
std::string runFunctionName(Shell* shell, Calc* calc);
std::string materializeFunctionName(Shell* shell, Calc* calc);
std::string buildShellSignature(Shell* shell);

std::string buildStencilWrapperCode(DacppFile* dacppFile,
                                    Shell* shell,
                                    Calc* calc,
                                    const clang::BinaryOperator* dacExpr);

}  // namespace mpi_stencil_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_COMMON_H
