#ifndef DACPP_REWRITER_MPI_SHARED_LOOP_LOWERED_REWRITE_H
#define DACPP_REWRITER_MPI_SHARED_LOOP_LOWERED_REWRITE_H

#include <string>

namespace clang {
class BinaryOperator;
class Rewriter;
class Stmt;
} // namespace clang

namespace dacppTranslator {
namespace mpi_rewriter {

struct LoopLoweredRewriteSpec {
    const clang::Stmt* outerLoop = nullptr;
    const clang::BinaryOperator* dacExpr = nullptr;
    std::string contextTypeName;
    std::string contextVariableName;
    std::string initFunctionName;
    std::string runFunctionName;
    std::string materializeFunctionName;
    std::string argumentText;
};

void rewriteLoopLoweredDacExpr(clang::Rewriter* rewriter,
                               const LoopLoweredRewriteSpec& spec);

} // namespace mpi_rewriter
} // namespace dacppTranslator

#endif // DACPP_REWRITER_MPI_SHARED_LOOP_LOWERED_REWRITE_H
