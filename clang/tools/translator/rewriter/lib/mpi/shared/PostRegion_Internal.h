#ifndef DACPP_TRANSLATOR_REWRITER_MPI_POST_REGION_INTERNAL_H
#define DACPP_TRANSLATOR_REWRITER_MPI_POST_REGION_INTERNAL_H

#include <cstddef>
#include <set>
#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace detail {

struct LoopRegionInfo {
    std::string loopVar;
    std::string lowerExpr;
    std::string upperExpr;
    bool upperInclusive = true;
    std::string bodyText;
    std::set<std::string> readTensors;
    std::set<std::string> writtenTensors;
};

bool extractLoopRegionInfo(const clang::ForStmt* forStmt,
                           clang::ASTContext* context,
                           Shell* shell,
                           const BufferRegionPlan& plan,
                           LoopRegionInfo& info);

bool isRootCentricRegionSupported(DacppFile* dacppFile,
                                  Shell* shell,
                                  const BufferRegionPlan& plan,
                                  const clang::Stmt* stmt);

std::string helperBaseName(Shell* shell, Calc* calc, int exprIdx);
std::string helperNameFor(Shell* shell,
                          Calc* calc,
                          int exprIdx,
                          std::size_t stmtIdx);

}  // namespace detail
}  // namespace mpi_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_TRANSLATOR_REWRITER_MPI_POST_REGION_INTERNAL_H
