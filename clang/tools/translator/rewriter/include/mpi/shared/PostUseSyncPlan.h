#ifndef DACPP_REWRITER_MPI_SHARED_POST_USE_SYNC_PLAN_H
#define DACPP_REWRITER_MPI_SHARED_POST_USE_SYNC_PLAN_H

#include <cstdint>
#include <string>
#include <vector>

namespace clang {
class Stmt;
} // namespace clang

namespace dacppTranslator {
namespace mpi_rewriter {

enum class PostUseSyncKind {
    None,
    BoundedIndexedRootRead,
    ScalarReductionCountEqOne,
    FullTensor
};

struct PostUseBoundedIndex {
    std::vector<int64_t> indices;
};

struct PostUseSyncPlan {
    PostUseSyncKind kind = PostUseSyncKind::FullTensor;
    std::string tensorName;
    std::string reason;
    std::vector<PostUseBoundedIndex> boundedIndices;
    const clang::Stmt* tensor2ArrayStmt = nullptr;
    std::string tensor2ArrayTargetName;
};

}  // namespace mpi_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_REWRITER_MPI_SHARED_POST_USE_SYNC_PLAN_H
