#ifndef DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_ANALYSIS_INTERNAL_H
#define DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_ANALYSIS_INTERNAL_H

#include <string>
#include <unordered_map>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"

#include "Rewriter_MPI_Stencil_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace detail {

struct RouteLoopInfo {
    std::string loopVar;
    const clang::VarDecl* loopVarDecl = nullptr;
};

struct RouteLoopInfo2D {
    RouteLoopInfo row;
    RouteLoopInfo col;
};

struct RouteAssignment {
    const clang::Expr* lhs = nullptr;
    const clang::Expr* rhs = nullptr;
    const clang::Stmt* stmt = nullptr;
};

std::string getStmtSourceText(const clang::Stmt* stmt,
                              clang::ASTContext* context);
std::string resolveActualTensorName(const std::string& shellParamName,
                                    const clang::BinaryOperator* dacExpr);

bool isVectorParam(Shell* shell, Calc* calc, int paramIdx);
bool isMatrixParam(Shell* shell, Calc* calc, int paramIdx);
void collectRootBridgeTensors(DistributedStencilSitePlan& plan,
                              DacppFile* dacppFile,
                              Shell* shell,
                              const BufferRegionPlan& regionPlan);

bool isEffectiveWriter(const std::string& tensorName,
                       Shell* shell,
                       const std::vector<IOTYPE>& transportModes);
bool isEffectiveReader(const std::string& tensorName,
                       Shell* shell,
                       const std::vector<IOTYPE>& transportModes);
int findShellParamIndex(Shell* shell, const std::string& tensorName);
bool writerRouteCoveredByInputDomain(
    Shell* shell,
    const std::vector<IOTYPE>& effectiveModes,
    int writerParamIndex,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta);
bool writerRouteCoveredByInputDomain2D(
    Shell* shell,
    const std::vector<IOTYPE>& effectiveModes,
    int writerParamIndex,
    const std::unordered_map<std::string, SplitBindMeta>& splitMeta);

bool extractRouteLoopInfo(const clang::ForStmt* forStmt,
                          clang::ASTContext* context,
                          const BufferRegionPlan& plan,
                          RouteLoopInfo& info);
bool extractRouteLoopInfo2D(const clang::ForStmt* outerFor,
                            clang::ASTContext* context,
                            const BufferRegionPlan& plan,
                            RouteLoopInfo2D& info);
bool parseIntegerLiteralExpr(const clang::Expr* expr, int& value);
bool parseAffineVectorAccessAST(const clang::Expr* expr,
                                const RouteLoopInfo& info,
                                std::string& tensorName,
                                AffineIndex1D& index);
bool parseAffineMatrixAccessAST(const clang::Expr* expr,
                                const RouteLoopInfo2D& info,
                                std::string& tensorName,
                                AffineIndex2D& index);
bool parseBoundaryMatrixAccessAST(const clang::Expr* expr,
                                  const RouteLoopInfo& loopInfo,
                                  clang::ASTContext* context,
                                  std::string& tensorName,
                                  bool& rowUsesLoop,
                                  int& rowOffset,
                                  std::string& rowExprText,
                                  bool& colUsesLoop,
                                  int& colOffset,
                                  std::string& colExprText);
bool parseBoundaryVectorAccessAST(const clang::Expr* expr,
                                  const RouteLoopInfo& loopInfo,
                                  clang::ASTContext* context,
                                  std::string& tensorName,
                                  bool& usesLoop,
                                  int& offset,
                                  std::string& exprText);
bool isBoundaryCopyPair(bool targetUsesLoop,
                        int targetOffset,
                        const std::string& targetExpr,
                        bool sourceUsesLoop,
                        int sourceOffset,
                        const std::string& sourceExpr);
bool isBoundaryConstantTarget(bool targetUsesLoop,
                              int targetOffset,
                              const std::string& targetExpr,
                              bool otherDimUsesLoop);
bool collectTopLevelAssignments(const clang::Stmt* stmt,
                                std::vector<RouteAssignment>& assignments);
bool collectNestedLoopAssignments2D(const clang::ForStmt* outerFor,
                                    std::vector<RouteAssignment>& assignments);

bool tryCollectDistributedFollowup(DistributedStencilSitePlan& plan,
                                   DacppFile* dacppFile,
                                   Shell* shell,
                                   const BufferRegionPlan& regionPlan,
                                   const std::vector<IOTYPE>& effectiveModes,
                                   const std::vector<IOTYPE>& transportModes,
                                   const clang::Stmt* stmt);
bool tryCollectDistributedFollowup2D(DistributedStencilSitePlan& plan,
                                     DacppFile* dacppFile,
                                     Shell* shell,
                                     const BufferRegionPlan& regionPlan,
                                     const std::vector<IOTYPE>& effectiveModes,
                                     const std::vector<IOTYPE>& transportModes,
                                     const clang::Stmt* stmt);
bool tryCollectReadCacheTransition2D(DistributedStencilSitePlan& plan,
                                     DacppFile* dacppFile,
                                     Shell* shell,
                                     const BufferRegionPlan& regionPlan,
                                     const std::vector<IOTYPE>& transportModes,
                                     const clang::Stmt* stmt);
bool tryCollectBoundaryLocalUpdate2D(DistributedStencilSitePlan& plan,
                                     DacppFile* dacppFile,
                                     Shell* shell,
                                     const BufferRegionPlan& regionPlan,
                                     const std::vector<IOTYPE>& transportModes,
                                     const clang::Stmt* stmt);
bool tryCollectBoundaryLocalUpdate1D(DistributedStencilSitePlan& plan,
                                     DacppFile* dacppFile,
                                     Shell* shell,
                                     const BufferRegionPlan& regionPlan,
                                     const std::vector<IOTYPE>& transportModes,
                                     const clang::Stmt* stmt);

}  // namespace detail
}  // namespace mpi_rewriter
}  // namespace dacppTranslator

#endif  // DACPP_TRANSLATOR_REWRITER_MPI_STENCIL_ANALYSIS_INTERNAL_H
