#ifndef DACPP_REWRITER_MPI_PLAN_H
#define DACPP_REWRITER_MPI_PLAN_H

#include <string>
#include <vector>

namespace clang {
class BinaryOperator;
class Stmt;
} // namespace clang

namespace dacppTranslator {

class DacppFile;
class Expression;
class Shell;
class Calc;

namespace mpi_rewriter {

struct MpiAnalysisContext {
    DacppFile *dacppFile = nullptr;
};

struct DacExprNode {
    int exprIndex = -1;
    Expression *expr = nullptr;
    Shell *shell = nullptr;
    Calc *calc = nullptr;
    const clang::BinaryOperator *dacExpr = nullptr;
    const clang::Stmt *parentStmt = nullptr;
};

enum class MpiPlanKind {
    LegacyAccessPattern,
    StencilPhaseC,
    OperatorResident,
    Unsupported
};

struct MpiPlanResult {
    MpiPlanKind kind = MpiPlanKind::Unsupported;
    int exprIndex = -1;
    std::string reason;
};

// Legacy AccessPattern wrapper path.
struct LegacyWrapperPlan : MpiPlanResult {
    DacExprNode exprNode;
};

// Stencil Phase-C path.
struct StencilPhaseCPlan : MpiPlanResult {
    DacExprNode exprNode;
};

struct MpiLoweringPlan {
    MpiPlanKind overallKind = MpiPlanKind::Unsupported;
    std::vector<DacExprNode> exprNodes;
    std::vector<MpiPlanResult> exprResults;
};

MpiLoweringPlan buildMpiLoweringPlan(DacppFile *dacppFile);

} // namespace mpi_rewriter
} // namespace dacppTranslator

#endif
