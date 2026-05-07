#ifndef DACPP_REWRITER_MPI_PLAN_BASE_H
#define DACPP_REWRITER_MPI_PLAN_BASE_H

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

} // namespace mpi_rewriter
} // namespace dacppTranslator

#endif
