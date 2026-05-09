#include "mpi/shared/LoopLoweredRewrite.h"

#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace dacppTranslator {
namespace mpi_rewriter {

void rewriteLoopLoweredDacExpr(clang::Rewriter* rewriter,
                               const LoopLoweredRewriteSpec& spec) {
    if (!rewriter || !spec.outerLoop || !spec.dacExpr) {
        return;
    }

    std::string initCode =
        "    " + spec.contextTypeName + " " + spec.contextVariableName + ";\n";
    initCode +=
        "    " + spec.initFunctionName + "(" + spec.contextVariableName;
    if (!spec.argumentText.empty()) {
        initCode += ", " + spec.argumentText;
    }
    initCode += ");\n";
    rewriter->InsertTextBefore(spec.outerLoop->getBeginLoc(), initCode);

    std::string runCall =
        spec.runFunctionName + "(" + spec.contextVariableName;
    if (!spec.argumentText.empty()) {
        runCall += ", " + spec.argumentText;
    }
    runCall += ")";
    rewriter->ReplaceText(spec.dacExpr->getSourceRange(), runCall);

    std::string materializeCall =
        "\n    " + spec.materializeFunctionName + "(" +
        spec.contextVariableName;
    if (!spec.argumentText.empty()) {
        materializeCall += ", " + spec.argumentText;
    }
    materializeCall += ");\n";
    rewriter->InsertTextAfterToken(spec.outerLoop->getEndLoc(),
                                   materializeCall);
}

} // namespace mpi_rewriter
} // namespace dacppTranslator
