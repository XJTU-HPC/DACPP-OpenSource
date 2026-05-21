#include "mpi/shared/LoopLoweredRewrite.h"

#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace dacppTranslator {
namespace mpi_rewriter {

namespace {

clang::CharSourceRange tokenCharRangeForStmt(clang::Rewriter* rewriter,
                                             const clang::Stmt* stmt) {
    if (!rewriter || !stmt) {
        return clang::CharSourceRange();
    }
    const clang::SourceManager& sourceManager = rewriter->getSourceMgr();
    const clang::LangOptions& langOpts = rewriter->getLangOpts();
    clang::SourceLocation begin = stmt->getBeginLoc();
    clang::SourceLocation end = clang::Lexer::getLocForEndOfToken(
        stmt->getEndLoc(), 0, sourceManager, langOpts);
    if (begin.isInvalid() || end.isInvalid()) {
        return clang::CharSourceRange();
    }
    return clang::CharSourceRange::getCharRange(begin, end);
}

clang::CharSourceRange compoundLoopCharRange(clang::Rewriter* rewriter,
                                             const clang::Stmt* stmt) {
    if (!rewriter || !stmt) {
        return clang::CharSourceRange();
    }
    const clang::SourceManager& sourceManager = rewriter->getSourceMgr();
    clang::SourceLocation begin = stmt->getBeginLoc();
    if (begin.isInvalid()) {
        return clang::CharSourceRange();
    }
    clang::FileID fileId = sourceManager.getFileID(begin);
    bool invalid = false;
    llvm::StringRef buffer = sourceManager.getBufferData(fileId, &invalid);
    if (invalid) {
        return clang::CharSourceRange();
    }
    const unsigned beginOffset = sourceManager.getFileOffset(begin);
    if (beginOffset >= buffer.size()) {
        return clang::CharSourceRange();
    }

    bool inLineComment = false;
    bool inBlockComment = false;
    bool inString = false;
    bool inChar = false;
    bool escaped = false;
    int braceDepth = 0;
    bool sawBodyBrace = false;
    for (unsigned pos = beginOffset; pos < buffer.size(); ++pos) {
        const char c = buffer[pos];
        const char next = pos + 1 < buffer.size() ? buffer[pos + 1] : '\0';
        if (inLineComment) {
            if (c == '\n') {
                inLineComment = false;
            }
            continue;
        }
        if (inBlockComment) {
            if (c == '*' && next == '/') {
                inBlockComment = false;
                ++pos;
            }
            continue;
        }
        if (inString) {
            if (!escaped && c == '"') {
                inString = false;
            }
            escaped = !escaped && c == '\\';
            if (c != '\\') {
                escaped = false;
            }
            continue;
        }
        if (inChar) {
            if (!escaped && c == '\'') {
                inChar = false;
            }
            escaped = !escaped && c == '\\';
            if (c != '\\') {
                escaped = false;
            }
            continue;
        }
        if (c == '/' && next == '/') {
            inLineComment = true;
            ++pos;
            continue;
        }
        if (c == '/' && next == '*') {
            inBlockComment = true;
            ++pos;
            continue;
        }
        if (c == '"') {
            inString = true;
            escaped = false;
            continue;
        }
        if (c == '\'') {
            inChar = true;
            escaped = false;
            continue;
        }
        if (c == '{') {
            ++braceDepth;
            sawBodyBrace = true;
            continue;
        }
        if (c == '}' && sawBodyBrace) {
            --braceDepth;
            if (braceDepth == 0) {
                clang::SourceLocation end =
                    begin.getLocWithOffset(static_cast<int>(pos + 1 -
                                                            beginOffset));
                if (end.isInvalid()) {
                    return clang::CharSourceRange();
                }
                return clang::CharSourceRange::getCharRange(begin, end);
            }
        }
    }
    return clang::CharSourceRange();
}

std::string selectedHostRowWriteback(const LoopLoweredRewriteSpec& spec) {
    if (!spec.writeBackSelectedHostRow ||
        spec.selectedHostTensorName.empty() ||
        spec.selectedOutputTensorName.empty() ||
        spec.selectedHostRowCondition.empty() ||
        spec.selectedHostRow < 0) {
        return "";
    }
    std::string code;
    code += "    if (__dacpp_mpi_is_root_rank() && (" +
            spec.selectedHostRowCondition + ")) {\n";
    code += "        " + spec.selectedHostTensorName + "[" +
            std::to_string(spec.selectedHostRow) + "] = " +
            spec.selectedOutputTensorName + ";\n";
    code += "    }\n";
    return code;
}

bool replaceLoopConditionWithFalse(clang::Rewriter* rewriter,
                                   const clang::Stmt* loop) {
    if (!rewriter || !loop) {
        return false;
    }
    const clang::Expr* cond = nullptr;
    if (const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(loop)) {
        cond = forStmt->getCond();
    } else if (const auto* whileStmt =
                   llvm::dyn_cast<clang::WhileStmt>(loop)) {
        cond = whileStmt->getCond();
    }
    if (!cond) {
        return false;
    }
    const auto condRange = tokenCharRangeForStmt(rewriter, cond);
    if (!condRange.isValid()) {
        return false;
    }
    rewriter->ReplaceText(condRange, "false");
    return true;
}

} // namespace

void rewriteLoopLoweredDacExpr(clang::Rewriter* rewriter,
                               const LoopLoweredRewriteSpec& spec) {
    if (!rewriter || !spec.outerLoop || !spec.dacExpr) {
        return;
    }

    auto buildInitCode = [&]() {
        std::string code =
            "    " + spec.contextTypeName + " " + spec.contextVariableName +
            ";\n";
        code +=
            "    " + spec.initFunctionName + "(" + spec.contextVariableName;
        if (!spec.argumentText.empty()) {
            code += ", " + spec.argumentText;
        }
        code += ");\n";
        return code;
    };

    auto buildRunLoopCall = [&]() {
        std::string call =
            (spec.runLoopFunctionName.empty() ? spec.runFunctionName
                                              : spec.runLoopFunctionName) +
            "(" + spec.contextVariableName;
        if (!spec.argumentText.empty()) {
            call += ", " + spec.argumentText;
        }
        call += ")";
        return call;
    };

    if (spec.replaceOuterLoopWithRunLoop) {
        std::string preLoopCode = buildInitCode();
        const std::string runLoopCall = buildRunLoopCall();
        if (spec.writeBackSelectedHostRow) {
            preLoopCode += "    const bool " +
                           spec.selectedHostRowCondition + " = " +
                           runLoopCall + ";\n";
        } else {
            preLoopCode += "    " + runLoopCall + ";\n";
        }
        if (!spec.materializeFunctionName.empty()) {
            preLoopCode += "    " + spec.materializeFunctionName + "(" +
                spec.contextVariableName;
            if (!spec.argumentText.empty()) {
                preLoopCode += ", " + spec.argumentText;
            }
            preLoopCode += ");\n";
        }
        preLoopCode += selectedHostRowWriteback(spec);
        rewriter->InsertTextBefore(spec.outerLoop->getBeginLoc(), preLoopCode);
        replaceLoopConditionWithFalse(rewriter, spec.outerLoop);
        const auto dacRange = tokenCharRangeForStmt(rewriter, spec.dacExpr);
        if (dacRange.isValid()) {
            rewriter->ReplaceText(dacRange, "((void)0)");
        } else {
            rewriter->ReplaceText(spec.dacExpr->getSourceRange(),
                                  "((void)0)");
        }
        return;
    }

    const std::string initCode = buildInitCode();
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
