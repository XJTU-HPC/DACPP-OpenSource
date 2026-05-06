#include <set>

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "OutputSyncAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace {

class RootOnlyPrintRewriteVisitor
    : public clang::RecursiveASTVisitor<RootOnlyPrintRewriteVisitor> {
public:
    clang::Rewriter* TheRewriter = nullptr;
    clang::ASTContext* Context = nullptr;
    std::set<const clang::Stmt*> RewrittenStmts;

    RootOnlyPrintRewriteVisitor(clang::Rewriter* rewriter,
                                clang::ASTContext* context)
        : TheRewriter(rewriter), Context(context) {}

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call || !TheRewriter) {
            return true;
        }
        if (!detail::isPrintMethodCall(call)) {
            return true;
        }

        rewriteOutputStatement(call);
        return true;
    }

    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* call) {
        if (!call || !TheRewriter) {
            return true;
        }
        if (call->getOperator() != clang::OO_LessLess ||
            !detail::containsCoutExpr(call)) {
            return true;
        }

        rewriteOutputStatement(call);
        return true;
    }

private:
    void rewriteOutputStatement(const clang::Stmt* output) {
        const clang::Stmt* stmt = enclosingStatement(output);
        if (!stmt || RewrittenStmts.count(stmt) != 0) {
            return;
        }
        const clang::SourceManager& sourceManager = TheRewriter->getSourceMgr();
        if (!stmt->getBeginLoc().isValid() ||
            !sourceManager.isWrittenInMainFile(stmt->getBeginLoc())) {
            return;
        }
        const clang::LangOptions& langOpts = TheRewriter->getLangOpts();
        clang::SourceLocation replacementEnd = statementEndIncludingSemi(stmt);
        if (replacementEnd.isInvalid()) {
            return;
        }
        const std::string stmtText =
            clang::Lexer::getSourceText(
                clang::CharSourceRange::getCharRange(stmt->getBeginLoc(),
                                                     replacementEnd),
                sourceManager, langOpts)
                .str();
        if (stmtText.empty()) {
            return;
        }

        TheRewriter->ReplaceText(
            clang::CharSourceRange::getCharRange(stmt->getBeginLoc(),
                                                 replacementEnd),
            "if (__dacpp_mpi_is_root_rank()) {\n        " + stmtText +
                "\n    }");
        RewrittenStmts.insert(stmt);
    }

    const clang::Stmt* enclosingStatement(const clang::Stmt* stmt) const {
        if (!stmt || !Context) {
            return nullptr;
        }

        clang::DynTypedNode current = clang::DynTypedNode::create(*stmt);
        for (int depth = 0; depth < detail::kMaxEnclosingStmtDepth; ++depth) {
            auto parents = Context->getParents(current);
            if (parents.empty()) {
                break;
            }
            const clang::DynTypedNode& parent = parents[0];
            if (parent.get<clang::CompoundStmt>()) {
                if (const auto* asStmt = current.get<clang::Stmt>()) {
                    return asStmt;
                }
                break;
            }
            if (parent.get<clang::ForStmt>() || parent.get<clang::WhileStmt>() ||
                parent.get<clang::IfStmt>()) {
                if (const auto* asStmt = current.get<clang::Stmt>()) {
                    return asStmt;
                }
                break;
            }
            if (parent.get<clang::ReturnStmt>()) {
                break;
            }
            current = parent;
        }
        return nullptr;
    }

    clang::SourceLocation statementEndIncludingSemi(
        const clang::Stmt* stmt) const {
        if (!stmt || !TheRewriter) {
            return clang::SourceLocation();
        }
        const clang::SourceManager& sourceManager = TheRewriter->getSourceMgr();
        const clang::LangOptions& langOpts = TheRewriter->getLangOpts();
        clang::SourceLocation afterToken =
            clang::Lexer::getLocForEndOfToken(stmt->getEndLoc(), 0,
                                              sourceManager, langOpts);
        if (afterToken.isInvalid()) {
            return clang::SourceLocation();
        }

        for (int offset = 0; offset < detail::kMaxSemicolonScanOffset; ++offset) {
            clang::SourceLocation loc = afterToken.getLocWithOffset(offset);
            if (loc.isInvalid() ||
                !sourceManager.isWrittenInSameFile(afterToken, loc)) {
                break;
            }
            bool invalid = false;
            const char* data = sourceManager.getCharacterData(loc, &invalid);
            if (invalid || !data) {
                break;
            }
            if (*data == ';') {
                return loc.getLocWithOffset(1);
            }
            if (*data != ' ' && *data != '\t' && *data != '\r' &&
                *data != '\n') {
                break;
            }
        }

        return afterToken;
    }
};

}  // namespace

void rewritePrintCallsRootOnly(clang::Rewriter* rewriter,
                               clang::TranslationUnitDecl* tuDecl) {
    if (!rewriter || !tuDecl) {
        return;
    }
    clang::SourceLocation insertLoc;
    const clang::SourceManager& sourceManager = rewriter->getSourceMgr();
    for (const clang::Decl* decl : tuDecl->decls()) {
        if (!decl) {
            continue;
        }
        clang::SourceLocation loc = decl->getBeginLoc();
        if (loc.isValid() && sourceManager.isWrittenInMainFile(loc)) {
            insertLoc = loc;
            break;
        }
    }
    if (insertLoc.isValid()) {
        rewriter->InsertText(insertLoc,
                             "static inline bool __dacpp_mpi_is_root_rank();\n");
    }
    RootOnlyPrintRewriteVisitor visitor(rewriter, &tuDecl->getASTContext());
    visitor.TraverseDecl(tuDecl);
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
