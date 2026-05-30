#ifndef ASTPARSE_H
#define ASTPARSE_H

#include <vector>
#include <queue>

#include "clang/AST/AST.h"

#include "Param.h"

using namespace clang;

namespace dacppTranslator {

// Depth-first search for the first matching child node.
template <typename NodeType>
NodeType* getNode(Stmt* curStmt) {
    if(!curStmt) return nullptr;
    for(Stmt::child_iterator it = curStmt->child_begin(); it != curStmt->child_end(); it++) {
        if(isa<NodeType>(*it)) return dyn_cast<NodeType>(*it);
        if(getNode<NodeType>(*it)) {
            return getNode<NodeType>(*it);
        }
    }
    return nullptr;
}

// Breadth-first search variant used when shallow matches are preferred.
template <typename NodeType>
NodeType* getNodeBFS(Stmt* curStmt) {
    std::queue<Stmt*> q;
    q.push(curStmt);
    while(!q.empty()) {
        Stmt* cur = q.front();
        q.pop();
        for(Stmt::child_iterator it = cur->child_begin(); it != cur->child_end(); it++) {
            if(isa<NodeType>(*it)) {
                return dyn_cast<NodeType>(*it);
            }
            q.push(*it);
        }
    }
    return nullptr;
}

void getSplitExpr(Expr* curExpr, std::string& name, std::vector<Expr*>& splits);

std::string stmt2String(Stmt *stmt);

IOTYPE inputOrOutput(const clang::ParmVarDecl* param);

}

#endif
