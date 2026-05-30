#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"

#include <string>
#include <vector>

#include "ASTParse.h"
#include "Param.h"

using namespace clang;

std::string dacppTranslator::stmt2String(Stmt *stmt) {
    clang::LangOptions lo;
    std::string out_str;
    llvm::raw_string_ostream outstream(out_str);
    stmt->printPretty(outstream, NULL, PrintingPolicy(lo));
    return out_str;
}

void dacppTranslator::getSplitExpr(Expr* curExpr, std::string& name, std::vector<Expr*>& splits) {
    if(!curExpr) { return; }
    CXXOperatorCallExpr* splitExpr = nullptr;
    if(!isa<CXXOperatorCallExpr>(curExpr)) {
        splitExpr = dacppTranslator::getNode<CXXOperatorCallExpr>(curExpr);
    }
    else {
        splitExpr = dyn_cast<CXXOperatorCallExpr>(curExpr);
    }
    getSplitExpr(getNode<CXXOperatorCallExpr>(splitExpr), name, splits);
    int count = 0;
    for(Stmt::child_iterator it = splitExpr->child_begin(); it != splitExpr->child_end(); it++) {
        if(count == 1 && !getNode<CXXOperatorCallExpr>(splitExpr)) {
            DeclRefExpr* tmp = isa<DeclRefExpr>(*it) ? dyn_cast<DeclRefExpr>(*it) : getNodeBFS<DeclRefExpr>(*it);
            name =  tmp->getNameInfo().getAsString();
        } else if(count == 2) {
            splits.push_back(dyn_cast<Expr>(*it));
        }
        count++;
    }
}

dacppTranslator::IOTYPE
dacppTranslator::inputOrOutput(const clang::ParmVarDecl* param) {

    for (const auto* attr : param->attrs()) {
        if (const auto* ann = llvm::dyn_cast<clang::AnnotateAttr>(attr)) {
            std::string tag = ann->getAnnotation().str();
            if (tag == "read")
                return IOTYPE::READ;
            if (tag == "write")
                return IOTYPE::WRITE;
            if (tag == "read_write")
                return IOTYPE::READ_WRITE;
        }
    }

    QualType QT = param->getType();

    QualType baseType = QT.getNonReferenceType();

    if (baseType.isConstQualified())
        return IOTYPE::READ;

    return IOTYPE::WRITE;
}

