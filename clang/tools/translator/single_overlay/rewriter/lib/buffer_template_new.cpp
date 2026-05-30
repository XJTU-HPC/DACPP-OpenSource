#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "buffer_template_new.h"
#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"
#include "DacppStructure.h"
#include <regex>
#include "dac_Reduction.h"
#include <set>
#include <sstream>
#include "Rewriter.h"
#include "Split.h"
#include "Param.h"
#include "dacInfo.h"
#include "Calc.h"
#include "ASTParse.h"

namespace BUFFER_TEMPLATE {
// Expand reduction_max(...) into an explicit SYCL reduction submission.
bool max_generate(std::string& reductionText, std::string original,
                  dacppTranslator::DacppFile dacFile) {

    std::string accText = "";
    std::size_t start = original.find("reduction_max");
    if (start == std::string::npos) return false;
    start += std::string("reduction_max(").length();

    int paren_depth = 0;
    std::size_t expr_end = start;
    for (std::size_t pos = start; pos < original.size(); ++pos) {
        if (original[pos] == '(') paren_depth++;
        else if (original[pos] == ')') paren_depth--;
        else if (original[pos] == ',' && paren_depth == 0) {
            expr_end = pos;
            break;
        }
    }
    std::string expr = original.substr(start, expr_end - start);
    std::vector<std::size_t> comma_pos;
    paren_depth = 0;
    for (std::size_t pos = start; pos < original.size(); ++pos) {
        if (original[pos] == '(') paren_depth++;
        else if (original[pos] == ')') paren_depth--;
        else if (original[pos] == ',' && paren_depth == 0) comma_pos.push_back(pos);
        if (comma_pos.size() >= 3) break;
    }

    std::string var_i = original.substr(comma_pos[0] + 1, comma_pos[1] - comma_pos[0] - 1);
    std::string initial_value_name = original.substr(comma_pos[1] + 1, comma_pos[2] - comma_pos[1] - 1);
    std::string N_name = original.substr(comma_pos[2] + 1);
    N_name = N_name.substr(0, N_name.find(')'));

    auto trim = [](std::string &s){
        size_t first = s.find_first_not_of(" \t\n");
        size_t last  = s.find_last_not_of(" \t\n");
        if(first == std::string::npos) s="";
        else s = s.substr(first,last-first+1);
    };
    trim(var_i); trim(initial_value_name); trim(N_name);
    std::vector<std::string> name;
    for (const auto& var_pair : dacFile.getForStatementVars()) {
        const std::string& var = var_pair.first;
        std::size_t pos = expr.find(var);
        while (pos != std::string::npos) {
            bool left_ok = (pos == 0) || (!std::isalnum(expr[pos-1]) && expr[pos-1] != '_');
            bool right_ok = (pos + var.length() == expr.size()) ||
                             (!std::isalnum(expr[pos + var.length()]) && expr[pos + var.length()] != '_');
            if (left_ok && right_ok) {
                name.push_back(var);
                break;
            }
            pos = expr.find(var, pos + 1);
        }
    }
    reductionText += " q.submit([&](sycl::handler &h) {\n";

for (const auto &var : name) {
    int i=0;
    dacppTranslator::Expression* expr = dacFile.getExpression(i);
    dacppTranslator::Shell* shell = expr->getShell();
    dacppTranslator::Calc* calc = expr->getCalc();
    bool flag = false;    bool found = false;
    for(int count = 0; count < shell->getNumParams(); count++) {
        dacppTranslator::Param* param = shell->getParam(count);
        if (param->getName() == var) {
            found = true;
            if (param->getRw()!= dacppTranslator::IOTYPE::READ) {
                flag = true;
                break;
            }
        }
    }
    if (!flag && found) {
        reductionText +=
            "    auto acc_" + var +
            " = r_" + var +
            ".get_access<sycl::access::mode::read>(h);\n";
    } else if(found&&flag){
        reductionText +=
            "    auto acc_" + var +
            " = r_" + var +
            "->get_access<sycl::access::mode::read_write>(h);\n";

    }
    else{
        reductionText +=
            "    auto acc_" + var +
            " = r_" + var +
            ".get_access<sycl::access::mode::read_write>(h);\n";
    }
}
    std::string max_buf = "r_max_error";
    reductionText += "        h.parallel_for(\n";
    reductionText += "            sycl::range<1>(" + N_name + "),\n";
    reductionText += "            sycl::reduction(" + max_buf + ", h, sycl::maximum<float>(), sycl::property::reduction::initialize_to_identity()),\n";
    reductionText += "            [=](sycl::id<1> idx, auto &reducer) {\n";
    std::string val_expr = expr;
for (const auto &var : name) {
    size_t pos = 0;
    while ((pos = val_expr.find(var, pos)) != std::string::npos) {
        bool left_ok = (pos == 0) || (!std::isalnum(val_expr[pos-1]) && val_expr[pos-1] != '_');
        bool right_ok = (pos + var.length() < val_expr.size()) &&
                        val_expr[pos + var.length()] == '[';

        if (!left_ok || !right_ok) {
            pos += var.length();
            continue;
        }
        size_t lbracket = pos + var.length();
        size_t rbracket = lbracket;
        int depth = 0;
        for (; rbracket < val_expr.size(); ++rbracket) {
            if (val_expr[rbracket] == '[') depth++;
            else if (val_expr[rbracket] == ']') {
                depth--;
                if (depth == 0) break;
            }
        }
        if (rbracket >= val_expr.size()) break;
        val_expr.replace(pos, rbracket - pos + 1,
                         "acc_" + var + "[idx]");
        pos += var.length() + 6;    }
}


    reductionText += "                float val = " + val_expr + ";\n";
    reductionText += "                reducer.combine(val);\n";
    reductionText += "            }\n";
    reductionText += "        );\n";
    reductionText += "    }).wait();\n";
    return true;
}
bool replaceReductionMax(std::string& fortext, std::string& reductionText,
                         dacppTranslator::DacppFile dacFile) {

    const std::string key = "reduction_max";
    std::string original;

    std::size_t pos = fortext.find(key);
    if (pos == std::string::npos) return false;

    std::size_t stmt_begin = fortext.rfind(';', pos);
    if (stmt_begin == std::string::npos) stmt_begin = 0;
    else ++stmt_begin;

    std::size_t stmt_end = fortext.find(';', pos);
    if (stmt_end == std::string::npos) return false;
    ++stmt_end;

    original = fortext.substr(stmt_begin, stmt_end - stmt_begin);

    max_generate(reductionText, original, dacFile);

    fortext.replace(stmt_begin, stmt_end - stmt_begin, reductionText);

    return true;
}

void replaceTextInString(std::string& text,
    const std::string &find,
    const std::string &replace){
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos){
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}
std::string templateString(std::string templ,
    std::vector<std::pair<std::string, std::string>> replacements){
	for(auto &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}
std::string BUFFER_ACCESSOR_LIST = "";
std::string ACCESSOR_POINTER_LIST = "";
const char *BUFFER_ACCESSOR_Template = R"~~~(
        accessor acc_{{NAME}}{r_{{NAME}}, h};)~~~";
const char *ACCESSOR_POINTER_Template = R"~~~(
            auto* d_{{NAME}} = acc_{{NAME}}.get_multi_ptr<access::decorated::no>().get();)~~~";
const char *DAC2SYCL_Template_2 = R"~~~(
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) {
    auto selector = default_selector_v;
    sycl::queue q(selector);
    ParameterGeneration para_gene_tool;
    {{OP_INIT}}
	{{ParameterGenerate}}
    {{DEVICE_MEM_ALLOC}}
    {{DATA_ASSOC_COMP}}
})~~~";

std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams,std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp){
    return templateString(DAC2SYCL_Template_2,
	{
		{"{{DAC_SHELL_NAME}}",    dacShellName},
		{"{{DAC_SHELL_PARAMS}}",  dacShellParams},
		{"{{OP_INIT}}",           opInit},
		{"{{ParameterGenerate}}", parameter_generate},
		{"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
		{"{{DATA_ASSOC_COMP}}",   dataAssocComp}
	});
}
bool containsNestedFor(const clang::ForStmt* FS) {
    using namespace clang;
    if (!FS) return false;

    class NestedForVisitor : public RecursiveASTVisitor<NestedForVisitor> {
    public:
        bool foundNested = false;
        const ForStmt* outer;

        NestedForVisitor(const ForStmt* outer) : outer(outer) {}

        bool VisitForStmt(ForStmt* innerFS) {
            if (innerFS != outer) {
                foundNested = true;
            }
            return true;
        }
    };

    NestedForVisitor v(FS);
    v.TraverseStmt(const_cast<clang::ForStmt*>(FS));
    return v.foundNested;
}
static bool forContainsDacExpr(
    const clang::ForStmt* FS,
    const clang::BinaryOperator* dacExpr,
    const clang::ASTContext* Context)
{
    if (!FS || !dacExpr || !Context) return false;
    const auto &SM = Context->getSourceManager();
    clang::SourceRange fsRange  = FS->getSourceRange();
    clang::SourceRange dacRange = dacExpr->getSourceRange();

    auto leq = [&](clang::SourceLocation A, clang::SourceLocation B){
        return A == B || SM.isBeforeInTranslationUnit(A, B);
    };

    bool beginOK = leq(fsRange.getBegin(), dacRange.getBegin());
    bool endOK   = leq(dacRange.getEnd(), fsRange.getEnd());
    return beginOK && endOK;
}
static std::string linearizeND(
    const std::string& body,
    const std::vector<std::string>& dvars
){
    std::string out = body;

    for (const auto& v : dvars) {
        std::string raw_name = v.substr(2);
        std::string shape_array = "info_" + raw_name + "_Shape";

        size_t start_pos = 0;
        while ((start_pos = out.find(v + "[", start_pos)) != std::string::npos) {
            if (start_pos > 0 && (std::isalnum(out[start_pos - 1]) || out[start_pos - 1] == '_')) {
                start_pos += v.length();
                continue;
            }

            std::vector<std::string> indices;
            size_t curr_pos = start_pos + v.length();
            size_t end_pos = curr_pos;
            while (curr_pos < out.length() && out[curr_pos] == '[') {
                int depth = 0;
                size_t bracket_start = curr_pos;
                size_t bracket_end = bracket_start;
                for (; bracket_end < out.length(); ++bracket_end) {
                    if (out[bracket_end] == '[') depth++;
                    else if (out[bracket_end] == ']') {
                        depth--;
                        if (depth == 0) break;
                    }
                }
                if (bracket_end < out.length() && depth == 0) {
                    indices.push_back(out.substr(bracket_start + 1, bracket_end - bracket_start - 1));
                    curr_pos = bracket_end + 1;
                    while (curr_pos < out.length() && std::isspace(out[curr_pos])) {
                        curr_pos++;
                    }
                    end_pos = curr_pos;
                } else {
                    break;                }
            }
            if (indices.size() > 1) {
                std::string flat_index = "";
                for (size_t d = 0; d < indices.size(); ++d) {
                    std::string term = "(" + indices[d] + ")";
                    for (size_t s = d + 1; s < indices.size(); ++s) {
                        term += " * " + shape_array + "[" + std::to_string(s) + "]";
                    }
                    if (d > 0) flat_index += " + ";
                    flat_index += term;
                }

                std::string replacement = v + "[" + flat_index + "]";
                out.replace(start_pos, end_pos - start_pos, replacement);
                start_pos += replacement.length();            } else {
                start_pos = end_pos;            }
        }
        std::regex scalarPat(
            "\\b" + v + "\\b"
            "(?!\\s*\\[)"            "(?!\\s*->)"            "(?!\\s*\\.)"        );
        std::regex alreadyDeref("\\*\\s*" + v);
        if (!std::regex_search(out, alreadyDeref)) {
            out = std::regex_replace(out, scalarPat, "*" + v);
        }
    }

    return out;
}
std::string parallelizeNestedFor(
    const clang::ForStmt* outerFS,
    const clang::ASTContext* Context,
    dacppTranslator::DacppFile* dacFile){
    using namespace clang;
    const LangOptions& LO = Context->getLangOpts();
    const SourceManager& SM = Context->getSourceManager();
    std::string iVar="i", iL="0", iR="0";
    if (auto* DS = dyn_cast<const DeclStmt>(outerFS->getInit())) {
        if (auto* VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
            iVar = VD->getNameAsString();
            if (VD->getInit())
                iL = Lexer::getSourceText(
                    CharSourceRange::getTokenRange(VD->getInit()->getSourceRange()),
                    SM, LO).str();
        }
    }
    if (auto* C = dyn_cast<BinaryOperator>(outerFS->getCond())) {
        iR = Lexer::getSourceText(
            CharSourceRange::getTokenRange(C->getRHS()->getSourceRange()),
            SM, LO).str();
    }
    const ForStmt* innerFS = nullptr;
    if (auto* CS = dyn_cast<CompoundStmt>(outerFS->getBody())) {
        for (auto* child : CS->body()) {
            if (auto* fs = dyn_cast<ForStmt>(child)) {
                innerFS = fs;
                break;
            }
        }
    }
    if (!innerFS)
        return "";    std::string jVar="j", jL="0", jR="0";
    if (auto* DS = dyn_cast<const DeclStmt>(innerFS->getInit())) {
        if (auto* VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
            jVar = VD->getNameAsString();
            if (VD->getInit())
                jL = Lexer::getSourceText(
                    CharSourceRange::getTokenRange(VD->getInit()->getSourceRange()),
                    SM, LO).str();
        }
    }
    if (auto* C = dyn_cast<BinaryOperator>(innerFS->getCond())) {
        jR = Lexer::getSourceText(
            CharSourceRange::getTokenRange(C->getRHS()->getSourceRange()),
            SM, LO).str();
    }
    std::string bodyText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(innerFS->getBody()->getSourceRange()),
        SM, LO).str();
    auto vars = dacFile->getForStatementVars();
    std::string accessorDecl;
    std::string ptrDecl;
    std::string replacedBody = bodyText;


    for (auto &p : vars) {
    const std::string& name = p.first;
    const std::string& type = p.second;
    std::regex wordExpr("\\b" + name + "\\b");
    if (!std::regex_search(replacedBody, wordExpr)) {
        continue;    }
    bool isConst = (type.find("const") != std::string::npos);
    if (isConst) {
        replacedBody = std::regex_replace(replacedBody, wordExpr, name);
        continue;
    }
    bool flag = false;    int i=0;
    bool found = false;
    dacppTranslator::Expression* expr = dacFile->getExpression(i);
    dacppTranslator::Shell* shell = expr->getShell();
    for(int count = 0; count < shell->getNumParams(); count++) {
        dacppTranslator::Param* param = shell->getParam(count);
        if (param->getName() == name) {
            found = true;
            if (param->getRw() != dacppTranslator::IOTYPE::READ){
                flag = true;
                break;
            }
        }
    }
if (!flag && found){
    accessorDecl +=
        "    auto acc_" + name +
        " = r_" + name +
        ".get_access<sycl::access::mode::read>(h);\n";
} else if(found&&flag){
    accessorDecl +=
        "    auto acc_" + name +
        " = r_" + name +
        "->get_access<sycl::access::mode::read_write>(h);\n";
}else{
    accessorDecl +=
        "    auto acc_" + name +
        " = r_" + name +
        ".get_access<sycl::access::mode::read_write>(h);\n";
}
    ptrDecl +=
        "      auto* d_" + name +
        " = acc_" + name +
        ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    replacedBody = std::regex_replace(
        replacedBody,
        wordExpr,
        "d_" + name
    );
}
    std::vector<std::string> dvars;
    for (auto &p : vars) {
        const std::string& name = p.first;
        const std::string& type = p.second;
        bool isConst = (type.find("const") != std::string::npos);
        if (isConst) continue;

        std::string dname = "d_" + name;
        std::regex w("\\b" + dname + "\\b");
        if (std::regex_search(replacedBody, w))
            dvars.push_back(dname);
    }
    replacedBody = linearizeND(replacedBody, dvars);
std::string iSuffix = "";
std::string jSuffix = "";
if (auto* biOp = dyn_cast<BinaryOperator>(outerFS->getCond())) {    if (biOp->getOpcode() == BO_LE) {
        iSuffix = " + 1";
    }
}
if (innerFS) {    if (auto* biOp = dyn_cast<BinaryOperator>(innerFS->getCond())) {        if (biOp->getOpcode() == BO_LE) {
            jSuffix = " + 1";
        }
    }
}
    std::string code;
    code += "{\n";
    code += "    int __iL = (" + iL + ");\n";
    code += "    int __iR = (" + iR + ");\n";
    code += "    int __iN = __iR - __iL" + iSuffix + ";\n";
    code += "    int __jL = (" + jL + ");\n";
    code += "    int __jR = (" + jR + ");\n";
    code += "    int __jN = __jR - __jL" + jSuffix + ";\n";

    code += "    q.submit([&](sycl::handler& h){\n";
    code += accessorDecl;

    code += "    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){\n";
    code += ptrDecl;

    code += "      int " + iVar + " = __iL + idx[0];\n";
    code += "      int " + jVar + " = __jL + idx[1];\n";
    code += "      " + replacedBody + "\n";

    code += "    });\n";
    code += "  });\n";
    code += "}\n";

    return code;
}

std::string parallelizeSingleFor(const clang::ForStmt* FS,clang::ASTContext* Context,dacppTranslator::DacppFile* dacFile)
{
    using namespace clang;

    if (containsNestedFor(FS)) {
        std::string nested = parallelizeNestedFor(FS, Context, dacFile);        if (!nested.empty())
            return nested;
    }

    const SourceManager& SM = Context->getSourceManager();
    const LangOptions& LO = Context->getLangOpts();

    std::string loopVar="i";
    std::string L="0", R="0";

    if (auto* DS = dyn_cast<const DeclStmt>(FS->getInit())) {
        if (auto* VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
            loopVar = VD->getNameAsString();
            if (VD->getInit())
                L = Lexer::getSourceText(
                    CharSourceRange::getTokenRange(VD->getInit()->getSourceRange()),
                    SM, LO);
        }
    }

    if (auto* C = dyn_cast<BinaryOperator>(FS->getCond())) {
        R = Lexer::getSourceText(
            CharSourceRange::getTokenRange(C->getRHS()->getSourceRange()),
            SM, LO);
    }

    std::string bodyText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(FS->getBody()->getSourceRange()),
        SM, LO).str();
    auto vars = dacFile->getForStatementVars();

    std::string accessorDecl;
    std::string ptrDecl;
    std::string replacedBody = bodyText;

    for (auto &p : vars) {
    const std::string& name = p.first;
    const std::string& type = p.second;
    std::regex wordExpr("\\b" + name + "\\b");
    if (!std::regex_search(replacedBody, wordExpr)) {
        continue;    }

    bool flag = true;
    bool found = false;
        int i=0;
        dacppTranslator::Expression* expr = dacFile->getExpression(i);
        dacppTranslator::Shell* shell = expr->getShell();

        for(int count = 0; count < shell->getNumParams(); count++) {
            if (shell->getParam(count)->getName() == name) {
                found = true;
                if (shell->getParam(count)->getRw() == dacppTranslator::IOTYPE::READ){
                    flag = false;
                    break;
                }
            }
        }
    if (!flag && found) {
    accessorDecl +=
        "    auto acc_" + name +
        " = r_" + name +
        ".get_access<sycl::access::mode::read>(h);\n";
    } else if(found&&flag){
    accessorDecl +=
        "    auto acc_" + name +
        " = r_" + name +
        "->get_access<sycl::access::mode::read_write>(h);\n";
    }
    else{
    accessorDecl +=
        "    auto acc_" + name +
        " = r_" + name +
        ".get_access<sycl::access::mode::read_write>(h);\n";
    }
    ptrDecl +=
        "      auto* d_" + name +
        " = acc_" + name +
        ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";
    replacedBody = std::regex_replace(
        replacedBody,
        wordExpr,
        "d_" + name
    );
    }
std::vector<std::string> dvars;
for (auto &p : vars) {
    const std::string& name = p.first;
    const std::string& type = p.second;
    if (type.find("const") != std::string::npos) continue;

    std::string dname = "d_" + name;
    std::regex w("\\b" + dname + "\\b");
    if (std::regex_search(replacedBody, w)) {
        dvars.push_back(dname);
    }
}
replacedBody = linearizeND(replacedBody, dvars);
std::string nSuffix = "";
if (auto* biOp = dyn_cast<BinaryOperator>(FS->getCond())) {
    if (biOp->getOpcode() == BO_LE) {
        nSuffix = " + 1";
    }
}
    std::string code;
    code += "{\n";
    code += "  int __L = (" + L + ");\n";
    code += "  int __R = (" + R + ");\n";
    code += "  int __N = __R - __L" + nSuffix +";\n";

    code += "  q.submit([&](sycl::handler& h){\n";
    code += accessorDecl;
    code += "    h.parallel_for(sycl::range<1>(__N), [=](sycl::id<1> idx){\n";
    code += ptrDecl;
    code += "      int " + loopVar + " = __L + idx[0];\n";
    code += "      " + replacedBody + "\n";
    code += "    });\n";
    code += "  });\n";
    code += "}\n";

    return code;
}
static void dfsCollectFors(
    const clang::Stmt* S,
    const clang::ForStmt* outerFor,
    const clang::BinaryOperator* dacExpr,
    clang::ASTContext* Context,
    const clang::ForStmt* currentTop,    std::vector<const clang::ForStmt*>& result
) {
    using namespace clang;
    if (!S) return;

    if (const auto* FS = llvm::dyn_cast<ForStmt>(S)) {
        if (FS == outerFor) {
            dfsCollectFors(FS->getBody(), outerFor, dacExpr, Context, nullptr, result);
            return;
        }
        if (dacExpr && forContainsDacExpr(FS, dacExpr, Context)) {
            dfsCollectFors(FS->getBody(), outerFor, dacExpr, Context, currentTop, result);
            return;
        }

        if (currentTop == nullptr) {
            result.push_back(FS);
            dfsCollectFors(FS->getBody(), outerFor, dacExpr, Context, FS, result);
        } else {
            dfsCollectFors(FS->getBody(), outerFor, dacExpr, Context, currentTop, result);
        }
        return;
    }
    if (const auto* CS = llvm::dyn_cast<CompoundStmt>(S)) {
        for (const Stmt* child : CS->body()) {
            dfsCollectFors(child, outerFor, dacExpr, Context, currentTop, result);
        }
        return;
    }
    for (const Stmt* child : S->children()) {
        dfsCollectFors(child, outerFor, dacExpr, Context, currentTop, result);
    }
}
std::vector<const clang::ForStmt*> collectInnerForsExceptDacpp(
    const clang::ForStmt* outerFor,
    const clang::BinaryOperator* dacExpr,
    clang::ASTContext* Context
){
    using namespace clang;
    std::vector<const ForStmt*> result;
    if (!outerFor || !Context) return result;

    const Stmt* body = outerFor->getBody();
    if (!body) return result;

    dfsCollectFors(body, outerFor, dacExpr, Context, nullptr, result);
    return result;
}
std::string rewriteSiblingForsToParallel(
    const std::string& originalForText,
    const clang::ForStmt* outerFor,
    const clang::BinaryOperator* dacExpr,
    clang::ASTContext* Context,
    dacppTranslator::DacppFile* dacFile){
    std::string newText = originalForText;
    auto targets = collectInnerForsExceptDacpp(outerFor, dacExpr, Context);
    std::sort(
        targets.begin(), targets.end(),
        [&](auto* A, auto* B){
            const auto &SM = Context->getSourceManager();
            return SM.isBeforeInTranslationUnit(B->getBeginLoc(), A->getBeginLoc());
        }
    );

    for (auto* FS : targets) {
        std::string oldFS = clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(FS->getSourceRange()),
            Context->getSourceManager(),
            Context->getLangOpts()
        ).str();
        std::string newFS = parallelizeSingleFor(FS, Context, dacFile);
        size_t pos = newText.find(oldFS);
        if (pos != std::string::npos) {
            newText.replace(pos, oldFS.size(), newFS);
        }
    }

    return newText;
}
const char *DATA_INFO_INIT_Template = R"~~~(
    DataInfo info_{{NAME}};
    info_{{NAME}}.dim = {{NAME}}.getDim();
    int info_{{NAME}}_Shape[{{DIM}}] = {0};
    for(int i = 0; i < info_{{NAME}}.dim; i++)
    {
        info_{{NAME}}.dimLength.push_back({{NAME}}.getShape(i));
        info_{{NAME}}_Shape[i] = {{NAME}}.getShape(i);
    }
	)~~~";
std::string CodeGen_DataInfoInit(std::string name, std::string dim){
    return templateString(DATA_INFO_INIT_Template,
	{
		{"{{NAME}}",    name},
        {"{{DIM}}",    dim}
	});
}

std::string generateBufferCode(dacppTranslator::DacppFile* dacppFile) {
    std::string code;

    if (!dacppFile) return code;
    std::vector<std::pair<std::string, std::string>> forStatementVars = dacppFile->getForStatementVars();

for (const auto& var : forStatementVars) {
    const std::string &name = var.first;
    const std::string &type = var.second;
    bool inShellVars = false;
    for (const auto &shellVar : dacppFile->shellVars) {
        if (shellVar.first == name) {
            inShellVars = true;
            break;
        }
    }
    if (inShellVars) {
        continue;
    }
    std::string bufferName = "r_" + name;
    code += "    sycl::buffer<" + type + ", 1> "
          + bufferName
          + "(&" + name + ", sycl::range<1>(1));\n";
}


    return code;
}
std::string extractAndReplaceSubmitCodeGeneric(
    const clang::ForStmt* forStatement,
    clang::ASTContext* Context,
    dacppTranslator::DacppFile* dacFile) {
    using namespace clang;
    if (!forStatement || !Context || !dacFile) return "";

    SourceManager &SM           = Context->getSourceManager();
    const LangOptions &LangOpts = Context->getLangOpts();
    class FindDacppOpVisitor : public RecursiveASTVisitor<FindDacppOpVisitor> {
    public:
        ASTContext *Ctx;
        std::vector<const BinaryOperator*> targets;

        FindDacppOpVisitor(ASTContext *ctx) : Ctx(ctx) {}

        bool VisitBinaryOperator(BinaryOperator *op) {
            if (!op || !Ctx) return true;

            std::string opText = Lexer::getSourceText(
                CharSourceRange::getTokenRange(op->getSourceRange()),
                Ctx->getSourceManager(),
                Ctx->getLangOpts()
            ).str();

            if (opText.find("<->") != std::string::npos)
                targets.push_back(op);

            return true;
        }
    };

    FindDacppOpVisitor visitor(Context);
    visitor.TraverseStmt(const_cast<clang::ForStmt*>(forStatement));

    if (visitor.targets.empty()) {
        llvm::errs() << "[extractAndReplaceSubmitCodeGeneric] No DACPP <-> expression found.\n";
        return "";
    }
    std::string forText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(forStatement->getSourceRange()),
        SM, LangOpts
    ).str();
    forText = generateBufferCode(dacFile) + forText;
    std::string reductionText="";
    replaceReductionMax(forText, reductionText,*dacFile);
    forText = rewriteSiblingForsToParallel(
        forText,
        forStatement,
        visitor.targets[0],
        Context,
        dacFile    );
    for (auto *op : visitor.targets) {
        std::string opText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(op->getSourceRange()),
            SM, LangOpts
        ).str();

        size_t pos = forText.find(opText);
        if (pos != std::string::npos)
            forText.replace(pos, opText.length(), "{{SUBMIT}}");
        else
            llvm::errs() << "[extractAndReplaceSubmitCodeGeneric] Warning: a DACPP <-> expression was not found in the loop body text.\n";
    }

    return forText;
}
const char *OP_REGULAR_SLICE_INIT_Template2 = R"~~~(
    RegularSlice {{OP_NAME}} = RegularSlice("{{OP_NAME}}", {{SIZE}}, {{STRIDE}});
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.SetSplitSize(para_gene_tool.init_operetor_splitnumber({{OP_NAME}},{{DATA_INFO_NAME}}));
)~~~";

std::string CodeGen_RegularSliceInit2(std::string opName,std::string size,std::string stride,std::string dim_id,std::string DATA_INFO_NAME){
    return templateString(OP_REGULAR_SLICE_INIT_Template2,
	{
		{"{{OP_NAME}}",    opName},
		{"{{SIZE}}",       size},
		{"{{STRIDE}}",     stride},
		{"{{DIM_ID}}",     dim_id},		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME}
	});
}
const char *OP_INDEX_INIT_Template2 = R"~~~(
    Index {{OP_NAME}} = Index("{{OP_NAME}}");
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.SetSplitSize(para_gene_tool.init_operetor_splitnumber({{OP_NAME}},{{DATA_INFO_NAME}}));
)~~~";

std::string CodeGen_IndexInit2(std::string opName,std::string dim_id,std::string DATA_INFO_NAME){
    return templateString(OP_INDEX_INIT_Template2,
	{
		{"{{OP_NAME}}",    opName},
		{"{{DIM_ID}}", dim_id},		{"{{DATA_INFO_NAME}}", DATA_INFO_NAME}
	});
}
const char *PARA_GENE_Template = R"~~~(
	{{InitOPS}}
	{{InitDeviceMemorySize}}
	{{InitSplitLength}}
	{{ItemNumber}}
)~~~";

std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string ItemNumber){
    return templateString(PARA_GENE_Template,
	{
		{"{{InitOPS}}", InitOPS},
		{"{{InitDeviceMemorySize}}", InitDeviceMemorySize},		{"{{InitSplitLength}}",InitSplitLength},
		{"{{ItemNumber}}",ItemNumber},
	});
}
const char *OPS_INIT_Template = R"~~~(
    Dac_Ops {{OPS_NAME}};
    {{ADD_OP2OPS}}
)~~~";

std::string CodeGen_DataOpsInit2(std::string OPS_NAME,std::string ADD_OP2OPS){
    return templateString(OPS_INIT_Template,
	{
		{"{{OPS_NAME}}",       OPS_NAME},
		{"{{ADD_OP2OPS}}",    ADD_OP2OPS}
	});
}
const char *ADD_OP2OPS_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OPS_NAME}}.push_back({{OP_NAME}});
)~~~";

std::string CodeGen_AddOp2Ops(std::string OP_NAME,std::string DIM_ID,std::string OPS_NAME){
    return templateString(ADD_OP2OPS_Template,
	{
		{"{{OP_NAME}}",    OP_NAME},
		{"{{DIM_ID}}",     DIM_ID},
		{"{{OPS_NAME}}",   OPS_NAME}
	});
}
const char *DEVICE_MEM_SIZE_Generate_Template1 = R"~~~(
    int {{NAME}} = para_gene_tool.init_device_memory_size({{DATA_INFO_NAME}},{{DACOPS_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME,std::string DACOPS_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template1,
	{
        {"{{NAME}}",        NAME},		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME},		{"{{DACOPS_NAME}}",        DACOPS_NAME}	});
}
const char *DEVICE_MEM_SIZE_Generate_Template2 = R"~~~(
    int {{NAME}} = para_gene_tool.init_device_memory_size({{DATA_INFO_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template2,
	{
        {"{{NAME}}",        NAME},		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME}	});
}
const char *DEVICE_MEM_SIZE_Generate_Template3 = R"~~~(
    int {{NAME}} = para_gene_tool.init_device_memory_size({{IN_DAC_OPS_NAME}},{{OUT_DAC_OPS_NAME}},{{DATA_INFO_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string IN_DAC_OPS_NAME,std::string OUT_DAC_OPS_NAME,std::string DATA_INFO_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template3,
	{
		{"{{NAME}}",            NAME},		{"{{IN_DAC_OPS_NAME}}", IN_DAC_OPS_NAME},		{"{{OUT_DAC_OPS_NAME}}",OUT_DAC_OPS_NAME},		{"{{DATA_INFO_NAME}}",      DATA_INFO_NAME}	});
}
const char *INIT_SPLIT_LENGTH_Template = R"~~~(
    para_gene_tool.init_op_split_length({{OPS_NAME}},{{SIZE}});
)~~~";

std::string CodeGen_Init_Split_Length(std::string OPS_NAME,std::string SIZE){
    return templateString(INIT_SPLIT_LENGTH_Template,
	{
		{"{{OPS_NAME}}",       OPS_NAME},
		{"{{SIZE}}",           SIZE}	});
}
const char *INIT_SPLIT_LENGTH_MATRIX_Template = R"~~~(
	{{DECLARE_DACOPS_VECTOR}}
    int SplitLength[{{ROW}}][{{COL}}] = {0};
    para_gene_tool.init_split_length_martix({{ROW}},{{COL}},&SplitLength[0][0],{{OPS_S_NAME}});
)~~~";

std::string CodeGen_Init_Split_Length_Matrix(std::string DECLARE_DACOPS_VECTOR,std::string ROW,std::string COL,std::string OPS_S_NAME){
    return templateString(INIT_SPLIT_LENGTH_MATRIX_Template,
	{
		{"{{DECLARE_DACOPS_VECTOR}}",       DECLARE_DACOPS_VECTOR},
		{"{{ROW}}",       ROW},		{"{{COL}}",       COL},		{"{{OPS_S_NAME}}",       OPS_S_NAME}	});
}
const char *DECLARE_DACOPS_VECTOR_Template = R"~~~(
    std::vector<Dac_Ops> {{OPSS_NAME}};
	{{PUSH_BACK_DAC_OPS}}
)~~~";

std::string CodeGen_Declare_DacOps_Vector(std::string OPSS_NAME,std::string PUSH_BACK_DAC_OPS){
    return templateString(DECLARE_DACOPS_VECTOR_Template,
	{
		{"{{OPSS_NAME}}",           OPSS_NAME},		{"{{PUSH_BACK_DAC_OPS}}",   PUSH_BACK_DAC_OPS}	});
}
const char *ADD_DACOPS2VECTOR_Template = R"~~~(
    {{OPSS_NAME}}.push_back({{OPS_NAME}});
)~~~";

std::string CodeGen_Add_DacOps2Vector(std::string OPSS_NAME,std::string OPS_NAME){
    return templateString(ADD_DACOPS2VECTOR_Template,
	{
		{"{{OPSS_NAME}}",       OPSS_NAME},		{"{{OPS_NAME}}",         OPS_NAME}	});
}
const char *INIT_WORK_ITEM_NUMBER_Template = R"~~~(
    int {{NAME}} = para_gene_tool.init_work_item_size({{OPS_NAME}});
)~~~";

std::string CodeGen_Init_Work_Item_Number(std::string NAME,std::string OPS_NAME){
    return templateString(INIT_WORK_ITEM_NUMBER_Template,
	{
		{"{{NAME}}",           NAME},
		{"{{OPS_NAME}}",       OPS_NAME}	});
}
const char *DEVICE_MEM_ALLOC_Template = R"~~~(
    buffer <{{TYPE}}> b_{{NAME}}{{{SIZE}}};
)~~~";

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size){
    BUFFER_ACCESSOR_LIST += templateString(BUFFER_ACCESSOR_Template,{
        {"{{NAME}}", name}
    });
    ACCESSOR_POINTER_LIST += templateString(ACCESSOR_POINTER_Template,{
        {"{{NAME}}", name}
    });
	string s = templateString(DEVICE_MEM_ALLOC_Template,{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}});

	ACCESSOR_POINTER_LIST = " ";
    return s;
}
const char *DATA_ASSOC_COMP_Template = R"~~~(
	{{H2D_MEM_MOV}}
	{{DATA_RECON}}
	{{KERNEL_EXECUTE}}
	{{REDUCTION}}
	{{D2H_MEM_MOV}}
)~~~";

std::string CodeGen_DataAssocComp(std::string H2DMemMove, std::string dataRecon, std::string kernelExecute, std::string reduction, std::string D2HMemMove){
    return templateString(DATA_ASSOC_COMP_Template,
	{
        {"{{H2D_MEM_MOV}}",       H2DMemMove},		{"{{DATA_RECON}}",        dataRecon},
        {"{{KERNEL_EXECUTE}}",    kernelExecute},
		{"{{REDUCTION}}",         reduction},
        {"{{D2H_MEM_MOV}}",       D2HMemMove}
	});
}
const char *D2B_MOV_BUFFER_Template_READ_WRITE = R"~~~(
    {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}.getSize()*sizeof({{TYPE}}));
    {{NAME}}.tensor2Array(h_{{NAME}});
)~~~";
std::string CodeGen_D2B_Mov_Buffer_read_write(std::string TYPE,std::string NAME,std::string SIZE)
{
    return templateString(D2B_MOV_BUFFER_Template_READ_WRITE,
	{
        {"{{TYPE}}",        TYPE},
		{"{{NAME}}",        NAME},
		{"{{SIZE}}",        SIZE}
	});
}





const char *D2B_MOV_BUFFER_Template = R"~~~(
    {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}.getSize()*sizeof({{TYPE}}));
    {{NAME}}.tensor2Array(h_{{NAME}});
	buffer<{{TYPE}}, 1> r_{{NAME}}(h_{{NAME}}, range<1>({{NAME}}.getSize()));
)~~~";

std::string CodeGen_D2B_Mov_Buffer(std::string TYPE,std::string NAME,std::string SIZE)
{
    return templateString(D2B_MOV_BUFFER_Template,
	{
        {"{{TYPE}}",        TYPE},
		{"{{NAME}}",        NAME},
		{"{{SIZE}}",        SIZE}
	});
}
const char *INIT_HOST_MEMORY_Template = R"~~~(
    {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}.getSize()*sizeof({{TYPE}}));
)~~~";
std::string CodeGen_Init_Host_Memory(std::string TYPE,std::string NAME)
{
    return templateString(INIT_HOST_MEMORY_Template,
	{
        {"{{TYPE}}",        TYPE},
		{"{{NAME}}",        NAME}
	});
}
const char *DEVICE_DATA_INIT_Template = R"~~~(
    {        host_accessor temp_accessor{b_{{NAME}}};
        for(int i = 0; i < {{SIZE}}; i++){
            temp_accessor[i] = 0;
        }
    }
)~~~";

std::string CodeGen_DeviceDataInit(std::string type,std::string name,std::string size){
    return templateString(DEVICE_DATA_INIT_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *DATA_RECON_BUFFER_Template = R"~~~(

    {{DATA_OPS_INIT}}

	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
)~~~";

std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit){
    return templateString(DATA_RECON_BUFFER_Template,
	{
		{"{{TYPE}}",       type},
		{"{{NAME}}",       name},
		{"{{SIZE}}",       size},
		{"{{DATA_OPS_INIT}}", dataOpsInit}
	});
}

const char *DATA_RECON_BUFFER_Template1 = R"~~~(
    {{DATA_OPS_INIT}}

    auto r_{{NAME}} = std::make_unique<sycl::buffer<{{TYPE}}, 1>>(h_{{NAME}},sycl::range<1>({{NAME}}.getSize()));
    r_{{NAME}}->set_final_data(h_{{NAME}});

	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
    sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
)~~~";

std::string CodeGen_DataReconstruct1(std::string type,std::string name,std::string size,std::string dataOpsInit){
    return templateString(DATA_RECON_BUFFER_Template1,
	{
		{"{{TYPE}}",       type},
		{"{{NAME}}",       name},
		{"{{SIZE}}",       size},
		{"{{DATA_OPS_INIT}}", dataOpsInit}
	});
}
const char *DATA_OPS_INIT_Template = R"~~~(
    Dac_Ops {{NAME}}_ops;
    {{OP_PUSH_BACK2OPS}}
)~~~";

std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack2Ops){
    return templateString(DATA_OPS_INIT_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_PUSH_BACK2OPS}}",    opPushBack2Ops},
	});
}
const char *OP_PUSH_BACK2OPS_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{NAME}}_ops.push_back({{OP_NAME}});)~~~";

std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId){
    return templateString(OP_PUSH_BACK2OPS_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{NAME}}",       name},
		{"{{DIM_ID}}",     dimId}
	});
}
const char *KERNEL_EXECUTE_Template = R"~~~(
    sycl::device device = q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);
    q.submit([&](handler &h) {
    {{ACCESSOR_LIST}}
    {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			{{INDEX_INIT}}
			{{GETPOS}}
            {{ACCESSOR_POINTER_LIST}}
			{{CALC_EMBED}}
        });
    }).wait();

)~~~";

const char *KERNEL_EXECUTE_Template1 = R"~~~(
    q.submit([&](handler &h) {
    {{ACCESSOR_LIST}}
    {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			{{INDEX_INIT}}
			{{GETPOS}}
            {{ACCESSOR_POINTER_LIST}}
			{{CALC_EMBED}}
        });
    }).wait();
)~~~";


const char *submitCode = R"~~~(
    q.submit([&](handler &h) {
    {{ACCESSOR_LIST}}
    {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
			{{INDEX_INIT}}
			{{GETPOS}}
            {{ACCESSOR_POINTER_LIST}}
			{{CALC_EMBED}}
        });
    }).wait();

)~~~";
const char *KERNEL_EXECUTE_Template2 = R"~~~(
    sycl::device device = q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);

    {{KERNEL}}

)~~~";
const char* ACCESSOR_LIST_K_read =  R"~~~(
        accessor<{{TYPE}}, 1, access::mode::read> acc_{{NAME}}(r_{{NAME}}, h);
        r_{{NAME}}.set_final_data(nullptr);
        )~~~";
const char* ACCESSOR_LIST_K_write =  R"~~~(
        accessor<{{TYPE}}, 1, sycl::access::mode::discard_write> acc_{{NAME}}(*r_{{NAME}}, h);)~~~";
const char* ACCESSOR_LIST_K_read_write =  R"~~~(
        accessor<{{TYPE}}, 1, sycl::access::mode::read_write> acc_{{NAME}}(*r_{{NAME}}, h);)~~~";
std::string CodeGen_AccessorInit0_read(std::string name, std::string type){
	return templateString(ACCESSOR_LIST_K_read,{
		{"{{NAME}}", name},
        {"{{TYPE}}", type}
	});
}
std::string CodeGen_AccessorInit0_write(std::string name, std::string type){
	return templateString(ACCESSOR_LIST_K_write,{
		{"{{NAME}}", name},
        {"{{TYPE}}", type}
	});
}
const char* ACCESSOR_POINTER_LIST_K =  R"~~~(
            auto* d_{{NAME}} = acc_{{NAME}}.get_multi_ptr<access::decorated::no>().get();)~~~";
std::string CodeGen_AccessorInit1(std::string name){
	return templateString(ACCESSOR_POINTER_LIST_K,{
		{"{{NAME}}", name}
	});
}
std::string CodeGen_AccessorInit0_read_write(std::string name, std::string type){
	return templateString(ACCESSOR_LIST_K_read_write,{
		{"{{NAME}}", name},
        {"{{TYPE}}", type}
	});
}
const char* GETPOS_1 =  R"~~~(
			const auto {{NAME}}_{{idx}} = {{splitname}}_ * {{splitname}}.stride;)~~~";

const char* GETPOS_0 =  R"~~~(
			const auto {{NAME}}_{{idx}} = 0;)~~~";
std::string CodeGen_getpos1(std::string name, std::string opsname, std::string idx){
	return templateString(GETPOS_1,{
		{"{{NAME}}", name},
		{"{{splitname}}", opsname},
		{"{{idx}}", idx}
	});
}

std::string CodeGen_getpos0(std::string name, std::string idx){
	return templateString(GETPOS_0,{
		{"{{NAME}}", name},
		{"{{idx}}", idx}
	});
}

std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit, std::string getpos, std::string ACCESSOR_LIST1, std::string ACCESSOR_LIST2, std::string CalcEmbed){
    return templateString(KERNEL_EXECUTE_Template,
	{
		{"{{SPLIT_SIZE}}",    SplitSize},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed},
        {"{{ACCESSOR_INIT}}", AccessorInit},
		{"{{ACCESSOR_LIST}}",  ACCESSOR_LIST1},
        {"{{ACCESSOR_POINTER_LIST}}",   ACCESSOR_LIST2},
		{"{{GETPOS}}",    getpos}
	});
}
std::string CodeGen_KernelExecute2(std::string SplitSize, std::string AccessorInit, std::string IndexInit, std::string getpos, std::string ACCESSOR_LIST1, std::string ACCESSOR_LIST2, std::string CalcEmbed, dacppTranslator::DacppFile* dacppFile){

    std::string forBodyTemplate=extractAndReplaceSubmitCodeGeneric(dacppFile->forStatement,dacppFile->Context,dacppFile);	std::string KERNEL_EXECUTE=templateString(forBodyTemplate,{{"{{SUBMIT}}",submitCode}});	std::string KERNEL_TEMPLATE=templateString(KERNEL_EXECUTE_Template2,{{"{{KERNEL}}",KERNEL_EXECUTE}});	return templateString(KERNEL_TEMPLATE,
	{
		{"{{SPLIT_SIZE}}",    SplitSize},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed},
        {"{{ACCESSOR_INIT}}", AccessorInit},
		{"{{ACCESSOR_LIST}}",  ACCESSOR_LIST1},
        {"{{ACCESSOR_POINTER_LIST}}",   ACCESSOR_LIST2},
        {"{{GETPOS}}",    getpos}
	});
}
const char *ACCESSOR_INIT_Template = R"~~~(
        auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read>(h);)~~~";
std::string CodeGen_AccessorInit(std::string name) {
	return templateString(ACCESSOR_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}
const char *INDEX_INIT_Template = R"~~~(
            const auto {{NAME}}={{EXPRESSION}};)~~~";
std::string CodeGen_IndexInit2(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets){
    std::set<std::string> sets_map;    std::vector<std::string> sets_order;    std::vector<std::string> sets_split;    for (int i = 0; i < sets.size(); ++i)
    {
		std::string ops_i_name = ops[i].name;
        if (sets_map.find(sets[i]) == sets_map.end())        {
            sets_map.insert(sets[i]);            sets_order.push_back(sets[i]);            sets_split.push_back(ops_i_name + ".split_size");        }
    }

    int sets_size = sets_map.size();    std::unordered_map<std::string,std::string> sets_sub_expression;
    for(int i = 0;i < sets_size; i++)    {
		std::string sub_expression = "item_id";
		for(int j = i + 1;j < sets_size;j ++){
			sub_expression = sub_expression + "/" + sets_split[j];
		}
        sets_sub_expression[sets_order[i]] = sub_expression;	}
    int len = ops.size;
	std::vector<std::string> index_expression_vector;
    for(int i = 0;i < len;i ++)
    {
        std::string index_expression = "(";
        index_expression = index_expression + sets_sub_expression[sets[i]];		index_expression = index_expression + "+" + "(" + offsets[i] + ")" + ")";
		index_expression = index_expression + "%" + ops[i].name + ".split_size";
		index_expression_vector.push_back(index_expression);
    }

	std::string expression = "";
	for(int i=0;i<len;i++){
		std::string opsname = ops[i].name;
		std::string index_i_expression = index_expression_vector[i];
		expression = expression + templateString(INDEX_INIT_Template,
		{
			{"{{NAME}}", opsname + "_"},			{"{{EXPRESSION}}", index_i_expression}
		});
	}
	return expression;
}
const char *CALC_EMBED_Template = R"~~~(
            {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

std::string CodeGen_CalcEmbed2(std::string Name, std::vector<std::string> splits, std::vector<int> splitNum,std::vector<std::string> accessor_names) {

    std::string DacCalcArgs = "(";        for (int z = 0; z < accessor_names.size(); z++) {
        DacCalcArgs += "d_" + accessor_names[z] +",";
    }
        for (int z = 0; z < accessor_names.size(); z++) {
            for(int zz = 0; zz < splitNum[z]; zz++){
                DacCalcArgs += accessor_names[z] + "_" + splits[zz] +",";
            }
    }
        for (int z = 0; z < accessor_names.size(); z++) {
            for(int zz = 0; zz < splitNum[z]; zz++){
                DacCalcArgs += "info_" + accessor_names[z] + "_Shape[" + splits[zz] + "],";
            }
    }
    for (int z = 0; z < accessor_names.size(); z++) {
        DacCalcArgs += "info_partition_" + accessor_names[z] + "_accessor";
        if (z != accessor_names.size() - 1) DacCalcArgs += ",";
    }

    DacCalcArgs += ");";

    return templateString(
        CALC_EMBED_Template,
        {
            {"{{DAC_CALC_NAME}}", Name},
            {"{{DAC_CALC_ARGS}}", DacCalcArgs}
        }
    );
}
const char *REDUCTION_Template_Span = R"~~~(
    if({{SPLIT_SIZE}} > 1)
    {
        for(int i=0;i<{{SPAN_SIZE}};i++) {
            q.submit([&](handler &h) {
                accessor d_{{NAME}}{r_{{NAME}}, h};
    	        h.parallel_for(
                range<1>({{SPLIT_SIZE}}),
                reduction(b_reduction_{{NAME}}[i], h,
                {{REDUCTION_RULE}},property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_{{NAME}}[(i/{{SPLIT_LENGTH}})*{{SPLIT_LENGTH}}*{{SPLIT_SIZE}}+i%{{SPLIT_LENGTH}}+idx*{{SPLIT_LENGTH}}]);
     	        });
         }).wait();
        }
        {
            host_accessor b_acc{r_{{NAME}}};
            for(int i = 0; i < {{SPAN_SIZE}}; i++){
                host_accessor temp_accessor{b_reduction_{{NAME}}[i]};
                b_acc[i] = temp_accessor[0];
            }
        }
    }

)~~~";

std::string CodeGen_Reduction_Span(std::string SpanSize,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule) {
    return templateString(REDUCTION_Template_Span,
	{
        {"{{SPAN_SIZE}}",        SpanSize},
		{"{{SPLIT_SIZE}}",       SplitSize},
		{"{{SPLIT_LENGTH}}",     SplitLength},
		{"{{TYPE}}",             Type},
		{"{{NAME}}",             Name},
		{"{{REDUCTION_RULE}}",   ReductionRule}
	});
}

const char *RESULT_B2H_MOV_Template = R"~~~(
    r_{{NAME}}.reset();
    {{NAME}}.array2Tensor(h_{{NAME}});
)~~~";

std::string CodeGen_Result_B2H_Mov(std::string NAME,std::string SIZE)
{
    return templateString(RESULT_B2H_MOV_Template,
	{
		{"{{NAME}}",             NAME},
		{"{{SIZE}}",             SIZE}
	});
}

}
