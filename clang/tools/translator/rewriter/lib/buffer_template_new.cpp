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
#include "usm_template.h"
#include "Calc.h"
#include "ASTParse.h"

namespace BUFFER_TEMPLATE {
// -------------------- max_generate --------------------
bool max_generate(std::string& reductionText, std::string original,
                  dacppTranslator::DacppFile dacFile) {

    std::string accText = "";
    // 1. 提取 expr 子串
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

    // 2. 提取剩余三个参数
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

    // 3. 识别 expr 中使用的 forStatementVars
    std::vector<std::string> name;
    for (const auto& var_pair : dacFile.getForStatementVars()) {
        std::cout<<"Checking variable: "<<var_pair.first<<std::endl;
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

    // 4. 生成 q.submit 代码
    reductionText += " dacpp_q.submit([&](sycl::handler &h) {\n";

    // 4.1 accessors
// 4.1 accessors

for (const auto &var : name) {
    int i=0;
    dacppTranslator::Expression* expr = dacFile.getExpression(i);
    dacppTranslator::Shell* shell = expr->getShell();
    dacppTranslator::Calc* calc = expr->getCalc();
    bool flag = false;  //flag 用于标记是否找到写权限
    bool found = false;
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




    // 4.2 d_指针
    for (const auto &var : name) {
        reductionText += "  //      auto d_" + var + " = acc_" + var + ".get_pointer();\n";
    }


    // 4.3 parallel_for
    std::string max_buf = "r_max_error"; // 假设已定义
    reductionText += "        h.parallel_for(\n";
    reductionText += "            sycl::range<1>(" + N_name + "),\n";
    reductionText += "            sycl::reduction(" + max_buf + ", h, sycl::maximum<float>(), sycl::property::reduction::initialize_to_identity()),\n";
    reductionText += "            [=](sycl::id<1> idx, auto &reducer) {\n";
    std::string val_expr = expr;
for (const auto &var : name) {
    size_t pos = 0;
    while ((pos = val_expr.find(var, pos)) != std::string::npos) {

        // 必须是独立标识符
        bool left_ok = (pos == 0) || (!std::isalnum(val_expr[pos-1]) && val_expr[pos-1] != '_');
        bool right_ok = (pos + var.length() < val_expr.size()) &&
                        val_expr[pos + var.length()] == '[';

        if (!left_ok || !right_ok) {
            pos += var.length();
            continue;
        }

        // 找匹配的 ]
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

        // 替换整个 x[...]
        val_expr.replace(pos, rbracket - pos + 1,
                         "acc_" + var + "[idx]");
        pos += var.length() + 6; // acc_ + [idx]
    }
}


    reductionText += "                float val = " + val_expr + ";\n";
    reductionText += "                reducer.combine(val);\n";
    reductionText += "            }\n";
    reductionText += "        );\n";
    reductionText += "    }).wait();\n";
    // std::cout<<"Generated reductionText:\n"<<reductionText<<std::endl;
    return true;
}

// -------------------- replaceReductionMax --------------------
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

//BUFFER_ACCESSOR_LIST存储了所有访问器的声明
//ACCESSOR_POINTER_LIST存储了所有访问器指针的声明 这些都是在内核中用到的
std::string BUFFER_ACCESSOR_LIST = "";
std::string ACCESSOR_POINTER_LIST = "";
//下面这个由b_name修改为r_name因为现在重组后的数据放到了r_name中
/*
const char *BUFFER_ACCESSOR_Template = R"~~~(
        accessor acc_{{NAME}}{b_{{NAME}}, h};)~~~";
*/
const char *BUFFER_ACCESSOR_Template = R"~~~(
        accessor acc_{{NAME}}{r_{{NAME}}, h};)~~~";
const char *ACCESSOR_POINTER_Template = R"~~~(
            auto* d_{{NAME}} = acc_{{NAME}}.get_multi_ptr<access::decorated::no>().get();)~~~";




//生成函数的总模板 基本包含了大致结构  buffer没有内存释放
const char *DAC2SYCL_Template_2 = R"~~~(
// 生成函数调用
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
    using namespace sycl;
    // 设备选择
    auto selector = default_selector_v;
    sycl::queue dacpp_q(selector);
    //声明参数生成工具
    ParameterGeneration para_gene_tool;
    // 算子初始化
    {{OP_INIT}}
    //参数生成
	{{ParameterGenerate}}
    // 设备内存分配
    {{DEVICE_MEM_ALLOC}}
    // 数据关联计算
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

// 判断 FS 的 body 中是否包含另一个 for 循环（用于判定是否做 2D 并行化）
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

// 判断一个 for 是否“包含” <-> 表达式对应的 BinaryOperator（防御性用）
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
// ================================================
// 将 d_A[i][j] 形式转换为 d_A[i * stride + j]
// ================================================
static std::string linearize2D(
    const std::string& body,
    const std::vector<std::string>& dvars,
    const std::vector<std::string>& strideExpr,
    bool& star
){
std::string out = body;
int i = 0;

for (const auto& v : dvars) {

    // ---------- 1. 二维 -> 一维 ----------
    {
        std::regex pat("\\b" + v + "\\s*\\[([^\\]]+)\\]\\s*\\[([^\\]]+)\\]");
        std::string rep = v + "[($1) * (" + strideExpr[i] + ") + ($2)]";
        out = std::regex_replace(out, pat, rep);
    }

    // ---------- 2. device pointer 标量解引用 ----------
    // 匹配“裸 v”，但排除：
    //   v[   *v   &v   v.   v->
    {
        std::regex scalarPat(
            "\\b" + v + "\\b"
            "(?!\\s*\\[)"     // not v[
            "(?!\\s*->)"     // not v->
            "(?!\\s*\\.)"    // not v.
        );

        // 为避免重复加 *，先排除 *v
        std::regex alreadyDeref("\\*\\s*" + v);

        if (!std::regex_search(out, alreadyDeref)) {
            out = std::regex_replace(out, scalarPat, "*" + v);
        }
    }

    i++;
}

return out;

}

// 二维自动并行化：for(i){ for(j){ body } } → 2D parallel_for
std::string parallelizeNestedFor(
    const clang::ForStmt* outerFS,
    const clang::ASTContext* Context,
    dacppTranslator::DacppFile* dacFile   // ✔ 你刚加的参数
){
    using namespace clang;
    const LangOptions& LO = Context->getLangOpts();
    const SourceManager& SM = Context->getSourceManager();

    // ====== 解析 i 循环 ======
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

    // ====== 获取内层 j 循环 ======
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
        return ""; // 不是二维循环，返回空让外层走 1D

    // ====== 解析 j 循环 ======
    std::string jVar="j", jL="0", jR="0";
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

    // ====== 获取 body ======
    std::string bodyText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(innerFS->getBody()->getSourceRange()),
        SM, LO).str();


    // ================================================================
    // ★ 第一步：读取所有外部变量（你之前 collectVarsFromForStatement 已经收集）
    // ================================================================
    auto vars = dacFile->getForStatementVars();


    // ================================================================
    // ★ 第二步：为所有变量生成 accessor + d_ptr，并准备替换 body
    // ================================================================
    std::string accessorDecl;
    std::string ptrDecl;
    std::string replacedBody = bodyText;


    for (auto &p : vars) {
    const std::string& name = p.first;
    const std::string& type = p.second;

    // 检查该变量是否真的出现在 body 中
    std::regex wordExpr("\\b" + name + "\\b");
    if (!std::regex_search(replacedBody, wordExpr)) {
        continue;   // 不在 body 中 → 完全忽略
    }

    // 判断是否为 const 类型
    bool isConst = (type.find("const") != std::string::npos);

    // =========================================================
    // 1) const 类型：不生成 accessor & d_ptr，不替换成 d_xxx
    // =========================================================
    if (isConst) {
        // const 保持原名，不修改 replacedBody
        replacedBody = std::regex_replace(replacedBody, wordExpr, name);
        continue;
    }

    // =========================================================
    // 2) 非 const 类型：生成 accessor & d_ptr 并替换为 d_xxx
    // =========================================================

    // accessor（注意 r_name：表示 buffer 名）
    // accessorDecl +=
    //     "    auto acc_" + name +
    //     " = r_" + name +
    //     ".get_access<sycl::access::mode::read_write>(h);\n";
    bool flag = false;//flag 用于标记是否找到写权限
    int i=0;
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
    // 不write的变量：把 . 换成 ->
    accessorDecl +=
        "    auto acc_" + name +
        " = r_" + name +
        ".get_access<sycl::access::mode::read>(h);\n";
} else if(found&&flag){
    // === 非 const 变量：把 . 换成 -> ===
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
    // 设备指针（注意 template 关键字）
    ptrDecl +=
        "      auto* d_" + name +
        " = acc_" + name +
        ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";

    // 替换 body 中出现的变量为 d_xxx
    replacedBody = std::regex_replace(
        replacedBody,
        wordExpr,
        "d_" + name
    );
}

    // ================================================================
    // ★ 第 X 步：二维下标线性化
    // ================================================================

    std::vector<std::string> stride ;

    // 收集所有 d_XXX
    std::vector<std::string> dvars;
    for (auto &p : vars) {

        const std::string& name = p.first;
        const std::string& type = p.second;
        bool isConst = (type.find("const") != std::string::npos);
        if (isConst) continue;
        stride.push_back("info_" + name+"_Shape[0]");
        // 之前生成了 d_name，因此这里加入 d_name
        std::string dname = "d_" + name;

        // 只有 body 中出现才需要加入
        std::regex w("\\b" + dname + "\\b");
        if (std::regex_search(replacedBody, w))
            dvars.push_back(dname);
    }
    bool star = false;
    // 进行二维访问替换
    replacedBody = linearize2D(replacedBody, dvars, stride , star);


    // ================================================================
    // ★ 第三步：生成完整的 2D kernel（和你原来的几乎相同，只是加了 accessor）
    // ================================================================
    std::string code;
    code += "{\n";
    code += "    int __iL = (" + iL + ");\n";
    code += "    int __iR = (" + iR + ");\n";
    code += "    int __iN = __iR - __iL +1;\n";
    code += "    int __jL = (" + jL + ");\n";
    code += "    int __jR = (" + jR + ");\n";
    code += "    int __jN = __jR - __jL + 1;\n";

    code += "    dacpp_q.submit([&](sycl::handler& h){\n";

    // ★ 插入 accessor
    code += accessorDecl;

    code += "    h.parallel_for(sycl::range<2>(__iN, __jN), [=](sycl::id<2> idx){\n";

    // ★ 插入 d_ptr
    code += ptrDecl;

    code += "      int " + iVar + " = __iL + idx[0];\n";
    code += "      int " + jVar + " = __jL + idx[1];\n";

    // ★ 使用替换后的 body
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
        std::string nested = parallelizeNestedFor(FS, Context, dacFile); // ★ 修改
        if (!nested.empty())
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

    // ★★★ 最关键：从 dacFile 取 forStatementVars
    auto vars = dacFile->getForStatementVars();

    std::string accessorDecl;
    std::string ptrDecl;
    std::string replacedBody = bodyText;
    // ====== 根据 body 实际使用的变量，生成 accessor 和设备指针 ======

    for (auto &p : vars) {
    const std::string& name = p.first;
    const std::string& type = p.second;

    // 检查该变量是否真的出现在 body 中
    std::regex wordExpr("\\b" + name + "\\b");
    if (!std::regex_search(replacedBody, wordExpr)) {
        continue;   // 不在 body 中 → 完全忽略
    }

    bool flag = true;
    bool found = false;
    // 查询 shellVars 中的类型是否包含 write,目前默认只有一个表达式
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
    // 不进行写的情况下，翻译成.
    accessorDecl +=
        "    auto acc_" + name +
        " = r_" + name +
        ".get_access<sycl::access::mode::read>(h);\n";
    } else if(found&&flag){
    // 进行写的情况下翻译成->
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


    // 设备指针（注意 template 关键字）
    ptrDecl +=
        "      auto* d_" + name +
        " = acc_" + name +
        ".template get_multi_ptr<sycl::access::decorated::no>().get();\n";

    // 替换 body 中出现的变量为 d_xxx
    replacedBody = std::regex_replace(
        replacedBody,
        wordExpr,
        "d_" + name
    );
    }


// ================================================================
// ★ 第 X 步：二维下标线性化（用于一维 parallel_for 场景）
// ================================================================

    // 尝试推断 stride，例如从 for(j=0;j<NY;j++) 得到 stride = __N 或者某 const 变量
    // 此处采用最简单稳定策略：stride = __N
    std::vector<std::string> stride ;

    // 找出所有 d_xxx
    std::vector<std::string> dvars;
    for (auto &p : vars) {
        const std::string& name = p.first;
        const std::string& type = p.second;
        bool isConst = (type.find("const") != std::string::npos);
        if (isConst) continue;
        stride.push_back("info_" + name+"_Shape[0]");
        std::string dname = "d_" + name;
        std::regex w("\\b" + dname + "\\b");
        if (std::regex_search(replacedBody, w))
            dvars.push_back(dname);
    }
    bool star = false;
    replacedBody = linearize2D(replacedBody, dvars, stride,star);



    std::string code;
    code += "{\n";
    code += "  int __L = (" + L + ");\n";
    code += "  int __R = (" + R + ");\n";
    code += "  int __N = __R - __L +1;\n";

    code += "  dacpp_q.submit([&](sycl::handler& h){\n";
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

// 辅助：在 outerFor 的 body 里 DFS，收集需要并行化的 for
// 规则：
//  - 跳过 outerFor 自己；
//  - 跳过内部仍然包含 <-> 的 for；
//  - 对于嵌套结构 for(i){ for(j){...} }：只把最外层的 i 加入 result，j 不再单独加入。
static void dfsCollectFors(
    const clang::Stmt* S,
    const clang::ForStmt* outerFor,
    const clang::BinaryOperator* dacExpr,
    clang::ASTContext* Context,
    const clang::ForStmt* currentTop,   // 当前所在的“顶层 for(i)”，如果为 nullptr 表示还没进入任何 for
    std::vector<const clang::ForStmt*>& result
) {
    using namespace clang;
    if (!S) return;

    if (const auto* FS = llvm::dyn_cast<ForStmt>(S)) {
        // 跳过最外层含 <-> 的那个 for
        if (FS == outerFor) {
            // 继续遍历它的 body
            dfsCollectFors(FS->getBody(), outerFor, dacExpr, Context, nullptr, result);
            return;
        }

        // 如果这个 for 自己包含 <->，也跳过（防御性）
        if (dacExpr && forContainsDacExpr(FS, dacExpr, Context)) {
            // 但它的子树里可能还有别的 for，可以继续往下找
            dfsCollectFors(FS->getBody(), outerFor, dacExpr, Context, currentTop, result);
            return;
        }

        if (currentTop == nullptr) {
            // 这是 outerFor 里面的一个新的顶层 for(i)
            result.push_back(FS);
            // 以它为当前“顶层 for”，继续往里找，但不再把内层 j 单独加入
            dfsCollectFors(FS->getBody(), outerFor, dacExpr, Context, FS, result);
        } else {
            // 这是嵌套在某个顶层 for(i) 里的 j 循环，不单独加入 result
            // 但它的 body 里可能还有更深的结构，继续往下
            dfsCollectFors(FS->getBody(), outerFor, dacExpr, Context, currentTop, result);
        }
        return;
    }

    // 如果是 CompoundStmt，就遍历其中每个子语句
    if (const auto* CS = llvm::dyn_cast<CompoundStmt>(S)) {
        for (const Stmt* child : CS->body()) {
            dfsCollectFors(child, outerFor, dacExpr, Context, currentTop, result);
        }
        return;
    }

    // 普通语句：遍历它的子节点
    for (const Stmt* child : S->children()) {
        dfsCollectFors(child, outerFor, dacExpr, Context, currentTop, result);
    }
}

// 收集 outerFor 体内的 “需要并行化的 for”
// 关键点：遇到 for(i){ for(j){...} } 时，只收集外层 i，不单独收集 j。
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


// 用 parallelizeSingleFor 替换掉 inner for；现在只会在嵌套的最外层做 2D
// 重写兄弟 for 为 parallel_for
std::string rewriteSiblingForsToParallel(
    const std::string& originalForText,
    const clang::ForStmt* outerFor,
    const clang::BinaryOperator* dacExpr,
    clang::ASTContext* Context,
    dacppTranslator::DacppFile* dacFile   // ★ 新增
){
    std::string newText = originalForText;

    // 收集 outerFor 内部的“可并行化的外层 for”
    auto targets = collectInnerForsExceptDacpp(outerFor, dacExpr, Context);

    // 为避免干扰文本位置，从靠后位置开始替换
    std::sort(
        targets.begin(), targets.end(),
        [&](auto* A, auto* B){
            const auto &SM = Context->getSourceManager();
            return SM.isBeforeInTranslationUnit(B->getBeginLoc(), A->getBeginLoc());
        }
    );

    for (auto* FS : targets) {

        // 旧 for 的源码
        std::string oldFS = clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(FS->getSourceRange()),
            Context->getSourceManager(),
            Context->getLangOpts()
        ).str();

        // ★★★ 使用带 dacFile 的 parallelizeSingleFor
        std::string newFS = parallelizeSingleFor(FS, Context, dacFile);

        // 在文本中替换
        size_t pos = newText.find(oldFS);
        if (pos != std::string::npos) {
            newText.replace(pos, oldFS.size(), newFS);
        }
    }

    return newText;
}




//数据信息初始化模板 这部分应该在上面的ParameterGeneration para_gene_tool;后面
//{{OP_INIT}}的前面 种种原因当时没有把这个写进去 Rewriter调用应该调用
const char *DATA_INFO_INIT_Template = R"~~~(
    // 数据信息初始化
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

    // 假设 dacppFile 提供获取 forStatementVars 的接口
    // 每个元素是 pair<type, name>
    std::vector<std::pair<std::string, std::string>> forStatementVars = dacppFile->getForStatementVars();

for (const auto& var : forStatementVars) {
    const std::string &name = var.first;
    const std::string &type = var.second;

    // 1. 检查是否已在 shellVars 中
    bool inShellVars = false;
    for (const auto &shellVar : dacppFile->shellVars) {
        if (shellVar.first == name) {
            inShellVars = true;
            break;
        }
    }

    // 2. 如果在 shellVars 中，跳过
    if (inShellVars) {
        continue;
    }

    // 3. 正常生成 buffer
    std::string bufferName = "r_" + name;
    code += "    sycl::buffer<" + type + ", 1> "
          + bufferName
          + "(&" + name + ", sycl::range<1>(1));\n";
}


    return code;
}
//2025.12.18新增，把if转换成single task交给gpu做
// std::string translateAllIfsToSingleTask(dacppTranslator::DacppFile* dacppFile, const std::string& codeStr) {
//     if (!dacppFile || !dacppFile->forStatement) return codeStr;
//     const ForStmt* forStmt = dacppFile->forStatement;
//     ASTContext* context = dacppFile->Context;
//     if (!context) return codeStr;

//     std::string newCode = codeStr;

//     for (const Stmt* s : forStmt->children()) {
//         if (!s) continue;

//         const IfStmt* targetIf = dyn_cast<IfStmt>(s);
//         if (!targetIf) continue;

//         Expr* cond = targetIf->getCond();
//         if (!cond) continue;
//         SourceRange condRange = cond->getSourceRange();
//         if (condRange.isInvalid()) continue;

//         std::string condText = Lexer::getSourceText(CharSourceRange::getTokenRange(condRange),
//                                                     context->getSourceManager(),
//                                                     context->getLangOpts()).str();
//         if (condText.empty()) continue;

//         // 获取 then 块赋值变量
//         std::vector<std::string> assignVars;
//         const Stmt* thenStmt = targetIf->getThen();
//         if (!thenStmt) continue;

//         if (const CompoundStmt* cs = dyn_cast<CompoundStmt>(thenStmt)) {
//             for (const Stmt* inner : cs->body()) {
//                 if (!inner) continue;
//                 if (const BinaryOperator* bo = dyn_cast<BinaryOperator>(inner)) {
//                     if (bo->isAssignmentOp()) {
//                         if (const DeclRefExpr* lhs = dyn_cast<DeclRefExpr>(bo->getLHS())) {
//                             assignVars.push_back(lhs->getNameInfo().getAsString());
//                         }
//                     }
//                 }
//             }
//         }

//         if (assignVars.empty()) continue;

//         // 用 string 拼接生成 GPU single_task 代码
//         std::string replacement;

//         for (auto& varPair : dacppFile->forStatementVars) {
//             std::string name = varPair.first;
//             replacement += "auto acc_" + name + " = r_" + name +
//                            ".get_access<cl::sycl::access::mode::read_write>(h);\n";
//             replacement += "auto d_" + name + " = acc_" + name + ".get_pointer();\n";
//         }

//         replacement += "h.single_task([=]() {\n";
//         replacement += "    if (" + condText + ") {\n";
//         for (auto& var : assignVars) {
//             replacement += "        *d_" + var + " = true;\n";
//         }
//         replacement += "    }\n";
//         replacement += "});\n";

//         // 替换原 if 语句
//         SourceRange ifRange = targetIf->getSourceRange();
//         std::string ifText = Lexer::getSourceText(CharSourceRange::getTokenRange(ifRange),
//                                                   context->getSourceManager(),
//                                                   context->getLangOpts()).str();
//         size_t pos = newCode.find(ifText);
//         if (pos != std::string::npos) {
//             newCode.replace(pos, ifText.size(), replacement);
//         }
//     }

//     return newCode;
// }

//2025.12.2新增，提取并替换 for 循环中的 DACPP 表达式为 {{SUBMIT}}
// 2025.12.2 新版：可并行化 + 保留原逻辑 + 保证替换顺序正确
// 2025.12.2 新版：可并行化 + 保留原逻辑 + 保证替换顺序正确
std::string extractAndReplaceSubmitCodeGeneric(
    const clang::ForStmt* forStatement,
    clang::ASTContext* Context,
    dacppTranslator::DacppFile* dacFile   // ★ 已修改
) {
    using namespace clang;
    if (!forStatement || !Context || !dacFile) return "";

    SourceManager &SM           = Context->getSourceManager();
    const LangOptions &LangOpts = Context->getLangOpts();

    // ======= Part 1：找到包含 <-> 的 BinaryOperator =======
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
        llvm::errs() << "[extractAndReplaceSubmitCodeGeneric] 未找到包含 <-> 的 DACPP 表达式\n";
        return "";
    }

    // ======= Part 2：提取 for 语句源代码 =======
    std::string forText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(forStatement->getSourceRange()),
        SM, LangOpts
    ).str();
    forText = generateBufferCode(dacFile) + forText;
    
    // ======= Part 2.5：重写规约代码 =======
    std::string reductionText="";
    // max_generate(reductionText,forText,*dacFile);
    std::cout<<"reductionText:"<<reductionText<<std::endl;
    replaceReductionMax(forText, reductionText,*dacFile);
    std::cout<<"finished "<<std::endl;
    // translateAllIfsToSingleTask(dacFile,forText);
    // ======= Part 3：重写兄弟 for → parallel_for =======
    forText = rewriteSiblingForsToParallel(
        forText,
        forStatement,
        visitor.targets[0],
        Context,
        dacFile       // ★ 必须传入
    );

    // ======= Part 4：将 <-> 替换为 {{SUBMIT}} =======
    for (auto *op : visitor.targets) {
        std::string opText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(op->getSourceRange()),
            SM, LangOpts
        ).str();

        size_t pos = forText.find(opText);
        if (pos != std::string::npos)
            forText.replace(pos, opText.length(), "{{SUBMIT}}");
        else
            llvm::errs() << "[extractAndReplaceSubmitCodeGeneric] 警告：在 forText 中未找到某个 <-> 表达式文本\n";
    }

    // ======= Part 5：替换变量：普通变量 → closure.varName =======
    // auto externalVars = dacFile->getForStatementVars();

    // for (auto &p : externalVars) {
    //     const std::string &varName  = p.first;
    //     const std::string &typeName = p.second;

    //     if (typeName.find("const") != std::string::npos)
    //         continue;

    //     bool isTensor = (typeName.find("Tensor") != std::string::npos);
    //     bool isMatrix = (typeName.find("Matrix") != std::string::npos);

    //     if (isTensor || isMatrix)
    //         continue;

    //     std::regex wordExpr("\\b" + varName + "\\b");
    //     forText = std::regex_replace(forText, wordExpr, "closure." + varName);
    // }

    return forText;
}



//算子初始化模板 {{OP_INIT}}
//分区算子初始化
const char *OP_REGULAR_SLICE_INIT_Template2 = R"~~~(
    // 规则分区算子初始化
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
		{"{{DIM_ID}}",     dim_id}, //需要通过dimId来计算算子的划分数了
		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME}
	});
}

//降维算子初始化
const char *OP_INDEX_INIT_Template2 = R"~~~(
    // 降维算子初始化
    Index {{OP_NAME}} = Index("{{OP_NAME}}");
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.SetSplitSize(para_gene_tool.init_operetor_splitnumber({{OP_NAME}},{{DATA_INFO_NAME}}));
)~~~";

std::string CodeGen_IndexInit2(std::string opName,std::string dim_id,std::string DATA_INFO_NAME){
    return templateString(OP_INDEX_INIT_Template2,
	{
		{"{{OP_NAME}}",    opName},
		{"{{DIM_ID}}", dim_id}, //需要通过dimId来计算算子的划分数
		{"{{DATA_INFO_NAME}}", DATA_INFO_NAME}
	});
}




//参数生成的总模板  {{ParameterGenerate}}
// const char *PARA_GENE_Template = R"~~~(
//     // 参数生成 提前计算后面需要用到的参数	
// 	{{InitOPS}}
// 	{{InitDeviceMemorySize}}
// 	{{InitSplitLength}}
// 	{{InitSpilitLengthMatrix}}
// 	{{ItemNumber}}
// 	{{InitReductionSplitSize}}
// 	{{InitReductionSplitLength}}
// )~~~";
//由于规约功能暂时没用，因此去掉{{InitReductionSplitSize}}和{{InitReductionSplitLength}}：
const char *PARA_GENE_Template = R"~~~(
    // 参数生成 提前计算后面需要用到的参数	
	{{InitOPS}}
	{{InitDeviceMemorySize}}
	{{InitSplitLength}}
	{{ItemNumber}}
)~~~";

// std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string InitSpilitLengthMatrix,std::string ItemNumber,std::string InitReductionSplitSize,std::string InitReductionSplitLength){
//     return templateString(PARA_GENE_Template,
// 	{
// 		{"{{InitOPS}}", InitOPS},
// 		{"{{InitDeviceMemorySize}}", InitDeviceMemorySize},//设备内存的分配大小计算
// 		{"{{InitSplitLength}}",InitSplitLength},
// 		{"{{InitSpilitLengthMatrix}}",InitSpilitLengthMatrix},
// 		{"{{ItemNumber}}",ItemNumber},
// 		{"{{InitReductionSplitSize}}",InitReductionSplitSize},
// 		{"{{InitReductionSplitLength}}",InitReductionSplitLength}
// 	});
// }
//由于规约功能暂时没用，因此去掉{{InitReductionSplitSize}}和{{InitReductionSplitLength}}：
// std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string InitSpilitLengthMatrix,std::string ItemNumber){
//     return templateString(PARA_GENE_Template,
// 	{
// 		{"{{InitOPS}}", InitOPS},
// 		{"{{InitDeviceMemorySize}}", InitDeviceMemorySize},//设备内存的分配大小计算
// 		{"{{InitSplitLength}}",InitSplitLength},
// 		{"{{InitSpilitLengthMatrix}}",InitSpilitLengthMatrix},
// 		{"{{ItemNumber}}",ItemNumber},
// 	});
// }

std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string ItemNumber){
    return templateString(PARA_GENE_Template,
	{
		{"{{InitOPS}}", InitOPS},
		{"{{InitDeviceMemorySize}}", InitDeviceMemorySize},//设备内存的分配大小计算
		{"{{InitSplitLength}}",InitSplitLength},
		{"{{ItemNumber}}",ItemNumber},
	});
}

//{{InitOPS}}
// 算子组初始化 
const char *OPS_INIT_Template = R"~~~(
    // 算子组初始化
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

//将算子添加到算子组的模板 数据重组时也有添加算子到算子组的模板 每次添加都将要重新设置作用的维度
//{{ADD_OP2OPS}}
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

//{{InitDeviceMemorySize}}
//生成设备内存分配大小的模板 对应mat[分区][分区] mat[分区][降维] mat[分区][] mat[降维][]
const char *DEVICE_MEM_SIZE_Generate_Template1 = R"~~~(
    //生成设备内存分配大小
    int {{NAME}} = para_gene_tool.init_device_memory_size({{DATA_INFO_NAME}},{{DACOPS_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME,std::string DACOPS_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template1,
	{
        {"{{NAME}}",        NAME}, //设备内存的名字 
		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME}, //tensor的名字
		{"{{DACOPS_NAME}}",        DACOPS_NAME} //算子组的名字
	});
}

//生成设备内存分配大小的模板 对应mat[][]
const char *DEVICE_MEM_SIZE_Generate_Template2 = R"~~~(
    //生成设备内存分配大小
    int {{NAME}} = para_gene_tool.init_device_memory_size({{DATA_INFO_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template2,
	{
        {"{{NAME}}",        NAME}, //设备内存的名字
		{"{{DATA_INFO_NAME}}",     DATA_INFO_NAME} //tensor的名字
	});
}

//生成设备内存分配的大小 对应数据重组需要分配的大小 
const char *DEVICE_MEM_SIZE_Generate_Template3 = R"~~~(
    //生成设备内存分配大小
    int {{NAME}} = para_gene_tool.init_device_memory_size({{IN_DAC_OPS_NAME}},{{OUT_DAC_OPS_NAME}},{{DATA_INFO_NAME}});
)~~~";

std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string IN_DAC_OPS_NAME,std::string OUT_DAC_OPS_NAME,std::string DATA_INFO_NAME){
    return templateString(DEVICE_MEM_SIZE_Generate_Template3,
	{
		{"{{NAME}}",            NAME}, //这个名字要注意 因为要和后面的名字对应
		{"{{IN_DAC_OPS_NAME}}", IN_DAC_OPS_NAME},//输入算子组的名字
		{"{{OUT_DAC_OPS_NAME}}",OUT_DAC_OPS_NAME},//输出算子组的名字
		{"{{DATA_INFO_NAME}}",      DATA_INFO_NAME}//输出数据TENSOR的名字
	});
}

//{{InitSplitLength}}
//计算算子组里面算子的划分数
const char *INIT_SPLIT_LENGTH_Template = R"~~~(
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length({{OPS_NAME}},{{SIZE}});
)~~~";

std::string CodeGen_Init_Split_Length(std::string OPS_NAME,std::string SIZE){
    return templateString(INIT_SPLIT_LENGTH_Template,
	{
		{"{{OPS_NAME}}",       OPS_NAME},
		{"{{SIZE}}",           SIZE}//这个是重组之后的数据的大小
	});
}

//{{InitSpilitLengthMatrix}}
//生成算子划分长度的二维矩阵
const char *INIT_SPLIT_LENGTH_MATRIX_Template = R"~~~(
	{{DECLARE_DACOPS_VECTOR}}
	// 生成划分长度的二维矩阵
    int SplitLength[{{ROW}}][{{COL}}] = {0};
    para_gene_tool.init_split_length_martix({{ROW}},{{COL}},&SplitLength[0][0],{{OPS_S_NAME}});
)~~~";

std::string CodeGen_Init_Split_Length_Matrix(std::string DECLARE_DACOPS_VECTOR,std::string ROW,std::string COL,std::string OPS_S_NAME){
    return templateString(INIT_SPLIT_LENGTH_MATRIX_Template,
	{
		{"{{DECLARE_DACOPS_VECTOR}}",       DECLARE_DACOPS_VECTOR},
		{"{{ROW}}",       ROW},//行 也就是算子组组的个数 后端可以提供
		{"{{COL}}",       COL},//列 算子组中最多的算子的个数作为列
		{"{{OPS_S_NAME}}",       OPS_S_NAME}//前面声明的算子组组的名字
	});
}

//声明std::vector<Dac_Ops>
//{{DECLARE_DACOPS_VECTOR}}
const char *DECLARE_DACOPS_VECTOR_Template = R"~~~(
    std::vector<Dac_Ops> {{OPSS_NAME}};
	{{PUSH_BACK_DAC_OPS}}
)~~~";

std::string CodeGen_Declare_DacOps_Vector(std::string OPSS_NAME,std::string PUSH_BACK_DAC_OPS){
    return templateString(DECLARE_DACOPS_VECTOR_Template,
	{
		{"{{OPSS_NAME}}",           OPSS_NAME},//声明的DAC_OPS算子组组的名字
		{"{{PUSH_BACK_DAC_OPS}}",   PUSH_BACK_DAC_OPS}//要添加的算子的语句
	});
}

//将算子组添加到std::vector<Dac_ops>这个算子组的vector里面
//{{PUSH_BACK_DAC_OPS}}
const char *ADD_DACOPS2VECTOR_Template = R"~~~(
    {{OPSS_NAME}}.push_back({{OPS_NAME}});
)~~~";

std::string CodeGen_Add_DacOps2Vector(std::string OPSS_NAME,std::string OPS_NAME){
    return templateString(ADD_DACOPS2VECTOR_Template,
	{
		{"{{OPSS_NAME}}",       OPSS_NAME},//算子组vector的名字 std::vector<Dac_ops>的名字
		{"{{OPS_NAME}}",         OPS_NAME}//要添加的算子组的名字
	});
}

//{{ItemNumber}}
//计算工作项的多少
const char *INIT_WORK_ITEM_NUMBER_Template = R"~~~(
    // 计算工作项的大小
    int {{NAME}} = para_gene_tool.init_work_item_size({{OPS_NAME}});
)~~~";

std::string CodeGen_Init_Work_Item_Number(std::string NAME,std::string OPS_NAME){
    return templateString(INIT_WORK_ITEM_NUMBER_Template,
	{
		{"{{NAME}}",           NAME},
		{"{{OPS_NAME}}",       OPS_NAME}//算子组的名字
	});
}

//{{InitReductionSplitSize}}
//计算归约中split_size的大小，由于规约功能暂时没用，因此注释掉
// const char *INIT_REDUCTION_SPLIT_SIZE_Template = R"~~~(
//     // 计算归约中split_size的大小
//     int {{NAME}} = para_gene_tool.init_reduction_split_size({{OPS_IN}},{{OPS_OUT}});
// )~~~";

// std::string CodeGen_Init_Reduction_Split_Size(std::string NAME,std::string OPS_IN,std::string OPS_OUT){
//     return templateString(INIT_REDUCTION_SPLIT_SIZE_Template,
// 	{
// 		{"{{NAME}}",           NAME},//归约中spilitsize的名字
// 		{"{{OPS_IN}}",       OPS_IN},//输入算子组的名字
// 		{"{{OPS_OUT}}",     OPS_OUT}//输出算子组的名字
// 	});
// }

//{{InitReductionSplitLength}}
//计算归约中split_length的大小，由于规约功能暂时没用，因此注释掉
// const char *INIT_REDUCTION_SPLIT_LENGTH_Template = R"~~~(
//     // 计算归约中split_length的大小
//     int {{NAME}} = para_gene_tool.init_reduction_split_length({{OPS_NAME}});
// )~~~";

// std::string CodeGen_Init_Reduction_Split_Length(std::string NAME,std::string OPS_NAME){
//     return templateString(INIT_REDUCTION_SPLIT_LENGTH_Template,
// 	{
// 		{"{{NAME}}",           NAME},//归约中spilitsize的名字
// 		{"{{OPS_NAME}}",   OPS_NAME} //算子组的名字
// 	});
// }




//{{DEVICE_MEM_ALLOC}}
//设备内存分配 使用buffer模拟设备内存
//相当于说每申请一个设备内存，在BUFFER_ACCESSOR_LIST中添加访问这个buffer的访问器声明
//为什么不会多声明呢？因为在数据重组中的buffer设备声明中没有没有往BUFFER_ACCESSOR_Template添加访问器声明
const char *DEVICE_MEM_ALLOC_Template = R"~~~(
    // Buffer设备内存分配
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

//目前归约还未使用
// const char *DEVICE_MEM_ALLOC_REDUCTION_Template = R"~~~(
//     // 规约Buffer设备内存分配
//     std::vector<sycl::buffer<{{TYPE}}, 1>> b_reduction_{{NAME}}({{SIZE}}, buffer<{{TYPE}}, 1>{1});
//     for(int i = 0; i < {{SIZE}}; i++){
//         host_accessor temp_accessor{b_reduction_{{NAME}}[i]};
//         temp_accessor[0] = 0;
//     })~~~";
//由于规约时不能用偏移量指示buffer,所以需要定义一个vector存结果
// std::string CodeGen_DeviceMemAllocReduction(std::string type,std::string name,std::string size){
//     return templateString(DEVICE_MEM_ALLOC_REDUCTION_Template,{
// 		{"{{TYPE}}", type},
// 		{"{{NAME}}", name},
// 		{"{{SIZE}}", size}
// 	});
// }




//{{DATA_ASSOC_COMP}}
//数据关联计算
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
        {"{{H2D_MEM_MOV}}",       H2DMemMove},//设备端数据重组 先移动数据到设备
		{"{{DATA_RECON}}",        dataRecon},
        {"{{KERNEL_EXECUTE}}",    kernelExecute},
		{"{{REDUCTION}}",         reduction},
        {"{{D2H_MEM_MOV}}",       D2HMemMove}
	});
}

//数据移动
//{{H2D_MEM_MOV}}
// const char *D2B_MOV_BUFFER_Template = R"~~~(
//     // 数据移动
//     {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}.getSize()*sizeof({{TYPE}}));
//     {{NAME}}.tensor2Array(h_{{NAME}});
//     {
//         host_accessor temp_accessor{b_{{NAME}}};
//         for(int i = 0; i < {{SIZE}}; i++){
//             temp_accessor[i] = h_{{NAME}}[i];
//         }
//     }
// )~~~";

// const char *D2B_MOV_BUFFER_Template = R"~~~(
//     // 数据移动
//     {{TYPE}}* h_{{NAME}} = ({{TYPE}}*)malloc({{NAME}}_Size*sizeof({{TYPE}}));
//     {{NAME}}.tensor2Array(h_{{NAME}});
// 	buffer<{{TYPE}}, 1> b_{{NAME}}(h_{{NAME}}, range<1>({{NAME}}_Size));
// )~~~";
const char *D2B_MOV_BUFFER_Template_READ_WRITE = R"~~~(
    // 数据移动
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
    // 数据移动
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

//用于buffer模板数据重组中申请主机内存
/*
    double* h_matNext=(double*)malloc(matNext.getSize()*sizeof(double));
*/
const char *INIT_HOST_MEMORY_Template = R"~~~(
    // 数据移动
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

//执行数据初始化为0
const char *DEVICE_DATA_INIT_Template = R"~~~(
    { //Buffer数据初始化
        host_accessor temp_accessor{b_{{NAME}}};
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

//数据重组
//{{DATA_RECON}}
//数据重组的buffer模板 支持在设备端完成数据重组
//注意逻辑是h_name(主机数据)传输到b_name(设备数据)，然后b_name到r_name(重组之后的数据)
//使用r_name进行计算
// const char *DATA_RECON_BUFFER_Template = R"~~~(
//     // 数据重组
//     DataReconstructor<{{TYPE}}> {{NAME}}_tool;
//     {{DATA_OPS_INIT}}
//     {{NAME}}_tool.init(info_{{NAME}},{{NAME}}_ops,q);
//     buffer<{{TYPE}}> r_{{NAME}}{{{SIZE}}};
//     {{NAME}}_tool.Reconstruct(r_{{NAME}},b_{{NAME}},q);
// 	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
//     sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
// )~~~";

const char *DATA_RECON_BUFFER_Template = R"~~~(
    // 数据重组
    
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

// const char *DATA_RECON_BUFFER_Template1 = R"~~~(
//     // 数据重组
//     DataReconstructor<{{TYPE}}> {{NAME}}_tool;
//     {{DATA_OPS_INIT}}

//     // buffer<{{TYPE}}> r_{{NAME}}{{{SIZE}}};
//     std::vector<{{TYPE}}> init({{SIZE}}, 0);
//     sycl::buffer<{{TYPE}}> r_{{NAME}}(init.data(), sycl::range<1>({{SIZE}}));

// 	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
//     sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
// )~~~";

// const char *DATA_RECON_BUFFER_Template1 = R"~~~(
//     // 数据重组
//     {{DATA_OPS_INIT}}

//     std::vector<{{TYPE}}> {{NAME}}_init({{NAME}}.getSize(), 0);
//     sycl::buffer<{{TYPE}}> r_{{NAME}}({{NAME}}_init.data(), sycl::range<1>({{NAME}}.getSize()));

// 	std::vector<int> info_partition_{{NAME}}=para_gene_tool.init_partition_data_shape(info_{{NAME}},{{NAME}}_ops);
//     sycl::buffer<int> info_partition_{{NAME}}_buffer(info_partition_{{NAME}}.data(), sycl::range<1>(info_partition_{{NAME}}.size()));
// )~~~";

const char *DATA_RECON_BUFFER_Template1 = R"~~~(
    // 数据重组
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

//{{DATA_OPS_INIT}}
//数据算子组初始化 用于计算数据重组时的相关数据
const char *DATA_OPS_INIT_Template = R"~~~(
    // 数据算子组初始化
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

//{{OP_PUSH_BACK2OPS}}
//将需要用到的算子加入到算子组
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

//内核执行
//{{KERNEL_EXECUTE}}
//一维划分的Buffer内核执行模板
//和usm不同的是增加获得访问器以及获得访问器指针的操作
// const char *KERNEL_EXECUTE_Template = R"~~~(
//     sycl::device device = q.get_device();
//     int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
// 	//工作项划分
//     int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
//     sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size)); 
//     sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

//     //队列提交命令组
//     q.submit([&](handler &h) {
//     {{ACCESSOR_LIST}}
//     {{ACCESSOR_INIT}}
//         h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
//             const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
//             if(item_id >= Item_Size)
//                 return;
//             // 索引初始化
// 			{{INDEX_INIT}}
//             // 获得accessor指针
//             {{ACCESSOR_POINTER_LIST}}
//             // 嵌入计算
// 			{{CALC_EMBED}}
//         });
//     }).wait();
    
// )~~~";

//二维划分的Buffer内核执行模板
const char *KERNEL_EXECUTE_Template = R"~~~(
    sycl::device device = dacpp_q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];

	// 二维划分（可测试三维拓展）
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);

    // 固定 local 为 16*16,但受设备上限约束
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);

    // 对齐 global 到 local 的整数倍（防止越界）
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);

    //队列提交命令组
    dacpp_q.submit([&](handler &h) {
    {{ACCESSOR_LIST}}
    {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			{{INDEX_INIT}}
			// 获得划分数据单元左上角（第一个元素）的位置
			{{GETPOS}}
            // 获得accessor指针
            {{ACCESSOR_POINTER_LIST}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
    
)~~~";

const char *KERNEL_EXECUTE_Template1 = R"~~~(
	    //队列提交命令组
    dacpp_q.submit([&](handler &h) {
    {{ACCESSOR_LIST}}
    {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			{{INDEX_INIT}}			
            // 获得划分数据单元左上角（第一个元素）的位置
			{{GETPOS}}
            // 获得accessor指针
            {{ACCESSOR_POINTER_LIST}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
)~~~";


const char *submitCode = R"~~~(
    dacpp_q.submit([&](handler &h) {
    {{ACCESSOR_LIST}}
    {{ACCESSOR_INIT}}
        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gx = item.get_global_id(0);
            int gy = item.get_global_id(1);
            int item_id = gx * global[1] + gy;
            if(item_id >= Item_Size)
                return;
            // 索引初始化
			{{INDEX_INIT}}            
            // 获得划分数据单元左上角（第一个元素）的位置
			{{GETPOS}}
            // 获得accessor指针
            {{ACCESSOR_POINTER_LIST}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
    
)~~~";
//for循环包含内核函数的情况下的二维划分的Buffer内核执行模板
const char *KERNEL_EXECUTE_Template2 = R"~~~(
    sycl::device device = dacpp_q.get_device();
    auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    int max_global_size_x = max_sizes[0];
    int max_global_size_y = max_sizes[1];
    int max_global_size_z = max_sizes[2];

	// 二维划分（可测试三维拓展）
    int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
    int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);

    // 固定 local 为 16*16,但受设备上限约束
    int local_x = std::min(16, max_global_size_x);
    int local_y = std::min(16, max_global_size_y);

    // 对齐 global 到 local 的整数倍（防止越界）
    int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
    int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

    sycl::range<2> local(local_x, local_y);
    sycl::range<2> global(global_x, global_y);

    {{KERNEL}}
    
)~~~";
//内核执行中的{{ACCESSOR_LIST}}，需要把A、B、C都传入，以免受到Rewriter_Buffer.cpp中第204行的判断读写的逻辑的干扰
// const char* ACCESSOR_LIST_K =  R"~~~(
//         accessor acc_{{NAME}}{r_{{NAME}}, h};)~~~";
// std::string CodeGen_AccessorInit0(std::string name){
// 	return templateString(ACCESSOR_LIST_K,{
// 		{"{{NAME}}", name}
// 	});
// }
//2025.12.4修改：对于只读的数据，将buffer的访问模式修改为只读 然后禁止数据写回操作。对于只写的数据，访问模式设置为覆盖写 同时注意 写成这样*r_matC。
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
//内核执行中的{{ACCESSOR_POINTER_LIST}}，需要把A、B、C都传入，以免受到Rewriter_Buffer.cpp中第204行的判断读写的逻辑的干扰
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

// 获得划分数据单元左上角（第一个元素）的位置
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

// std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit,std::string ACCESSOR_LIST1, std::string ACCESSOR_LIST2, std::string CalcEmbed){
//     return templateString(KERNEL_EXECUTE_Template,
// 	{
// 		{"{{SPLIT_SIZE}}",    SplitSize},
// 		{"{{INDEX_INIT}}",    IndexInit},
// 		{"{{CALC_EMBED}}",    CalcEmbed},
//         {"{{ACCESSOR_INIT}}", AccessorInit},
//         // {"{{ACCESSOR_LIST}}",   BUFFER_ACCESSOR_LIST},
// 		{"{{ACCESSOR_LIST}}",  ACCESSOR_LIST1},
//         {"{{ACCESSOR_POINTER_LIST}}",   ACCESSOR_LIST2}
// 	});
// }

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
//当匹配到包含内核的for循环时，我们采用这个函数来重写内核提交
std::string CodeGen_KernelExecute2(std::string SplitSize, std::string AccessorInit, std::string IndexInit, std::string getpos, std::string ACCESSOR_LIST1, std::string ACCESSOR_LIST2, std::string CalcEmbed, dacppTranslator::DacppFile* dacppFile){

    std::string forBodyTemplate=extractAndReplaceSubmitCodeGeneric(dacppFile->forStatement,dacppFile->Context,dacppFile);//得到对应for循环的源代码，并将for循环中的表达式替换成{{SUBMIT}},还能够把子for循环替换成parallel_for
	std::string KERNEL_EXECUTE=templateString(forBodyTemplate,{{"{{SUBMIT}}",submitCode}});//将{{SUBMIT}}替换成内核提交代码模板
	std::string KERNEL_TEMPLATE=templateString(KERNEL_EXECUTE_Template2,{{"{{KERNEL}}",KERNEL_EXECUTE}});//将内核执行模板中的{{KERNEL}}替换成for循环的内核提交代码，然后下一行返回最终生成的代码
	return templateString(KERNEL_TEMPLATE,
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

// 访问器初始化
// {{ACCESSOR_INIT}}
// const char *ACCESSOR_INIT_Template = R"~~~(
//         auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read_write>(h);)~~~";
// std::string CodeGen_AccessorInit(std::string name) {
// 	return templateString(ACCESSOR_INIT_Template,
// 	{
// 		{"{{NAME}}",    name}
// 	});
// }
//2025.12.4修改：对于数据单元形状的访问器，将模式从读写模式修改为只读模式
const char *ACCESSOR_INIT_Template = R"~~~(
        auto info_partition_{{NAME}}_accessor = info_partition_{{NAME}}_buffer.get_access<sycl::access::mode::read>(h);)~~~";
std::string CodeGen_AccessorInit(std::string name) {
	return templateString(ACCESSOR_INIT_Template,
	{
		{"{{NAME}}",    name}
	});
}

//索引初始化
//{{INDEX_INIT}}
const char *INDEX_INIT_Template = R"~~~(
            const auto {{NAME}}={{EXPRESSION}};)~~~";
//新的索引生成模板 相当于现在的ops能用的只有算子的名字了 算子的划分数是不会改变的
std::string CodeGen_IndexInit2(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets)//sets表示每个算子属于的集合的名字 offsets表示每个算子相对于集合的偏移量
{ 
    std::set<std::string> sets_map;//用于辅助找到不同的集合的个数
    std::vector<std::string> sets_order;//记录了不同的集合出现的顺序，储存集合的名字： idx idy idz
    std::vector<std::string> sets_split;//记录了不同集合对应的划分数，与集合名相对应： idx的划分数 idy的划分数 idz的划分数 
    for (int i = 0; i < sets.size(); ++i) 
    {
		std::string ops_i_name = ops[i].name;
        if (sets_map.find(sets[i]) == sets_map.end())//如果容器里没有
        {
            sets_map.insert(sets[i]);//将集合插入容器
            sets_order.push_back(sets[i]);//将集合放入到集合的数组中
            sets_split.push_back(ops_i_name + ".split_size");//将集合对应的划分数放入数组中
        }
    }
    
    int sets_size = sets_map.size();//得到各类集合总个数
    std::unordered_map<std::string,std::string> sets_sub_expression;//<集合的名称，集合对应的索引表达式>

    for(int i = 0;i < sets_size; i++)//有几个集合就循环几次
    {
		std::string sub_expression = "item_id";
		for(int j = i + 1;j < sets_size;j ++){
			sub_expression = sub_expression + "/" + sets_split[j];
		}
		//sub_expression = sub_expression + "%" + std::to_string(sets_split[i]);//取模操作应该在偏移之后
        sets_sub_expression[sets_order[i]] = sub_expression;//将子表达式和集合的名字进行关联
	}

    //下面根据偏移量来计算各个算子对应的索引
    int len = ops.size;
	std::vector<std::string> index_expression_vector;
    for(int i = 0;i < len;i ++)
    {
        std::string index_expression = "(";
        index_expression = index_expression + sets_sub_expression[sets[i]];//得到集合的索引
        //index_expression = index_expression + "+" + "(" + offsets[i] + ")" + "+" + std::to_string(ops[i].split_size) + ")";//加上偏移量和划分数 防止出现负数
		index_expression = index_expression + "+" + "(" + offsets[i] + ")" + ")";
		index_expression = index_expression + "%" + ops[i].name + ".split_size";
		index_expression_vector.push_back(index_expression);
    }

	std::string expression = "";
	for(int i=0;i<len;i++){
		std::string opsname = ops[i].name;
		std::string index_i_expression = index_expression_vector[i];
		expression = expression + templateString(INDEX_INIT_Template,
		{
			{"{{NAME}}", opsname + "_"},//注意这里加了下划线
			{"{{EXPRESSION}}", index_i_expression}
		});
	}
	return expression;
}

//嵌入计算
//{{CALC_EMBED}}
// const char *CALC_EMBED_Template = R"~~~(
//             {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

//waveEq(d_matCur+(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),d_matPrev+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),d_matNext+(sp1_*SplitLength[2][0]+sp2_*SplitLength[2][1]),info_partition_matCur_accessor,info_partition_matPrev_accessor,info_partition_matNext_accessor);
//所有的d_name需要换成r_name
// std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names){
// 	std::string DacCalcArgs = "(";//name(参数) 中的(
// 	int len = args.size;//len 表示有几组数据
// 	for(int i=0;i<len;i++)
// 	{
// 		std::string IndexComb="(";//这个左括号代表数据在长向量中偏移量
// 		for(int j=0;j<args[i].ops.size;j++)//第j组数据中算子组中包含的算子的个数
// 		{
// 			std::string opsname = args[i].ops[j].name;//算子是char[5]类型的，这里需要转换为string后面方便使用。得到第j个算子的名字
// 			IndexComb+= opsname + "_" + "*" + "SplitLength[" + std::to_string(i) + "][" + std::to_string(j) + "]";//给算子加一个下划线 这里是为了得到算子的划分长度 具体逻辑忘了
// 			if(j!=args[i].ops.size-1) IndexComb+="+";//还没有到最后一个算子，就继续添加+号
// 		}
// 		IndexComb+=")";//添加偏移量中的右括号
// 		if(IndexComb == "()")//如果是空的话，就不要加IndexComb了
// 		{
// 			DacCalcArgs+=args[i].name;
// 		}
// 		else
// 		{
// 			DacCalcArgs+=args[i].name + "+" + IndexComb;
// 		}		
// 		DacCalcArgs += ",";//因为后面还有其他的参数，所以就继续添加 ,
// 	}
// 	//添加数据访问器的相关参数
// 	for (int z = 0; z < accessor_names.size(); z++) 
// 	{
// 		DacCalcArgs += "info_partition_" + accessor_names[z]+"_accessor";
// 		if (z == accessor_names.size() - 1) 
// 		{
// 			DacCalcArgs += ");";
// 		} 
// 		else 
// 		{
// 			DacCalcArgs += ",";
// 		}
// 	}
// 	return templateString(CALC_EMBED_Template,
// 	{
// 		{"{{DAC_CALC_NAME}}",    Name},
// 		{"{{DAC_CALC_ARGS}}",    DacCalcArgs}
// 	});
// }
const char *CALC_EMBED_Template = R"~~~(
            {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

std::string CodeGen_CalcEmbed2(std::string Name, std::vector<std::string> splits, std::vector<int> splitNum,std::vector<std::string> accessor_names) {

    std::string DacCalcArgs = "("; // calc(

    // 1) 添加d_name
        for (int z = 0; z < accessor_names.size(); z++) {
        DacCalcArgs += "d_" + accessor_names[z] +",";
    }
    // 2) 添加name_dim
        for (int z = 0; z < accessor_names.size(); z++) {
            for(int zz = 0; zz < splitNum[z]; zz++){
                DacCalcArgs += accessor_names[z] + "_" + splits[zz] +",";
            }
    }
    // 3）添加info_name_Shape[dim]
        for (int z = 0; z < accessor_names.size(); z++) {
            for(int zz = 0; zz < splitNum[z]; zz++){
                DacCalcArgs += "info_" + accessor_names[z] + "_Shape[" + splits[zz] + "],";
            }
    }
    // 4) 添加 info_partition_name_accessor
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


//归约
//{{REDUCTION}}
//归约中将b_name修改为了r_name，因为现在是使用r_name进行计算
const char *REDUCTION_Template_Span = R"~~~(
    // 归约
    if({{SPLIT_SIZE}} > 1)
    {
        for(int i=0;i<{{SPAN_SIZE}};i++) {
            dacpp_q.submit([&](handler &h) {
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

//结果返回
//{{D2H_MEM_MOV}}
//先将r_name中的数据逆重组到b_name,再将b_name数据传输到h_name
// const char *RESULT_B2H_MOV_Template = R"~~~(
//     //结果返回
//     {{NAME}}_tool.UpdateData(r_{{NAME}},b_{{NAME}},q,{{NAME}}_Size);
//     {
//         host_accessor temp_accessor{b_{{NAME}}};
//         for(int i = 0; i < {{SIZE}}; i++){
//             h_{{NAME}}[i] = temp_accessor[i];
//         }
//     }
//     {{NAME}}.array2Tensor(h_{{NAME}});
// )~~~";

// const char *RESULT_B2H_MOV_Template = R"~~~(
//     //结果返回
    
//     {
//         host_accessor temp_accessor{r_{{NAME}}};
//         for(int i = 0; i < {{NAME}}.getSize(); i++){
//             h_{{NAME}}[i] = temp_accessor[i];
//         }
//     }
//     {{NAME}}.array2Tensor(h_{{NAME}});
// )~~~";

const char *RESULT_B2H_MOV_Template = R"~~~(
    //结果返回语句改为析构语句
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
