#include <set>
#include <iostream>
#include <sstream>
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringExtras.h"
#include "Rewriter.h"
#include "Split.h"
#include "Param.h"
#include "dacInfo.h"
#include "universal_template.h"
#include "buffer_template.h"
#include "usm_template.h"
#include "mpi_template.h"

using namespace MPI_TEMPLATE;

void dacppTranslator::Rewriter::rewriteMPI() {
    CompoundStmt *Node =
        dyn_cast<CompoundStmt>(dacppFile->getMainFuncLoc()->getBody());
    Stmt *first = nullptr, *last;
    for (Stmt *I : Node->body()) {
    if (!first)
        first = I;
    last = I;
    }

    rewriter->InsertTextBefore(last->getSourceRange().getBegin(),
    MPI_FINISH_Template);

    for (int exprCount = dacppFile->dacExprs.size() - 1; exprCount >= 0; exprCount--) {
        //找到数据关联计算表达式
        const BinaryOperator *dacExpr = dacppFile->dacExprs[exprCount];
        CallExpr *shellCall = dacppTranslator::getNode<CallExpr>(dacExpr->getLHS());
        DeclRefExpr *declRefExpr;
        if (isa<DeclRefExpr>(dacExpr->getRHS())) {
            declRefExpr = dyn_cast<DeclRefExpr>(dacExpr->getRHS());
        } else {
            declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());
        }

        // 数据分发代码和数据收集代码
        string scatter_code;
        string gather_code;

        // 访问shellCall的每一个参数
        Expression* expr = dacppFile->getExpression(exprCount);
        Shell* shell = expr->getShell();

        for(int NumShellParam = 0; NumShellParam < shell->getNumShellParams(); NumShellParam++) {
            ShellParam* shellParam = shell->getShellParam(NumShellParam);
            // string varName = shellParam->getName();
            string varType = shellParam->getBasicType();

            Expr *argExpr = shellCall->getArg(NumShellParam)->IgnoreImpCasts();
            std::string varName;

            // 处理变量名
            if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(argExpr)) {
                varName = DRE->getDecl()->getNameAsString();
            } else if (MemberExpr *ME = dyn_cast<MemberExpr>(argExpr)) {
                varName = ME->getMemberDecl()->getNameAsString();
            } else {
                varName = "<unknown>"; // 非直接变量引用的情况
            } 
            if(shellParam->getRw() == 1) {
                //数据转换：dacpp转换为std
                if(shellParam->getNumSplit() == 1) {
                    Split* split = shellParam->getSplit(0);
                    if(split->getId() == "void") {
                        continue;
                    }
                    else if(split->type.compare("IndexSplit") == 0) {
                        IndexSplit* indexSplit = static_cast<IndexSplit*>(split);
                        scatter_code = CodeGen_res_data_size(varName) + scatter_code;
                        scatter_code = UNIVERSAL_TEMPLATE::CodeGen_DataInfoInit(varName) + scatter_code;
                        scatter_code += CodeGen_dacpp2std_index(varType, varName, "res_data_size", "mpi_size");
                        scatter_code += CodeGen_std2dacpp(varType, "1", varName);
                        gather_code += CodeGen_dacpp2std_gather(varName);
                        gather_code += CodeGen_gather1D(varName, "res_data_size", "mpi_size", varType);
                        gather_code += CodeGen_std2dacpp_gather(varName);
                    }
                    else if(split->type.compare("RegularSplit") == 0) {
                        RegularSplit* regularSplit = static_cast<RegularSplit*>(split);
                        string split_length = std::to_string(regularSplit->getSplitSize());
                        string split_step = std::to_string(regularSplit->getSplitStride());
                        scatter_code = CodeGen_res_data_size(varName) + scatter_code;
                        scatter_code = UNIVERSAL_TEMPLATE::CodeGen_DataInfoInit(varName) + scatter_code;
                        scatter_code += CodeGen_dacpp2std_split(varType, varName, "res_data_size",
                                                            "mpi_size", split_length, split_step);
                        scatter_code += CodeGen_std2dacpp(varType, "1", varName);
                        gather_code += CodeGen_dacpp2std_gather(varName);
                        gather_code += CodeGen_gather1D(varName, "res_data_size", "mpi_size", varType);
                        gather_code += CodeGen_std2dacpp_gather(varName);
                    }
                }
                else if(shellParam->getNumSplit() == 2) {
                    Split* split = shellParam->getSplit(0);
                    if(split->getId() == "void") {
                        continue;
                    }
                    else if(split->type.compare("IndexSplit") == 0) {
                        IndexSplit* indexSplit = static_cast<IndexSplit*>(split);
                        scatter_code = CodeGen_res_data_size(varName) + scatter_code;
                        scatter_code = UNIVERSAL_TEMPLATE::CodeGen_DataInfoInit(varName) + scatter_code;
                        scatter_code += CodeGen_dacpp2std_index(varType, varName, "res_data_size", "mpi_size");
                        scatter_code += CodeGen_std2dacpp_2d(varType, "mpi_size", varName);
                        gather_code += CodeGen_dacpp2std_gather(varName);
                        gather_code += CodeGen_gather2D_row(varName, "res_data_size", "mpi_size", varType);
                        gather_code += CodeGen_std2dacpp_gather(varName);
                    }
                }
            }
            else {
                if(shellParam->getNumSplit() == 1) {
                    Split* split = shellParam->getSplit(0);
                    if(split->getId() == "void") {
                        scatter_code += CodeGen_local_dacpp_keep(varType, "1", varName);
                    }
                    else if(split->type.compare("IndexSplit") == 0) {
                        IndexSplit* indexSplit = static_cast<IndexSplit*>(split);
                        scatter_code += CodeGen_dacpp2std_index(varType, varName, "res_data_size", "mpi_size");
                        scatter_code += CodeGen_Scatter1DIndex(varType, varName, "res_data_size", "mpi_size");
                        scatter_code += CodeGen_std2dacpp(varType, "1", varName);
                    }
                    else if(split->type.compare("RegularSplit") == 0) {
                        RegularSplit* regularSplit = static_cast<RegularSplit*>(split);
                        string split_length = std::to_string(regularSplit->getSplitSize());
                        string split_step = std::to_string(regularSplit->getSplitStride());
                        scatter_code += CodeGen_dacpp2std_split(varType, varName, "res_data_size",
                                                            "mpi_size", split_length, split_step);
                        scatter_code += CodeGen_Scatter1DSplit(varType, varName, "res_data_size", "mpi_size",
                                                                split_length, split_step);
                        scatter_code += CodeGen_std2dacpp(varType, "1", varName);
                    }
                }
                else if(shellParam->getNumSplit() == 2) {
                    Split* split = shellParam->getSplit(0);
                    if(split->getId() == "void") {
                        scatter_code = UNIVERSAL_TEMPLATE::CodeGen_DataInfoInit(varName) + scatter_code;
                        scatter_code += CodeGen_local_dacpp_keep(varType, "2", varName);
                    }
                    else if(split->type.compare("IndexSplit") == 0) {
                        IndexSplit* indexSplit = static_cast<IndexSplit*>(split);
                        scatter_code = UNIVERSAL_TEMPLATE::CodeGen_DataInfoInit(varName) + scatter_code;
                        scatter_code += CodeGen_dacpp2std_index(varType, varName, "res_data_size", "mpi_size");
                        scatter_code += CodeGen_Scatter2D_row_index(varType, varName, "mpi_size");
                        scatter_code += CodeGen_std2dacpp_2d(varType, "mpi_size", varName);
                    }
                    else if(split->type.compare("RegularSplit") == 0) {
                        RegularSplit* regularSplit = static_cast<RegularSplit*>(split);
                        string split_length = std::to_string(regularSplit->getSplitSize());
                        string split_step = std::to_string(regularSplit->getSplitStride());
                        scatter_code = UNIVERSAL_TEMPLATE::CodeGen_DataInfoInit(varName) + scatter_code;
                        scatter_code += CodeGen_dacpp2std_split2d(varType, varName, "res_data_size",
                                                            "mpi_size", split_length, split_step);
                        scatter_code += CodeGen_Scatter2D_row_split(varType, varName, "res_data_size", "mpi_size",
                                                                split_length, split_step);
                        scatter_code += CodeGen_std2dacpp_2dsplit(varType, "mpi_size", varName);
                    }
                }
            }
            
        }
        // 生成新函数名
        std::string newFuncName = shellCall->getDirectCallee()->getNameAsString() + 
                                "_" + declRefExpr->getDecl()->getNameAsString();

        // 生成带前缀的参数列表
        std::vector<std::string> newArgs;
        LangOptions langOpts;
        langOpts.CPlusPlus = true;
        PrintingPolicy policy(langOpts);

        for (auto *arg : shellCall->arguments()) {
            std::string argText;
            llvm::raw_string_ostream rso(argText);
            
            // 处理参数中的变量引用
            if (auto *dref = dyn_cast<DeclRefExpr>(arg->IgnoreParenImpCasts())) {
                rso << "local_dacpp_" << dref->getDecl()->getName();
            } 
            // 处理成员变量（如需要）
            else if (auto *member = dyn_cast<MemberExpr>(arg->IgnoreParenImpCasts())) {
                if (auto *memberDref = dyn_cast<DeclRefExpr>(member->getBase())) {
                    rso << "local_dacpp_" << memberDref->getDecl()->getName()
                        << member->getMemberDecl()->getName();
                }
            }
            // 其他类型参数保持原样
            else {
                arg->printPretty(rso, nullptr, policy);
            }
            
            newArgs.push_back(rso.str());
        }

        // 构建最终代码
        std::string newCode = newFuncName + "(" + llvm::join(newArgs, ", ") + ")";
        // 替换原始代码
        rewriter->ReplaceText(dacExpr->getSourceRange(), newCode);

        rewriter->InsertTextBefore(dacExpr->getSourceRange().getBegin(), scatter_code);

        // rewriter->InsertTextAfterToken(dacStmt->getSourceRange().getEnd(), GATHER_1D_Template);

        /* 写入MPI_Gather代码。 */
        if (dacExpr) { // 确保找到了 dacStmt
            // 找到语句的结束位置
            SourceLocation stmtEndLoc = dacExpr->getEndLoc(); // 使用 getEndLoc()

            // 使用 Lexer 查找下一个分号之后的位置
            SourceLocation semicolonEndLoc = Lexer::findLocationAfterToken(
                stmtEndLoc,            // 从语句结束位置开始查找
                tok::semi,             // 查找分号 token
                rewriter->getSourceMgr(), // SourceManager
                rewriter->getLangOpts(),  // LanguageOptions
                true                   // 跳过注释
            );

            // 检查是否成功找到分号之后的位置
            if (semicolonEndLoc.isValid()) {
                // 插入文本到分号之后的位置
                
                rewriter->InsertTextBefore(semicolonEndLoc, gather_code); // 确保模板包含换行符
            } else {
                llvm::errs() << "[警告] 无法为 DAC 语句定位分号后的位置, MPI_Gather可能插入错误。\n";
            }
        } else {
             llvm::errs() << "[警告] 未找到 DAC 语句，跳过 MPI_Gather 插入。\n";
        }
        
    }
    
    rewriter->InsertTextBefore(first->getSourceRange().getBegin(), MPI_INIT_Template);
    rewriter->InsertText(dacppFile->node->getBeginLoc(), MPI_HEADER_Template);
    // 在rewriteMPI函数中插入主函数参数（新增代码）
    rewriter->InsertText(
        dacppFile->getMainFuncLoc()->getBeginLoc().getLocWithOffset(9), // 定位到main(位置
        "int argc, char** argv" 
    );
}