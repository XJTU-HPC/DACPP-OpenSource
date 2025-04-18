#include <string>

#include "clang/AST/Attr.h"
#include "llvm/ADT/StringExtras.h"

#include "DacppStructure.h"
#include "ASTParse.h"


using namespace clang;


/**
 * 存储头文件信息类实现
 */
dacppTranslator::HeaderFile::HeaderFile() {
}

dacppTranslator::HeaderFile::HeaderFile(std::string name) {
    this->name = name;
}

void dacppTranslator::HeaderFile::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::HeaderFile::getName() {
    return name;
}


/**
 * 存储命名空间信息类实现
 */
dacppTranslator::NameSpace::NameSpace() {
}

dacppTranslator::NameSpace::NameSpace(std::string name) {
    this->name = name;
}

void dacppTranslator::NameSpace::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::NameSpace::getName() {
    return name;
}


/**
 * 存储数据关联计算表达式信息类实现
 */
dacppTranslator::Expression::Expression() {
}

void dacppTranslator::Expression::setShell(Shell* shell) {
    this->shell = shell;
}

dacppTranslator::Shell* dacppTranslator::Expression::getShell() {
    return shell;
}

void dacppTranslator::Expression::setCalc(Calc* calc) {
    this->calc = calc;
}

dacppTranslator::Calc* dacppTranslator::Expression::getCalc() {
    return calc;
}

void dacppTranslator::Expression::setDacExpr(const clang::BinaryOperator* dacExpr) {
    this->dacExpr = dacExpr;
}

const clang::BinaryOperator* dacppTranslator::Expression::getDacExpr() {
    return dacExpr;
}

bool dacppTranslator::Expression::shellLHS_p(const BinaryOperator *dacExpr) {
  bool found_p = false;
  Expr *LHS;
  const CXXBindTemporaryExpr *BTE;
  const CallExpr *Call;
  const ImplicitCastExpr *ICE;
  const DeclRefExpr *DRE;
  const FunctionDecl *FD;

  do {
    LHS = dacExpr->getLHS();
    BTE = dyn_cast<CXXBindTemporaryExpr>(LHS);
    if (!BTE)
      break;

    Call = dyn_cast<CallExpr>(BTE->getSubExpr());
    if (!Call)
      break;

    ICE = dyn_cast<ImplicitCastExpr>(Call->getCallee());
    if (!ICE)
      break;

    DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
    if (!DRE)
      break;

    FD = dyn_cast<FunctionDecl>(DRE->getDecl());
    if (!FD)
      break;

    if (!FD->getDescribedFunctionTemplate() &&
        !FD->isFunctionTemplateSpecialization()) {
      if (FD->hasAttrs()) {
        const AttrVec &Attrs = FD->getAttrs();
        for (auto *A : Attrs) {
          std::string BSBuf;
          llvm::raw_string_ostream BSStream(BSBuf);
          A->printPretty(BSStream, PrintingPolicy(LangOptions()));
          if (!strcmp(BSBuf.c_str(), "shell")) {
            found_p = true;
            break;
          }
        }
      }
    }
  } while (0);
  return found_p;
}

/**
 * 存储DACPP文件信息类实现
 */
dacppTranslator::DacppFile::DacppFile() {
    setHeaderFile("<sycl/sycl.hpp>");
    setHeaderFile("\"DataReconstructor.h\"");
    setHeaderFile("\"ParameterGeneration.h\"");
    setNameSpace("sycl");
}

void dacppTranslator::DacppFile::setHeaderFile(std::string headerFile) {
    headerFiles.push_back(new HeaderFile(headerFile));
}

dacppTranslator::HeaderFile* dacppTranslator::DacppFile::getHeaderFile(int idx) {
    return headerFiles[idx];
}

int dacppTranslator::DacppFile::getNumHeaderFile() {
    return headerFiles.size();
}

void dacppTranslator::DacppFile::setNameSpace(std::string nameSpace) {
    nameSpaces.push_back(new NameSpace(nameSpace));
}

dacppTranslator::NameSpace* dacppTranslator::DacppFile::getNameSpace(int idx) {
    return nameSpaces[idx];
}

int dacppTranslator::DacppFile::getNumNameSpace() {
    return nameSpaces.size();
}

void dacppTranslator::DacppFile::setExpression(const BinaryOperator* dacExpr) {
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacppTranslator::Expression::shellLHS_p (dacExpr) ?  dacExpr->getLHS() : dacExpr->getRHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    // 获取实参的形状
    // 这里的实参目前指的是 DACPP:Tensor
    // TODO 
    // 实参形状这里直接在定义实参的位置找到了实参的初始化列表，在翻译的时候将实参形状硬编码到了函数中，如果多次调用同一函数，如果实参不同，会生成多个SYCL函数
    // Tensor的初始化用到了两个std::vector，如果在生成之后对其进行了push_back，则不能得到正确的形状，会出现bug
    // 如果Tensor初始化列表中的vector是从文件中读取的，也无法获得正确形状，这种情况需要在SYCL文件中把形状修改为软编码，比如Tensor.getShape(idx)
    std::vector<std::vector<int>> shapes(shellCall->getNumArgs());
    for(unsigned int paramsCount = 0; paramsCount < shellCall->getNumArgs(); paramsCount++) {
        Expr* curExpr = shellCall->getArg(paramsCount);
        DeclRefExpr* declRefExpr;
        if(isa<DeclRefExpr>(curExpr)) {
            declRefExpr = dyn_cast<DeclRefExpr>(curExpr);
        } else if(isa<ImplicitCastExpr>(curExpr)) {
            declRefExpr = getNode<DeclRefExpr>(curExpr);
        } else {
            // 带切片的Tensor
            while(getNode<CXXOperatorCallExpr>(curExpr)) {
                curExpr = getNode<CXXOperatorCallExpr>(curExpr);
            }
            int count = 0;
            for(Stmt::child_iterator it = curExpr->child_begin(); it != curExpr->child_end(); it++, count++) {
                if(count != 1) {
                    continue;
                }
                declRefExpr = getNode<DeclRefExpr>(*it);
            }
        }
        int count = 0;
        for(Stmt::child_iterator it = dyn_cast<VarDecl>(declRefExpr->getDecl())->getInit()->child_begin(); it != dyn_cast<VarDecl>(declRefExpr->getDecl())->getInit()->child_end(); it++) {
            if(count != 1) {
                count++;
                continue;
            }
            VarDecl* shapeDecl = dyn_cast<VarDecl>(getNode<DeclRefExpr>(*it)->getDecl());
            InitListExpr* initListExpr = getNode<InitListExpr>(shapeDecl->getInit());
            for(Stmt::child_iterator it = initListExpr->child_begin(); it != initListExpr->child_end(); it++) {
                IntegerLiteral* integer = dyn_cast<IntegerLiteral>(*it);
                shapes[paramsCount].push_back(std::stoi(toString(integer->getValue(), 10, true)));
            }
            count++;
        }
    }
    Shell* shell = new Shell();
    Calc* calc = new Calc();
    Expression* expr = new Expression();
    expr->setDacExpr(dacExpr);
    expr->setShell(shell);
    expr->setCalc(calc);
    shell->setFather(expr);
    calc->setFather(expr);
    shell->parseShell(dacExpr, shapes);
    calc->parseCalc(dacExpr);
    exprs.push_back(expr);
}

dacppTranslator::Expression* dacppTranslator::DacppFile::getExpression(int idx) {
    return exprs[idx];
}

int dacppTranslator::DacppFile::getNumExpression() {
    return exprs.size();
}

void dacppTranslator::DacppFile::setMainFuncLoc(const FunctionDecl* mainFuncLoc) {
    this->mainFuncLoc = mainFuncLoc;
}

const FunctionDecl* dacppTranslator::DacppFile::getMainFuncLoc() {
    return mainFuncLoc;
}