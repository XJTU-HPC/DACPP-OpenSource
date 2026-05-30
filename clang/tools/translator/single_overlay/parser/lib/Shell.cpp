#include <string>
#include <iostream>

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/StringExtras.h"

#include "ASTParse.h"
#include "DacppStructure.h"
#include "Param.h"
#include "Shell.h"
#include <vector>

typedef struct ArcNode
{
    int adjvex;
    struct ArcNode *nextarc;
    char *offset;
}ArcNode;
std::vector <int> v_dim;
struct VNode {
    int id;
    clang::ValueDecl *D;
    dacppTranslator::Split *s;
    ArcNode *firstarc;
} ;

struct ALGraph {

    VNode *vertices;

    int vexnum;

    int allocated;
} ;

static int LocateVex(ALGraph *G,const clang::ValueDecl *u)
{

  int i;
  for(i=0;i<G->vexnum;++i)
    if(u == G->vertices[i].D)
      return i;
  return -1;
}

static void DestroyGraph(ALGraph *G)
{
  int i;
  ArcNode *p,*q;
  G->vexnum = G->allocated = 0;
  for(i=0;i<(*G).vexnum;++i)
  {
    p=(*G).vertices[i].firstarc;
    while(p)
    {
      q=p->nextarc;
      free ((void *)p->offset);
      free(p);
      p=q;
    }
  }
  free(G);
}

static ALGraph *CreateGraph(void)
{
  ALGraph *G = (ALGraph*) malloc (sizeof (struct ALGraph));
  memset (G, 0, sizeof (*G));
  return G;
}

static VNode* GetVex(ALGraph *G,int v)
{
  return &G->vertices[v];
}

static int InsertVex(ALGraph *G,clang::ValueDecl *v, dacppTranslator::Split *s)
{

  if (G->allocated <= G->vexnum)
  {
    G->vertices = (VNode *)realloc (G->vertices, (1 + G->allocated) * sizeof (VNode));
    G->allocated += 1;
  }
  (*G).vertices[(*G).vexnum].D = v;
  (*G).vertices[(*G).vexnum].s = s;
  (*G).vertices[(*G).vexnum].id = (*G).vexnum;
  (*G).vertices[(*G).vexnum].firstarc=NULL;
  (*G).vexnum++;
  return (*G).vexnum - 1;
}

static void InsertArc(ALGraph *G,int i,int j, const char *offset)
{

  ArcNode *p;
  p=(ArcNode*)malloc(sizeof(ArcNode));
  p->offset = (char *) malloc (sizeof (char) *
         (strlen (offset) + 1));
  strcpy (p->offset, offset);
  p->adjvex=j;
  p->nextarc=(*G).vertices[i].firstarc;
  (*G).vertices[i].firstarc=p;
}

dacppTranslator::Shell::Shell() {
    this->G = CreateGraph ();
}

dacppTranslator::Shell::~Shell() {
    DestroyGraph (this->G);
}

void dacppTranslator::Shell::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::Shell::getName() {
    return name;
}

void dacppTranslator::Shell::setParam(Param* param) {
    params.push_back(param);
}

dacppTranslator::Param* dacppTranslator::Shell::getParam(int idx) {
    return params[idx];
}

int dacppTranslator::Shell::getNumParams() {
    return params.size();
}

void dacppTranslator::Shell::setSplit(Split* split) {
    splits.push_back(split);
}

dacppTranslator::Split* dacppTranslator::Shell::getSplit(int idx) {
    return splits[idx];
}

int dacppTranslator::Shell::getNumSplits() {
    return splits.size();
}

void dacppTranslator::Shell::setShellParam(ShellParam* param) {
    shellParams.push_back(param);
}

dacppTranslator::ShellParam* dacppTranslator::Shell::getShellParam(int idx) {
    return shellParams[idx];
}

int dacppTranslator::Shell::getNumShellParams() {
    return shellParams.size();
}

void dacppTranslator::Shell::setFather(Expression* expr) {
    this->father = expr;
}

dacppTranslator::Expression* dacppTranslator::Shell::getFather() {
    return father;
}

void dacppTranslator::Shell::setShellLoc(FunctionDecl* shellLoc) {
    this->shellLoc = shellLoc;
}

FunctionDecl* dacppTranslator::Shell::getShellLoc() {
    return shellLoc;
}

struct Visitor : RecursiveASTVisitor<Visitor> {
    dacppTranslator::Shell *sh;
    const BinaryOperator* dacExpr;

    Visitor(dacppTranslator::Shell *sh, const BinaryOperator* dacExpr)
        : sh(sh), dacExpr(dacExpr) {}

    Expr *ignoreImplicitSemaNodes(Expr *Node) {
        Expr *E = Node;
        while (true) {
          if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
            E = MTE->getSubExpr();
          if (auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E))
            E = BTE->getSubExpr();
          if (auto *ICE = dyn_cast<ImplicitCastExpr>(E))
            E = ICE->getSubExpr();
          else
            break;
        }

        return E;
    }

    bool VisitVarDecl(VarDecl *D);
    bool VisitCallExpr(CallExpr *Call);
};

bool Visitor::VisitCallExpr(CallExpr *Call) {
  std::string TempString;
  llvm::raw_string_ostream SS(TempString);
  unsigned i, e;
  int v1 = -1,v2 = -1;
  std::string offset1, offset2;

  Call->getCallee()->printPretty(SS, nullptr, PrintingPolicy(LangOptions()));

  if (!strcmp(SS.str().c_str(), "binding")) {
    for (i = 0, e = Call->getNumArgs(); i != e; ++i) {
      if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {

        break;
      }
      if (Expr *tempExpr = ignoreImplicitSemaNodes(Call->getArg(i))) {

        if (CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(tempExpr)) {
          if (CCE->getNumArgs() == 1) {
            tempExpr = ignoreImplicitSemaNodes(CCE->getArg(0));
            if (const auto *DeclRef = dyn_cast<DeclRefExpr>(tempExpr))

              (i % 2 ? v2 : v1) = LocateVex (sh->G, DeclRef->getDecl());
          }
        }

        else if (auto *OpCallExpr = dyn_cast<CXXOperatorCallExpr>(tempExpr)) {

          if (OpCallExpr->getNumArgs() == 2) {
            llvm::raw_string_ostream Buf(i % 2 ? offset2 : offset1);
            tempExpr = ignoreImplicitSemaNodes(OpCallExpr->getArg(0));
            if (const auto *DeclRef = dyn_cast<DeclRefExpr>(tempExpr))

              (i % 2 ? v2 : v1) = LocateVex (sh->G, DeclRef->getDecl());

            Buf << ' ' << getOperatorSpelling(OpCallExpr->getOperator()) << ' ';
            OpCallExpr->getArg(1)->printPretty(Buf, nullptr, PrintingPolicy(LangOptions()));
          }
        }
      }
    }
    if (v1 != v2) {

      if (offset1.c_str()[0])
        offset2 += " - (" + offset1 + ")";

      InsertArc(sh->G, v1, v2, offset2.c_str());
    }
  }
  return true;
}
int countBrackInExpr(clang::Expr *expr){
   if(!expr) return 0;
   int count=0;
   if(auto *op = llvm::dyn_cast<CXXOperatorCallExpr>(expr)){
    auto kind =op->getOperator();
    if(kind == clang::OO_Subscript){
      count++;
    }
   }
   for(auto *child :expr->children()){
    if(auto *childexpr = llvm::dyn_cast<Expr>(child)){
      count+=countBrackInExpr(childexpr);
    }
   }
   return count;
}
bool Visitor::VisitVarDecl (VarDecl *D)
{
  VarDecl *curVarDecl = D;
  do

  {
    if (curVarDecl->getType().getAsString().compare("dacpp::list") != 0 &&
        curVarDecl->getType().getAsString().compare("dacpp::index") != 0 &&
        curVarDecl->getType().getAsString().compare("dacpp::split") != 0) {
      break;
    }

    if (curVarDecl->getType().getAsString().compare("dacpp::index") == 0) {
      dacppTranslator::IndexSplit *sp = new dacppTranslator::IndexSplit(nullptr);
      sp->setId(curVarDecl->getNameAsString());
      sp->type = "IndexSplit";
      sh->setSplit(sp);
      if (-1 == LocateVex(sh->G, curVarDecl)) {
        sp->v = GetVex (sh->G, InsertVex (sh->G, curVarDecl, sp));
      }
      break;
    }

    if (curVarDecl->getType().getAsString().compare("dacpp::split") ==
        0) {
      dacppTranslator::RegularSplit *sp = new dacppTranslator::RegularSplit(nullptr);
      sp->setId(curVarDecl->getNameAsString());
      CXXConstructExpr *CCE = nullptr;
      if (isa<CXXConstructExpr>(curVarDecl->getInit())) {
        CCE = dyn_cast<CXXConstructExpr>(curVarDecl->getInit());
      } else {
        CCE = dacppTranslator::getNode<CXXConstructExpr>(curVarDecl->getInit());
      }
      int count = 0;
      for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                          E = CCE->arg_end();
           I != E; ++I) {
        if (count == 0) {

          sp->setSplitSize(std::stoi(
              toString((dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
        } else if (count == 1) {

          sp->setSplitStride(std::stoi(
              toString((dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
        }
        count++;
      }
      sp->type = "RegularSplit";
      sh->setSplit(sp);
      if (-1 == LocateVex (sh->G, curVarDecl)) {
        sp->v = GetVex (sh->G, InsertVex (sh->G, curVarDecl, sp));
      }
      break;
    }

    InitListExpr *ILE =
        dacppTranslator::getNode<InitListExpr>(curVarDecl->getInit());
    for (unsigned int i = 0; i < ILE->getNumInits(); i++) {
      Expr* childExpr = ILE->getInit(i);
      int bracketCount = countBrackInExpr(childExpr);
      v_dim.push_back(bracketCount);
      dacppTranslator::ShellParam *shellParam = new dacppTranslator::ShellParam();
      shellParam->setDimension(bracketCount);
      Expr *curExpr = ILE->getInit(i);
      std::vector<Expr *> astExprs;

      std::string name = "";
      dacppTranslator::getSplitExpr(curExpr, name, astExprs);
      shellParam->setName(name);
      for (unsigned int paramsCount = 0;
           paramsCount < sh->getShellLoc()->getNumParams(); paramsCount++) {
        if (shellParam->getName() !=
            sh->getShellLoc()->getParamDecl(paramsCount)->getNameAsString()) {
          continue;
        }
        shellParam->setRw(
            dacppTranslator::inputOrOutput(sh->getShellLoc()->getParamDecl(paramsCount)));

        shellParam->setType(sh->getParam(paramsCount)->newType);
        shellParam->setName(sh->getParam(paramsCount)->getName());

      }
      for (unsigned int i = 0; i < astExprs.size(); i++) {
        if (dacppTranslator::getNode<DeclRefExpr>(astExprs[i])) {
          VarDecl *vd = dyn_cast<VarDecl>(
              dacppTranslator::getNode<DeclRefExpr>(astExprs[i])->getDecl());

          if (vd->getType().getAsString().compare("dacpp::split") == 0) {
            dacppTranslator::RegularSplit *sp =
                new dacppTranslator::RegularSplit((dacppTranslator::RegularSplit *) GetVex(sh->G, LocateVex (sh->G, vd))->s);
            sp->type = "RegularSplit";
            sp->setId(vd->getNameAsString());
            sp->setDimIdx(i);
            CXXConstructExpr *CCE = nullptr;
            if (isa<CXXConstructExpr>(vd->getInit())) {
              CCE = dyn_cast<CXXConstructExpr>(vd->getInit());
            } else {
                dacppTranslator::getNode<CXXConstructExpr>(vd->getInit());
            }
            int count = 0;
            for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                                E = CCE->arg_end();
                 I != E; ++I) {
              if (count == 0) {

                sp->setSplitSize(std::stoi(toString(
                    (dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
              } else if (count == 1) {

                sp->setSplitStride(std::stoi(toString(
                    (dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
              }
              count++;
            }

            for (int m = 0; m < sh->getNumSplits(); m++) {
              if (sh->getSplit(m)->getId().compare(sp->getId()) == 0 &&
                  sh->getSplit(m)->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit *isp =
                    static_cast<dacppTranslator::RegularSplit *>(
                        sh->getSplit(m));
                isp->setSplitNumber(sp->getSplitNumber());
              }
            }
            shellParam->setSplit(sp);
          } else if (vd->getType().getAsString().compare("dacpp::index") == 0) {
            dacppTranslator::IndexSplit *sp = new dacppTranslator::IndexSplit((dacppTranslator::IndexSplit *) GetVex(sh->G, LocateVex (sh->G, vd))->s);
            sp->type = "IndexSplit";
            sp->setId(vd->getNameAsString());
            sp->setDimIdx(i);

            for (int m = 0; m < sh->getNumSplits(); m++) {
              if (sh->getSplit(m)->getId().compare(sp->getId()) == 0 &&
                  sh->getSplit(m)->type.compare("IndexSplit") == 0) {
                dacppTranslator::IndexSplit *isp =
                    static_cast<dacppTranslator::IndexSplit *>(sh->getSplit(m));
                isp->setSplitNumber(sp->getSplitNumber());
              }
            }
            shellParam->setSplit(sp);
          }
        } else {
          dacppTranslator::Split *sp = new dacppTranslator::Split(nullptr);
          sp->type = "Split";
          sp->setId("void");
          sp->setDimIdx(i);
          shellParam->setSplit(sp);
        }
      }
      sh->setShellParam(shellParam);
    }
  } while (0);

  return true;
}

void dacppTranslator::Shell::GetBindInfo(
    std::vector<BINDINFO> *pbindInfo)
{
  bool *visited;
  std::queue<BINDINFO *> Q;
  int v;
  BINDINFO bindinfo ;
  std::string parent;
  int icls = 0;
  ArcNode *p;
  int *refs;

  pbindInfo->clear();
  visited = (bool *)malloc(sizeof(bool) * G->vexnum);
  memset(visited, 0, sizeof(bool) * G->vexnum);
  refs = (int *)malloc(sizeof(int) * G->vexnum);
  memset(refs, 0, sizeof(int) * G->vexnum);

  for(v=0;v<G->vexnum;v++)
    for (p = G->vertices[v].firstarc; p; p = p->nextarc)
      refs[p->adjvex]++;

  for(v=0;v<G->vexnum;v++)
  {
    if(!visited[v] && !refs[v])
    {
      icls++;
      visited[v]=true;
      bindinfo.icls = icls;
      bindinfo.v = GetVex (G, v);
      bindinfo.offset = "";
      pbindInfo->push_back(bindinfo);
      Q.push (&bindinfo);
      while(!Q.empty())
      {
        bindinfo = *Q.front();
        parent = bindinfo.offset;
        Q.pop();
        for (p = G->vertices[bindinfo.v->id].firstarc; p; p = p->nextarc)
        {
          if(!visited[p->adjvex])
          {
            visited[p->adjvex]=true;
            bindinfo.icls = icls;
            bindinfo.v = GetVex (G, p->adjvex);
            bindinfo.offset = parent + p->offset;
            pbindInfo->push_back(bindinfo);
            Q.push (&bindinfo);
          }
        }
      }
    }
  }
  free(refs);
  free(visited);
}

dacppTranslator::Split *dacppTranslator::Shell::search_symbol(VNode *v) {
  dacppTranslator::Split *s = v->s;
  while (s->parent) s = s->parent;
  return s;
}

void dacppTranslator::Shell::parseShell(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes) {
  std::string Msg;
  llvm::raw_string_ostream SS(Msg);
    Visitor V (this, dacExpr);

  Expr *dacExprLHS = dacppTranslator::Expression::shellLHS_p (dacExpr) ? dacExpr->getLHS() : dacExpr->getRHS();
  CallExpr *shellCall = getNode<CallExpr>(dacExprLHS);
  FunctionDecl *shellFunc = shellCall->getDirectCallee();

  setShellLoc(shellFunc);

  setName(shellFunc->getNameAsString());

  for (unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams();
       paramsCount++) {
    Param *param = new Param();

    param->setRw(inputOrOutput(shellFunc->getParamDecl(paramsCount)));

    param->setType(shellFunc->getParamDecl(paramsCount)->getType());

    param->setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());

    for (unsigned int i = 0; i < shapes[paramsCount].size(); i++) {
      param->setShape(shapes[paramsCount][i]);
    }

    setParam(param);
    }

    Stmt* shellFuncBody = shellFunc->getBody();
    V.TraverseStmt (shellFuncBody);

    std::vector<BINDINFO> bindInfo;
    GetBindInfo (&bindInfo);
}
