#ifndef TRANSLATOR_PARSER_PARAM_H
#define TRANSLATOR_PARSER_PARAM_H

#include <string>
#include <vector>
#include "clang/AST/Type.h"

#include "Split.h"

namespace dacppTranslator {

enum class IOTYPE{
    READ,
    WRITE,
    READ_WRITE
};
class Param {

private:
    IOTYPE rw;
    std::string name;
    std::vector<int> shape;
    int dimension;

public:
    Param();
    std::string rule;

    void setRw(IOTYPE type);
    IOTYPE getRw();

    void setType(clang::QualType newType);
    std::string getType();
    std::string getBasicType();

    void setName(std::string name);
    std::string getName();

    void __attribute__(( unused, deprecated )) setShape(int size) ;
    void __attribute__(( unused, deprecated )) setShape(int idx, int size);
    int __attribute__(( unused, deprecated )) getShape(int idx);
    int __attribute__(( unused, deprecated )) getDim();

    int dim;
    void setDimension(int id);
    int getDimension();

    clang::QualType newType, BasicType;
};

class ShellParam : public Param {

private:
    std::vector<Split*> splits;

public:
    ShellParam();

    void setSplit(Split* split);
    Split* getSplit(int idx);
    int getNumSplit();

};

}

#endif