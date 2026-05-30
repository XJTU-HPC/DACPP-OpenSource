#ifndef TRANSLATOR_PARSER_SPLIT_H
#define TRANSLATOR_PARSER_SPLIT_H

#include <string>

typedef struct ALGraph ALGraph;
typedef struct VNode VNode;

namespace dacppTranslator {

class Split {

private:
    std::string id;
    int dimIdx;
    int splitNumber;

public:
    std::string type;
    Split(Split *parent);
    virtual ~Split() {}
    Split(Split *parent, std::string id, int dimIdx);

    void setId(std::string id);
    std::string getId();

    void setDimIdx(int dimIdx);
    int getDimIdx();

    void setSplitNumber(int splitNumber);
    int getSplitNumber();

    VNode *v;
    Split *parent;

};

class IndexSplit : public Split {

private:
    int splitNumber;

public:
    IndexSplit(Split *parent);
    IndexSplit(Split *parent, std::string id, int dimIdx, int splitNumber);

    void setSplitNumber(int splitNumber);
    int getSplitNumber();

    std::string toString(){
        return "id: " + getId() + "\n" +
               "dimIdx: " + std::to_string(getDimIdx()) + "\n" +
               "splitNumber: " + std::to_string(splitNumber) + "\n";
    }
};

class RegularSplit : public Split {

private:
    int splitSize;
    int splitStride;
    int splitNumber;

public:
    RegularSplit(Split *parent);
    RegularSplit(Split *parent, std::string id, int dimIdx, int splitSize, int splitStride, int splitNumber);

    void setSplitSize(int splitSize);
    int getSplitSize();

    void setSplitStride(int splitStride);
    int getSplitStride();

    void setSplitNumber(int splitNumber);
    int getSplitNumber();

    std::string toString(){
        return "id: " + getId() + "\n" +
               "dimIdx: " + std::to_string(getDimIdx()) + "\n" +
               "splitSize: " + std::to_string(splitSize) + "\n" +
               "splitStride: " + std::to_string(splitStride) + "\n" +
               "splitNumber: " + std::to_string(splitNumber) + "\n";
    }
};

}

#endif