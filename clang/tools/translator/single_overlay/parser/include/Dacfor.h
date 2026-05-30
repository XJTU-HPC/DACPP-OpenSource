#ifndef TRANSLATOR_PARSER_DACFOR_H
#define TRANSLATOR_PARSER_DACFOR_H

#include <string>
#include <vector>
#include <memory>

#include "clang/AST/Type.h"
#include "Param.h"

namespace dacppTranslator {

    // Semantic operations extracted from DAC-for control flow.
    enum class SemanticOperation {
        None,
        Swap,
        Memset,
        Copy,
        Custom
    };

    class ParamControl {
    private:
        Param* paramA_;
        Param* paramB_;
        std::vector<std::string> shapeA;
        std::vector<std::string> shapeB;
        SemanticOperation operation_;
    public:
        ParamControl();

        void setParamA(Param* param);
        void setParamB(Param* param);
        std::vector<std::string> getShapeA() const { return shapeA; }
        std::vector<std::string> getShapeB() const { return shapeB; }
        void setShapeA(std::vector<std::string> shape) { shapeA = shape; }
        void setShapeB(std::vector<std::string> shape) { shapeB = shape; }
        void setOperation(SemanticOperation op);

        Param* getParamA();
        Param* getParamB();
        SemanticOperation getOperation() const;
    };

    // Captures the structural controls associated with a DAC-for region.
    class ControlBlock {
    private:
        std::vector<std::shared_ptr<ParamControl>> paramControls_;
        std::string loopBound_;
    public:
        ControlBlock(){
            loopBound_ = "ERROR";
        };

        void addParamControl(const std::shared_ptr<ParamControl>& control);
        void setLoopBound(std::string Bound);

        const std::vector<std::shared_ptr<ParamControl>>& getParamControls() const;
        std::string getLoopBound() const;
    };

}

#endif
