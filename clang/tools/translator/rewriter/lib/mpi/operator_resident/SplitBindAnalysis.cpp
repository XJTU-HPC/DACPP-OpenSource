#include <algorithm>
#include <vector>

#include "DacppStructure.h"
#include "ShellPartitionAnalysis_Internal.h"

namespace dacppTranslator {
namespace mpi_rewriter {
namespace operator_resident {

const char* shellDimKindName(ShellDimKind kind) {
    switch (kind) {
    case ShellDimKind::Index:
        return "Index";
    case ShellDimKind::Void:
        return "Void";
    case ShellDimKind::Split:
        return "Split";
    }
    return "Unknown";
}

int64_t shapeValueFor(Shell* shell, int paramIdx, int dimIdx) {
    if (!shell || paramIdx < 0 || paramIdx >= shell->getNumParams() ||
        dimIdx < 0) {
        return -1;
    }
    Param* param = shell->getParam(paramIdx);
    if (!param || dimIdx >= param->getDim()) {
        return -1;
    }
    return param->getShape(dimIdx);
}

bool splitIsVoid(Split* split) {
    return !split || split->getId() == "void";
}

bool splitIsIndex(Split* split) {
    return split && split->type == "IndexSplit";
}

bool splitIsRegular(Split* split) {
    return split && split->type == "RegularSplit";
}

bool sameOrder(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    return lhs == rhs;
}

std::vector<int> uniquePreserveOrder(const std::vector<int>& values) {
    std::vector<int> result;
    for (int value : values) {
        if (std::find(result.begin(), result.end(), value) == result.end()) {
            result.push_back(value);
        }
    }
    return result;
}

} // namespace operator_resident
} // namespace mpi_rewriter
} // namespace dacppTranslator
