#include <string>

#include "Rewriter_MPI_Common.h"

namespace dacppTranslator {
namespace mpi_rewriter {

std::string mpiDatatypeFor(const std::string& type) {
    if (type == "float") return "MPI_FLOAT";
    if (type == "double") return "MPI_DOUBLE";
    if (type == "long double") return "MPI_LONG_DOUBLE";
    if (type == "int") return "MPI_INT";
    if (type == "short" || type == "short int") return "MPI_SHORT";
    if (type == "long") return "MPI_LONG";
    if (type == "long long") return "MPI_LONG_LONG";
    if (type == "char") return "MPI_CHAR";
    if (type == "signed char") return "MPI_SIGNED_CHAR";
    if (type == "unsigned") return "MPI_UNSIGNED";
    if (type == "unsigned int") return "MPI_UNSIGNED";
    if (type == "unsigned short" || type == "unsigned short int") return "MPI_UNSIGNED_SHORT";
    if (type == "unsigned long") return "MPI_UNSIGNED_LONG";
    if (type == "unsigned long long") return "MPI_UNSIGNED_LONG_LONG";
    if (type == "unsigned char") return "MPI_UNSIGNED_CHAR";
    if (type == "bool") return "MPI_CXX_BOOL";
    if (type == "std::complex<float>" || type == "complex<float>") {
        return "MPI_C_FLOAT_COMPLEX";
    }
    if (type == "std::complex<double>" || type == "complex<double>") {
        return "MPI_C_DOUBLE_COMPLEX";
    }
    if (type == "std::complex<long double>" || type == "complex<long double>") {
        return "MPI_C_LONG_DOUBLE_COMPLEX";
    }
    return "MPI_BYTE";
}

bool usesByteTransport(const std::string& type) {
    return mpiDatatypeFor(type) == "MPI_BYTE";
}

std::string mpiPayloadCountExpr(const std::string& elemCountExpr,
                                const std::string& type) {
    if (usesByteTransport(type)) {
        return "static_cast<int>((" + elemCountExpr + ") * sizeof(" + type + "))";
    }
    return elemCountExpr;
}

std::string toPlannerMode(IOTYPE mode) {
    switch (mode) {
    case IOTYPE::READ:
        return "dacpp::mpi::AccessMode::Read";
    case IOTYPE::WRITE:
        return "dacpp::mpi::AccessMode::Write";
    case IOTYPE::READ_WRITE:
        return "dacpp::mpi::AccessMode::ReadWrite";
    }
    return "dacpp::mpi::AccessMode::Read";
}

std::string toAccessorMode(IOTYPE mode) {
    return mode == IOTYPE::READ ? "sycl::access::mode::read"
                                : "sycl::access::mode::read_write";
}

int inferViewRank(ShellParam* shellParam, Param* calcParam) {
    const std::string calcType = calcParam->getType();
    if (calcType.find('*') != std::string::npos) {
        return 1;
    }
    if (calcType.find("Vector<") != std::string::npos) {
        return 1;
    }
    if (calcType.find("Matrix<") != std::string::npos) {
        return 2;
    }

    const int calcDim = calcParam->getDimension();
    if (calcDim > 0) {
        return calcDim;
    }

    const int shellDim = shellParam->getDimension();
    return shellDim > 0 ? shellDim : 1;
}

std::string toViewType(ShellParam* shellParam, Param* calcParam, IOTYPE mode) {
    const std::string qualifiedType = viewElementType(calcParam, mode);
    const int dim = inferViewRank(shellParam, calcParam);

    if (dim <= 1) {
        return "dacpp::mpi::View1D<" + qualifiedType + ">";
    }
    return "dacpp::mpi::View2D<" + qualifiedType + ">";
}

std::string viewElementType(Param* calcParam, IOTYPE mode) {
    const std::string baseType = calcParam->getBasicType();
    const bool isReadOnly = mode == IOTYPE::READ;
    return isReadOnly ? ("const " + baseType) : baseType;
}

}  // namespace mpi_rewriter
}  // namespace dacppTranslator
