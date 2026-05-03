#ifndef DACPP_MPI_MPI_TYPES_H
#define DACPP_MPI_MPI_TYPES_H

#include <complex>
#include <cstddef>
#include <type_traits>

#include <mpi.h>

namespace dacpp {
namespace mpi {

template <typename T>
inline MPI_Datatype mpi_datatype_for_value() {
    if constexpr (std::is_same_v<T, float>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, long double>) {
        return MPI_LONG_DOUBLE;
    } else if constexpr (std::is_same_v<T, int>) {
        return MPI_INT;
    } else if constexpr (std::is_same_v<T, short> ||
                         std::is_same_v<T, short int>) {
        return MPI_SHORT;
    } else if constexpr (std::is_same_v<T, long>) {
        return MPI_LONG;
    } else if constexpr (std::is_same_v<T, long long>) {
        return MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<T, char>) {
        return MPI_CHAR;
    } else if constexpr (std::is_same_v<T, signed char>) {
        return MPI_SIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, unsigned>) {
        return MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T, unsigned short>) {
        return MPI_UNSIGNED_SHORT;
    } else if constexpr (std::is_same_v<T, unsigned long>) {
        return MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        return MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        return MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, bool>) {
        return MPI_CXX_BOOL;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return MPI_C_FLOAT_COMPLEX;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return MPI_C_DOUBLE_COMPLEX;
    } else if constexpr (std::is_same_v<T, std::complex<long double>>) {
        return MPI_C_LONG_DOUBLE_COMPLEX;
    }
    return MPI_BYTE;
}

template <typename T>
inline bool uses_byte_transport_for_value() {
    return mpi_datatype_for_value<T>() == MPI_BYTE;
}

template <typename T>
inline int mpi_payload_count_for_values(std::size_t count) {
    if (uses_byte_transport_for_value<T>()) {
        return static_cast<int>(count * sizeof(T));
    }
    return static_cast<int>(count);
}

}  // namespace mpi
}  // namespace dacpp

#endif
