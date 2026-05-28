#ifndef DACPP_MPI_KERNEL_VIEWS_H
#define DACPP_MPI_KERNEL_VIEWS_H

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#define DACPP_MPI_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define DACPP_MPI_ALWAYS_INLINE inline
#endif

namespace dacpp {
namespace mpi {

template <typename T>
struct View1D {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;

    DACPP_MPI_ALWAYS_INLINE decltype(auto) operator[](int idx) const {
        return data[slots[offset + idx]];
    }
};

template <typename T>
struct View2DRow {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;

    DACPP_MPI_ALWAYS_INLINE decltype(auto) operator[](int idx) const {
        return data[slots[offset + idx]];
    }
};

template <typename T>
struct View2D {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;
    int cols = 0;

    DACPP_MPI_ALWAYS_INLINE View2DRow<T> operator[](int row) const {
        return View2DRow<T>{data, slots, offset + row * cols};
    }
};

template <typename T>
struct ContiguousView1D {
    T* data = nullptr;
    int offset = 0;

    DACPP_MPI_ALWAYS_INLINE decltype(auto) operator[](int idx) const {
        return data[offset + idx];
    }
};

template <typename T>
struct ResidentHaloView1D {
    T* data = nullptr;
    int offset = 0;

    DACPP_MPI_ALWAYS_INLINE decltype(auto) operator[](int idx) const {
        return data[offset + idx];
    }
};

template <typename T>
struct ResidentHaloInteriorView1D {
    T* data = nullptr;
    int fullCols = 0;
    int interiorCols = 0;
    int rowOffset = 0;
    int colOffset = 0;

    DACPP_MPI_ALWAYS_INLINE decltype(auto) operator[](int idx) const {
        const int row = idx / interiorCols;
        const int col = idx % interiorCols;
        return data[(row + rowOffset) * fullCols + (col + colOffset)];
    }
};

template <typename T>
struct ResidentHaloView2DRow {
    T* data = nullptr;
    int offset = 0;

    DACPP_MPI_ALWAYS_INLINE decltype(auto) operator[](int idx) const {
        return data[offset + idx];
    }
};

template <typename T>
struct ResidentHaloView2D {
    T* data = nullptr;
    int offset = 0;
    int cols = 0;

    DACPP_MPI_ALWAYS_INLINE ResidentHaloView2DRow<T> operator[](int row) const {
        return ResidentHaloView2DRow<T>{data, offset + row * cols};
    }
};

template <typename T>
struct ContiguousView2DRow {
    T* data = nullptr;
    int offset = 0;

    DACPP_MPI_ALWAYS_INLINE decltype(auto) operator[](int idx) const {
        return data[offset + idx];
    }
};

template <typename T>
struct ContiguousView2D {
    T* data = nullptr;
    int offset = 0;
    int cols = 0;

    DACPP_MPI_ALWAYS_INLINE ContiguousView2DRow<T> operator[](int row) const {
        return ContiguousView2DRow<T>{data, offset + row * cols};
    }
};

}  // namespace mpi
}  // namespace dacpp

#undef DACPP_MPI_ALWAYS_INLINE

#endif
