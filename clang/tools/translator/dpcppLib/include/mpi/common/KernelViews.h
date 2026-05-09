#ifndef DACPP_MPI_KERNEL_VIEWS_H
#define DACPP_MPI_KERNEL_VIEWS_H

#include <cstdint>

namespace dacpp {
namespace mpi {

template <typename T>
struct View1D {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;

    decltype(auto) operator[](int idx) const {
        return data[slots[offset + idx]];
    }
};

template <typename T>
struct View2DRow {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;

    decltype(auto) operator[](int idx) const {
        return data[slots[offset + idx]];
    }
};

template <typename T>
struct View2D {
    T* data = nullptr;
    const int32_t* slots = nullptr;
    int offset = 0;
    int cols = 0;

    View2DRow<T> operator[](int row) const {
        return View2DRow<T>{data, slots, offset + row * cols};
    }
};

template <typename T>
struct ContiguousView1D {
    T* data = nullptr;
    int offset = 0;

    decltype(auto) operator[](int idx) const {
        return data[offset + idx];
    }
};

template <typename T>
struct ResidentHaloView1D {
    T* data = nullptr;
    int offset = 0;

    decltype(auto) operator[](int idx) const {
        return data[offset + idx];
    }
};

template <typename T>
struct ResidentHaloView2DRow {
    T* data = nullptr;
    int offset = 0;

    decltype(auto) operator[](int idx) const {
        return data[offset + idx];
    }
};

template <typename T>
struct ResidentHaloView2D {
    T* data = nullptr;
    int offset = 0;
    int cols = 0;

    ResidentHaloView2DRow<T> operator[](int row) const {
        return ResidentHaloView2DRow<T>{data, offset + row * cols};
    }
};

template <typename T>
struct ContiguousView2DRow {
    T* data = nullptr;
    int offset = 0;

    decltype(auto) operator[](int idx) const {
        return data[offset + idx];
    }
};

template <typename T>
struct ContiguousView2D {
    T* data = nullptr;
    int offset = 0;
    int cols = 0;

    ContiguousView2DRow<T> operator[](int row) const {
        return ContiguousView2DRow<T>{data, offset + row * cols};
    }
};

}  // namespace mpi
}  // namespace dacpp

#endif
