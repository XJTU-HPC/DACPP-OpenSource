#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

template <typename NextAccessor>
struct StencilSunwayBoundaryKernel3 {
    NextAccessor next_acc;
    bool has_top_boundary;
    bool has_bottom_boundary;
    std::size_t pitch;
    std::size_t bottom_dst_base;
    std::size_t bottom_src_base;

    StencilSunwayBoundaryKernel3(NextAccessor acc,
                                 bool top_boundary,
                                 bool bottom_boundary,
                                 std::size_t pitch_value,
                                 std::size_t dst_base,
                                 std::size_t src_base)
        : next_acc(acc),
          has_top_boundary(top_boundary),
          has_bottom_boundary(bottom_boundary),
          pitch(pitch_value),
          bottom_dst_base(dst_base),
          bottom_src_base(src_base) {}

    void operator()(sycl_compat::id<1> idx) const {
        const std::size_t j = idx[0];
        if (has_top_boundary) {
            next_acc[pitch + j] = next_acc[2 * pitch + j];
        }
        if (has_bottom_boundary) {
            next_acc[bottom_dst_base + j] = next_acc[bottom_src_base + j];
        }
    }
};

#ifndef STENCIL_NX
#define STENCIL_NX 8192
#endif

#ifndef STENCIL_NY
#define STENCIL_NY 8192
#endif

#ifndef STENCIL_TIME_STEPS
#define STENCIL_TIME_STEPS 5000
#endif

constexpr int NX = STENCIL_NX;
constexpr int NY = STENCIL_NY;
constexpr int TIME_STEPS = STENCIL_TIME_STEPS;
constexpr double Lx = 10.0;
constexpr double Ly = 10.0;
constexpr double alpha = 0.01;

static int rows_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return base + (rank < rem ? 1 : 0);
}

static int row_begin_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return rank * base + std::min(rank, rem);
}








struct HostMatrix {
    std::vector<int> shape;
    std::vector<double> data;

    HostMatrix() = default;
    explicit HostMatrix(std::vector<int> s) : shape(std::move(s)) {
        std::size_t total = 1;
        for (int dim : shape) {
            total *= static_cast<std::size_t>(dim);
        }
        data.assign(total, 0.0);
    }

    std::size_t linearIndex(int i, int j) const {
        const int coords[2] = {i, j};
        std::size_t offset = 0;
        std::size_t stride = 1;
        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
            offset += static_cast<std::size_t>(coords[dim]) * stride;
            stride *= static_cast<std::size_t>(shape[static_cast<std::size_t>(dim)]);
        }
        return offset;
    }

    double getElement(int i, int j) const { return data[linearIndex(i, j)]; }
    void setElement(int i, int j, double v) { data[linearIndex(i, j)] = v; }

    void tensor2Array(std::vector<double>& out) const {
        out.resize(data.size());
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                out[static_cast<std::size_t>(i) * shape[1] + j] = getElement(i, j);
            }
        }
    }

    void array2Tensor(const std::vector<double>& in) {
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                setElement(i, j, in[static_cast<std::size_t>(i) * shape[1] + j]);
            }
        }
    }
};

int main(int argc, char** argv) {
    const auto e2e_start = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NX) {
        if (rank == 0) {
            std::cerr << "MPI size must be <= STENCIL_NX\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int local_rows = rows_for_rank(rank, size);
    const int row_begin = row_begin_for_rank(rank, size);
    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
    const double dx = Lx / (NX - 1);
    const double dy = Ly / (NY - 1);
    const double dt_stability =
        (dx * dx * dy * dy) / (2.0 * alpha * (dx * dx + dy * dy));
    const double delta_t = 0.4 * dt_stability;
    const std::size_t pitch = static_cast<std::size_t>(NY);
    const std::size_t local_extent =
        static_cast<std::size_t>(local_rows + 2) * pitch;

    std::vector<double> local_curr(local_extent, 0.0);
    std::vector<double> local_next(local_extent, 0.0);





    std::vector<int> counts(static_cast<std::size_t>(size));
    std::vector<int> displs(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        counts[static_cast<std::size_t>(r)] = rows_for_rank(r, size) * NY;
        displs[static_cast<std::size_t>(r)] = row_begin_for_rank(r, size) * NY;
    }





    const double sigma = 1.0;
    HostMatrix global_matrix;
    std::vector<double> transport;
    if (rank == 0) {
        global_matrix = HostMatrix({NX, NY});
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                const double x = i * dx;
                const double y = j * dy;
                global_matrix.setElement(
                    i, j,
                    std::exp(-((x - Lx / 2.0) * (x - Lx / 2.0) +
                               (y - Ly / 2.0) * (y - Ly / 2.0)) /
                             (2.0 * sigma * sigma)));
            }
        }
        global_matrix.tensor2Array(transport);
    }
    MPI_Scatterv(rank == 0 ? transport.data() : nullptr,
                 counts.data(), displs.data(), MPI_DOUBLE,
                 local_curr.data() + pitch, local_rows * NY, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    sycl_compat::queue q(sycl_compat::default_selector_v);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int step = 0; step < TIME_STEPS; ++step) {
        MPI_Sendrecv(local_curr.data() + pitch, NY, MPI_DOUBLE, prev, 10,
                     local_curr.data(), NY, MPI_DOUBLE, prev, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(local_curr.data() +
                         static_cast<std::size_t>(local_rows) * pitch,
                     NY, MPI_DOUBLE, next, 11,
                     local_curr.data() +
                         static_cast<std::size_t>(local_rows + 1) * pitch,
                     NY, MPI_DOUBLE, next, 10, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        {
            sycl_compat::buffer<double, 1> curr_buf(
                local_curr.data(), sycl_compat::range<1>(local_curr.size()));
            sycl_compat::buffer<double, 1> next_buf(
                local_next.data(), sycl_compat::range<1>(local_next.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto curr =
                    curr_buf.get_access<sycl_compat::access::mode::read>(h);
                auto next_acc =
                    next_buf.get_access<sycl_compat::access::mode::write>(h);
                h.parallel_for<class stencil_MPI_StandardSycl_test_kernel_1>(
                    sycl_compat::range<2>(static_cast<std::size_t>(local_rows),
                                          static_cast<std::size_t>(NY - 2)),
                    [=](sycl_compat::id<2> idx) {
                        const int lr = static_cast<int>(idx[0]) + 1;
                        const int j = static_cast<int>(idx[1]) + 1;
                        const int gi = row_begin + lr - 1;
                        if (gi <= 0 || gi >= NX - 1) {
                            return;
                        }
                        const std::size_t center =
                            static_cast<std::size_t>(lr) * NY + j;
                        const double u_xx =
                            (curr[center + NY] - 2.0 * curr[center] +
                             curr[center - NY]) /
                            (dx * dx);
                        const double u_yy =
                            (curr[center + 1] - 2.0 * curr[center] +
                             curr[center - 1]) /
                            (dy * dy);
                        next_acc[center] =
                            curr[center] + alpha * delta_t * (u_xx + u_yy);
                    });
            });






            q.submit([&](sycl_compat::handler& h) {
                auto curr_acc =
                    curr_buf.get_access<sycl_compat::access::mode::read_write>(h);
                auto next_acc =
                    next_buf.get_access<sycl_compat::access::mode::read>(h);
                h.parallel_for<class stencil_MPI_StandardSycl_test_kernel_copyback>(
                    sycl_compat::range<2>(static_cast<std::size_t>(local_rows),
                                          static_cast<std::size_t>(NY - 2)),
                    [=](sycl_compat::id<2> idx) {
                        const int lr = static_cast<int>(idx[0]) + 1;
                        const int j = static_cast<int>(idx[1]) + 1;
                        const int gi = row_begin + lr - 1;
                        if (gi <= 0 || gi >= NX - 1) {
                            return;
                        }
                        const std::size_t center =
                            static_cast<std::size_t>(lr) * NY + j;
                        curr_acc[center] = next_acc[center];
                    });
            });
            q.submit([&](sycl_compat::handler& h) {
                auto curr_acc =
                    curr_buf.get_access<sycl_compat::access::mode::read_write>(h);
                h.parallel_for<class stencil_MPI_StandardSycl_test_kernel_2>(
                    sycl_compat::range<1>(static_cast<std::size_t>(local_rows)),
                    [=](sycl_compat::id<1> idx) {
                        const int lr = static_cast<int>(idx[0]) + 1;
                        const int gi = row_begin + lr - 1;
                        const std::size_t row =
                            static_cast<std::size_t>(lr) * NY;
                        if (gi > 0 && gi < NX - 1) {
                            curr_acc[row] = curr_acc[row + 1];
                            curr_acc[row + NY - 1] =
                                curr_acc[row + NY - 2];
                        }
                    });
            });
            q.submit([&](sycl_compat::handler& h) {
                auto curr_acc =
                    curr_buf.get_access<sycl_compat::access::mode::read_write>(h);
                const bool kernel3_has_top_boundary = (row_begin == 0);
                const bool kernel3_has_bottom_boundary =
                    (row_begin + local_rows == NX);
                const std::size_t kernel3_bottom_dst_base =
                    static_cast<std::size_t>(local_rows) * pitch;
                const std::size_t kernel3_bottom_src_base =
                    static_cast<std::size_t>(local_rows - 1) * pitch;
                h.parallel_for<class stencil_MPI_StandardSycl_test_kernel_3>(
                    sycl_compat::range<1>(static_cast<std::size_t>(NY)),
                    StencilSunwayBoundaryKernel3<decltype(curr_acc)>(
                        curr_acc,
                        kernel3_has_top_boundary,
                        kernel3_has_bottom_boundary,
                        pitch,
                        kernel3_bottom_dst_base,
                        kernel3_bottom_src_base));
            });
            q.wait();
        }
    }






    MPI_Gatherv(local_curr.data() + pitch, local_rows * NY, MPI_DOUBLE,
                rank == 0 ? transport.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        global_matrix.array2Tensor(transport);
        for (int j = 0; j < NY; ++j) {
            global_matrix.setElement(0, j, global_matrix.getElement(1, j));
            global_matrix.setElement(NX - 1, j, global_matrix.getElement(NX - 2, j));
        }
        for (int i = 0; i < NX; ++i) {
            global_matrix.setElement(i, 0, global_matrix.getElement(i, 1));
            global_matrix.setElement(i, NY - 1, global_matrix.getElement(i, NY - 2));
        }


        double checksum = 0.0;
        for (int j = 0; j < NY; ++j) {
            checksum += global_matrix.getElement(0, j);
        }
        std::cerr << "[MPI_StandardSycl][stencil][test] row0_checksum="
                  << checksum << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto e2e_end = std::chrono::steady_clock::now();
    const double e2e_local_seconds =
        std::chrono::duration<double>(e2e_end - e2e_start).count();
    double e2e_seconds = 0.0;
    MPI_Reduce(&e2e_local_seconds,
               &e2e_seconds,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][stencil][test] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
