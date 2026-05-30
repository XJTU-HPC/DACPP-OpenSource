#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef WAVE_NX
#define WAVE_NX 8192
#endif

#ifndef WAVE_NY
#define WAVE_NY 8192
#endif

#ifndef WAVE_TIME_STEPS
#define WAVE_TIME_STEPS 10000
#endif

class WaveEquationStandardHaloKernel;

namespace {

constexpr int NX = WAVE_NX;
constexpr int NY = WAVE_NY;
constexpr int TIME_STEPS = WAVE_TIME_STEPS;
constexpr double Lx = 10.0;
constexpr double Ly = 10.0;
constexpr double c = 1.0;
constexpr double medium_variation = 0.08;
constexpr double damping_strength = 0.03;
constexpr double sponge_strength = 0.18;

int rows_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return base + (rank < rem ? 1 : 0);
}

int row_begin_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return rank * base + std::min(rank, rem);
}

template <typename PrevAccessor, typename CurrAccessor, typename NextAccessor>
struct WaveEquationStandardHaloFunctor {
    PrevAccessor prev;
    CurrAccessor curr;
    NextAccessor next;
    int global_begin;
    int nx;
    int ny;
    int sponge_width;
    double dx;
    double dy;
    double dt;
    double lx;
    double ly;
    double wave_speed;
    double medium_variation;
    double damping_strength;
    double sponge_strength;

    void operator()(sycl_compat::id<1> idx) const {
        const std::size_t item = static_cast<std::size_t>(idx[0]);
        const int lr = static_cast<int>(item / ny) + 1;
        const int j = static_cast<int>(item % ny);
        const int gi = global_begin + lr - 1;
        const std::size_t pos = static_cast<std::size_t>(lr) * ny + j;

        if (gi == 0 || gi == nx - 1 || j == 0 || j == ny - 1) {
            next[pos] = 0.0;
            return;
        }

        const double center = curr[pos];
        const double u_xx = (curr[pos + ny] - 2.0 * center + curr[pos - ny]) /
                            (dx * dx);
        const double u_yy = (curr[pos + 1] - 2.0 * center + curr[pos - 1]) /
                            (dy * dy);
        const double x = gi * dx;
        const double y = j * dy;
        const double rx = (x - lx * 0.5) / lx;
        const double ry = (y - ly * 0.5) / ly;
        const double radius2 = rx * rx + ry * ry;
        const double local_c =
            wave_speed * (1.0 + medium_variation * radius2);
        const double damping = damping_strength * (1.0 + radius2);
        const double damp_dt = damping * dt;
        double next_value = (2.0 - damp_dt) * center -
                            (1.0 - damp_dt) * prev[pos] +
                            local_c * local_c * dt * dt * (u_xx + u_yy);

        int dist = gi;
        dist = std::min(dist, nx - 1 - gi);
        dist = std::min(dist, j);
        dist = std::min(dist, ny - 1 - j);
        if (dist < sponge_width) {
            const double depth = static_cast<double>(sponge_width - dist) /
                                 static_cast<double>(sponge_width);
            const double attenuation = sponge_strength * depth * depth;
            next_value = next_value / (1.0 + attenuation * dt);
        }
        next[pos] = next_value;
    }
};

}

int main(int argc, char** argv) {
    const auto e2e_t0 = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NX) {
        if (rank == 0) {
            std::cerr << "MPI size must be <= WAVE_NX\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int local_rows = rows_for_rank(rank, size);
    const int global_begin = row_begin_for_rank(rank, size);
    const std::size_t pitch = static_cast<std::size_t>(NY);
    const std::size_t global_size = static_cast<std::size_t>(NX) * pitch;
    const std::size_t local_with_halo =
        static_cast<std::size_t>(local_rows + 2) * pitch;

    const double dx = Lx / (NX - 1);
    const double dy = Ly / (NY - 1);
    const double dt = 0.5 * std::fmin(dx, dy) / c;
    const int sponge_width = std::max(1, std::min(NX, NY) / 16);

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[static_cast<std::size_t>(r)] = rows_for_rank(r, size) * NY;
        displs[static_cast<std::size_t>(r)] = offset;
        offset += counts[static_cast<std::size_t>(r)];
    }

    std::vector<double> h_prev(local_with_halo, 0.0);
    std::vector<double> h_curr(local_with_halo, 0.0);
    std::vector<double> h_next(local_with_halo, 0.0);

    const double sigma = 0.5;
    std::vector<double> u_prev;
    std::vector<double> u_curr;
    std::vector<double> u_next;
    if (rank == 0) {
        u_prev.assign(global_size, 0.0);
        u_curr.assign(global_size, 0.0);
        u_next.assign(global_size, 0.0);

        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                const double x = i * dx;
                const double y = j * dy;
                u_prev[static_cast<std::size_t>(i) * NY + j] =
                    std::exp(-((x - Lx / 2) * (x - Lx / 2) +
                               (y - Ly / 2) * (y - Ly / 2)) /
                             (2 * sigma * sigma));
            }
        }
    }
    MPI_Scatterv(rank == 0 ? u_prev.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_DOUBLE,
                 h_prev.data() + pitch,
                 counts[static_cast<std::size_t>(rank)],
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    const int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int down = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < TIME_STEPS; ++step) {
        MPI_Request reqs[4];
        int req_count = 0;
        if (up != MPI_PROC_NULL) {
            MPI_Irecv(h_curr.data(),
                      NY,
                      MPI_DOUBLE,
                      up,
                      1,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
            MPI_Isend(h_curr.data() + pitch,
                      NY,
                      MPI_DOUBLE,
                      up,
                      0,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Irecv(h_curr.data() +
                          static_cast<std::size_t>(local_rows + 1) * pitch,
                      NY,
                      MPI_DOUBLE,
                      down,
                      0,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
            MPI_Isend(h_curr.data() +
                          static_cast<std::size_t>(local_rows) * pitch,
                      NY,
                      MPI_DOUBLE,
                      down,
                      1,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }
        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        {
            sycl_compat::buffer<double, 1> prev_buf(
                h_prev.data(), sycl_compat::range<1>(h_prev.size()));
            sycl_compat::buffer<double, 1> curr_buf(
                h_curr.data(), sycl_compat::range<1>(h_curr.size()));
            sycl_compat::buffer<double, 1> next_buf(
                h_next.data(), sycl_compat::range<1>(h_next.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto prev =
                    prev_buf.get_access<sycl_compat::access::mode::read>(h);
                auto curr =
                    curr_buf.get_access<sycl_compat::access::mode::read>(h);
                auto next =
                    next_buf
                        .get_access<sycl_compat::access::mode::read_write>(h);
                WaveEquationStandardHaloFunctor<decltype(prev),
                                                decltype(curr),
                                                decltype(next)>
                    wave_kernel{prev,
                                curr,
                                next,
                                global_begin,
                                NX,
                                NY,
                                sponge_width,
                                dx,
                                dy,
                                dt,
                                Lx,
                                Ly,
                                c,
                                medium_variation,
                                damping_strength,
                                sponge_strength};

                h.parallel_for<WaveEquationStandardHaloKernel>(
                    sycl_compat::range<1>(static_cast<std::size_t>(local_rows) *
                                          static_cast<std::size_t>(NY)),
                    wave_kernel);
            });
            q.wait();
        }

        std::swap(h_prev, h_curr);
        std::swap(h_curr, h_next);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> local_out(static_cast<std::size_t>(local_rows) * NY);
    for (int lr = 0; lr < local_rows; ++lr) {
        std::copy(h_curr.begin() +
                      static_cast<std::ptrdiff_t>((lr + 1) * NY),
                  h_curr.begin() +
                      static_cast<std::ptrdiff_t>((lr + 2) * NY),
                  local_out.begin() + static_cast<std::ptrdiff_t>(lr * NY));
    }

    std::vector<double> global_out;
    if (rank == 0) {
        global_out.resize(static_cast<std::size_t>(NX) * NY);
    }
    MPI_Gatherv(local_out.data(),
                static_cast<int>(local_out.size()),
                MPI_DOUBLE,
                rank == 0 ? global_out.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        std::copy(global_out.begin(), global_out.end(), u_curr.begin());
        for (int i = 0; i < NX; ++i) {
            const std::size_t row_base = static_cast<std::size_t>(i) * NY;
            u_curr[row_base] = 0.0;
            u_curr[row_base + static_cast<std::size_t>(NY - 1)] = 0.0;
        }
        const std::size_t last_row_base = static_cast<std::size_t>(NX - 1) * NY;
        for (int j = 0; j < NY; ++j) {
            u_curr[static_cast<std::size_t>(j)] = 0.0;
            u_curr[last_row_base + static_cast<std::size_t>(j)] = 0.0;
        }
        (void)u_next;
    }

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][wave] seconds=" << max_seconds
                  << std::endl;
    }

    const auto e2e_t1 = std::chrono::steady_clock::now();
    const double e2e_local_seconds =
        std::chrono::duration<double>(e2e_t1 - e2e_t0).count();
    double e2e_seconds = 0.0;
    MPI_Reduce(&e2e_local_seconds,
               &e2e_seconds,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][wave] e2e_seconds=" << e2e_seconds
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
