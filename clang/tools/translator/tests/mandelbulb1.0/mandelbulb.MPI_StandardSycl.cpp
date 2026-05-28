#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef MANDELBULB_NX
#define MANDELBULB_NX 1024
#endif

#ifndef MANDELBULB_NY
#define MANDELBULB_NY 1024
#endif

#ifndef MANDELBULB_SAMPLES
#define MANDELBULB_SAMPLES 2
#endif

#ifndef MANDELBULB_MAX_RAY_STEPS
#define MANDELBULB_MAX_RAY_STEPS 640
#endif

#ifndef MANDELBULB_FRACTAL_ITER
#define MANDELBULB_FRACTAL_ITER 100
#endif

#ifndef MANDELBULB_WORK_REPEATS
#define MANDELBULB_WORK_REPEATS 1500
#endif

class MandelbulbRowBlockKernel;

namespace {

constexpr int NX = MANDELBULB_NX;
constexpr int NY = MANDELBULB_NY;
constexpr int SAMPLES = MANDELBULB_SAMPLES;
constexpr int MAX_RAY_STEPS = MANDELBULB_MAX_RAY_STEPS;
constexpr int FRACTAL_ITER = MANDELBULB_FRACTAL_ITER;
constexpr int WORK_REPEATS = MANDELBULB_WORK_REPEATS;

constexpr double FOV_SCALE = 1.15;
constexpr double CAMERA_Z = -4.0;
constexpr double MAX_DISTANCE = 8.0;
constexpr double HIT_EPSILON = 0.0008;
constexpr double NORMAL_EPSILON = 0.0015;
constexpr double POWER = 8.0;
constexpr double BAILOUT = 4.0;

int rows_for_rank(int rank, int size) {
    const int base = NY / size;
    const int rem = NY % size;
    return base + (rank < rem ? 1 : 0);
}

int row_begin_for_rank(int rank, int size) {
    const int base = NY / size;
    const int rem = NY % size;
    return rank * base + std::min(rank, rem);
}

double mandelbulb_de(double px, double py, double pz) {
    double zx = px;
    double zy = py;
    double zz = pz;
    double dr = 1.0;
    double r = 0.0;

    for (int i = 0; i < FRACTAL_ITER; i++) {
        r = sycl_compat::sqrt(zx * zx + zy * zy + zz * zz);
        if (r > BAILOUT) {
            break;
        }

        double safe_r = r;
        if (safe_r < 1.0e-12) {
            safe_r = 1.0e-12;
        }

        double ratio = zz / safe_r;
        if (ratio > 1.0) {
            ratio = 1.0;
        }
        if (ratio < -1.0) {
            ratio = -1.0;
        }

        double theta = sycl_compat::acos(ratio);
        double phi = sycl_compat::atan2(zy, zx);
        double zr = sycl_compat::pow(safe_r, POWER);

        dr = sycl_compat::pow(safe_r, POWER - 1.0) * POWER * dr + 1.0;
        theta = theta * POWER;
        phi = phi * POWER;

        const double sin_theta = sycl_compat::sin(theta);
        zx = zr * sin_theta * sycl_compat::cos(phi) + px;
        zy = zr * sin_theta * sycl_compat::sin(phi) + py;
        zz = zr * sycl_compat::cos(theta) + pz;
    }

    double safe_r = r;
    if (safe_r < 1.0e-12) {
        safe_r = 1.0e-12;
    }

    double dist = 0.5 * sycl_compat::log(safe_r) * safe_r / dr;
    if (dist < 1.0e-6) {
        dist = 1.0e-6;
    }
    return dist;
}

} // namespace

int main(int argc, char** argv) {
    const auto e2e_t0 = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NY) {
        if (rank == 0) {
            std::cerr << "MPI size must be <= MANDELBULB_NY\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int local_rows = rows_for_rank(rank, size);
    const int local_row_begin = row_begin_for_rank(rank, size);
    const int local_count = local_rows * NX;
    constexpr int total_count = NX * NY;

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[static_cast<std::size_t>(r)] = rows_for_rank(r, size) * NX;
        displs[static_cast<std::size_t>(r)] = offset;
        offset += counts[static_cast<std::size_t>(r)];
    }

    std::vector<int> pixels;
    if (rank == 0) {
        pixels.resize(static_cast<std::size_t>(total_count));
        for (int i = 0; i < NY; ++i) {
            for (int j = 0; j < NX; ++j) {
                pixels[static_cast<std::size_t>(i) * NX + j] = i * NX + j;
            }
        }
    }

    std::vector<int> local_pixels(static_cast<std::size_t>(local_count), 0);
    std::vector<double> local_image(static_cast<std::size_t>(local_count), 0.0);

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    MPI_Scatterv(rank == 0 ? pixels.data() : nullptr,
                 rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_INT,
                 local_pixels.data(),
                 local_count,
                 MPI_INT,
                 0,
                 MPI_COMM_WORLD);

    if (local_count > 0) {
        sycl_compat::buffer<int, 1> pixels_buf(
            local_pixels.data(), sycl_compat::range<1>(local_pixels.size()));
        sycl_compat::buffer<double, 1> image_buf(
            local_image.data(), sycl_compat::range<1>(local_image.size()));

        q.submit([&](sycl_compat::handler& h) {
            auto pixels_acc =
                pixels_buf.get_access<sycl_compat::access::mode::read>(h);
            auto image_acc =
                image_buf.get_access<sycl_compat::access::mode::write>(h);

            h.parallel_for<MandelbulbRowBlockKernel>(
                sycl_compat::range<1>(static_cast<std::size_t>(local_count)),
                [=](sycl_compat::id<1> idx) {
                    const int local_idx = static_cast<int>(idx[0]);
                    const int id = pixels_acc[idx];
                    const int pixel_x = id % NX;
                    const int pixel_y = id / NX;
                    double result = 0.0;

                    for (int repeat = 0; repeat < WORK_REPEATS; repeat++) {
                        for (int sample = 0; sample < SAMPLES; sample++) {
                            const double jitter_x =
                                ((sample * 13 + repeat * 5) % 17) / 17.0 - 0.5;
                            const double jitter_y =
                                ((sample * 7 + repeat * 3) % 19) / 19.0 - 0.5;

                            const double fx =
                                (pixel_x + 0.5 + jitter_x * 0.35) / NX;
                            const double fy =
                                (pixel_y + 0.5 + jitter_y * 0.35) / NY;
                            const double aspect =
                                static_cast<double>(NX) / static_cast<double>(NY);

                            double dir_x =
                                (2.0 * fx - 1.0) * aspect * FOV_SCALE;
                            double dir_y = (1.0 - 2.0 * fy) * FOV_SCALE;
                            double dir_z = 1.35;
                            const double dir_len =
                                sycl_compat::sqrt(dir_x * dir_x +
                                                  dir_y * dir_y +
                                                  dir_z * dir_z);
                            dir_x = dir_x / dir_len;
                            dir_y = dir_y / dir_len;
                            dir_z = dir_z / dir_len;

                            double total_dist = 0.0;
                            double hit_x = 0.0;
                            double hit_y = 0.0;
                            double hit_z = 0.0;
                            int hit = 0;
                            int final_step = MAX_RAY_STEPS;

                            for (int step = 0; step < MAX_RAY_STEPS; step++) {
                                const double px = dir_x * total_dist;
                                const double py = dir_y * total_dist;
                                const double pz =
                                    CAMERA_Z + dir_z * total_dist;

                                const double dist = mandelbulb_de(px, py, pz);
                                if (dist < HIT_EPSILON) {
                                    hit = 1;
                                    hit_x = px;
                                    hit_y = py;
                                    hit_z = pz;
                                    final_step = step;
                                    break;
                                }

                                total_dist = total_dist + dist * 0.86;
                                if (total_dist > MAX_DISTANCE) {
                                    final_step = step;
                                    break;
                                }
                            }

                            double value = 0.0;
                            if (hit == 1) {
                                double nx =
                                    mandelbulb_de(hit_x + NORMAL_EPSILON,
                                                  hit_y,
                                                  hit_z) -
                                    mandelbulb_de(hit_x - NORMAL_EPSILON,
                                                  hit_y,
                                                  hit_z);
                                double ny =
                                    mandelbulb_de(hit_x,
                                                  hit_y + NORMAL_EPSILON,
                                                  hit_z) -
                                    mandelbulb_de(hit_x,
                                                  hit_y - NORMAL_EPSILON,
                                                  hit_z);
                                double nz =
                                    mandelbulb_de(hit_x,
                                                  hit_y,
                                                  hit_z + NORMAL_EPSILON) -
                                    mandelbulb_de(hit_x,
                                                  hit_y,
                                                  hit_z - NORMAL_EPSILON);
                                const double n_len =
                                    sycl_compat::sqrt(nx * nx + ny * ny +
                                                      nz * nz);
                                if (n_len > 1.0e-12) {
                                    nx = nx / n_len;
                                    ny = ny / n_len;
                                    nz = nz / n_len;
                                }

                                double lx = -0.55;
                                double ly = 0.70;
                                double lz = -0.45;
                                const double l_len =
                                    sycl_compat::sqrt(lx * lx + ly * ly +
                                                      lz * lz);
                                lx = lx / l_len;
                                ly = ly / l_len;
                                lz = lz / l_len;

                                double diffuse = nx * lx + ny * ly + nz * lz;
                                if (diffuse < 0.0) {
                                    diffuse = 0.0;
                                }

                                const double occlusion =
                                    1.0 -
                                    static_cast<double>(final_step) /
                                        static_cast<double>(MAX_RAY_STEPS);
                                value =
                                    0.20 + 0.65 * diffuse + 0.15 * occlusion;
                            } else {
                                value = 0.025 *
                                        sycl_compat::exp(-0.05 * total_dist);
                            }

                            result += value;
                        }
                    }

                    image_acc[local_idx] =
                        result / static_cast<double>(SAMPLES * WORK_REPEATS);
                });
        });
        q.wait();
    }

    constexpr int center_row = NY / 2;
    constexpr int center_col = NX / 2;
    const int center_owner = [&]() {
        for (int r = 0; r < size; ++r) {
            const int begin = row_begin_for_rank(r, size);
            const int end = begin + rows_for_rank(r, size);
            if (center_row >= begin && center_row < end) {
                return r;
            }
        }
        return 0;
    }();

    double center_value = 0.0;
    if (rank == center_owner) {
        const int local_row = center_row - local_row_begin;
        center_value = local_image[static_cast<std::size_t>(local_row) * NX +
                                   center_col];
    }
    if (center_owner != 0) {
        if (rank == center_owner) {
            MPI_Send(&center_value, 1, MPI_DOUBLE, 0, 4701, MPI_COMM_WORLD);
        } else if (rank == 0) {
            MPI_Recv(&center_value,
                     1,
                     MPI_DOUBLE,
                     center_owner,
                     4701,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds = std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][mandelbulb][rowblock] seconds="
                  << max_seconds << std::endl;
        std::cout << center_value << std::endl;
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
        std::cerr << "[MPI_StandardSycl][mandelbulb][rowblock] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
