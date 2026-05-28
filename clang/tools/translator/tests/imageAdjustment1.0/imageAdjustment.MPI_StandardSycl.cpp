#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace sycl_compat = cl::sycl;

#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH 4
#endif

#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT 4
#endif

struct Pixel {
    int r;
    int g;
    int b;
};

std::ostream& operator<<(std::ostream& os, const Pixel& p) {
    os << "(" << p.r << ", " << p.g << ", " << p.b << ")";
    return os;
}

class ImageAdjustRedKernel;
class ImageAdjustBrightnessKernel;

namespace {

constexpr int width = IMAGE_WIDTH;
constexpr int height = IMAGE_HEIGHT;
constexpr int total = width * height;

int rows_for_rank(int rank, int size) {
    const int base = height / size;
    const int rem = height % size;
    return base + (rank < rem ? 1 : 0);
}

int row_begin_for_rank(int rank, int size) {
    const int base = height / size;
    const int rem = height % size;
    return rank * base + std::min(rank, rem);
}

} // namespace

int main(int argc, char** argv) {
    const auto e2e_t0 = std::chrono::steady_clock::now();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int local_rows = rows_for_rank(rank, size);
    const int row_begin = row_begin_for_rank(rank, size);
    const int local_count = local_rows * width;

    std::vector<Pixel> image(total, {100, 100, 100});
    std::vector<Pixel> local_image(local_count, {100, 100, 100});
    for (int lr = 0; lr < local_rows; ++lr) {
        const int gi = row_begin + lr;
        for (int j = 0; j < width; ++j) {
            local_image[static_cast<std::size_t>(lr) * width + j] =
                image[static_cast<std::size_t>(gi) * width + j];
        }
    }
    std::vector<Pixel> local_image2(local_count, {100, 100, 100});
    std::vector<Pixel> local_image3(local_count, {0, 0, 0});

    sycl_compat::queue q{sycl_compat::default_selector_v};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    if (local_count > 0) {
        {
            sycl_compat::buffer<Pixel, 1> in_buf(
                local_image.data(), sycl_compat::range<1>(local_image.size()));
            sycl_compat::buffer<Pixel, 1> out_buf(
                local_image2.data(), sycl_compat::range<1>(local_image2.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto in =
                    in_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    out_buf.get_access<sycl_compat::access::mode::read_write>(h);
                h.parallel_for<ImageAdjustRedKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_count)),
                    [=](sycl_compat::id<1> idx) {
                        Pixel p = out[idx];
                        p.r = sycl_compat::min(255, in[idx].r + 50);
                        out[idx] = p;
                    });
            });
            q.wait();
        }

        {
            sycl_compat::buffer<Pixel, 1> in_buf(
                local_image2.data(), sycl_compat::range<1>(local_image2.size()));
            sycl_compat::buffer<Pixel, 1> out_buf(
                local_image3.data(), sycl_compat::range<1>(local_image3.size()));

            q.submit([&](sycl_compat::handler& h) {
                auto in =
                    in_buf.get_access<sycl_compat::access::mode::read>(h);
                auto out =
                    out_buf.get_access<sycl_compat::access::mode::write>(h);
                h.parallel_for<ImageAdjustBrightnessKernel>(
                    sycl_compat::range<1>(
                        static_cast<std::size_t>(local_count)),
                    [=](sycl_compat::id<1> idx) {
                        Pixel p;
                        p.r = sycl_compat::min(255, in[idx].r + 30);
                        p.g = sycl_compat::min(255, in[idx].g + 30);
                        p.b = sycl_compat::min(255, in[idx].b + 30);
                        out[idx] = p;
                    });
            });
            q.wait();
        }
    }

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = rows_for_rank(r, size) * width;
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<int> byte_counts(size, 0);
    std::vector<int> byte_displs(size, 0);
    for (int r = 0; r < size; ++r) {
        byte_counts[r] = counts[r] * static_cast<int>(sizeof(Pixel));
        byte_displs[r] = displs[r] * static_cast<int>(sizeof(Pixel));
    }

    std::vector<Pixel> global_image(total, {0, 0, 0});
    MPI_Gatherv(local_image3.data(),
                local_count * static_cast<int>(sizeof(Pixel)),
                MPI_BYTE,
                global_image.data(),
                byte_counts.data(),
                byte_displs.data(),
                MPI_BYTE,
                0,
                MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    double max_seconds = 0.0;
    MPI_Reduce(
        &local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cerr << "[MPI_StandardSycl][imageAdjustment][naive] seconds="
                  << max_seconds << std::endl;
        std::cout << "Original Image:" << std::endl;
        std::cout << "\nImage After Color Adjustment:" << std::endl;
        std::cout << "\nImage After Brightness Enhancement:" << std::endl;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                std::cout << global_image[static_cast<std::size_t>(i) * width +
                                          j]
                          << " ";
            }
            std::cout << std::endl;
        }
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
        std::cerr << "[MPI_StandardSycl][imageAdjustment][naive] e2e_seconds="
                  << e2e_seconds << std::endl;
    }

    MPI_Finalize();
    return 0;
}
