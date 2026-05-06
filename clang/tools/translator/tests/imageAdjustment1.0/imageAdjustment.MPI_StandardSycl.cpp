// Image Adjustment — MPI + SYCL standard implementation
// Element-parallel operations: color adjustment and brightness enhancement.
// Split image rows across ranks.

#include <CL/sycl.hpp>
#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace sycl = cl::sycl;

struct Pixel {
    int r, g, b;
    friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
        os << "(" << p.r << ", " << p.g << ", " << p.b << ")";
        return os;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int width = 4;
    constexpr int height = 4;
    constexpr int total = width * height;

    // Divide rows across ranks
    const int base = height / size;
    const int rem  = height % size;
    const int local_rows = base + (rank < rem ? 1 : 0);
    const int row_begin = rank * base + std::min(rank, rem);
    const int local_count = local_rows * width;

    // Initialize local portion of the image
    std::vector<Pixel> image(local_count, {100, 100, 100});
    std::vector<Pixel> image2(local_count, {0, 0, 0});
    std::vector<Pixel> image3(local_count, {0, 0, 0});

    sycl::queue q{sycl::default_selector_v};

    // Step 1: Color adjustment — increase red by 50
    {
        sycl::buffer<Pixel, 1> buf_in(image.data(), sycl::range<1>(local_count));
        sycl::buffer<Pixel, 1> buf_out(image2.data(), sycl::range<1>(local_count));

        q.submit([&](sycl::handler& h) {
            auto acc_in  = buf_in.get_access<sycl::access::mode::read>(h);
            auto acc_out = buf_out.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                acc_out[idx].r = std::min(255, acc_in[idx].r + 50);
                acc_out[idx].g = acc_in[idx].g;
                acc_out[idx].b = acc_in[idx].b;
            });
        });
        q.wait();
    }

    // Step 2: Brightness enhancement — increase RGB by 30
    {
        sycl::buffer<Pixel, 1> buf_in(image2.data(), sycl::range<1>(local_count));
        sycl::buffer<Pixel, 1> buf_out(image3.data(), sycl::range<1>(local_count));

        q.submit([&](sycl::handler& h) {
            auto acc_in  = buf_in.get_access<sycl::access::mode::read>(h);
            auto acc_out = buf_out.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(local_count), [=](sycl::id<1> idx) {
                acc_out[idx].r = std::min(255, acc_in[idx].r + 30);
                acc_out[idx].g = std::min(255, acc_in[idx].g + 30);
                acc_out[idx].b = std::min(255, acc_in[idx].b + 30);
            });
        });
        q.wait();
    }

    // Gather final image to rank 0
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int lr = base + (r < rem ? 1 : 0);
        counts[r] = lr * width;
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<int> byte_counts(size), byte_displs(size);
    for (int r = 0; r < size; ++r) {
        byte_counts[r] = counts[r] * sizeof(Pixel);
        byte_displs[r] = displs[r] * sizeof(Pixel);
    }

    std::vector<Pixel> global_image;
    if (rank == 0) global_image.resize(total);

    MPI_Gatherv(image3.data(), local_count * sizeof(Pixel), MPI_BYTE,
                rank == 0 ? global_image.data() : nullptr,
                byte_counts.data(), byte_displs.data(),
                MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Original Image:" << std::endl;
        std::cout << "\nImage After Color Adjustment:" << std::endl;
        std::cout << "\nImage After Brightness Enhancement:" << std::endl;
        // Print as 2D grid
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                std::cout << global_image[i * width + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
