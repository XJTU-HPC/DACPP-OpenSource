#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

constexpr size_t NUM_NEURONS = 8192;
constexpr size_t INPUT_SIZE  = 4096;

int main() {
    std::vector<float> host_grads(NUM_NEURONS * INPUT_SIZE);
    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        for (size_t j = 0; j < INPUT_SIZE; ++j) {
            host_grads[i * INPUT_SIZE + j] = static_cast<float>(j % 32);
        }
    }

    std::vector<float> partial_sum(NUM_NEURONS, 0.0f);
    queue q;

    {
        buffer<float, 1> grad_buf(host_grads.data(), range<1>(NUM_NEURONS * INPUT_SIZE));
        buffer<float, 1> partial_buf(partial_sum.data(), range<1>(NUM_NEURONS));

        q.submit([&](handler& h) {
            auto grads_acc = grad_buf.get_access<access::mode::read>(h);
            auto sum_acc = partial_buf.get_access<access::mode::discard_write>(h);
            h.parallel_for(range<1>(NUM_NEURONS), [=](id<1> neuron_id) {
                float sum = 0.0f;
                for (size_t j = 0; j < INPUT_SIZE; ++j) {
                    sum += grads_acc[j * NUM_NEURONS + neuron_id[0]];
                }
                sum_acc[neuron_id] = sum;
            });
        }).wait();
    }

    std::cout << "First 5 neuron gradient sums:\n";
    for (size_t i = 0; i < std::min<size_t>(5, partial_sum.size()); ++i) {
        std::cout << partial_sum[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
