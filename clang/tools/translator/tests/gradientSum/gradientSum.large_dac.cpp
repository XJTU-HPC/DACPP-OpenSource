#include <iostream>
#include <vector>
#include <random>
#include <any>
#include "ReconTensor.h"

using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int NUM_NEURONS = 1024;
const int INPUT_SIZE  = 512;
 
shell dacpp::list gradSumShell(dacpp::Matrix<float>& matGrads READ,
                               dacpp::Matrix<float>& matNeuronSum WRITE) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{matGrads[{}][idx1], matNeuronSum[idx1][idx2]};
    return dataList;
}

calc void gradSum(dacpp::Vector<float>& grads, dacpp::Vector<float>& neuronSum) {
    int sum = 0;
    for (int j = 0; j < INPUT_SIZE; ++j) {
        sum += static_cast<int>(grads[j]);
    }
    neuronSum[0] = static_cast<float>(sum);
}

int main() {
    vector<float> host_grads(NUM_NEURONS * INPUT_SIZE);
    for (int i = 0; i < NUM_NEURONS; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            host_grads[i * INPUT_SIZE + j] = static_cast<float>(j % 32);
        }
    }

    dacpp::Matrix<float> matGrads({NUM_NEURONS, INPUT_SIZE}, host_grads);
    vector<float> host_neuron_sum(NUM_NEURONS, 0.0f);
    dacpp::Matrix<float> matNeuronSum({NUM_NEURONS, 1}, host_neuron_sum);

    gradSumShell(matGrads, matNeuronSum) <-> gradSum;

    std::cout << "First 5 neuron gradient sums:\n";
    for (size_t i = 0; i < std::min<size_t>(5, NUM_NEURONS); ++i) {
        std::cout << matNeuronSum[i][0] << " ";
    }
    std::cout << std::endl;

    return 0;
}
