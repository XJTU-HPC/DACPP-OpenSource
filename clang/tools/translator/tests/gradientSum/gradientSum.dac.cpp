#include <iostream>
#include <vector>
#include <random>
#include <any>
#include "ReconTensor.h"

using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int NUM_NEURONS = 8;   // 神经元数量（层宽度）
const int INPUT_SIZE  = 8;   // 每个神经元输入数
 
shell dacpp::list gradSumShell(dacpp::Matrix<float>& matGrads READ,
                               dacpp::Matrix<float>& matNeuronSum WRITE) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{matGrads[{}][idx1], matNeuronSum[idx1][idx2]};
    return dataList;
}

// -----------------------------
// DAC calc 函数：规约加法
// -----------------------------
calc void gradSum(dacpp::Vector<float>& grads, dacpp::Vector<float>& neuronSum) {
    int sum=0;
    for (int j = 0; j < INPUT_SIZE; ++j) {
        sum += grads[j];   // 规约加法
    }
    neuronSum[0]=sum;
}

// -----------------------------
// 主函数
// -----------------------------
int main() {
    // 初始化梯度矩阵（模拟随机梯度）
    vector<float> host_grads(NUM_NEURONS * INPUT_SIZE);
    mt19937 gen(42);
    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    // for (auto &v : host_grads) v = dist(gen);
    for(int i=0;i<NUM_NEURONS;i++){
        for(int j=0;j<INPUT_SIZE;j++){
            host_grads[i*INPUT_SIZE+j]=i + j;
        }
    }
    // DAC Tensor 初始化
    dacpp::Matrix<float> matGrads({NUM_NEURONS, INPUT_SIZE}, host_grads);
    vector<float> host_neuron_sum(NUM_NEURONS, 0.0f);
    dacpp::Matrix<float> matNeuronSum({NUM_NEURONS, 1}, host_neuron_sum);

    // 执行 DAC shell -> calc
    gradSumShell(matGrads, matNeuronSum) <-> gradSum;

    // 输出结果
    std::cout << "First 5 neuron gradient sums:\n";
    for (size_t i = 0; i < std::min(5, NUM_NEURONS) ; ++i)
        std::cout << matNeuronSum[i][0] << " ";
    std::cout << std::endl;

    return 0;
}
