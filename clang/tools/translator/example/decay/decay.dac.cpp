#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}


const double dt = 0.1;
const double T = 5.0;
const size_t numIsotopes = 10;
shell dacpp::list DECAY(const dacpp::Vector<double>& N0s,
                        const dacpp::Vector<double>& lambdas,
                        dacpp::Vector<double>& local_A,
                        const dacpp::Vector<double>& t) {
    dacpp::index i;


    dacpp::list dataList{N0s[i], lambdas[i], local_A[i], t[{}]};
    return dataList;
}

calc void decay(double* N0s,
                double* lambdas,
                double* local_A,
                double* t) {
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);
}


void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s, double dt, double T) {
    size_t numIsotopes = lambdas.size();
    std::vector<double> A(T/dt*numIsotopes, 0.0);
    std::vector<double> time;
    std::vector<double> t;
    t.push_back(static_cast<double>(0));


    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({static_cast<int>(T/dt), static_cast<int>(numIsotopes)}, A);


    while(t_tensor[0] <= T){
        DECAY(N0s_tensor, lambdas_tensor, local_A_tensor,  t_tensor) <-> decay;
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    A_tensor[1].print();
}

int main() {



    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);


    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01 + 0.01*i;
    }




    calculateDecay(lambdas, N0s, dt, T);

    return 0;
}