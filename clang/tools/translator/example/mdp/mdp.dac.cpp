#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}

const double A = 1.0;
const double D = 0.1;
const double dx = 0.1;
const double dt = 0.01;
const int N = 150;
const int T = 1000;


void initialize(std::vector<double>& p) {
    for (int i = 0; i < N; ++i) {

        double x = i * dx;
        p[i] = std::exp(-std::pow(x - 5.0, 2) / 2.0);
    }
}


void normalize(dacpp::Vector<double>& p) {
    double sum = 0.0;
    for (int i = 0;i < N-2; i++) {
        sum += p[i];
    }
    for (int i = 0;i < N-2; i++) {
        p[i] /= sum;
    }
}

shell dacpp::list mdp_shell( dacpp::Vector<double>& p READ_WRITE, dacpp::Vector<double>& new_p READ_WRITE){
    dacpp::index idx;
    dacpp::split sp(3,1);
    binding(idx, sp);
    dacpp::list dataList{p[sp],new_p[idx]};
    return dataList;
}

calc void mdp(dacpp::Vector<double>& p, double* new_p){
    double diffusion = D * (p[2] - 2 * p[1] + p[0]) / (dx * dx) ;
    double drift = (-A) * (p[2] - p[0]) / (2 * dx);
    new_p[0] = p[1] + dt * (diffusion+ drift);
}



int main() {
    std::vector<double> p1(N, 0.0);

    initialize(p1);

    std::vector<double> new_p1(N-2, 0.0);
    dacpp::Vector<double> p(p1);
    dacpp::Vector<double> new_p(new_p1);
    for (int t = 0; t < T; ++t) {
        mdp_shell(p, new_p) <->  mdp;


        for(int i = 0; i <= N-3; i++){
            p[i+1] = new_p[i];
        }




    }
    std::cout << p[2] << std::endl;

    return 0;
}