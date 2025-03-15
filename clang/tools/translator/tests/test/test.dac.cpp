#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
shell dacpp::list MC(const dacpp::Vector<double>& x,
                       const dacpp::Vector<double>& y,
                        const dacpp::Vector<int>& inside_circle) {
    dacpp::index i;
    //dacpp::split s(3,1);
   // binding(i, s);
    dacpp::list dataList{x[{i}], y[{i}], inside_circle[{i}]};
    return dataList;
}

calc void mc(double x,
                double y,
                int inside_circle) {
    x = rand() / (double)RAND_MAX;
    y = rand() / (double)RAND_MAX;

    // 判断是否在单位圆内
    if (x * x + y * y <= 1.0) {
        inside_circle = 1;
    }

}


double monte_carlo_pi(int num_samples) {
    std::vector<double> x(num_samples,0.0);
    std::vector<double> y(num_samples,0.0);
    std::vector<int> inside_circle(num_samples,0.0);
    int inside_circle_sum = 0;

    dacpp::Vector<double> x_tensor(x);
    dacpp::Vector<double> y_tensor(y);
    dacpp::Vector<int> inside_circle_tensor(inside_circle);

    MC(x_tensor, y_tensor, inside_circle_tensor) <-> mc;
    // 计算π的估算值

    for(int  i = 0; i < num_samples; i++ ){
        if(inside_circle_tensor[i] == 1)   inside_circle_sum++;
    }

    return 4.0 * inside_circle_sum / num_samples;
}

int main() {
    srand(time(0));  // 用当前时间作为随机种子
    int num_samples = 1000000;  // 选择模拟的随机点数量
    double pi_estimate = monte_carlo_pi(num_samples);

    cout << "Estimated Pi: " << pi_estimate << endl;
    return 0;
}
