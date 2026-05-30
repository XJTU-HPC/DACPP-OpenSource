#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"


using namespace std;
namespace dacpp {
    typedef std::vector<std::any> list;
}



const int NX = 8;
const int NY = 8;
const double Lx = 10.0f;
const double Ly = 10.0f;
const double alpha = 0.01f;
const int TIME_STEPS = 10;

const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);


const double dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
const double delta_t = 0.4f * dt_stability;

shell dacpp::list stencilShell( dacpp::Matrix<double>& matIn READ_WRITE,
                                dacpp::Matrix<double>& matOut READ_WRITE) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matIn[sp1][sp2], matOut[idx1][idx2]};
    return dataList;
}

calc void stencil(dacpp::Matrix<double>& mat,
                    double* out) {
    out[0] = mat[1][1] + alpha *delta_t * (((mat[2][1] - 2.0f * mat[1][1] + mat[0][1]) / (dx * dx))+ ((mat[1][2] - 2.0f * mat[1][1] + mat[1][0]) / (dy * dy)));
}

int main() {






    vector<double> u_prev(NX * NY, 0.0f);
    vector<double> u_curr(NX * NY, 0.0f);
    vector<double> u_next(NX * NY, 0.0f);


    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;

            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }


    dacpp::Matrix<double> matIn({NX, NY}, u_curr);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matOut = u_next_tensor[{1,NX-1}][{1,NY-1}];
    for(int i=0;i<TIME_STEPS;i++) {
        stencilShell(matIn, matOut) <-> stencil;

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                matIn[i][j]=matOut[i-1][j-1];
            }
        }


        for (int j = 0; j <= NY-1; ++j) {

            matIn[0][j]=matIn[1][j];
            matIn[NX - 1][j]=matIn[NX-2][j];

        }
        for (int i = 0; i < NX-1; ++i) {

            matIn[i][0]=matIn[i][1];
            matIn[i][NY-1]=matIn[i][NY-2];
        }

    }
    matIn[0].print();





    return 0;
}