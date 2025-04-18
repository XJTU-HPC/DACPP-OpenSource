#include "include/ReconTensor.h"
#include <iostream>
#include <vector>
//using dacpp::TensorProxy;
using dacpp::Tensor;

using std::vector;
using std::cout;
using std::endl;


void func(dacpp::Vector<int> t){
    vector<int> data2{2,5,8};
    t.array2Tensor(data2);
    return;
}

int main() {
    vector<int> data{1,2,3,4,5,6,7,8,9};
    Tensor<int,2> u_tensor({3,3}, data);
    Tensor<int,2> b_tensor= u_tensor;
    Tensor<int,1> tt_tensor= u_tensor[{}][1];
    vector<int> data2{20,50,80};
    tt_tensor.array2Tensor(data2);
    u_tensor.print();
    func(u_tensor[{}][1]);
    u_tensor.print();
    // cout<<std::string(50,'=')<<endl;
    // for(int i=0;i<3;i++){
    //     for(int j=0;j<3;j++)
    //         cout<<u_tensor[i][j]<< " ";
    //     cout<<std::endl;
    // }
    // cout<<std::string(50,'=')<<endl;
    // Tensor<int, 2> c_tensor;
    // c_tensor = b_tensor[{}][{1}];
    // b_tensor.print();
    // c_tensor.print();
    // vector<int> dip{9,8,1,4,5,2,3,6,7};
    // u_tensor.array2Tensor(dip);
    // cout<<std::string(50,'=')<<endl;
    // u_tensor.print();
    // cout<<std::string(50,'=')<<endl;

    // u_tensor.tensor2Array(data);
    // for(auto it : data)cout<<it<<" ";
    // cout<<endl;
    // cout<<std::string(50,'=')<<endl;
    // c_tensor.tensor2Array(data);
    // for(auto it : data)cout<<it<<" ";
    // cout<<endl;
    // cout<<std::string(50,'=')<<endl;

    // u_tensor[{}][{1,3}] = b_tensor[{}][{1,3}];
    // Tensor<int, 2> d_tensor = b_tensor[{}][{}];
    // b_tensor.array2Tensor(dip);
    // std::vector<int> indices;
    // for(int i=0;i<3;i++){
    //     indices.push_back(i);
    //     for(int j=0;j<2;j++){
    //         indices.push_back(j);
    //         cout<<d_tensor.getElement(indices)<< " ";
    //         indices.pop_back();
    //     }
    //     indices.pop_back();
    //     cout<<std::endl;
    // }
    // cout<<std::string(50,'=')<<endl;
    return 0;
}
