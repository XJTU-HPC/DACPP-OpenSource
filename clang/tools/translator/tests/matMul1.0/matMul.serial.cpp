#include <iostream>
#include <vector>

using namespace std;

// 矩阵乘法函数
void matrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int m = A.size();     // A 的行数
    int n = A[0].size();  // A 的列数 / B 的行数
    int p = B[0].size();  // B 的列数

    // 执行矩阵乘法
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i][j] = 0;  // 初始化 C[i][j]
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // 初始化两个矩阵 A 和 B
    vector<vector<int>> A = {{1, 2}, {3, 4}};
    vector<vector<int>> B = {{5, 6}, {7, 8}};
    
    // 结果矩阵 C
    vector<vector<int>> C(2, vector<int>(2, 0)); // 初始化 C 为 2x2 矩阵，值为 0

    // 调用矩阵乘法函数
    matrixMultiply(A, B, C);

    // 打印结果矩阵 C
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
