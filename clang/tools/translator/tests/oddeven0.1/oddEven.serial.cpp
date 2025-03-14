#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

const int N = 8;  // 假设数组的大小为8

// 交换函数
void swap(vector<int>& array, int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

// 奇偶归并排序的核心操作
void oddEvenMergeSort(vector<int>& array, int n) {
    // 每一轮排序进行多次比较
    for (int phase = 0; phase < n; phase++) {
        // 奇数阶段：比较相邻的奇数索引
        for (int i = 0; i < n - 1; i++) {
            if (phase % 2 == 0 && (i % 2 == 0)) {
                if (array[i] > array[i + 1]) {
                    swap(array, i, i + 1);
                }
            }
            // 偶数阶段：比较相邻的偶数索引
            if (phase % 2 != 0 && (i % 2 == 1)) {
                if (array[i] > array[i + 1]) {
                    swap(array, i, i + 1);
                }
            }
        }
    }
}

// 主函数：初始化数据并调用奇偶归并排序
int main() {
    vector<int> array(N);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        array[i] = N - i;  // 初始化为递减的数组
    }

    // 打印排序前的数组
    //cout << "Array before sorting:" << endl;
    for (int i = 0; i < N; i++) {
        //cout << array[i] << " ";
    }
    //cout << endl;

    // 执行奇偶归并排序
    oddEvenMergeSort(array, N);

    // 打印排序后的数组
    //cout << "Array after sorting:" << endl;
    std::cout << "{";
    for (int i = 0; i < N; i++) {
        cout << array[i] ;
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "}" << std:: endl;

    return 0;
}
