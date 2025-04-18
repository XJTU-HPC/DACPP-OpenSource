#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

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
        for (int i = 0; i < n / 2; i++) {
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
    const int N = 1024;  // 假设数组的大小为1024
    vector<int> array(N);

    // 初始化数据
    srand(time(0));
    for (int i = 0; i < N; i++) {
        array[i] = rand() % 1000;  // 随机生成0到1000之间的整数
    }

    // 打印排序前的数组（前10个）
    cout << "Array before sorting (first 10 elements):" << endl;
    for (int i = 0; i < 10; i++) {
        cout << array[i] << " ";
    }
    cout << endl;

    // 执行奇偶归并排序
    oddEvenMergeSort(array, N);

    // 打印排序后的数组（前10个）
    cout << "Array after sorting (first 10 elements):" << endl;
    for (int i = 0; i < 10; i++) {
        cout << array[i] << " ";
    }
    cout << endl;

    return 0;
}
