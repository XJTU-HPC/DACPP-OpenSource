#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <vector>
#include <iostream>

#include "Slice.h"

namespace dacpp {

template<typename ImplType>
class Tensor {
private:
    std::shared_ptr<ImplType> data_;
    int offset_;
    int dim_;
    std::shared_ptr<int> shape_;
    std::shared_ptr<int> stride_;

    void recursiveTake(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_.get()[i];
            }
            data[idx++] = data_.get()[index];
            return;
        }
        for(int i = 0; i < shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveTake(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }

    void recursiveBring(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_.get()[i];
            }
            data_.get()[index] = data[idx++];
            return;
        }
        for(int i = 0; i < shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveBring(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }

    void recursivePrint(std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_.get()[i];
            }
            std::cout << data_.get()[index];
            return;
        }
        std::cout << "{";
        for(int i = 0; i < shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursivePrint(indices, dimIdx + 1);
            if(i != shape_.get()[dimIdx] - 1) std::cout << ", ";
            indices.pop_back();
        }
        std::cout << "}";
    }

public:
    Tensor() {
    }

    /**
     * 标量 Tensor 构造函数
    */
    Tensor(ImplType data) {
        data_ = std::shared_ptr<ImplType>(new ImplType[1], std::default_delete<ImplType[]>());
        data_.get()[0] = data;
        offset_ = 0;
        dim_ = 0;
        shape_ = std::shared_ptr<ImplType>(new int[1], std::default_delete<int[]>());
        stride_ = std::shared_ptr<ImplType>(new int[1], std::default_delete<int[]>());
    }

    Tensor(ImplType* data, int size, int* shape, int dim) {
        // 参数检查
        // int count = 1;
        // for(int i = 0; i < dim; i++) count *= shape[i];
        // if(count != size) {}

        data_ = std::shared_ptr<ImplType>(new ImplType[size], std::default_delete<ImplType[]>());
        for(int idx = 0; idx < size; idx++) {
            data_.get()[idx] = data[idx];
        }
        offset_ = 0;
        dim_ = dim;
        shape_ = std::shared_ptr<ImplType>(new int[dim], std::default_delete<int[]>());
        stride_ = std::shared_ptr<ImplType>(new int[dim], std::default_delete<int[]>());
        for(int idx = dim - 1; idx >= 0; idx--) {
            shape_.get()[idx] = shape[idx];
            if(idx == dim_ - 1) stride_.get()[idx] = 1;
            else stride_.get()[idx] = stride_.get()[idx + 1] * shape[idx + 1];
        }
    }

    Tensor(const std::vector<ImplType>& data, const std::vector<int>& shape) {
        // 参数检查
        // int count = 1;
        // for(int i = 0; i < shape.size(); i++) count *= shape[i];
        // if(count != data.size()) {}

        data_ = std::shared_ptr<ImplType>(new ImplType[data.size()], std::default_delete<ImplType[]>());
        for(int idx = 0; idx < data.size(); idx++) {
            data_.get()[idx] = data[idx];
        }
        offset_ = 0;
        dim_ = shape.size();
        shape_ = std::shared_ptr<int>(new int[dim_], std::default_delete<int[]>());
        stride_ = std::shared_ptr<int>(new int[dim_], std::default_delete<int[]>());
        for(int idx = dim_ - 1; idx >= 0; idx--) {
            shape_.get()[idx] = shape[idx];
            if(idx == dim_ - 1) stride_.get()[idx] = 1;
            else stride_.get()[idx] = stride_.get()[idx + 1] * shape[idx + 1];
        }
    }

    Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) : data_(data), offset_(offset), dim_(dim), shape_(shape), stride_(stride) {}

    ~Tensor() {
    }

    std::shared_ptr<ImplType> getDataPtr() const {
        return data_;
    }

    int getOffset() const {
        return offset_;
    }

    // 获得 Tensor 的维度
    int getDim() const {
        return dim_;
    }

    std::shared_ptr<int> getShapePtr() const {
        return shape_;
    }

    std::shared_ptr<int> getStridePtr() const {
        return stride_;
    }

    // 获得 Tensor 某一维的大小
    // dimIdx：维度索引
    int getShape(int dimIdx) const {
        // 参数检查
        // if(dimIdx >= dim_) {}

        return shape_.get()[dimIdx]; 
    }

    int getStride(int dimIdx) const {
        return stride_.get()[dimIdx];
    }

    int getSize() const {
        int size = 1;
        for(int dimIdx = 0; dimIdx < getDim(); dimIdx++) {
            size *= shape_.get()[dimIdx];
        }
        return size;
    }

    // 获得 Tensor 指定下标数据
    // indices：所有维度的下标组成的数组
    ImplType getData(int* indices) const {
        int index = offset_;
        for(int idx = 0; idx < dim_; idx++) {
            index += indices[idx] * stride_.get()[idx];
        }
        return data_.get()[index];
    }

    Tensor<ImplType> operator+(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] + operand.getData(indices);
            delete[] indices;
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    // Tensor<ImplType> operator+(ImplType> operand) const {

    // }

    Tensor<ImplType> operator-(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] - operand.getData(indices);
            delete[] indices;
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    Tensor<ImplType> operator*(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] * operand.getData(indices);
            delete[] indices;
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    Tensor<ImplType> operator/(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] / operand.getData(indices);
            delete[] indices;
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    Tensor<ImplType> operator%(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] % operand.getData(indices);
            delete[] indices;
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator+=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] += operand.getData(indices);
            delete[] indices;
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator-=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] -= operand.getData(indices);
            delete[] indices;
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator*=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] *= operand.getData(indices);
            delete[] indices;
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator/=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] /= operand.getData(indices);
            delete[] indices;
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator%=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_.get()[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] %= operand.getData(indices);
            delete[] indices;
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator=(const Tensor<ImplType>& operand) {
        data_ = operand.getDataPtr();
        offset_ = operand.getOffset();
        dim_ = operand.getDim();
        shape_ = operand.getShapePtr();
        stride_ = operand.getStridePtr();
    }

    Tensor<ImplType> operator[](int idx) const {
        return slice(0, idx);
    }

    Tensor<ImplType> operator[](Slice slc) const {
        if(slc.isAll_) {
            return slice(0, 0, shape_.get()[0]);
        }
        else {
            return slice(0, slc.start_, slc.end_, slc.stride_);
        }
    }

    Tensor<ImplType> operator[](split sp) const {
        return *this;
    }

    // Tensor 切片操作
    // dimIdx：进行切片的维度
    // idx：索引
    Tensor<ImplType> slice(int dimIdx, int idx) const {
        // 参数检查
        // if(dimIdx >= dim_ || idx >= shape_.get()[dimIdx]) {}

        int offset = offset_ + idx * stride_.get()[dimIdx];
        int dim = dim_ - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = shape_.get()[idx];
                stride.get()[idx] = stride_.get()[idx];
            }
            else {
                shape.get()[idx] = shape_.get()[idx + 1];
                stride.get()[idx] = stride_.get()[idx + 1];
            }
        }
        return Tensor<ImplType>(data_, offset, dim, shape, stride);
    }

    // Tensor 切片操作
    // dimIdx：进行切片的维度
    // start：开始索引
    // end：结束索引
    // sliceStride：步长
    Tensor<ImplType> slice(int dimIdx, int start, int end, int sliceStride = 1) const {
        // 参数检查
        // if(dimIdx >= dim_ || start >= shape_.get()[dimIdx] || end > shape_.get()[dimIdx]) {}

        int offset = start * stride_.get()[dimIdx];
        int dim = dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = shape_.get()[i];
            stride.get()[i] = stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = stride_.get()[dimIdx] * sliceStride;
        return Tensor<ImplType>(data_, offset, dim, shape, stride);
    }

    ImplType getElement(std::vector<int> indices) {
        Tensor<ImplType> temp = slice(0, indices[0]);
        for(int i = 1; i < indices.size(); i++) {
            temp = temp.slice(0, indices[i]);
        }
        return temp.getDataPtr().get()[temp.getOffset()];
    }

    ImplType getValue() {
        ImplType a;
        return a;
    }

    // 获取 Tensor 中的数据，用基本类型数组保存
    // data：基本类型数组
    void tensor2Array(ImplType* data) const {
        std::vector<int> indices;
        int idx = 0;
        recursiveTake(data, idx, indices, 0);
    }

    // 改变 Tensor 中的数据
    // data：基本类型数组
    void array2Tensor(ImplType* data) {
        std::vector<int> indices;
        int idx = 0;
        recursiveBring(data, idx, indices, 0);
    }

    void print() const {
        std::vector<int> indices;
        recursivePrint(indices, 0);
        std::cout << "\n";
    }

    /**
     * 修改 Tensor 中的值
     * 
     * @param value: 新值
     * @param pos: 值的坐标
     */
    void reviseValue(ImplType value, std::vector<int> pos) {
        int idx = offset_;
        for(int i = 0; i < pos.size(); i++) {
            idx += pos[i] * stride_.get()[i];
        }
        data_.get()[idx] = value;
    }

    // // 测试代码
    // void test() {
    //     std::cout << "数据引用次数" << data_.use_count() << "\n";
    //     std::cout << "偏移" << offset_ << "\n";
    //     std::cout << "维度" << dim_ << "\n";
    //     for(int i = 0; i < dim_; i++) {
    //         std::cout << shape_.get()[i] << " ";
    //     }
    //     std::cout << "\n";
    //     for(int i = 0; i < dim_; i++) {
    //         std::cout << stride_.get()[i] << " ";
    //     }
    //     std::cout << "\n";
    // }

};

} // namespace dacpp

#endif