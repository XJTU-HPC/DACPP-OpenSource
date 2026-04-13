#ifndef MYTENSOR_HPP
#define MYTENSOR_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>

#include "Slice.h"

namespace dacpp {



template<class ImplType>
class TensorBase{
public:
    TensorBase(){}
    ~TensorBase(){}
    std::shared_ptr<ImplType> getDataPtr() const {
        return this->data_;
    }
    int getOffset() const {
        return this->offset_;
    }
    int getDim() const {
        return this->dim_;
    }
    std::shared_ptr<int> getShapePtr() const {
        return this->shape_;
    }
    std::shared_ptr<int> getStridePtr() const {
        return this->stride_;
    }
    int getShape(int dimIdx) const {
        return this->shape_.get()[dimIdx]; 
    }
    int getStride(int dimIdx) const {
        return this->stride_.get()[dimIdx];
    }
    int getSize() const {
        int size = 1;
        for(int dimIdx = 0; dimIdx < getDim(); dimIdx++) {
            size *= this->shape_.get()[dimIdx];
        }
        return size;
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
    int getCurrentDim() const {
        return this->current_dim;
    }
protected:

    void NextDim(){
        this->current_dim ++;
    }
    void recursiveTake(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) {
                index += indices[i] * this->stride_.get()[i];
            }
            data[idx++] = this->data_.get()[index];
            return;
        }
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveTake(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }

    void recursiveBring(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) {
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) {
                index += indices[i] * this->stride_.get()[i];
            }
            this->data_.get()[index] = data[idx++];
            return;
        }
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveBring(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }

    void recursivePrint(std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) {
                index += indices[i] * this->stride_.get()[i];
            }
            std::cout << this->data_.get()[index];
            return;
        }
        std::cout << "{";
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursivePrint(indices, dimIdx + 1);
            if(i != this->shape_.get()[dimIdx] - 1) std::cout << ", ";
            indices.pop_back();
        }
        std::cout << "}";
    }
    std::shared_ptr<ImplType> data_;
    int offset_;
    int dim_;
    std::shared_ptr<int> shape_;
    std::shared_ptr<int> stride_;
    int current_dim = 0;
};

template<class ImplType, int N>
class Tensor: public TensorBase <ImplType>{
public:
    Tensor() {}
    Tensor(const Tensor<ImplType, N> &x){
        this->data_ = x.data_;
        this->offset_ = x.offset_;
        this->dim_ = x.dim_;
        this->shape_ = x.shape_;
        this->stride_ = x.stride_;
        this->current_dim = 0;
    }
    Tensor(const std::initializer_list<int> values, const ImplType* data){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[ElementSize], std::default_delete<ImplType[]>());
        for(int i = 0; i < ElementSize; i++)
            this->data_.get()[i] = data[i];
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        const auto *it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    Tensor(const std::initializer_list<int> values, const std::vector<ImplType> data){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        if(ElementSize != data.size())  
            throw("The number of elements in the vector does not correspond to the Shape.");
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[data.size()], std::default_delete<ImplType[]>());
        for(size_t i = 0; i < data.size(); i++)
            this->data_.get()[i] = data[i];
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        const auto *it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    Tensor(const std::initializer_list<int> values, ImplType data = 0){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[ElementSize], std::default_delete<ImplType[]>());
        for(int i = 0; i < ElementSize; i++)
            this->data_.get()[i] = data;
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        const auto *it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
    }
    Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    ~Tensor() {}
    Tensor& operator=(const Tensor<ImplType, N>& operand) {
        this->data_ = operand.getDataPtr();
        this->offset_ = operand.getOffset();
        this->dim_ = operand.getDim();
        this->shape_ = operand.getShapePtr();
        this->stride_ = operand.getStridePtr();
        this->current_dim = 0;
        return *this;
    }
    Tensor<ImplType, N> operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0){
            // this->current_dim++;
            // Tensor<ImplType, N> tmp = *this;
            // tmp.current_dim = this->current_dim + 1;
            //if(tmp.current_dim > tmp.dim_)
            if(this->current_dim + 1 > this->dim_)
                throw "Slice operates on dimensions that exceed those of Tensor.";
            std::cout<<this->current_dim<<std::endl;
            return slice(this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        }
        else if(idx.size() == 1){
            const int i = *(idx.begin());
            //std::cout<<this->current_dim + 1<<std::endl;
            return slice(this->current_dim, i, i + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            //std::cout<<this->current_dim + 1<<std::endl;
            return slice(this->current_dim, start, end, stride, 1);
        }
    }
    Tensor<ImplType, N-1> operator[](int idx) const {
        return slice(this->current_dim, idx);
    }

    Tensor<ImplType, N> operator[](RegularSplit sp) const {
        return *this;
    }
    Tensor<ImplType, N - 1> operator[](Index sp) const {
        return Tensor<ImplType, N-1>();
    }
    // Tensor 切片操作
    // dimIdx：进行切片的维度
    // idx：索引
    Tensor<ImplType, N-1> slice(int dimIdx, int idx) const {
        // 参数检查
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw("Slice operates on dimensions that exceed those of Tensor.");

        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        int dim = this->dim_ - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = this->shape_.get()[idx];
                stride.get()[idx] = this->stride_.get()[idx];
            }
            else {
                shape.get()[idx] = this->shape_.get()[idx + 1];
                stride.get()[idx] = this->stride_.get()[idx + 1];
            }
        }
        return Tensor<ImplType, N - 1>(this->data_, offset, dim, shape, stride, this->current_dim);
    }
    Tensor<ImplType, N> slice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const {
        // 参数检查
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw("Slice operates on dimensions that exceed those of Tensor.");
        int offset = start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return Tensor<ImplType, N>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    Tensor<ImplType, N> operator+(const Tensor<ImplType, N>& operand) const {

    }
    Tensor<ImplType, N> operator-(const Tensor<ImplType, N>& operand) const {

    }
    Tensor<ImplType, N> operator*(const Tensor<ImplType, N>& operand) const {

    }
    Tensor<ImplType, N> operator/(const Tensor<ImplType, N>& operand) const {

    }
    Tensor<ImplType, N> operator%(const Tensor<ImplType, N>& operand) const {

    }
    void operator+=(const Tensor<ImplType, N>& operand) {

    }
    void operator-=(const Tensor<ImplType, N>& operand) {

    }
    void operator*=(const Tensor<ImplType, N>& operand) {

    }
    void operator/=(const Tensor<ImplType, N>& operand) {

    }
    void operator%=(const Tensor<ImplType, N>& operand) {

    }
};

template<class ImplType>
class Tensor <ImplType, 1> : public TensorBase<ImplType>{//todo :this->data_ -> shared_ptr
private:
    template <class InputIt>
    void initialize(InputIt first, InputIt last) {
        size_t size = std::distance(first, last);

        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[size], std::default_delete<ImplType[]>());

        std::copy(first, last, this->data_.get());

        this->offset_ = 0;
        this->dim_ = 1;
        this->shape_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->shape_.get()[0] = size;
        this->stride_.get()[0] = 1;
    }
public:
    Tensor(){}
    ~Tensor(){}
    Tensor(std::vector<ImplType> init){
        initialize(init.begin(), init.end());
    }
    Tensor(std::initializer_list<ImplType> init){
        initialize(init.begin(), init.end());
    }
    Tensor(int len, const ImplType *init){
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[len], std::default_delete<ImplType[]>());

        std::copy_n(init, len, this->data_.get());

        this->offset_ = 0;
        this->dim_ = 1;
        this->shape_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->shape_.get()[0] = len;
        this->stride_.get()[0] = 1;
    }
    Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
    }
    Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    Tensor& operator=(const Tensor<ImplType, 1>& operand) {
        this->data_ = operand.getDataPtr();
        this->offset_ = operand.getOffset();
        this->dim_ = operand.getDim();
        this->shape_ = operand.getShapePtr();
        this->stride_ = operand.getStridePtr();
        this->current_dim = 0;
        return *this;
    }
    Tensor<ImplType, 1> operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0){
            Tensor<ImplType, 1> tmp = *this;
            tmp.current_dim = this->current_dim + 1;
            if(tmp.current_dim > tmp.dim_)
                throw "Slice operates on dimensions that exceed those of Tensor.";
            return tmp;
        }
        else if(idx.size() == 1){
            const int i = *(idx.begin());
            return slice(this->current_dim, i, i + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return slice(this->current_dim, start, end, stride, 1);
        }
    }
    ImplType& operator[](int idx) {
        return slice(this->current_dim, idx);
    }
    Tensor<ImplType, 1> operator[](RegularSplit sp) const {
        return *this;
    }
    ImplType& operator[](Index sp) const ;
    ImplType& slice(int dimIdx, int idx) const {
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw("Slice operates on dimensions that exceed those of Tensor.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        return this->data_.get()[offset];
    }
    Tensor<ImplType, 1> slice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const {
        // 参数检查
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw("Slice operates on dimensions that exceed those of Tensor.");
        int offset = start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return Tensor<ImplType, 1>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    Tensor<ImplType, 1> operator+(const Tensor<ImplType, 1>& operand) const {

    }

    Tensor<ImplType, 1> operator-(const Tensor<ImplType, 1>& operand) const {

    }

    Tensor<ImplType, 1> operator*(const Tensor<ImplType, 1>& operand) const {

    }

    Tensor<ImplType, 1> operator/(const Tensor<ImplType, 1>& operand) const {

    }

    Tensor<ImplType, 1> operator%(const Tensor<ImplType, 1>& operand) const {

    }

    void operator+=(const Tensor<ImplType, 1>& operand) {

    }

    void operator-=(const Tensor<ImplType, 1>& operand) {

    }

    void operator*=(const Tensor<ImplType, 1>& operand) {

    }

    void operator/=(const Tensor<ImplType, 1>& operand) {

    }

    void operator%=(const Tensor<ImplType, 1>& operand) {

    }
};

template <class T>
using Vector = Tensor<T, 1>;
template <class T>
using Matrix = Tensor<T, 2>;

} // namespace dacpp

#endif
