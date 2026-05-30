#ifndef TENSOR_EXCEPTION_H_
#define TENSOR_EXCEPTION_H_

#include <exception>
#include <string>
#include <sstream>

namespace tensor {

class TensorException : public std::exception {
public:
    TensorException(const std::string& message, const char* file, const char* function, int line)
        : message_(message), file_(file), function_(function), line_(line) {
        constructFullMessage();
    }

    virtual const char* what() const noexcept override {
        return fullMessage_.c_str();
    }

private:
    void constructFullMessage() {
        std::ostringstream oss;
        oss << "Error: " << message_ << "\n"
            << "File: " << file_ << "\n"
            << "Function: " << function_ << "\n"
            << "Line: " << line_;
        fullMessage_ = oss.str();
    }

    std::string message_;
    const char* file_;
    const char* function_;
    int line_;
    std::string fullMessage_;
};
#define THROW_TENSOR_EXCEPTION(fmt, ...) \
    { \
        char buffer[256]; \
        snprintf(buffer, sizeof(buffer), fmt, ##__VA_ARGS__); \
        throw tensor::TensorException(buffer, __FILE__, __FUNCTION__, __LINE__); \
    }

#define CHECK_TENSOR_SIZE(data_size, shape, dim) \
    do { \
        int expected_size = 1; \
        for (int i = 0; i < (dim); i++) expected_size *= (shape)[i]; \
        if (expected_size != (data_size)) { \
            THROW_TENSOR_EXCEPTION("Expected size %d, but got %d", expected_size, data_size); \
        } \
    } while (0)

#define CHECK_INDEX_BOUNDS(idx, dim) \
    do { \
        if ((idx) < 0 || (idx) >= (dim)) { \
            THROW_TENSOR_EXCEPTION("Index out of bounds: idx = %d, dim = %d", (idx), (dim)); \
        } \
    } while (0)

#define CHECK_INDEX_VALID(idx, dim) \
    do { \
        if ((idx) < 0 || (idx) >= (dim)) { \
            THROW_TENSOR_EXCEPTION("Index out of bounds: idx = %d, dim = %d", (idx), (dim)); \
        } \
    } while (0)

#define CHECK_SHAPE_MATCH(a, b) \
    do { \
        if ((a).getDim() != (b).getDim()) { \
            THROW_TENSOR_EXCEPTION("Tensor dimensions do not match: %d vs %d", (a).getDim(), (b).getDim()); \
        } \
        for (int i = 0; i < (a).getDim(); ++i) { \
            if ((a).getShape(i) != (b).getShape(i)) { \
                THROW_TENSOR_EXCEPTION("Tensor shapes do not match at dimension %d: %d vs %d", i, (a).getShape(i), (b).getShape(i)); \
            } \
        } \
    } while (0)

#define CHECK_MATRIX_MULTIPLY_COMPATIBLE(a, b) \
    do { \
        if ((a).getDim() != 2 || (b).getDim() != 2) { \
            THROW_TENSOR_EXCEPTION("Both tensors must be 2-dimensional for matrix multiplication: %d vs %d", (a).getDim(), (b).getDim()); \
        } \
        if ((a).getShape(1) != (b).getShape(0)) { \
            THROW_TENSOR_EXCEPTION("Incompatible shapes for matrix multiplication: %d (A columns) vs %d (B rows)", (a).getShape(1), (b).getShape(0)); \
        } \
    } while (0)

#define CHECK_SLICE(dimIdx, idx_or_start, end, dim_, shape_) \
    do { \
        if ((dimIdx) < 0 || (dimIdx) >= (dim_) || \
            ((sizeof(idx_or_start) == sizeof(int)) && \
             ((idx_or_start) < 0 || (idx_or_start) >= (shape_).get()[(dimIdx)])) || \
            ((sizeof(idx_or_start) != sizeof(int)) && \
             ((idx_or_start) < 0 || (idx_or_start) >= (shape_).get()[(dimIdx)])) || \
            ((end) <= (idx_or_start)) || (end) > (shape_).get()[(dimIdx)]) { \
            THROW_TENSOR_EXCEPTION("Invalid slice parameters: dimIdx = %d, idx_or_start = %d, end = %d", (dimIdx), (idx_or_start), (end)); \
        } \
    } while (0)

}

#endif
