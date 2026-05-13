#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

/* 
 Matrix header for numerical 2D arrays
 Bertalan Szuchovszky 12.28.2025
 
 Matrix is a 2D array class with:
  -> element access: m(i,j) or m[i][j] both are valid
  -> Basic operations: +, -, *, / (element-wise) //including +=, -=, *=, /=
  -> Other important tilities: fill, max, min, norm
  
 Vector is a 1D array wrapper for std::vector<double> with new vector operations in VecOps
*/

using Vector = std::vector<double>;

//utility functions for Vector
namespace VecOps {
    //operator overrides
    inline Vector operator+(const Vector& a, const Vector& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vector size mismatch"); //only if both are same len
        Vector result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
        return result;
    }
    
    inline Vector operator-(const Vector& a, const Vector& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vector size mismatch"); //only if... - || -
        Vector result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
        return result;
    }
    
    inline Vector operator*(const Vector& a, double scalar) { //const multiplication
        Vector result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] * scalar;
        return result;
    }
    
    inline Vector operator*(double scalar, const Vector& a) {//other way around, as turns out these only work one way
        return a * scalar;
    }
    
    //max/min
    inline double max(const Vector& v) {
        return *std::max_element(v.begin(), v.end()); //there exists a std command for this but whatever
    }
    
    inline double min(const Vector& v) {
        return *std::min_element(v.begin(), v.end());//again, std command but this is shorter
    }
    
    //norm
    inline double norm(const Vector& v) {
        double sum = 0.0;
        for (double x : v) sum += x * x; //sum squared of elements
        return std::sqrt(sum); //sqrt(sum^2)
    }
}

//matrix class, a data matrix should be constructed with the input data, row num and col num
class Matrix {
private:
    std::vector<double> data_;  //flattened storage (row-major)
    size_t rows_;
    size_t cols_;
    
public:
    //constructors
    Matrix() : rows_(0), cols_(0) {} //0 if not specified
    
    Matrix(size_t rows, size_t cols, double init_val = 0.0) //specified
        : data_(rows * cols, init_val), rows_(rows), cols_(cols) {}
    
    Matrix(size_t rows, size_t cols, const std::vector<double>& values)
        : rows_(rows), cols_(cols) {
        if (values.size() != rows * cols) {
            throw std::invalid_argument("Data size mismatch"); //size check
        }
        data_ = values;
    }
    
    //size accessors - displays row len, col len and row*col len as size of the whole thing (n x m matrix logic)
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    
    //element access: m(i,j) 
    double& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j]; //we treat the 1D vector as 2D using "modulo"
    }
    
    const double& operator()(size_t i, size_t j) const {//this is needed for i,j-th element to be decl-ed as cb
        return data_[i * cols_ + j];
    }
    
    //row access: m[i] returns pointer to row i (allows m[i][j] which is similar to python and the usual stuff)
    double* operator[](size_t i) {
        return &data_[i * cols_];
    }
    
    const double* operator[](size_t i) const { //so that i-th could be cast as const double
        return &data_[i * cols_];
    }
    
    //direct data access (for interop with C functions)
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
    
    //fill with value
    void fill(double val) {
        std::fill(data_.begin(), data_.end(), val);
    }
    
    //resize (destroys content) THIS MIGHT NEED SOME WORK 
    void resize(size_t rows, size_t cols, double init_val = 0.0) {
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows * cols, init_val);
    }
    
    //copy a row to a Vector
    Vector row(size_t i) const {
        Vector result(cols_); //using vector "class"
        for (size_t j = 0; j < cols_; ++j) {
            result[j] = (*this)(i, j);
        }
        return result;
    }
    
    //copy a column to a Vector
    Vector col(size_t j) const { //same as before but for j
        Vector result(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            result[i] = (*this)(i, j);
        }
        return result;
    }
    
    //set a row from a Vector
    void setRow(size_t i, const Vector& v) {
        if (v.size() != cols_) throw std::invalid_argument("Vector size mismatch"); //MEMORY LEAK PREVENTION
        for (size_t j = 0; j < cols_; ++j) {
            (*this)(i, j) = v[j];
        }
    }
    
    // Set a column from a Vector
    void setCol(size_t j, const Vector& v) {
        if (v.size() != rows_) throw std::invalid_argument("Vector size mismatch"); //MEMORY LEAK PREVENTION
        for (size_t i = 0; i < rows_; ++i) {
            (*this)(i, j) = v[i];
        }
    }
    
    //element-wise operations:
    //addition
    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {//only matrices with same ij can be added
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i]; //override, add i,j-th elements
        }
        return result;
    }
    
    Matrix& operator+=(const Matrix& other) {//addition logic for += too
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }
    
    //subtraction - same as addition
    Matrix operator-(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }
    
    Matrix& operator-=(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }
    
    //scalar multiplication - refer to Vector again
    Matrix operator*(double scalar) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }
    
    Matrix& operator*=(double scalar) { //both ways 
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }
    
    //scalar division (basically multiplication)
    Matrix operator/(double scalar) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        return result;
    }
    
    Matrix& operator/=(double scalar) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] /= scalar;
        }
        return *this;
    }
    
    //element-wise multiplication (Hadamard product)
    Matrix multiply(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * other.data_[i]; //A_ij*B_ij the ij element of A is multiplied by ij el of B
        }
        return result;
    }
    
    //utility funcs
    
    //maximum element
    double max() const {
        return *std::max_element(data_.begin(), data_.end()); //scan the whole vector<double>
    }
    
    //minimum element - same logic as max
    double min() const {
        return *std::min_element(data_.begin(), data_.end());
    }
    
    //norm of dat basically (Forbenius)
    double norm() const {
        double sum = 0.0;
        for (double x : data_) sum += x * x;
        return std::sqrt(sum);
    }
    
    //sum of all elements
    double sum() const {
        double total = 0.0;
        for (double x : data_) total += x;
        return total;
    }
    
    //apply func to each element
    template<typename Func>
    Matrix apply(Func f) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = f(data_[i]);
        }
        return result;
    }
    
    //apply function in-place
    template<typename Func>
    void applyInPlace(Func f) {
        for (double& x : data_) {
            x = f(x);
        }
    }
};

//scalar * matrix
inline Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar;
}


//create matrix from function: m(i,j) = f(i,j)
template<typename Func>
Matrix createMatrix(size_t rows, size_t cols, Func f) {
    Matrix m(rows, cols); //n*m matrix containing 0-s at this point
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = f(i, j); //assuming f(i,j) is the func, we declare the ij-th element via f
        }
    }
    return m;
}

//create vector from function: v[i] = f(i) same as before
template<typename Func>
Vector createVector(size_t n, Func f) {
    Vector v(n); //contains 0-s
    for (size_t i = 0; i < n; ++i) {
        v[i] = f(i); //i-th element declared via f
    }
    return v;
}

//linear interpolation between two vectors
inline Vector linspace(double start, double end, size_t n) {
    Vector v(n);
    if (n == 1) {
        v[0] = start;
        return v;
    }
    double step = (end - start) / (n - 1);
    for (size_t i = 0; i < n; ++i) {
        v[i] = start + i * step;
    }
    return v;
}




#endif // MATRIX_H
