#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include "Kokkos_Macros.hpp"
#include "hydro_config.hpp"

//fixed 4D vector for Hydro state Q = [rho, rho*u, rho*v, E]

//the usual vector operations that we will need
namespace VecOps{

  KOKKOS_INLINE_FUNCTION
  Vec4 add(const Vec4& a, const Vec4& b){
    return Vec4{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 sub(const Vec4& a, const Vec4& b){
    return Vec4{a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 mul(double s, const Vec4& v){
    return Vec4{s * v[0], s * v[1], s * v[2], s * v[3]};
  }

  //Vec4 operators moved below the namespace (global, next to Vec4) - see there

  struct Matrix{
    //signed size for the loops below (comparing int with size_t NVAR gave -Wsign-compare everywhere)
    static constexpr int N = static_cast<int>(NVAR);
    Real data[N][N];

    KOKKOS_INLINE_FUNCTION
    Matrix(){
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          data[i][j] = 0.0;
        }
      }
    }

    //explicit, otherwise a plain Real silently converts to val*I (e.g. m + 2.0 == m + 2*I)
    KOKKOS_INLINE_FUNCTION
    explicit Matrix(Real val){
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          data[i][j] = (i==j) ? val : 0.0;
        }
      }
    }

    KOKKOS_INLINE_FUNCTION Real& operator()(int i, int j){ return data[i][j]; }
    KOKKOS_INLINE_FUNCTION Real operator()(int i, int j) const { return data[i][j]; }

    KOKKOS_INLINE_FUNCTION
    Matrix operator+(const Matrix& o) const {
      Matrix res;
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          res.data[i][j] = data[i][j] + o.data[i][j];
        }
      }
      return res;
    }

    KOKKOS_INLINE_FUNCTION
    Matrix operator-(const Matrix& o) const {
      Matrix res;
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          res.data[i][j] = data[i][j] - o.data[i][j];
        }
      }
      return res;
    }

    KOKKOS_INLINE_FUNCTION
    Matrix& operator+=(const Matrix& o){
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          data[i][j] += o.data[i][j];
        }
      }
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Matrix& operator-=(const Matrix& o){
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          data[i][j] -= o.data[i][j];
        }
      }
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Matrix operator*(Real s) const {
      Matrix res;
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          res.data[i][j] = data[i][j]*s;
        }
      }
      return res;
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec4 operator*(const Vec4& v) const {
      Vec4 res {0.0, 0.0, 0.0, 0.0};
      for(int i = 0; i < N; ++i){
        Real sum = 0.0;
        for(int j = 0; j < N; ++j){
          sum += data[i][j]*v[j];
        }
        res[i] = sum;
      }
      return res;
    }

    KOKKOS_INLINE_FUNCTION
    Matrix operator*(const Matrix& o) const {
      Matrix res;
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          Real sum = 0.0;
          for(int k = 0; k < N; ++k){
            sum += data[i][k]*o.data[k][j];
          }
          res.data[i][j] = sum;
        }
      }
      return res;
    }


    KOKKOS_INLINE_FUNCTION
    static Matrix invert_4x4(const Matrix& m){
      Matrix inv;
      Real A[16], invOut[16];
      int idx = 0;
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
          A[idx++] = m(i, j);
        }
      }

      invOut[0] = A[5]*A[10]*A[15]-A[5]*A[11]*A[14]-A[9]*A[6]*A[15]+A[9]*A[7]*A[14]+A[13]*A[6]*A[11]- A[13]*A[7]*A[10];
      invOut[4] = -A[4]*A[10]*A[15]+A[4]*A[11]*A[14]+A[8]*A[6]*A[15]-A[8]*A[7]*A[14]-A[12]*A[6]*A[11]+A[12]*A[7]*A[10];
      invOut[8] = A[4]*A[9]*A[15]-A[4]*A[11]*A[13]-A[8]*A[5]*A[15]+A[8]*A[7]*A[13]+A[12]*A[5]*A[11]-A[12]*A[7]*A[9];
      invOut[12] = -A[4]*A[9]*A[14]+A[4]*A[10]*A[13]+A[8]*A[5]*A[14]-A[8]*A[6]*A[13]-A[12]*A[5]*A[10]+A[12]*A[6]*A[9];
      invOut[1] = -A[1]*A[10]*A[15]+A[1]*A[11]*A[14]+A[9]*A[2]*A[15]-A[9]*A[3]*A[14]-A[13]*A[2]*A[11]+A[13]*A[3]*A[10];
      invOut[5] = A[0]*A[10]*A[15]-A[0]*A[11]*A[14]-A[8]*A[2]*A[15]+A[8]*A[3]*A[14]+A[12]*A[2]*A[11]-A[12]*A[3]*A[10];
      invOut[9] = -A[0]*A[9]*A[15]+A[0]*A[11]*A[13]+A[8]*A[1]*A[15]-A[8]*A[3]*A[13]-A[12]*A[1]*A[11]+A[12]*A[3]*A[9];
      invOut[13] =  A[0]*A[9]*A[14]-A[0]*A[10]*A[13]-A[8]*A[1]*A[14]+A[8]*A[2]*A[13]+A[12]*A[1]*A[10]-A[12]*A[2]*A[9];
      invOut[2] = A[1]*A[6]*A[15]-A[1]*A[7]*A[14]-A[5]*A[2]*A[15]+A[5]*A[3]*A[14]+A[13]*A[2]*A[7]-A[13]*A[3]*A[6];
      invOut[6] = -A[0]*A[6]*A[15]+A[0]*A[7]*A[14]+A[4]*A[2]*A[15]-A[4]*A[3]*A[14]-A[12]*A[2]*A[7]+A[12]*A[3]*A[6];
      invOut[10] = A[0]*A[5]*A[15]-A[0]*A[7]*A[13]-A[4]*A[1]*A[15]+A[4]*A[3]*A[13]+A[12]*A[1]*A[7]-A[12]*A[3]*A[5];
      invOut[14] = -A[0]*A[5]*A[14]+A[0]*A[6]*A[13]+A[4]*A[1]*A[14]-A[4]*A[2]*A[13]-A[12]*A[1]*A[6]+A[12]*A[2]*A[5];
      invOut[3] = -A[1]*A[6]*A[11]+A[1]*A[7]*A[10]+A[5]*A[2]*A[11]-A[5]*A[3]*A[10]-A[9]*A[2]*A[7]+A[9]*A[3]*A[6];
      invOut[7] =  A[0]*A[6]*A[11]-A[0]*A[7]*A[10]-A[4]*A[2]*A[11]+A[4]*A[3]*A[10]+A[8]*A[2]*A[7]-A[8]*A[3]*A[6];
      invOut[11] = -A[0]*A[5]*A[11]+A[0]*A[7]*A[9]+A[4]*A[1]*A[11]-A[4]*A[3]*A[9]-A[8]*A[1]*A[7]+A[8]*A[3]*A[5];
      invOut[15] =  A[0]*A[5]*A[10]-A[0]*A[6]*A[9]-A[4]*A[1]*A[10]+A[4]*A[2]*A[9]+A[8]*A[1]*A[6]-A[8]*A[2]*A[5];

      Real det = A[0]*invOut[0] + A[1]*invOut[4] + A[2]*invOut[8] + A[3]*invOut[12];
      //KOKKOS_IF_ON_HOST / KOKKOS_IF_ON_DEVICE pick the right branch for the code being compiled.
      if(Kokkos::fabs(det) < 1e-20){
        KOKKOS_IF_ON_HOST((throw std::runtime_error("Matrix inversion failed: near zero det");))
        KOKKOS_IF_ON_DEVICE((Kokkos::abort("Matrix inversion failed: near zero det");))
      }

      Real invDet = 1.0/det;
      for(int i = 0; i < N*N; ++i){
        inv.data[i/4][i%4] = invOut[i]*invDet;
      }

      return inv;
    }
  };

  //solves M X = B and M x = b at once (Gaussian elimination with partial pivoting).
  //M is a copy, B and b are overwritten with the solutions.
  //returns false if M is singular (or contains NaN).
  KOKKOS_INLINE_FUNCTION
  bool solve_4x4(Matrix M, Matrix& B, Vec4& b){
    constexpr int N = Matrix::N;
    for(int c = 0; c < N; ++c){
      //pivot: largest entry in column c
      int piv = c;
      Real amax = Kokkos::fabs(M(c, c));
      for(int r = c + 1; r < N; ++r){
        if(Kokkos::fabs(M(r, c)) > amax){ amax = Kokkos::fabs(M(r, c)); piv = r; }
      }
      if(!(amax > 0.0)) return false;
      if(piv != c){
        for(int k = 0; k < N; ++k){
          Real tm = M(c, k); M(c, k) = M(piv, k); M(piv, k) = tm;
          Real tb = B(c, k); B(c, k) = B(piv, k); B(piv, k) = tb;
        }
        Real t = b[c]; b[c] = b[piv]; b[piv] = t;
      }
      //eliminate below the pivot
      for(int r = c + 1; r < N; ++r){
        Real f = M(r, c)/M(c, c);
        if(f == 0.0) continue;
        for(int k = c; k < N; ++k) M(r, k) -= f*M(c, k);
        for(int k = 0; k < N; ++k) B(r, k) -= f*B(c, k);
        b[r] -= f*b[c];
      }
    }
    //back substitution
    for(int r = N - 1; r >= 0; --r){
      for(int k = 0; k < N; ++k){
        Real sum = B(r, k);
        for(int m = r + 1; m < N; ++m) sum -= M(r, m)*B(m, k);
        B(r, k) = sum/M(r, r);
      }
      Real sum = b[r];
      for(int m = r + 1; m < N; ++m) sum -= M(r, m)*b[m];
      b[r] = sum/M(r, r);
    }
    return true;
  }

} // namespace VecOps

//Vec4 lives in the global namespace, so its operators must too: then argument-dependent lookup finds
//them everywhere and headers no longer need a file-scope using namespace VecOps; (which leaked into every includer)
KOKKOS_INLINE_FUNCTION
Vec4 operator+(const Vec4& a, const Vec4& b){ return VecOps::add(a, b); }

KOKKOS_INLINE_FUNCTION
Vec4 operator-(const Vec4& a, const Vec4& b){ return VecOps::sub(a, b); }

KOKKOS_INLINE_FUNCTION
Vec4 operator*(double s, const Vec4& v){ return VecOps::mul(s, v); }
