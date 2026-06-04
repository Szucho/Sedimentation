#ifndef GRID_H
#define GRID_H

#include <vector>
#include <stdexcept>
#include "matrix.h"

/*
 Grid BC application for 2D Grids
 Bertalan Szuchovszky 02.03.2026

 Cell: cell point struct, at each cell point we have a Q state vector, Cell[0] = rho, Cell[1] = rho u ...
 Grid: the whole grid, every element of the grid is a cell Grid(i,j)[0] = rho, Grid(i,j)[1] = rho u ...
 BCType: Boundary condition type - just names, the solver just needs the names
 BoundaryCondition: sets the BCType, for Dirichlet we need Cell Q_fixed constant state vector
 GridBC: BC type at each grid wall

 The other 2 functions are defined in grid_setup.cpp
 

 MODIFIED 12.05.2026
 -Grid is i*j*k 1D Vector, cell is a temporary value holder.
 -Added some helper functions into Grid so that the new cell can be accessed.
*/



//at each cell we have a 4D state vector Q
static constexpr size_t NVAR = 4; //# of primitive variables

//this is just a temporary value type
struct Cell {
  double q[NVAR] = {};
  double& operator[](int c)       { return q[c]; }
  double  operator[](int c) const { return q[c]; }
};

class Grid {
private:
  std::vector<double> data_;
  size_t rows_, cols_;

public:
  Grid(size_t rows, size_t cols)
    : data_(rows * cols * NVAR, 0.0), rows_(rows), cols_(cols) {}

  //access a single primitive value with index k at cell (i,j)
  double& operator()(size_t i, size_t j, size_t k) {
    return data_[(i * cols_ + j) * NVAR + k];
  }
  const double& operator()(size_t i, size_t j, size_t k) const {
    return data_[(i * cols_ + j) * NVAR + k];
  }

  //access a pointer to the start of cell (i,j) variables
  double* cell(size_t i, size_t j) {
    return data_.data() + (i * cols_ + j) * NVAR;
  }
  const double* cell(size_t i, size_t j) const {
    return data_.data() + (i * cols_ + j) * NVAR;
  }

  Cell getCell(size_t i, size_t j) const {
    Cell c;
    const double* q = data_.data() + (i * cols_ + j) * NVAR;
    c[0]=q[0]; c[1]=q[1]; c[2]=q[2]; c[3]=q[3];
    return c;
  }
  void setCell(size_t i, size_t j, const Cell& c) {
    double* q = data_.data() + (i * cols_ + j) * NVAR;
    q[0]=c[0]; q[1]=c[1]; q[2]=c[2]; q[3]=c[3];
  }

  void copyCell(size_t i_dst, size_t j_dst, size_t i_src, size_t j_src) {
    const double* src = data_.data() + (i_src * cols_ + j_src) * NVAR;
    double*       dst = data_.data() + (i_dst * cols_ + j_dst) * NVAR;
    for (size_t k = 0; k < NVAR; ++k) dst[k] = src[k];
  }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t nvar() const { return NVAR; }
};


enum class BCType {Open, Closed, Periodic, Dirichlet}; //BC types available

struct BoundaryCondition {
  BCType type;
  Cell Q_fixed;  //Dirichlet
};

struct GridBC {
  BoundaryCondition left, right, top, bottom; //BC on all walls
};



inline Vector CellToVec(const Cell& c) {
  return {c[0], c[1], c[2], c[3]};
}
inline Cell VecToCell(const Vector& v) {
  Cell c; c[0]=v[0]; c[1]=v[1]; c[2]=v[2]; c[3]=v[3];
  return c;
}


//defined in grid_setup.cpp
void validateBC(const GridBC& bc); //validateBC checks if Periodic BC was set correctly
// void applyBC(Grid& grid, const GridBC& bc); //applies BC type at chosen wall
void applyBC(Grid& grid, const GridBC& bc, double Omega, double cs2, double gamma, double dz, double zmin);

#endif // GRID_H
