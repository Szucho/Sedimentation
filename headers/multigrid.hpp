#pragma once
#include <vector>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include "hydro_config.hpp"
#include "mesh.hpp"

/*
 Multigrid-preconditioned CG for the implicit (backward Euler) heat conduction step

   A x = b,   A x|_i = C_i x_i + sum_f w_f (x_i - x_nb),   w_f = kappa_f * (shared face length)/(centre distance)

 in integrated (per cell) form: C_i = rho c_v V_i/dt, kappa_f = arithmetic mean of the two cells (like Athena++).
 A is symmetric positive definite, solved with CG preconditioned by ONE multigrid V-cycle per iteration.

 The multigrid follows Athena++ (Tomida & Stone 2023, ApJS 266, 7; src/multigrid/ and the variable coefficient
 smoother/matrix of src/crdiffusion/mg_crdiffusion.cpp):
  - hierarchy: AMR leaf cells -> refined regions coarsened level by level (Athena++: block levels + octets, here
    the 2D quadtree analogue) -> uniform root grid -> root grid halved until it can not be reduced (coarsest level)
  - coefficients restricted by volume average (kappa) and the operator rediscretised on every level
  - smoother "jacobi-rb": colour-wise Jacobi in two passes (all cells of one colour from the old neighbour values,
    then relaxed with omega), colours by index parity
  - V(1,1) cycle, coarsest level smoothed a fixed number of times instead of an exact solve
 Deviations from Athena++, all needed because the V-cycle is a CG preconditioner (must be a FIXED SYMMETRIC operator):
  - prolongation is piecewise constant with restriction = its transpose (sum of the children's defects, the
    integrated form of Athena++'s volume average); Athena++ uses tri-linear/tri-cubic prolongation, whose
    transpose is not the volume average
  - the colour order is not flipped between cycles (Athena++ flips coffset_ every V-cycle), post-smoothing runs
    the colours in reverse order of the pre-smoothing
  - no average subtraction (Athena++ subtracts it for the singular all-periodic problem; A here is regular, C > 0)
 Levels are general cell graphs (CSR), so composite AMR levels and fine/coarse faces need no special cases, and
 every kernel is a flat loop over cells -> GPU friendly.
*/

using IdxView = Kokkos::View<int*, MemorySpace>;

struct MGLevel {
  int n = 0;
  IdxView row_ptr, col;     //CSR adjacency (neighbours across the cell faces)
  View1D g;                 //geometric factor of the face: shared length/centre distance
  View1D dist;              //centre distance across the face (leaf level only, for gradients)
  View1D W;                 //face weight kappa_f*g (set by set_coefficients)
  View1D C, kap, vol, D;    //time term (integrated), conductivity, cell volume, diagonal C + sum W
  IdxView parent;           //cell of the next coarser level containing this cell (-1 on the coarsest)
  IdxView child_ptr, child; //CSR list of the children on the next finer level (empty on level 0)
  IdxView color_cells[2];   //cells of each colour
  View1D x, b, work;        //solution, source, smoother scratch
};

class MGConduction {
public:
  //rebuilds the hierarchy if the mesh changed since the last call (block layout signature)
  void setup(Mesh& mesh);

  int n_leaf() const { return lv_.empty() ? 0 : lv_[0].n; }
  int n_levels() const { return (int)lv_.size(); }
  //leaf cell index of active cell (i,j) of block slot b
  KOKKOS_INLINE_FUNCTION static int leaf_index(int b, int i, int j, int is, int js, int nx1, int nx2){
    return b*nx1*nx2 + (i - is)*nx2 + (j - js);
  }
  //leaf coefficients to fill before set_coefficients(): C (integrated time term) and kappa
  View1D leaf_C()   { return lv_[0].C; }
  View1D leaf_kap() { return lv_[0].kap; }
  //restricts the coefficients and builds the face weights / diagonals on every level
  void set_coefficients();
  //grad(c) = max over the faces of leaf cell c of |v(c) - v(nb)|/(centre distance), a |grad v| estimate
  void max_face_gradient(const View1D& v, const View1D& grad);

  //y = A x on the leaf level; with_C = false gives the pure conduction operator L x
  void apply(const View1D& x, const View1D& y, bool with_C);
  //z = M^{-1} r: one symmetric V-cycle (copies r in and z out; solve() works in place instead)
  void precondition(const View1D& r, const View1D& z);
  //PCG for A x = b (x = 0 start); converged when |r| <= tol*|b| (L2) or when the last correction of x is below
  //x_tol everywhere (max norm, in the units of x). The second test is needed when the diagonal ratio w/C is so large
  //that A x can not resolve x in double precision: CG then stalls on a b that is only round-off of a
  //uniform state. Returns the number of iterations.
  int solve(const View1D& b, const View1D& x, Real tol, int max_iter, bool& converged, Real x_tol = 0.0);

  //smoothing parameters (Athena++ defaults: npresmooth_ = npostsmooth_ = 1, omega = 1)
  int npre = 1, npost = 1;
  Real omega = 1.0;
  int ncoarsest = 64; //smoothing sweeps (colour pairs) on the coarsest level, forward and then in reverse order
  int min_coarse_cells = 64;
  //levels with at most this many cells are cycled inside ONE team kernel (small_vcycle): on a GPU a kernel launch
  //costs more than smoothing a few thousand cells, and the coarse levels would otherwise need ~10 launches each
  int small_level_cells = 2048;
  //(public: CUDA extended lambdas can not live in private member functions)
  void smooth(int l, int color);
  void defect_restrict(int l);
  void vcycle(int l);
  void small_vcycle();
  Real apply_dot(const View1D& x, const View1D& y);
private:
  std::vector<MGLevel> lv_;
  std::vector<std::int64_t> signature_;
  View1D cg_p_, cg_Ap_;

  //the small levels ls_ .. L-1 stored back to back ("flat"), so that one kernel can walk all of them: global cell
  //index = level offset + local index. W, D, x, b, work of these levels are views into the flat arrays.
  static constexpr int kMaxSmall = 32;
  int ls_ = 0, nsmall_ = 0;
  Kokkos::Array<int, kMaxSmall + 1> s_cell_off_;      //first flat cell of each small level
  Kokkos::Array<int, 2*kMaxSmall + 1> s_color_off_;   //first entry of each (level, colour) in s_colcells_
  IdxView s_rb_, s_re_, s_col_;                       //rows (flat edge ranges) and neighbours (flat cell index)
  IdxView s_parent_, s_cb_, s_ce_, s_child_, s_colcells_;
  View1D s_W_, s_D_, s_x_, s_b_, s_work_;
  void build_small_levels();
};
