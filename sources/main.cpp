#include <cmath>
#include <ios>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <sstream>
#include <algorithm>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#include <fstream>
#include <string>
#include <Kokkos_Core.hpp>
#include "hydro_config.hpp"
#include "grid.hpp"
#include "mesh.hpp"
#include "hllc.hpp"
#include "slope_limiters.hpp"
#include "hydro_io.hpp"
#include "particle.hpp"
#include "multigrid.hpp"

using VecOps::Matrix;
constexpr Real k_B = 1.380649e-16;  //erg/K
constexpr Real PI = 3.141592653589793;
constexpr Real m_p = 1.672622e-24;  //g

/*
 2D Hydrodynamics HLLC Euler Equation solver on a block-structured AMR mesh
 Bertalan Szuchovszky 03.03.2026

 Solves the 2D compressible Euler equations in conservative form
 on a quadtree AMR mesh (see mesh.hpp) using the HLLC approximate Riemann solver (see hllc.hpp)
 with minmod slope-limited interpolation at cell walls (see slope_limiters.hpp).

 Equation to be solved: dQ/dt + div F(Q) = S(Q) + div(kappa grad T)
 Heat conduction: operator split, after the hydro step. Backward Euler in T with kappa frozen, solved over the whole
 mesh with multigrid-preconditioned CG (see conduction_step, multigrid.hpp)
 State vector: Q = [rho, rho*u, rho*v, rho*e_tot]
               e_tot = e_th + 0.5*(u^2+v^2),  e_th = 1/(gamma-1) * k_BT/(mu*m_p) - ideal gas law

 Spatial discretization: 2nd order finite volume (~MUSCL-Hancock, approximate Riemann solver)
 Time integration: implicit B2 scheme (Batten et al. 1997, eqs. 70-72), two approximately factored
                   backward-Euler (B1) solves per step: dt/2, then dt from the half-step state.
 Boundary conditions: Open, Closed, Periodic, Dirichlet (user specified)
 Open BC obeys hydrostatic equilibrium (numerical) in z
 Well balanced method allows hydrostatic equilibrium gas initial condition (Käppeli & Mishra 2016)

 Modified October 2026:
 - ported the B2 step of the uniform-grid code onto the Kokkos AMR mesh. Every block solves its own
   block-tridiagonal lines (z-sweep then r-sweep), the implicit operator is truncated at block
   boundaries (neighbour dU treated as 0, like the old patch solves), the explicit residual sees the
   neighbours through the ghost zones. All blocks use the same dt (no subcycling), as in the explicit AMR code.
 - the particle refines the mesh around itself instead of carrying its own fine patch
*/


//ideal gas heat conduction coeff, f=5 for H2
//kappa = f/(3d^2) * sqrt(k_B^3 T/(pi^3 mu*m_p))
KOKKOS_INLINE_FUNCTION
Real kappa(Real mu, Real d, Real T){
  return 5.0/(3.0*d*d)*Kokkos::sqrt(k_B*k_B*k_B*T/(PI*PI*PI*mu*m_p));
}

using View5D = Kokkos::View<Real*****, Kokkos::LayoutRight, MemorySpace>;

struct Physics {
  Real gamma, Omega, mu;
  Real d; //effective cross section for the heat conduction
  Real cond_phi; //heat flux saturation q_sat = phi rho c_s^3 (phi <= 0: no limiter)
  Real cs2; //isothermal sound speed^2 of the hydrostatic open BC
};


//everything the hydro kernels need to read one block's state: data + block geometry
//copied by value into the Kokkos lambdas, so only Views and scalars live here
struct HydroOps {
  View4D q;
  View1D x1min, dx1, dx2;
  IntView2D nlevel;
  int is, ie, js, je;
  Real gamma, Omega, mu, d;
  Real z0; //absolute z of the local origin (cell coordinates are local)

  KOKKOS_INLINE_FUNCTION Vec4 Q(int b, int i, int j) const {
    return Vec4{q(b, i, j, 0), q(b, i, j, 1), q(b, i, j, 2), q(b, i, j, 3)};
  }

  KOKKOS_INLINE_FUNCTION Real pressure(int b, int i, int j) const {
    Real rho = q(b, i, j, 0);
    return (gamma - 1.0)*(q(b, i, j, 3) - 0.5*(q(b, i, j, 1)*q(b, i, j, 1) + q(b, i, j, 2)*q(b, i, j, 2))/rho);
  }

  KOKKOS_INLINE_FUNCTION Real temperature(int b, int i, int j) const {
    return pressure(b, i, j)*m_p*mu/(q(b, i, j, 0)*k_B); //p = rho/(mu*m_p)*k_B*T
  }

  //local cell centre z coordinate (also valid for ghost cells) and the potential difference phi(z_a) - phi(z_b),
  //phi(z) = 1/2 Omega^2 (z0 + z)^2 (see phi_diff)
  KOKKOS_INLINE_FUNCTION Real z(int b, int i) const { return x1min(b) + (i - is + 0.5)*dx1(b); }
  KOKKOS_INLINE_FUNCTION Real dphi(Real za, Real zb) const { return phi_diff(za, zb, z0, Omega); }

  KOKKOS_INLINE_FUNCTION bool physical(int b, FaceDir f) const {
    return nlevel(b, (int)f) == (int)NeighborLevel::PhysicalBoundary;
  }

  //minmod slopes, zero in the edge cells next to (and in the ghosts beyond) a physical wall
  //Slopes belong to the cell, so both cells of a face see the same interface states -> unique face flux
  KOKKOS_INLINE_FUNCTION Vec4 slope_x(int b, int i, int j) const {
    if((i <= is && physical(b, FaceDir::XM)) || (i >= ie && physical(b, FaceDir::XP))) return Vec4{0.0, 0.0, 0.0, 0.0};
    return sigma_minmod(Q(b, i - 1, j), Q(b, i, j), Q(b, i + 1, j), dx1(b));
  }
  KOKKOS_INLINE_FUNCTION Vec4 slope_y(int b, int i, int j) const {
    if((j <= js && physical(b, FaceDir::YM)) || (j >= je && physical(b, FaceDir::YP))) return Vec4{0.0, 0.0, 0.0, 0.0};
    return sigma_minmod(Q(b, i, j - 1), Q(b, i, j), Q(b, i, j + 1), dx2(b));
  }

  //slope-limited states at face i+1/2 (x) and j+1/2 (y)
  KOKKOS_INLINE_FUNCTION void states_x(int b, int i, int j, Vec4& QL, Vec4& QR) const {
    Real dz = dx1(b);
    QL = Q(b, i, j)     + (0.5*dz)*slope_x(b, i, j);
    QR = Q(b, i + 1, j) - (0.5*dz)*slope_x(b, i + 1, j);
  }
  KOKKOS_INLINE_FUNCTION void states_y(int b, int i, int j, Vec4& QL, Vec4& QR) const {
    Real dr = dx2(b);
    QL = Q(b, i, j)     + (0.5*dr)*slope_y(b, i, j);
    QR = Q(b, i, j + 1) - (0.5*dr)*slope_y(b, i, j + 1);
  }

  //well balanced pressure of cell m extrapolated to its face m+s/2 (s = +1 or -1), Käppeli et al. 2016:
  //p = p0 + p1, p0 the discrete hydrostatic profile anchored at cell m (p1 = 0 at m),
  //p1 reconstructed with a minmod slope from the neighbours' deviations from p0
  KOKKOS_INLINE_FUNCTION Real wb_face_pressure(int b, int m, int j, int s) const {
    Real dz     = dx1(b);
    Real rho_m  = q(b, m, j, 0);
    Real p_m    = pressure(b, m, j);
    Real z_m    = z(b, m);
    Real dphi_mm = dphi(z(b, m - 1), z_m); //phi_{m-1} - phi_m
    Real dphi_mp = dphi(z(b, m + 1), z_m); //phi_{m+1} - phi_m

    Real p1_mm = pressure(b, m - 1, j) - (p_m - 0.5*(rho_m + q(b, m - 1, j, 0))*dphi_mm);
    Real p1_mp = pressure(b, m + 1, j) - (p_m - 0.5*(rho_m + q(b, m + 1, j, 0))*dphi_mp);
    Real sigma_p1 = minmod_one(p1_mm, 0.0, p1_mp, dz);

    Real dphi_s = (s > 0) ? dphi_mp : dphi_mm;
    return p_m - 0.5*rho_m*dphi_s + 0.5*s*dz*sigma_p1;
  }

  KOKKOS_INLINE_FUNCTION void set_pressure(Vec4& Qf, Real p) const {
    Qf[3] = p/(gamma - 1.0) + 0.5*(Qf[1]*Qf[1] + Qf[2]*Qf[2])/Qf[0];
  }

  //HLLC fluxes at face i+1/2 (with the well balanced energy) and j+1/2
  KOKKOS_INLINE_FUNCTION Vec4 flux_x(int b, int i, int j) const {
    Vec4 QL, QR;
    states_x(b, i, j, QL, QR);
    set_pressure(QL, wb_face_pressure(b, i,     j, +1));
    set_pressure(QR, wb_face_pressure(b, i + 1, j, -1));
    return FluxhllcX(QL, QR, gamma);
  }
  KOKKOS_INLINE_FUNCTION Vec4 flux_y(int b, int i, int j) const {
    Vec4 QL, QR;
    states_y(b, i, j, QL, QR);
    return FluxhllcY(QL, QR, gamma);
  }

  //flux Jacobians at face i+1/2 / j+1/2. 
  //The well-balanced energy correction is omitted from the Jacobian states:
  //it is already in R (the explicit RHS), the Jacobian uses the plain slope-limited states
  KOKKOS_INLINE_FUNCTION HLLCJac jac_x(int b, int i, int j) const {
    Vec4 QL, QR;
    states_x(b, i, j, QL, QR);
    return hllc_jacobian_x(QL, QR, gamma);
  }
  KOKKOS_INLINE_FUNCTION HLLCJac jac_y(int b, int i, int j) const {
    Vec4 QL, QR;
    states_y(b, i, j, QL, QR);
    return hllc_jacobian_y(QL, QR, gamma);
  }

  //gravity source S = (0, -rho dphi/dz, 0, -rho u dphi/dz), centred difference of phi (well balanced with flux_x)
  KOKKOS_INLINE_FUNCTION Real grav_g(int b, int i) const {
    return dphi(z(b, i + 1), z(b, i - 1))/(2.0*dx1(b));
  }
};


//scratch memory of the implicit solver (resized when the number of active blocks grows)
struct ImplicitWork {
  View4D R, dW, dU;  //residual, z-sweep result, r-sweep result
  View4D flux1;      //block-face fluxes of the first B1 solve (for the reflux)
  View5D Cp;         //block-Thomas D'^{-1} C blocks, one per line cell
  View1D x1min;      //lower z edge of every block
  View1D::host_mirror_type h_x1min; //its host copy (persistent)
  bool x1min_valid = false;
  int capacity = 0;

  void ensure(Mesh& mesh){
    int n_active = mesh.num_active_blocks();
    const MeshInputs& in = mesh.inputs();
    int ni = in.nx1_block + 2*in.nghost;
    int nj = in.nx2_block + 2*in.nghost;
    int lmax = std::max(in.nx1_block, in.nx2_block);

    if(n_active > capacity){
      capacity = n_active;
      R     = View4D("implicit_R",  capacity, ni, nj, NVAR);
      dW    = View4D("implicit_dW", capacity, ni, nj, NVAR);
      dU    = View4D("implicit_dU", capacity, ni, nj, NVAR);
      Cp    = View5D("implicit_Cp", capacity, lmax, lmax, NVAR, NVAR);
      x1min = View1D("block_x1min", capacity);
      h_x1min = Kokkos::create_mirror_view(x1min);
      x1min_valid = false;
      View4D f = mesh.get_flux_view();
      flux1 = View4D("implicit_flux1", f.extent(0), f.extent(1), f.extent(2), f.extent(3));
    }

    //block geometry changes with AMR updates: compare on the host, upload only if something changed
    //(a host->device copy every step is a synchronisation on a GPU)
    bool changed = !x1min_valid;
    for(const auto& bptr : mesh.blocks()){
      if(h_x1min(bptr->slot_id) != bptr->block_size.x1min){
        h_x1min(bptr->slot_id) = bptr->block_size.x1min;
        changed = true;
      }
    }
    if(changed) Kokkos::deep_copy(x1min, h_x1min);
    x1min_valid = true;
  }
};


HydroOps make_ops(Mesh& mesh, const ImplicitWork& w, const Physics& ph){
  HydroOps ops;
  ops.q      = mesh.get_data_view();
  ops.x1min  = w.x1min;
  ops.dx1    = mesh.get_block_dx1_view();
  ops.dx2    = mesh.get_block_dx2_view();
  ops.nlevel = mesh.get_neighbor_level_view();
  int ng = mesh.inputs().nghost;
  ops.is = ng;  ops.ie = ng + mesh.inputs().nx1_block - 1;
  ops.js = ng;  ops.je = ng + mesh.inputs().nx2_block - 1;
  ops.gamma = ph.gamma;  ops.Omega = ph.Omega;  ops.mu = ph.mu;  ops.d = ph.d;
  ops.z0 = mesh.inputs().x1_origin;
  return ops;
}


//R(Q) = -div F + S + div(kappa grad T) on every active cell (= explicit time derivative, without dt)
//The fluxes on the block faces are stored in flux_out for the fine/coarse reflux.
void compute_residual(int n_active, const HydroOps& ops, const View4D& R, const View4D& flux_out){
  int is = ops.is, ie = ops.ie, js = ops.js, je = ops.je;

  Kokkos::parallel_for(
    "ComputeResidual",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      Real dz = ops.dx1(b);
      Real dr = ops.dx2(b);

      Vec4 FZimh = ops.flux_x(b, i - 1, j);
      Vec4 FZiph = ops.flux_x(b, i,     j);
      Vec4 FRjmh = ops.flux_y(b, i, j - 1);
      Vec4 FRjph = ops.flux_y(b, i, j);

      if(i == is) for(size_t k = 0; k < NVAR; ++k) flux_out(b, (int)FaceDir::XM, j - js, k) = FZimh[k];
      if(i == ie) for(size_t k = 0; k < NVAR; ++k) flux_out(b, (int)FaceDir::XP, j - js, k) = FZiph[k];
      if(j == js) for(size_t k = 0; k < NVAR; ++k) flux_out(b, (int)FaceDir::YM, i - is, k) = FRjmh[k];
      if(j == je) for(size_t k = 0; k < NVAR; ++k) flux_out(b, (int)FaceDir::YP, i - is, k) = FRjph[k];

      //flux divergence + gravity
      Vec4 Rij = (-1.0/dz)*(FZiph - FZimh) - (1.0/dr)*(FRjph - FRjmh);
      Real G = ops.grav_g(b, i);
      Rij[1] -= ops.q(b, i, j, 0)*G;
      Rij[3] -= ops.q(b, i, j, 1)*G;


      for(size_t k = 0; k < NVAR; ++k) R(b, i, j, k) = Rij[k];
    }
  );
}


//fine/coarse flux correction of the residual (same as ApplyReflux in mesh.cpp, but on R instead of on Q, so no dt):
//a coarse cell next to a finer block uses the average of the two fine face fluxes instead of its own face flux.
//doing this inside every B1 solve (instead of an ApplyReflux after the step) keeps the explicit part exactly
//conservative and puts the correction in front of the implicit operator
//NOTE an explicit dt*dF/dx correction after an implicit step at Courant numbers of 10^3+ is unstable.
void reflux_residual(Mesh& mesh, const View4D& R, const View4D& flux){
  int n_active = mesh.num_active_blocks();
  IntView2D n_gid   = mesh.get_neighbor_gid_view();
  IntView2D n_level = mesh.get_neighbor_level_view();
  IntView2D n_gid2  = mesh.get_neighbor_gid2_view();
  View1D dx1 = mesh.get_block_dx1_view();
  View1D dx2 = mesh.get_block_dx2_view();
  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int ng  = mesh.inputs().nghost;
  int is = ng, ie = ng + nx1 - 1;
  int js = ng, je = ng + nx2 - 1;

  Kokkos::parallel_for(
    "RefluxResidual",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, 4, std::max(nx1, nx2)}),
    KOKKOS_LAMBDA(const int a, const int fi, const int t){
      if(n_level(a, fi) != (int)NeighborLevel::Finer) return;
      FaceDir face = static_cast<FaceDir>(fi);
      bool xface = (face == FaceDir::XM || face == FaceDir::XP);
      int N = xface ? nx2 : nx1;
      if(t >= N) return;

      //the two fine faces covering coarse face t, in the fine block's opposite face register
      int half_N = N/2;
      int fb  = (t/half_N == 0) ? n_gid(a, fi) : n_gid2(a, fi);
      int tb  = 2*(t%half_N);
      int opp = xface ? (fi == (int)FaceDir::XP ? (int)FaceDir::XM : (int)FaceDir::XP)
                      : (fi == (int)FaceDir::YP ? (int)FaceDir::YM : (int)FaceDir::YP);

      Real d    = xface ? dx1(a) : dx2(a);
      Real sign = (face == FaceDir::XP || face == FaceDir::YP) ? 1.0 : -1.0;
      int ai = (face == FaceDir::XP) ? ie : (face == FaceDir::XM) ? is : is + t;
      int aj = (face == FaceDir::YP) ? je : (face == FaceDir::YM) ? js : js + t;

      for(size_t k = 0; k < NVAR; ++k){
        Real f_fine = 0.5*(flux(fb, opp, tb, k) + flux(fb, opp, tb + 1, k));
        //atomic: a corner coarse cell with finer neighbours on two faces is updated by two threads
        Kokkos::atomic_add(&R(a, ai, aj, k), sign*(flux(a, fi, t, k) - f_fine)/d);
      }
    }
  );
}


//one block-tridiagonal sweep along every line of every block (block-Thomas algorithm)
//dir = 0: z-sweep, lines of constant j, solves (I/tau - dR/dU|_z) out = in_scale*in
//dir = 1: r-sweep, lines of constant i, solves (I/tau - dR/dU|_r) out = in_scale*in
//the line ends at the block boundary: the ghost dU is treated as 0 there (A[0] = C[N-1] = 0),
//both at block-block faces and at physical walls
void line_solve(int n_active, const HydroOps& ops, int dir, Real tau,
                const View4D& in, Real in_scale, const View4D& out, const View5D& Cp){
  int N = (dir == 0) ? ops.ie - ops.is + 1 : ops.je - ops.js + 1; //cells per line
  int L = (dir == 0) ? ops.je - ops.js + 1 : ops.ie - ops.is + 1; //lines per block

  Kokkos::parallel_for(
    "ImplicitLineSolve",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n_active, L}),
    KOKKOS_LAMBDA(const int b, const int l){
      Real h = (dir == 0) ? ops.dx1(b) : ops.dx2(b);
      auto cell_i = [&](int s){ return (dir == 0) ? ops.is + s : ops.is + l; };
      auto cell_j = [&](int s){ return (dir == 0) ? ops.js + l : ops.js + s; };
      //Jacobian of the face between line cells s and s+1
      auto jac = [&](int s){
        return (dir == 0) ? ops.jac_x(b, cell_i(s), cell_j(s)) : ops.jac_y(b, cell_i(s), cell_j(s));
      };

      HLLCJac Jm = jac(-1);  //face s-1/2
      Matrix C_prev;
      Vec4 y_prev{0.0, 0.0, 0.0, 0.0};

      //forward elimination: D'_s = D_s - A_s D'^{-1}_{s-1} C_{s-1}, stored as Cp_s = D'^{-1}_s C_s
      for(int s = 0; s < N; ++s){
        int i = cell_i(s), j = cell_j(s);
        HLLCJac Jp = jac(s); //face s+1/2

        //diagonal: I/tau + (1/h) J_{s+1/2} dF_dUl - (1/h) J_{s-1/2} dF_dUr
        Matrix D(1.0/tau);
        D += Jp.dF_dUl*(1.0/h);
        D -= Jm.dF_dUr*(1.0/h);
        if(dir == 0){
          //gravity source subtracted from D: dS/dU has (1,0) = -G and (3,1) = -G
          Real G = ops.grav_g(b, i);
          D(1, 0) += G;
          D(3, 1) += G;
        }
        //upper block: C_s = (1/h) J_{s+1/2} dF_dUr, zero at the block edge
        Matrix C = (s < N - 1) ? Jp.dF_dUr*(1.0/h) : Matrix();

        Vec4 r;
        for(size_t k = 0; k < NVAR; ++k) r[k] = in_scale*in(b, i, j, k);

        if(s > 0){
          //lower block: A_s = -(1/h) J_{s-1/2} dF_dUl, zero at the block edge (s = 0)
          Matrix A = Jm.dF_dUl*(-1.0/h);
          D -= A*C_prev;
          r = r - A*y_prev;
        }

        if(!VecOps::solve_4x4(D, C, r)){
          KOKKOS_IF_ON_HOST((throw std::runtime_error("line_solve: singular diagonal block");))
          KOKKOS_IF_ON_DEVICE((Kokkos::abort("line_solve: singular diagonal block");))
        }

        for(int m = 0; m < 4; ++m){
          for(int k = 0; k < 4; ++k) Cp(b, l, s, m, k) = C(m, k);
          out(b, i, j, m) = r[m];
        }
        C_prev = C;
        y_prev = r;
        Jm = Jp;
      }

      //back substitution: x_s = y_s - Cp_s x_{s+1}
      for(int s = N - 2; s >= 0; --s){
        int i = cell_i(s), j = cell_j(s);
        int in1 = cell_i(s + 1), jn1 = cell_j(s + 1);
        for(int m = 0; m < 4; ++m){
          Real sum = 0.0;
          for(int k = 0; k < 4; ++k) sum += Cp(b, l, s, m, k)*out(b, in1, jn1, k);
          out(b, i, j, m) -= sum;
        }
      }
    }
  );
}


//1 backward-Euler implicit step (Batten et al. 1997), increment only: w.dU = U^{n+1} - U^n
//approximate factorisation: z-sweep then r-sweep, each a block-tridiagonal solve.
//(I/tau - J_z) dW = R,   (I/tau - J_r) dU = dW/tau
void implicit_B1(Mesh& mesh, const HydroOps& ops, ImplicitWork& w, const View4D& flux_out, Real tau){
  int n_active = mesh.num_active_blocks();
  compute_residual(n_active, ops, w.R, flux_out);
  reflux_residual(mesh, w.R, flux_out);
  line_solve(n_active, ops, 0, tau, w.R,  1.0,     w.dW, w.Cp);
  line_solve(n_active, ops, 1, tau, w.dW, 1.0/tau, w.dU, w.Cp);
}


//interior update q += fac*dU on every active block
void add_increment(int n_active, const HydroOps& ops, const View4D& dU, Real fac){
  View4D q = ops.q;
  Kokkos::parallel_for(
    "AddIncrement",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ops.is, ops.js}, {n_active, ops.ie + 1, ops.je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j){
      for(size_t k = 0; k < NVAR; ++k) q(b, i, j, k) += fac*dU(b, i, j, k);
    }
  );
}


//B2 scheme eqs. (70)-(72) Batten et al. 1997
//2 backward-Euler steps: dt/2 then dt, twice the convergence rate of B1.
//  U_bar   = U^n + dU1,       (I/(dt/2) - J(U^n)) dU1 = R(U^n)
//  U^{n+1} = U_bar + dU2/2,   (I/dt - J(U_bar))   dU2 = R(U_bar)
//Fine/coarse fluxes are corrected inside R (reflux_residual), so no ApplyReflux is needed afterwards.
//On return the mesh flux view holds the time-averaged block-face fluxes 0.5*(F(U^n) + F(U_bar)), which is
//exactly the flux of the update in the explicit limit (J -> 0): U^{n+1} = U^n + dt/2 (R(U^n) + R(U_bar)).
//Ghost zones of U^{n+1} are NOT refreshed here.
void implicit_step_B2(Mesh& mesh, const GridBC& bc, ImplicitWork& w, const Physics& ph, Real dt){
  int n_active = mesh.num_active_blocks();
  if(n_active == 0) return;

  w.ensure(mesh);
  HydroOps ops = make_ops(mesh, w, ph);
  View4D flux2 = mesh.get_flux_view();

  //half step
  implicit_B1(mesh, ops, w, w.flux1, 0.5*dt);
  add_increment(n_active, ops, w.dU, 1.0);
  ExchangeGhostZones(mesh);
  ApplyPhysicalBoundaryConditions(mesh, bc, ph.Omega, ph.cs2, ph.gamma);

  //full step from U_bar
  implicit_B1(mesh, ops, w, flux2, dt);
  add_increment(n_active, ops, w.dU, 0.5);

  //time averaged block-face fluxes for ApplyReflux
  View4D flux1 = w.flux1;
  Kokkos::parallel_for(
    "AverageFaceFluxes",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, 4, (int)flux2.extent(2)}),
    KOKKOS_LAMBDA(const int b, const int f, const int t){
      for(size_t k = 0; k < NVAR; ++k) flux2(b, f, t, k) = 0.5*(flux1(b, f, t, k) + flux2(b, f, t, k));
    }
  );
}


//CFL time step (only informative for the implicit scheme: dt/dt_cfl is the Courant number)
Real compute_cfl_dt(Mesh& mesh, Real gamma, Real cfl_max){
  int n_active = mesh.num_active_blocks();
  if(n_active == 0) return 1.0;

  View4D data = mesh.get_data_view();
  View1D dx1 = mesh.get_block_dx1_view();
  View1D dx2 = mesh.get_block_dx2_view();
  int ng = mesh.inputs().nghost;
  int ie = ng + mesh.inputs().nx1_block - 1;
  int je = ng + mesh.inputs().nx2_block - 1;

  Real max_wave_speed = 0.0;
  Kokkos::parallel_reduce(
    "ComputeCFL",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& local_max){
      Real rho = data(b, i, j, 0);
      Real u = data(b, i, j, 1)/rho;
      Real v = data(b, i, j, 2)/rho;
      Real p = (gamma - 1.0)*(data(b, i, j, 3) - rho*0.5*(u*u + v*v));
      Real cs = Kokkos::sqrt(gamma*p/rho);
      Real speed = (Kokkos::fabs(u) + cs)/dx1(b) + (Kokkos::fabs(v) + cs)/dx2(b);
      if(speed > local_max) local_max = speed;
    },
    Kokkos::Max<Real>(max_wave_speed)
  );
  return cfl_max/max_wave_speed;
}


//volume integral of conserved variable k (for the conservation check)
Real compute_total(Mesh& mesh, int k){
  int n_active = mesh.num_active_blocks();
  View4D data = mesh.get_data_view();
  View1D dx1 = mesh.get_block_dx1_view();
  View1D dx2 = mesh.get_block_dx2_view();
  int ng = mesh.inputs().nghost;
  int ie = ng + mesh.inputs().nx1_block - 1;
  int je = ng + mesh.inputs().nx2_block - 1;

  Real total = 0.0;
  Kokkos::parallel_reduce(
    "ComputeTotal",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, ng, ng}, {n_active, ie + 1, je + 1}),
    KOKKOS_LAMBDA(const int b, const int i, const int j, Real& sum){
      sum += data(b, i, j, k)*dx1(b)*dx2(b);
    },
    total
  );
  return total;
}


//net outflow rate of conserved variable k through the physical domain walls,
//from the block-face fluxes that implicit_step_B2 leaves in the mesh flux view
Real boundary_outflow_rate(Mesh& mesh, int k){
  int n_active = mesh.num_active_blocks();
  View4D flux = mesh.get_flux_view();
  IntView2D nlevel = mesh.get_neighbor_level_view();
  View1D dx1 = mesh.get_block_dx1_view();
  View1D dx2 = mesh.get_block_dx2_view();
  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;

  Real rate = 0.0;
  Kokkos::parallel_reduce(
    "BoundaryOutflow",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_active, 4, std::max(nx1, nx2)}),
    KOKKOS_LAMBDA(const int b, const int f, const int t, Real& sum){
      if(nlevel(b, f) != (int)NeighborLevel::PhysicalBoundary) return;
      bool xface = (f == (int)FaceDir::XM || f == (int)FaceDir::XP);
      if(t >= (xface ? nx2 : nx1)) return;
      Real sign = (f == (int)FaceDir::XP || f == (int)FaceDir::YP) ? 1.0 : -1.0;
      Real area = xface ? dx2(b) : dx1(b);
      sum += sign*flux(b, f, t, k)*area;
    },
    rate
  );
  return rate;
}


//HEAT CONDUCTION (operator split, after the hydro step)
//Backward Euler in T with kappa frozen at T* (the state after the hydro step), for the increment delta = T - T*:
//  C_i delta_i + sum_f w_f (delta_i - delta_nb) = -sum_f w_f (T*_i - T*_nb),   C_i = rho c_v V_i/dt
//w_f = kappa_f * (face length)/(centre distance), the same number seen from both cells of a face -> symmetric positive
//definite, solved over the WHOLE mesh with multigrid-preconditioned CG (multigrid.hpp, after Athena++).
//Solving for delta instead of T: here w/C ~ 1e17, so in the T form C*T would drown in the round-off of w*T.
//Physical walls are insulating (zero conduction flux, whatever the hydro BC) -> sum C_i delta_i = 0 exactly, which
//is enforced after the solve (removes the round-off drift of the total energy).
struct ConductionWork {
  MGConduction mg;
  View1D Ts, rcv, rhs, x, gradT; //leaf cells: T*, rho c_v, right hand side, delta, |grad T*|
  long iters_total = 0;
  int  iters_max = 0, solves = 0, not_converged = 0;

  void ensure(Mesh& mesh){
    mg.setup(mesh);
    int n = mg.n_leaf();
    if((int)Ts.extent(0) != n){
      Ts = View1D("cond_Ts", n);  rcv = View1D("cond_rcv", n);
      rhs = View1D("cond_rhs", n);  x = View1D("cond_x", n);  gradT = View1D("cond_gradT", n);
    }
  }
};

void conduction_step(Mesh& mesh, ConductionWork& cw, const Physics& ph, Real dt){
  int n_active = mesh.num_active_blocks();
  if(n_active == 0 || dt <= 0.0) return;
  cw.ensure(mesh);

  const MeshInputs& in = mesh.inputs();
  int is = in.nghost, ie = in.nghost + in.nx1_block - 1, nx1 = in.nx1_block;
  int js = in.nghost, je = in.nghost + in.nx2_block - 1, nx2 = in.nx2_block;
  View4D q = mesh.get_data_view();
  View1D dx1 = mesh.get_block_dx1_view(), dx2 = mesh.get_block_dx2_view();
  View1D Ts = cw.Ts, rcv = cw.rcv, rhs = cw.rhs, x = cw.x, gradT = cw.gradT;
  View1D C = cw.mg.leaf_C(), kap = cw.mg.leaf_kap();
  Real gamma = ph.gamma, mu = ph.mu, d = ph.d, phi = ph.cond_phi;
  auto cells = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, is, js}, {n_active, ie + 1, je + 1});

  //T*, kappa(T*), rho c_v (per volume), C = rho c_v V/dt
  Kokkos::parallel_for("CondSetup", cells, KOKKOS_LAMBDA(int b, int i, int j){
    int c = MGConduction::leaf_index(b, i, j, is, js, nx1, nx2);
    Real rho = q(b, i, j, 0);
    Real p = (gamma - 1.0)*(q(b, i, j, 3) - 0.5*(q(b, i, j, 1)*q(b, i, j, 1) + q(b, i, j, 2)*q(b, i, j, 2))/rho);
    Real T = p*m_p*mu/(rho*k_B);
    Ts(c)  = T;
    kap(c) = kappa(mu, d, T);
    rcv(c) = rho*k_B/((gamma - 1.0)*mu*m_p);
    C(c)   = rcv(c)*dx1(b)*dx2(b)/dt;
  });

  //saturated (flux limited) conduction: the Fourier flux q_F = kappa |grad T| can not exceed the free streaming
  //limit q_sat = phi rho c_s^3 (c_s isothermal sound speed; Cowie & McKee 1977 write it as 5 phi_s rho c_s^3).
  //kappa_eff = kappa/(1 + q_F/q_sat) -> flux q_F/(1 + q_F/q_sat): Fourier where q_F << q_sat (dense gas),
  //q_sat where q_F >> q_sat (mean free path >> gradient length), smooth in between.
  //|grad T| from T* (lagged), so kappa_eff is frozen over the step like kappa
  if(phi > 0.0){
    cw.mg.max_face_gradient(Ts, gradT);
    Kokkos::parallel_for("CondFluxLimiter", cells, KOKKOS_LAMBDA(int b, int i, int j){
      int c = MGConduction::leaf_index(b, i, j, is, js, nx1, nx2);
      Real rho  = q(b, i, j, 0);
      Real cs   = Kokkos::sqrt(k_B*Ts(c)/(mu*m_p));
      Real qsat = phi*rho*cs*cs*cs;
      kap(c) = kap(c)/(1.0 + kap(c)*gradT(c)/qsat);
    });
  }
  cw.mg.set_coefficients();

  //rhs = -L T*
  cw.mg.apply(Ts, rhs, false);
  int n = cw.mg.n_leaf();
  Kokkos::parallel_for("CondRHS", Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int c){ rhs(c) = -rhs(c); });

  //insulating walls: sum rhs = 0 exactly (the face fluxes cancel pairwise), in floating point it is round-off. The
  //uniform mode of A has the tiny eigenvalue sum C (w/C up to ~1e21), so CG would amplify that round-off into a
  //spurious uniform shift of delta (~1 K at 100mikron's dt) and spend its iterations on it -> remove it first
  Real s_rhs = 0.0, s_C = 0.0;
  Kokkos::parallel_reduce("CondRHSSum", Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int c, Real& a, Real& s){
    a += rhs(c);  s += C(c);
  }, s_rhs, s_C);
  Real rhs_shift = s_rhs/s_C;
  Kokkos::parallel_for("CondRHSZeroSum", Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int c){
    rhs(c) -= C(c)*rhs_shift;
  });

  bool converged;
  const int max_iter = 500;
  //relative defect 1e-10, or the last CG correction of delta below 1e-9 K everywhere: at low density and large dt
  //(w/C ~ 1e21, e.g. 100mikron) A x can not resolve delta in double precision and a uniform T* gives an rhs of pure
  //round-off on which CG would only stall
  int it = cw.mg.solve(rhs, x, 1e-10, max_iter, converged, 1e-9);
  cw.solves++;
  cw.iters_total += it;
  cw.iters_max = std::max(cw.iters_max, it);
  if(!converged){
    if(cw.not_converged == 0)
      std::cout << "\nWarning: conduction MG-CG not converged after " << max_iter << " iterations\n";
    cw.not_converged++;
  }

  //insulating walls: sum C_i delta_i = 0 exactly; remove the round-off drift with a uniform shift of delta
  Real sCx = 0.0, sC = 0.0;
  Kokkos::parallel_reduce("CondDrift", Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int c, Real& a, Real& s){
    a += C(c)*x(c);  s += C(c);
  }, sCx, sC);
  Real shift = sCx/sC;

  //E += rho c_v delta (density and momentum unchanged)
  Kokkos::parallel_for("CondEnergy", cells, KOKKOS_LAMBDA(int b, int i, int j){
    int c = MGConduction::leaf_index(b, i, j, is, js, nx1, nx2);
    q(b, i, j, 3) += rcv(c)*(x(c) - shift);
  });
  Kokkos::fence();
}


//everything one particle half-kick needs from the gas: its CIC stencil (with the deposit volumes), the
//interpolated gas state and 1/M_eff of the gas the stencil reaches (implicit two-way drag)
struct ParticleCoupling {
  CICStencil S;
  GasAtParticle gas;
  Real inv_gas_mass;
};

//deposit volume dz*dr*depth of a cell of blk (planar 2D slab): the "depth" is the azimuthal length of gas the particle
//drags along. "cell" = local dr (what a 3D run would use), "ring" = 2*pi*r (axisymmetric), or a fixed length in cm
Real particle_cell_volume(const MeshBlock& blk, const Particle& p, Real r0, const std::string& depth_mode){
  Real depth;
  if(depth_mode == "cell")      depth = blk.dx2();
  else if(depth_mode == "ring") depth = 2.0*PI*(r0 + p.r); //absolute r
  else                          depth = std::stod(depth_mode);
  return blk.dx1()*blk.dx2()*depth;
}

//device result of the gas interpolation + its host copy, allocated once (allocating and freeing Views every call
//means cudaMalloc/cudaFree twice per step on a GPU, and cudaFree synchronises the device)
struct GasAtParticleResult {
  GasAtParticle gas;
  Real inv_gas_mass;
};
struct ParticleScratch {
  Kokkos::View<GasAtParticleResult, MemorySpace> d{"gas_at_particle"};
  typename Kokkos::View<GasAtParticleResult, MemorySpace>::host_mirror_type h = Kokkos::create_mirror_view(d);
};

//false if no block contains the particle
bool gas_at_particle(Mesh& mesh, const Particle& p, Real gamma, const std::string& depth_mode, ParticleCoupling& out,
                     ParticleScratch& scratch){
  CICStencil S;
  if(!build_cic_stencil(mesh, p.z, p.r, S)) return false;
  //each stencil cell deposits into its own block's cell volume (blocks are stored at index slot_id)
  for(int k = 0; k < S.n; ++k) S.vol[k] = particle_cell_volume(*mesh.blocks()[S.b[k]], p, mesh.inputs().x2_origin, depth_mode);

  View4D data = mesh.get_data_view();
  auto d = scratch.d;
  Kokkos::parallel_for("InterpolateGas", Kokkos::RangePolicy<ExecutionSpace>(0, 1), KOKKOS_LAMBDA(int){
    d().gas          = interpolate_gas(data, S, gamma);
    d().inv_gas_mass = cic_inv_gas_mass(data, S);
  });
  Kokkos::deep_copy(scratch.h, d); //one copy back for both results

  out.S = S;
  out.gas = scratch.h().gas;
  out.inv_gas_mass = scratch.h().inv_gas_mass;
  return true;
}


//width of the terminal stdout is attached to (80 if it is not a terminal, e.g. redirected to a file)
int terminal_columns(){
#ifdef _WIN32
  CONSOLE_SCREEN_BUFFER_INFO info;
  if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
    return info.srWindow.Right - info.srWindow.Left + 1;
#else
  struct winsize w;
  if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
#endif
  return 80;
}


int main(int argc, char* argv[]){
  Kokkos::initialize(argc, argv);
  {
    std::string param_file = (argc > 1) ? argv[1] : "params.txt";
    SimParams par = readParams(param_file);

    Physics ph;
    ph.gamma = par.gamma;
    ph.Omega = par.Omega;
    ph.mu    = par.gas_mu;
    ph.d     = 0.289e-7; //d: effective diameter of H2
    ph.cs2   = par.gas_p0/par.gas_rho0;
    ph.cond_phi = par.cond_phi;

    MeshInputs inputs;
    //local coordinates: the mesh starts at (0, 0), zmin / rmin of params.txt is the origin (see phi_diff)
    inputs.x1_origin = par.zmin;
    inputs.x2_origin = par.rmin;
    inputs.domain = RegionSize{0.0, par.zmax - par.zmin, 0.0, par.rmax - par.rmin, (int)par.Nz, (int)par.Nr};
    inputs.nx1_block = par.nx1_block;
    inputs.nx2_block = par.nx2_block;
    inputs.nghost = par.nghost;
    inputs.max_blocks = 1024;
    inputs.refine_regions = par.refine_regions; //absolute in params.txt -> local
    for(auto& reg : inputs.refine_regions){
      reg.x1min -= par.zmin;  reg.x1max -= par.zmin;
      reg.x2min -= par.rmin;  reg.x2max -= par.rmin;
    }

    //the residual reads i+-2 (slopes of the neighbours, well-balanced p1 slopes)
    if(inputs.nghost < 2) throw std::invalid_argument("nghost must be at least 2");

    GridBC bc;
    bc.left.type   = parseBC(par.bc_left);
    bc.right.type  = parseBC(par.bc_right);
    bc.bottom.type = parseBC(par.bc_bottom);
    bc.top.type    = parseBC(par.bc_top);
    validateBC(bc);

    inputs.periodic_x1 = (bc.left.type == BCType::Periodic);
    inputs.periodic_x2 = (bc.bottom.type == BCType::Periodic);

    Mesh mesh(inputs);
    mesh.BuildTree();
    mesh.CreateMeshBlocks();
    mesh.SetupNeighbors();

    setupOutputDir(par.outdir);
    writeMetadata(par.outdir, par);
    std::ofstream gridfile     = openGridFile(par.outdir, mesh);
    std::ofstream particlefile = openParticleFile(par.outdir);

    std::cout << "Active leaf blocks created: " << mesh.num_active_blocks() << "\n";
    std::cout << "Particle back-reaction volume: dz*dr*depth, particle_depth = " << par.particle_depth << "\n";

    init_cond(mesh, par);
    mesh.CaptureInitialState();
    ExchangeGhostZones(mesh);
    ApplyPhysicalBoundaryConditions(mesh, bc, ph.Omega, ph.cs2, ph.gamma);

    Particle p = initParticle(par); //local position
    ParticleScratch particle_scratch;
    const Real z0 = inputs.x1_origin;
    const RegionSize dom = inputs.domain;

    //AMR: refine around the particle (replaces the old particle patch) and where the density contrast is large (bow shock)
    //(refinement needs even block sizes holding at least 2*nghost cells, see Mesh::CheckRefinable)
    const int  amr_interval = 5;
    const int  max_level    = par.amr_max_level;
    const bool amr_enabled = inputs.nx1_block%2 == 0 && inputs.nx2_block%2 == 0 &&
                             inputs.nx1_block >= 2*inputs.nghost && inputs.nx2_block >= 2*inputs.nghost;
    if(!amr_enabled){
      std::cout << "Warning: nx1_block/nx2_block must be even and >= 2*nghost for refinement -> AMR disabled\n";
    }

    //particle inside the block, with a margin of 2 cells of that block
    //plus the distance it travels until the next regrid
    //a supersonic particle would otherwise run out of the refined region between two AMR updates
    Real dt = 0.0; //current time step, set below
    auto particle_in_block = [&p, &dt, amr_interval](const MeshBlock& block){
      if(!p.active) return false;
      Real mz = 2.0*block.dx1() + 1.5*Kokkos::fabs(p.vz)*dt*amr_interval;
      Real mr = 2.0*block.dx2() + 1.5*Kokkos::fabs(p.vr)*dt*amr_interval;
      return p.z >= block.block_size.x1min - mz && p.z <= block.block_size.x1max + mz &&
             p.r >= block.block_size.x2min - mr && p.r <= block.block_size.x2max + mr;
    };


    //density contrast |ln(rho/rho_init)| (rho_init = initial state, remapped on every regrid, so the stratified
    //background is never flagged). Measured on supersonic (Mach 2, edge cells included): bow shock within 0.5 cm of
    //the particle 0.15, Mach cone 1-2 cm out 0.03, trail (the jet the particle drags) <= 0.033, wake 0.013,
    //undisturbed 1e-10 -> refine 0.08 / derefine 0.05 refines the bow shock and lets the trail derefine.
    auto max_contrast = [](const MeshBlock& block, const View4D& data, const View3D& rho_init){
      int b = block.slot_id;
      Real max_c = 0.0;
      Kokkos::parallel_reduce("DensityContrast",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({block.is, block.js}, {block.ie + 1, block.je + 1}),
        KOKKOS_LAMBDA(int i, int j, Real& local_max){
          Real c = Kokkos::fabs(Kokkos::log(data(b, i, j, 0)/rho_init(b, i, j)));
          if(c > local_max) local_max = c;
        },
        Kokkos::Max<Real>(max_c)
      );
      return max_c;
    };
    const Real refine_contrast   = par.amr_refine;
    const Real derefine_contrast = par.amr_derefine;
    std::cout << "AMR: max_level " << max_level << ", density contrast |ln(rho/rho_init)|: refine if > "
              << refine_contrast << ", derefine if < " << derefine_contrast;
    if(par.amr_radius > 0.0) std::cout << ", only within " << par.amr_radius << " cm of the particle";
    std::cout << "\n";

    //optional: contrast refinement only within amr_radius of the particle (distance = particle to the nearest point
    //of the block; for more particles: true if within the radius of ANY particle). Off by default (<= 0).
    const Real amr_radius = par.amr_radius;
    auto near_particle = [&p, amr_radius](const MeshBlock& block){
      if(amr_radius <= 0.0) return true;
      if(!p.active) return false;
      Real dz = Kokkos::fmax(0.0, Kokkos::fmax(block.block_size.x1min - p.z, p.z - block.block_size.x1max));
      Real dr = Kokkos::fmax(0.0, Kokkos::fmax(block.block_size.x2min - p.r, p.r - block.block_size.x2max));
      return dz*dz + dr*dr <= amr_radius*amr_radius;
    };

    BlockCriterion refine_crit = [&](const MeshBlock& block, const View4D& data, const View3D& rho_init){
      return particle_in_block(block) || (near_particle(block) && max_contrast(block, data, rho_init) > refine_contrast);
    };
    BlockCriterion derefine_crit = [&](const MeshBlock& block, const View4D& data, const View3D& rho_init){
      return !particle_in_block(block) && (!near_particle(block) || max_contrast(block, data, rho_init) < derefine_contrast);
    };

    //time step: dt = (tf-t0)/Nt, capped by the leapfrog stability limit of the gravity source and by a maximum
    //Courant number. The B2 step is implicit, but the approximate factorisation + per-block line solves lose
    //stability at very large Courant numbers: ~4e4 on a single level, a few 1e3 across fine/coarse boundaries
    //(tested it extensively on the hydrostatic gaussian_z column). 
    //The cap is evaluated on the finest level after every AMR update.
    const Real courant_max = 2000.0;
    Real t0 = par.t0;
    Real tf = par.tf;
    Real dt_user = (tf - t0)/par.Nt;
    Real dt_leapfrog = 1.0/(2.0*ph.Omega);
    if(dt_user > dt_leapfrog){
      std::cout << "Warning: dt=" << dt_user << " exceeds the leapfrog limit, using dt=" << dt_leapfrog << "\n";
      dt_user = dt_leapfrog;
    }
    //particle CFL: the particle moves at most particle_cfl cells of its block per step, so the CIC coupling sees
    //every cell it passes. For a supersonic particle this also keeps the hydro Courant number ~1 near it (the cone
    //is a resolved shock, not something the implicit scheme may step over). Subsonic runs are barely affected.
    const Real particle_cfl = 0.5;
    auto particle_dt = [&](){
      if(!p.active) return dt_user;
      int b = find_particle_block(mesh, p.z, p.r);
      if(b < 0) return dt_user;
      const MeshBlock& blk = *mesh.blocks()[b];
      Real rate = Kokkos::fabs(p.vz)/blk.dx1() + Kokkos::fabs(p.vr)/blk.dx2();
      return rate > 0.0 ? particle_cfl/rate : dt_user;
    };
    auto choose_dt = [&](){
      return std::min({dt_user, compute_cfl_dt(mesh, ph.gamma, courant_max), particle_dt()});
    };
    dt = choose_dt();
    std::cout << "dt = " << dt << "  (Courant number " << dt/compute_cfl_dt(mesh, ph.gamma, 1.0) << ")\n";
    {
      ParticleCoupling pc0;
      if(gas_at_particle(mesh, p, ph.gamma, par.particle_depth, pc0, particle_scratch)){
        Real vrel = std::hypot(p.vz - pc0.gas.u, p.vr - pc0.gas.v);
        std::cout << "Particle start: z - zmin = " << p.z << " cm, r - rmin = " << p.r << " cm, Mach "
                  << vrel/pc0.gas.cs << "\n";
      }
    }

    Real t = t0;
    int n = 0;
    Real t_save = (tf - t0)/100.0;
    Real t_next_save = t0;

    //progressbar
    int lastpercentshown = -1;
    auto draw_progress = [&](Real t_now, bool force){
      Real progress = (tf > t0) ? (t_now - t0)/(tf - t0) : 1.0;
      if(progress > 1.0) progress = 1.0;
      int percent = (int)(progress*100.0);
      if(!force && percent == lastpercentshown) return;
      lastpercentshown = percent;
      //the line must fit the terminal: \r only returns to the start of the current screen line, so a wrapped line
      //leaves its first part behind on every redraw. Bar width adapts, the step/blocks part is dropped if needed.
      std::ostringstream info, extra;
      info << " " << std::setw(3) << percent << "%  t=" << std::scientific << std::setprecision(3) << t_now;
      extra << "  step " << n << "  blocks " << mesh.num_active_blocks();
      int cols = terminal_columns() - 1; //never write into the last column (some terminals wrap there)
      int fixed = 2 + (int)info.str().size(); //"[" + "]" + info
      bool show_extra = cols - fixed - (int)extra.str().size() >= 10;
      if(show_extra) fixed += (int)extra.str().size();
      int barwidth = std::clamp(cols - fixed, 0, 40);
      int filled = (int)(progress*barwidth);
      std::string line = "[" + std::string(filled, '#') + std::string(barwidth - filled, '-') + "]" + info.str()
                       + (show_extra ? extra.str() : "");
      if((int)line.size() > cols) line.resize(std::max(cols, 0));
      line.resize(std::max(cols, (int)line.size()), ' '); //pad: clears what a longer previous line left behind
      std::cout << "\r" << line << std::flush;
    };

    ImplicitWork work;
    ConductionWork cond_work;
    cond_work.mg.small_level_cells = par.mg_small_level_cells;
    Real initial_mass   = compute_total(mesh, 0);
    Real initial_energy = compute_total(mesh, 3);
    Real cumulative_mass_out = 0.0;

    auto start_time = std::chrono::high_resolution_clock::now();

    try{
      //initial state
      writeFrame(gridfile, mesh, t);
      writeParticle(particlefile, p, t, n);
      gridfile.flush();
      particlefile.flush();
      t_next_save += t_save;

      while(t < tf*(1.0 - 1e-14)){
        //AMR part
        if(amr_enabled && n%amr_interval == 0){
          mesh.UpdateDynamicMesh(refine_crit, derefine_crit, max_level, par.amr_derefine);
          //populating ghost zones and boundaries for the new mesh layout immediately!!!
          ExchangeGhostZones(mesh);
          ApplyPhysicalBoundaryConditions(mesh, bc, ph.Omega, ph.cs2, ph.gamma);
        }
        //time step every step (also right after regridding!!! the finest level sets the Courant cap)
        dt = std::min(choose_dt(), tf - t);

        draw_progress(t, n == 0);

        //particle: first half kick with the gas at t_n (implicit two-way drag), its back-reaction on the gas at the
        //old position, then the drift. The gas gets the impulse BEFORE the hydro step / the next kick sees it.
        bool has_feedback = false;
        if(p.active){
          ParticleCoupling pc;
          if(gas_at_particle(mesh, p, ph.gamma, par.particle_depth, pc, particle_scratch)){
            has_feedback = leapfrog_first_half(p, pc.gas, ph.Omega, z0, dom.x1min, dom.x2min, dom.x1max, dom.x2max, dt, ph.mu,
                                               pc.inv_gas_mass);
            source_projection(mesh.get_data_view(), p, pc.S); //even if it left in the drift
            ExchangeGhostZones(mesh);
            ApplyPhysicalBoundaryConditions(mesh, bc, ph.Omega, ph.cs2, ph.gamma);
          }else{
            p.active = false;
          }
        }

        //gas
        implicit_step_B2(mesh, bc, work, ph, dt);
        cumulative_mass_out += boundary_outflow_rate(mesh, 0)*dt;

        //particle: second half kick with the gas at t_{n+1} (implicit two-way drag), then its back-reaction
        if(has_feedback){
          ParticleCoupling pc;
          if(gas_at_particle(mesh, p, ph.gamma, par.particle_depth, pc, particle_scratch)){
            leapfrog_second_half(p, pc.gas, ph.Omega, z0, dt, ph.mu, pc.inv_gas_mass);
            source_projection(mesh.get_data_view(), p, pc.S);
          }else{
            p.active = false;
          }
        }

        //heat conduction over the whole mesh (also conducts the particle's deposited heat)
        conduction_step(mesh, cond_work, ph, dt);

        ExchangeGhostZones(mesh);
        ApplyPhysicalBoundaryConditions(mesh, bc, ph.Omega, ph.cs2, ph.gamma);

        n += 1;
        t += dt;

        //output at the END of the step: a frame written right after a regrid (before this step's conduction) showed
        //the regrid's restriction heating (kinetic energy of averaged shear -> heat) before conduction spreads it.
        //The final frame is written after the loop.
        if(t >= t_next_save && t < tf*(1.0 - 1e-14)){
          writeFrame(gridfile, mesh, t);
          writeParticle(particlefile, p, t, n);
          gridfile.flush();
          particlefile.flush();
          t_next_save += t_save;
        }
      }

      writeFrame(gridfile, mesh, t);
      writeParticle(particlefile, p, t, n);
      gridfile.flush();
      particlefile.flush();
      draw_progress(tf, true);

      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;

      std::cout << "\nSimulation completed in " << n << " steps!\n";
      std::cout << "Output saved to " << par.outdir << "/grid.bin\n";

      Real final_mass   = compute_total(mesh, 0);
      Real final_energy = compute_total(mesh, 3);
      Real expected_mass = initial_mass - cumulative_mass_out;

      std::cout << std::scientific << std::setprecision(12);
      std::cout << "  Initial Mass M(t=0)      = " << initial_mass << "\n";
      std::cout << "  Final Mass M(t=tf)       = " << final_mass << "\n";
      std::cout << "  Net Boundary Outflow     = " << cumulative_mass_out << "\n";
      std::cout << "  Mass Discrepancy         = " << std::abs(final_mass - expected_mass) << "\n\n";
      std::cout << "  Initial Energy E(t=0)    = " << initial_energy << "\n";
      std::cout << "  Final Energy E(t=tf)     = " << final_energy << "\n";
      if(cond_work.solves > 0)
        std::cout << "  Conduction MG-CG: " << cond_work.solves << " solves, mean "
                  << (double)cond_work.iters_total/cond_work.solves << " / max " << cond_work.iters_max
                  << " iterations, " << cond_work.not_converged << " not converged\n";
      std::cout << "RUNTIME: " << elapsed.count() << " seconds.\n";
    }catch(const std::exception& e){
      std::cerr << "\n[CRASH] at step=" << n << " t=" << t << ": " << e.what() << "\n";
    }
  }
  Kokkos::finalize();

  return 0;
}
