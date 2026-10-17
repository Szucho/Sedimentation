#include "particle.hpp"
#include "mesh.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

//constants and all KOKKOS_INLINE_FUNCTIONs (interpolate_gas, drag_accel, grav_accel,
//leapfrog_first_half, leapfrog_second_half) moved to particle.hpp so main.cpp can link against them.


//particle to grid interpolation (back-reaction of the drag), see particle.hpp
void source_projection(const View4D& grid_data, const Particle& p, const CICStencil& S){

  //momentum the particle got from drag in the last (implicit) half-kick;
  //N3 dictates that the gas gets the opposite (entry k gets w_k of it, as a momentum DENSITY in its own cell)
  Real Jz = -p.mass*p.dvz_drag;
  Real Jr = -p.mass*p.dvr_drag;

  //single-iteration kernel: the stencil entries are distinct cells (build_cic_stencil merges duplicates),
  //so each cell is read and written once, no atomics are needed and it runs on any backend
  Kokkos::parallel_for("SourceProjection", Kokkos::RangePolicy<ExecutionSpace>(0, 1), KOKKOS_LAMBDA(int){
    for(int k = 0; k < S.n; ++k){
      int b = S.b[k], i = S.i[k], j = S.j[k];
      Real rho = grid_data(b, i, j, 0); //0th index is rho, but the chondrite stays solid -> no mass source
      Real mz  = grid_data(b, i, j, 1);
      Real mr  = grid_data(b, i, j, 2);
      Real mz_new = mz + S.w[k]*Jz/S.vol[k];
      Real mr_new = mr + S.w[k]*Jr/S.vol[k];
      grid_data(b, i, j, 1) = mz_new;
      grid_data(b, i, j, 2) = mr_new;
      //exact kinetic energy change -> internal energy (pressure) unchanged
      grid_data(b, i, j, 3) += 0.5*((mz_new*mz_new + mr_new*mr_new) - (mz*mz + mr*mr))/rho;
    }
  });
}


//CLAUDE FIX: new helper - leaf blocks don't overlap, so at most one block contains (z, r).
//Half-open intervals, except the domain's upper edge which is included.
int find_particle_block(const Mesh& mesh, Real z, Real r){
  const RegionSize& dom = mesh.inputs().domain;
  for(const auto& bptr : mesh.blocks()){
    const RegionSize& s = bptr->block_size;
    bool in_z = (z >= s.x1min) && (z < s.x1max || (z == s.x1max && s.x1max == dom.x1max));
    bool in_r = (r >= s.x2min) && (r < s.x2max || (r == s.x2max && s.x2max == dom.x2max));
    if(in_z && in_r) return bptr->slot_id;
  }
  return -1;
}


//adds weight w to cell (b, i, j) of the stencil, merging duplicates
static void add_stencil_cell(CICStencil& S, int b, int i, int j, Real w){
  for(int k = 0; k < S.n; ++k){
    if(S.b[k] == b && S.i[k] == i && S.j[k] == j){ S.w[k] += w; return; }
  }
  if(S.n == CIC_MAX_CELLS) throw std::runtime_error("build_cic_stencil: more than CIC_MAX_CELLS cells");
  S.b[S.n] = b; S.i[S.n] = i; S.j[S.n] = j; S.w[S.n] = w; S.vol[S.n] = 0.0;
  ++S.n;
}

//1D overlap of cell I of a level-Lh grid with the cells of a block axis (first global cell B, n cells, level Lb),
//done in exact integer global cell indices: physical coordinates of tiny cells far from the origin (z ~ 1e12 cm,
//dz ~ 1e-2 cm) are only a few ulps wide. Calls f(local cell index, covered fraction of cell I).
template<typename F>
static void overlap_1d(std::int64_t I, int Lh, std::int64_t B, int n, int Lb, F&& f){
  const int M = std::max(Lh, Lb);
  const int sh = M - Lh, sb = M - Lb;
  const std::int64_t lo = I << sh, hi = (I + 1) << sh; //cell I in level-M units, [lo, hi)
  const std::int64_t c0 = std::max<std::int64_t>(B,         lo >> sb);
  const std::int64_t c1 = std::min<std::int64_t>(B + n - 1, (hi - 1) >> sb);
  for(std::int64_t c = c0; c <= c1; ++c){
    std::int64_t ov = std::min(hi, (c + 1) << sb) - std::max(lo, c << sb);
    if(ov > 0) f((int)(c - B), (Real)ov/(Real)(std::int64_t(1) << sh));
  }
}

bool build_cic_stencil(const Mesh& mesh, Real z, Real r, CICStencil& S){
  S.n = 0;
  int home = find_particle_block(mesh, z, r);
  if(home < 0) return false;
  const MeshInputs& in = mesh.inputs();
  const MeshBlock& hb = *mesh.blocks()[home];
  const int nx1 = in.nx1_block, nx2 = in.nx2_block;
  const int Lh = hb.loc.level;

  //CIC cells in the home block's local coordinates (cell index relative to its first active cell, may be -1 or n)
  Real z_g = (z - hb.block_size.x1min)/hb.dx1() - 0.5; //-1/2: weights refer to cell centres
  Real r_g = (r - hb.block_size.x2min)/hb.dx2() - 0.5;
  int  i0  = (int)std::floor(z_g);
  int  j0  = (int)std::floor(r_g);
  Real p   = z_g - i0;
  Real q   = r_g - j0;

  //number of cells across the domain at the home level
  const std::int64_t N1 = (std::int64_t)mesh.nx1_blocks_root()*nx1 << Lh;
  const std::int64_t N2 = (std::int64_t)mesh.nx2_blocks_root()*nx2 << Lh;

  for(int di = 0; di < 2; ++di){
    for(int dj = 0; dj < 2; ++dj){
      Real Wc = (di ? p : 1.0 - p)*(dj ? q : 1.0 - q);
      if(Wc <= 0.0) continue;

      //global cell index of the stencil cell at the home level
      std::int64_t I = hb.loc.lx1*nx1 + i0 + di;
      std::int64_t J = hb.loc.lx2*nx2 + j0 + dj;
      //outside the domain: periodic image, or folded back onto the boundary cell (it lies in the home block)
      if(I < 0)   I = in.periodic_x1 ? I + N1 : 0;
      if(I >= N1) I = in.periodic_x1 ? I - N1 : N1 - 1;
      if(J < 0)   J = in.periodic_x2 ? J + N2 : 0;
      if(J >= N2) J = in.periodic_x2 ? J - N2 : N2 - 1;

      //spread Wc over the leaf cells covering the stencil cell, by covered area (any block, any level)
      for(const auto& bptr : mesh.blocks()){
        const int Lb = bptr->loc.level;
        overlap_1d(I, Lh, bptr->loc.lx1*nx1, nx1, Lb, [&](int ii, Real fz){
          overlap_1d(J, Lh, bptr->loc.lx2*nx2, nx2, Lb, [&](int jj, Real fr){
            add_stencil_cell(S, bptr->slot_id, ii + bptr->nghost, jj + bptr->nghost, Wc*fz*fr);
          });
        });
      }
    }
  }

  //the leaves tile the domain, so the weights sum to 1 (momentum conservation of the deposit)
  Real wsum = 0.0;
  for(int k = 0; k < S.n; ++k) wsum += S.w[k];
  if(S.n == 0 || std::fabs(wsum - 1.0) > 1e-12) throw std::runtime_error("build_cic_stencil: stencil not covered by the mesh");
  return true;
}
