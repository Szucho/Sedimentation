#include "multigrid.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <tuple>
#include <stdexcept>

/*
 Multigrid-preconditioned CG for the implicit heat conduction, see multigrid.hpp.
 Structure and smoother follow Athena++ (Tomida & Stone 2023; src/multigrid/multigrid.cpp, multigrid_driver.cpp,
 src/crdiffusion/mg_crdiffusion.cpp), the deviations needed for the CG preconditioner are listed in multigrid.hpp.
*/

namespace {

//a cell of a multigrid level on the host: tree level L (negative below the root grid), global index (I, J) on
//that level and its physical extent
struct HCell {
  int L;
  std::int64_t I, J;
  Real z0, z1, r0, r1, vol;
};

struct Key {
  int L;
  std::int64_t I, J;
  bool operator==(const Key& o) const { return L == o.L && I == o.I && J == o.J; }
  bool operator<(const Key& o) const { return std::tie(L, I, J) < std::tie(o.L, o.I, o.J); }
};
struct KeyHash {
  std::size_t operator()(const Key& k) const {
    std::size_t h = std::hash<std::int64_t>()(k.I);
    h ^= std::hash<std::int64_t>()(k.J) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(k.L) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

//a host level: cells, faces (a, b, dir) with a < b (dir 0 = z face, 1 = r face), parent on the next coarser level
struct HLevel {
  std::vector<HCell> cells;
  std::vector<std::tuple<int, int, int>> faces;
  std::vector<int> parent;
};

//centre distance across the face between a and b (normal direction dir)
Real face_distance(const HCell& a, const HCell& b, int dir){
  return (dir == 0) ? 0.5*((a.z1 - a.z0) + (b.z1 - b.z0)) : 0.5*((a.r1 - a.r0) + (b.r1 - b.r0));
}

//geometric factor of the face between a and b: shared face length/centre distance (two-point flux, as on the leaf
//level: Same level -> ht/hn, fine/coarse -> h_fine,t/(1.5 h_fine,n))
Real face_factor(const HCell& a, const HCell& b, int dir){
  Real shared = (dir == 0) ? std::min(a.r1, b.r1) - std::max(a.r0, b.r0)
                           : std::min(a.z1, b.z1) - std::max(a.z0, b.z0);
  return (shared > 0.0) ? shared/face_distance(a, b, dir) : 0.0;
}

//AMR leaf cells and their faces. Neighbours are found by key: same level, else the coarser cell, else the two
//finer cells (2:1 balance), periodic wrap on the root grid index range of that level.
HLevel build_leaf_level(Mesh& mesh){
  const MeshInputs& in = mesh.inputs();
  int nx1 = in.nx1_block, nx2 = in.nx2_block, is = in.nghost, js = in.nghost;
  std::int64_t Nz0 = (std::int64_t)mesh.nx1_blocks_root()*nx1;
  std::int64_t Nr0 = (std::int64_t)mesh.nx2_blocks_root()*nx2;
  int n_active = mesh.num_active_blocks();

  HLevel lv;
  lv.cells.resize((std::size_t)n_active*nx1*nx2);
  std::unordered_map<Key, int, KeyHash> idx;
  idx.reserve(lv.cells.size()*2);
  for(const auto& bptr : mesh.blocks()){
    const MeshBlock& blk = *bptr;
    Real dz = blk.dx1(), dr = blk.dx2();
    for(int i = 0; i < nx1; ++i){
      for(int j = 0; j < nx2; ++j){
        int c = MGConduction::leaf_index(blk.slot_id, i + is, j + js, is, js, nx1, nx2);
        HCell& h = lv.cells[c];
        h.L = blk.loc.level;
        h.I = blk.loc.lx1*nx1 + i;
        h.J = blk.loc.lx2*nx2 + j;
        h.z0 = blk.block_size.x1min + i*dz;  h.z1 = h.z0 + dz;
        h.r0 = blk.block_size.x2min + j*dr;  h.r1 = h.r0 + dr;
        h.vol = dz*dr;
        idx[Key{h.L, h.I, h.J}] = c;
      }
    }
  }

  //index on level L after the periodic wrap, -1 = outside the domain
  auto wrap = [&](std::int64_t K, std::int64_t N0, int L, bool periodic) -> std::int64_t {
    std::int64_t N = N0 << L;
    if(periodic) return ((K % N) + N) % N;
    return (K < 0 || K >= N) ? -1 : K;
  };
  auto find = [&](int L, std::int64_t I, std::int64_t J) -> int {
    auto it = idx.find(Key{L, I, J});
    return (it == idx.end()) ? -1 : it->second;
  };

  for(int c = 0; c < (int)lv.cells.size(); ++c){
    const HCell& h = lv.cells[c];
    for(int dir = 0; dir < 2; ++dir){
      for(int s = -1; s <= 1; s += 2){
        std::int64_t In = h.I + (dir == 0 ? s : 0), Jn = h.J + (dir == 1 ? s : 0);
        In = wrap(In, Nz0, h.L, in.periodic_x1);
        Jn = wrap(Jn, Nr0, h.L, in.periodic_x2);
        if(In < 0 || Jn < 0) continue; //physical wall: insulating, no face
        std::vector<int> nbs;
        int nb = find(h.L, In, Jn);
        if(nb >= 0) nbs.push_back(nb);
        if(nbs.empty() && h.L > 0){
          nb = find(h.L - 1, In >> 1, Jn >> 1);
          if(nb >= 0) nbs.push_back(nb);
        }
        if(nbs.empty()){
          //the two finer cells touching this face
          std::int64_t If = 2*In + ((dir == 0) ? (s > 0 ? 0 : 1) : 0);
          std::int64_t Jf = 2*Jn + ((dir == 1) ? (s > 0 ? 0 : 1) : 0);
          for(int t = 0; t < 2; ++t){
            nb = (dir == 0) ? find(h.L + 1, If, Jf + t) : find(h.L + 1, If + t, Jf);
            if(nb >= 0) nbs.push_back(nb);
          }
        }
        if(nbs.empty()) throw std::runtime_error("MGConduction: no neighbour found for a leaf cell");
        for(int b : nbs) if(c < b) lv.faces.emplace_back(c, b, dir);
      }
    }
  }
  std::sort(lv.faces.begin(), lv.faces.end());
  lv.faces.erase(std::unique(lv.faces.begin(), lv.faces.end()), lv.faces.end());
  return lv;
}

//next coarser level: the cells of the finest tree level present are merged 2x2 into their parents, all others are
//kept (Athena++ coarsens the refined octets level by level down to the root grid, then the root grid itself; on a
//uniform level this rule halves the whole grid). Returns false if nothing could be merged.
bool coarsen(HLevel& fine, HLevel& coarse){
  int Lmax = fine.cells[0].L;
  for(const auto& h : fine.cells) Lmax = std::max(Lmax, h.L);

  std::vector<Key> keys(fine.cells.size());
  for(std::size_t c = 0; c < fine.cells.size(); ++c){
    const HCell& h = fine.cells[c];
    keys[c] = (h.L == Lmax) ? Key{h.L - 1, h.I >> 1, h.J >> 1} : Key{h.L, h.I, h.J};
  }
  std::vector<Key> uniq = keys;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  if(uniq.size() == fine.cells.size()) return false;

  std::unordered_map<Key, int, KeyHash> idx;
  idx.reserve(uniq.size()*2);
  coarse.cells.resize(uniq.size());
  for(std::size_t k = 0; k < uniq.size(); ++k){
    idx[uniq[k]] = (int)k;
    HCell& h = coarse.cells[k];
    h.L = uniq[k].L;  h.I = uniq[k].I;  h.J = uniq[k].J;
    h.z0 = h.r0 = 1e300;  h.z1 = h.r1 = -1e300;  h.vol = 0.0;
  }
  fine.parent.resize(fine.cells.size());
  for(std::size_t c = 0; c < fine.cells.size(); ++c){
    int p = idx[keys[c]];
    fine.parent[c] = p;
    HCell& h = coarse.cells[p];
    const HCell& f = fine.cells[c];
    h.z0 = std::min(h.z0, f.z0);  h.z1 = std::max(h.z1, f.z1);
    h.r0 = std::min(h.r0, f.r0);  h.r1 = std::max(h.r1, f.r1);
    h.vol += f.vol;
  }
  //coarse faces = fine faces between different parents
  for(const auto& [a, b, dir] : fine.faces){
    int pa = fine.parent[a], pb = fine.parent[b];
    if(pa == pb) continue;
    coarse.faces.emplace_back(std::min(pa, pb), std::max(pa, pb), dir);
  }
  std::sort(coarse.faces.begin(), coarse.faces.end());
  coarse.faces.erase(std::unique(coarse.faces.begin(), coarse.faces.end()), coarse.faces.end());
  return true;
}

IdxView to_device(const std::vector<int>& v, const char* name){
  IdxView d(name, v.size());
  auto h = Kokkos::create_mirror_view(d);
  for(std::size_t k = 0; k < v.size(); ++k) h(k) = v[k];
  Kokkos::deep_copy(d, h);
  return d;
}
View1D to_device(const std::vector<Real>& v, const char* name){
  View1D d(name, v.size());
  auto h = Kokkos::create_mirror_view(d);
  for(std::size_t k = 0; k < v.size(); ++k) h(k) = v[k];
  Kokkos::deep_copy(d, h);
  return d;
}

Real dot(const View1D& a, const View1D& b, int n){
  Real s = 0.0;
  Kokkos::parallel_reduce("MGDot", Kokkos::RangePolicy<ExecutionSpace>(0, n),
    KOKKOS_LAMBDA(int i, Real& acc){ acc += a(i)*b(i); }, s);
  return s;
}

} //namespace


void MGConduction::setup(Mesh& mesh){
  //block layout signature: rebuild only after a regrid
  std::vector<std::int64_t> sig;
  sig.reserve(4*mesh.num_active_blocks() + 2);
  sig.push_back(mesh.inputs().nx1_block);
  sig.push_back(mesh.inputs().nx2_block);
  for(const auto& b : mesh.blocks()){
    sig.push_back(b->slot_id);  sig.push_back(b->loc.level);
    sig.push_back(b->loc.lx1);  sig.push_back(b->loc.lx2);
  }
  if(sig == signature_ && !lv_.empty()) return;
  signature_ = sig;

  //host hierarchy
  std::vector<HLevel> h;
  h.push_back(build_leaf_level(mesh));
  while((int)h.back().cells.size() > min_coarse_cells){
    HLevel c;
    if(!coarsen(h.back(), c)) break;
    h.push_back(std::move(c));
  }
  h.back().parent.assign(h.back().cells.size(), -1);

  //device levels
  lv_.assign(h.size(), MGLevel{});
  for(std::size_t l = 0; l < h.size(); ++l){
    const HLevel& H = h[l];
    MGLevel& L = lv_[l];
    L.n = (int)H.cells.size();

    std::vector<int> deg(L.n + 1, 0);
    for(const auto& [a, b, dir] : H.faces){ deg[a + 1]++; deg[b + 1]++; }
    for(int c = 0; c < L.n; ++c) deg[c + 1] += deg[c];
    std::vector<int> col(deg[L.n]), fill(deg.begin(), deg.end() - 1);
    std::vector<Real> g(deg[L.n]), dist(deg[L.n]);
    for(const auto& [a, b, dir] : H.faces){
      Real gf = face_factor(H.cells[a], H.cells[b], dir);
      Real df = face_distance(H.cells[a], H.cells[b], dir);
      dist[fill[a]] = df;  col[fill[a]] = b;  g[fill[a]++] = gf;
      dist[fill[b]] = df;  col[fill[b]] = a;  g[fill[b]++] = gf;
    }
    if(l == 0) L.dist = to_device(dist, "mg_dist");
    L.row_ptr = to_device(deg, "mg_row_ptr");
    L.col = to_device(col, "mg_col");
    L.g = to_device(g, "mg_g");
    L.W = View1D("mg_W", col.size());

    std::vector<Real> vol(L.n);
    std::vector<int> cl[2];
    for(int c = 0; c < L.n; ++c){
      vol[c] = H.cells[c].vol;
      cl[(H.cells[c].I + H.cells[c].J) & 1].push_back(c);
    }
    L.vol = to_device(vol, "mg_vol");
    L.color_cells[0] = to_device(cl[0], "mg_red");
    L.color_cells[1] = to_device(cl[1], "mg_black");
    L.parent = to_device(H.parent, "mg_parent");

    if(l > 0){
      const HLevel& F = h[l - 1];
      std::vector<int> cptr(L.n + 1, 0), ch(F.cells.size());
      for(int p : F.parent) cptr[p + 1]++;
      for(int c = 0; c < L.n; ++c) cptr[c + 1] += cptr[c];
      std::vector<int> f2(cptr.begin(), cptr.end() - 1);
      for(int c = 0; c < (int)F.cells.size(); ++c) ch[f2[F.parent[c]]++] = c;
      L.child_ptr = to_device(cptr, "mg_child_ptr");
      L.child = to_device(ch, "mg_child");
    }

    L.C = View1D("mg_C", L.n);  L.kap = View1D("mg_kap", L.n);  L.D = View1D("mg_D", L.n);
    L.x = View1D("mg_x", L.n);  L.b = View1D("mg_b", L.n);
    L.work = View1D("mg_work", L.n);
  }
  int n0 = lv_[0].n;
  cg_p_ = View1D("cg_p", n0);  cg_Ap_ = View1D("cg_Ap", n0);
  build_small_levels();
}


//flat storage of the small levels (see multigrid.hpp). Small levels: from the first level after the leaf level with
//at most small_level_cells cells down to the coarsest (always at least the coarsest level).
void MGConduction::build_small_levels(){
  int L = (int)lv_.size();
  ls_ = L - 1;
  for(int l = std::min(1, L - 1); l < L; ++l) if(lv_[l].n <= small_level_cells){ ls_ = l; break; }
  nsmall_ = L - ls_;
  if(nsmall_ > kMaxSmall) throw std::runtime_error("MGConduction: too many small levels");

  //host copies of the per-level index arrays
  std::vector<int> cell_off(nsmall_ + 1, 0), edge_off(nsmall_ + 1, 0);
  for(int k = 0; k < nsmall_; ++k){
    cell_off[k + 1] = cell_off[k] + lv_[ls_ + k].n;
    edge_off[k + 1] = edge_off[k] + (int)lv_[ls_ + k].col.extent(0);
  }
  int Nc = cell_off[nsmall_], Ne = edge_off[nsmall_];
  std::vector<int> rb(Nc), re(Nc), colv(Ne), par(Nc, -1), cb(Nc, 0), ce(Nc, 0), chv, colcells;
  std::vector<int> color_off(2*nsmall_ + 1, 0);
  for(int k = 0; k < nsmall_; ++k){
    const MGLevel& Lv = lv_[ls_ + k];
    auto rp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Lv.row_ptr);
    auto cl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Lv.col);
    auto pa = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Lv.parent);
    for(int c = 0; c < Lv.n; ++c){
      rb[cell_off[k] + c] = edge_off[k] + rp(c);
      re[cell_off[k] + c] = edge_off[k] + rp(c + 1);
      if(k + 1 < nsmall_) par[cell_off[k] + c] = cell_off[k + 1] + pa(c);
    }
    for(int e = 0; e < (int)cl.extent(0); ++e) colv[edge_off[k] + e] = cell_off[k] + cl(e);
    if(k > 0){ //children of level k live on level k-1 (also small)
      auto cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Lv.child_ptr);
      auto ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Lv.child);
      for(int c = 0; c < Lv.n; ++c){
        cb[cell_off[k] + c] = (int)chv.size() + cp(c);
        ce[cell_off[k] + c] = (int)chv.size() + cp(c + 1);
      }
      int base = (int)chv.size();
      chv.resize(base + ch.extent(0));
      for(int i = 0; i < (int)ch.extent(0); ++i) chv[base + i] = cell_off[k - 1] + ch(i);
    }
    for(int color = 0; color < 2; ++color){
      auto cc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Lv.color_cells[color]);
      for(int i = 0; i < (int)cc.extent(0); ++i) colcells.push_back(cell_off[k] + cc(i));
      color_off[2*k + color + 1] = (int)colcells.size();
    }
  }
  for(int k = 0; k <= nsmall_; ++k) s_cell_off_[k] = cell_off[k];
  for(int k = 0; k <= 2*nsmall_; ++k) s_color_off_[k] = color_off[k];
  s_rb_ = to_device(rb, "mg_s_rb");  s_re_ = to_device(re, "mg_s_re");  s_col_ = to_device(colv, "mg_s_col");
  s_parent_ = to_device(par, "mg_s_parent");  s_cb_ = to_device(cb, "mg_s_cb");  s_ce_ = to_device(ce, "mg_s_ce");
  s_child_ = to_device(chv.empty() ? std::vector<int>{0} : chv, "mg_s_child");
  s_colcells_ = to_device(colcells, "mg_s_colcells");
  s_W_ = View1D("mg_s_W", std::max(Ne, 1));  s_D_ = View1D("mg_s_D", Nc);
  s_x_ = View1D("mg_s_x", Nc);  s_b_ = View1D("mg_s_b", Nc);  s_work_ = View1D("mg_s_work", Nc);

  //the small levels' W, D, x, b, work become (unmanaged) views into the flat arrays, so set_coefficients and the
  //restriction from the last large level write directly into the flat storage
  for(int k = 0; k < nsmall_; ++k){
    MGLevel& Lv = lv_[ls_ + k];
    Lv.W    = View1D(s_W_.data() + edge_off[k], edge_off[k + 1] - edge_off[k]);
    Lv.D    = View1D(s_D_.data() + cell_off[k], Lv.n);
    Lv.x    = View1D(s_x_.data() + cell_off[k], Lv.n);
    Lv.b    = View1D(s_b_.data() + cell_off[k], Lv.n);
    Lv.work = View1D(s_work_.data() + cell_off[k], Lv.n);
  }
}


//Athena++ RestrictCoefficients (volume average) + CalculateMatrix on every level: here C is summed (it is the
//integrated time term), kappa is volume averaged, the face weights and diagonals are rebuilt from them
void MGConduction::set_coefficients(){
  for(std::size_t l = 0; l < lv_.size(); ++l){
    MGLevel& L = lv_[l];
    View1D C = L.C, kap = L.kap, vol = L.vol, D = L.D, W = L.W, g = L.g;
    IdxView row_ptr = L.row_ptr, col = L.col;
    if(l > 0){
      const MGLevel& F = lv_[l - 1];
      View1D Cf = F.C, kf = F.kap, vf = F.vol;
      IdxView cp = L.child_ptr, ch = L.child;
      Kokkos::parallel_for("MGRestrictCoeff", Kokkos::RangePolicy<ExecutionSpace>(0, L.n), KOKKOS_LAMBDA(int c){
        Real sc = 0.0, sk = 0.0;
        for(int k = cp(c); k < cp(c + 1); ++k){ int f = ch(k); sc += Cf(f); sk += vf(f)*kf(f); }
        C(c) = sc;
        kap(c) = sk/vol(c);
      });
    }
    Kokkos::parallel_for("MGMatrix", Kokkos::RangePolicy<ExecutionSpace>(0, L.n), KOKKOS_LAMBDA(int c){
      Real d = C(c);
      for(int k = row_ptr(c); k < row_ptr(c + 1); ++k){
        Real w = 0.5*(kap(c) + kap(col(k)))*g(k); //face coefficient = mean of the two cells (Athena++)
        W(k) = w;
        d += w;
      }
      D(c) = d;
    });
  }
  Kokkos::fence();
}


void MGConduction::max_face_gradient(const View1D& v, const View1D& grad){
  const MGLevel& L = lv_[0];
  View1D dist = L.dist;
  IdxView row_ptr = L.row_ptr, col = L.col;
  Kokkos::parallel_for("MGMaxGradient", Kokkos::RangePolicy<ExecutionSpace>(0, L.n), KOKKOS_LAMBDA(int c){
    Real gmax = 0.0;
    for(int k = row_ptr(c); k < row_ptr(c + 1); ++k) gmax = Kokkos::fmax(gmax, Kokkos::fabs(v(c) - v(col(k)))/dist(k));
    grad(c) = gmax;
  });
}


void MGConduction::apply(const View1D& x, const View1D& y, bool with_C){
  const MGLevel& L = lv_[0];
  View1D C = L.C, D = L.D, W = L.W;
  IdxView row_ptr = L.row_ptr, col = L.col;
  Kokkos::parallel_for("MGApply", Kokkos::RangePolicy<ExecutionSpace>(0, L.n), KOKKOS_LAMBDA(int c){
    Real s = (with_C ? D(c) : D(c) - C(c))*x(c);
    for(int k = row_ptr(c); k < row_ptr(c + 1); ++k) s -= W(k)*x(col(k));
    y(c) = s;
  });
}


//Athena++ "jacobi-rb" smoother (mg_crdiffusion.cpp Smooth): pass 1 computes the Jacobi value of every cell of one
//colour from the current neighbour values, pass 2 relaxes u += omega*(work - u). Two passes because on composite
//levels cells of the same colour can be neighbours across fine/coarse faces.
void MGConduction::smooth(int l, int color){
  MGLevel& L = lv_[l];
  View1D x = L.x, b = L.b, D = L.D, W = L.W, work = L.work;
  IdxView row_ptr = L.row_ptr, col = L.col, cells = L.color_cells[color];
  Real om = omega;
  int nc = (int)cells.extent(0);
  Kokkos::parallel_for("MGSmooth1", Kokkos::RangePolicy<ExecutionSpace>(0, nc), KOKKOS_LAMBDA(int k){
    int c = cells(k);
    Real s = b(c);
    for(int e = row_ptr(c); e < row_ptr(c + 1); ++e) s += W(e)*x(col(e));
    work(c) = s/D(c);
  });
  Kokkos::parallel_for("MGSmooth2", Kokkos::RangePolicy<ExecutionSpace>(0, nc), KOKKOS_LAMBDA(int k){
    int c = cells(k);
    x(c) += om*(work(c) - x(c));
  });
}


//Athena++ CalculateDefectBlock + RestrictBlock fused: the coarse source is the sum of the children's defects
//r = b - A x (integrated form of the volume average), computed on the fly. Also zeroes the coarse solution (the coarse
//V-cycle starts from zero), so no separate fill kernel is needed.
void MGConduction::defect_restrict(int l){
  MGLevel& L = lv_[l];
  MGLevel& Cl = lv_[l + 1];
  View1D x = L.x, b = L.b, D = L.D, W = L.W, bc = Cl.b, xc = Cl.x;
  IdxView row_ptr = L.row_ptr, col = L.col, cp = Cl.child_ptr, ch = Cl.child;
  Kokkos::parallel_for("MGDefectRestrict", Kokkos::RangePolicy<ExecutionSpace>(0, Cl.n), KOKKOS_LAMBDA(int c){
    Real sum = 0.0;
    for(int k = cp(c); k < cp(c + 1); ++k){
      int f = ch(k);
      Real r = b(f) - D(f)*x(f);
      for(int e = row_ptr(f); e < row_ptr(f + 1); ++e) r += W(e)*x(col(e));
      sum += r;
    }
    bc(c) = sum;
    xc(c) = 0.0;
  });
}


//the V-cycle of the small levels ls_ .. L-1 in ONE team kernel (team barriers between the steps), same operations
//in the same order as vcycle(): pre-smoothing colours 0,1 -> defect + restriction -> ... -> coarsest level smoothed
//ncoarsest colour pairs forward and in reverse (Athena++ SolveCoarsestGrid) -> prolongation + post-smoothing 1,0.
//Expects b and x = 0 of level ls_ to be set (by defect_restrict of the level above, or by the caller).
void MGConduction::small_vcycle(){
  IdxView rb = s_rb_, re = s_re_, col = s_col_, par = s_parent_, cb = s_cb_, ce = s_ce_, child = s_child_,
          colcells = s_colcells_;
  View1D W = s_W_, D = s_D_, x = s_x_, b = s_b_, work = s_work_;
  auto coff = s_cell_off_;
  auto kof = s_color_off_;
  int K = nsmall_, npre_ = npre, npost_ = npost, nsweep = ncoarsest;
  Real om = omega;
  using member_t = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  Kokkos::parallel_for("MGSmallVcycle", Kokkos::TeamPolicy<ExecutionSpace>(1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const member_t& t){
      //jacobi-rb sweep of one colour of level k (two passes, see smooth())
      auto sweep = [&](int k, int color){
        int c0 = kof[2*k + color], c1 = kof[2*k + color + 1];
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, c0, c1), [&](int i){
          int c = colcells(i);
          Real s = b(c);
          for(int e = rb(c); e < re(c); ++e) s += W(e)*x(col(e));
          work(c) = s/D(c);
        });
        t.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, c0, c1), [&](int i){
          int c = colcells(i);
          x(c) += om*(work(c) - x(c));
        });
        t.team_barrier();
      };
      //down
      for(int k = 0; k < K - 1; ++k){
        for(int s = 0; s < npre_; ++s){ sweep(k, 0); sweep(k, 1); }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, coff[k + 1], coff[k + 2]), [&](int c){
          Real sum = 0.0;
          for(int i = cb(c); i < ce(c); ++i){
            int f = child(i);
            Real r = b(f) - D(f)*x(f);
            for(int e = rb(f); e < re(f); ++e) r += W(e)*x(col(e));
            sum += r;
          }
          b(c) = sum;
          x(c) = 0.0;
        });
        t.team_barrier();
      }
      //coarsest
      for(int s = 0; s < nsweep; ++s){ sweep(K - 1, 0); sweep(K - 1, 1); }
      for(int s = 0; s < nsweep; ++s){ sweep(K - 1, 1); sweep(K - 1, 0); }
      //up
      for(int k = K - 2; k >= 0; --k){
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, coff[k], coff[k + 1]), [&](int c){ x(c) += x(par(c)); });
        t.team_barrier();
        for(int s = 0; s < npost_; ++s){ sweep(k, 1); sweep(k, 0); }
      }
    });
}


//V-cycle from zero initial guess on level l (Athena++ OneStepToCoarser / SolveCoarsestGrid / OneStepToFiner).
//Pre-smoothing colours 0,1, post-smoothing 1,0 (reverse) -> symmetric preconditioner.
//Expects b and x = 0 of level l to be set.
void MGConduction::vcycle(int l){
  if(l == ls_){ small_vcycle(); return; }
  MGLevel& L = lv_[l];
  MGLevel& Cl = lv_[l + 1];
  for(int s = 0; s < npre; ++s){ smooth(l, 0); smooth(l, 1); }
  defect_restrict(l);
  vcycle(l + 1);
  //prolongation (piecewise constant) and correction
  {
    View1D x = L.x, xc = Cl.x;
    IdxView parent = L.parent;
    Kokkos::parallel_for("MGProlongate", Kokkos::RangePolicy<ExecutionSpace>(0, L.n), KOKKOS_LAMBDA(int c){
      x(c) += xc(parent(c));
    });
  }
  for(int s = 0; s < npost; ++s){ smooth(l, 1); smooth(l, 0); }
}


void MGConduction::precondition(const View1D& r, const View1D& z){
  Kokkos::deep_copy(lv_[0].b, r);
  Kokkos::deep_copy(lv_[0].x, 0.0);
  vcycle(0);
  Kokkos::deep_copy(z, lv_[0].x);
}


//y = A x on the leaf level and returns x.y (fused: one kernel, one host sync)
Real MGConduction::apply_dot(const View1D& x, const View1D& y){
  const MGLevel& L = lv_[0];
  View1D D = L.D, W = L.W;
  IdxView row_ptr = L.row_ptr, col = L.col;
  Real xy = 0.0;
  Kokkos::parallel_reduce("MGApplyDot", Kokkos::RangePolicy<ExecutionSpace>(0, L.n), KOKKOS_LAMBDA(int c, Real& acc){
    Real s = D(c)*x(c);
    for(int k = row_ptr(c); k < row_ptr(c + 1); ++k) s -= W(k)*x(col(k));
    y(c) = s;
    acc += x(c)*s;
  }, xy);
  return xy;
}


//CG works in place on the leaf level of the multigrid: the residual r IS the V-cycle's source lv_[0].b and the
//preconditioned residual z IS its solution lv_[0].x, so the preconditioner needs no copies. z is zeroed in the
//update kernel (the next V-cycle starts from zero) once p has been built from it.
int MGConduction::solve(const View1D& b, const View1D& x, Real tol, int max_iter, bool& converged, Real x_tol){
  int n = lv_[0].n;
  View1D r = lv_[0].b, z = lv_[0].x, p = cg_p_, Ap = cg_Ap_;
  Kokkos::deep_copy(x, 0.0);
  Kokkos::deep_copy(r, b);
  Real bb = dot(b, b, n);
  converged = (bb == 0.0);
  if(converged) return 0;

  Kokkos::deep_copy(z, 0.0);
  vcycle(0);
  Kokkos::deep_copy(p, z);
  Real rz = dot(r, z, n);
  int it = 0;
  while(it < max_iter){
    Real alpha = rz/apply_dot(p, Ap);
    Real rr = 0.0, dx_max = 0.0;
    Kokkos::parallel_reduce("CGUpdate", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(int i, Real& acc, Real& dmax){
        Real dx = alpha*p(i);
        x(i) += dx;
        Real ri = r(i) - alpha*Ap(i);
        r(i) = ri;
        z(i) = 0.0;
        acc += ri*ri;
        dmax = Kokkos::fmax(dmax, Kokkos::fabs(dx));
      }, rr, Kokkos::Max<Real>(dx_max));
    ++it;
    if(rr <= tol*tol*bb || dx_max <= x_tol){ converged = true; break; }
    vcycle(0);
    Real rz_new = dot(r, z, n);
    Real beta = rz_new/rz;
    rz = rz_new;
    Kokkos::parallel_for("CGDirection", Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i){
      p(i) = z(i) + beta*p(i);
    });
  }
  return it;
}
