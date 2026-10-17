#include "grid.hpp"
#include "mesh.hpp"
#include <stdexcept>

//validates that periodic boundary conditions are symmetrically paired
//across opposing domain walls (left/right and bottom/top).
void validateBC(const GridBC& bc){
  if((bc.left.type == BCType::Periodic) != (bc.right.type == BCType::Periodic))
    throw std::invalid_argument("Periodic BC must be applied to both left and right walls");
  if((bc.bottom.type == BCType::Periodic) != (bc.top.type == BCType::Periodic))
    throw std::invalid_argument("Periodic BC must be applied to both bottom and top walls");
}


//static (file-local)
KOKKOS_INLINE_FUNCTION
static void applyHydrostaticOpenBC(const View4D& data, int b, int i_ghost, int i_phys, int j,
                                   Real z_ghost, Real z_phys, Real z0, Real Omega, Real cs2, Real gamma){
  //discrete hydrostatic ratio (Käppeli et al. 2016, Eq. 18)
  //isothermal, K = cs2 = p/rho (same K as init_cond's gaussian_z)
  //phi = 0.5*Omega^2*x^2, K = cs2, ratio = (2K - Dphi)/(2K + Dphi)
  Real K     = cs2;
  Real Dphi  = phi_diff(z_ghost, z_phys, z0, Omega); //local z, z0 = absolute origin
  Real ratio = (2.0*K - Dphi)/(2.0*K + Dphi);

  //sűrűség és impulzusok skálázása
  data(b, i_ghost, j, 0) = data(b, i_phys, j, 0)*ratio;
  data(b, i_ghost, j, 1) = data(b, i_phys, j, 1)*ratio; // u-momentum
  data(b, i_ghost, j, 2) = data(b, i_phys, j, 2)*ratio; // v-momentum

  //belső energia skálázása (p_g = p_p * ratio, mert izoterm)
  //először kiszámoljuk a fizikai cella nyomását
  Real rho_p   = data(b, i_phys, j, 0);
  Real u_p     = data(b, i_phys, j, 1)/rho_p;
  Real v_p     = data(b, i_phys, j, 2)/rho_p;
  Real e_kin_p = 0.5*rho_p*(u_p*u_p + v_p*v_p);
  Real p_p     = (gamma - 1.0)*(data(b, i_phys, j, 3) - e_kin_p);

  Real p_ghost = p_p*ratio;
  //impulzusok már skálázva vannak fent, az új ghost sűrűséggel számoljuk az energiát
  Real rho_g = data(b, i_ghost, j, 0);
  //kinetikus energia
  Real e_kin_g = 0.5*(data(b, i_ghost, j, 1)*data(b, i_ghost, j, 1) +
                      data(b, i_ghost, j, 2)*data(b, i_ghost, j, 2))/rho_g;

  data(b, i_ghost, j, 3) = p_ghost/(gamma - 1.0) + e_kin_g;
}


//applies physical boundary conditions across all active mesh blocks that
//border the external domain boundaries using a Kokkos parallel kernel.
//Supports Open, Closed (reflecting momentum), and Dirichlet BCs.
//The x1 wall positions come from the mesh domain and the spacing
//from each block's own dx1, so it is correct for multiple / refined blocks.
//The declaration in mesh.hpp matches this.
void ApplyPhysicalBoundaryConditions(Mesh& mesh, const GridBC& bc, Real Omega, Real cs2, Real gamma){
  validateBC(bc);

  int n_active = mesh.num_active_blocks();
  if (n_active == 0) return;

  View4D data = mesh.get_data_view();
  IntView2D n_level = mesh.get_neighbor_level_view();
  //per-block cell size (refined blocks have a smaller dx1 than the root grid)
  View1D dx1_view = mesh.get_block_dx1_view();

  int nx1 = mesh.inputs().nx1_block;
  int nx2 = mesh.inputs().nx2_block;
  int nghost = mesh.inputs().nghost;
  int is = nghost, ie = nghost + nx1 - 1;
  int js = nghost, je = nghost + nx2 - 1;
  //a block with a physical XM / XP wall starts / ends exactly at the domain edge
  Real zmin = mesh.inputs().domain.x1min;
  Real zmax = mesh.inputs().domain.x1max;
  Real z0   = mesh.inputs().x1_origin;

  //boundary conditions packed into POD for lambda capture
  GridBC bc_copy = bc;

  //split into two kernels: Pass 1 fills the x1 ghosts for interior j only; pass 2 fills the x2 ghosts
  //for ALL i (ghost i included), so corners are filled once from already-set x1 ghosts. In the old single kernel
  //corner cells were written by the left/right and bottom/top branches at the same time (data race).

  //pass 1: left / right walls
  Kokkos::parallel_for(
    "ApplyPhysicalBC_x1",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n_active, nx2}),
    KOKKOS_LAMBDA(const int b, const int t){
      int j = js + t;
      Real dz = dx1_view(b);

      //Left boundary
      if(n_level(b, (int)FaceDir::XM) == (int)NeighborLevel::PhysicalBoundary){
        for(int g = 0; g < nghost; ++g){
          int ghost_i = is - 1 - g;
          switch(bc_copy.left.type){
            case BCType::Open:{
              //one ghost layer per g, each built from the cell just inside it (is -> is-1 -> is-2 ...),
              //instead of hardcoded indices 0,1,2 rewritten on every g iteration
              int  ref_i   = ghost_i + 1;
              Real z_ref   = zmin + (ref_i - is + 0.5)*dz;
              Real z_ghost = z_ref - dz;
              applyHydrostaticOpenBC(data, b, ghost_i, ref_i, j, z_ghost, z_ref, z0, Omega, cs2, gamma);
              break;
            }
            case BCType::Closed:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = data(b, is + g, j, k);
              data(b, ghost_i, j, 1) *= -1.0;
              break;
            case BCType::Dirichlet:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = bc_copy.left.Q_fixed[k];
              break;
            default: break;
          }
        }
      }

      //Right
      if(n_level(b, (int)FaceDir::XP) == (int)NeighborLevel::PhysicalBoundary){
        for (int g = 0; g < nghost; ++g){
          int ghost_i = ie + 1 + g;
          switch(bc_copy.right.type){
            case BCType::Open:{
              //block-local indices ie+1, ie+2...
              //and each ghost is built from the cell just inside it
              int  ref_i   = ghost_i - 1;
              Real z_ref   = zmax - (ie - ref_i + 0.5)*dz;
              Real z_ghost = z_ref + dz;
              applyHydrostaticOpenBC(data, b, ghost_i, ref_i, j, z_ghost, z_ref, z0, Omega, cs2, gamma);
              break;
            }
            case BCType::Closed:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = data(b, ie - g, j, k);
              data(b, ghost_i, j, 1) *= -1.0;
              break;
            case BCType::Dirichlet:
              for(size_t k = 0; k < NVAR; ++k) data(b, ghost_i, j, k) = bc_copy.right.Q_fixed[k];
              break;
            default: break;
          }
        }
      }
    }
  );

  //pass 2: bottom / top walls (runs after pass 1, same execution space instance -> ordered)
  Kokkos::parallel_for(
    "ApplyPhysicalBC_x2",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n_active, nx1 + 2*nghost}),
    KOKKOS_LAMBDA(const int b, const int t){
      int i = is - nghost + t;

      //Bottom
      if(n_level(b, (int)FaceDir::YM) == (int)NeighborLevel::PhysicalBoundary){
        for(int g = 0; g < nghost; ++g){
          int ghost_j = js - 1 - g;
          switch(bc_copy.bottom.type){
            case BCType::Open:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, js, k);
              break;
            case BCType::Closed:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, js + g, k);
              data(b, i, ghost_j, 2) *= -1.0;
              break;
            case BCType::Dirichlet:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = bc_copy.bottom.Q_fixed[k];
              break;
            default: break;
          }
        }
      }

      //Top
      if(n_level(b, (int)FaceDir::YP) == (int)NeighborLevel::PhysicalBoundary){
        for(int g = 0; g < nghost; ++g){
          int ghost_j = je + 1 + g;
          switch(bc_copy.top.type){
            case BCType::Open:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, je, k);
              break;
            case BCType::Closed:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = data(b, i, je - g, k);
              data(b, i, ghost_j, 2) *= -1.0;
              break;
            case BCType::Dirichlet:
              for(size_t k = 0; k < NVAR; ++k) data(b, i, ghost_j, k) = bc_copy.top.Q_fixed[k];
              break;
            default: break;
          }
        }
      }
    }
  );
}
