#include "../headers/grid.h"
#include <cmath>
#include <stdexcept>

/*
 Grid BC application for 2D Grids
 Bertalan Szuchovszky 02.03.2026

 Grid, BCType, Cell, GridBC, BoundaryCondition are defined in the grid.h header file

 validateBC: checks if the Periodic BC was set on a pair of opposing walls - if not, error message
  Input:
    GridBC
  Output:
    nothing, throws error message if periodic BC was set incorrectly

 appyBC: applies chosen BCType on a specific wall allowing for different BC at different walls
  -> Open:      what goes out the wall vanishes (Neumann BC: derivative = 0 at walls)
  -> Closed:    what collides with the wall bounces back (normal momentum flipping)
  -> Periodic:  what goes out one wall comes back on the opposing side
  -> Dirichlet: given constant state vector at walls
  Input:
    Grid & GridBC - the whole grid and the boundary conditions
  Output:
    nothing, modifies the state vector at grid walls depending on BC type
*/


void validateBC(const GridBC& bc) { //If periodic, the other wall needs to be set to periodic BC aswell
    if ((bc.left.type == BCType::Periodic) != (bc.right.type == BCType::Periodic))
        throw std::invalid_argument("Periodic BC must be applied to both left and right walls");
    if ((bc.bottom.type == BCType::Periodic) != (bc.top.type == BCType::Periodic))
        throw std::invalid_argument("Periodic BC must be applied to both bottom and top walls");
}



void applyHydrostaticOpenBC(Grid& grid, size_t i_ghost, size_t i_phys, size_t j, 
                            double z_ghost, double z_phys, double Omega, double cs2, double gamma) {
  //discrete hydrostatic ratio (Käppeli et al. 2016, Eq. 18)
  //phi = 0.5*Omega^2*x^2, K = cs2/gamma, ratio = (2K - Dphi)/(2K + Dphi)
  double K     = cs2;
  double Dphi  = 0.5 * Omega * Omega * (z_ghost*z_ghost - z_phys*z_phys);
  double ratio = (2.0*K - Dphi) / (2.0*K + Dphi);

  //sűrűség és impulzusok skálázása
  grid(i_ghost, j, 0) = grid(i_phys, j, 0) * ratio;
  grid(i_ghost, j, 1) = grid(i_phys, j, 1) * ratio; // u-momentum
  grid(i_ghost, j, 2) = grid(i_phys, j, 2) * ratio; // v-momentum

  //belső energia skálázása (p_g = p_p * ratio, mert izoterm)
  //először kiszámoljuk a fizikai cella nyomását
  double rho_p   = grid(i_phys, j, 0);
  double u_p     = grid(i_phys, j, 1) / rho_p;
  double v_p     = grid(i_phys, j, 2) / rho_p;
  double e_kin_p = 0.5 * rho_p * (u_p*u_p + v_p*v_p);
  double p_p     = (gamma - 1.0) * (grid(i_phys, j, 3) - e_kin_p);

  double p_ghost = p_p * ratio;
  //impulzusok már skálázva vannak fent, az új ghost sűrűséggel számoljuk az energiát
  double rho_g = grid(i_ghost, j, 0);
  double e_kin_g = 0.5 * (grid(i_ghost, j, 1)*grid(i_ghost, j, 1) + 
                          grid(i_ghost, j, 2)*grid(i_ghost, j, 2) / rho_g);

  grid(i_ghost, j, 3) = p_ghost / (gamma - 1.0) + e_kin_g;
}



void applyBC(Grid& grid, const GridBC& bc, double Omega, double cs2, double gamma, double dz, double zmin) {
  validateBC(bc); //check if Periodic BC was used correctly
  size_t Nz = grid.rows() - 4; //ghost points not included
  size_t Nr = grid.cols() - 4;
  //every wall has a BC -> switch case for every wall depending on BCType
  //then just apply the BC at the walls

  //left wall
  for (size_t j = 0; j < grid.cols(); ++j) {
    switch (bc.left.type) {
      case BCType::Open:{
        double z_p =  zmin + 0.5 * dz; 
        double z_g1 = z_p - dz;       //i = 1
        double z_g0 = z_p - 2.0 * dz; //i = 0
        applyHydrostaticOpenBC(grid, 1, 2, j, z_g1, z_p, Omega, cs2, gamma);
        applyHydrostaticOpenBC(grid, 0, 1, j, z_g0, z_g1, Omega, cs2, gamma);
        break;
      }
      case BCType::Closed:
        grid.copyCell(1,j, 2,j); grid(1,j,1) *=-1.0; //flip normal momentum rho*u
        grid.copyCell(0,j, 3,j); grid(0,j,0) *=-1.0;
        break;
      case BCType::Periodic:
        grid.copyCell(0,j, Nz,  j); //goes out this side enters at the other side
        grid.copyCell(1,j, Nz+1,j);
        break;
      case BCType::Dirichlet:
        grid.setCell(0,j, bc.left.Q_fixed); //const BC value
        grid.setCell(1,j, bc.left.Q_fixed);
        break;
    }
  }
  //right wall
  for (size_t j = 0; j < grid.cols(); ++j) {
    switch (bc.right.type) {
      case BCType::Open:{
        double z_p = zmin + (Nz + 1 - 2) * dz + 0.5 * dz; //last phys cell (Nz+1)
        double z_g2 = z_p + dz;       //i = Nz+2
        double z_g3 = z_p + 2.0 * dz; //i = Nz+3
        applyHydrostaticOpenBC(grid, Nz+2, Nz+1, j, z_g2, z_p, Omega, cs2, gamma);
        applyHydrostaticOpenBC(grid, Nz+3, Nz+1, j, z_g3, z_g2, Omega, cs2, gamma);
        break;
      }
      case BCType::Closed:
        grid.copyCell(Nz+2,j, Nz+1,j); grid(Nz+2,j,1) *= -1.0;
        grid.copyCell(Nz+3,j, Nz,  j); grid(Nz+3,j,1) *= -1.0;
        break;
      case BCType::Periodic:
        grid.copyCell(Nz+2,j, 2,j);
        grid.copyCell(Nz+3,j, 3,j);
        break;
      case BCType::Dirichlet:
        grid.setCell(Nz+2,j, bc.right.Q_fixed);
        grid.setCell(Nz+3,j, bc.right.Q_fixed);
        break;
    }
  }
  //bottom wall
  for (size_t i = 0; i < grid.rows(); ++i) {
    switch (bc.bottom.type) {
      case BCType::Open:
        grid.copyCell(i,0, i,2);
        grid.copyCell(i,1, i,2);
        break;
      case BCType::Closed:
        grid.copyCell(i,1, i,2); grid(i,1,2) *= -1.0;
        grid.copyCell(i,0, i,3); grid(i,0,2) *= -1.0;
        break;
      case BCType::Periodic:
        grid.copyCell(i,0, i,Nr);
        grid.copyCell(i,1, i,Nr+1);
        break;
      case BCType::Dirichlet:
        grid.setCell(i,0, bc.bottom.Q_fixed);
        grid.setCell(i,1, bc.bottom.Q_fixed);
        break;
    }
  }
  //top wall
  for (size_t i = 0; i < grid.rows(); ++i) {
    switch (bc.top.type) {
      case BCType::Open:
        grid.copyCell(i,Nr+2, i,Nr+1);
        grid.copyCell(i,Nr+3, i,Nr+1);
        break;
      case BCType::Closed:
        grid.copyCell(i,Nr+2, i,Nr+1); grid(i,Nr+2,2) *= -1.0;
        grid.copyCell(i,Nr+3, i,Nr  ); grid(i,Nr+3,2) *= -1.0;
        break;
      case BCType::Periodic:
        grid.copyCell(i,Nr+2, i,2);
        grid.copyCell(i,Nr+3, i,3);
        break;
      case BCType::Dirichlet:
        grid.setCell(i,Nr+2, bc.top.Q_fixed);
        grid.setCell(i,Nr+3, bc.top.Q_fixed);
        break;
    }
  }
}
