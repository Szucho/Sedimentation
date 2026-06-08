#include <cmath>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include "../headers/matrix.h"
#include "../headers/grid.h"
#include "../headers/HLLC.h"
#include "../headers/slopelim.h"
#include "../headers/particle.h"
#include "../headers/io.h"

using namespace VecOps;
constexpr double PI = 3.141592653589793;
constexpr double k_B = 1.380649e-16;  //erg/K
constexpr double m_p = 1.672622e-24;  //g

/*
 2D Hydrodynamics HLLC Euler Equation solver
 Bertalan Szuchovszky 03.03.2026

 Solves the 2D compressible Euler equations in conservative form
 on a uniform Cartesian grid using the HLLC approximate Riemann solver (see HLLC.cpp)
 with Superbee slope-limited interpolation at cell walls (see slope_limiters.cpp).

 Equation to be solved: dQ/dt + dF/dQ * div Q = 0 with dF/dQ = J Jacobi matrix 
 State vector: Q = [rho, rho*u, rho*v, rho*e_tot]
               e_tot = e_th + 0.5*(u^2+v^2),  e_th = 1/(gamma-1) * k_BT/(mu*m_p) - ideal gas law

 Spatial discretization: 2nd order finite volume (~MUSCL-Hancock, approximate Riemann solver)
 Time integration: Explicit Euler
 => 2nd order FV in space & 1st order in time
 Boundary conditions: Open, Closed, Periodic, Dirichlet (user specified)

 Usage: set init_cond() for your problem (DON'T FORGET THIS), then build and run.
 Build: clang++ -std=c++17 -O3 -march=native -I./headers sources/HLLC.cpp sources/slope_limiters.cpp sources/grid_setup.cpp sources/particle.cpp sources/main.cpp -o builds/solver_name
 No CMake yet as I can't be bothered to write one
 Validated on Sod shock tube (Toro, Chapter 4)

 THE CODE SERVES AS A BASE THAT ONE CAN MODIFY TO SUIT THEIR OWN PROJECT

 MODIFIED from 20.04.2026 - 07.05.2026
  - implemented well balanced method so hydrostatic equilibrium could be achieved
  - added particle.cpp & corresponding header for solid particle making this a Lagrangian-Eulerian scheme
  - added io.h that handles in- / output (file writing etc.)
  - tested well balanced method on particle sedimentation - the whole LE scheme is working properly
  - added heat diffusion TO DO: modify io.h to require kappa, test it

 MODIFIED 10.05 (maybe) - 28.05.2026
  - added logarithmic writing
  - implemented and tested the implicit + well-balanced B2 method found in Batten et al. 1997
*/


//FIRST: CFL condition -> need to check if dt > dt_{cfl}
double cfl_dt(const Grid& grid, double dz, double dr, double gamma, double CFL_max=0.3) {
  size_t nz = grid.rows() - 4;
  size_t nr = grid.cols() - 4;
  double smax_z = 0.0, smax_r = 0.0; //s characteristics
  for (size_t i = 2; i < nz+2; i++) {
    for (size_t j = 2; j < nr+2; j++) {
      Cell c = grid.getCell(i,j); //c contains [rho, rho u, rho v, rho e_tot]
      double rho = c[0];
      double u   = c[1]/rho;
      double v   = c[2]/rho;
      double p   = (gamma-1.0)*(c[3] - 0.5*rho*(u*u+v*v));
      double cs  = std::sqrt(gamma*p/rho); //adiabatic soundspeed
      smax_z = std::max(smax_z, std::abs(u)+cs);
      smax_r = std::max(smax_r, std::abs(v)+cs);
    }
  }
  return CFL_max / (smax_z/dz + smax_r/dr);
}


double diffuse_dt(const Grid& grid, double dz, double dr, 
                  double gamma, double mu, double kappa_func(double T)){

  size_t nz = grid.rows()-4;
  size_t nr = grid.cols()-4;
  double dt_min = std::numeric_limits<double>::max();//numeric error minimum
  for (size_t i=2; i<nz+2; i++){
    for (size_t j=2; j<nr+2; j++){
      Vector Q = CellToVec(grid.getCell(i, j));
      double rho = Q[0];
      double u   = Q[1]/rho;
      double v   = Q[2]/rho;
      double p   = (gamma-1.0)*(Q[3]-0.5*rho*(u*u-v*v));
      double T   = p*m_p*mu/(rho*k_B);
      double kappa = kappa_func(T);
      double cv = k_B/(mu*m_p*(gamma-1.0));
      double alpha = kappa/(rho*cv); //diffusivity
      double dt_cell = 1.0/(2.0*alpha*(1.0/(dz*dz)+1.0/(dr*dr)));
      dt_min = std::min(dt_min, dt_cell);
    }
  }
  return dt_min;
}


static Vector grav_source(const Vector& Q, double z_im1, double z_ip1, double Omega, double dz) {
  double rho = Q[0];
  double u   = Q[1] / rho;
  Vector S(4, 0.0);
  S[1] = -0.5*rho * Omega*Omega * (z_ip1*z_ip1-z_im1*z_im1)/(2.0*dz);
  S[3] = u*S[1];
  return S;
}


//HLLC step of a single grid point Q_{ij}
Vector HLLC_step(const Vector& Qi, 
                 const Vector& Flux_q1iph, 
                 const Vector& Flux_q2iph, 
                 const Vector& Flux_q1imh, 
                 const Vector& Flux_q2imh,
                 const Vector& S,
                 double dq1, double dq2, double dt){
  Vector Qi_new; //return the new state vector at ij gridpoint
  Qi_new = Qi - dt/dq1*(Flux_q1iph - Flux_q1imh) - dt/dq2*(Flux_q2iph-Flux_q2imh); //Euler timestep
  Qi_new = Qi_new + S; //OPEREATOR SPLITTING this source is the back reaction of the particle
  return Qi_new;
}


template <typename Func>
Grid timestep(const Grid& grid,
              const Grid& S_grid,
              double zmin,
              double dz, double dr, double dt,
              double gamma, double Omega,
              double mu, Func kappa_func){
  size_t nz = grid.rows();
  size_t nr = grid.cols();
  Grid grid_new(nz, nr);


  //helper for temperature diffusion
  auto getT = [&](size_t i, size_t j) -> double {
    auto Q = CellToVec(grid.getCell(i,j));
    double rho = Q[0], u = Q[1]/rho, v = Q[2]/rho;
    double p = (gamma-1.0)*(Q[3] - 0.5*rho*(u*u+v*v));
    return p*m_p*mu/(rho*k_B); //since p = rho/(mu*m_p)*k_B*T
  };


  for (size_t i = 2; i<nz-2; i++){
    double z_i   = zmin + (i-2)*dz + 0.5*dz;  //cell center i, BC ghost cells are in numerical equilibrium too
    double z_im1 = z_i - dz;
    double z_ip1 = z_i + dz;
    double z_im2 = z_im1 - dz;
    double z_ip2 = z_ip1 + dz;
    for(size_t j = 2; j<nr-2; j++){
      Vector Qi = CellToVec(grid.getCell(i,j)); //bunch of Q_{ij} vals needed for the slope calculations 
      Vector Qim1 = CellToVec(grid.getCell(i-1,j));
      Vector Qim2 = CellToVec(grid.getCell(i-2,j));
      Vector Qip1 = CellToVec(grid.getCell(i+1,j));
      Vector Qip2 = CellToVec(grid.getCell(i+2,j));
      Vector Qjm1 = CellToVec(grid.getCell(i,j-1));
      Vector Qjm2 = CellToVec(grid.getCell(i,j-2));
      Vector Qjp1 = CellToVec(grid.getCell(i,j+1));
      Vector Qjp2 = CellToVec(grid.getCell(i,j+2));

      //slopes in the x direction
      Vector sigma_im1_z, sigma_i_z, sigma_ip1_z;

      if (i == 2) {
        sigma_im1_z = Vector(4, 0.0);
        sigma_i_z   = Vector(4, 0.0);
        sigma_ip1_z = sigma_minmod(Qi, Qip1, Qip2, dz);
      } else if (i == nz-3) {
        sigma_im1_z = sigma_minmod(Qim2, Qim1, Qi, dz);
        sigma_i_z   = Vector(4, 0.0);
        sigma_ip1_z = Vector(4, 0.0);
      } else {
        sigma_im1_z = sigma_minmod(Qim2, Qim1, Qi,   dz);
        sigma_i_z   = sigma_minmod(Qim1, Qi,   Qip1, dz);
        sigma_ip1_z = sigma_minmod(Qi,   Qip1, Qip2, dz);
      }

      //slopes in the y direction
      Vector sigma_jm1_r, sigma_j_r, sigma_jp1_r;

      if (j == 2) {
        // első fizikai cella: első rendű, ne használj ghost cellákat slope-hoz
        sigma_jm1_r = Vector(4, 0.0);
        sigma_j_r   = Vector(4, 0.0);
        sigma_jp1_r = sigma_minmod(Qi, Qjp1, Qjp2, dr);
      } else if (j == nr-3) {
        // utolsó fizikai cella
        sigma_jm1_r = sigma_minmod(Qjm2, Qjm1, Qi, dr);
        sigma_j_r   = Vector(4, 0.0);
        sigma_jp1_r = Vector(4, 0.0);
      } else {
        sigma_jm1_r = sigma_minmod(Qjm2, Qjm1, Qi,   dr);
        sigma_j_r   = sigma_minmod(Qjm1, Qi,   Qjp1, dr);
        sigma_jp1_r = sigma_minmod(Qi,   Qjp1, Qjp2, dr);
      }
      
      //q1 interfaces
      Vector QL_imh = Qim1 + 0.5*dz * sigma_im1_z;  //left  state at i-1/2
      Vector QR_imh = Qi   - 0.5*dz * sigma_i_z;    //right state at i-1/2
      Vector QL_iph = Qi   + 0.5*dz * sigma_i_z;    //left  state at i+1/2
      Vector QR_iph = Qip1 - 0.5*dz * sigma_ip1_z;  //right state at i+1/2
      
      //well balanced - Käppeli et al. 2016
      Primitive_Vals prim_im2 = QtoPrim(Qim2, gamma);
      Primitive_Vals prim_im1 = QtoPrim(Qim1, gamma);
      Primitive_Vals prim_i   = QtoPrim(Qi,   gamma);
      Primitive_Vals prim_ip1 = QtoPrim(Qip1, gamma);
      Primitive_Vals prim_ip2 = QtoPrim(Qip2, gamma);

      //phi(x) = 1/2*Omega^2 x^2
      double p0_at_imh = prim_i.p + 0.5*prim_i.rho*(0.5*Omega*Omega*(z_i*z_i - z_im1*z_im1)); 
      double p0_at_iph = prim_i.p - 0.5*prim_i.rho*(0.5*Omega*Omega*(z_ip1*z_ip1 - z_i*z_i));
      double p0_at_im1h = prim_im1.p - 0.5*prim_im1.rho*(0.5*Omega*Omega*(z_i*z_i - z_im1*z_im1));
      double p0_at_ip1h = prim_ip1.p + 0.5*prim_ip1.rho*(0.5*Omega*Omega*(z_ip1*z_ip1 - z_i*z_i));

      double p0_at_im1 = prim_i.p + 0.5*(prim_im1.rho+prim_i.rho) * 0.5*Omega*Omega*(z_i*z_i - z_im1*z_im1);
      double p0_at_ip1 = prim_i.p - 0.5*(prim_ip1.rho+prim_i.rho) * 0.5*Omega*Omega*(z_ip1*z_ip1 - z_i*z_i);
      double p0_at_im2 = prim_im1.p + 0.5*(prim_im2.rho+prim_im1.rho) * 0.5*Omega*Omega*(z_im1*z_im1 - z_im2*z_im2);
      double p0_at_ip2 = prim_ip1.p - 0.5*(prim_ip2.rho+prim_ip1.rho) * 0.5*Omega*Omega*(z_ip2*z_ip2 - z_ip1*z_ip1);
 
      //p1,i(x_i) = 0 by design
      double p1_ip1 = prim_ip1.p - p0_at_ip1;
      double p1_im1 = prim_im1.p - p0_at_im1;
      double p1_im2 = prim_im2.p - p0_at_im2;
      double p1_ip2 = prim_ip2.p - p0_at_ip2;
      //slope with p1,i(x_i)=0
      double sigma_pi = minmod_one(p1_im1, 0.0, p1_ip1, dz);
      double sigma_pim1 = minmod_one(p1_im2, p1_im1, 0.0, dz);
      double sigma_pip1 = minmod_one(0.0, p1_ip1, p1_ip2, dz);


      QL_imh[3] = (p0_at_im1h + 0.5*sigma_pim1*dz)/(gamma-1.0)
                + 0.5*(QL_imh[1]*QL_imh[1]+QL_imh[2]*QL_imh[2])/QL_imh[0];
      QR_imh[3] = (p0_at_imh - 0.5*sigma_pi*dz)/(gamma-1.0)
                + 0.5*(QR_imh[1]*QR_imh[1]+QR_imh[2]*QR_imh[2])/QR_imh[0];
      QR_iph[3] = (p0_at_ip1h - 0.5*sigma_pip1*dz)/(gamma-1.0)
                + 0.5*(QR_iph[1]*QR_iph[1] + QR_iph[2]*QR_iph[2])/QR_iph[0];
      QL_iph[3] = (p0_at_iph + 0.5*sigma_pi*dz)/(gamma-1.0)
                + 0.5*(QL_iph[1]*QL_iph[1] + QL_iph[2]*QL_iph[2])/QL_iph[0];

      //q2 interfaces
      Vector QL_jmh = Qjm1 + 0.5*dr * sigma_jm1_r;  //left  state at j-1/2
      Vector QR_jmh = Qi   - 0.5*dr * sigma_j_r;    //right state at j-1/2
      Vector QL_jph = Qi   + 0.5*dr * sigma_j_r;    //left  state at j+1/2
      Vector QR_jph = Qjp1 - 0.5*dr * sigma_jp1_r;  //right state at j+1/2

      
      //fluxes -> HLLC method (see HLLC.cpp, Toro)
      Vector FZimh = Fluxhllc_q1(QL_imh, QR_imh, gamma);
      Vector FZiph = Fluxhllc_q1(QL_iph, QR_iph, gamma);
      Vector FRjmh = Fluxhllc_q2(QL_jmh, QR_jmh, gamma);
      Vector FRjph = Fluxhllc_q2(QL_jph, QR_jph, gamma); 
       
      //apply HLLC timestep at gridcell Q_{ij}
      Vector Q_new = HLLC_step(Qi, FZiph, FRjph, FZimh, FRjmh, CellToVec(S_grid.getCell(i,j)), dz, dr, dt);
      
      //gravity as source term
      Q_new = Q_new + grav_source(Qi, z_im1, z_ip1, Omega, dz)*dt;

      //dE/dt + div F = div(kappa grad T) + S_particle + S_grav + other sources (maybe in the future)
      //heat diffusion will be added as another source term via operator splitting
      double T_ij   = getT(i,   j);
      double T_ip1  = getT(i+1, j);
      double T_im1  = getT(i-1, j);
      double T_jp1  = getT(i, j+1);
      double T_jm1  = getT(i, j-1);

      //mean kappa between neighbouring cells
      double k_iph = 0.5*(kappa_func(T_ij) + kappa_func(T_ip1));
      double k_imh = 0.5*(kappa_func(T_ij) + kappa_func(T_im1));
      double k_jph = 0.5*(kappa_func(T_ij) + kappa_func(T_jp1));
      double k_jmh = 0.5*(kappa_func(T_ij) + kappa_func(T_jm1));

      // double k_iph = 2.0*kappa_func(T_ij)*kappa_func(T_ip1)/(kappa_func(T_ij) + kappa_func(T_ip1));
      // double k_imh = 2.0*kappa_func(T_im1)*kappa_func(T_ij)/(kappa_func(T_im1) + kappa_func(T_ij));
      // double k_jph = 2.0*kappa_func(T_ij)*kappa_func(T_jp1)/(kappa_func(T_ij) + kappa_func(T_jp1));
      // double k_jmh = 2.0*kappa_func(T_jm1)*kappa_func(T_ij)/(kappa_func(T_jm1) + kappa_func(T_ij));

      //temperature flux divergence div(F_T) = div(kappa grad T)
      double divFlux_z = (k_iph*(T_ip1 - T_ij) - k_imh*(T_ij - T_im1)) / (dz*dz);
      double divFlux_r = (k_jph*(T_jp1 - T_ij) - k_jmh*(T_ij - T_jm1)) / (dr*dr);

      Q_new[3] += dt * (divFlux_z + divFlux_r);  //only E updated

      grid_new.setCell(i,j, VecToCell(Q_new)); //convert Q_new to Cell and then append it to the new grid
    }
  }
  return grid_new; //return new grid after dt timestep
}


// Grid heat_diffusion_step(const Grid& grid, double dx, double dy, double dt,
//                          double gamma, double mu, double kappa_func(double T)) {
//   //dE/dt + div F = div(kappa grad T) + Source_particle + other sources (maybe in the future)
//   //heat diffusion will be added as another source term via operator splitting
//   size_t nx = grid.rows(), ny = grid.cols();
//   Grid grid_new = grid;  //only rho*e changes
//
//   auto getT = [&](size_t i, size_t j) -> double {
//     auto Q = CellToVec(grid(i,j));
//     double rho = Q[0], u = Q[1]/rho, v = Q[2]/rho;
//     double p = (gamma-1.0)*(Q[3] - 0.5*rho*(u*u+v*v));
//     return p*m_p*mu/(rho*k_B); //since p = rho/(mu*m_p)*k_B*T
//   };
//
//   for (size_t i = 2; i < nx-2; i++) {
//     for (size_t j = 2; j < ny-2; j++) {
//       double T_ij   = getT(i,   j);
//       double T_ip1  = getT(i+1, j);
//       double T_im1  = getT(i-1, j);
//       double T_jp1  = getT(i, j+1);
//       double T_jm1  = getT(i, j-1);
//
//       //mean kappa between neighbouring cells
//       double k_iph = 0.5*(kappa_func(T_ij) + kappa_func(T_ip1));
//       double k_imh = 0.5*(kappa_func(T_ij) + kappa_func(T_im1));
//       double k_jph = 0.5*(kappa_func(T_ij) + kappa_func(T_jp1));
//       double k_jmh = 0.5*(kappa_func(T_ij) + kappa_func(T_jm1));
//
//       // double k_iph = 2.0*kappa_func(T_ij)*kappa_func(T_ip1)/(kappa_func(T_ij) + kappa_func(T_ip1));
//       // double k_imh = 2.0*kappa_func(T_im1)*kappa_func(T_ij)/(kappa_func(T_im1) + kappa_func(T_ij));
//       // double k_jph = 2.0*kappa_func(T_ij)*kappa_func(T_jp1)/(kappa_func(T_ij) + kappa_func(T_jp1));
//       // double k_jmh = 2.0*kappa_func(T_jm1)*kappa_func(T_ij)/(kappa_func(T_jm1) + kappa_func(T_ij));
//
//       //temperature flux divergence div(F_T) = div(kappa grad T)
//       double divFlux_x = (k_iph*(T_ip1 - T_ij) - k_imh*(T_ij - T_im1)) / (dx*dx);
//       double divFlux_y = (k_jph*(T_jp1 - T_ij) - k_jmh*(T_ij - T_jm1)) / (dy*dy);
//
//       Cell c = grid(i,j);
//       c[3] += dt * (divFlux_x + divFlux_y);  //only E updated
//       grid_new(i,j) = c;
//     }
//   }
//   return grid_new;
// }


//IMPLICIT IMPLEMENTATION (Q & U are the same, the Batten et al. 1997 article uses U instead of Q)
//residual vaiting for minimalization
template <typename Func>
Grid compute_residual(const Grid& grid,
                      double zmin,
                      double dz, double dr,
                      double gamma, double Omega,
                      double mu, Func kappa_func) {
  size_t nz = grid.rows();
  size_t nr = grid.cols();
  Grid R(nz, nr);   //residual grid, basically what we had in timestep without dt

  
  auto getT = [&](size_t i, size_t j) -> double {
    auto Q = CellToVec(grid.getCell(i,j));
    double rho = Q[0], u = Q[1]/rho, v = Q[2]/rho;
    double p = (gamma-1.0)*(Q[3] - 0.5*rho*(u*u+v*v));
    return p*m_p*mu/(rho*k_B);
  };

  for (size_t i = 2; i < nz-2; i++) {
    double z_i   = zmin + (i-2)*dz + 0.5*dz;
    double z_im1 = z_i - dz,  z_ip1 = z_i + dz;
    double z_im2 = z_im1 - dz, z_ip2 = z_ip1 + dz;

    for (size_t j = 2; j < nr-2; j++) {
      Vector Qi   = CellToVec(grid.getCell(i,  j));
      Vector Qim1 = CellToVec(grid.getCell(i-1,j));
      Vector Qim2 = CellToVec(grid.getCell(i-2,j));
      Vector Qip1 = CellToVec(grid.getCell(i+1,j));
      Vector Qip2 = CellToVec(grid.getCell(i+2,j));
      Vector Qjm1 = CellToVec(grid.getCell(i,j-1));
      Vector Qjm2 = CellToVec(grid.getCell(i,j-2));
      Vector Qjp1 = CellToVec(grid.getCell(i,j+1));
      Vector Qjp2 = CellToVec(grid.getCell(i,j+2));

      //slopes stay the same
      Vector sigma_im1_z, sigma_i_z, sigma_ip1_z;
      if (i == 2) {
        sigma_im1_z = Vector(4,0.0); sigma_i_z = Vector(4,0.0);
        sigma_ip1_z = sigma_minmod(Qi, Qip1, Qip2, dz);
      } else if (i == nz-3) {
        sigma_im1_z = sigma_minmod(Qim2,Qim1,Qi,dz);
        sigma_i_z   = Vector(4,0.0); sigma_ip1_z = Vector(4,0.0);
      } else {
        sigma_im1_z = sigma_minmod(Qim2,Qim1,Qi,  dz);
        sigma_i_z   = sigma_minmod(Qim1,Qi,  Qip1,dz);
        sigma_ip1_z = sigma_minmod(Qi,  Qip1,Qip2,dz);
      }
      Vector sigma_jm1_r, sigma_j_r, sigma_jp1_r;
      if (j == 2) {
        sigma_jm1_r = Vector(4,0.0); sigma_j_r = Vector(4,0.0);
        sigma_jp1_r = sigma_minmod(Qi, Qjp1, Qjp2, dr);
      } else if (j == nr-3) {
        sigma_jm1_r = sigma_minmod(Qjm2,Qjm1,Qi,dr);
        sigma_j_r   = Vector(4,0.0); sigma_jp1_r = Vector(4,0.0);
      } else {
        sigma_jm1_r = sigma_minmod(Qjm2,Qjm1,Qi,  dr);
        sigma_j_r   = sigma_minmod(Qjm1,Qi,  Qjp1,dr);
        sigma_jp1_r = sigma_minmod(Qi,  Qjp1,Qjp2,dr);
      }

      //state vectors at interfaces, same as before
      Vector QL_imh = Qim1 + 0.5*dz*sigma_im1_z;
      Vector QR_imh = Qi   - 0.5*dz*sigma_i_z;
      Vector QL_iph = Qi   + 0.5*dz*sigma_i_z;
      Vector QR_iph = Qip1 - 0.5*dz*sigma_ip1_z;

      //well balanced - Käppeli et al. 2016
      Primitive_Vals prim_im2, prim_im1, prim_i, prim_ip1, prim_ip2;
      try {
          prim_im2 = QtoPrim(Qim2, gamma);
          prim_im1 = QtoPrim(Qim1, gamma);
          prim_i   = QtoPrim(Qi,   gamma);
          prim_ip1 = QtoPrim(Qip1, gamma);
          prim_ip2 = QtoPrim(Qip2, gamma);
      } catch (const std::invalid_argument& e) {
          std::cerr << "[QtoPrim FAIL] cell=(" << i << "," << j << ")"
                    << "  Qim2=[" << Qim2[0] <<","<< Qim2[3] << "]"
                    << "  Qim1=[" << Qim1[0] <<","<< Qim1[3] << "]"
                    << "  Qi  =[" << Qi[0]   <<","<< Qi[3]   << "]"
                    << "  Qip1=[" << Qip1[0] <<","<< Qip1[3] << "]"
                    << "  Qip2=[" << Qip2[0] <<","<< Qip2[3] << "]\n";
          throw;
      }

      //phi(x) = 1/2*Omega^2 x^2
      double p0_at_imh = prim_i.p + 0.5*prim_i.rho*(0.5*Omega*Omega*(z_i*z_i - z_im1*z_im1)); 
      double p0_at_iph = prim_i.p - 0.5*prim_i.rho*(0.5*Omega*Omega*(z_ip1*z_ip1 - z_i*z_i));
      double p0_at_im1h = prim_im1.p - 0.5*prim_im1.rho*(0.5*Omega*Omega*(z_i*z_i - z_im1*z_im1));
      double p0_at_ip1h = prim_ip1.p + 0.5*prim_ip1.rho*(0.5*Omega*Omega*(z_ip1*z_ip1 - z_i*z_i));

      double p0_at_im1 = prim_i.p + 0.5*(prim_im1.rho+prim_i.rho) * 0.5*Omega*Omega*(z_i*z_i - z_im1*z_im1);
      double p0_at_ip1 = prim_i.p - 0.5*(prim_ip1.rho+prim_i.rho) * 0.5*Omega*Omega*(z_ip1*z_ip1 - z_i*z_i);
      double p0_at_im2 = prim_im1.p + 0.5*(prim_im2.rho+prim_im1.rho) * 0.5*Omega*Omega*(z_im1*z_im1 - z_im2*z_im2);
      double p0_at_ip2 = prim_ip1.p - 0.5*(prim_ip2.rho+prim_ip1.rho) * 0.5*Omega*Omega*(z_ip2*z_ip2 - z_ip1*z_ip1);
 
      //p1,i(x_i) = 0 by design
      double p1_ip1 = prim_ip1.p - p0_at_ip1;
      double p1_im1 = prim_im1.p - p0_at_im1;
      double p1_im2 = prim_im2.p - p0_at_im2;
      double p1_ip2 = prim_ip2.p - p0_at_ip2;
      //slope with p1,i(x_i)=0
      double sigma_pi = minmod_one(p1_im1, 0.0, p1_ip1, dz);
      double sigma_pim1 = minmod_one(p1_im2, p1_im1, 0.0, dz);
      double sigma_pip1 = minmod_one(0.0, p1_ip1, p1_ip2, dz);


      QL_imh[3] = (p0_at_im1h + 0.5*sigma_pim1*dz)/(gamma-1.0)
                + 0.5*(QL_imh[1]*QL_imh[1]+QL_imh[2]*QL_imh[2])/QL_imh[0];
      QR_imh[3] = (p0_at_imh - 0.5*sigma_pi*dz)/(gamma-1.0)
                + 0.5*(QR_imh[1]*QR_imh[1]+QR_imh[2]*QR_imh[2])/QR_imh[0];
      QR_iph[3] = (p0_at_ip1h - 0.5*sigma_pip1*dz)/(gamma-1.0)
                + 0.5*(QR_iph[1]*QR_iph[1] + QR_iph[2]*QR_iph[2])/QR_iph[0];
      QL_iph[3] = (p0_at_iph + 0.5*sigma_pi*dz)/(gamma-1.0)
                + 0.5*(QL_iph[1]*QL_iph[1] + QL_iph[2]*QL_iph[2])/QL_iph[0];

      //q2 direction
      Vector QL_jmh = Qjm1 + 0.5*dr*sigma_jm1_r;
      Vector QR_jmh = Qi   - 0.5*dr*sigma_j_r;
      Vector QL_jph = Qi   + 0.5*dr*sigma_j_r;
      Vector QR_jph = Qjp1 - 0.5*dr*sigma_jp1_r;

      //HLLC fluxes, same as in timestep func
      Vector FZimh = Fluxhllc_q1(QL_imh, QR_imh, gamma);
      Vector FZiph = Fluxhllc_q1(QL_iph, QR_iph, gamma);
      Vector FRjmh = Fluxhllc_q2(QL_jmh, QR_jmh, gamma);
      Vector FRjph = Fluxhllc_q2(QL_jph, QR_jph, gamma);

      //flux divergence + gravity + diffusion (no dt)
      Vector Rij = -(1.0/dz)*(FZiph - FZimh) - (1.0/dr)*(FRjph - FRjmh)
                 + grav_source(Qi, z_im1, z_ip1, Omega, dz);

      //heat diffusion
      double T_ij  = getT(i,j), T_ip1 = getT(i+1,j), T_im1 = getT(i-1,j);
      double T_jp1 = getT(i,j+1), T_jm1 = getT(i,j-1);
      double k_iph = 0.5*(kappa_func(T_ij)+kappa_func(T_ip1));
      double k_imh = 0.5*(kappa_func(T_ij)+kappa_func(T_im1));
      double k_jph = 0.5*(kappa_func(T_ij)+kappa_func(T_jp1));
      double k_jmh = 0.5*(kappa_func(T_ij)+kappa_func(T_jm1));
      Rij[3] += (k_iph*(T_ip1-T_ij) - k_imh*(T_ij-T_im1))/(dz*dz)
              + (k_jph*(T_jp1-T_ij) - k_jmh*(T_ij-T_jm1))/(dr*dr);

      R.setCell(i, j, VecToCell(Rij));
    }
  }
  return R;
}


//1 backward-Euler implicit step (Batten et al. 1997)
//approximate factorisation: z-sweep then r-sweep, each a block-tridiagonal solve.
//S_grid (particle back-reaction) is operator split, added explicitly at the end for now.
template <typename Func>
Grid implicit_step_B1(const Grid& grid, const Grid& S_grid,
                      double zmin, double dz, double dr, double dt,
                      double gamma, double Omega,
                      double mu, Func kappa_func) {
  size_t nz = grid.rows(), nr = grid.cols();
  size_t Nz = nz - 4, Nr = nr - 4;

  Grid R  = compute_residual(grid, zmin, dz, dr, gamma, Omega, mu, kappa_func);
  Grid dW(nz, nr);   //result of z-sweep



  std::vector<Matrix> LU_z(Nz, Matrix(4,4,0.0));
  std::vector<std::vector<int>> piv_z(Nz, std::vector<int>(4,0));
  //z-sweep, for each fixed j, solve block-tridiagonal in i
  for (size_t j = 2; j < nr-2; j++) {
    std::vector<Matrix> Az(Nz, Matrix(4,4,0.0));
    std::vector<Matrix> Dz(Nz, Matrix(4,4,0.0));
    std::vector<Matrix> Cz(Nz, Matrix(4,4,0.0));
    std::vector<Vector> rz(Nz, Vector(4,0.0));

    for (size_t ii = 0; ii < Nz; ii++) {
      size_t i = ii + 2;

      //gravity source
      double z_i = zmin + ii*dz + 0.5*dz;
      double G = Omega*Omega*z_i;
      Matrix dSdU(4,4,0.0);
      dSdU(1,0) = -G;
      dSdU(3,1) = -G;

      rz[ii] = CellToVec(R.getCell(i, j));
      for (int k = 0; k < 4; k++) Dz[ii](k,k) = 1.0/dt; //I/dt

      //slope-limited interface states with same logic as compute_residual
      Vector Qi   = CellToVec(grid.getCell(i,  j));
      Vector Qim1 = CellToVec(grid.getCell(i-1,j));
      Vector Qim2 = CellToVec(grid.getCell(i-2,j));
      Vector Qip1 = CellToVec(grid.getCell(i+1,j));
      Vector Qip2 = CellToVec(grid.getCell(i+2,j));

      Vector sig_im1, sig_i, sig_ip1;
      if (ii == 0) {
        sig_im1 = Vector(4,0.0); sig_i = Vector(4,0.0);
        sig_ip1 = sigma_minmod(Qi, Qip1, Qip2, dz);
      } else if (ii == Nz-1) {
        sig_im1 = sigma_minmod(Qim2,Qim1,Qi,dz);
        sig_i   = Vector(4,0.0); sig_ip1 = Vector(4,0.0);
      } else {
        sig_im1 = sigma_minmod(Qim2,Qim1,Qi,  dz);
        sig_i   = sigma_minmod(Qim1,Qi,  Qip1,dz);
        sig_ip1 = sigma_minmod(Qi,  Qip1,Qip2,dz);
      }

      Vector QL_imh = Qim1 + 0.5*dz*sig_im1;
      Vector QR_imh = Qi   - 0.5*dz*sig_i;
      Vector QL_iph = Qi   + 0.5*dz*sig_i;
      Vector QR_iph = Qip1 - 0.5*dz*sig_ip1;

      //well-balanced energy correction omitted from Jacobi states.
      //The correction is already in R (the explicit RHS); 
      //Jacobi uses the unmodified slope-limited states for stability.

      HLLCJac Jimh = hllc_jacobian_q1(QL_imh, QR_imh, gamma);
      HLLCJac Jiph = hllc_jacobian_q1(QL_iph, QR_iph, gamma);

      //lower block: A[ii] = -(1/dz)*J_{i-1/2} dF_dUl
      //zero for ii=0: ghost cell delta U treated as 0
      if (ii > 0)
        Az[ii] = Jimh.dF_dUl * (-1.0/dz);

      //diagonal: I/dt + (1/dz)*J_{i+1/2}dF_dUl - (1/dz)*J_{i-1/2}dF_dUr
      Dz[ii] += Jiph.dF_dUl * (1.0/dz);
      Dz[ii] -= Jimh.dF_dUr * (1.0/dz);

      //gravity source subtracted from Dz
      Dz[ii] = Dz[ii] - dSdU;

      //upper block: C[ii] = (1/dz)*J_{i+1/2}dF_dUr
      if (ii < Nz-1)
        Cz[ii] = Jiph.dF_dUr * (1.0/dz);
    }

    block_thomas(Az, Dz, Cz, rz, Nz, LU_z, piv_z);
    for (size_t ii = 0; ii < Nz; ii++)
      dW.setCell(ii+2, j, VecToCell(rz[ii]));
  }

  std::vector<Matrix> LU_r(Nr, Matrix(4,4,0.0));
  std::vector<std::vector<int>> piv_r(Nr, std::vector<int>(4,0));
  //r-sweep, for each fixed i, solve block-tridiagonal in j using delta W as RHS
  Grid dU(nz, nr);

  for (size_t i = 2; i < nz-2; i++) {
    std::vector<Matrix> Ar(Nr, Matrix(4,4,0.0));
    std::vector<Matrix> Dr(Nr, Matrix(4,4,0.0));
    std::vector<Matrix> Cr(Nr, Matrix(4,4,0.0));
    std::vector<Vector> rr(Nr, Vector(4,0.0));

    for (size_t jj = 0; jj < Nr; jj++) {
      size_t j = jj + 2;

      rr[jj] = 1/dt * CellToVec(dW.getCell(i, j));
      for (int k = 0; k < 4; k++) Dr[jj](k,k) = 1.0/dt;

      Vector Qi   = CellToVec(grid.getCell(i,j  ));
      Vector Qjm1 = CellToVec(grid.getCell(i,j-1));
      Vector Qjm2 = CellToVec(grid.getCell(i,j-2));
      Vector Qjp1 = CellToVec(grid.getCell(i,j+1));
      Vector Qjp2 = CellToVec(grid.getCell(i,j+2));

      Vector sig_jm1, sig_j, sig_jp1;
      if (jj == 0) {
        sig_jm1 = Vector(4,0.0); sig_j = Vector(4,0.0);
        sig_jp1 = sigma_minmod(Qi, Qjp1, Qjp2, dr);
      } else if (jj == Nr-1) {
        sig_jm1 = sigma_minmod(Qjm2,Qjm1,Qi,dr);
        sig_j   = Vector(4,0.0); sig_jp1 = Vector(4,0.0);
      } else {
        sig_jm1 = sigma_minmod(Qjm2,Qjm1,Qi,  dr);
        sig_j   = sigma_minmod(Qjm1,Qi,  Qjp1,dr);
        sig_jp1 = sigma_minmod(Qi,  Qjp1,Qjp2,dr);
      }

      Vector QL_jmh = Qjm1 + 0.5*dr*sig_jm1;
      Vector QR_jmh = Qi   - 0.5*dr*sig_j;
      Vector QL_jph = Qi   + 0.5*dr*sig_j;
      Vector QR_jph = Qjp1 - 0.5*dr*sig_jp1;

      HLLCJac Jjmh = hllc_jacobian_q2(QL_jmh, QR_jmh, gamma);
      HLLCJac Jjph = hllc_jacobian_q2(QL_jph, QR_jph, gamma);

      if (jj > 0)
        Ar[jj] = Jjmh.dF_dUl * (-1.0/dr);

      Dr[jj] += Jjph.dF_dUl * (1.0/dr);
      Dr[jj] -= Jjmh.dF_dUr * (1.0/dr);

      if (jj < Nr-1)
        Cr[jj] = Jjph.dF_dUr * (1.0/dr);
    }

    block_thomas(Ar, Dr, Cr, rr, Nr, LU_r, piv_r);
    for (size_t jj = 0; jj < Nr; jj++)
      dU.setCell(i, jj+2, VecToCell(rr[jj]));
  }

  //applying delta U and particle source (operator split, already contains dt)
  Grid grid_new = grid;
  for (size_t i = 2; i < nz-2; i++)
    for (size_t j = 2; j < nr-2; j++) {
      Vector Q_new = CellToVec(grid.getCell(i,j))
                   + CellToVec(dU.getCell(i,j))
                   + CellToVec(S_grid.getCell(i,j));
      grid_new.setCell(i, j, VecToCell(Q_new));
    }
  return grid_new;
}


//B2 scheme eqs. (70)-(72) Batten et al. 1997
//2 backward-Euler steps: dt/2 then dt, twice the convergence rate of B1.
template <typename Func>
Grid implicit_step_B2(const Grid& grid, const Grid& S_grid,
                      const GridBC& bc, double cs2,
                      double zmin, double dz, double dr, double dt,
                      double gamma, double Omega,
                      double mu, Func kappa_func) {
  Grid S_empty(grid.rows(), grid.cols());   //S_grid applied once at the end only
  
  Grid U_bar = implicit_step_B1(grid,  S_empty, zmin, dz, dr, dt/2.0,
                                gamma, Omega, mu, kappa_func);
  
  applyBC(U_bar, bc, Omega, cs2, gamma, dz, zmin);
  
    Grid dU2   = implicit_step_B1(U_bar, S_empty, zmin, dz, dr, dt,
                                gamma, Omega, mu, kappa_func);

  //U^{n+1} = U_bar + (dU2 - U_bar)/2 + S_grid
  //so U_bar + dU_tilde/2 where dU_tilde = dU2 - U_bar
  size_t nz = grid.rows(), nr = grid.cols();
  Grid result = grid;

  for (size_t i = 2; i < nz-2; i++){
    for (size_t j = 2; j < nr-2; j++) {
      Vector Q_bar = CellToVec(U_bar.getCell(i,j));
      Vector dU2_val = CellToVec(dU2.getCell(i,j));
      Vector S_val = CellToVec(S_grid.getCell(i,j));
      
      Vector Q = Q_bar + 0.5*(dU2_val - Q_bar) + S_val;
      result.setCell(i, j, VecToCell(Q));
    }
  }
  return result;
}


//ideal gas heat conduction coeff, f=3
double kappa(double mu, double d, double T){
  return 3/(3*d*d) * std::sqrt(k_B*k_B*k_B*T/(PI*PI*PI*mu*m_p)); //kappa = f/(3d^2) * sqrt(k_B^3 T/(pi^3 mu*m_p))
}


int main(int argc, char*argv[]){
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  auto t1 = high_resolution_clock::now();

  std::string paramfile = (argc > 1) ? argv[1] : "params.txt";
  SimParams par = readParams(paramfile);
  double cs2 = par.gas_p0/par.gas_rho0;

  double Nz = par.Nz, Nr = par.Nr;
  double Nt = par.Nt;
  double t0 = par.t0, tf = par.tf;
  double zmin = par.zmin, zmax = par.zmax;
  double rmin = par.rmin, rmax = par.rmax;
  double gamma = par.gamma;
  double Omega = par.Omega;
  double mu    = par.gas_mu;

  double dt = (tf - t0) / Nt;
  double dz = (zmax - zmin) / Nz;
  double dr = (rmax - rmin) / Nr;

  GridBC bc;
  bc.left.type   = parseBC(par.bc_left);
  bc.right.type  = parseBC(par.bc_right);
  bc.top.type    = parseBC(par.bc_top);
  bc.bottom.type = parseBC(par.bc_bottom);
  validateBC(bc);

  size_t nz = (size_t)Nz + 4;
  size_t nr = (size_t)Nr + 4;
  Grid grid(nz, nr);
  init_cond(grid, par);
  applyBC(grid, bc, Omega, cs2, gamma, dz, zmin);

  //d: effective cross section
  double d = 1.0;
  auto kappa_func = [mu, d](double T) -> double {
    return kappa(mu, d, T);
  };

  
  Particle p = initParticle(par);

  //cfl condition check: either new dt or continue with CFL
  // double cfl = cfl_dt(grid, dz, dr, gamma);
  // double diffuse = diffuse_dt(grid, dz, dr, gamma, mu, kappa_null);
  double dt_leapfrog = 1.0/(2.0*Omega); //stability condition of gravity source
  // if (dt > cfl && cfl < dt_leapfrog && cfl < diffuse){
  //   std::cout << "Warning: dt=" << dt << " exceeds CFL limit=" << cfl
  //             << ", using CFL dt"<<std::endl;
  //   dt = cfl;
  //   Nt = (int)std::ceil((tf - t0) / dt);
  //   std::cout << "Nt updated to " << Nt <<std::endl;
  // } else if (dt > cfl && cfl > dt_leapfrog){
  //   std::cout << "Warning: dt=" << dt << " exceeds Leapfrog limit=" << dt_leapfrog
  //             << ", using Leapfrog dt"<<std::endl;
  //   dt = dt_leapfrog;
  //   Nt = (int)std::ceil((tf - t0) / dt);
  //   std::cout << "Nt updated to " << Nt <<std::endl;
  // } else if (dt > cfl && cfl > diffuse){
  //   std::cout << "Warning: dt=" << dt << " exceeds diffusive CFL limit=" << diffuse
  //             << ", using diffusive CFL dt"<<std::endl;
  //   dt = diffuse;
  //   Nt = (int)std::ceil((tf - t0) / dt);
  //   std::cout << "Nt updated to " << Nt <<std::endl;
  // }

  if (dt > dt_leapfrog) {
    dt = dt_leapfrog;
    Nt = (int)std::ceil((tf - t0) / dt);
  }

  setupOutputDir(par.outdir);
  writeMetadata(par.outdir, par);
  auto gridfile     = openGridFile(par.outdir, (size_t)Nz, (size_t)Nr);
  auto particlefile = openParticleFile(par.outdir);
  double t = t0;

  double write_fact = 1;
  int next_save = 1;

  int print_interval = (int)Nt/10;
  if (print_interval==0) print_interval = 1;
  int n=0;
  try{
    for (; n<(int)Nt; n++){
      t = t0 + n*dt;

      // if (n % 5000 == 0) {
      //   double progress = (double)n / Nt * 100.0;
      //   std::cout << "Step: " << n << " / " << (int)Nt 
      //             << " [" << std::fixed << std::setprecision(2) << progress << "%] "
      //             << "t = " << t << std::endl; 
      // }
      if (n%print_interval == 0 || n == (int)Nt-1){
        double progress = (double)n/((int)Nt-1)*100.0;
        std::cout << "Step: " << n << " / " << (int)Nt
                  << " [" << std::fixed << std::setprecision(2) << progress << "%] "
                  << "t = " << t << std::endl;
      }

      if(n%10==0){
        writeParticle(particlefile, p, t, n);
        writeFrame(gridfile, grid,  t, (size_t)Nz, (size_t)Nr);
      }

      // if (n>=next_save){
      //   writeParticle(particlefile, p, t, n);
      //   particlefile.flush(); //DEBUG
      //   writeFrame(gridfile, grid, t, (size_t)Nz, (size_t)Nr);
      //   gridfile.flush();
      //   int current_next = next_save;
      //   next_save = static_cast<int>(current_next*write_fact);
      //   if(next_save<=current_next){
      //     next_save = current_next+1;
      //   }
      //   if (next_save <=n){
      //     next_save = n+1;
      //   }
      //   // std::cout << "Saved step " << n << ". Next target step: " << next_save << std::endl;
      // }


      applyBC(grid, bc, Omega, cs2, gamma, dz, zmin);
      // if (n == 0) {
      // std::cout << "ghost i="<< nz-2 <<": rho=" << grid(nz-2,2, 0) 
      //           << " rhou=" << grid(nz-2,2, 1) <<std::endl;
      // }

      if (p.active){
        CICWeights W = calc_CIC(p.z, p.r, zmin, rmin, dz, dr, (int)Nz, (int)Nr);
        GasAtParticle gas = interpolate_gas(grid, W, gamma);
        
        if(leapfrog_first_half(p, gas, Omega, zmin, rmin, zmax, rmax, dz, dr, (int)Nz, (int)Nr, dt)){
          CICWeights W_new = calc_CIC(p.z, p.r, zmin, rmin, dz, dr, (int)Nz, (int)Nr);
          GasAtParticle gas_new = interpolate_gas(grid, W_new, gamma);

          Grid S_grid(nz, nr);
          source_projection(S_grid, p, gas_new, W_new, dz, dr, dt, gamma);
          grid = implicit_step_B2(grid, S_grid, bc, cs2, zmin, dz, dr, dt, gamma, Omega, mu, kappa_func);
          applyBC(grid, bc, Omega, cs2, gamma, dz, zmin);

          CICWeights W_np1 = calc_CIC(p.z, p.r, zmin, rmin, dz, dr, (int)Nz, (int)Nr);
          GasAtParticle gas_np1 = interpolate_gas(grid, W_np1, gamma);
          leapfrog_second_half(p, gas_np1, Omega, dt);
        } else {
          Grid S_grid(nz, nr);
          grid = implicit_step_B2(grid, S_grid, bc, cs2, zmin, dz, dr, dt, gamma, Omega, mu, kappa_func);
          applyBC(grid, bc, Omega, cs2, gamma, dz, zmin);
        }
      } else {
        Grid S_grid(nz, nr);  //empty source, just advance gas
        grid = implicit_step_B2(grid, S_grid, bc, cs2, zmin, dz, dr, dt, gamma, Omega, mu, kappa_func);
        applyBC(grid, bc, Omega, cs2, gamma, dz, zmin);
      }
      if (n == 0) {
        std::cout << "\nAFTER FIRST TIMESTEP\n";
        std::cout << std::scientific << std::setprecision(4);
        for (size_t i = 2; i < std::min((size_t)10, nz); i++) {
          double rho = grid(i,2, 0);
          double u = grid(i,2, 1) / rho;
          double p = (gamma-1.0)*(grid(i,2,3) - 0.5*rho*u*u);
          std::cout << "i=" << i << " : rho=" << rho << " u=" << u << " p=" << p << std::endl;
        }
        std::cout << "i=" << nz-4 << " : rho=" << grid(nz-4,2,0) << " u=" << grid(nz-4,2,1)/grid(nz-4,2,0)<<
          " p="<<(gamma-1.0)*(grid(nz-4, 2, 3)-0.5*grid(nz-4,2,1)*grid(nz-4,2,1)/grid(nz-4,2,0))<<std::endl;
        std::cout << "i=" << nz-3 << " : rho=" << grid(nz-3,2,0) << " u=" << grid(nz-3,2,1)/grid(nz-3,2,0)<<
          " p="<<(gamma-1.0)*(grid(nz-3, 2, 3)-0.5*grid(nz-3,2,1)*grid(nz-4,2,1)/grid(nz-3,2,0))<<std::endl;
        std::cout << "i=" << nz-2 << " : rho=" << grid(nz-2,2,0) << " u=" << grid(nz-2,2,1)/grid(nz-2,2,0)<<
          " p="<<(gamma-1.0)*(grid(nz-2, 2, 3)-0.5*grid(nz-2,2,1)*grid(nz-2,2,1)/grid(nz-2,2,0))<<std::endl;
        std::cout << ".........................\n"<<std::endl;
      }
    }
  } catch (const std::invalid_argument& e){
    std::cerr << "[CRASH] at step=" <<n<<" t="<<t<<std::endl;
    throw;
  }
  
  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << ms_int.count() << "ms\n";
  std::cout << ms_double.count() << "ms\n";
  return 0;
}
