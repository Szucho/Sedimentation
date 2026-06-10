#include "../headers/particle.h"
#include <cmath>

constexpr double PI = 3.141592653589793;
constexpr double c_heat  = 8e6;       // erg/g/K
constexpr double h_conv  = 1e3;       // erg/s/cm^2/K
constexpr double eps_rad = 0.8;
constexpr double sigma_SB = 5.67e-5;  // erg/s/cm^2/K^4
constexpr double rho_m   = 3.0;       // g/cm^2
constexpr double T_melt  = 1800.0;    // K
constexpr double k_B = 1.380649e-16;
constexpr double m_p = 1.672622e-24;


//CIC weighing
CICWeights calc_CIC(double pz, double pr, double zmin, double rmin, double dz, double dr, int nz, int nr){
  CICWeights W;
  
  //fractional cell indices
  double z_g = (pz-zmin)/dz - 0.5; //-1/2 moves particle to cell centre instead of cell corner (my FSM mistake :/)
  double r_g = (pr-rmin)/dr - 0.5;

  int i = (int)std::floor(z_g);
  int j = (int)std::floor(r_g);

  double p = z_g-i;  //fractional offset in x = [0,1]
  double q = r_g-j;  //        - || -       y = [0,1]
  
  //offset by 2 because of ghost points in grid
  W.i0 = i+2;
  W.j0 = j+2;
  W.i1 = i+3;
  W.j1 = j+3;

  //clamp to physical domain (I use 2 ghost cells on each boundary here I ignore them)
  W.i0 = std::max(2, std::min(W.i0, nz+1));
  W.j0 = std::max(2, std::min(W.j0, nr+1));
  W.i1 = std::max(2, std::min(W.i1, nz+1));
  W.j1 = std::max(2, std::min(W.j1, nr+1));

  //weights
  W.W00 = (1.0-p)*(1.0-q);
  W.W01 = (1.0-p)*q;
  W.W10 = p*(1.0-q);
  W.W11 = p*q;

  return W;
}


//from HLLC grid to particle interpolation
GasAtParticle interpolate_gas(const Grid& grid, const CICWeights& W, double gamma){
  double rho  = W.W00*grid(W.i0, W.j0, 0)
              + W.W01*grid(W.i0, W.j1, 0)
              + W.W10*grid(W.i1, W.j0, 0)
              + W.W11*grid(W.i1, W.j1, 0);

  double rhou = W.W00*grid(W.i0, W.j0, 1)
              + W.W01*grid(W.i0, W.j1, 1)
              + W.W10*grid(W.i1, W.j0, 1)
              + W.W11*grid(W.i1, W.j1, 1);

  double rhov = W.W00*grid(W.i0, W.j0, 2)
              + W.W01*grid(W.i0, W.j1, 2)
              + W.W10*grid(W.i1, W.j0, 2)
              + W.W11*grid(W.i1, W.j1, 2);

  double rhoe = W.W00*grid(W.i0, W.j0, 3)
              + W.W01*grid(W.i0, W.j1, 3)
              + W.W10*grid(W.i1, W.j0, 3)
              + W.W11*grid(W.i1, W.j1, 3);

  GasAtParticle g;

  g.rho = rho;
  g.u   = rhou/rho;
  g.v   = rhov/rho;
  g.p   = (gamma-1.0)*(rhoe-0.5*rho*(g.u*g.u + g.v*g.v));
  g.cs  = std::sqrt(gamma*g.p/g.rho);

  return g;
}


//particle to grid interpolation
void source_projection(Grid& grid_new, 
                         const Particle& p, 
                         const GasAtParticle gas, 
                         const CICWeights W,
                         double dz, double dr, 
                         double dt, double gamma){

  auto [az, ar] = drag_accel(p, gas); //drag acceleration

  //force on particle from gas drag
  double Fx = p.mass*az;
  double Fy = p.mass*ar;
  //N3 dictates that the gas loses this momentum
  double cell_area = dz*dr;
  
  //source contributions to primitive vals of the gas
  double dSz = -Fx*dt/cell_area;
  double dSr = -Fy*dt/cell_area;
  double dSE = -(Fx*gas.u + Fy*gas.v)*dt/cell_area; //work against gas velocity

  //distribute back-reaction to 4 surrounding cells using CIC weights
  auto distribute = [&](int i, int j, double w){
    grid_new(i,j, 1) += dSz*w; //0th index is for rho but the chondrite will
    grid_new(i,j, 2) += dSr*w; //stay solid so no mixing -> no mass source
    grid_new(i,j, 3) += dSE*w;
  };

  distribute(W.i0, W.j0, W.W00);
  distribute(W.i0, W.j1, W.W01);
  distribute(W.i1, W.j0, W.W10);
  distribute(W.i1, W.j1, W.W11);
}


// //drag acceleration
// std::array<double,2> drag_accel(const Particle& p, const GasAtParticle& gas) {
//   //Stokes: F = 6*pi*eta*r*(v_p - v_gas)
//   //eta ~ rho*cs*lambda (~rho*cs*1 for now; TO DO: implement lambda mean free path)
//   double eta = gas.rho * gas.cs * p.radius;  //effective viscosity scale
//   double drag_coeff = 6.0 * PI * eta;
//   
//   double ax = -drag_coeff / p.mass * (p.vx - gas.u);
//   double ay = -drag_coeff / p.mass * (p.vy - gas.v);
//   return {ax, ay};
// }

//drag acceleration Epstein
std::array<double,2> drag_accel(const Particle& p, const GasAtParticle& gas) {
  double vth = std::sqrt(8.0*gas.p/(PI*gas.rho));
  double coeff = 4.0*PI/3.0 * gas.rho * p.radius*p.radius * vth;
  double ax = -coeff / p.mass * (p.vz-gas.u);
  double ay = -coeff / p.mass * (p.vr-gas.v);
  return {ax, ay};
}

//tidal acceleration
std::array<double,2> grav_accel(const Particle& p, double Omega) {
    return { -Omega*Omega * p.z, 0.0};
}


// bool leapfrog_first_half(Particle& p,
//                          const GasAtParticle& gas,
//                          double Omega,
//                          double xmin, double ymin,
//                          double xmax, double ymax,
//                          double dx, double dy,
//                          int nx, int ny,
//                          double dt) {
//
//   auto [drag_x, drag_y] = drag_accel(p, gas);
//   auto [g_x, g_y]       = grav_accel(p, Omega);
//   double ax = drag_x + g_x;
//   double ay = drag_y + g_y;
//
//   //half kick
//   p.vx += 0.5 * ax * dt;
//   p.vy += 0.5 * ay * dt;
//   //drift
//   p.x += p.vx * dt;
//   p.y += p.vy * dt;
//   //boundary check
//   if (p.x < xmin || p.x > xmax || p.y < ymin || p.y > ymax) {
//       p.active = false;
//       return false;
//   }
//   return true;
// }
// void leapfrog_second_half(Particle& p,
//                           const GasAtParticle& gas_new,  //gas at t_{n+1}
//                           double Omega,
//                           double dt) {
//
//   auto [drag_x2, drag_y2] = drag_accel(p, gas_new);
//   auto [g_x2, g_y2]       = grav_accel(p, Omega);
//   double ax2 = drag_x2 + g_x2;
//   double ay2 = drag_y2 + g_y2;
//
//   //second half kick
//   p.vx += 0.5 * ax2 * dt;
//   p.vy += 0.5 * ay2 * dt;
// }


bool leapfrog_first_half(Particle& p,
                         const GasAtParticle& gas,
                         double Omega,
                         double zmin, double rmin,
                         double zmax, double rmax,
                         double dz, double dr,
                         int nz, int nr,
                         double dt, double mu) {

  auto [g_z, g_r] = grav_accel(p, Omega);
  
  // t_stop kiszámítása
  double vth   = std::sqrt(8.0 * gas.p / (PI * gas.rho));
  double coeff = 4.0*PI/3.0 * gas.rho * p.radius*p.radius * vth;
  double t_stop = p.mass / coeff;

  // implicit drag half-kick: v* = (v + (g + u_gas/t_stop)*dt/2) / (1 + dt/(2*t_stop))
  double factor = 1.0 / (1.0 + 0.5 * dt / t_stop);
  double A = 4.0 * PI * p.radius * p.radius; //surface area
  double T_gas = gas.p * m_p * mu / (gas.rho * k_B);

  p.vz = (p.vz + (g_z + gas.u / t_stop) * 0.5 * dt) * factor;
  p.vr = (p.vr + (g_r + gas.v / t_stop) * 0.5 * dt) * factor;

  // drift
  double dTdt = (vth / (c_heat * rho_m * p.radius)) * gas.rho * p.vz * p.vz;
            // - (h_conv * A / (p.mass * c_heat)) * (p.T - T_gas)
            // - (eps_rad * sigma_SB * A / (p.mass * c_heat)) * (std::pow(p.T, 4) - std::pow(T_gas, 4));

  p.T += 0.5 * dTdt * dt;
  p.z += p.vz * dt;
  p.r += p.vr * dt;

  if (p.z < zmin || p.z > zmax || p.r < rmin || p.r > rmax) {
      p.active = false;
      return false;
  }
  return true;
}

void leapfrog_second_half(Particle& p,
                          const GasAtParticle& gas_new,  //gas at t_{n+1}
                          double Omega,
                          double dt, double mu) {

  auto [g_x2, g_y2] = grav_accel(p, Omega);

  double vth2   = std::sqrt(8.0 * gas_new.p / (PI * gas_new.rho));
  double coeff2 = 4.0*PI/3.0 * gas_new.rho * p.radius*p.radius * vth2;
  double t_stop2 = p.mass / coeff2;

  double factor2 = 1.0 / (1.0 + 0.5 * dt / t_stop2);
  p.vz = (p.vz + (g_x2 + gas_new.u / t_stop2) * 0.5 * dt) * factor2;
  p.vr = (p.vr + (g_y2 + gas_new.v / t_stop2) * 0.5 * dt) * factor2;

  double A     = 4.0 * PI * p.radius * p.radius;
  double T_gas = gas_new.p * m_p * mu / (gas_new.rho * k_B);
  double dTdt  = (vth2 / (c_heat * rho_m * p.radius)) * gas_new.rho * p.vz * p.vz;
               // - (h_conv * A / (p.mass * c_heat)) * (p.T - T_gas)
               // - (eps_rad * sigma_SB * A / (p.mass * c_heat))
               //   * (std::pow(p.T, 4) - std::pow(T_gas, 4));
  p.T += 0.5 * dTdt * dt;
}
