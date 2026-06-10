#ifndef PARTICLE
#define PARTICLE

#include <array>
#include "grid.h"
#include "matrix.h"


/*
 Particle script
 Bertalan Szuchovszky - 01.05.2026. (I think)

 Lagrange view of a single solid particle. 
 The particle is interpolated to the gas grid via Cloud In Cell (CIC) method,
 back-reaction is calculated the same way.
 The particle equation of motion is integrated with a Leapfrog algorightm.
    -> gravitational potential calulated explicitly
    -> implicit calculation of Epstein drag in gas interacting with particle ~ explicit became unstable :(
*/


struct Particle {
  double z, r;   //positions
  double vz, vr; //velocities
  double mass;   //total mass of the solid chondrite particle
  double radius; //it will be a circle
  double T;
  bool active=true;
  //id
};

struct GasAtParticle {
  double rho, u, v, p, cs; //state of gas will be interpolated to the particle location from the HLLC grid
};


struct CICWeights { //Kernel will be the same for particle -> grid and grid -> particle interpolations
  int i0, j0, i1, j1;        //4 neighbouring cell indicies
  double W00, W01, W10, W11; //corresponding weights
};


CICWeights calc_CIC(double pz, double pr, double zmin, double rmin, double dz, double dr, int nz, int nr);
GasAtParticle interpolate_gas(const Grid& grid, const CICWeights& W, double gamma);
std::array<double,2> drag_accel(const Particle& p, const GasAtParticle& gas);

void source_projection(Grid& grid_new, 
                         const Particle& p, 
                         const GasAtParticle gas, 
                         const CICWeights W,
                         double dz, double dr, 
                         double dt, double gamma);


bool leapfrog_first_half(Particle& p,
                         const GasAtParticle& gas,
                         double Omega,
                         double zmin, double rmin,
                         double zmax, double rmax,
                         double dz, double dr,
                         int nz, int nr,
                         double dt, double mu);

void leapfrog_second_half(Particle& p,
                          const GasAtParticle& gas_new,
                          double Omega,
                          double dt, double mu);



#endif // !PARTICLE
