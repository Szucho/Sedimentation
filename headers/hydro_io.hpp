#pragma once

#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <map>
#include <iomanip>
#include <cstdint>
#include <vector>
#include "hydro_config.hpp"
#include "grid.hpp"
#include "mesh.hpp"
#include "particle.hpp"

/*
 Input / Output header for Kokkos AMR Hydro Simulation
 Bertalan Szuchovszky 20.04.2026
 
 Modified on 21.09.2026:
  - Adapted to parse full params.txt configuration files.
  - Added support for exporting binary grid structures and simulation frames directly from flat Kokkos device views

*/

//structure holding all simulation, grid, physics, and I/O parameters parsed from config files
struct SimParams {
  //grid
  Real Nz, Nr, Nt;
  Real t0, tf;
  Real zmin, zmax, rmin, rmax;
  //refinement
  int nx1_block = 0, nx2_block = 0;
  int nghost = 2;
  std::vector<RefinementRegion> refine_regions;
  //AMR density contrast |ln(rho/rho_init)|: refine above / derefine below (hysteresis)
  Real amr_refine = 0.08, amr_derefine = 0.05;
  int amr_max_level = 3; //deepest refinement level (0 = root grid only)
  Real amr_radius = 0.0; //contrast refinement only within this distance of the particle [cm] (<= 0: everywhere)
  //physics
  Real gamma;
  Real Omega;
  //boundary conditions
  std::string bc_left, bc_right, bc_top, bc_bottom;
  //gas initial condition
  std::string gas_profile;
  Real gas_rho0, gas_p0, gas_u0, gas_v0; 
  Real gas_mu = 2.3; //mean molecular weight (for heat conduction, gas temperature)
  Real cond_phi = 1.0; //heat conduction saturation q_sat = cond_phi rho c_s^3 (<= 0: no flux limiter)
  int mg_small_level_cells = 2048; //multigrid levels up to this size run in one team kernel (GPU launch tuning)
  //particle initial condition
  Real pz0, pr0, pvz0, pvr0, pmass, pT0, pradius;
  //position relative to zmin/rmin (exact, unlike particle_z - zmin): used instead of particle_z/particle_r if given
  bool has_pdz = false, has_pdr = false;
  Real pdz0 = 0.0, pdr0 = 0.0;
  //initial speed in units of the gas sound speed sqrt(gamma p0/rho0), towards the midplane (z = 0);
  //overrides particle_vz if given (sets vr = particle_vr as usual)
  bool has_pmach = false;
  Real pmach = 0.0;
  //depth of the gas the particle's drag acts on (deposit volume = dz*dr*depth):
  //"cell" = local dr (default, like a 3D cell), "ring" = 2*pi*r (axisymmetric), or a length in cm
  std::string particle_depth = "cell";
  //output
  std::string outdir;
};

//parameter and setup functions
SimParams readParams(const std::string& filename);
BCType parseBC(const std::string& s);
void init_cond(Mesh& mesh, const SimParams& par);
Particle initParticle(const SimParams& par);
void setupOutputDir(const std::string& dir);
void writeMetadata(const std::string& outdir, const SimParams& par);
std::ofstream openGridFile(const std::string& outdir, Mesh& mesh);
void writeFrame(std::ofstream& f, Mesh& mesh, Real t);
std::ofstream openParticleFile(const std::string &outdir);
void writeParticle(std::ofstream &f, const Particle &p, Real t, int n);
