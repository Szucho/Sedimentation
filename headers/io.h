#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include "grid.h"
#include "particle.h"
#include <filesystem>

constexpr double PI_IO = 3.141592653589793;


/*
 Input / Output header
 Bertalan Szuchovszky 20.04.2026.

 Only active for one instance, reads a params.txt file containing initial values of particle, gas, BC-s, ...
 Hanldes file writing into a specified folder (in params.txt), creates
  -> meta.txt containing metadata (grid params & physical constants i.e gamma)
  -> grid.bin containing all state vals on every gridpoint at every timestep
  -> particle.bin containing the position, velocity,... of particle
 There are some initial conditions available 
 !!! DO NOT FORGET TO CHANGE IT FOR A SPECIFIC INITIAL CONDITION !!!

*/

//everything the simulation needs to run
struct SimParams {
  //grid
  double Nz, Nr, Nt;
  double t0, tf;
  double zmin, zmax, rmin, rmax;
  //physics
  double gamma, Omega;
  //boundary conditions
  std::string bc_left, bc_right, bc_top, bc_bottom;
  //gas initial condition
  std::string gas_profile;
  double gas_rho0, gas_p0, gas_u0, gas_v0;
  //particle initial condition
  double pz0, pr0, pvz0, pvr0, pmass, pradius;
  //output
  std::string outdir;
};


//parameter file reader
inline SimParams readParams(const std::string& filename) {
  std::ifstream f(filename);
  if (!f) throw std::runtime_error("Cannot open param file: " + filename);

  std::map<std::string, std::string> kv;
  std::string line;
  while (std::getline(f, line)) {
    auto comment = line.find('#');
    if (comment != std::string::npos) line = line.substr(0, comment);
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    std::istringstream ss(line);
    std::string key, val;
    if (ss >> key >> val) kv[key] = val;
  }

  auto get = [&](const std::string& k) -> std::string {
    auto it = kv.find(k);
    if (it == kv.end()) throw std::runtime_error("Missing parameter: " + k);
    return it->second;
  };
  auto getd = [&](const std::string& k) {
    try { return std::stod(get(k)); }
    catch (const std::invalid_argument&) {
      throw std::runtime_error("Parameter '" + k + "' is not a valid number");
    }
  };

  SimParams p;

  //grid
  p.Nz   = getd("Nz");    p.Nr  = getd("Nr");   p.Nt  = getd("Nt");
  p.t0   = getd("t0");    p.tf  = getd("tf");
  p.zmin = getd("zmin");  p.zmax = getd("zmax");
  p.rmin = getd("rmin");  p.rmax = getd("rmax");

  //physics
  p.gamma = getd("gamma");
  p.Omega = getd("Omega");

  //BC
  p.bc_left   = get("bc_left");
  p.bc_right  = get("bc_right");
  p.bc_top    = get("bc_top");
  p.bc_bottom = get("bc_bottom");

  //gas IC
  p.gas_profile = get("gas_profile");
  p.gas_rho0    = getd("gas_rho0");
  p.gas_p0      = getd("gas_p0");
  p.gas_u0      = getd("gas_u0");
  p.gas_v0      = getd("gas_v0");

  //particle IC
  p.pz0     = getd("particle_x");
  p.pr0     = getd("particle_y");
  p.pvz0    = getd("particle_vx");
  p.pvr0    = getd("particle_vy");
  p.pmass   = getd("particle_mass");
  p.pradius = getd("particle_radius");

  //output
  p.outdir = get("outdir");

  //sanity checks
  if (p.Nz <= 0 || p.Nr <= 0 || p.Nt <= 0)
    throw std::invalid_argument("Nz, Nr, Nt must be positive");
  if (p.zmax <= p.zmin || p.rmax <= p.rmin)
    throw std::invalid_argument("zmax <= zmin or rmax <= rmin");
  if (p.tf <= p.t0)
    throw std::invalid_argument("tf must be greater than t0");
  if (p.gas_rho0 <= 0 || p.gas_p0 <= 0)
    throw std::invalid_argument("gas_rho0 and gas_p0 must be positive");
  if (p.pmass <= 0 || p.pradius <= 0)
    throw std::invalid_argument("particle mass and radius must be positive");

  return p;
}


//BC string -> BC type
inline BCType parseBC(const std::string& s) {
  if      (s == "Open")      return BCType::Open;
  else if (s == "Closed")    return BCType::Closed;
  else if (s == "Periodic")  return BCType::Periodic;
  else if (s == "Dirichlet") return BCType::Dirichlet;
  else throw std::invalid_argument("Unknown BC type: " + s);
}


//gas initial conditions
inline void init_cond(Grid& grid, const SimParams& par) {
  size_t nz = grid.rows() - 4;
  size_t nr = grid.cols() - 4;
  double dz = (par.zmax - par.zmin) / par.Nz;

  if (par.gas_profile == "uniform") {
    for (size_t i = 2; i < nz+2; i++) {
      for (size_t j = 2; j < nr+2; j++) {
        Cell c = grid.getCell(i,j);
        c[0] = par.gas_rho0;
        c[1] = par.gas_rho0 * par.gas_u0;
        c[2] = par.gas_rho0 * par.gas_v0;
        c[3] = par.gas_p0 / (par.gamma - 1.0)
             + 0.5 * par.gas_rho0 * (par.gas_u0*par.gas_u0 + par.gas_v0*par.gas_v0);
        grid.setCell(i,j,c);
      }
    }
  } else if (par.gas_profile == "gaussian_z") {
    double cs2 = par.gas_p0 / par.gas_rho0;
    double H2  = 2.0 * cs2 / (par.Omega * par.Omega);

    std::cout << "gaussian_z: adiabatic equilibrium\n"
              << "  cs = " << std::sqrt(cs2) << " cm/s\n"
              << "  H      = " << std::sqrt(H2) << " cm\n";

    double K = cs2;
    double z_prev = par.zmin + 0.5 * dz;
    double p_prev = par.gas_p0 * std::exp(-z_prev*z_prev/H2);
    double rho_prev = par.gas_rho0 * std::exp(-z_prev*z_prev/H2);

    for (size_t i = 2; i < nz+2; i++) {
      double p_i, rho_i;
      if (i == 2) {
        p_i   = p_prev;
        rho_i = rho_prev;
      } else {
        double z_i  = par.zmin + (i - 2) * dz + 0.5 * dz;
        double Dphi = 0.5 * par.Omega * par.Omega * (z_i*z_i - z_prev*z_prev);
        double ratio = (2.0*K - Dphi) / (2.0*K + Dphi);
        p_i   = p_prev   * ratio;
        rho_i = rho_prev * ratio;
        z_prev   = z_i;
        p_prev   = p_i;
        rho_prev = rho_i;
      }

      for (size_t j = 2; j < nr+2; j++) {
        Cell c;
        c[0] = rho_i;
        c[1] = 0.0;
        c[2] = 0.0;
        c[3] = p_i / (par.gamma - 1.0);
        grid.setCell(i,j,c);
      }
    }
  } else {
    throw std::runtime_error("Unknown gas_profile: '" + par.gas_profile + "'. Available: uniform, gaussian_z");
  }
}


//particle init cond
inline Particle initParticle(const SimParams& par) {
  Particle p;
  p.z      = par.pz0;
  p.r      = par.pr0;
  p.vz     = par.pvz0;
  p.vr     = par.pvr0;
  p.mass   = par.pmass;
  p.radius = par.pradius;
  p.active = true;
  return p;
}


//output directory
inline void setupOutputDir(const std::string& dir) {
  std::error_code ec;
  // create_directories creates the full path and doesn't error if it exists
  std::filesystem::create_directories(dir, ec); 
  
  if (ec) {
    throw std::runtime_error("Could not create output directory: " + dir + " - " + ec.message());
  }
  std::cout << "Output directory: " << dir << "\n";
}


//metadata file txt containing some important values of the simulation
inline void writeMetadata(const std::string& outdir, const SimParams& par) {
  std::string path = outdir + "/meta.txt";
  std::ofstream f(path);
  if (!f) throw std::runtime_error("Cannot open meta.txt: " + path);
  f << "zmin "  << par.zmin  << "\n"
    << "zmax "  << par.zmax  << "\n"
    << "rmin "  << par.rmin  << "\n"
    << "rmax "  << par.rmax  << "\n"
    << "gamma " << par.gamma << "\n"
    << "Nz "    << (int)par.Nz    << "\n"
    << "Nr "    << (int)par.Nr    << "\n"
    << "cs_iso " << par.gas_p0/par.gas_rho0 <<"\n"
    << "H_iso "  << 1.0/par.Omega*std::sqrt(2.0*par.gas_p0/par.gas_rho0);
}


//grid output, single binary file, all frames sequential suggested by Claude
inline std::ofstream openGridFile(const std::string& outdir, size_t nz, size_t nr) {
  std::string path = outdir + "/grid.bin";
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open grid output file: " + path);
  uint64_t nz64 = nz; uint64_t nr64 = nr;
  f.write(reinterpret_cast<const char*>(&nz64), sizeof(uint64_t));
  f.write(reinterpret_cast<const char*>(&nr64), sizeof(uint64_t));
  return f;
}

inline void writeFrame(std::ofstream& f, const Grid& grid, double t, size_t nz, size_t nr) {
  f.write(reinterpret_cast<const char*>(&t), sizeof(double));
  for (size_t i = 2; i < nz+2; i++)
    for (size_t j = 2; j < nr+2; j++)
      f.write(reinterpret_cast<const char*>(grid.cell(i,j)), 4 * sizeof(double));
}


//particle output bin, one row per timestep, open file once
inline std::ofstream openParticleFile(const std::string& outdir) {
  std::string path = outdir + "/particle.bin";
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open particle.bin: " + path);
  return f;
}

//write the raw data structure directly
inline void writeParticle(std::ofstream& f, const Particle& p, double t, int n) {
  //write 'n' and 't' followed by the particle struct data
  f.write(reinterpret_cast<const char*>(&n), sizeof(int));
  f.write(reinterpret_cast<const char*>(&t), sizeof(double));
  
  //writing particle data (x, y, vx, vy, m, r, active)
  f.write(reinterpret_cast<const char*>(&p.z), sizeof(double));
  f.write(reinterpret_cast<const char*>(&p.r), sizeof(double));
  f.write(reinterpret_cast<const char*>(&p.vz), sizeof(double));
  f.write(reinterpret_cast<const char*>(&p.vr), sizeof(double));
}
