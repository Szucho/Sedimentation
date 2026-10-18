#include "hydro_io.hpp"
#include "particle.hpp"
#include <fstream>
#include <stdexcept>
//CLAUDE FIX: std::lround / std::exp and std::min / std::max used in init_cond
#include <cmath>
#include <algorithm>

/*
 * Here we handle input output
 *
 * parameter file parsing, boundary condition mapping, 
 * initial state setup via Kokkos device kernels,
 * simulation metadata logging, and binary grid/frame serialization.
*/

SimParams readParams(const std::string& filename){ //reads in params.txt
  std::ifstream f(filename);
  if(!f) throw std::runtime_error("Cannot open param file: " + filename);

  std::map<std::string, std::string> kv;
  std::vector<RefinementRegion> refine_regions;
  std::string line;
  while(std::getline(f, line)){ //file read in
    auto comment = line.find('#'); //# denotes comments
    if (comment != std::string::npos) line = line.substr(0, comment);
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    std::istringstream ss(line);
    std::string key; //we use keys in the input file
    ss >> key;
    if(key == "refine"){ //static refine
      RefinementRegion r;
      if(!(ss >> r.x1min >> r.x1max >> r.x2min >> r.x2max >> r.level))
        throw std::runtime_error("Malformed 'refine' line: " + line);
      refine_regions.push_back(r);
      continue;
    }
    std::string val;
    if(ss >> val) kv[key] = val;
  }

  auto has = [&](const std::string& k) { return kv.find(k) != kv.end(); };
  auto get = [&](const std::string& k) -> std::string{
      auto it = kv.find(k);
      if(it == kv.end()) throw std::runtime_error("Missing parameter: " + k);
      return it->second;
  };
  auto getd = [&](const std::string& k){ return std::stod(get(k)); };
  auto geti = [&](const std::string& k){ return std::stoi(get(k)); };

  SimParams p; //just reading in all the other parameters
  p.Nz   = getd("Nz");    p.Nr  = getd("Nr");   p.Nt  = getd("Nt");
  p.t0   = getd("t0");    p.tf  = getd("tf");
  p.zmin = getd("zmin");  p.zmax = getd("zmax");
  p.rmin = getd("rmin");  p.rmax = getd("rmax");
  //refinement parameters
  p.nx1_block = has("nx1_block") ? geti("nx1_block") : (int)p.Nz;
  p.nx2_block = has("nx2_block") ? geti("nx2_block") : (int)p.Nr;
  p.refine_regions = std::move(refine_regions);
  if(has("amr_refine"))     p.amr_refine     = getd("amr_refine");
  if(has("amr_derefine"))   p.amr_derefine   = getd("amr_derefine");
  if(has("amr_max_level"))  p.amr_max_level  = geti("amr_max_level");
  if(has("amr_radius"))     p.amr_radius     = getd("amr_radius");
  //physics
  p.gamma = getd("gamma");
  p.Omega = getd("Omega");
  //BC-s
  p.bc_left   = get("bc_left");
  p.bc_right  = get("bc_right");
  p.bc_top    = get("bc_top");
  p.bc_bottom = get("bc_bottom");
  //gas initial condition
  p.gas_profile = get("gas_profile");
  p.gas_rho0    = getd("gas_rho0");
  p.gas_p0      = getd("gas_p0");
  p.gas_u0      = getd("gas_u0");
  p.gas_v0      = getd("gas_v0");
  if(has("gas_mu")) p.gas_mu = getd("gas_mu");
  if(has("cond_phi")) p.cond_phi = getd("cond_phi");
  if(has("mg_small_level_cells")) p.mg_small_level_cells = geti("mg_small_level_cells");
  //particle initial conditions
  //particle_dz / particle_dr (relative to zmin / rmin) or particle_z / particle_r (absolute)
  p.has_pdz     = has("particle_dz");
  p.has_pdr     = has("particle_dr");
  if(p.has_pdz) p.pdz0 = getd("particle_dz"); else p.pz0 = getd("particle_z");
  if(p.has_pdr) p.pdr0 = getd("particle_dr"); else p.pr0 = getd("particle_r");
  if(p.has_pdz) p.pz0 = p.zmin + p.pdz0; //absolute values only for printing / metadata
  if(p.has_pdr) p.pr0 = p.rmin + p.pdr0;
  p.has_pmach   = has("particle_mach");
  if(p.has_pmach) p.pmach = getd("particle_mach");
  p.pvz0        = p.has_pmach ? 0.0 : getd("particle_vz");
  p.pvr0        = getd("particle_vr");
  p.pmass       = getd("particle_mass");
  p.pT0         = getd("particle_T");
  p.pradius      = getd("particle_radius");
  if(has("particle_depth")){
    p.particle_depth = get("particle_depth");
    if(p.particle_depth != "cell" && p.particle_depth != "ring"){
      std::size_t pos = 0;
      Real depth = 0.0;
      try{ depth = std::stod(p.particle_depth, &pos); }catch(const std::exception&){ pos = 0; }
      if(pos != p.particle_depth.size() || !(depth > 0.0))
        throw std::invalid_argument("particle_depth must be cell, ring or a positive length in cm");
    }
  }

  p.outdir      = get("outdir");

  if(p.pmass <= 0 || p.pradius <= 0 || p.pT0 <= 0){
    throw std::invalid_argument("particle mass, radius and temperature must be positive");
  }
  return p;
}

//now that we have BC-s in the input, we want to use those as keys for the simulation
BCType parseBC(const std::string& s){
  if      (s == "Open")      return BCType::Open;
  else if (s == "Closed")    return BCType::Closed;
  else if (s == "Periodic")  return BCType::Periodic;
  else if (s == "Dirichlet") return BCType::Dirichlet;
  else throw std::invalid_argument("Unknown BC type: " + s);
}

//just some initial conditions
//now only scalars and Kokkos Views are captured so that GPU can handle it
void init_cond(Mesh& mesh, const SimParams& par){
  View4D data = mesh.get_data_view();
  Real gamma = par.gamma;
  //local mesh coordinates (the mesh domain starts at 0), z0 = absolute z of the local origin
  Real zmin = mesh.inputs().domain.x1min;
  Real zmax = mesh.inputs().domain.x1max;
  Real z0_abs = mesh.inputs().x1_origin;
  std::string profile = par.gas_profile;
  Real rho0 = par.gas_rho0;
  Real p0 = par.gas_p0;
  //scalar copies of the SimParams fields used inside the kernels
  Real u0 = par.gas_u0;
  Real v0 = par.gas_v0;
  Real Omega = par.Omega;
  int nghost = mesh.inputs().nghost;

  //check the profile and print the gaussian_z info once
  if(profile != "uniform" && profile != "gaussian_z" && profile != "Sod_x" && profile != "contact_x")
    throw std::runtime_error("Unknown gas_profile: " + profile);
  if(profile == "gaussian_z"){
    Real cs2 = p0/rho0;
    Real H2  = 2.0*cs2/(Omega*Omega);
    std::cout << "gaussian_z: adiabatic equilibrium\n"
              << "  cs = " << std::sqrt(cs2) << " cm/s\n"
              << "  H      = " << std::sqrt(H2) << " cm\n";
  }

  for(auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    int b_slot = b.slot_id;
    int total_i = b.block_size.nx1 + 2*nghost;
    int total_j = b.block_size.nx2 + 2*nghost;
    //block geometry as plain scalars for the lambdas
    int  is    = b.is;
    Real x1min = b.block_size.x1min;
    Real dx1   = b.dx1();

    if(profile == "uniform"){
      Kokkos::parallel_for("InitUniform", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
        KOKKOS_LAMBDA(int i, int j) {
          data(b_slot, i, j, 0) = rho0;
          data(b_slot, i, j, 1) = rho0*u0;
          data(b_slot, i, j, 2) = rho0*v0;
          data(b_slot, i, j, 3) = p0/(gamma - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);
        }
      );
    }else if(profile == "gaussian_z"){
      Real cs2 = p0/rho0;
      Real H2  = 2.0*cs2/(Omega*Omega);
      Real K = cs2;
      Real dz = dx1;

      //the recurrence runs on this block's level grid in GLOBAL cell indices:
      //anchored with the exact Gaussian at the domain's first physical cell
      //(I = 0, like before), stepped up to the block's last ghost and down to the domain's lower ghosts.
      //Blocks on the same level therefore get identical, discretely hydrostatic values.
      auto z_of = [&](long I){ return zmin + (I + 0.5)*dz; };
      long I0   = std::lround((x1min - zmin)/dz);   //global index of this block's first physical cell
      long Ilo  = I0 - nghost;                      //global index of local cell 0
      long Ihi  = I0 + b.block_size.nx1 + nghost - 1;
      long Imin = std::min(Ilo, 0L);
      long Imax = std::max(Ihi, 0L);

      std::vector<Real> rho_glob(Imax - Imin + 1), p_glob(Imax - Imin + 1);
      auto at = [&](long I) -> long { return I - Imin; };
      Real z0 = z0_abs + z_of(0); //absolute z of the anchor cell
      rho_glob[at(0)] = rho0*std::exp(-z0*z0/H2);
      p_glob[at(0)]   = p0*std::exp(-z0*z0/H2);
      //upward / downward sequential recurrence relation (the ratio is symmetric, so both directions agree)
      for(long I = 1; I <= Imax; ++I){
        Real Dphi  = phi_diff(z_of(I), z_of(I-1), z0_abs, Omega);
        Real ratio = (2.0*K - Dphi)/(2.0*K + Dphi);
        rho_glob[at(I)] = rho_glob[at(I-1)]*ratio;
        p_glob[at(I)]   = p_glob[at(I-1)]*ratio;
      }
      for(long I = -1; I >= Imin; --I){
        Real Dphi  = phi_diff(z_of(I), z_of(I+1), z0_abs, Omega);
        Real ratio = (2.0*K - Dphi)/(2.0*K + Dphi);
        rho_glob[at(I)] = rho_glob[at(I+1)]*ratio;
        p_glob[at(I)]   = p_glob[at(I+1)]*ratio;
      }

      //column copied into Kokkos Views (std::vector can't be captured in device code)
      View1D rho_col("rho_col", total_i), p_col("p_col", total_i);
      auto h_rho = Kokkos::create_mirror_view(rho_col);
      auto h_p   = Kokkos::create_mirror_view(p_col);
      for(int i = 0; i < total_i; i++){
        h_rho(i) = rho_glob[at(Ilo + i)];
        h_p(i)   = p_glob[at(Ilo + i)];
      }
      Kokkos::deep_copy(rho_col, h_rho);
      Kokkos::deep_copy(p_col, h_p);

      //copy or assign the precomputed column profile across all radial coordinates (j) in parallel
      Kokkos::parallel_for("InitGaussianZ_Recursive",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
        KOKKOS_LAMBDA(int i, int j) {
          Real rho_i = rho_col(i);
          Real p_i   = p_col(i);
          Real E     = p_i/(gamma - 1.0) + 0.5*rho_i*(u0*u0 + v0*v0);

          data(b_slot, i, j, 0) = rho_i;
          data(b_slot, i, j, 1) = rho_i * u0;
          data(b_slot, i, j, 2) = rho_i * v0;
          data(b_slot, i, j, 3) = E;
        }
      );
    }else if(profile == "Sod_x" || profile == "contact_x"){
      //contact_x: density (temperature) jump at constant pressure, at rest -> only heat conduction evolves it
      bool contact = (profile == "contact_x");
      Real rho_L = rho0,     p_L = p0,     u_L = 0.0, v_L = 0.0;
      Real rho_R = contact ? 0.5*rho0 : rho0/8.0, p_R = contact ? p0 : 0.1*p0, u_R = 0.0, v_R = 0.0;
      Real x_mid = 0.5 * (zmin + zmax);

      //initialize across the ENTIRE block including ghosts so no cell is left with NaN/zeros
      Kokkos::parallel_for("InitSod", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {total_i, total_j}),
        KOKKOS_LAMBDA(int i, int j) {
          //captured scalars
          Real x = x1min + (i - is + 0.5)*dx1;
          Real rho = (x < x_mid) ? rho_L : rho_R;
          Real p   = (x < x_mid) ? p_L   : p_R;
          Real u   = (x < x_mid) ? u_L   : u_R;
          Real v   = (x < x_mid) ? v_L   : v_R;
          Real E   = p/(gamma - 1.0) + 0.5*rho*(u*u + v*v);

          data(b_slot, i, j, 0) = rho;
          data(b_slot, i, j, 1) = rho*u;
          data(b_slot, i, j, 2) = rho*v;
          data(b_slot, i, j, 3) = E;
        }
      );
    }
  }
  //making sure the device data is complete before anything reads it on the host
  Kokkos::fence();
}


Particle initParticle(const SimParams &par){
  Particle p;
  //local coordinates, relative to zmin / rmin (see MeshInputs::x1_origin)
  p.z      = par.has_pdz ? par.pdz0 : par.pz0 - par.zmin;
  p.r      = par.has_pdr ? par.pdr0 : par.pr0 - par.rmin;
  p.vz     = par.pvz0;
  if(par.has_pmach){
    //adiabatic sound speed of the initial gas: p/rho = p0/rho0 everywhere for uniform and gaussian_z
    Real cs   = std::sqrt(par.gamma*par.gas_p0/par.gas_rho0);
    Real sign = (par.zmin + p.z > 0.0) ? -1.0 : 1.0; //towards the midplane
    p.vz = sign*par.pmach*cs;
  }
  p.vr     = par.pvr0;
  p.mass   = par.pmass;
  p.radius = par.pradius;
  p.T      = par.pT0;
  p.active = true;
  return p;
}



//output handling
void setupOutputDir(const std::string& dir){
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) throw std::runtime_error("Could not create output directory: " + dir);
    std::cout << "Output directory: " << dir << "\n";
}

//metadata contains simulation parameters
void writeMetadata(const std::string& outdir, const SimParams& par){
    std::ofstream f(outdir + "/meta.txt");
    if(!f) throw std::runtime_error("Cannot open meta.txt");
    f << std::setprecision(17) << std::scientific;
    f << "zmin "  << par.zmin  << "\n"
      << "zmax "  << par.zmax  << "\n"
      << "rmin "  << par.rmin  << "\n"
      << "rmax "  << par.rmax  << "\n"
      << "gamma " << par.gamma << "\n"
      << "Nz "    << (int)par.Nz << "\n"
      << "Nr "    << (int)par.Nr << "\n"
      << "nx1_block " << par.nx1_block << "\n"
      << "nx2_block " << par.nx2_block << "\n"
      << "Omega "     << par.Omega     << "\n"
      << "gas_rho0 "  << par.gas_rho0  << "\n"
      << "gas_p0 "    << par.gas_p0    << "\n"
      << "gas_mu "    << par.gas_mu    << "\n"
      << "particle_radius " << par.pradius << "\n"
      //same keys as the old uniform-grid writer (scripts/particle.py): cs_iso is p0/rho0 = c_s^2 (!),
      //H_iso = sqrt(2 p0/rho0)/Omega is the gaussian_z scale height
      << "cs_iso " << par.gas_p0/par.gas_rho0 << "\n"
      << "H_iso "  << std::sqrt(2.0*par.gas_p0/par.gas_rho0)/par.Omega << "\n"
      //grid.bin block bounds and particle.bin z, r are LOCAL: relative to zmin (z) and rmin (r)
      << "local_coords 1\n";
}

//grid.bin contains simulation data
std::ofstream openGridFile(const std::string& outdir, Mesh& mesh){
  //NOTE: mesh is unused (block geometry is written per frame in writeFrame);
  //kept it for the existing call sites °-~°
  (void)mesh;
  std::string path = outdir + "/grid.bin";
  std::ofstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("Cannot open grid output file: " + path);
  return f;
}

//writing into grid.bin
void writeFrame(std::ofstream& f, Mesh& mesh, Real t){
  //write timestamp
  f.write(reinterpret_cast<const char*>(&t), sizeof(Real));

  //write active block count for this specific frame
  uint64_t nblocks = mesh.blocks().size();
  f.write(reinterpret_cast<const char*>(&nblocks), sizeof(uint64_t));

  //write block geometries for this frame
  for(const auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    int32_t level = b.loc.level;
    int64_t lx1 = b.loc.lx1, lx2 = b.loc.lx2;
    uint64_t nx1 = (uint64_t)b.block_size.nx1, nx2 = (uint64_t)b.block_size.nx2;
    f.write(reinterpret_cast<const char*>(&level), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&lx1), sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(&lx2), sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(&b.block_size.x1min), sizeof(Real));
    f.write(reinterpret_cast<const char*>(&b.block_size.x1max), sizeof(Real));
    f.write(reinterpret_cast<const char*>(&b.block_size.x2min), sizeof(Real));
    f.write(reinterpret_cast<const char*>(&b.block_size.x2max), sizeof(Real));
    f.write(reinterpret_cast<const char*>(&nx1), sizeof(uint64_t));
    f.write(reinterpret_cast<const char*>(&nx2), sizeof(uint64_t));
  }

  //write cell data payload
  View4D device_data = mesh.get_data_view();
  HostView4D host_data = Kokkos::create_mirror_view(device_data);
  Kokkos::deep_copy(host_data, device_data);

  for(const auto& bptr : mesh.blocks()){
    const MeshBlock& b = *bptr;
    for(int i = b.is; i <= b.ie; i++){
      for(int j = b.js; j <= b.je; j++){
        Real cell_data[NVAR];
        for(size_t k = 0; k < NVAR; ++k){
          cell_data[k] = host_data(b.slot_id, i, j, k);
        }
        f.write(reinterpret_cast<const char*>(cell_data), sizeof(Real) * NVAR);
      }
    }
  }
}


std::ofstream openParticleFile(const std::string &outdir){
  std::string path = outdir + "/particle.bin";
  std::ofstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("Cannot open particle.bin: " + path);
  return f;
}

void writeParticle(std::ofstream &f, const Particle &p, Real t, int n){
  f.write(reinterpret_cast<const char*>(&n), sizeof(int));
  f.write(reinterpret_cast<const char*>(&t), sizeof(Real));
  f.write(reinterpret_cast<const char*>(&p.z), sizeof(Real));
  f.write(reinterpret_cast<const char*>(&p.r), sizeof(Real));
  f.write(reinterpret_cast<const char*>(&p.vz), sizeof(Real));
  f.write(reinterpret_cast<const char*>(&p.vr), sizeof(Real));
  f.write(reinterpret_cast<const char*>(&p.T), sizeof(Real));

  int active_val = p.active ? 1 : 0;
  f.write(reinterpret_cast<const char*>(&active_val), sizeof(int));
}
