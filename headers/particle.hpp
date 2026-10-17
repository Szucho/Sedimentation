#pragma once

#include "hydro_config.hpp"


/*
 Particle script
 Bertalan Szuchovszky - 01.05.2026. (I think)

 Lagrange view of a single solid particle.
 The particle is interpolated to the gas grid via Cloud In Cell (CIC) method,
 back-reaction is calculated the same way.
 The particle equation of motion is integrated with a Leapfrog algorightm.
    -> gravitational potential calulated explicitly
    -> implicit calculation of Epstein drag in gas interacting with particle ~ explicit became unstable :(
       (subsonic/supersonic interpolated Epstein drag, Kwok 1975)
    -> implicit (backward Euler) particle temperature: heating by drag, convective + radiative exchange with the gas

 UPDATED: 28.09.2026
*/

class Mesh;

//constants moved here from particle.cpp (the inline functions below need them in every TU)
//they live in a namespace so they don't clash with the PI / k_B / m_p that main.cpp defines globally.
namespace particle_const {
  inline constexpr Real PI       = 3.141592653589793;
  inline constexpr Real c_heat   = 8e6;       // erg/g/K
  inline constexpr Real h_conv   = 1e3;       // erg/s/cm^2/K
  inline constexpr Real eps_rad  = 0.8;
  inline constexpr Real sigma_SB = 5.67e-5;   // erg/s/cm^2/K^4
  inline constexpr Real T_melt   = 1800.0;    // K
  inline constexpr Real k_B      = 1.380649e-16;
  inline constexpr Real m_p      = 1.672622e-24;
  //heating now uses the particle's own mass, so it can't disagree with mass/radius
}

template<typename T>
KOKKOS_INLINE_FUNCTION constexpr T QUAD(T x){ return x*x*x*x; }


struct Particle {
  Real z, r;   //positions, LOCAL (relative to zmin/rmin, see MeshInputs::x1_origin)
  Real vz, vr; //velocities
  Real mass;   //total mass of the solid chondrite particle
  Real radius; //it will be a circle
  Real T;
  bool active=true;
  //velocity change due to drag only, of the LAST half-kick (set by half_kick_and_heat).
  //source_projection gives exactly -mass*dv_drag to the gas, so momentum is conserved with the implicit drag.
  Real dvz_drag = 0.0, dvr_drag = 0.0;
};

struct GasAtParticle {
  Real rho, u, v, p, cs; //state of gas will be interpolated to the particle location from the HLLC grid
};


//CIC stencil flattened onto the cells that actually hold the gas. The 4 CIC cells are laid out with the spacing of
//the block containing the particle (block-local coordinates!!);
//each one is then mapped onto the leaf cells covering its footprint,
//in THEIR block's local indices (same block, a same-level/coarser/finer neighbour, or the periodic image). 
//entry k reads/receives weight w[k] = sum_c W_c * (area of cell c covered by k)/(area of c), sum_k w[k] = 1.
//interpolation and back-reaction use the same list, so they are exact adjoints and never touch ghost cells.
inline constexpr int CIC_MAX_CELLS = 64;
struct CICStencil {
  int n = 0;
  int b[CIC_MAX_CELLS], i[CIC_MAX_CELLS], j[CIC_MAX_CELLS];
  Real w[CIC_MAX_CELLS];
  Real vol[CIC_MAX_CELLS]; //deposit volume of each entry's cell (filled by the caller, see particle_cell_volume)
};

//device-safe replacement for std::array<Real,2> (structured bindings still work)
struct Accel2 {
  Real z, r;
};



//ALL KOKKOS_INLINE_FUNCTION for GPU acceleration
//from HLLC grid to particle interpolation
KOKKOS_INLINE_FUNCTION
GasAtParticle interpolate_gas(const View4D& data, const CICStencil& S, Real gamma){
  Real rho = 0.0, rhou = 0.0, rhov = 0.0, rhoe = 0.0;
  for(int k = 0; k < S.n; ++k){
    rho  += S.w[k]*data(S.b[k], S.i[k], S.j[k], 0);
    rhou += S.w[k]*data(S.b[k], S.i[k], S.j[k], 1);
    rhov += S.w[k]*data(S.b[k], S.i[k], S.j[k], 2);
    rhoe += S.w[k]*data(S.b[k], S.i[k], S.j[k], 3);
  }

  GasAtParticle g;

  g.rho = rho;
  g.u   = rhou/rho;
  g.v   = rhov/rho;
  g.p   = (gamma-1.0)*(rhoe-0.5*rho*(g.u*g.u + g.v*g.v));
  //interpolating conserved vars can give a slightly negative pressure -> NaN in sqrt; floor it at 0
  g.p   = Kokkos::fmax(g.p, 0.0);
  g.cs  = Kokkos::sqrt(gamma*g.p/g.rho);

  return g;
}


//1/M_eff: how fast the CIC-interpolated gas velocity responds to a momentum J given to the gas through the same
//stencil (entry k gets w_k*J/V_k): delta u_interp = J * sum(w_k^2/V_k)/rho_interp = J/M_eff.
//used by the implicit two-way drag, so the gas velocity the kick predicts is exactly the one the deposit produces.
KOKKOS_INLINE_FUNCTION
Real cic_inv_gas_mass(const View4D& data, const CICStencil& S){
  Real rho = 0.0, w2v = 0.0;
  for(int k = 0; k < S.n; ++k){
    rho += S.w[k]*data(S.b[k], S.i[k], S.j[k], 0);
    w2v += S.w[k]*S.w[k]/S.vol[k];
  }
  return w2v/rho;
}


//Epstein drag coefficient with the subsonic/supersonic interpolation of Kwok (1975), the fit to the exact
//Baines et al. (1965) / Draine & Salpeter (1979) drag: F_drag = -K(w)*(v_p - u_gas), w = |v_p - u_gas|,
//  K(w) = 4pi/3 rho a^2 v_th sqrt(1 + 9pi/128 w^2/c^2) = pi a^2 rho sqrt(w^2 + 128/(9pi) c^2),
//  c^2 = p/rho = k_B T/(mu m_p) (isothermal), v_th = sqrt(8p/(pi rho)).
//w << c: linear Epstein K = 4pi/3 rho a^2 v_th, w >> c: ram pressure F = pi a^2 rho w^2.
//Written as sqrt(rho^2 w^2 + 128/(9pi) rho p), so zero gas density gives K = 0 instead of NaN.
KOKKOS_INLINE_FUNCTION
Real epstein_coeff(const Particle& p, const GasAtParticle& gas, Real w){
  using particle_const::PI;
  return PI*p.radius*p.radius*Kokkos::sqrt(gas.rho*gas.rho*w*w + 128.0/(9.0*PI)*gas.rho*gas.p);
}

KOKKOS_INLINE_FUNCTION
Accel2 drag_accel(const Particle& p, const GasAtParticle& gas){
  Real wz = p.vz - gas.u, wr = p.vr - gas.v;
  Real coeff = epstein_coeff(p, gas, Kokkos::sqrt(wz*wz + wr*wr));
  return {-coeff/p.mass*wz, -coeff/p.mass*wr};
}

KOKKOS_INLINE_FUNCTION
Accel2 grav_accel(const Particle& p, Real Omega, Real z0){
  return { -Omega*Omega*(z0 + p.z), 0.0}; //p.z is local, z0 its absolute origin
}


//relative speed s = |w'| after the implicit two-way kick with the speed dependent K (see half_kick_and_heat):
//  s*(1 + h*beta*K(s)) = W,   W = |v - u + g h|,   beta = 1/m + 1/M_eff
//the left side is increasing and convex in s, so Newton started from s = W (>= root) decreases monotonically
//onto the unique root in [0, W] without overshooting.
KOKKOS_INLINE_FUNCTION
Real implicit_drag_speed(const Particle& p, const GasAtParticle& gas, Real W, Real h, Real beta){
  using particle_const::PI;
  if(!(W > 0.0)) return 0.0;
  Real a  = PI*p.radius*p.radius*h*beta;              //K(s)*h*beta = a*q(s)
  Real r2 = gas.rho*gas.rho;
  Real c2 = 128.0/(9.0*PI)*gas.rho*gas.p;             //q(s) = sqrt(rho^2 s^2 + c2)
  Real s  = W;
  for(int it = 0; it < 200; ++it){
    Real q  = Kokkos::sqrt(r2*s*s + c2);
    Real g  = s + a*s*q - W;
    Real dg = 1.0 + a*(q + (q > 0.0 ? r2*s*s/q : 0.0));
    Real ds = g/dg;
    s -= ds;
    if(ds <= 1e-14*s) break;
  }
  return s;
}

//backward-Euler temperature step of length h:
//  T' = T + h*[H - a_c (T' - T_gas) - a_r (T'^4 - T_gas^4)]
//H = frictional heating rate, a_c / a_r = convective / radiative rate coefficients (all in K/s, K/s/K, K/s/K^4).
//f(T') = T' - T - h H + h a_c (T' - T_gas) + h a_r (T'^4 - T_gas^4) is increasing and convex for T' > 0, and
//f(T_up) >= 0 for T_up = T + h (H + a_c T_gas + a_r T_gas^4), so Newton from T_up converges monotonically from above:
//unconditionally stable, never undershoots below the equilibrium, never goes negative.
KOKKOS_INLINE_FUNCTION
Real implicit_temperature(Real T, Real T_gas, Real H, Real a_c, Real a_r, Real h){
  Real Tg4 = QUAD(T_gas);
  Real Tn  = T + h*(H + a_c*T_gas + a_r*Tg4);
  for(int it = 0; it < 200; ++it){
    Real f  = Tn - T - h*H + h*a_c*(Tn - T_gas) + h*a_r*(QUAD(Tn) - Tg4);
    Real df = 1.0 + h*a_c + 4.0*h*a_r*Tn*Tn*Tn;
    Real dT = f/df;
    Tn -= dT;
    if(dT <= 1e-14*Tn) break;
  }
  return Tn;
}


//shared implicit half-kick + half temperature step
//two-way implicit drag (backward Euler for the particle AND the gas it sees, momentum conserving):
//  m v' = m v + m g h - K h w',   M_eff u' = M_eff u + K h w',   w' = v' - u',   h = dt/2, K = epstein_coeff(|w'|)
//  -> w' = (v - u + g h)/(1 + K(|w'|) h (1/m + 1/M_eff)),  v' = v + g h - (K h/m) w'
//w' is parallel to v - u + g h, only its length needs the nonlinear solve (implicit_drag_speed).
//inv_gas_mass = 1/M_eff (cic_inv_gas_mass). 0 means infinitely heavy gas (no back-reaction), which is the
//one-way formula v' = (v + (g + u/t_s) h)/(1 + h/t_s). The explicit back-reaction was unstable once
//K h/M_eff > ~1, i.e. when the gas in the CIC cells is lighter than what the drag moves in one step (small cells).
//dvz_drag / dvr_drag hold the drag velocity change of THIS half-kick; deposit it with source_projection right after.
//the temperature then takes a backward-Euler step of h with the frictional heating K(|w'|) |w'|^2 of the kick.
KOKKOS_INLINE_FUNCTION
void half_kick_and_heat(Particle& p, const GasAtParticle& gas, Real Omega, Real z0, Real dt, Real mu,
                        Real inv_gas_mass = 0.0){
  using namespace particle_const;
  auto [g_z, g_r] = grav_accel(p, Omega, z0);
  Real h = 0.5*dt;

  //relative velocity after the kick, then the particle velocity
  Real w0z = p.vz - gas.u + g_z*h;
  Real w0r = p.vr - gas.v + g_r*h;
  Real W   = Kokkos::sqrt(w0z*w0z + w0r*w0r);
  Real s   = implicit_drag_speed(p, gas, W, h, 1.0/p.mass + inv_gas_mass);
  Real factor = (W > 0.0) ? s/W : 1.0;
  Real wz = w0z*factor;
  Real wr = w0r*factor;

  //work with 1/t_stop instead of t_stop, so zero gas density/pressure gives "no drag"
  Real coeff      = epstein_coeff(p, gas, s);
  Real inv_t_stop = coeff/p.mass;

  p.dvz_drag = -h*inv_t_stop*wz;
  p.dvr_drag = -h*inv_t_stop*wr;
  p.vz += g_z*h + p.dvz_drag;
  p.vr += g_r*h + p.dvr_drag;

  Real A     = 4.0*PI*p.radius*p.radius; //surface area
  Real T_gas = gas.p*m_p*mu/(gas.rho*k_B);
  Real mc    = p.mass*c_heat;

  //frictional heating uses the RELATIVE speed |v_p - u_gas|^2 in both directions (was p.vz^2 only),
  //and the drag power coeff*dv^2 divided by the particle's own mass (was a hardcoded rho_m = 3 g/cm^3)
  //relative speed after the kick (gas response included); equals p.v - gas.u when inv_gas_mass = 0
  Real H   = coeff*s*s/mc;
  Real a_c = h_conv*A/mc;
  Real a_r = eps_rad*sigma_SB*A/mc;
  p.T = implicit_temperature(p.T, T_gas, H, a_c, a_r, h);
}


KOKKOS_INLINE_FUNCTION
bool leapfrog_first_half(Particle& p,
                         const GasAtParticle& gas,
                         Real Omega, Real z0,
                         Real zmin, Real rmin,
                         Real zmax, Real rmax,
                         Real dt, Real mu, Real inv_gas_mass = 0.0) {

  half_kick_and_heat(p, gas, Omega, z0, dt, mu, inv_gas_mass);

  //drift (local coordinates: a drift of ~1e-4 cm is no longer rounded away)
  p.z += p.vz*dt;
  p.r += p.vr*dt;

  if (p.z < zmin || p.z > zmax || p.r < rmin || p.r > rmax){
      p.active = false;
      return false;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
void leapfrog_second_half(Particle& p,
                          const GasAtParticle& gas_new,  //gas at t_{n+1}
                          Real Omega, Real z0,
                          Real dt, Real mu, Real inv_gas_mass = 0.0){
  half_kick_and_heat(p, gas_new, Omega, z0, dt, mu, inv_gas_mass);
}


//back reaction: gives the gas -mass*(dvz_drag, dvr_drag), the momentum the particle got from drag in the LAST
//half-kick. Call it right after each leapfrog half (with the stencil used for that kick's inv_gas_mass),
//so the next kick sees the pushed gas. Only the kinetic energy of the gas changes (exactly, per cell),
//its internal energy / pressure is untouched - the frictional heat goes into the particle temperature.
//S.vol must be VOLUMES: dz*dr*depth for the planar 2D solver (see particle_depth in main.cpp).
void source_projection(const View4D& grid_data, const Particle& p, const CICStencil& S);

//helper to find which active block (slot_id) contains the particle; -1 if none.
int find_particle_block(const Mesh& mesh, Real z, Real r);

//CIC stencil of a particle at (z, r), see CICStencil. Returns false if no block contains the particle.
//Stencil cells outside a non-periodic domain are folded back onto the boundary cell (momentum stays in the domain);
//periodic directions wrap. S.vol is left at 0 for the caller to fill.
bool build_cic_stencil(const Mesh& mesh, Real z, Real r, CICStencil& S);
