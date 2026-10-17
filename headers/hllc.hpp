#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "hydro_config.hpp"
#include "vector_ops.hpp"

/*
 HLLC method for 2D Euler HD equation header file
 Bertalan Szuchovszky 26.02.2026
 
 Modified on 21.09.2026:
  - Ported state vectors to zero-allocation Kokkos::Array<double, 4> (Vec4).
  - Added KOKKOS_INLINE_FUNCTION decorators to facilitate execution on host and device backends.
  - Consolidated header includes for Modern CMake / Kokkos integration.

 state vector Q = [rho, rho*u, rho*v, rho*e_tot] 
 with e_tot = e_th + 0.5*|v|^2 & e_th = 1/(gamma-1) k_BT/mu m_p thermal energy

 We only need the hllc flux from the HLLC.cpp file
 FluxhllcX : numerical flux at an interface in x direction
 FluxhllcY : numerical flux at an interface in y direction

 Input:
  QL, QR: left and right state vectors at the interface!!! USE SLOPE LIMITERS TO GET THEM!
  gamma: (f+2)/f adiabatic index
 Returns:
  HLLC flux vectors in x direction or y direction


  For details read Toro - Riemann Solvers and Numerical Methods for Fluid Dynamics (3rd ed)
  I implemented chapter 10 using my matrix.h header for numpy like arrays.
  This code serves to calculate the hllc flux with given Q state vector and gamma adiabatic index.
  The Toro book uses SK instead of lambdas, but to stay consistent with the FSM lecture notation
  I will use lambda_{m}, lambda_{0} and lambda_{+} instead.

*/

//removed file-scope 'using namespace VecOps;' since it leaked into every file including this header.
//the Vec4 operators are global now (vector_ops.hpp), so nothing here needs it.

struct Primitive_Vals{
  double rho, u, v, p, cs;
};

KOKKOS_INLINE_FUNCTION
static Primitive_Vals QtoPrim(const Vec4& Q, double gamma){
  Primitive_Vals q;
  q.rho = Q[0];
  q.u = Q[1]/q.rho; //rho u / rho
  q.v = Q[2]/q.rho; //rho v / rho

  //Q[3] = rho e_tot = rho(e_th + 0.5(u^2 + v^2)), e_th = 1/(gamma-1)k_BT/mu m_p
  //p = rho k_BT/mu m_p ideal gas law -> p = (gamma-1)*e_th = (gamma-1)rho(e_tot-0.5(u^2+v^2))
  q.p = (gamma - 1.0)*(Q[3] - q.rho*0.5*(q.u*q.u + q.v*q.v));

  //FIX: check BEFORE the sqrt (was after -> cs already NaN), include rho == 0 (division above gives inf/NaN),
  //and no plain throw in device code: throw on host, Kokkos::abort on device
  if(!(q.rho > 0.0) || q.p < 0.0){
    KOKKOS_IF_ON_HOST((throw std::invalid_argument("Density or pressure is negative");))
    KOKKOS_IF_ON_DEVICE((Kokkos::abort("Density or pressure is negative");))
  }
  //Kokkos::sqrt is device-safe
  q.cs = Kokkos::sqrt(gamma*q.p/q.rho);

  return q;
}

//flux in the x direction - Toro chapter 10
KOKKOS_INLINE_FUNCTION
static Vec4 xFlux(const Vec4& Q, double gamma){
  Primitive_Vals q = QtoPrim(Q, gamma);
  double e_tot = Q[3]/q.rho;
  double h_tot = e_tot + q.p/q.rho;
  //Fx = (rho u, rho u^2 + p, rho u v, rho u h_tot)
  return Vec4{
    q.rho*q.u,
    q.rho*q.u*q.u + q.p,
    q.rho*q.v*q.u,
    q.rho*q.u*h_tot
  };
}

struct lambdas {
  double lm, l0, lp; //lambda_{-}, lambda_{0}, lambda_{+} eigenvals of the Jacobi matrix
};

KOKKOS_INLINE_FUNCTION
static lambdas WavecX(const Primitive_Vals& L, const Primitive_Vals& R){
  //Toro 10.48
  //Kokkos::min / Kokkos::max device-safe
  double SL = Kokkos::min(L.u - L.cs, R.u - R.cs); //lambda_{-} = u-cs but let it be the min of these
  double SR = Kokkos::max(L.u + L.cs, R.u + R.cs); //lambda_{+} = u+cs and the max of these

  //Toro eq. 10.36
  double Sstar = (R.p - L.p + L.rho*L.u*(SL - L.u) - R.rho*R.u*(SR - R.u))/(L.rho*(SL - L.u) - R.rho*(SR - R.u));
  return {SL, Sstar, SR};
}

KOKKOS_INLINE_FUNCTION
static Vec4 Qtilde(const Vec4& Q, double gamma, double SK, double Sstar){
  //Q is Q_K with K=L || K=R => q is also q_k
  Primitive_Vals q = QtoPrim(Q, gamma);
  double mul = q.rho*(SK - q.u)/(SK - Sstar);
  double E = Q[3];
  //Toro 10.39
  return Vec4{
    mul,
    mul*Sstar,
    mul*q.v,
    mul*(E/q.rho + (Sstar - q.u)*(Sstar + q.p/(q.rho*(SK - q.u))))
  };
}

KOKKOS_INLINE_FUNCTION
Vec4 FluxhllcX(const Vec4& QL, const Vec4& QR, double gamma){
  //left and right primitive vals
  Primitive_Vals L = QtoPrim(QL, gamma);
  Primitive_Vals R = QtoPrim(QR, gamma);

  lambdas w = WavecX(L, R);

  //Toro 10.26
  if(0.0 <= w.lm){
    return xFlux(QL, gamma);
  }

  if(w.lp <= 0.0){
    return xFlux(QR, gamma);
  }

  if(w.lm <= 0.0 && 0.0 <= w.l0){
    Vec4 QtL = Qtilde(QL, gamma, w.lm, w.l0); //Toro 10.27
    return xFlux(QL, gamma) + w.lm*(QtL - QL);
  }

  Vec4 QtR = Qtilde(QR, gamma, w.lp, w.l0); //Toro 10.29
  return xFlux(QR, gamma) + w.lp*(QtR - QR);
}

//trick to get the flux in y direction instead of writing the whole thing again
KOKKOS_INLINE_FUNCTION
static Vec4 rotateQxy(const Vec4& Q){
  //[rho, rho*u, rho*v, E]  ->  [rho, rho*v, rho*u, E]
  return Vec4{Q[0], Q[2], Q[1], Q[3]};
}

KOKKOS_INLINE_FUNCTION
Vec4 FluxhllcY(const Vec4& QL, const Vec4& QR, double gamma){
  //rotate so that y becomes the normal direction
  Vec4 QL_rot = rotateQxy(QL);
  Vec4 QR_rot = rotateQxy(QR);

  //solve as an x direction flux but now with rotated Q
  Vec4 Frot = FluxhllcX(QL_rot, QR_rot, gamma);

  //rotate flux back:  G = [rho*v, rho*u*v, rho*v^2+p, (E+p)*v]
  return Vec4{Frot[0], Frot[2], Frot[1], Frot[3]};
}


/*
 IMPLICIT PART (Q & U are the same, the Batten et al. 1997 article uses U instead of Q)
 Jacobians of the HLLC flux with frozen S_L, S_R and S_M differentiated (Batten, Leschziner, Goldberg 1997)
 Ported from the old HLLC.cpp (std::vector / Matrix(4,4)) to Vec4 / VecOps::Matrix with Kokkos decorators.
*/

//dF_hllc/dU_L and dF_hllc/dU_R at one interface
struct HLLCJac {
  VecOps::Matrix dF_dUl;
  VecOps::Matrix dF_dUr;
};

//Euler flux Jacobian in x, used for the fully supersonic cases in eq. (38)
KOKKOS_INLINE_FUNCTION
static VecOps::Matrix euler_jac_x(double u, double v, double E, double p, double rho, double gamma){
  double H  = (E + p)/rho; //specific enthalpy
  double c2 = u*u + v*v;
  VecOps::Matrix A;
  //d(rho u)/dU
  A(0,1) = 1.0;
  //d(rho u^2+p)/dU
  A(1,0) = (gamma-3.0)/2.0*u*u + (gamma-1.0)/2.0*v*v;
  A(1,1) = (3.0-gamma)*u;
  A(1,2) = -(gamma-1.0)*v;
  A(1,3) = gamma-1.0;
  //d(rho uv)/dU
  A(2,0) = -u*v;   A(2,1) = v;   A(2,2) = u;
  //d((E+p)u)/dU
  A(3,0) = u*((gamma-1.0)*c2/2.0 - H);
  A(3,1) = H - (gamma-1.0)*u*u;
  A(3,2) = -(gamma-1.0)*u*v;
  A(3,3) = gamma*u;
  return A;
}

//star-region Jacobian, eqs. (40)-(54). K is the side whose star state is used (L if S_M >= 0, R otherwise),
//"loc" derivatives are w.r.t. U_K and "far" ones w.r.t. the other side (paper note after eq. 54: L<->R swap).
KOKKOS_INLINE_FUNCTION
static void hllc_star_jacobian(const Vec4& QK, const Primitive_Vals& K, double SK, double SM, double pstar,
                               const Vec4& dSM_loc, const Vec4& dSM_far,
                               const Vec4& dp_loc,  const Vec4& dp_far, double gamma,
                               VecOps::Matrix& J_loc, VecOps::Matrix& J_far){
  double V = 1.0/(SK - SM); //eq. (9)

  //*-state components eq. (7), x direction n_x=1, n_y=0
  double rhos  = K.rho*(SK - K.u)*V;
  double rhous = V*((SK - K.u)*QK[1] + (pstar - K.p));
  double rhovs = V*(SK - K.u)*QK[2];
  double Es    = V*((SK - K.u)*QK[3] - K.p*K.u + pstar*SM);
  double EpP   = Es + pstar;

  double c2 = K.u*K.u + K.v*K.v; //|v_K|^2
  double h  = (QK[3] + K.p)/K.rho;

  //d(rho*)/dU eq. (46)
  Vec4 drho_loc  = {V*SK, -V, 0.0, 0.0};
  //d(rhou*)/dU eq. (47), n_x=1
  Vec4 drhou_loc = {V*(K.u*K.u - 0.5*(gamma-1.0)*c2), V*(SK + (gamma-3.0)*K.u), V*((gamma-1.0)*K.v), V*(-(gamma-1.0))};
  //d(rhov*)/dU eq. (48), n_y=0 -> no pressure term
  Vec4 drhov_loc = {V*K.u*K.v, -V*K.v, V*(SK - K.u), 0.0};
  //d(E*)/dU eq. (54)
  Vec4 dE_loc    = {V*(K.u*h - 0.5*(gamma-1.0)*K.u*c2), V*(-h + (gamma-1.0)*K.u*K.u), V*((gamma-1.0)*K.u*K.v), V*(SK - gamma*K.u)};
  Vec4 drho_far, drhou_far, drhov_far, dE_far;

  for(int k = 0; k < 4; k++){
    drho_loc[k]  += V*rhos*dSM_loc[k];
    drho_far[k]   = V*rhos*dSM_far[k];
    drhou_loc[k] += V*(dp_loc[k] + rhous*dSM_loc[k]);
    drhou_far[k]  = V*(dp_far[k] + rhous*dSM_far[k]);
    drhov_loc[k] += V*rhovs*dSM_loc[k];
    drhov_far[k]  = V*rhovs*dSM_far[k];
    dE_loc[k]    += V*(SM*dp_loc[k] + EpP*dSM_loc[k]);
    dE_far[k]     = V*(SM*dp_far[k] + EpP*dSM_far[k]);
  }

  //assemble eqs. (40-41): F* = (S_M rho*, S_M rhou* + p*, S_M rhov*, S_M (E* + p*))
  for(int k = 0; k < 4; k++){
    J_loc(0,k) = SM*drho_loc[k]  + rhos*dSM_loc[k];
    J_far(0,k) = SM*drho_far[k]  + rhos*dSM_far[k];
    J_loc(1,k) = SM*drhou_loc[k] + rhous*dSM_loc[k] + dp_loc[k];
    J_far(1,k) = SM*drhou_far[k] + rhous*dSM_far[k] + dp_far[k];
    J_loc(2,k) = SM*drhov_loc[k] + rhovs*dSM_loc[k];
    J_far(2,k) = SM*drhov_far[k] + rhovs*dSM_far[k];
    J_loc(3,k) = SM*(dE_loc[k] + dp_loc[k]) + EpP*dSM_loc[k];
    J_far(3,k) = SM*(dE_far[k] + dp_far[k]) + EpP*dSM_far[k];
  }
}

KOKKOS_INLINE_FUNCTION
HLLCJac hllc_jacobian_x(const Vec4& QL, const Vec4& QR, double gamma){
  Primitive_Vals L = QtoPrim(QL, gamma);
  Primitive_Vals R = QtoPrim(QR, gamma);

  lambdas w = WavecX(L, R);
  double SL = w.lm, SM = w.l0, SR = w.lp;

  HLLCJac J; //zero-initialised 4x4 blocks

  //supersonic cases reduce to the plain Euler flux Jacobian, eq. (38)
  if(SL > 0.0){ //fully supersonic rightward, F = F_L
    J.dF_dUl = euler_jac_x(L.u, L.v, QL[3], L.p, L.rho, gamma);
    return J;
  }
  if(SR < 0.0){ //fully supersonic leftward, F = F_R
    J.dF_dUr = euler_jac_x(R.u, R.v, QR[3], R.p, R.rho, gamma);
    return J;
  }

  //p* eq. (11) & (37), pressure in the star region
  double pstar = L.rho*(L.u - SL)*(L.u - SM) + L.p;

  //dS_M/dU_L and dS_M/dU_R, eqs. (A),(B); r-tilde denominator eq. (12)
  double inv_r = 1.0/(R.rho*(SR - R.u) - L.rho*(SL - L.u));
  double cL = L.u*L.u + L.v*L.v;
  double cR = R.u*R.u + R.v*R.v;
  Vec4 dSM_dUL = {
    inv_r*(-L.u*L.u + 0.5*(gamma-1.0)*cL + SM*SL), //d/d(rho)
    inv_r*((3.0-gamma)*L.u - SL - SM),             //d/d(rho*u)
    inv_r*((1.0-gamma)*L.v),                       //d/d(rho*v)
    inv_r*(gamma-1.0)                              //d/d(E)
  };
  Vec4 dSM_dUR = {
    inv_r*(R.u*R.u - 0.5*(gamma-1.0)*cR - SM*SR),
    inv_r*(SR + SM - (3.0-gamma)*R.u),
    inv_r*((gamma-1.0)*R.v),
    inv_r*(1.0-gamma)
  };

  //dp*/dU eq. (45)
  Vec4 dp_dUL = (R.rho*(SR - R.u))*dSM_dUL;
  Vec4 dp_dUR = (L.rho*(SL - L.u))*dSM_dUR;

  if(SM >= 0.0){ //left star region
    hllc_star_jacobian(QL, L, SL, SM, pstar, dSM_dUL, dSM_dUR, dp_dUL, dp_dUR, gamma, J.dF_dUl, J.dF_dUr);
  }else{         //right star region S_M < 0 <= S_R
    hllc_star_jacobian(QR, R, SR, SM, pstar, dSM_dUR, dSM_dUL, dp_dUR, dp_dUL, gamma, J.dF_dUr, J.dF_dUl);
  }
  return J;
}

//y direction: rotate (rho, rho v, rho u, E), solve as x, permute rows and columns back
KOKKOS_INLINE_FUNCTION
HLLCJac hllc_jacobian_y(const Vec4& QL, const Vec4& QR, double gamma){
  HLLCJac Jr = hllc_jacobian_x(rotateQxy(QL), rotateQxy(QR), gamma);
  HLLCJac J;
  const int p[4] = {0, 2, 1, 3};
  for(int i = 0; i < 4; i++){
    for(int k = 0; k < 4; k++){
      J.dF_dUl(p[i], p[k]) = Jr.dF_dUl(i, k);
      J.dF_dUr(p[i], p[k]) = Jr.dF_dUr(i, k);
    }
  }
  return J;
}
