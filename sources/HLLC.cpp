#include <cmath>
#include "../headers/matrix.h"
#include "../headers/HLLC.h"


/*
  HLLC method for 2D Euler HD equation
  Bertalan Szuchovszky 26.02.2026

  For details read Toro - Riemann Solvers and Numerical Methods for Fluid Dynamics (3rd ed)
  I implemented chapter 10 using my matrix.h header for numpy like arrays.
  This code serves to calculate the hllc flux with given Q state vector and gamma adiabatic index.
  The Toro book uses SK instead of lambdas, but to stay consistent with the FSM lecture notation
  I will use lambda_{m}, lambda_{0} and lambda_{+} instead.

  MODIFIED 24.04.2026:
  Added well balanced HLLC flux 
  - 02.05.2026 I opted not to use it as I have a better implementation in main following Käppeli et al. 2016
*/

using namespace VecOps; //vector operation namespace for std::vector<double> in matrix.h


Vector Fluxhllc_q1(const Vector& QL, const Vector& QR, double gamma){
  //left and right primitive vals
  Primitive_Vals L = QtoPrim(QL, gamma);
  Primitive_Vals R = QtoPrim(QR, gamma);

  lambdas w = Wavec_q1(L, R);

  //Toro 10.26
  if (0.0<=w.lm){
    return q1Flux(QL, gamma);
  }
  
  if (w.lp<=0.0){
    return q1Flux(QR, gamma);
  }

  if (w.lm<=0.0 && 0.0<=w.l0){
    Vector QtL = Qtilde(QL, gamma, w.lm, w.l0); //Toro 10.27
    return q1Flux(QL, gamma) + w.lm*(QtL-QL);
  }

  Vector QtR = Qtilde(QR, gamma, w.lp, w.l0);
  return q1Flux(QR, gamma) + w.lp*(QtR-QR); //Toro 10.29
}

// Vector FluxhllcX_wellbalanced(const Vector& QL, const Vector& QR, double gamma, 
//                               double Omega, double x_L, double x_R) {
//   //standard HLLC flux
//   Vector F = FluxhllcX(QL, QR, gamma);
//   
//   Primitive_Vals L = QtoPrim(QL, gamma);
//   Primitive_Vals R = QtoPrim(QR, gamma);
//   
//   //avg density at interface
//   double rho_avg = 0.5 * (L.rho + R.rho);
//   
//   //gravitational accel @ cell centers
//   double g_L = Omega * Omega * x_L;
//   double g_R = Omega * Omega * x_R;
//   double g_avg = 0.5 * (g_L + g_R);
//   
//   //dist between cell centers
//   double dx = x_R - x_L;
//   
//   //well-balanced correction to momentum flux (F[1])
//   //this term: -int_xL^xR rho(x)*g(x) dx ≈ -rho_avg * g_avg * dx
//   F[1] -= rho_avg * g_avg * dx;
//   
//   //well-balanced correction to energy flux (F[3])
//   //the work done by gravity: W = -u*rho*g*dx
//   //with avg velocity
//   double u_avg = 0.5 * (L.u + R.u);
//   F[3] -= u_avg * rho_avg * g_avg * dx;
//   
//   return F;
// }


//trick to get the flux in y direction instead of writing the whole thing again
static Vector rotateQq1q2(const Vector& Q) {
  //[rho, rho*u, rho*v, E]  ->  [rho, rho*v, rho*u, E]
  return {Q[0], Q[2], Q[1], Q[3]};
}

Vector Fluxhllc_q2(const Vector& QL, const Vector& QR, double gamma) {
  //rotate so that y becomes the normal direction
  Vector QL_rot = rotateQq1q2(QL);
  Vector QR_rot = rotateQq1q2(QR);

  //solve as an x direction flux but now with rotated Q
  Vector Frot = Fluxhllc_q1(QL_rot, QR_rot, gamma);

  //rotate flux back:  G = [rho*v, rho*u*v, rho*v^2+p, (E+p)*v]
  return {Frot[0], Frot[2], Frot[1], Frot[3]};
}

// Vector FluxhllcY_wellbalanced(const Vector& QL, const Vector& QR, double gamma, 
//                               double Omega, double y_L, double y_R) {
//   //standard HLLC flux in y direction
//   Vector F = FluxhllcY(QL, QR, gamma);
//   
//   Primitive_Vals L = QtoPrim(QL, gamma);
//   Primitive_Vals R = QtoPrim(QR, gamma);
//   
//   //avg density at interface
//   double rho_avg = 0.5 * (L.rho + R.rho);
//   
//   //gravitational accel @ cell centers
//   double g_L = Omega * Omega * y_L;
//   double g_R = Omega * Omega * y_R;
//   double g_avg = 0.5 * (g_L + g_R);
//   
//   //dist between cell centers
//   double dy = y_R - y_L;
//   
//   //well-balanced correction to momentum flux (F[1])
//   //this term: -int_xL^xR rho(x)*g(x) dx ≈ -rho_avg * g_avg * dx
//   F[1] -= rho_avg * g_avg * dy;
//   
//   //well-balanced correction to energy flux (F[3])
//   //the work done by gravity: W = -u*rho*g*dx
//   //with avg velocity
//   double v_avg = 0.5 * (L.v + R.v);
//   F[3] -= v_avg * rho_avg * g_avg * dy;
//   
//   return F;
// }

