#ifndef HLLC_H
#define HLLC_H

#include "matrix.h"
#include <iostream>

/*
 HLLC method for 2D Euler HD equation header file
 Bertalan Szuchovszky 26.02.2026
 
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
*/

struct Primitive_Vals{
  double rho, u, v, p, cs;
};

static Primitive_Vals QtoPrim(const Vector& Q, double gamma){
  if (Q.size() != 4) throw std::invalid_argument("State vector must be 4D");

  Primitive_Vals q;
  q.rho = Q[0];
  q.u = Q[1]/q.rho; //rho u / rho
  q.v = Q[2]/q.rho; //rho v / rho
  //Q[3] = rho e_tot = rho(e_th + 0.5(u^2 + v^2)), e_th = 1/(gamma-1)k_BT/mu m_p
  //p = rho k_BT/mu m_p ideal gas law -> p = (gamma-1)*e_th = (gamma-1)rho(e_tot-0.5(u^2+v^2)) 
  q.p = (gamma-1.0)*(Q[3]-q.rho * 0.5*(q.u*q.u + q.v*q.v));
  q.cs = std::sqrt(gamma * q.p/q.rho);

  if (q.rho <0.0||q.p<0.0){
    std::cout << "QtoPrim failed:\n"
              << "  Q = [" << Q[0] << ", " << Q[1] << ", "
                           << Q[2] << ", " << Q[3] << "]\n"
              << "  rho=" << q.rho << "  p=" << q.p << "\n";
    throw std::invalid_argument("Density or pressure is negative");
  }
  return q;
}


//flux in the x direction - Toro chapter 10
static Vector q1Flux(const Vector&Q, double gamma){
  Primitive_Vals q = QtoPrim(Q, gamma);
  double e_tot = Q[3]/q.rho;
  double h_tot = e_tot + q.p/q.rho;
  //Fx = (rho u, rho u^2 + p, rho u v, rho u h_tot)
  return{
    q.rho*q.u,
    q.rho*q.u*q.u + q.p,
    q.rho*q.v*q.u,
    q.rho*q.u*h_tot,
  };
}

//flux in the y direction
static Vector q2Flux(const Vector&Q, double gamma){
  Primitive_Vals q = QtoPrim(Q, gamma);
  double e_tot = Q[3]/q.rho;
  double h_tot = e_tot + q.p/q.rho;
  //Fy = (rho u, rho u v, rho v^2 + p, rho v h_tot)
  return{
    q.rho*q.u,
    q.rho*q.u*q.v,
    q.rho*q.v*q.v + q.p,
    q.rho*q.v*h_tot,
  };
}


struct lambdas{
  double lm, l0, lp; //lambda_{-}, lambda_{0}, lambda_{+} eigenvals of the Jacobi matrix
};


static lambdas Wavec_q1(const Primitive_Vals& L, const Primitive_Vals& R){ //L:left, R:right
  //Toro 10.48
  double SL = std::min(L.u-L.cs, R.u-R.cs); //lambda_{-} = u-cs but let it be the min of these
  double SR = std::max(L.u+L.cs, R.u+R.cs); //lambda_{+} = u+cs and the max of these 

  double Sstar = (R.p-L.p + L.rho*L.u*(SL-L.u) - R.rho*R.u*(SR-R.u))/(L.rho*(SL-L.u)-R.rho*(SR-R.u)); //Toro eq. 10.36
  return {SL, Sstar, SR};
}


static Vector Qtilde(const Vector&Q, double gamma, double SK, double Sstar){
  //Q is Q_K with K=L || K=R => q is also q_K
  Primitive_Vals q = QtoPrim(Q, gamma);
  double mul = q.rho*(SK-q.u)/(SK-Sstar);
  double E = Q[3];
  //Toro 10.39
  return{
    mul,
    mul*Sstar,
    mul*q.v,
    mul*(E/q.rho + (Sstar-q.u)*(Sstar + q.p/(q.rho*(SK-q.u)))),
  };
}

Vector q1Flux(const Vector& Q, double gamma);
Vector q2Flux(const Vector& Q, double gamma);
Vector Fluxhllc_q1(const Vector& QL, const Vector& QR, double gamma);
Vector Fluxhllc_q2(const Vector& QL, const Vector& QR, double gamma);
#endif // HLLC_H
