#ifndef SLOPELIM_H
#define SLOPELIM_H

#include "matrix.h"

/*
 Superbee slope limiter header file
 Bertalan Szuchovszky 26.06.2026

 state vector Q = [rho, rho*u, rho*v, rho*e_tot] 
 with e_tot = e_th + 0.5*|v|^2 & e_th = 1/(gamma-1) k_BT/mu m_p thermal energy

 Func to import: sigma_superbee

 Input:
  Qim1, Qi, Qip1: Q{i-1}, Q_{i}; Q_{i+1} state vectors 
  dk: either dx or dy depending on direction
 Output:
  sigma_k superbee slope
*/

double superbee_one(double qim1, double qi, double qip1, double dx);
Vector sigma_superbee(const Vector& Qim1, const Vector& Qi, const Vector& Qip1, double dk);
Vector sigma_van_leer(const Vector& Qim1, const Vector& Qi, const Vector& Qip1, double dk);
Vector sigma_minmod(const Vector& Qim1, const Vector& Qi, const Vector& Qip1, double dk);
double minmod_one(double qim1, double qi, double qip1, double dk);
#endif // !SLOPELIM_H
