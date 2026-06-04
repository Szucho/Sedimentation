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

  MODIFIED 28.05.2026
  - added implicit implementation
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


//IMPLICIT IMPLEMENTATION (Q & U are the same, the Batten et al. 1997 article uses U instead of Q)
//Frozen S_L, S_R + avg S_M ~ Batten, Leschziner, Goldberg 1997

//Euler flux Jacobi used for the fully supersonic cases in eq. (38)
static Matrix euler_jac_q1(double u, double v, double E, double p, double rho, double gamma) {
  double H  = (E + p)/rho; //specific enthalpy
  double c2 = u*u + v*v;
  Matrix A(4, 4, 0.0);
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


HLLCJac hllc_jacobian_q1(const Vector& QL, const Vector& QR, double gamma) {

  
  Primitive_Vals primL = QtoPrim(QL, gamma);
  Primitive_Vals primR = QtoPrim(QR, gamma);
  
  double rhoL = primL.rho;
  double uL   = primL.u;
  double vL   = primL.v;
  double pL   = primL.p;
  double EL   = QL[3];
  
  double rhoR = primR.rho;
  double uR   = primR.u;
  double vR   = primR.v;
  double pR   = primR.p;
  double ER   = QR[3];

  lambdas w = Wavec_q1(primL, primR);
  double SL = w.lm, SM = w.l0, SR = w.lp;

  HLLCJac J;   //zero-initialised 4x4 blocks

  //supersonic case of Jacobian reduces to plain Euler flux Jacobian
  //eq. (38): if S_L > 0, F* = F_L, so dF*/dU_L = dF_L/dU_L, dF*/dU_R = 0
  //so here fully supersonic rightward — F* = F_L
  if (SL > 0.0) {
    J.dF_dUl = euler_jac_q1(uL, vL, EL, pL, rhoL, gamma);
    //J.dF_dUr stays zero
    return J;
  }
  //and fully supersonic leftward — F* = F_R
  if (SR < 0.0) {
    //J.dF_dUl stays zero
    J.dF_dUr = euler_jac_q1(uR, vR, ER, pR, rhoR, gamma);
    return J;
  }

  //shared quantities
  //r-tilde denominator for dS_M/dU eq. (12)
  double rtilde = rhoR*(SR - uR) - rhoL*(SL - uL);
  double VL     = 1.0/(SL - SM);   //eq. (9)

  //p* eq. (11) & eq. (37) pressure in the star region
  double pstar = rhoL*(uL - SL)*(uL - SM) + pL;

  //*-state components U*_L eq. (7), q1 direction n_x=1, n_y=0
  double rhostarL  = rhoL*(SL - uL)*VL;
  double rhoustarL = VL*((SL - uL)*QL[1] + (pstar - pL));
  double rhovstarL = VL*(SL - uL)*QL[2];
  double EstarL    = VL*((SL - uL)*EL - pL*uL + pstar*SM);

  //dS_M/dU_L  eq. A in paper, q1: n_x=1, n_y=0, q_L = u_L
  double inv_r = 1.0/rtilde;
  double cL    = uL*uL + vL*vL; //|v_L|^2
  Vector dSM_dUL = {
    inv_r*(-uL*uL + 0.5*(gamma-1.0)*cL + SM*SL),//d/d(rho)
    inv_r*((3.0-gamma)*uL - SL - SM), //d/d(rho*u)
    inv_r*((1.0-gamma)*vL), //d/d(rho*v)
    inv_r*(gamma-1.0) //d/d(E)
  };

  //dS_M/dU_R  eq. B in paper, q1: n_x=1, n_y=0, q_R = u_R
  double cR = uR*uR + vR*vR;
  Vector dSM_dUR = {
    inv_r*(uR*uR - 0.5*(gamma-1.0)*cR - SM*SR),
    inv_r*(SR + SM - (3.0-gamma)*uR),
    inv_r*((gamma-1.0)*vR),
    inv_r*(1.0-gamma)
  };

  //dp* / dU  (eq. 45)
  //dp*/dU_L = rho_R*(S_R - u_R) * dS_M/dU_L
  //dp*/dU_R = rho_L*(S_L - u_L) * dS_M/dU_R
  double factorL = rhoR*(SR - uR);
  double factorR = rhoL*(SL - uL);
  Vector dpstar_dUL(4), dpstar_dUR(4);
  for (int k = 0; k < 4; k++) {
    dpstar_dUL[k] = factorL * dSM_dUL[k];
    dpstar_dUR[k] = factorR * dSM_dUR[k];
  }

  if (SM >= 0.0) {
    double VL = 1.0/(SL - SM); // eq. (9)

    double rhostarL  = rhoL*(SL - uL)*VL; // eq. (7)
    double rhoustarL = VL*((SL - uL)*QL[1] + (pstar - pL));
    double rhovstarL = VL*(SL - uL)*QL[2];
    double EstarL    = VL*((SL - uL)*EL - pL*uL + pstar*SM);
    double EpPL      = EstarL + pstar;

    //d(rho*_L)/dU  eq. (46)
    Vector drho_L = { VL*SL, -VL, 0.0, 0.0 };
    Vector drho_R(4, 0.0);
    for (int k = 0; k < 4; k++) {
      drho_L[k] += VL*rhostarL*dSM_dUL[k];
      drho_R[k]  = VL*rhostarL*dSM_dUR[k];
    }
    //d(rhou*_L)/dU  eq. (47), n_x=1
    Vector drhou_L = {
      VL*(uL*uL - 0.5*(gamma-1.0)*cL),
      VL*(SL + (gamma-3.0)*uL),
      VL*((gamma-1.0)*vL),
      VL*(-(gamma-1.0))
    };
    Vector drhou_R(4, 0.0);
    for (int k = 0; k < 4; k++) {
      drhou_L[k] += VL*(dpstar_dUL[k] + rhoustarL*dSM_dUL[k]);
      drhou_R[k]  = VL*(dpstar_dUR[k] + rhoustarL*dSM_dUR[k]);
    }
    //d(rhov*_L)/dU  eq. (48), n_y=0, no pressure term
    Vector drhov_L = { VL*uL*vL, -VL*vL, VL*(SL - uL), 0.0 };
    Vector drhov_R(4, 0.0);
    for (int k = 0; k < 4; k++) {
      drhov_L[k] += VL*rhovstarL*dSM_dUL[k];
      drhov_R[k]  = VL*rhovstarL*dSM_dUR[k];
    }
    //d(E*_L)/dU  eq. (54), n_x=1, n_y=0
    double hL = (EL + pL)/rhoL;
    Vector dE_L = {
      VL*(uL*hL - 0.5*(gamma-1.0)*uL*cL),
      VL*(-hL   + (gamma-1.0)*uL*uL),
      VL*((gamma-1.0)*uL*vL),
      VL*(SL - gamma*uL)
    };
    Vector dE_R(4, 0.0);
    for (int k = 0; k < 4; k++) {
      dE_L[k] += VL*(SM*dpstar_dUL[k] + EpPL*dSM_dUL[k]);
      dE_R[k]  = VL*(SM*dpstar_dUR[k] + EpPL*dSM_dUR[k]);
    }
    //assemble  eqs. (40-41)
    for (int k = 0; k < 4; k++) {
      J.dF_dUl(0,k) = SM*drho_L[k]  + rhostarL *dSM_dUL[k];
      J.dF_dUr(0,k) = SM*drho_R[k]  + rhostarL *dSM_dUR[k];
      J.dF_dUl(1,k) = SM*drhou_L[k] + rhoustarL*dSM_dUL[k] + dpstar_dUL[k];
      J.dF_dUr(1,k) = SM*drhou_R[k] + rhoustarL*dSM_dUR[k] + dpstar_dUR[k];
      J.dF_dUl(2,k) = SM*drhov_L[k] + rhovstarL*dSM_dUL[k];
      J.dF_dUr(2,k) = SM*drhov_R[k] + rhovstarL*dSM_dUR[k];
      J.dF_dUl(3,k) = SM*(dE_L[k] + dpstar_dUL[k]) + EpPL*dSM_dUL[k];
      J.dF_dUr(3,k) = SM*(dE_R[k] + dpstar_dUR[k]) + EpPL*dSM_dUR[k];
    }
    return J;
  }

  //eq. (38) right star region  S_M < 0 <= S_R
  //eqs. (46)-(54) with subscripts l<->r and L<->R exchanged (paper note after eq. 54)
  //"local" derivative is now dSM_dUR; "far" is dSM_dUL
  {
    double VR = 1.0/(SR - SM);

    double rhostarR  = rhoR*(SR - uR)*VR;
    double rhoustarR = VR*((SR - uR)*QR[1] + (pstar - pR));
    double rhovstarR = VR*(SR - uR)*QR[2];
    double EstarR    = VR*((SR - uR)*ER - pR*uR + pstar*SM);
    double EpPR      = EstarR + pstar;

    //d(rho*_R)/dU  eq. (46), L<->R
    Vector drho_R = { VR*SR, -VR, 0.0, 0.0 };
    Vector drho_L(4, 0.0);
    for (int k = 0; k < 4; k++) {
      drho_R[k] += VR*rhostarR*dSM_dUR[k];
      drho_L[k]  = VR*rhostarR*dSM_dUL[k];
    }
    //d(rhou*_R)/dU  eq. (47), L<->R, n_x=1
    Vector drhou_R = {
      VR*(uR*uR - 0.5*(gamma-1.0)*cR),
      VR*(SR + (gamma-3.0)*uR),
      VR*((gamma-1.0)*vR),
      VR*(-(gamma-1.0))
    };
    Vector drhou_L(4, 0.0);
    for (int k = 0; k < 4; k++) {
      drhou_R[k] += VR*(dpstar_dUR[k] + rhoustarR*dSM_dUR[k]);
      drhou_L[k]  = VR*(dpstar_dUL[k] + rhoustarR*dSM_dUL[k]);
    }
    //d(rhov*_R)/dU  eq. (48), L<->R, n_y=0
    Vector drhov_R = { VR*uR*vR, -VR*vR, VR*(SR - uR), 0.0 };
    Vector drhov_L(4, 0.0);
    for (int k = 0; k < 4; k++) {
      drhov_R[k] += VR*rhovstarR*dSM_dUR[k];
      drhov_L[k]  = VR*rhovstarR*dSM_dUL[k];
    }
    //d(E*_R)/dU  eq. (54), L<->R, n_x=1, n_y=0
    double hR = (ER + pR)/rhoR;
    Vector dE_R = {
      VR*(uR*hR - 0.5*(gamma-1.0)*uR*cR),
      VR*(-hR   + (gamma-1.0)*uR*uR),
      VR*((gamma-1.0)*uR*vR),
      VR*(SR - gamma*uR)
    };
    Vector dE_L(4, 0.0);
    for (int k = 0; k < 4; k++) {
      dE_R[k] += VR*(SM*dpstar_dUR[k] + EpPR*dSM_dUR[k]);
      dE_L[k]  = VR*(SM*dpstar_dUL[k] + EpPR*dSM_dUL[k]);
    }
    //assemble  eqs. (40-41), L<->R
    for (int k = 0; k < 4; k++) {
      J.dF_dUl(0,k) = SM*drho_L[k]  + rhostarR*dSM_dUL[k];
      J.dF_dUr(0,k) = SM*drho_R[k]  + rhostarR*dSM_dUR[k];
      J.dF_dUl(1,k) = SM*drhou_L[k] + rhoustarR*dSM_dUL[k] + dpstar_dUL[k];
      J.dF_dUr(1,k) = SM*drhou_R[k] + rhoustarR*dSM_dUR[k] + dpstar_dUR[k];
      J.dF_dUl(2,k) = SM*drhov_L[k] + rhovstarR*dSM_dUL[k];
      J.dF_dUr(2,k) = SM*drhov_R[k] + rhovstarR*dSM_dUR[k];
      J.dF_dUl(3,k) = SM*(dE_L[k] + dpstar_dUL[k]) + EpPR*dSM_dUL[k];
      J.dF_dUr(3,k) = SM*(dE_R[k] + dpstar_dUR[k]) + EpPR*dSM_dUR[k];
    }
    return J;
  }
}


HLLCJac hllc_jacobian_q2(const Vector& QL, const Vector& QR, double gamma) {
  auto rot = [](const Vector& Q) -> Vector { return {Q[0], Q[2], Q[1], Q[3]}; };
  HLLCJac Jr = hllc_jacobian_q1(rot(QL), rot(QR), gamma);
  HLLCJac J;
  int p[4] = {0,2,1,3};
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 4; k++) {
      J.dF_dUl(p[i], p[k]) = Jr.dF_dUl(i,k);
      J.dF_dUr(p[i], p[k]) = Jr.dF_dUr(i,k);
    }
  return J;
}



//article uses Block-Thomas algorithm
void block_thomas(std::vector<Matrix>& A, std::vector<Matrix>& D, std::vector<Matrix>& C,
                  std::vector<Vector>& rhs, size_t N, std::vector<Matrix>& LU_work, 
                  std::vector<std::vector<int>>& piv_work){

  std::vector<Matrix> LU(N, Matrix(4,4,0.0));
  std::vector<std::vector<int>> piv(N, std::vector<int>(4,0));

  //LU-factor first diagonal block
  LU[0] = D[0];
  piv[0] = mat_lu_factor(LU[0]);

  //forward elimination
  for (size_t i = 1; i < N; i++) {
    //solve LU[i-1] * W_C = C[i-1]  column by column → W_C = D[i-1]^{-1} C[i-1}
    Matrix W_C(4, 4, 0.0);
    for (int k = 0; k < 4; k++) {
      Vector col = C[i-1].col(k);
      W_C.setCol(k, mat_lu_solve(LU[i-1], piv[i-1], col));
    }
    //solve LU[i-1] * W_b = rhs[i-1]
    Vector W_b = mat_lu_solve(LU[i-1], piv[i-1], rhs[i-1]);

    //update row i
    D[i]   =D[i]   - mat_mul(A[i], W_C);   //D[i] = D[i] - A[i] * D[i-1]^{-1} * C[i-1]
    rhs[i] =rhs[i] - mat_vec_mul(A[i], W_b);

    LU[i] = D[i];
    piv[i] = mat_lu_factor(LU[i]);
  }

  //back substitution
  rhs[N-1] = mat_lu_solve(LU[N-1], piv[N-1], rhs[N-1]);
  for (int i = (int)N-2; i >= 0; i--) {
    Vector tmp = rhs[i] - mat_vec_mul(C[i], rhs[i+1]);
    rhs[i] = mat_lu_solve(LU[i], piv[i], tmp);
  }
}
