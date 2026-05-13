# Sedimentation simulation

The goal of this project is to simulate the sedimentation process of a particle in a protoplanetary disk.
The simulation has 2 main parts:
  * Well-balanced HLLC that handles the interacting gas background which is initialized in equilibrium
  * Leapfrog algorithm which integrates the equations of motion (and other variables) of the single solid particle
\

The 2 methods are connected via Cloud In Cell (CIC) method; the CIC interpolates the position of the particle onto the grid where the HLLC works, 
then using the same weights for positions the CIC recovers the amounts of gas that are (=cells) in contact with the particle to calculate the
back-reaction of the gas onto the particle (implicit Epstein drag).
\

**TO DO**:
  * ask for variables needed for heat diffusion term in io.h
  * add temperature diffusion timescale function
  * test temperature diffusion
  * IMPLICIT (ASAP): Average-State Jacobians and Implicit Methods for Compressible Viscous and Turbulent Flows - Batten, Leschziner, Goldberg
  * GPU?
  * Quality README
