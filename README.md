#Vorticity-Vector-Potential-JAX

A GPU-accelerated Navier-Stokes solver for 3D incompressible flows using the vorticity-vector potential formulation, implemented in JAX.
##Overview
This solver implements the vorticity-vector potential (ψ-ω) formulation for incompressible flows, eliminating the pressure term and satisfying the divergence-free condition automatically. The code is optimized for GPU execution using JAX's JIT compilation and automatic differentiation capabilities.
Key Features:

##Vorticity-vector potential formulation for 3D incompressible Navier-Stokes equations
GPU-accelerated computations with JAX
JIT compilation for optimized performance
Finite difference discretization (2nd-order spatial, 1st-order temporal)
Uniform collocated grid structure
Supported geometries: 3D lid-driven cavity and square duct flow

##Mathematical Formulation
