# Vorticity-Vector-Potential-JAX

A GPU-accelerated Navier-Stokes solver for 3D incompressible flows using the vorticity-vector potential formulation, implemented in JAX.
## Overview
This solver implements the vorticity-vector potential (ψ-ω) formulation for incompressible flows, eliminating the pressure term and satisfying the divergence-free condition automatically. The code is using JAX's JIT compilation capability for decreasing theexecution time.
Key Features:

## Vorticity-vector potential formulation for 3D incompressible Navier-Stokes equations
GPU-accelerated computations with JAX
JIT compilation for optimized performance
Finite difference discretization (2nd-order spatial, 1st-order temporal)
Uniform collocated grid structure
Supported geometries: 3D lid-driven cavity and square duct flow

## Mathematical Formulation
Solve the Vorticity Transport Equation (VTE) in the non-conservative form:

$\frac{ \partial\vec{\omega}}{\partial t} + (\vec{\omega} \cdot \nabla)\vec{u} = \frac{1}{Re} \nabla^2\vec{\omega} + (\vec{u} \cdot \nabla)\vec{\omega}$

This requires first solving the Poisson equation for the vector potential:
$\nabla^2\vec{\psi} = - \vec{\omega}$

and after that computing and updating the vorticity by solving the VTE.

## Discretization:

Time derivative: 1st-order Euler scheme
Spatial derivatives: 2nd-order central differences
Grid: Uniform collocated mesh

Installation
Requirements

Python 3.8+
JAX with GPU support
NumPy
Matplotlib (for visualization)
