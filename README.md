# Vorticity-Vector-Potential-JAX

A GPU-accelerated Navier–Stokes solver for 3D incompressible flows using the
vorticity–vector potential formulation, implemented in **JAX**. 

---

## Overview

This solver implements the **vorticity–vector potential (ψ–ω) formulation**
for incompressible Navier–Stokes equations. The formulation eliminates the
pressure term and automatically enforces the divergence-free constraint on
the velocity field.

The code leverages **JAX** for:
- Just-In-Time (JIT) compilation
- GPU acceleration

More information regarding the vorticity - vector potential algorithm implemented in this repository can be found in: [Thesis PDF](https://github.com/Trolias/Vorticity-Vector-Potential-JAX/releases/download/final/Thesis.pdf)

---

## Key Features

- Vorticity–vector potential formulation for 3D incompressible flows
- GPU-accelerated computation using JAX
- JIT compilation for optimized performance
- Finite-difference discretization
- Uniform collocated grid
- Supported test cases:
  - 3D lid-driven cavity flow
  - Square duct flow

---

## Mathematical Formulation

The incompressible vorticity transport equation (VTE) is solved in
non-conservative form:

$\frac{\partial \boldsymbol{\omega}}{\partial t}+ (\mathbf{u} \cdot \nabla)\boldsymbol{\omega}= (\boldsymbol{\omega} \cdot \nabla)\mathbf{u}+ \frac{1}{Re}\nabla^2 \boldsymbol{\omega}$

The velocity field is recovered from the vector potential by solving the
Poisson equation:

$\nabla^2 \boldsymbol{\psi} = - \boldsymbol{\omega}$

and computing:

$\mathbf{u} = \nabla \times \boldsymbol{\psi}$

This formulation removes the pressure variable and ensures
$\nabla \cdot \mathbf{u} = 0$ by construction.

In Scalar - VVP file the code follows the approach of Tan, J., Li, Q., Xi, H., Zheng, Q., & Pan, C. *A 3D DNS solver using the vorticity-vector potential-scalar potential formulation for incompressible turbulent flow with periodic boundary conditions*. [SSRN: 6552360](https://ssrn.com/abstract=6552360) / [DOI: 10.2139/ssrn.6552360](https://doi.org/10.2139/ssrn.6552360).

This includes the introduction of scalar potential $Φ$ by decomposing velocity as:

$\mathbf{u} = \nabla \times \boldsymbol{\psi} + \nabla Φ$

This decomposition yields another Poisson equation for Phi. This approach was used for the numerical simulation of Taylor-Green Vortex in 3 dimensions.

---

## Discretization

- **Time integration:** First-order explicit Euler scheme, (ADDED !) 4th order explicit Runge-Kutta
- **Spatial discretization:** Second-order central finite differences, (ADDED !), Central Difference Schemes 4th and 6th order,
- **Poisson Solver:** Jacobi and Conjugate Gradient Options
- **Grid:** Uniform collocated mesh with ghost cells options

---

## Installation

### Requirements

- Python 3.8+
- JAX 
- NumPy

---

## Output

Simulation results are exported in **Tecplot-compatible format** for
post-processing and visualization.
