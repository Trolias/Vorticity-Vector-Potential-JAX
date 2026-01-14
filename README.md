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

$
\frac{\partial \boldsymbol{\omega}}{\partial t}
+ (\mathbf{u} \cdot \nabla)\boldsymbol{\omega}
= (\boldsymbol{\omega} \cdot \nabla)\mathbf{u}
+ \frac{1}{Re}\nabla^2 \boldsymbol{\omega}
$

The velocity field is recovered from the vector potential by solving the
Poisson equation:

$
\nabla^2 \boldsymbol{\psi} = - \boldsymbol{\omega}
$

and computing:

$
\mathbf{u} = \nabla \times \boldsymbol{\psi}
$

This formulation removes the pressure variable and ensures
$\nabla \cdot \mathbf{u} = 0$ by construction.

---

## Discretization

- **Time integration:** First-order explicit Euler scheme
- **Spatial discretization:** Second-order central finite differences
- **Grid:** Uniform collocated mesh

---

## Installation

### Requirements

- Python 3.8+
- JAX (with GPU support recommended)
- NumPy

---

## Output

Simulation results are exported in **Tecplot-compatible format** for
post-processing and visualization.
