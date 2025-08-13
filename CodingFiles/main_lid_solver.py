import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"  # Tell JAX to use GPU before any JAX import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config
config.update("jax_enable_x64", True)  # Must be before any JAX ops
import jax
import jax.numpy as jnp
from jax import jit, lax
from JAX_Numerical_Schemes_3D import Grad_x, Grad_y, Grad_z, Laplacian_x, Laplacian_y, Laplacian_z, Adv_x, Adv_y, Adv_z
from JAX_PoissonSolver_3D import Poisson_solver, apply_potential_bcs_cavity, apply_potential_bcs_cavity_2ndOrder
from JAX_Vorticity_solver import apply_velocity_BCs, apply_vorticity_bcs, main_vorticity_loop 
from JAX_Utils_3D import* 
import time
import yaml


start_time = time.time()
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found. Please ensure it's in the same directory.")
    exit() 

# Access YAML parameters
Nx = config["Grid"]["Nx"]
Ny = config["Grid"]["Ny"]
Nz = config["Grid"]["Nz"]
L = float(config["Geometry"]["L"])
H = float(config["Geometry"]["H"])
D = float(config["Geometry"]["D"])
Re = float(config["Re"])
tol = float(config["Poisson"]["tol"])
ITERATIONS = config["Poisson"]["Iterations"] # Max iterations for internal Poisson solver
Uwall = float(config["Uwall"])
conv_tol = float(config["conv_tol"])         # Convergence tolerance for the main loop

# --- Grid Spacing Calculation ---
dx = L / (Nx - 1)
dy = H / (Ny - 1)
dz = D / (Nz - 1)


# Time step for stability (1st time step after that we use adaptive time step)
nu = Uwall * H / Re
print(f"Kinematic Viscosity is: {nu}")
dt = min(0.25 * dx * dx * Re / Uwall / H, 4 * H / Re / Uwall)  
print(f'dt = {dt:.4e}')
print(f'Reynolds number Re: {Re}')
print(f'CFL number is: {Uwall * dt / dy:.4e}')


# Initializing all fields with zeros as JAX NumPy arrays
psi_x = jnp.zeros((Ny, Nx, Nz))
psi_y = jnp.zeros((Ny, Nx, Nz))
psi_z = jnp.zeros((Ny, Nx, Nz))

omega_x = jnp.zeros((Ny, Nx, Nz))
omega_y = jnp.zeros((Ny, Nx, Nz))
omega_z = jnp.zeros((Ny, Nx, Nz))

omega_new_x = jnp.zeros_like(omega_x)
omega_new_y = jnp.zeros_like(omega_y)
omega_new_z = jnp.zeros_like(omega_z)

u = jnp.zeros_like(omega_x) # Initialize u, v, w as zeros
v = jnp.zeros_like(omega_y)
w = jnp.zeros_like(omega_z)


# --- Run the Main Simulation Loop ---
psi_final_x, psi_final_y, psi_final_z, \
omega_final_x, omega_final_y, omega_final_z, \
u_final, v_final, w_final, \
k_final, conv_error_final = main_vorticity_loop(
    psi_x, psi_y, psi_z,                        # Initial psi fields (zeros)
    omega_new_x, omega_new_y, omega_new_z,      # Initial omega fields (with BCs)
    u, v, w,                                    # Initial velocity fields (with BCs)
    dx, dy, dz,
    dt, Uwall, Re, nu, conv_tol, ITERATIONS, tol
)

# --- Post-Simulation Output ---
print(f"Final Error: {conv_error_final:.2e}")
print(f"Simulation completed in {k_final} iterations.")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")

# Calculate final divergence for diagnostics 
div_omega_final_field = calculate_divergence(omega_final_x, omega_final_y, omega_final_z, dx, dy, dz)
max_div_omega_final = jnp.max(jnp.abs(div_omega_final_field))
print(f"Max abs(Divergence Ω) at end: {max_div_omega_final:.2e}")
div_vel_final_field = calculate_divergence(u_final, v_final, w_final, dx, dy, dz)
max_div_u_final = jnp.max(jnp.abs(div_vel_final_field))
print(f"Max abs(Divergence U) at end: {max_div_u_final:.2e}")


# --- Save Results to VTK ---
filename = "NS_solver.vtk" 

save_to_vtk_fast(psi_final_x, psi_final_y, psi_final_z,
            omega_final_x, omega_final_y, omega_final_z,
            u_final, v_final, w_final, div_omega_final_field, div_vel_final_field, # Use the final velocity fields from the loop
            dx, dy, dz,
            filename)

save_checkpoint_binary('./hello_1',psi_final_x,psi_final_y,psi_final_z,
                       omega_final_x, omega_final_y, omega_final_z,
                       u_final, v_final, w_final,
                       k_final,conv_error_final)
