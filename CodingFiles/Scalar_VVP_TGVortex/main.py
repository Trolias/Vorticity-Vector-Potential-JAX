import os
# os.environ["JAX_PLATFORM_NAME"] = "gpu"                 # Tell JAX to use GPU before any JAX import
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config
config.update("jax_enable_x64", True)                     
import jax
import jax.numpy as jnp
from Vorticity_solver import main_vorticity_loop
from helpers import calculate_divergence, write_tecplot_3d, save_checkpoint_binary
from Boundary_Conditions import apply_all_velocity_bcs_TG, apply_all_vorticity_bcs_TG
import time
import yaml
from nums import curl


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
gc = config["ghost_cells"]

# --- Grid Spacing Calculation ---
dx = L / (Nx)
dy = H / (Ny)
dz = D / (Nz)


# ---- TGV Parameters ----
V0   = 1.0
Lref = L / (2*jnp.pi)
nu   = V0 * Lref / Re            # = 1/1600 for L = 2*pi
t_end = 10.0 * Lref / V0         # t* = 10; change as needed
t_end = 2.0
kx = 2.0 * jnp.pi / L
ky = 2.0 * jnp.pi / H
kz = 2.0 * jnp.pi / D
#  ----------------------

# ---- Grid Coordinates ----
x = (jnp.arange(Nx) + 0.5) * dx
y = (jnp.arange(Ny) + 0.5) * dy
z = (jnp.arange(Nz) + 0.5) * dz
Y, X, Z = jnp.meshgrid(y, x, z, indexing='ij')  # shape (Ny, Nx, Nz)


# Initializing all fields with zeros as JAX NumPy arrays
psi_x = jnp.zeros((Ny+2*gc, Nx+2*gc, Nz+2*gc))
psi_y = jnp.zeros((Ny+2*gc, Nx+2*gc, Nz+2*gc))
psi_z = jnp.zeros((Ny+2*gc, Nx+2*gc, Nz+2*gc))

omega_x = jnp.zeros((Ny+2*gc, Nx+2*gc, Nz+2*gc))
omega_y = jnp.zeros((Ny+2*gc, Nx+2*gc, Nz+2*gc))
omega_z = jnp.zeros((Ny+2*gc, Nx+2*gc, Nz+2*gc))

omega_new_x = jnp.zeros_like(omega_x)
omega_new_y = jnp.zeros_like(omega_y)
omega_new_z = jnp.zeros_like(omega_z)

u = jnp.zeros_like(omega_x) # Initialize u, v, w as zeros
v = jnp.zeros_like(omega_y)
w = jnp.zeros_like(omega_z)


# ---- TGV Velocity Initial Conditions ----
u_phys =  V0 * jnp.sin(kx*X) * jnp.cos(ky*Y) * jnp.cos(kz*Z)
v_phys = -V0 * jnp.cos(kx*X) * jnp.sin(ky*Y) * jnp.cos(kz*Z)
w_phys =  jnp.zeros_like(u_phys)

u = u.at[gc:-gc, gc:-gc, gc:-gc].set(u_phys)
v = v.at[gc:-gc, gc:-gc, gc:-gc].set(v_phys)
w = w.at[gc:-gc, gc:-gc, gc:-gc].set(w_phys)

u, v, w = apply_all_velocity_bcs_TG(u, v, w, Uwall=0.0)
omega_x, omega_y, omega_z = apply_all_vorticity_bcs_TG(omega_x, omega_y, omega_z, u, v, w, dx, dy, dz)
psi_x, psi_y, psi_z = apply_all_velocity_bcs_TG(psi_x, psi_y, psi_z, Uwall=0)

print(f'Div(ω) = {jnp.max(jnp.abs(calculate_divergence(omega_x, omega_y, omega_z, dx, dy, dz)))}')
print(f'Div(u) = {jnp.max(jnp.abs(calculate_divergence(u, v, w, dx, dy, dz)))}')
print(f'Div(Ψ) = {jnp.max(jnp.abs(calculate_divergence(psi_x, psi_y, psi_z, dx, dy, dz)))}')

# --- Run the Main Simulation Loop ---
psi_final_x, psi_final_y, psi_final_z, \
omega_final_x, omega_final_y, omega_final_z, \
u_final, v_final, w_final, \
k_final, t_final, conv_error_final = main_vorticity_loop(
    psi_x, psi_y, psi_z,                        # Initial psi fields (zeros)
    omega_new_x, omega_new_y, omega_new_z,      # Initial omega fields (with BCs)
    u, v, w,                                    # Initial velocity fields (with BCs)
    dx, dy, dz,
    Uwall, Re, nu, conv_tol, t_end
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
filename = "TG_T8.dat" 

write_tecplot_3d(filename,
    psi_final_x, psi_final_y, psi_final_z,
    omega_final_x, omega_final_y, omega_final_z,
    u_final, v_final, w_final,
    div_omega_final_field, div_vel_final_field,
    dx, dy, dz, gc, L, H, D)

save_checkpoint_binary('./hello_1',psi_final_x,psi_final_y,psi_final_z,
                       omega_final_x, omega_final_y, omega_final_z,
                       u_final, v_final, w_final,
                       k_final,conv_error_final)