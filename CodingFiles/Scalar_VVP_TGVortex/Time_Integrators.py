import jax
import jax.numpy as jnp
from jax import config
import yaml
from nums import Laplacian, grad, grad4, grad8
from Boundary_Conditions import apply_all_vorticity_bcs, apply_all_vorticity_bcs_TG, apply_all_potential_bcs_TG, apply_all_velocity_bcs_TG
from Poisson_solver_JAX import Spectral_Tricks
from nums import shift

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found. Please ensure it's in the same directory.")
    exit()

gc = config["ghost_cells"]
L = float(config["Geometry"]["L"])
H = float(config["Geometry"]["H"])
D = float(config["Geometry"]["D"])
Nx = config["Grid"]["Nx"]
Ny = config["Grid"]["Ny"]
Nz = config["Grid"]["Nz"]
dx = L / Nx
dy = H / Ny
dz = D / Nz
Re = config["Re"]

def solve_rhs(omega_x, omega_y, omega_z, u_new, v_new, w_new, dx, dy, dz):

    # 4. Vorticity transport equation
    laplacian_omega_x = Laplacian(omega_x, dx, dy, dz)
    laplacian_omega_y = Laplacian(omega_y, dx, dy, dz)
    laplacian_omega_z = Laplacian(omega_z, dx, dy, dz)

    skew_x = u_new*grad8(omega_x, dx, 1) + v_new*grad8(omega_x, dy, 0) + w_new*grad8(omega_x, dz, 2)
    skew_y = u_new*grad8(omega_y, dx, 1) + v_new*grad8(omega_y, dy, 0) + w_new*grad8(omega_y, dz, 2)
    skew_z = u_new*grad8(omega_z, dx, 1) + v_new*grad8(omega_z, dy, 0) + w_new*grad8(omega_z, dz, 2)

    conserv_x = grad4(omega_x*u_new, dx, 1) + grad4(omega_x*v_new, dy, 0) + grad4(omega_x*w_new, dz, 2)
    conserv_y = grad4(omega_y*u_new, dx, 1) + grad4(omega_y*v_new, dy, 0) + grad4(omega_y*w_new, dz, 2)
    conserv_z = grad4(omega_z*u_new, dx, 1) + grad4(omega_z*v_new, dy, 0) + grad4(omega_z*w_new, dz, 2)

    div_corr_x = omega_x*grad4(u_new, dx, 1) + omega_x*grad4(v_new, dy, 0) + omega_x*grad4(w_new, dz, 2)
    div_corr_y = omega_y*grad4(u_new, dx, 1) + omega_y*grad4(v_new, dy, 0) + omega_y*grad4(w_new, dz, 2)
    div_corr_z = omega_z*grad4(u_new, dx, 1) + omega_z*grad4(v_new, dy, 0) + omega_z*grad4(w_new, dz, 2)

    convection_term_x = 0.5 * (skew_x + conserv_x + div_corr_x)
    convection_term_y = 0.5 * (skew_y + conserv_y + div_corr_y)
    convection_term_z = 0.5 * (skew_z + conserv_z + div_corr_z)

    vortex_stretching_x = omega_x*grad4(u_new, dx, 1) + omega_y*grad4(u_new, dy, 0) + omega_z*grad4(u_new, dz, 2)
    vortex_stretching_y = omega_x*grad4(v_new, dx, 1) + omega_y*grad4(v_new, dy, 0) + omega_z*grad4(v_new, dz, 2)
    vortex_stretching_z = omega_x*grad4(w_new, dx, 1) + omega_y*grad4(w_new, dy, 0) + omega_z*grad4(w_new, dz, 2)

    rhs_x = (1.0/Re)*laplacian_omega_x - convection_term_x + vortex_stretching_x
    rhs_y = (1.0/Re)*laplacian_omega_y - convection_term_y + vortex_stretching_y
    rhs_z = (1.0/Re)*laplacian_omega_z - convection_term_z + vortex_stretching_z

    return rhs_x, rhs_y, rhs_z


def RK4(fx, fy, fz, u, v, w, dx, dy, dz, dt):
    k1_x, k1_y, k1_z = solve_rhs(fx, fy, fz, u, v, w, dx, dy, dz)
    s1_x, s1_y, s1_z = apply_all_vorticity_bcs_TG(fx + 0.5*dt*k1_x,fy + 0.5*dt*k1_y,fz + 0.5*dt*k1_z,u,v,w,dx,dy,dz)
    psi1_x, psi1_y, psi1_z, u1, v1, w1 = Spectral_Tricks(u, v, w, shift(s1_x), shift(s1_y), shift(s1_z))
    psi1_x, psi1_y, psi1_z = apply_all_potential_bcs_TG(psi1_x, psi1_y, psi1_z)
    u1, v1, w1 = apply_all_velocity_bcs_TG(u1, v1, w1, 0.0)

    k2_x, k2_y, k2_z = solve_rhs(s1_x, s1_y, s1_z, u1, v1, w1, dx, dy, dz)
    s2_x, s2_y, s2_z = apply_all_vorticity_bcs_TG(fx + 0.5*dt*k2_x,fy + 0.5*dt*k2_y,fz + 0.5*dt*k2_z,u1,v1,w1,dx,dy,dz)
    psi2_x, psi2_y, psi2_z, u2, v2, w2 = Spectral_Tricks(u1, v1, w1, shift(s2_x), shift(s2_y), shift(s2_z))
    psi2_x, psi2_y, psi2_z = apply_all_potential_bcs_TG(psi2_x, psi2_y, psi2_z)
    u2, v2, w2 = apply_all_velocity_bcs_TG(u2, v2, w2, 0.0)


    k3_x, k3_y, k3_z = solve_rhs(s2_x, s2_y, s2_z, u2, v2, w2, dx, dy, dz)
    s3_x, s3_y, s3_z = apply_all_vorticity_bcs_TG(fx + dt*k3_x,fy + dt*k3_y,fz + dt*k3_z,u2,v2,w2,dx,dy,dz)
    psi3_x, psi3_y, psi3_z, u3, v3, w3 = Spectral_Tricks(u2, v2, w2, shift(s3_x), shift(s3_y), shift(s3_z))
    psi3_x, psi3_y, psi3_z = apply_all_potential_bcs_TG(psi3_x, psi3_y, psi3_z)
    u3, v3, w3 = apply_all_velocity_bcs_TG(u3, v3, w3, 0.0)

    k4_x, k4_y, k4_z = solve_rhs(s3_x, s3_y, s3_z, u3, v3, w3, dx, dy, dz)
    f_new_x = fx + dt/6.0*(k1_x + 2.0*k2_x + 2.0*k3_x + k4_x)
    f_new_y = fy + dt/6.0*(k1_y + 2.0*k2_y + 2.0*k3_y + k4_y)
    f_new_z = fz + dt/6.0*(k1_z + 2.0*k2_z + 2.0*k3_z + k4_z)
    f_new_x, f_new_y, f_new_z = apply_all_vorticity_bcs_TG(f_new_x,f_new_y,f_new_z,u3,v3,w3,dx,dy,dz)
    psi4_x, psi4_y, psi4_z, u4, v4, w4 = Spectral_Tricks(u3, v3, w3, shift(f_new_x), shift(f_new_y), shift(f_new_z))
    psi4_x, psi4_y, psi4_z = apply_all_potential_bcs_TG(psi4_x, psi4_y, psi4_z)
    u4, v4, w4 = apply_all_velocity_bcs_TG(u4, v4, w4, 0.0)

    return f_new_x, f_new_y, f_new_z, u4, v4, w4, psi4_x, psi4_y, psi4_z

def Forward_Euler(fx, fy, fz, u, v, w, dx, dy, dz, dt):
    rhs_x, rhs_y, rhs_z = solve_rhs(fx, fy, fz, u, v, w, dx, dy, dz)
    f_new_x = fx + dt*rhs_x
    f_new_y = fy + dt*rhs_y
    f_new_z = fz + dt*rhs_z

    return f_new_x, f_new_y, f_new_z