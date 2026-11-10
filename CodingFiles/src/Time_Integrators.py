import jax
import jax.numpy as jnp
from jax import config
import yaml
from nums import Laplacian, grad, grad4, grad8, weno_flux_div, weno_advect
from Boundary_Conditions import apply_all_vorticity_bcs

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found.")
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


def solve_rhs(omega_x, omega_y, omega_z, u_new, v_new, w_new, dx, dy, dz,):
    laplacian_omega_x = Laplacian(omega_x, dx, dy, dz)
    laplacian_omega_y = Laplacian(omega_y, dx, dy, dz)
    laplacian_omega_z = Laplacian(omega_z, dx, dy, dz)

    divu8 = grad8(u_new, dx, 1) + grad8(v_new, dy, 0) + grad8(w_new, dz, 2)

    def convection(om):
        skew = u_new*grad(om, dx, 1) + v_new*grad(om, dy, 0) + w_new*grad(om, dz, 2)
        conserv = grad4(om*u_new, dx, 1) + grad4(om*v_new, dy, 0) + grad4(om*w_new, dz, 2)
        return 0.5*(skew + conserv + om*divu8)
        # return skew

    convection_term_x = convection(omega_x)
    convection_term_y = convection(omega_y)
    convection_term_z = convection(omega_z)

    # vortex stretching unchanged
    vortex_stretching_x = omega_x*grad(u_new, dx, 1) + omega_y*grad(u_new, dy, 0) + omega_z*grad(u_new, dz, 2)
    vortex_stretching_y = omega_x*grad(v_new, dx, 1) + omega_y*grad(v_new, dy, 0) + omega_z*grad(v_new, dz, 2)
    vortex_stretching_z = omega_x*grad(w_new, dx, 1) + omega_y*grad(w_new, dy, 0) + omega_z*grad(w_new, dz, 2)

    rhs_x = (1.0/Re)*laplacian_omega_x - convection_term_x + vortex_stretching_x
    rhs_y = (1.0/Re)*laplacian_omega_y - convection_term_y + vortex_stretching_y
    rhs_z = (1.0/Re)*laplacian_omega_z - convection_term_z + vortex_stretching_z
    return rhs_x, rhs_y, rhs_z


def RK4(fx, fy, fz, u, v, w, dx, dy, dz, dt):
    k1_x, k1_y, k1_z = solve_rhs(fx, fy, fz, u, v, w, dx, dy, dz)
    s1_x, s1_y, s1_z = apply_all_vorticity_bcs(fx + 0.5*dt*k1_x,fy + 0.5*dt*k1_y,fz + 0.5*dt*k1_z,u,v,w,dx,dy,dz)

    k2_x, k2_y, k2_z = solve_rhs(s1_x, s1_y, s1_z, u, v, w, dx, dy, dz)
    s2_x, s2_y, s2_z = apply_all_vorticity_bcs(fx + 0.5*dt*k2_x,fy + 0.5*dt*k2_y,fz + 0.5*dt*k2_z,u,v,w,dx,dy,dz)


    k3_x, k3_y, k3_z = solve_rhs(s2_x, s2_y, s2_z, u, v, w, dx, dy, dz)
    s3_x, s3_y, s3_z = apply_all_vorticity_bcs(fx + dt*k3_x,fy + dt*k3_y,fz + dt*k3_z,u,v,w,dx,dy,dz)

    k4_x, k4_y, k4_z = solve_rhs(s3_x, s3_y, s3_z, u, v, w, dx, dy, dz)
    f_new_x = fx + dt/6.0*(k1_x + 2.0*k2_x + 2.0*k3_x + k4_x)
    f_new_y = fy + dt/6.0*(k1_y + 2.0*k2_y + 2.0*k3_y + k4_y)
    f_new_z = fz + dt/6.0*(k1_z + 2.0*k2_z + 2.0*k3_z + k4_z)

    return f_new_x, f_new_y, f_new_z

def Forward_Euler(fx, fy, fz, u, v, w, dx, dy, dz, dt):
    rhs_x, rhs_y, rhs_z = solve_rhs(fx, fy, fz, u, v, w, dx, dy, dz)
    f_new_x = fx + dt*rhs_x
    f_new_y = fy + dt*rhs_y
    f_new_z = fz + dt*rhs_z

    return f_new_x, f_new_y, f_new_z