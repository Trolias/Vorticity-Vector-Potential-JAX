import jax
import jax.numpy as jnp
from jax import jit, lax
from Poisson_solver_JAX import build_pressure_coeffs, potential_poisson_cg
from nums import grad, shift, curl, grad4
from helpers import adaptive_time_step, calculate_divergence
from Boundary_Conditions import apply_potential_bcs_cavity, apply_all_vorticity_bcs, apply_all_velocity_bcs
from jax import config
from Time_Integrators import RK4, Forward_Euler
config.update("jax_enable_x64", True)
import yaml

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


def calc_velocities(psi_x, psi_y, psi_z, dx, dy, dz):
    u = grad(psi_z, dy, 0) - grad(psi_y, dz, 2)
    v = grad(psi_x, dz, 2) - grad(psi_z, dx, 1)
    w = grad(psi_y, dx, 1) - grad(psi_x, dy, 0)
    return u, v, w


def calc_vorticity(u, v, w, dx, dy, dz):
    omega_x = grad(w, dy, 0) - grad(v, dz, 2)
    omega_y = grad(u, dz, 2) - grad(w, dx, 1)
    omega_z = grad(v, dx, 1) - grad(u, dy, 0)
    return omega_x, omega_y, omega_z


def solve_potential_component(psi, omega, bc_type, Nx, Ny, Nz, dx, dy, dz):
    ap, bc_s, cx, cy, cz = build_pressure_coeffs(Nx, Ny, Nz, dx, dy, dz, bc_type)
    return psi.at[gc:-gc, gc:-gc, gc:-gc].set(
        potential_poisson_cg(shift(psi), shift(omega), ap, bc_s, cx, cy, cz))


PSI_BC_TYPES = {
    "psi_x": ("Dirichlet", "Dirichlet", "Neumann",  "Neumann",  "Dirichlet", "Dirichlet"),
    "psi_y": ("Neumann",   "Neumann",   "Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet"),
    "psi_z": ("Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet", "Neumann",   "Neumann"),
}


def vorticity_step(psi_x, psi_y, psi_z,
                    omega_x, omega_y, omega_z,
                    dx, dy, dz,
                    Uwall, Re, nu, dt_max):

    # 1. Solve the vector-potential Poisson equations
    psi_new_x = apply_potential_bcs_cavity(
        solve_potential_component(psi_x, omega_x, PSI_BC_TYPES["psi_x"], Nx, Ny, Nz, dx, dy, dz), "psi_x")
    psi_new_y = apply_potential_bcs_cavity(
        solve_potential_component(psi_y, omega_y, PSI_BC_TYPES["psi_y"], Nx, Ny, Nz, dx, dy, dz), "psi_y")
    psi_new_z = apply_potential_bcs_cavity(
        solve_potential_component(psi_z, omega_z, PSI_BC_TYPES["psi_z"], Nx, Ny, Nz, dx, dy, dz), "psi_z")


    # 2. Calculate velocities from the new streamfunction values
    u_new, v_new, w_new = calc_velocities(psi_new_x, psi_new_y, psi_new_z, dx, dy, dz)
    u_new, v_new, w_new = apply_all_velocity_bcs(u_new, v_new, w_new, Uwall)

    # 3. Adaptive time step and vorticity BCs ahead of the transport update
    dt = jnp.minimum(adaptive_time_step(u_new, v_new, w_new, dx, dy, dz, nu), dt_max)

    omega_x, omega_y, omega_z = apply_all_vorticity_bcs(omega_x, omega_y, omega_z, u_new, v_new, w_new, dx, dy, dz)
    omega_new_x, omega_new_y, omega_new_z = RK4(omega_x, omega_y, omega_z, u_new, v_new, w_new, dx, dy, dz, dt)
    # omega_new_x, omega_new_y, omega_new_z = Forward_Euler(omega_x, omega_y, omega_z, u_new, v_new, w_new, dx, dy, dz, dt)


    # 5. Apply BCs to the new vorticity fields
    omega_new_x, omega_new_y, omega_new_z = apply_all_vorticity_bcs(
        omega_new_x, omega_new_y, omega_new_z, u_new, v_new, w_new, dx, dy, dz)

    divergence     = calculate_divergence(omega_new_x, omega_new_y, omega_new_z, dx, dy, dz)
    divergence_vel = calculate_divergence(u_new, v_new, w_new, dx, dy, dz)
    divergence_pot = calculate_divergence(psi_new_x, psi_new_y, psi_new_z, dx, dy, dz)
    jax.debug.print(
        "Time step {dti}, vorticity divergence, div(omega) = {div:.6e}, "
        "velocity divergence, div(U) = {divu:.6e} and potential divergence, div(Psi) = {divpsi:.6e}",
        dti=dt, div=jnp.max(jnp.abs(divergence)),
        divu=jnp.max(jnp.abs(divergence_vel)), divpsi=jnp.max(jnp.abs(divergence_pot)))
    jax.debug.print("Max omega = {vor:.6e}", vor=jnp.max(jnp.abs(omega_z)))

    return psi_new_x, psi_new_y, psi_new_z, omega_new_x, omega_new_y, omega_new_z, u_new, v_new, w_new, dt


@jit
def main_vorticity_loop(psi_init_x, psi_init_y, psi_init_z,
                        omega_init_x, omega_init_y, omega_init_z,
                        u_init, v_init, w_init,
                        dx, dy, dz, Uwall, Re, nu, conv_tol, t_end):

    def cond_fn(state):
        k, t, conv_error = state[9], state[10], state[11]
        return (k < 100000) & (t < t_end * (1.0 - 1e-12)) & (conv_error > conv_tol)

    def body_fn(state):
        psi_x, psi_y, psi_z, omega_x, omega_y, omega_z, u, v, w, k, t, _ = state

        (psi_new_x, psi_new_y, psi_new_z,
         omega_new_x, omega_new_y, omega_new_z,
         u_new, v_new, w_new, dt) = vorticity_step(
            psi_x, psi_y, psi_z, omega_x, omega_y, omega_z,
            dx, dy, dz, Uwall, Re, nu, t_end - t)

        norm_delta_omega = jnp.sqrt(jnp.linalg.norm(omega_new_x - omega_x)**2 +
                                    jnp.linalg.norm(omega_new_y - omega_y)**2 +
                                    jnp.linalg.norm(omega_new_z - omega_z)**2)
        norm_omega = jnp.sqrt(jnp.linalg.norm(omega_x)**2 +
                              jnp.linalg.norm(omega_y)**2 +
                              jnp.linalg.norm(omega_z)**2)
        conv_error = norm_delta_omega / (norm_omega + 1e-10)
        jax.debug.print("Iteration {kt}, t = {tt:.5f}, convergence error {error:.6e}",
                        kt=k, tt=t + dt, error=conv_error)

        return (psi_new_x, psi_new_y, psi_new_z,
                omega_new_x, omega_new_y, omega_new_z,
                u_new, v_new, w_new,
                k + 1, t + dt, conv_error)

    u_initial_bcs, v_initial_bcs, w_initial_bcs = apply_all_velocity_bcs(u_init, v_init, w_init, Uwall)
    omega_init_x, omega_init_y, omega_init_z = curl(u_initial_bcs, v_initial_bcs, w_initial_bcs, dx, dy, dz)
    omega_x_initial_bcs, omega_y_initial_bcs, omega_z_initial_bcs = apply_all_vorticity_bcs(
        omega_init_x, omega_init_y, omega_init_z, u_init, v_init, w_init, dx, dy, dz)

    state = (psi_init_x, psi_init_y, psi_init_z,
             omega_x_initial_bcs, omega_y_initial_bcs, omega_z_initial_bcs,
             u_initial_bcs, v_initial_bcs, w_initial_bcs,
             0, 0.0, 1.0)

    (psi_final_x, psi_final_y, psi_final_z,
     omega_final_x, omega_final_y, omega_final_z,
     u_final, v_final, w_final,
     steps, t_final, final_error) = lax.while_loop(cond_fn, body_fn, state)

    return (psi_final_x, psi_final_y, psi_final_z,
            omega_final_x, omega_final_y, omega_final_z,
            u_final, v_final, w_final,
            steps, t_final, final_error)


