import jax
import jax.numpy as jnp
from jax import jit, lax
from JAX_PoissonSolver_3D import Poisson_solver 
from JAX_Numerical_Schemes_3D import*
from JAX_Utils_3D import*  
from functools import partial
from JAX_BCs_cavity import*
from jax import config
config.update("jax_enable_x64", True)



def calc_velocities(psi_x, psi_y, psi_z, dx, dy, dz):

    u = Grad_y(psi_z, dy) - Grad_z(psi_y, dz)
    v = Grad_z(psi_x, dz) - Grad_x(psi_z, dx)
    w = Grad_x(psi_y, dx) - Grad_y(psi_x, dy)
    return u, v, w

def calc_vorticity(u,v,w,dx,dy,dz):

    omega_x = Grad_y(w,dy) - Grad_z(v,dz)
    omega_y = Grad_z(u,dz) - Grad_x(w,dx)
    omega_z = Grad_x(v,dx) - Grad_y(u,dy)

    return omega_x, omega_y, omega_z 


def vorticity_step(psi_x, psi_y, psi_z,
                   omega_x, omega_y, omega_z,
                   u, v, w, # u, v, w are current velocities, used for transport terms
                   dx, dy, dz,
                   dt, Uwall, Re, nu, Poisson_iterations, Poisson_tol):
    """
    Performs a single step of the vorticity-streamfunction solver.

    Args:
        psi_x, psi_y, psi_z (jnp.ndarray): Current streamfunction fields.
        omega_x, omega_y, omega_z (jnp.ndarray): Current vorticity fields.
        u, v, w (jnp.ndarray): Current velocity fields (used for advection/stretching).
        dx, dy, dz (float): Grid spacings.
        dt (float): Time step.
        Uwall (float): Wall velocity.
        Re (float): Reynolds number.
        Poisson_iterations (int): Max iterations for the internal Poisson solver.
        Poisson_tol (float): Tolerance for the internal Poisson solver.

    Returns:
        tuple: (psi_new_x, psi_new_y, psi_new_z,
                omega_new_x, omega_new_y, omega_new_z,
                u_new, v_new, w_new) updated fields.
    """
    # 1. Solve Poisson equation for streamfunctions (psi_x, psi_y, psi_z)
    psi_new_x, _, _ = Poisson_solver(psi_x, dx, dy, dz, -omega_x, Poisson_iterations, Poisson_tol, 'psi_x')
    psi_new_y, _, _ = Poisson_solver(psi_y, dx, dy, dz, -omega_y, Poisson_iterations, Poisson_tol, 'psi_y')
    psi_new_z, _, _ = Poisson_solver(psi_z, dx, dy, dz, -omega_z, Poisson_iterations, Poisson_tol, 'psi_z')

    # 2. Calculate Velocities from new streamfunction values
    u_new, v_new, w_new = calc_velocities(psi_new_x, psi_new_y, psi_new_z, dx, dy, dz)

    # 3. Apply boundary conditions to the newly calculated velocities
    u_new = apply_velocity_BCs(u_new, Uwall, "u")
    v_new = apply_velocity_BCs(v_new, Uwall, "v")
    w_new = apply_velocity_BCs(w_new, Uwall, "w")

    # 3.5 Calculate time step and apply BCs for vorticity
    dt = adaptive_time_step(u_new,v_new,w_new,dx,dy,dz,nu)

    omega_x = apply_vorticity_bcs(omega_x, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_x")
    omega_y = apply_vorticity_bcs(omega_y, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_y")
    omega_z = apply_vorticity_bcs(omega_z, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_z")

    # 4. Compute vorticity transport equation:
    # Diffusion Term (Laplacian of vorticity)
    laplacian_omega_x = Laplacian_x(omega_x, dx) + Laplacian_y(omega_x, dy) + Laplacian_z(omega_x, dz)
    laplacian_omega_y = Laplacian_x(omega_y, dx) + Laplacian_y(omega_y, dy) + Laplacian_z(omega_y, dz)
    laplacian_omega_z = Laplacian_x(omega_z, dx) + Laplacian_y(omega_z, dy) + Laplacian_z(omega_z, dz)

    # Convection Term (using Adv_x, Adv_y, Adv_z for upwind scheme)
    convection_term_x = Adv_x(u_new, omega_x, dx) + Adv_y(v_new, omega_x, dy) + Adv_z(w_new, omega_x, dz)
    convection_term_y = Adv_x(u_new, omega_y, dx) + Adv_y(v_new, omega_y, dy) + Adv_z(w_new, omega_y, dz)
    convection_term_z = Adv_x(u_new, omega_z, dx) + Adv_y(v_new, omega_z, dy) + Adv_z(w_new, omega_z, dz)


    # Uncomment to use Central Differences
    # convection_term_x = u_new*Grad_x(omega_x, dx) + v_new*Grad_y(omega_x, dy) + w_new*Grad_z(omega_x, dz)
    # convection_term_y = u_new*Grad_x(omega_y, dx) + v_new*Grad_y(omega_y, dy) + w_new*Grad_z(omega_y, dz)
    # convection_term_z = u_new*Grad_x(omega_z, dx) + v_new*Grad_y(omega_z, dy) + w_new*Grad_z(omega_z, dz)

    # Vortex Stretching Term
    vortex_stretching_x = omega_x * Grad_x(u_new, dx) + omega_y * Grad_y(u_new, dy) + omega_z * Grad_z(u_new, dz)
    vortex_stretching_y = omega_x * Grad_x(v_new, dx) + omega_y * Grad_y(v_new, dy) + omega_z * Grad_z(v_new, dz)
    vortex_stretching_z = omega_x * Grad_x(w_new, dx) + omega_y * Grad_y(w_new, dy) + omega_z * Grad_z(w_new, dz)

    # Update new vorticity components 
    omega_new_x = omega_x + dt * (
        (1.0 / Re) * ( laplacian_omega_x )
        - convection_term_x
        + vortex_stretching_x 
    )

    omega_new_y = omega_y + dt * (
        (1.0 / Re) * ( laplacian_omega_y  )
        - convection_term_y
        + vortex_stretching_y 
    )

    omega_new_z = omega_z + dt * (
        (1.0 / Re) * (laplacian_omega_z )
        - convection_term_z
        + vortex_stretching_z 
    )

    # 5. Apply boundary conditions to the new vorticity fields

    omega_new_x = apply_vorticity_bcs(omega_new_x, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_x")
    omega_new_y = apply_vorticity_bcs(omega_new_y, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_y")
    omega_new_z = apply_vorticity_bcs(omega_new_z, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_z")

    # omega_new_x = apply_vorticity_bcs_2ndOrder(omega_new_x, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_x")
    # omega_new_y = apply_vorticity_bcs_2ndOrder(omega_new_y, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_y")
    # omega_new_z = apply_vorticity_bcs_2ndOrder(omega_new_z, u_new, v_new, w_new, dx, dy, dz, Uwall, "omega_new_z")

    divergence = calculate_divergence(omega_new_x, omega_new_y,omega_new_z,dx,dy,dz)
    div_max = jnp.max(jnp.abs(divergence))
    divergence_vel = calculate_divergence(u_new, v_new,w_new,dx,dy,dz)
    div_max_vel = jnp.max(jnp.abs(divergence_vel))
    divergence_pot = calculate_divergence(psi_new_x, psi_new_y, psi_new_z,dx,dy,dz)
    div_max_pot = jnp.max(jnp.abs(divergence_pot))
    jax.debug.print("Time step {dti}, vorticity divergence, div(ω) = {div:.6e}, velocity divergence, div(U) = {divu:.6e} and potential divergence, div(Ψ) = {divpsi:.6e} ", dti=dt, div= div_max, divu = div_max_vel, divpsi = div_max_pot)
    jax.debug.print("Max omega = {vor:.6e}", vor = jnp.max(jnp.abs(omega_z)) )

    # Return all updated fields for the next iteration of the main loop
    return psi_new_x, psi_new_y, psi_new_z, omega_new_x, omega_new_y, omega_new_z, u_new, v_new, w_new

# Use @jit flag here to speed up the execution of the code
@jit
def main_vorticity_loop(psi_init_x, psi_init_y, psi_init_z,
                        omega_init_x, omega_init_y, omega_init_z,
                        u_init, v_init, w_init, # Initial velocities as part of state
                        dx, dy, dz,
                        dt, Uwall, Re, nu, conv_tol, Poisson_iterations, Poisson_tol):
    """
    Main 3D vorticity-streamfunction solver loop using JAX.

    Args:
        psi_init_x, psi_init_y, psi_init_z (jnp.ndarray): Initial streamfunction fields.
        omega_init_x, omega_init_y, omega_init_z (jnp.ndarray): Initial vorticity fields.
        u_init, v_init, w_init (jnp.ndarray): Initial velocity fields.
        dx, dy, dz (float): Grid spacing.
        dt (float): Time step.
        Uwall (float): Wall velocity.
        Re (float): Reynolds number.
        conv_tol (float): Convergence tolerance for the main loop.
        Poisson_iterations (int): Max iterations for the internal Poisson solver.
        Poisson_tol (float): Tolerance for the internal Poisson solver.

    Returns:
        tuple: (psi_final_x, psi_final_y, psi_final_z,
                omega_final_x, omega_final_y, omega_final_z,
                u_final, v_final, w_final, # Final velocities
                steps, final_error)
    """
    # Define the condition for the while loop
    def cond_fn(state):
        _, _, _, _, _, _, _, _, _, k, conv_error = state
        # The loop continues as long as k is less than a max number of steps (e.g., 30000)
        # AND the convergence error is greater than the specified tolerance.
        return jnp.logical_and(k < 100000, conv_error > conv_tol)

    # Define the body of the while loop
    def body_fn(state):
        psi_x, psi_y, psi_z, omega_x, omega_y, omega_z, u, v, w, k, _ = state

        # Take one vorticity step, which updates all fields
        psi_new_x, psi_new_y, psi_new_z, \
        omega_new_x, omega_new_y, omega_new_z, \
        u_new, v_new, w_new = vorticity_step(psi_x, psi_y, psi_z,
                                             omega_x, omega_y, omega_z,
                                             u, v, w, 
                                             dx, dy, dz,
                                             dt, Uwall, Re, nu, Poisson_iterations, Poisson_tol)

        # Compute convergence error based on vorticity change
        delta_omega_x = omega_new_x - omega_x
        delta_omega_y = omega_new_y - omega_y
        delta_omega_z = omega_new_z - omega_z

        norm_delta_omega_sq = (jnp.linalg.norm(delta_omega_x)**2 +
                               jnp.linalg.norm(delta_omega_y)**2 +
                               jnp.linalg.norm(delta_omega_z)**2)
        norm_delta_omega = jnp.sqrt(norm_delta_omega_sq)

        norm_omega_sq = (jnp.linalg.norm(omega_x)**2 +
                         jnp.linalg.norm(omega_y)**2 +
                         jnp.linalg.norm(omega_z)**2)
        norm_omega = jnp.sqrt(norm_omega_sq)

        # Add a small epsilon to the denominator to avoid division by zero
        conv_error = norm_delta_omega / (norm_omega + 1e-10)
        jax.debug.print("Iteration {kt}, convergence error {error:.6e}", kt=k, error=conv_error)

        # Return the updated state for the next iteration
        return (psi_new_x, psi_new_y, psi_new_z,
                omega_new_x, omega_new_y, omega_new_z,
                u_new, v_new, w_new, 
                k + 1, conv_error)

    # Initial state for the while loop
    # Initial convergence error is set to a value > conv_tol to ensure the loop runs
    # k (iteration count) starts at 0
    conv_error_init = 1.0 # Set high to ensure loop starts
    k_init = 0

    # Apply initial boundary conditions for vorticity and velocities
    u_initial_bcs = apply_velocity_BCs(u_init, Uwall, "u")
    v_initial_bcs = apply_velocity_BCs(v_init, Uwall, "v")
    w_initial_bcs = apply_velocity_BCs(w_init, Uwall, "w")

    omega_init_x, omega_init_y, omega_init_z = curl(u_initial_bcs, v_initial_bcs, w_initial_bcs, dx, dy, dz)
    omega_x_initial_bcs = apply_vorticity_bcs(omega_init_x, u_init, v_init, w_init, dx, dy, dz, Uwall, "omega_new_x")
    omega_y_initial_bcs = apply_vorticity_bcs(omega_init_y, u_init, v_init, w_init, dx, dy, dz, Uwall, "omega_new_y")
    omega_z_initial_bcs = apply_vorticity_bcs(omega_init_z, u_init, v_init, w_init, dx, dy, dz, Uwall, "omega_new_z")

    # jax.debug.print("ω_x max: {x}, ω_y max: {y}, ω_z max: {z}", 
    #             x=jnp.max(jnp.abs(omega_x_initial_bcs)), 
    #             y=jnp.max(jnp.abs(omega_y_initial_bcs)), 
    #             z=jnp.max(jnp.abs(omega_z_initial_bcs)))
    # omega_x_initial_bcs = apply_vorticity_bcs_2ndOrder(omega_init_x, u_init, v_init, w_init, dx, dy, dz, Uwall, "omega_new_x")
    # omega_y_initial_bcs = apply_vorticity_bcs_2ndOrder(omega_init_y, u_init, v_init, w_init, dx, dy, dz, Uwall, "omega_new_y")
    # omega_z_initial_bcs = apply_vorticity_bcs_2ndOrder(omega_init_z, u_init, v_init, w_init, dx, dy, dz, Uwall, "omega_new_z")

    # psi_init_x, psi_init_y, psi_init_z, \
    # omega_x_initial_bcs, omega_y_initial_bcs, omega_z_initial_bcs, \
    # u_initial_bcs, v_initial_bcs, w_initial_bcs, \
    # k_init, conv_error_init = load_checkpoint_binary('./hello_1')

    # Initial state tuple for lax.while_loop
    state = (psi_init_x, psi_init_y, psi_init_z,
             omega_x_initial_bcs, omega_y_initial_bcs, omega_z_initial_bcs,
             u_initial_bcs, v_initial_bcs, w_initial_bcs,
             k_init, conv_error_init)

    # Run the main simulation loop
    psi_final_x, psi_final_y, psi_final_z, \
    omega_final_x, omega_final_y, omega_final_z, \
    u_final, v_final, w_final, \
    steps, final_error = lax.while_loop(cond_fn, body_fn, state)

    return psi_final_x, psi_final_y, psi_final_z, \
           omega_final_x, omega_final_y, omega_final_z, \
           u_final, v_final, w_final, \
           steps, final_error

#
