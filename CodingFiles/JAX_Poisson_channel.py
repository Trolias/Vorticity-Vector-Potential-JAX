import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from JAX_Utils_3D import CornersBC
from functools import partial
from jax import config
config.update("jax_enable_x64", True) 


def apply_potential_BCs_channel(u_new, dx, dy, dz, volumetric_flux, H, field_name):

    Ny = u_new.shape[0]
    Y = jnp.linspace(0.0, H, Ny)
    Y_profile = (volumetric_flux * Y[1:-1]).reshape(-1, 1)
    value = volumetric_flux * Y[-1]
    face_shape = u_new[-1, 1:-1, 1:-1].shape  

    if field_name == "psi_x":
        # Dirichlet BCs for psi_x: Setting the field to zero on specific walls.
        # Walls normal to z (front/back)
        u_new = u_new.at[1:-1, 1:-1, 0].set(0.0)                            # z = 0 (front wall)
        u_new = u_new.at[1:-1, 1:-1, -1].set(0.0)                           # z = Lz (back wall)
        # Walls normal to y (bottom/top)
        u_new = u_new.at[0, 1:-1, 1:-1].set(0.0)                            # y = 0 (bottom wall)
        u_new = u_new.at[-1, 1:-1, 1:-1].set(0.0)                           # y = Ly (top wall)
        # Walls normal to x (left/right) - also set to zero 
        u_new = u_new.at[1:-1, 0, 1:-1].set(u_new[1:-1,1,1:-1])             # x = 0 (left wall)
        u_new = u_new.at[1:-1, -1, 1:-1].set(u_new[1:-1,-2,1:-1])           # x = Lx (right wall)

    elif field_name == "psi_y":
        # Dirichlet BCs for psi_y: Setting the field to zero on specific walls.
        # Walls normal to x (left/right)
        u_new = u_new.at[1:-1, 0, 1:-1].set(0.0)                            # x = 0 (left wall)
        u_new = u_new.at[1:-1, -1, 1:-1].set(u_new[1:-1,-2,1:-1])           # x = Lx (right wall)

        # Walls normal to z (front/back)
        u_new = u_new.at[1:-1, 1:-1, 0].set(0.0)                            # z = 0 (front wall)
        u_new = u_new.at[1:-1, 1:-1, -1].set(0.0)                           # z = Lz (back wall)
        # Walls normal to y (bottom/top) - also set to zero
        u_new = u_new.at[0, 1:-1, 1:-1].set(u_new[1,1:-1,1:-1])             # y = 0 (bottom wall)
        u_new = u_new.at[-1, 1:-1, 1:-1].set(u_new[-2,1:-1,1:-1])           # y = Ly (top wall)
    elif field_name == "psi_z":
        # Dirichlet BCs for psi_z: Setting the field to zero on specific walls.

        # Walls normal to z (front/back) - also set to zero
        u_new = u_new.at[1:-1, 1:-1, 0].set(u_new[1:-1,1:-1,1])             # z = 0 (front wall)
        u_new = u_new.at[1:-1, 1:-1, -1].set(u_new[1:-1,1:-1,-2])           # z = Lz (back wall)
        # Walls normal to x (left/right)
        u_new = u_new.at[1:-1, 0, 1:-1].set(Y_profile)                      # x = 0 (left wall) INFLOW
        u_new = u_new.at[1:-1, -1, 1:-1].set(u_new[1:-1,-2,1:-1])           # x = Lx (right wall)
        # Walls normal to y (bottom/top)
        u_new = u_new.at[0, 1:-1, 1:-1].set(0.0)    # y = 0 (bottom wall)
        u_new = u_new.at[-1, :, 1:-1].set(1.0)  # y = Ly (top wall)

    # u_new = CornersBC(u_new) # Apply additional corner BCs

    return u_new

def apply_potential_BCs_channel_2ndOrder(u_new, dx, dy, dz, volumetric_flux, H, field_name):

    Ny = u_new.shape[0]
    Y = jnp.linspace(0.0, H, Ny)
    Y_profile = (volumetric_flux * Y[1:-1]).reshape(-1, 1)
    value = volumetric_flux * Y[-1]
    face_shape = u_new[-1, 1:-1, 1:-1].shape  # e.g., (Ny-2, Nz-2)

    if field_name == "psi_x":
        # Dirichlet BCs for psi_x: Setting the field to zero on specific walls.
        # Walls normal to z (front/back)
        u_new = u_new.at[1:-1, 1:-1, 0].set(0.0)                            # z = 0 (front wall)
        u_new = u_new.at[1:-1, 1:-1, -1].set(0.0)                           # z = Lz (back wall)
        # Walls normal to y (bottom/top)
        u_new = u_new.at[0, 1:-1, 1:-1].set(0.0)                            # y = 0 (bottom wall)
        u_new = u_new.at[-1, 1:-1, 1:-1].set(0.0)                           # y = Ly (top wall)
        # Walls normal to x (left/right) - also set to zero
        u_new = u_new.at[1:-1, 0, 1:-1].set( (4.0*u_new[1:-1,1,1:-1] - u_new[1:-1,2,1:-1]) / 3.0)             # x = 0 (left wall)
        u_new = u_new.at[1:-1, -1, 1:-1].set( (4.0*u_new[1:-1,-2,1:-1] - u_new[1:-1,-3,1:-1]) / 3.0)          # x = Lx (right wall)

    elif field_name == "psi_y":
        # Dirichlet BCs for psi_y: Setting the field to zero on specific walls.
        # Walls normal to x (left/right)
        u_new = u_new.at[1:-1, 0, 1:-1].set(0.0)                            # x = 0 (left wall)
        u_new = u_new.at[1:-1, -1, 1:-1].set( (4.0*u_new[1:-1,-2,1:-1] - u_new[1:-1,-3,1:-1]) / 3.0)           # x = Lx (right wall)

        # Walls normal to z (front/back)
        u_new = u_new.at[1:-1, 1:-1, 0].set(0.0)                            # z = 0 (front wall)
        u_new = u_new.at[1:-1, 1:-1, -1].set(0.0)                           # z = Lz (back wall)
        # Walls normal to y (bottom/top) - also set to zero
        u_new = u_new.at[0, 1:-1, 1:-1].set( (4.0*u_new[1,1:-1,1:-1] - u_new[2,1:-1,1:-1]) / 3.0)             # y = 0 (bottom wall)
        u_new = u_new.at[-1, 1:-1, 1:-1].set( (4.0*u_new[-2,1:-1,1:-1] - u_new[-3,1:-1,1:-1]) / 3.0)           # y = Ly (top wall)
    elif field_name == "psi_z":
        # Dirichlet BCs for psi_z: Setting the field to zero on specific walls.

        # Walls normal to z (front/back) - also set to zero
        u_new = u_new.at[1:-1, 1:-1, 0].set( (4.0*u_new[1:-1,1:-1,1] - u_new[1:-1,1:-1,2]) / 3.0)             # z = 0 (front wall)
        u_new = u_new.at[1:-1, 1:-1, -1].set( (4.0*u_new[1:-1,1:-1,-2] - u_new[1:-1,1:-1,-3]) / 3.0)           # z = Lz (back wall)
        # Walls normal to x (left/right)
        u_new = u_new.at[1:-1, 0, 1:-1].set(Y_profile)                      # x = 0 (left wall) INFLOW
        u_new = u_new.at[1:-1, -1, 1:-1].set( (4.0*u_new[1:-1,-2,1:-1] - u_new[1:-1,-3,1:-1]) / 3.0)           # x = Lx (right wall)
        # u_new = u_new.at[1:-1, -1, 1:-1].set(0.0)                         # x = Lx (right wall)
        # Walls normal to y (bottom/top)
        u_new = u_new.at[0, 1:-1, 1:-1].set(0.0)    # y = 0 (bottom wall)
        u_new = u_new.at[-1, 1:-1, 1:-1].set(jnp.full(face_shape, value))  # y = Ly (top wall)
        # u_new = u_new.at[-1, :, :].set(1.0)    # y = H (top wall)

    # u_new = CornersBC(u_new) # Apply additional corner BCs

    return u_new


# POISSON STEP
def poisson_step(u, b, volumetric_flux, H, dx, dy, dz, field_name):
    """
    Performs one iteration of the Jacobi solver for the Poisson equation
    and applies boundary conditions specific to the 'field_name'.

    Args:
        u (jnp.ndarray): The current 3D solution field (Ny, Nx, Nz).
        b (jnp.ndarray): The 3D source term (Ny, Nx, Nz).
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        field_name (str): Identifier for applying specific boundary conditions
                          ("psi_x", "psi_y", "psi_z"). 

    Returns:
        jnp.ndarray: The updated 3D solution field after one iteration and BCs.
    """

    u_updated_interior = (
        + (u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / dy**2
        + (u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1]) / dx**2
        + (u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2]) / dz**2
        - b[1:-1, 1:-1, 1:-1] 
    ) / (
        # Denominator: sum of coefficients from the Laplacian discretization
        2.0 / dx**2 + 2.0 / dy**2 + 2.0 / dz**2
    )

    u_new = u.at[1:-1, 1:-1, 1:-1].set(u_updated_interior)

    # Apply boundary conditions based on the `field_name`.
    u_new = apply_potential_BCs_channel(u_new, dx, dy, dz, volumetric_flux, H, field_name)
    # u_new = apply_potential_BCs_channel_2ndOrder(u_new, dx, dy, dz, volumetric_flux, H, field_name)

    return u_new


def Poisson_solver(u_init, dx, dy, dz, b, volumetric_flux, H, max_iterations, tol=None, field_name=None):
    """
    Solves the 3D Poisson equation iteratively using the Jacobi method with JAX.

    Args:
        u_init (jnp.ndarray): Initial guess for the 3D solution field (Ny, Nx, Nz).
                              Should be a JAX array.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        b (jnp.ndarray): The 3D source term (Ny, Nx, Nz). Should be a JAX array.
        max_iterations (int): Maximum number of iterations for the solver.
        tol (float, optional): Tolerance for convergence (relative L2 norm difference
                               between successive iterations). If None, the solver
                               will run for `max_iterations` regardless of convergence.
        field_name (str, optional): A string identifying the type of field
                                    ("psi_x", "psi_y", "psi_z") to apply
                                    the correct boundary conditions.

    Returns:
        jnp.ndarray: The final 3D solution field after convergence or reaching
                     `max_iterations`.
    """

    initial_diff = jnp.array(tol + 1.0) if tol is not None else jnp.array(float('inf'))
    state = (u_init, 0, initial_diff)

    def cond_fn(state):
        current_u, k, current_diff = state         
        if tol is not None:
            return jnp.logical_and(k < max_iterations, current_diff > tol)
        else:
            return k < max_iterations               

    def body_fn(state):
        current_u, k, _ = state                    
        u_next = poisson_step(current_u, b, volumetric_flux, H, dx, dy, dz, field_name=field_name)
        diff = jnp.linalg.norm(u_next - current_u) / (jnp.linalg.norm(current_u) + 1e-10)
        return (u_next, k + 1, diff) 

    # Execute the iterative solver using `jax.lax.while_loop`.
    # The loop will continue until `cond_fn` returns False.
    final_u, final_k, final_diff = lax.while_loop(cond_fn, body_fn, state)

    return final_u, final_k, final_diff 

