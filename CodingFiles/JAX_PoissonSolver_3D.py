import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from JAX_Utils_3D import CornersBC
from functools import partial
from jax import config
config.update("jax_enable_x64", True) # Enable 64-bit precision for better accuracy in simulations



# POISSON STEP
@partial(jit, static_argnames=('field_name',))
def poisson_step(u, b, dx, dy, dz, field_name):
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
                          ("psi_x", "psi_y", "psi_z"). This should be a static argument
                          to the `Poisson` function for efficient JAX compilation.

    Returns:
        jnp.ndarray: The updated 3D solution field after one iteration and BCs.
    """
    u_updated_interior = (
        + (u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / dy**2
        + (u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1]) / dx**2
        + (u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2]) / dz**2
        - b[1:-1, 1:-1, 1:-1]  
    ) / (
        2.0 / dx**2 + 2.0 / dy**2 + 2.0 / dz**2
    )

    u_new = u.at[1:-1, 1:-1, 1:-1].set(u_updated_interior)
    u_new = apply_potential_bcs_cavity(u_new,dx,dy,dz,field_name)

    return u_new


@partial(jit, static_argnames=('field_name', 'max_iterations', 'tol'))
def Poisson_solver(u_init, dx, dy, dz, b, max_iterations, tol=None, field_name=None):
    """
    Solves the 3D Poisson equation iteratively using the Jacobi method with JAX.

    Args:
        u_init (jnp.ndarray): Initial guess for the 3D solution field (Ny, Nx, Nz).
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        b (jnp.ndarray): The 3D source term (Ny, Nx, Nz). 
        max_iterations (int): Maximum number of iterations for the solver.
        tol (float, optional): Tolerance for convergence (relative L2 norm difference
                               between successive iterations). If None, the solver
                               will run for `max_iterations` regardless of convergence.
        field_name (str, optional): A string identifying the type of field
                                    ("psi_x", "psi_y", "psi_z") to apply
                                    the correct boundary conditions. This must be
                                    a **static argument** for JAX compilation.

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
        u_next = poisson_step(current_u, b, dx, dy, dz, field_name)
        diff = jnp.linalg.norm(u_next - current_u) / (jnp.linalg.norm(current_u) + 1e-10)
        return (u_next, k + 1, diff) 

    final_u, final_k, final_diff = lax.while_loop(cond_fn, body_fn, state)

    return final_u, final_k, final_diff 


@partial(jit, static_argnames=('field_name',))
def apply_potential_bcs_cavity(psi, dx, dy, dz, field_name):
    """Applies vorticity boundary conditions based on the field component and velocities.

    Args:
        omega (jnp.ndarray): The vorticity component field.
        u, v, w (jnp.ndarray): Velocity components used for derivative-based BCs.
        dx, dy, dz (float): Grid spacings.
        Uwall (float): The velocity of the moving wall (used for omega_z top wall).
        field_name (str): The name of the vorticity component ("omega_new_x", etc.).
                          This is a static argument for JAX compilation.

    Returns:
        jnp.ndarray: The vorticity field with applied boundary conditions.
    """
    if field_name == "psi_x":
        # Boundary conditions derived from velocity components based on omega_x = dW/dY - dV/dZ
        psi = psi.at[1:-1,1:-1,0].set(0.0)          # Front wall (dZ)
        psi = psi.at[1:-1, 1:-1, -1].set(0.0)       # Back wall (dZ)
        psi = psi.at[0, 1:-1, 1:-1].set(0.0)        # Bottom Wall (dY)
        psi = psi.at[-1, 1:-1, 1:-1].set(0.0)       # Top Wall (dY)
        # Normal Component (Dirichlet = 0.0)
        psi = psi.at[1:-1, 0, 1:-1].set(psi[1:-1,1,1:-1])   # Left Wall
        psi = psi.at[1:-1, -1, 1:-1].set(psi[1:-1,-2,1:-1]) # Right Wall

    elif field_name == "psi_y":
        # Boundary conditions derived from velocity components based on omega_y = dU/dZ - dW/dX
        psi = psi.at[1:-1, 1:-1, 0].set(0.0)        # Front Wall (dZ)
        psi = psi.at[1:-1, 1:-1, -1].set(0.0)       # Back Wall (dZ)
        psi = psi.at[1:-1, 0, 1:-1].set(0.0)        # Left Wall (dX)
        psi = psi.at[1:-1, -1, 1:-1].set(0.0)       # Right Wall (dX)
        # Normal Component (Dirichlet = 0.0)
        psi = psi.at[0, 1:-1, 1:-1].set(psi[1,1:-1,1:-1]) # Bottom Wall
        psi = psi.at[-1, 1:-1, 1:-1].set(psi[-2,1:-1,1:-1]) # Top Wall

    elif field_name == "psi_z":
        # Boundary conditions derived from velocity components based on omega_z = dV/dX - dU/dY
        psi = psi.at[1:-1, 0, 1:-1].set(0.0)        # Left Wall (dX)
        psi = psi.at[1:-1, -1, 1:-1].set(0.0)       # Right Wall (dX)
        psi = psi.at[0, 1:-1, 1:-1].set(0.0)        # Bottom Wall (dY)
        psi = psi.at[-1, 1:-1, 1:-1].set(0.0)       # Top Wall (dY)
        # Normal Component (Neumann)
        psi = psi.at[1:-1, 1:-1, 0].set(psi[1:-1,1:-1,1])   # Front Wall
        psi = psi.at[1:-1, 1:-1, -1].set(psi[1:-1,1:-1,-2]) # Back Wall

    # psi = CornersBC(psi) # Apply corner BCs (Uncomment if desire corner correction)
    return psi

@partial(jit, static_argnames=('field_name',))
def apply_potential_bcs_cavity_2ndOrder(psi, dx, dy, dz, field_name):
    """
        Same as before but 2nd-order (NOT TESTED)
    """
    if field_name == "psi_x":
        # Boundary conditions derived from velocity components based on omega_x = dW/dY - dV/dZ
        psi = psi.at[1:-1,1:-1,0].set(0.0)          # Front wall (dZ)
        psi = psi.at[1:-1, 1:-1, -1].set(0.0)       # Back wall (dZ)
        psi = psi.at[0, 1:-1, 1:-1].set(0.0)        # Bottom Wall (dY)
        psi = psi.at[-1, 1:-1, 1:-1].set(0.0)       # Top Wall (dY)
        # Normal Component (Dirichlet = 0.0, as per your original code)
        psi = psi.at[1:-1, 0, 1:-1].set( (4.0*psi[1:-1,1,1:-1] - psi[1:-1,2,1:-1]) / 3.0)   # Left Wall
        psi = psi.at[1:-1, -1, 1:-1].set( (-psi[1:-1,-3,1:-1] + 4.0*psi[1:-1,-2,1:-1]) / 3.0) # Right Wall

    elif field_name == "psi_y":
        # Boundary conditions derived from velocity components based on omega_y = dU/dZ - dW/dX
        psi = psi.at[1:-1, 1:-1, 0].set(0.0)        # Front Wall (dZ)
        psi = psi.at[1:-1, 1:-1, -1].set(0.0)       # Back Wall (dZ)
        psi = psi.at[1:-1, 0, 1:-1].set(0.0)        # Left Wall (dX)
        psi = psi.at[1:-1, -1, 1:-1].set(0.0)       # Right Wall (dX)
        # Normal Component (Dirichlet = 0.0)
        psi = psi.at[0, 1:-1, 1:-1].set( (4.0*psi[1,1:-1,1:-1] - psi[2,1:-1,1:-1]) / 3.0) # Bottom Wall
        psi = psi.at[-1, 1:-1, 1:-1].set( (-psi[-3,1:-1,1:-1] + 4.0*psi[-2,1:-1,1:-1]) / 3.0) # Top Wall

    elif field_name == "psi_z":
        # Boundary conditions derived from velocity components based on omega_z = dV/dX - dU/dY
        psi = psi.at[1:-1, 0, 1:-1].set(0.0)        # Left Wall (dX)
        psi = psi.at[1:-1, -1, 1:-1].set(0.0)       # Right Wall (dX)
        psi = psi.at[0, 1:-1, 1:-1].set(0.0)        # Bottom Wall (dY)
        psi = psi.at[-1, 1:-1, 1:-1].set(0.0)       # Top Wall (dY)
        # Normal Component (Neumann-like, based on original NumPy uncommented lines)
        psi = psi.at[1:-1, 1:-1, 0].set( (4.0*psi[1:-1,1:-1,1] - psi[1:-1,1:-1,2]) / 3.0)   # Front Wall
        psi = psi.at[1:-1, 1:-1, -1].set( (-psi[1:-1,1:-1,-3] + 4.0*psi[1:-1,1:-1,-2]) / 3.0) # Back Wall

    psi = CornersBC(psi) # Apply corner BCs 
    return psi