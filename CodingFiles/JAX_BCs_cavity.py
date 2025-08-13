import jax
from jax import jit
from JAX_Utils_3D import CornersBC
from functools import partial
from JAX_Numerical_Schemes_3D import*

@partial(jit, static_argnames=('field_name',))
def apply_velocity_BCs(vel, Uwall, field_name):
    """Applies velocity boundary conditions to a field based on its component name.

    Args:
        vel (jnp.ndarray): The velocity component field (u, v, or w).
        Uwall (float): The velocity of the moving wall.
        field_name (str): The name of the velocity component ("u", "v", or "w").

    Returns:
        jnp.ndarray: The velocity field with applied boundary conditions.
    """
    if field_name == "u":
        vel = vel.at[1:-1, 0, 1:-1].set(0.0)    # Left Wall (x=0)
        vel = vel.at[1:-1, -1, 1:-1].set(0.0)   # Right Wall (x=Lx)
        vel = vel.at[0, 1:-1, 1:-1].set(0.0)    # Bottom Wall (y=0)
        vel = vel.at[-1, 1:-1, 1:-1].set(Uwall) # Top Wall (y=Ly), moving lid
        vel = vel.at[1:-1, 1:-1, 0].set(0.0)    # Front Wall (z=0)
        vel = vel.at[1:-1, 1:-1, -1].set(0.0)   # Back Wall (z=Lz)
    elif field_name == "v":
        vel = vel.at[1:-1, 0, 1:-1].set(0.0)
        vel = vel.at[1:-1, -1, 1:-1].set(0.0)
        vel = vel.at[0, 1:-1, 1:-1].set(0.0)
        vel = vel.at[-1, 1:-1, 1:-1].set(0.0)
        vel = vel.at[1:-1, 1:-1, 0].set(0.0)
        vel = vel.at[1:-1, 1:-1, -1].set(0.0)
    elif field_name == "w":
        vel = vel.at[1:-1, 0, 1:-1].set(0.0)
        vel = vel.at[1:-1, -1, 1:-1].set(0.0)
        vel = vel.at[0, 1:-1, 1:-1].set(0.0)
        vel = vel.at[-1, 1:-1, 1:-1].set(0.0)
        vel = vel.at[1:-1, 1:-1, 0].set(0.0)
        vel = vel.at[1:-1, 1:-1, -1].set(0.0)

    # vel = CornersBC(vel) # Apply corner BCs (If want Corner correction then uncomment)
    return vel

@partial(jit, static_argnames=('field_name',))
def apply_vorticity_bcs(omega, u, v, w, dx, dy, dz, Uwall, field_name):
    """Applies vorticity boundary conditions based on the field component and velocities.

    Args:
        omega (jnp.ndarray): The vorticity component field.
        u, v, w (jnp.ndarray): Velocity components used for derivative-based BCs.
        dx, dy, dz (float): Grid spacings.
        Uwall (float): The velocity of the moving wall (used for omega_z top wall).
        field name (static argument): Declare the field i.e, omega_X for prescribing suitable BCs

    Returns:
        jnp.ndarray: The vorticity field with applied boundary conditions.
    """
    if field_name == "omega_new_x":
        # Boundary conditions derived from velocity components based on omega_x = dW/dY - dV/dZ
        omega = omega.at[1:-1, 1:-1, 0].set(-v[1:-1, 1:-1, 1] / dz) # Front wall (dZ)
        omega = omega.at[1:-1, 1:-1, -1].set(v[1:-1, 1:-1, -2] / dz) # Back wall (dZ)

        omega = omega.at[0, 1:-1, 1:-1].set(w[1, 1:-1, 1:-1] / dy) # Bottom Wall (dY)
        omega = omega.at[-1, 1:-1, 1:-1].set(-w[-2, 1:-1, 1:-1] / dy) # Top Wall (dY)

        # Normal Component (Dirichlet = 0.0)
        omega = omega.at[1:-1, 0, 1:-1].set(0.0) # Left Wall
        omega = omega.at[1:-1, -1, 1:-1].set(0.0) # Right Wall

    elif field_name == "omega_new_y":
        # Boundary conditions derived from velocity components based on omega_y = dU/dZ - dW/dX
        omega = omega.at[1:-1, 1:-1, 0].set(u[1:-1, 1:-1, 1] / dz) # Front Wall (dZ)
        omega = omega.at[1:-1, 1:-1, -1].set(-u[1:-1, 1:-1, -2] / dz) # Back Wall (dZ)

        omega = omega.at[1:-1, 0, 1:-1].set(-w[1:-1, 1, 1:-1] / dx) # Left Wall (dX)
        omega = omega.at[1:-1, -1, 1:-1].set(w[1:-1, -2, 1:-1] / dx) # Right Wall (dX)

        # Normal Component (Dirichlet = 0.0)
        omega = omega.at[0, 1:-1, 1:-1].set(0.0) # Bottom Wall
        omega = omega.at[-1, 1:-1, 1:-1].set(0.0) # Top Wall

    elif field_name == "omega_new_z":
        # Boundary conditions derived from velocity components based on omega_z = dV/dX - dU/dY
        omega = omega.at[1:-1, 0, 1:-1].set(v[1:-1, 1, 1:-1] / dx) # Left Wall (dX)
        omega = omega.at[1:-1, -1, 1:-1].set(-v[1:-1, -2, 1:-1] / dx) # Right Wall (dX)

        omega = omega.at[0, 1:-1, 1:-1].set(-u[1, 1:-1, 1:-1] / dy) # Bottom Wall (dY)
        omega = omega.at[-1, :, :].set(-(Uwall - u[-2, :, :]) / dy) # Top Wall (dY)

        # Normal Component (Neumann BC)
        omega = omega.at[1:-1, 1:-1, 0].set(0.0) # Front Wall
        omega = omega.at[1:-1, 1:-1, -1].set(0.0) # Back Wall

    omega = CornersBC(omega) # Apply corner correction BCs 
    return omega


@partial(jit, static_argnames=('field_name',))
def apply_vorticity_bcs_2ndOrder(omega, u, v, w, dx, dy, dz, Uwall, field_name):
    """
        Same as before but 2nd Order (Not tested yet)
    """
    if field_name == "omega_new_x":
        # Boundary conditions derived from velocity components based on omega_x = dW/dY - dV/dZ
        omega = omega.at[1:-1, 1:-1, 0].set( - (4.0*v[1:-1, 1:-1, 1] - v[1:-1,1:-1,2] ) / (2.0*dz) ) # Front wall (dZ)
        omega = omega.at[1:-1, 1:-1, -1].set( - (v[1:-1,1:-1,-3] - 4.0*v[1:-1, 1:-1, -2] ) / (2.0*dz) ) # Back wall (dZ)
        omega = omega.at[0, 1:-1, 1:-1].set( (4.0*w[1, 1:-1, 1:-1] - w[2,1:-1,1:-1]) / (2.0*dy) ) # Bottom Wall (dY)
        omega = omega.at[-1, 1:-1, 1:-1].set( (w[-3,1:-1,1:-1] - 4.0*w[-2, 1:-1, 1:-1]) / (2.0*dy) ) # Top Wall (dY)
        # Normal Component (Dirichlet = 0.0)
        omega = omega.at[1:-1, 0, 1:-1].set(0.0) # Left Wall
        omega = omega.at[1:-1, -1, 1:-1].set(0.0) # Right Wall

    elif field_name == "omega_new_y":
        # Boundary conditions derived from velocity components based on omega_y = dU/dZ - dW/dX
        omega = omega.at[1:-1, 1:-1, 0].set( (4.0*u[1:-1, 1:-1, 1] - u[1:-1,1:-1,2]) / (2.0*dz) ) # Front Wall (dZ)
        omega = omega.at[1:-1, 1:-1, -1].set( (u[1:-1,1:-1,-3] - 4.0*u[1:-1, 1:-1, -2]) / (2.0*dz) ) # Back Wall (dZ)
        omega = omega.at[1:-1, 0, 1:-1].set( - (4.0*w[1:-1, 1, 1:-1] - w[1:-1,2,1:-1]) / (2.0*dx) ) # Left Wall (dX)
        omega = omega.at[1:-1, -1, 1:-1].set( - (w[1:-1,-3,1:-1] - 4.0*w[1:-1, -2, 1:-1]) / (2.0*dx) ) # Right Wall (dX)
        # Normal Component (Dirichlet = 0.0)
        omega = omega.at[0, 1:-1, 1:-1].set(0.0) # Bottom Wall
        omega = omega.at[-1, 1:-1, 1:-1].set(0.0) # Top Wall

    elif field_name == "omega_new_z":
        # Boundary conditions derived from velocity components based on omega_z = dV/dX - dU/dY
        omega = omega.at[1:-1, 0, 1:-1].set( (4.0*v[1:-1, 1, 1:-1] - v[1:-1,2,1:-1]) / (2.0*dx) ) # Left Wall (dX)
        omega = omega.at[1:-1, -1, 1:-1].set( (v[1:-1,-3,1:-1] - 4.0*v[1:-1, -2, 1:-1]) / (2.0*dx) ) # Right Wall (dX)
        omega = omega.at[0, 1:-1, 1:-1].set( - (4.0*u[1, 1:-1, 1:-1] - u[2,1:-1,1:-1]) / (2.0*dy) ) # Bottom Wall (dY)
        # Special BC for the moving lid: dU/dY on top wall
        omega = omega.at[-1, 1:-1, 1:-1].set(-(3.0*Uwall - 4.0*u[-2, 1:-1, 1:-1] + u[-3,1:-1,1:-1]) / (2.0*dy) ) # Top Wall (dY)
        # Normal Component (Neumann)
        omega = omega.at[1:-1, 1:-1, 0].set(0.0) # Front Wall
        omega = omega.at[1:-1, 1:-1, -1].set(0.0) # Back Wall

    omega = CornersBC(omega) # Apply corner BCs 
    return omega