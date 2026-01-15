import jax
import jax.numpy as jnp
from jax import jit
import os
from JAX_Numerical_Schemes_3D import*
from jax import config
import numpy as np
from functools import partial
config.update("jax_enable_x64", True) 
import pickle

def CornersBC(field): # Use the field for correction near the corners
 
    updated_field = field 

    # FRONT PLANE (z=0) corners
    # Corner (0, 0, 0) - Bottom-Left-Front
    updated_field = updated_field.at[0, 0, 0].set((field[1, 0, 0] + field[0, 1, 0] + field[0, 0, 1]) / 3.0)
    updated_field = updated_field.at[1, 1, 1].set((field[1, 0, 0] + field[0, 1, 0] + field[0, 0, 1]) / 3.0)

    # Corner (0, Nx-1, 0) - Bottom-Right-Front
    updated_field = updated_field.at[0, -1, 0].set((field[1, -1, 0] + field[0, -2, 0] + field[0, -1, 1]) / 3.0)
    updated_field = updated_field.at[1, -2, 1].set((field[1, -1, 0] + field[0, -2, 0] + field[0, -1, 1]) / 3.0)

    # Corner (Ny-1, Nx-1, 0) - Top-Right-Front
    updated_field = updated_field.at[-1, -1, 0].set((field[-2, -1, 0] + field[-1, -2, 0] + field[-1, -1, 1]) / 3.0)
    updated_field = updated_field.at[-2, -2, 1].set((field[-2, -1, 0] + field[-1, -2, 0] + field[-1, -1, 1]) / 3.0)

    # Corner (Ny-1, 0, 0) - Top-Left-Front
    updated_field = updated_field.at[-1, 0, 0].set((field[-2, 0, 0] + field[-1, 1, 0] + field[-1, 0, 1]) / 3.0)
    updated_field = updated_field.at[-2, 1, 1].set((field[-2, 0, 0] + field[-1, 1, 0] + field[-1, 0, 1]) / 3.0)


    # BACK PLANE (z=Nz-1) corners
    # Corner (0, 0, Nz-1) - Bottom-Left-Back
    updated_field = updated_field.at[0, 0, -1].set((field[1, 0, -1] + field[0, 1, -1] + field[0, 0, -2]) / 3.0)
    updated_field = updated_field.at[1, 1, -2].set((field[1, 0, -1] + field[0, 1, -1] + field[0, 0, -2]) / 3.0)

    # Corner (0, Nx-1, Nz-1) - Bottom-Right-Back
    updated_field = updated_field.at[0, -1, -1].set((field[1, -1, -1] + field[0, -2, -1] + field[0, -1, -2]) / 3.0)
    updated_field = updated_field.at[1, -2, -2].set((field[1, -1, -1] + field[0, -2, -1] + field[0, -1, -2]) / 3.0)

    # Corner (Ny-1, Nx-1, Nz-1) - Top-Right-Back
    updated_field = updated_field.at[-1, -1, -1].set((field[-2, -1, -1] + field[-1, -2, -1] + field[-1, -1, -2]) / 3.0)
    updated_field = updated_field.at[-2, -2, -2].set((field[-2, -1, -1] + field[-1, -2, -1] + field[-1, -1, -2]) / 3.0)

    # Corner (Ny-1, 0, Nz-1) - Top-Left-Back
    updated_field = updated_field.at[-1, 0, -1].set((field[-2, 0, -1] + field[-1, 1, -1] + field[-1, 0, -2]) / 3.0)
    updated_field = updated_field.at[-2, 1, -2].set((field[-2, 0, -1] + field[-1, 1, -1] + field[-1, 0, -2]) / 3.0)


    # Top Left Corners Interior
    updated_field = updated_field.at[-1, 0, 1:-1].set( 0.5 * (field[-1,1,1:-1] + field[-2,0,1:-1])  )
    updated_field = updated_field.at[-2, 1, 1:-1].set( 0.5 * (field[-1,1,1:-1] + field[-2,0,1:-1])  )

    updated_field = updated_field.at[-1, 1:-1, 0].set( 0.5 * (field[-1,1:-1,1] + field[-2,1:-1,0])  )
    updated_field = updated_field.at[-2, 1:-1, 1].set( 0.5 * (field[-1,1:-1,1] + field[-2,1:-1,0])  )

    # Top right Corners Interior
    updated_field = updated_field.at[-1, -1,1:-1].set( 0.5 * (field[-2,-1,1:-1]+ field[-1,-2,1:-1] ) )
    updated_field = updated_field.at[-2, -2,1:-1].set( 0.5 * (field[-2,-1,1:-1]+ field[-1,-2,1:-1] ) )

    updated_field = updated_field.at[-1, 1:-1,-1].set( 0.5 * (field[-2,1:-1,-1]+ field[-1,1:-1,-2] ) )
    updated_field = updated_field.at[-2, 1:-1,-2].set( 0.5 * (field[-2,1:-1,-1]+ field[-1,1:-1,-2] ) )

    # Bottom Letf
    updated_field = updated_field.at[0, 0, 1:-1].set( 0.5 * (field[0,1,1:-1] + field[1,0,1:-1])  )
    updated_field = updated_field.at[1, 1, 1:-1].set( 0.5 * (field[0,1,1:-1] + field[1,0,1:-1])  )

    updated_field = updated_field.at[0, 1:-1, 0].set( 0.5 * (field[0,1:-1,1] + field[1,1:-1,0])  )
    updated_field = updated_field.at[1, 1:-1, 1].set( 0.5 * (field[0,1:-1,1] + field[1,1:-1,0])  )

    # Bottom Right
    updated_field = updated_field.at[0, -1,1:-1].set( 0.5 * (field[1,-1,1:-1]+ field[0,-2,1:-1] ) )
    updated_field = updated_field.at[1, -2,1:-1].set( 0.5 * (field[1,-1,1:-1]+ field[0,-2,1:-1] ) )

    updated_field = updated_field.at[0, 1:-1,-1].set( 0.5 * (field[1,1:-1,-1]+ field[0,1:-1,-2] ) )
    updated_field = updated_field.at[1, 1:-1,-2].set( 0.5 * (field[1,1:-1,-1]+ field[0,1:-1,-2] ) )

    return updated_field


def calculate_divergence(u_field, v_field, w_field, dx, dy, dz):
    """
    Calculates the divergence of a 3D vector field (u, v, w) using central differences.
    The divergence is computed for the interior points of the domain.
    """
    div_u = jnp.zeros_like(u_field)
    du_dx = Grad_x(u_field, dx)
    dv_dy = Grad_y(v_field, dy)
    dw_dz = Grad_z(w_field, dz)

    interior_divergence = du_dx[1:-1, 1:-1, 1:-1] + \
                          dv_dy[1:-1, 1:-1, 1:-1] + \
                          dw_dz[1:-1, 1:-1, 1:-1]

    return div_u.at[1:-1,1:-1,1:-1].set(interior_divergence)


def save_to_vtk_fast(psi_x, psi_y, psi_z,
                     omega_x, omega_y, omega_z,
                     u, v, w,
                     divergence, vel_div,
                     dx, dy, dz,
                     filename):
    """
    Export ψ, ω, velocity, and divergence to VTK with correct axis orientation.
    Input arrays are shaped (Ny, Nx, Nz)  ↔ axes (y, x, z).
    """

    psi_x, psi_y, psi_z = map(np.asarray, (psi_x, psi_y, psi_z))
    omega_x, omega_y, omega_z = map(np.asarray, (omega_x, omega_y, omega_z))
    u, v, w = map(np.asarray, (u, v, w))
    divergence = np.asarray(divergence)
    vel_div = np.asarray(vel_div)

    Ny, Nx, Nz = psi_x.shape
    total_pts = Nx * Ny * Nz

    def to_z_y_x(field):
        return np.transpose(field, (2, 0, 1))

    psi      = np.stack([to_z_y_x(psi_x),
                         to_z_y_x(psi_y),
                         to_z_y_x(psi_z)], axis=-1).reshape(-1, 3)
    omega    = np.stack([to_z_y_x(omega_x),
                         to_z_y_x(omega_y),
                         to_z_y_x(omega_z)], axis=-1).reshape(-1, 3)
    velocity = np.stack([to_z_y_x(u),
                         to_z_y_x(v),
                         to_z_y_x(w)], axis=-1).reshape(-1, 3)
    div_flat = to_z_y_x(divergence).reshape(-1)
    vel_div_flat = to_z_y_x(vel_div).reshape(-1)


    # write VTK
    os.makedirs("vtk_output", exist_ok=True)
    full_path = os.path.join("vtk_output", filename)
    with open(full_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vector & scalar fields\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {dx} {dy} {dz}\n")
        f.write(f"POINT_DATA {total_pts}\n\n")

        def write_vec(name, data):
            f.write(f"VECTORS {name} float\n")
            np.savetxt(f, data, fmt="%.6e %.6e %.6e")
            f.write("\n")

        def write_scalar(name, data):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, data[:, None], fmt="%.6e")
            f.write("\n")

        write_vec("psi",      psi)
        write_vec("omega",    omega)
        write_vec("velocity", velocity)
        write_scalar("vorticity_divergence", div_flat)
        write_scalar("velocity_divergence", vel_div_flat)


    print(f"Saved VTK file: {full_path}")


def adaptive_time_step(u,v,w,dx,dy,dz,nu, safety: float = 0.95): 

    #Inner Rule 
    H = (1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)
    Inner_dt = 1.0 / (2.0*nu*H)

    #Outer Rule
    Outer_dt = 2.0 * nu / ( u**2 + v**2 + w**2 + 1e-09 )
    dt_local =  jnp.minimum(Inner_dt,Outer_dt)
    dt_global = jnp.min(dt_local)

    return safety * dt_global


# -------------------------------------------------------------------------------------------------------------------------------

# Function to load the data saved by { save_checkpoint_binary } function 
def load_checkpoint_binary(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    return (
        jnp.array(state["psi_x"]),
        jnp.array(state["psi_y"]),
        jnp.array(state["psi_z"]),
        jnp.array(state["omega_x"]),
        jnp.array(state["omega_y"]),
        jnp.array(state["omega_z"]),
        jnp.array(state["u"]),
        jnp.array(state["v"]),
        jnp.array(state["w"]),
        state["iteration"],
        state["conv_error"]
    )

# Save the data in a format that can be used as initialisation for further runs
def save_checkpoint_binary(path, psi_x, psi_y, psi_z,
                           omega_x, omega_y, omega_z,
                           u, v, w,
                           iteration, conv_error):
    state = {
        "psi_x": np.array(psi_x),
        "psi_y": np.array(psi_y),
        "psi_z": np.array(psi_z),
        "omega_x": np.array(omega_x),
        "omega_y": np.array(omega_y),
        "omega_z": np.array(omega_z),
        "u": np.array(u),
        "v": np.array(v),
        "w": np.array(w),
        "iteration": iteration,
        "conv_error": conv_error
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)

