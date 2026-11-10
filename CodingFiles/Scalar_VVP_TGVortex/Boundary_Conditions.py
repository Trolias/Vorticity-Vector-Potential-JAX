import jax
import jax.numpy as jnp
from jax import config
import yaml


try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found. Please ensure it's in the same directory.")
    exit() 

L = float(config["Geometry"]["L"])
H = float(config["Geometry"]["H"])
D = float(config["Geometry"]["D"])
gc = config["ghost_cells"]
Nx = config["Grid"]["Nx"]
Ny = config["Grid"]["Ny"]
Nz = config["Grid"]["Nz"]
dx = L / (Nx)
dy = H / (Ny)
dz = D / (Nz)


def apply_BCs(f, bc_type="Dirichlet", val=0.0, axis=0, position="start"):
    h = (dy, dx, dz)[axis]
    dim = axis

    f = jnp.moveaxis(f, dim, 0)              # bring this axis to the front
    if bc_type == "Periodic":
        f = f.at[:gc].set(f[-2*gc:-gc])    # start ghost <- last gc interior cells
        f = f.at[-gc:].set(f[gc:2*gc])     # end ghost   <- first gc interior cells

    inner = slice(gc, -gc)                   # restrict the OTHER two dims to interior

    if position == "start":
        ghost, mirror_src = slice(0, gc), slice(gc, 2*gc)
    else:  # "end"
        ghost, mirror_src = slice(-gc, None), slice(-2*gc, -gc)

    if bc_type == "Dirichlet":
        mirror = f[mirror_src, inner, inner][::-1]
        f = f.at[ghost, inner, inner].set(2.0*val - mirror)

    elif bc_type == "Neumann":
        mirror = f[mirror_src, inner, inner][::-1]
        sign = -1.0 if position == "start" else 1.0
        f = f.at[ghost, inner, inner].set(sign*h*val + mirror)

    return jnp.moveaxis(f, 0, dim)           # move the axis back

#                                       VECTOR POTENTIAL
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

def apply_potential_bcs_cavity(psi, field_name):
    if field_name == "psi_x":
        # Boundary conditions derived from velocity components based on omega_x = dW/dY - dV/dZ
        psi = apply_BCs(psi,"Dirichlet", 0.0, 2, "start") # Front wall (dZ)
        psi = apply_BCs(psi,"Dirichlet", 0.0, 2, "end")   # Back wall (dZ)
        psi = apply_BCs(psi,"Dirichlet", 0.0, 0, "start") # Bottom wall (dY)
        psi = apply_BCs(psi,'Dirichlet', 0.0, 0, "end")   # Top wall (dY)

        # Normal Component (Dirichlet = 0.0)
        psi = apply_BCs(psi,"Neumann", 0.0, 1, "start")   # Left Wall
        psi = apply_BCs(psi,"Neumann", 0.0, 1, "end")     # Right Wall

    elif field_name == "psi_y":
        # Boundary conditions derived from velocity components based on omega_y = dU/dZ - dW/dX
        psi = apply_BCs(psi,"Dirichlet", 0.0, 2, "start") # Front wall (dZ)
        psi = apply_BCs(psi,"Dirichlet", 0.0, 2, "end")   # Back wall (dZ)
        psi = apply_BCs(psi,"Dirichlet", 0.0, 1, "start")   # Left Wall (dX)
        psi = apply_BCs(psi,"Dirichlet", 0.0, 1, "end")     # Right Wall (dX)

        # Normal Component (Dirichlet = 0.0)
        psi = apply_BCs(psi,"Neumann", 0.0, 0, "start") # Bottom wall (dY)
        psi = apply_BCs(psi,'Neumann', 0.0, 0, "end")   # Top wall (dY)

    elif field_name == "psi_z":
        # Boundary conditions derived from velocity components based on omega_z = dV/dX - dU/dY
        psi = apply_BCs(psi,"Dirichlet", 0.0, 1, "start")   # Left Wall (dX)
        psi = apply_BCs(psi,"Dirichlet", 0.0, 1, "end")     # Right Wall (dX)
        psi = apply_BCs(psi,"Dirichlet", 0.0, 0, "start") # Bottom wall (dY)
        psi = apply_BCs(psi,'Dirichlet', 0.0, 0, "end")   # Top wall (dY)

        # Normal Component (Neumann)
        psi = apply_BCs(psi,"Neumann", 0.0, 2, "start") # Front wall (dZ)
        psi = apply_BCs(psi,"Neumann", 0.0, 2, "end")   # Back wall (dZ)

    return psi


#                                       VELOCITY
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

def apply_velocity_BCs(vel, Uwall, field_name,):

    if field_name == "u":
        vel = apply_BCs(vel, "Dirichlet", 0.0, 1, "start") # Left Wall (x=0)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 1, "end")   # Right Wall (x=Lx)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 0, "start") # Bottom Wall (y=0)
        vel = apply_BCs(vel, "Dirichlet", Uwall, 0, "end") # Top Wall (y=Ly), moving lid
        vel = apply_BCs(vel, "Dirichlet", 0.0, 2, "start") # Front Wall (z=0)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 2, "end")  # Back Wall (z=Lz)

    elif field_name == "v":
        vel = apply_BCs(vel, "Dirichlet", 0.0, 1, "start") # Left Wall (x=0)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 1, "end")   # Right Wall (x=Lx)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 0, "start") # Bottom Wall (y=0)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 0, "end")   # Top Wall (y=Ly)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 2, "start") # Front Wall (z=0)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 2, "end")  # Back Wall (z=Lz)
    elif field_name == "w":
        vel = apply_BCs(vel, "Dirichlet", 0.0, 1, "start") # Left Wall (x=0)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 1, "end")   # Right Wall (x=Lx)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 0, "start") # Bottom Wall (y=0)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 0, "end")   # Top Wall (y=Ly)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 2, "start") # Front Wall (z=0)
        vel = apply_BCs(vel, "Dirichlet", 0.0, 2, "end")  # Back Wall (z=Lz)

    return vel



#                                       VORTICITY
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


def apply_vorticity_bcs(omega, u, v, w, dx, dy, dz, field_name):

    if field_name == "omega_new_x":
        vor_val_front  = - (v[gc:-gc, gc:-gc, gc+1] - v[gc:-gc, gc:-gc, gc-1]) / (2.0*dz)
        vor_val_back   = - (v[gc:-gc, gc:-gc, -gc]  - v[gc:-gc, gc:-gc, -gc-2]) / (2.0*dz)
        vor_val_bottom =   (w[gc+1, gc:-gc, gc:-gc] - w[gc-1, gc:-gc, gc:-gc]) / (2.0*dy)
        vor_val_top    =   (w[-gc, gc:-gc, gc:-gc]  - w[-gc-2, gc:-gc, gc:-gc]) / (2.0*dy)
        omega = apply_BCs(omega, "Dirichlet", vor_val_front,  axis=2, position="start")
        omega = apply_BCs(omega, "Dirichlet", vor_val_back,   axis=2, position="end")
        omega = apply_BCs(omega, "Dirichlet", vor_val_bottom, axis=0, position="start")
        omega = apply_BCs(omega, "Dirichlet", vor_val_top,    axis=0, position="end")
        omega = apply_BCs(omega, "Dirichlet", 0.0, 1, "start")
        omega = apply_BCs(omega, "Dirichlet", 0.0, 1, "end")

    elif field_name == "omega_new_y":
        vor_val_front = (u[gc:-gc, gc:-gc, gc+1] - u[gc:-gc, gc:-gc, gc-1]) / (2.0*dz)
        vor_val_back  = (u[gc:-gc, gc:-gc, -gc]  - u[gc:-gc, gc:-gc, -gc-2]) / (2.0*dz)
        vor_val_left  = - (w[gc:-gc, gc+1, gc:-gc] - w[gc:-gc, gc-1, gc:-gc]) / (2.0*dx)
        vor_val_right = - (w[gc:-gc, -gc, gc:-gc]  - w[gc:-gc, -gc-2, gc:-gc]) / (2.0*dx)
        omega = apply_BCs(omega, "Dirichlet", vor_val_front, axis=2, position="start")
        omega = apply_BCs(omega, "Dirichlet", vor_val_back,  axis=2, position="end")
        omega = apply_BCs(omega, "Dirichlet", vor_val_left,  axis=1, position="start")
        omega = apply_BCs(omega, "Dirichlet", vor_val_right, axis=1, position="end")
        omega = apply_BCs(omega, "Dirichlet", 0.0, 0, "start")
        omega = apply_BCs(omega, "Dirichlet", 0.0, 0, "end")

    elif field_name == "omega_new_z":
        vor_val_left   =   (v[gc:-gc, gc+1, gc:-gc] - v[gc:-gc, gc-1, gc:-gc]) / (2.0*dx)
        vor_val_right  =   (v[gc:-gc, -gc, gc:-gc]  - v[gc:-gc, -gc-2, gc:-gc]) / (2.0*dx)
        vor_val_bottom = - (u[gc+1, gc:-gc, gc:-gc] - u[gc-1, gc:-gc, gc:-gc]) / (2.0*dy)
        vor_val_top    = - (u[-gc, gc:-gc, gc:-gc]  - u[-gc-2, gc:-gc, gc:-gc]) / (2.0*dy)
        omega = apply_BCs(omega, "Dirichlet", vor_val_left,   axis=1, position="start")
        omega = apply_BCs(omega, "Dirichlet", vor_val_right,  axis=1, position="end")
        omega = apply_BCs(omega, "Dirichlet", vor_val_bottom, axis=0, position="start")
        omega = apply_BCs(omega, "Dirichlet", vor_val_top,    axis=0, position="end")
        omega = apply_BCs(omega, "Dirichlet", 0.0, 2, "start")
        omega = apply_BCs(omega, "Dirichlet", 0.0, 2, "end")

    return omega


# ALL BCs at ONCE
def apply_all_velocity_bcs(u, v, w, Uwall):
    return (apply_velocity_BCs(u, Uwall, "u"),
            apply_velocity_BCs(v, Uwall, "v"),
            apply_velocity_BCs(w, Uwall, "w"))


def apply_all_vorticity_bcs(omega_x, omega_y, omega_z, u, v, w, dx, dy, dz):
    return (apply_vorticity_bcs(omega_x, u, v, w, dx, dy, dz, "omega_new_x"),
            apply_vorticity_bcs(omega_y, u, v, w, dx, dy, dz, "omega_new_y"),
            apply_vorticity_bcs(omega_z, u, v, w, dx, dy, dz, "omega_new_z"))



#                                       TAYLOR GREEN VORTEX
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
def apply_potential_bcs_TG(psi, field_name,):
    if field_name == "psi_x":
        psi = apply_BCs(psi,"Periodic", 0.0, 0, None,) # Front wall (dZ)
        psi = apply_BCs(psi,"Periodic", 0.0, 1, None,)   # Back wall (dZ)
        psi = apply_BCs(psi,"Periodic", 0.0, 2, None,) # Bottom wall (dY)


    elif field_name == "psi_y":
        psi = apply_BCs(psi,"Periodic", 0.0, 0, None,) # Front wall (dZ)
        psi = apply_BCs(psi,"Periodic", 0.0, 1, None,)   # Back wall (dZ)
        psi = apply_BCs(psi,"Periodic", 0.0, 2, None,) # Bottom wall (dY)

    elif field_name == "psi_z":
        psi = apply_BCs(psi,"Periodic", 0.0, 0, None,) # Front wall (dZ)
        psi = apply_BCs(psi,"Periodic", 0.0, 1, None,)   # Back wall (dZ)
        psi = apply_BCs(psi,"Periodic", 0.0, 2, None,) # Bottom wall (dY)
    return psi


def apply_velocity_BCs_TG(vel, Uwall, field_name,):

    if field_name == "u":
        vel = apply_BCs(vel, "Periodic", 0.0, 0, None,) # Left Wall (x=0)
        vel = apply_BCs(vel, "Periodic", 0.0, 1, None,)   # Right Wall (x=Lx)
        vel = apply_BCs(vel, "Periodic", 0.0, 2, None,) # Bottom Wall (y=0)

    elif field_name == "v":
        vel = apply_BCs(vel, "Periodic", 0.0, 0, None,) # Left Wall (x=0)
        vel = apply_BCs(vel, "Periodic", 0.0, 1, None,)   # Right Wall (x=Lx)
        vel = apply_BCs(vel, "Periodic", 0.0, 2, None,) # Bottom Wall (y=0)

    elif field_name == "w":
        vel = apply_BCs(vel, "Periodic", 0.0, 0, None,) # Left Wall (x=0)
        vel = apply_BCs(vel, "Periodic", 0.0, 1, None,)   # Right Wall (x=Lx)
        vel = apply_BCs(vel, "Periodic", 0.0, 2, None,) # Bottom Wall (y=0)

    return vel
    

def apply_vorticity_bcs_TG(omega, field_name):


    if field_name == "omega_new_x":
        omega = apply_BCs(omega, "Periodic", 0.0,  axis=0, position=None,)
        omega = apply_BCs(omega, "Periodic", 0.0,   axis=1, position=None,)
        omega = apply_BCs(omega, "Periodic", 0.0, axis=2, position=None, )

    elif field_name == "omega_new_y":
        omega = apply_BCs(omega, "Periodic", 0.0,  axis=0, position=None, )
        omega = apply_BCs(omega, "Periodic", 0.0,   axis=1, position=None,)
        omega = apply_BCs(omega, "Periodic", 0.0, axis=2, position=None, )


    elif field_name == "omega_new_z":
        omega = apply_BCs(omega, "Periodic", 0.0,  axis=0, position=None,)
        omega = apply_BCs(omega, "Periodic", 0.0,   axis=1, position=None,)
        omega = apply_BCs(omega, "Periodic", 0.0, axis=2, position=None, )

    return omega

def apply_all_velocity_bcs_TG(u, v, w, Uwall):
    return (apply_velocity_BCs_TG(u, Uwall, "u"),
            apply_velocity_BCs_TG(v, Uwall, "v"),
            apply_velocity_BCs_TG(w, Uwall, "w"))


def apply_all_vorticity_bcs_TG(omega_x, omega_y, omega_z, u, v, w, dx, dy, dz):
    return (apply_vorticity_bcs_TG(omega_x, "omega_new_x"),
            apply_vorticity_bcs_TG(omega_y, "omega_new_y"),
            apply_vorticity_bcs_TG(omega_z, "omega_new_z"))

def apply_all_potential_bcs_TG(psi_x, psi_y, psi_z):
    return (apply_potential_bcs_TG(psi_x, "psi_x"),
            apply_potential_bcs_TG(psi_y, "psi_y"),
            apply_potential_bcs_TG(psi_z, "psi_z"))

