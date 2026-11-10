import jax
import jax.numpy as jnp
import numpy as np
import pickle
from nums import grad, grad4
import yaml
from jax import config

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found. Please ensure it's in the same directory.")
    exit() 
gc = config['ghost_cells']

def _to_uniform_1d(field, h, gc, L, n, axis):
    """
    Interpolate a cell-centered field (gc ghost cells on both sides along
    `axis`) from non-uniform cell centers onto a uniform grid of n points
    spanning [0, L] along that axis. Works for any gc, any array rank.
    """
    field = np.moveaxis(field, axis, 0)
    Ntot = field.shape[0]
    n_interior = Ntot - 2*gc
    x_c = (np.arange(n_interior) + 0.5) * h

    # ghost cell ADJACENT to each wall (not the outermost ghost layer)
    wall_start = 0.5*(field[gc-1]      + field[gc])          # start wall value
    wall_end   = 0.5*(field[-gc]       + field[-gc-1])        # end wall value

    interior = field[gc:gc+n_interior]

    x_ext = np.concatenate([[0.0], x_c, [L]])
    vals_ext = np.concatenate([wall_start[None, ...], interior, wall_end[None, ...]], axis=0)

    x_query = np.linspace(0, L, n)

    other_shape = vals_ext.shape[1:]
    vals_flat = vals_ext.reshape(vals_ext.shape[0], -1)
    out_flat = np.empty((n, vals_flat.shape[1]))
    for j in range(vals_flat.shape[1]):
        out_flat[:, j] = np.interp(x_query, x_ext, vals_flat[:, j])

    out = out_flat.reshape((n,) + other_shape)
    return np.moveaxis(out, 0, axis)


def to_uniform_grid_3d(field, dx, dy, dz, gc, Lx, Ly, Lz, nx, ny, nz):
    """Sequential 1D interpolation along x, then y, then z == trilinear
    interpolation from the cell-centered FVM grid onto a uniform grid."""
    field = _to_uniform_1d(field, dx, gc, Lx, nx, axis=1)   # x -> array dim 1
    field = _to_uniform_1d(field, dy, gc, Ly, ny, axis=0)   # y -> array dim 0
    field = _to_uniform_1d(field, dz, gc, Lz, nz, axis=2)   # z -> array dim 2
    return field


def write_tecplot_3d(filename, psi_x, psi_y, psi_z,
                      omega_x, omega_y, omega_z,
                      u, v, w,
                      div_omega, div_vel,
                      dx, dy, dz, gc, Lx, Ly, Lz,
                      nx=None, ny=None, nz=None,
                      title="VVP Cavity Flow"):
    """
    Write VVP fields to Tecplot ASCII on a uniform grid spanning exactly
    [0,Lx] x [0,Ly] x [0,Lz] (matching the FD convention: boundary points
    included), reconstructed from the cell-centered FVM data via the exact
    wall value recovered from the ghost/interior mirror pair.

    Pass the FULL arrays here (still containing their gc ghost layers) —
    NOT pre-sliced with [gc:-gc, ...] — otherwise there's no ghost value
    left to reconstruct the x=0 / x=Lx (etc.) boundary points from.
    """
    fields = [np.asarray(f) for f in
              (psi_x, psi_y, psi_z, omega_x, omega_y, omega_z,
               u, v, w, div_omega, div_vel)]

    Ny_tot, Nx_tot, Nz_tot = fields[0].shape
    if nx is None: nx = Nx_tot - 2*gc   # default: same resolution as the interior grid
    if ny is None: ny = Ny_tot - 2*gc
    if nz is None: nz = Nz_tot - 2*gc

    fields_uniform = [to_uniform_grid_3d(f, dx, dy, dz, gc, Lx, Ly, Lz, nx, ny, nz)
                       for f in fields]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    Y, X, Z = np.meshgrid(y, x, z, indexing="ij")   # shape (ny, nx, nz)

    def i_fastest(field):
        # I fastest, then J, then K -> Tecplot POINT ordering
        return np.transpose(field, (2, 0, 1)).reshape(-1)

    cols = [i_fastest(X), i_fastest(Y), i_fastest(Z)] + [i_fastest(f) for f in fields_uniform]
    data = np.column_stack(cols)

    names = ["X", "Y", "Z",
             "PSI_X", "PSI_Y", "PSI_Z",
             "OMEGA_X", "OMEGA_Y", "OMEGA_Z",
             "U", "V", "W",
             "DIV_OMEGA", "DIV_VEL"]

    with open(filename, "w") as f:
        f.write(f'TITLE = "{title}"\n')
        f.write("VARIABLES = " + ", ".join(f'"{n}"' for n in names) + "\n")
        f.write(f'ZONE T="Zone1", I={nx}, J={ny}, K={nz}, DATAPACKING=POINT\n')
        np.savetxt(f, data, fmt="%.8e")

    print(f"Wrote {nx*ny*nz} points to {filename}")


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

def calculate_divergence(u_field, v_field, w_field, dx, dy, dz):
    """
    Calculates the divergence of a 3D vector field (u, v, w) using central differences.
    The divergence is computed for the interior points of the domain.
    """
    div_u = jnp.zeros_like(u_field)
    du_dx = grad4(u_field, dx,1)
    dv_dy = grad4(v_field, dy,0)
    dw_dz = grad4(w_field, dz,2)

    interior_divergence = du_dx[gc:-gc, gc:-gc, gc:-gc] + \
                          dv_dy[gc:-gc, gc:-gc, gc:-gc] + \
                          dw_dz[gc:-gc, gc:-gc, gc:-gc]

    return div_u.at[gc:-gc, gc:-gc, gc:-gc].set(interior_divergence)