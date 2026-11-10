import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
from jax import config
from Boundary_Conditions import*
import yaml
from jax.scipy.sparse.linalg import cg

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found.")
    exit() 

gc = config["ghost_cells"]


def build_pressure_coeffs(nx, ny, nz, dx, dy, dz,
                            bc_type=("Dirichlet",) * 6,
                            bc_value=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
    cx = 1.0 / dx**2
    cy = 1.0 / dy**2
    cz = 1.0 / dz**2

    ap = 2.0*(cx + cy + cz) * jnp.ones((ny, nx, nz))
    bc_s = jnp.zeros((ny, nx, nz))

    (bt_type, tp_type, lf_type, rt_type, front, back) = bc_type
    (bt_val, tp_val, lf_val, rt_val, front_val, back_val) = bc_value

    # bottom (j=0) / top (j=ny-1) -- y-direction, use cy
    if bt_type == "Dirichlet":
        ap = ap.at[0, :, :].add(cy)
        bc_s = bc_s.at[0, :, :].add(2*cy*bt_val)
    elif bt_type == "Neumann":  
        ap = ap.at[0, :, :].add(-cy)

    if tp_type == "Dirichlet":
        ap = ap.at[-1, :, :].add(cy)
        bc_s = bc_s.at[-1, :, :].add(2*cy*tp_val)
    elif tp_type == "Neumann": 
        ap = ap.at[-1, :, :].add(-cy)

    # left (i=0) / right (i=nx-1) -- x-direction, use cx
    if lf_type == "Dirichlet":
        ap = ap.at[:, 0, :].add(cx)
        bc_s = bc_s.at[:, 0, :].add(2*cx*lf_val)
    elif lf_type == "Neumann": 
        ap = ap.at[:, 0, :].add(-cx)

    if rt_type == "Dirichlet":
        ap = ap.at[:, -1, :].add(cx)
        bc_s = bc_s.at[:, -1, :].add(2*cx*rt_val)
    elif rt_type == "Neumann": 
        ap = ap.at[:, -1, :].add(-cx)

    if front == "Dirichlet":
        ap = ap.at[:,:,0].add(cz)
        bc_s = bc_s.at[:,:,0].add(2*cz*front_val)
    elif front == "Neumann":
        ap = ap.at[:,:,0].add(-cz)

    if back == "Dirichlet":
        ap = ap.at[:,:,-1].add(cz)
        bc_s = bc_s.at[:,:,-1].add(2*cz*back_val)
    elif back == "Neumann":
        ap = ap.at[:,:,-1].add(-cz)
        

    return ap, bc_s, cx, cy, cz

def _poisson_matvec(u3d, ap, cx, cy, cz, axis=(0, 0, 0)):
    Au = ap * u3d

    if axis[0] == 1:
        Au = Au + (-cx) * jnp.roll(u3d, shift=-1, axis=1)   # east, wraps around
        Au = Au + (-cx) * jnp.roll(u3d, shift=+1, axis=1)   # west, wraps around
    else:
        Au = Au.at[:, :-1].add(-cx * u3d[:, 1:, :])   # east
        Au = Au.at[:, 1:].add(-cx * u3d[:, :-1, :])   # west

    if axis[1] == 1:
        Au = Au + (-cy) * jnp.roll(u3d, shift=-1, axis=0)   # north, wraps around
        Au = Au + (-cy) * jnp.roll(u3d, shift=+1, axis=0)   # south, wraps around
    else:
        Au = Au.at[:-1, :].add(-cy * u3d[1:, :, :])   # north
        Au = Au.at[1:, :].add(-cy * u3d[:-1, :, :])   # south

    if axis[2] == 1:
        Au = Au + (-cz) * jnp.roll(u3d, shift=-1, axis=2)   # front, wraps around
        Au = Au + (-cz) * jnp.roll(u3d, shift=+1, axis=2)   # back, wraps around
    else:
        Au = Au.at[:, :, :-1].add(-cz * u3d[:, :, 1:])   # front
        Au = Au.at[:, :, 1:].add(-cz * u3d[:, :, :-1])   # back

    return Au


def potential_poisson_cg(p, source, ap, bc_s, cx, cy, cz, tol=1e-8, maxiter=800,):
    b = bc_s + source

    def A_op(v):
        Av = _poisson_matvec(v, ap, cx, cy, cz, (0,0,0)) 
        return Av      
    

    p_new, _ = cg(A_op, b, x0=p, tol=tol, maxiter=maxiter)
    return p_new