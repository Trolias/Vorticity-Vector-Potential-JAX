import jax 
import jax.numpy as jnp
import yaml
from jax import config

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found. Please ensure it's in the same directory.")
    exit() 

gcp = config["ghost_cells"]
Nx = config['Grid']['Nx']
Ny = config["Grid"]["Ny"]
Nz = config["Grid"]["Nz"]

CENTRAL2 = {-1: -0.5, 1: 0.5}
LAPL2 = {-1: 1.0, 0: -2, 1: 1.0}

LAPL4    = {-2: -1/12, -1: 4/3,  0: -5/2, 1: 4/3, 2: -1/12}   
CENTRAL4 = {-2:  1/12, -1: -2/3, 1: 2/3,  2: -1/12}        

CENTRAL6 = {-3: -1.0/60, -2: 3.0/20, -1: -3.0/4, 0: 0.0, 1: 3.0/4, 2: -3.0/20, 3: 1.0/60}
LAPL6 = {-3: 1.0/90, -2: -3.0/20, -1: 3.0/2, 0: -49.0/18, 1: 3.0/2, 2: -3.0/20, 3: 1.0/90}
CENTRAL8 = {-4: 1.0/280, -3: -4.0/105, -2: 1.0/5, -1: -4.0/5, 0: 0.0, 1: 4.0/5, 2: -1.0/5, 3: 4.0/105, 4: -1.0/280}


def shift(field, off_x =0, off_y =0, off_z=0):
    x0, y0, z0 = gcp+off_x, gcp+off_y, gcp+off_z
    return field[y0:y0+Ny, x0:x0+Nx, z0:z0+Nz]

def apply_stencil(field, coeffs, axis, h, h_power=1):
    total = 0
    for offset, w in coeffs.items():
        if axis == 0:
            s = shift(field, off_y = offset)
        elif axis == 1:
            s = shift(field, off_x = offset)
        elif axis == 2:
            s = shift(field, off_z = offset)
        total = total + w * s
    return total / (h**h_power)

def grad(field, dx, axis):
    out = jnp.zeros_like(field)
    grad = apply_stencil(field, CENTRAL2, axis, dx, 1)
    out = out.at[gcp:-gcp, gcp:-gcp, gcp:-gcp].set(grad)
    return out

def grad4(field, dx, axis):
    out = jnp.zeros_like(field)
    grad = apply_stencil(field, CENTRAL4, axis, dx, 1)
    out = out.at[gcp:-gcp, gcp:-gcp, gcp:-gcp].set(grad)
    return out

def grad8(field, dx, axis):
    out = jnp.zeros_like(field)
    grad = apply_stencil(field, CENTRAL8, axis, dx, 1)
    out = out.at[gcp:-gcp, gcp:-gcp, gcp:-gcp].set(grad)
    return out

def Laplacian_x(u, dx):
    out = jnp.zeros_like(u)
    interior = apply_stencil(u, LAPL6, axis=1, h=dx, h_power=2)    
    return out.at[gcp:-gcp, gcp:-gcp, gcp:-gcp].set(interior)

def Laplacian_y(u, dy):
    out = jnp.zeros_like(u)
    interior = apply_stencil(u, LAPL6, axis=0, h=dy, h_power=2) 
    return out.at[gcp:-gcp, gcp:-gcp, gcp:-gcp].set(interior)

def Laplacian_z(u, dz):
    out = jnp.zeros_like(u)
    interior = apply_stencil(u, LAPL6, axis=2, h=dz, h_power=2) 
    
    return out.at[gcp:-gcp, gcp:-gcp, gcp:-gcp].set(interior)

def Laplacian(u, dx, dy, dz):
    out = jnp.zeros_like(u)
    laplacian_x = apply_stencil(u, LAPL6, axis=1, h=dx, h_power=2)
    laplacian_y = apply_stencil(u, LAPL6, axis=0, h=dy, h_power=2)
    laplacian_z = apply_stencil(u, LAPL6, axis=2, h=dz, h_power=2) 

    return out.at[gcp:-gcp, gcp:-gcp, gcp:-gcp].set(laplacian_x + laplacian_y + laplacian_z)

def curl (field_x, field_y, field_z, dx,dy,dz):

    u_x = grad(field_z,dy,0) - grad(field_y,dz,2)
    u_y = grad(field_x,dz,2) - grad(field_z,dx,1)
    u_z = grad(field_y,dx,1) - grad(field_x,dy,0)

    return u_x, u_y, u_z


def weno5_local(v, eps=1e-6):
    v0, v1, v2, v3, v4 = v[0], v[1], v[2], v[3], v[4]
    b0 = (13/12)*(v0 - 2*v1 + v2)**2 + 0.25*(v0 - 4*v1 + 3*v2)**2
    b1 = (13/12)*(v1 - 2*v2 + v3)**2 + 0.25*(v1 - v3)**2
    b2 = (13/12)*(v2 - 2*v3 + v4)**2 + 0.25*(3*v2 - 4*v3 + v4)**2
    a0, a1, a2 = 0.1/(eps + b0)**2, 0.6/(eps + b1)**2, 0.3/(eps + b2)**2
    p0 = (2*v0 - 7*v1 + 11*v2)/6
    p1 = (-v1 + 5*v2 + 2*v3)/6
    p2 = (2*v2 + 5*v3 - v4)/6
    return (a0*p0 + a1*p1 + a2*p2)/(a0 + a1 + a2)

def weno_flux_div(w, u, h, axis):
    g, ug = jnp.moveaxis(w, axis, 0), jnp.moveaxis(u, axis, 0)
    N = g.shape[0]
    cw = [g[a:N-5+a] for a in range(6)]            # cells k-2..k+3 for face k+1/2
    cu = [ug[a:N-5+a] for a in range(6)]
    alpha = jnp.max(jnp.abs(jnp.stack(cu)), axis=0)   # local max |u|
    fp = [0.5*(a*q + alpha*q) for a, q in zip(cu, cw)]
    fm = [0.5*(a*q - alpha*q) for a, q in zip(cu, cw)]
    F = weno5_local(fp[0:5]) + weno5_local((fm[5], fm[4], fm[3], fm[2], fm[1]))
    d = (F[gcp-2:N-gcp-2] - F[gcp-3:N-gcp-3]) / h
    d = jnp.pad(d, [(gcp, gcp), (0, 0), (0, 0)])
    return jnp.moveaxis(d, 0, axis)

def weno_advect(w, u, h, axis):
    g, ug = jnp.moveaxis(w, axis, 0), jnp.moveaxis(u, axis, 0)
    N = g.shape[0]
    cw = [g[a:N-5+a] for a in range(6)]              
    wL = weno5_local(cw[0:5])
    wR = weno5_local((cw[5], cw[4], cw[3], cw[2], cw[1]))
    sr, sl = slice(gcp-2, N-gcp-2), slice(gcp-3, N-gcp-3)
    ui = ug[gcp:N-gcp]; up = ui > 0
    d = ui*(jnp.where(up, wL[sr], wR[sr]) - jnp.where(up, wL[sl], wR[sl])) / h
    d = jnp.pad(d, [(gcp, gcp), (0, 0), (0, 0)])
    return jnp.moveaxis(d, 0, axis)

