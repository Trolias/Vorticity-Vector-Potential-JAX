import jax
import jax.numpy as jnp
from jax import jit
from jax import config
config.update("jax_enable_x64", True) # Enable 64-bit precision for better accuracy in simulations

"""""
    Below we present the functon of multiple useful operators such as: the nabla operator, the Laplacian, the curl and the div.
    Grad_{x,y,z) refers to the corresponding differentiating variable
    Adv_{x,y,z} refers to direction first-order upwind scheme describing the convective term in NS equations
"""
@jit 
def curl (field_x, field_y, field_z, dx,dy,dz):

    u_x = Grad_y(field_z,dy) - Grad_z(field_y,dz)
    u_y = Grad_z(field_x,dz) - Grad_x(field_z,dx)
    u_z = Grad_x(field_y,dx) - Grad_y(field_x,dy)

    return u_x, u_y, u_z

@jit
def Grad_x(u, dx):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 2:,1:-1] - u[1:-1, :-2,1:-1]) / (2.0 * dx)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Grad_y(u, dy):
    out = jnp.zeros_like(u)
    interior = (u[2:, 1:-1,1:-1] - u[:-2, 1:-1,1:-1]) / (2.0 * dy)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Grad_z(u, dz):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 1:-1,2:] - u[1:-1, 1:-1,:-2]) / (2.0 * dz)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Laplacian_x(u, dx):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 2:,1:-1] - 2.0 * u[1:-1, 1:-1,1:-1] + u[1:-1, :-2,1:-1]) / (dx * dx)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Laplacian_y(u, dy):
    out = jnp.zeros_like(u)
    interior = (u[2:, 1:-1,1:-1] - 2.0 * u[1:-1, 1:-1,1:-1] + u[:-2, 1:-1,1:-1]) / (dy * dy)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Laplacian_z(u, dz):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 1:-1,2:] - 2.0 * u[1:-1, 1:-1,1:-1] + u[1:-1, 1:-1,:-2]) / (dz * dz)
    return out.at[1:-1,1:-1,1:-1].set(interior)


@jit
def Adv_x(velocity_adv_field, phi_field, dx):

    out = jnp.zeros_like(phi_field)
    interior_slice = (slice(1,-1), slice(1,-1), slice(1,-1))

    vel_interior = velocity_adv_field[interior_slice]
    phi_interior = phi_field[interior_slice]

    # Backward difference for positive velocity (flow from left in x-direction)
    phi_prev_x = phi_field[1:-1, :-2, 1:-1]
    advection_positive = vel_interior * (phi_interior - phi_prev_x) / dx

    # Forward difference for negative velocity (flow from right in x-direction)
    phi_next_x = phi_field[1:-1, 2:, 1:-1]
    advection_negative = vel_interior * (phi_next_x - phi_interior) / dx

    advection_term = jnp.where(vel_interior > 0, advection_positive, 0.0) + \
                     jnp.where(vel_interior < 0, advection_negative, 0.0)

    return out.at[interior_slice].set(advection_term)

@jit
def Adv_y(velocity_adv_field, phi_field, dy):

    out = jnp.zeros_like(phi_field)
    interior_slice = (slice(1,-1), slice(1,-1), slice(1,-1))

    vel_interior = velocity_adv_field[interior_slice]
    phi_interior = phi_field[interior_slice]

    # Backward difference for positive velocity (flow from bottom in y-direction)
    phi_prev_y = phi_field[:-2, 1:-1, 1:-1]
    advection_positive = vel_interior * (phi_interior - phi_prev_y) / dy

    # Forward difference for negative velocity (flow from top in y-direction)
    phi_next_y = phi_field[2:, 1:-1, 1:-1]
    advection_negative = vel_interior * (phi_next_y - phi_interior) / dy

    advection_term = jnp.where(vel_interior > 0, advection_positive, 0.0) + \
                     jnp.where(vel_interior < 0, advection_negative, 0.0)

    return out.at[interior_slice].set(advection_term)

@jit
def Adv_z(velocity_adv_field, phi_field, dz):

    out = jnp.zeros_like(phi_field)
    interior_slice = (slice(1,-1), slice(1,-1), slice(1,-1))

    vel_interior = velocity_adv_field[interior_slice]
    phi_interior = phi_field[interior_slice]

    # Backward difference for positive velocity (flow from front in z-direction)
    phi_prev_z = phi_field[1:-1, 1:-1, :-2]
    advection_positive = vel_interior * (phi_interior - phi_prev_z) / dz

    # Forward difference for negative velocity (flow from back in z-direction)
    phi_next_z = phi_field[1:-1, 1:-1, 2:]
    advection_negative = vel_interior * (phi_next_z - phi_interior) / dz

    advection_term = jnp.where(vel_interior > 0, advection_positive, 0.0) + \
                     jnp.where(vel_interior < 0, advection_negative, 0.0)

    return out.at[interior_slice].set(advection_term)

@jit
def Backward_1st_x(u, dx):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 1:-1,1:-1] - u[1:-1, :-2,1:-1]) / (dx)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Backward_1st_y(u, dy):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 1:-1,1:-1] - u[:-2, 1:-1,1:-1]) / (dy)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Backward_1st_z(u, dz):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 1:-1,1:-1] - u[1:-1, 1:-1,:-2]) / (dz)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Forward_1st_x(u, dx):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 2:,1:-1] - u[1:-1, 1:-1,1:-1]) / (dx)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Forward_1st_y(u, dy):
    out = jnp.zeros_like(u)
    interior = (u[2:, 1:-1,1:-1] - u[1:-1, 1:-1,1:-1]) / (dy)
    return out.at[1:-1,1:-1,1:-1].set(interior)

@jit
def Forward_1st_z(u, dz):
    out = jnp.zeros_like(u)
    interior = (u[1:-1, 1:-1,2:] - u[1:-1, 1:-1,1:-1]) / (dz)
    return out.at[1:-1,1:-1,1:-1].set(interior)


# --------------------------------------------------------  TIME INTEGRATION -----------------------------------------------------------------

""""
    Runge-Kutta 4 time integration scheme (NOT TESTED)
"""
def rhs_omega(omega_x, omega_y, omega_z, u, v, w, dx, dy, dz, Re):
    # Laplacian terms
    lap_x = Laplacian_x(omega_x, dx) + Laplacian_y(omega_x, dy) + Laplacian_z(omega_x, dz)
    lap_y = Laplacian_x(omega_y, dx) + Laplacian_y(omega_y, dy) + Laplacian_z(omega_y, dz)
    lap_z = Laplacian_x(omega_z, dx) + Laplacian_y(omega_z, dy) + Laplacian_z(omega_z, dz)

    # Convection terms
    conv_x = u * Grad_x(omega_x, dx) + v * Grad_y(omega_x, dy) + w * Grad_z(omega_x, dz)
    conv_y = u * Grad_x(omega_y, dx) + v * Grad_y(omega_y, dy) + w * Grad_z(omega_y, dz)
    conv_z = u * Grad_x(omega_z, dx) + v * Grad_y(omega_z, dy) + w * Grad_z(omega_z, dz)

    # Vortex stretching terms
    stretch_x = omega_x * Grad_x(u, dx) + omega_y * Grad_y(u, dy) + omega_z * Grad_z(u, dz)
    stretch_y = omega_x * Grad_x(v, dx) + omega_y * Grad_y(v, dy) + omega_z * Grad_z(v, dz)
    stretch_z = omega_x * Grad_x(w, dx) + omega_y * Grad_y(w, dy) + omega_z * Grad_z(w, dz)

    # Combine
    rhs_x = (1.0 / Re) * lap_x - conv_x + stretch_x
    rhs_y = (1.0 / Re) * lap_y - conv_y + stretch_y
    rhs_z = (1.0 / Re) * lap_z - conv_z + stretch_z

    return rhs_x, rhs_y, rhs_z


@jit 
def RK4(omega_x, omega_y, omega_z, u, v, w, dx, dy, dz, dt, Re):

    k1 = rhs_omega(omega_x, omega_y, omega_z, u, v, w, dx, dy, dz, Re)

    ωx_temp = omega_x + 0.5 * dt * k1[0]
    ωy_temp = omega_y + 0.5 * dt * k1[1]
    ωz_temp = omega_z + 0.5 * dt * k1[2]
    k2 = rhs_omega(ωx_temp, ωy_temp, ωz_temp, u, v, w, dx, dy, dz, Re)

    ωx_temp = omega_x + 0.5 * dt * k2[0]
    ωy_temp = omega_y + 0.5 * dt * k2[1]
    ωz_temp = omega_z + 0.5 * dt * k2[2]
    k3 = rhs_omega(ωx_temp, ωy_temp, ωz_temp, u, v, w, dx, dy, dz, Re)

    ωx_temp = omega_x + dt * k3[0]
    ωy_temp = omega_y + dt * k3[1]
    ωz_temp = omega_z + dt * k3[2]
    k4 = rhs_omega(ωx_temp, ωy_temp, ωz_temp, u, v, w, dx, dy, dz, Re)

    omega_new_x = omega_x + (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    omega_new_y = omega_y + (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    omega_new_z = omega_z + (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])


    return omega_new_x, omega_new_y, omega_new_z

""""
    Alternative way to express the convective term in Vorticity Transport equation as a function of teh vector-potential ONLY. 
    (NOT TESTED YET)

"""

@jit
def int_grad_x(f, dx):
    df = jnp.zeros_like(f)

    df = df.at[1:-1, 2:-2, 1:-1].set((f[1:-1, 3:-1, 1:-1] - f[1:-1, 1:-3, 1:-1]) / (2.0 * dx))
    df = df.at[1:-1, 1, 1:-1].set((f[1:-1, 2, 1:-1] - f[1:-1, 1, 1:-1]) / dx)
    df = df.at[1:-1, -2, 1:-1].set((f[1:-1, -2, 1:-1] - f[1:-1, -3, 1:-1]) / dx)

    return df[1:-1,1:-1,1:-1]
@jit
def int_grad_y(f, dy):
    df = jnp.zeros_like(f)
    df = df.at[2:-2, 1:-1, 1:-1].set((f[3:-1, 1:-1, 1:-1] - f[1:-3, 1:-1, 1:-1]) / (2.0 * dy))
    df = df.at[1, 1:-1, 1:-1].set((f[2, 1:-1, 1:-1] - f[1, 1:-1, 1:-1]) / dy)
    df = df.at[-2, 1:-1, 1:-1].set((f[-2, 1:-1, 1:-1] - f[-3, 1:-1, 1:-1]) / dy)
    return df[1:-1,1:-1,1:-1]

@jit
def int_grad_z(f, dz):
    df = jnp.zeros_like(f)
    df = df.at[1:-1, 1:-1, 2:-2].set((f[1:-1, 1:-1, 3:-1] - f[1:-1, 1:-1, 1:-3]) / (2.0 * dz))
    df = df.at[1:-1, 1:-1, 1].set((f[1:-1, 1:-1, 2] - f[1:-1, 1:-1, 1]) / dz)
    df = df.at[1:-1, 1:-1, -2].set((f[1:-1, 1:-1, -2] - f[1:-1, 1:-1, -3]) / dz)
    return df[1:-1,1:-1,1:-1]



@jit
def NonLinearConvectivePotentialBasedTerms(psi_new_x,psi_new_y,psi_new_z,dx,dy,dz):
    viscous_term_x = jnp.zeros_like(psi_new_x)
    viscous_term_y = jnp.zeros_like(psi_new_y)
    viscous_term_z = jnp.zeros_like(psi_new_z)
    NonLinear_Conv_x = jnp.zeros_like(psi_new_x)
    NonLinear_Conv_y = jnp.zeros_like(psi_new_y)
    NonLinear_Conv_z = jnp.zeros_like(psi_new_z)

    viscous_term_x = viscous_term_x.at[1:-1,1:-1,1:-1].set(
        + ( psi_new_y[2:,2:,1:-1] - psi_new_y[:-2,2:,1:-1] - psi_new_y[2:,:-2,1:-1] + psi_new_y[:-2,:-2,1:-1] ) / (4.0*dx*dy)
        - ( psi_new_x[2:,1:-1,1:-1] - 2.0*psi_new_x[1:-1,1:-1,1:-1] + psi_new_x[:-2,1:-1,1:-1] ) / (dy**2)
        - ( psi_new_x[1:-1,1:-1,2:] - 2.0*psi_new_x[1:-1,1:-1,1:-1] + psi_new_x[1:-1,1:-1,:-2] ) / (dz**2)
        + ( psi_new_z[1:-1,2:,2:] - psi_new_z[1:-1,2:,:-2] - psi_new_z[1:-1,:-2,2:] + psi_new_z[1:-1,:-2,:-2] ) / (4.0*dx*dz)
    ) 

    viscous_term_y = viscous_term_y.at[1:-1,1:-1,1:-1].set(
        + ( psi_new_z[2:,1:-1,2:] - psi_new_z[2:,1:-1,:-2] - psi_new_z[:-2,1:-1,2:] + psi_new_z[:-2,1:-1,:-2] ) / (4.0*dy*dz)
        - ( psi_new_y[1:-1,1:-1,2:] - 2.0*psi_new_y[1:-1,1:-1,1:-1] + psi_new_y[1:-1,1:-1,:-2] ) / (dz**2)
        - ( psi_new_y[1:-1,2:,1:-1] - 2.0*psi_new_y[1:-1,1:-1,1:-1] + psi_new_y[:-2,1:-1,1:-1] ) / (dx**2)
        + ( psi_new_x[2:,2:,1:-1] - psi_new_x[2:,:-2,1:-1] - psi_new_x[:-2,2:,1:-1] + psi_new_x[:-2,:-2,1:-1] ) / (4.0*dx*dy)
    )

    viscous_term_z = viscous_term_z.at[1:-1,1:-1,1:-1].set(
        + ( psi_new_x[1:-1,2:,2:] - psi_new_x[1:-1,:-2,2:] - psi_new_x[1:-1,2:,:-2] + psi_new_x[1:-1,:-2,:-2] ) / (4.0*dx*dz)
        + ( psi_new_y[2:,1:-1,2:] - psi_new_y[2:,1:-1,:-2] - psi_new_y[:-2,1:-1,2:] + psi_new_y[:-2,1:-1,:-2] ) / (4.0*dy*dz)
        - ( psi_new_z[1:-1,2:,1:-1] - 2.0*psi_new_z[1:-1,1:-1,1:-1] + psi_new_z[1:-1,:-2,1:-1] ) / (dx**2)
        - ( psi_new_z[2:,1:-1,1:-1] - 2.0*psi_new_z[1:-1,1:-1,1:-1] + psi_new_z[:-2,1:-1,1:-1] ) / (dy**2)
    )

    NonLinear_Conv_x = NonLinear_Conv_x.at[1:-1,1:-1,1:-1].set(
        + viscous_term_x[1:-1,1:-1,1:-1] * (
            + (psi_new_x[2:,1:-1,2:] - psi_new_x[2:,1:-1,:-2] - psi_new_x[:-2,1:-1,2:] + psi_new_x[:-2,1:-1,:-2]) / (4.0*dy*dz)
            - (psi_new_z[2:,2:,1:-1] - psi_new_z[:-2,2:,1:-1] - psi_new_z[2:,:-2,1:-1] + psi_new_z[:-2,:-2,1:-1]) / (4.0*dx*dy)
        )
        - viscous_term_y[1:-1,1:-1,1:-1] * (
            + (psi_new_z[2:,1:-1,1:-1] - 2.0*psi_new_z[1:-1,1:-1,1:-1] + psi_new_z[:-2,1:-1,1:-1]) / (dy**2)\
            - (psi_new_y[2:,1:-1,2:] - psi_new_y[2:,1:-1,:-2] - psi_new_y[:-2,1:-1,2:] + psi_new_y[:-2,1:-1,:-2]) / (4.0*dy*dz)
        )
        - viscous_term_z[1:-1,1:-1,1:-1] * (
            + (psi_new_z[2:,1:-1,2:] - psi_new_z[2:,1:-1,:-2] - psi_new_z[:-2,1:-1,2:] + psi_new_z[:-2,1:-1,:-2]) / (4.0*dz*dy)
            - (psi_new_y[1:-1,1:-1,2:] - 2.0*psi_new_y[1:-1,1:-1,1:-1] - psi_new_y[1:-1,1:-1,:-2]) / (dz**2)
        )
        + viscous_term_x[1:-1,1:-1,1:-1] * (
            + (psi_new_y[1:-1,2:,2:] - psi_new_y[1:-1,2:,:-2] - psi_new_y[1:-1,:-2,2:] + psi_new_y[1:-1,:-2,:-2]) / (4.0*dx*dz)
            - (psi_new_x[2:,1:-1,2:] - psi_new_x[2:,1:-1,:-2] - psi_new_x[:-2,1:-1,2:] + psi_new_x[:-2,1:-1,:-2]) / (4.0*dy*dz)
        )
        + ( (psi_new_x[1:-1,1:-1,2:] - psi_new_x[1:-1,1:-1,:-2])/(2.0*dz) - (psi_new_z[1:-1,2:,1:-1]-psi_new_z[1:-1,:-2,1:-1])/(2.0*dx) ) * (
            int_grad_y(viscous_term_x,dy)
        )
        - ( (psi_new_z[2:,1:-1,1:-1] - psi_new_z[:-2,1:-1,1:-1])/(2.0*dy) - (psi_new_y[1:-1,1:-1,2:]-psi_new_y[1:-1,1:-1,:-2])/(2.0*dz) ) * (
            int_grad_y(viscous_term_y,dy)
        )
        - ( (psi_new_z[2:,1:-1,1:-1] - psi_new_z[:-2,1:-1,1:-1])/(2.0*dy) - (psi_new_y[1:-1,1:-1,2:]-psi_new_y[1:-1,1:-1,:-2])/(2.0*dz) ) * (
            int_grad_z(viscous_term_z,dz)
        )
        + ( (psi_new_y[1:-1,2:,1:-1] - psi_new_y[1:-1,:-2,1:-1])/(2.0*dx) - (psi_new_x[2:,1:-1,1:-1]-psi_new_x[:-2,1:-1,1:-1])/(2.0*dy) ) * (
            int_grad_z(viscous_term_x,dz)
        )
    )
    # --------------------------------------------------------------------------------------------------------------------------------------

    NonLinear_Conv_y = NonLinear_Conv_y.at[1:-1,1:-1,1:-1].set(
        + viscous_term_y[1:-1,1:-1,1:-1] * (
            + (psi_new_y[1:-1,2:,2:] - psi_new_y[1:-1,2:,:-2] - psi_new_y[1:-1,:-2,2:] + psi_new_y[1:-1,:-2,:-2]) / (4.0*dx*dz)
            - (psi_new_x[2:,1:-1,2:] - psi_new_x[2:,1:-1,:-2] - psi_new_x[:-2,1:-1,2:] + psi_new_x[:-2,1:-1,:-2]) / (4.0*dy*dz)
        )
        - viscous_term_z[1:-1,1:-1,1:-1] * (
            + (psi_new_x[1:-1,1:-1,2:] - 2.0*psi_new_x[1:-1,1:-1,1:-1] + psi_new_x[1:-1,1:-1,:-2]) / (dz**2)
            - (psi_new_z[1:-1,2:,2:] - psi_new_z[1:-1,2:,:-2] - psi_new_z[1:-1,:-2,2:] + psi_new_z[1:-1,:-2,:-2]) / (4.0*dx*dz)
        )
        - viscous_term_x[1:-1,1:-1,1:-1] * (
            + (psi_new_x[1:-1,2:,2:] - psi_new_x[1:-1,2:,:-2] - psi_new_x[1:-1,:-2,2:] + psi_new_x[1:-1,:-2,:-2]) / (4.0*dx*dz)
            - (psi_new_z[1:-1,2:,1:-1] - 2.0*psi_new_z[1:-1,1:-1,1:-1] + psi_new_z[1:-1,:-2,1:-1]) / (dx**2)
        )
        + viscous_term_y[1:-1,1:-1,1:-1] * (
            + (psi_new_z[2:,2:,1:-1] - psi_new_z[:-2,2:,1:-1] - psi_new_z[2:,:-2,1:-1] + psi_new_z[:-2,:-2,1:-1]) / (4.0*dx*dy)
            - (psi_new_y[1:-1,2:,2:] - psi_new_y[1:-1,2:,:-2] - psi_new_y[1:-1,:-2,2:] + psi_new_y[1:-1,:-2,:-2]) / (4.0*dz*dx)
        )
        + ( (psi_new_y[1:-1,2:,1:-1] - psi_new_y[1:-1,:-2,1:-1])/(2.0*dx) - (psi_new_x[2:,1:-1,1:-1]-psi_new_x[:-2,1:-1,1:-1])/(2.0*dy) ) * (
            int_grad_z(viscous_term_y,dz)
        )
        - ( (psi_new_x[1:-1,1:-1,2:] - psi_new_x[1:-1,1:-1,:-2])/(2.0*dz) - (psi_new_z[1:-1,2:,1:-1]-psi_new_z[1:-1,:-2,1:-1])/(2.0*dx) ) * (
            int_grad_z(viscous_term_z,dz)
        )
        + ( (psi_new_z[2:,1:-1,1:-1] - psi_new_z[:-2,1:-1,1:-1])/(2.0*dy) - (psi_new_y[1:-1,1:-1,2:]-psi_new_y[1:-1,1:-1,:-2])/(2.0*dz) ) * (
            int_grad_x(viscous_term_y,dx)
        )
        - ( (psi_new_x[1:-1,1:-1,2:] - psi_new_x[1:-1,1:-1,:-2])/(2.0*dz) - (psi_new_z[1:-1,2:,1:-1]-psi_new_z[1:-1,:-2,1:-1])/(2.0*dx) ) * (
            int_grad_x(viscous_term_x,dx)
        )
    )
    # -----------------------------------------------------------------------------------------------------------------------------------------

    NonLinear_Conv_z = NonLinear_Conv_z.at[1:-1,1:-1,1:-1].set(
        + viscous_term_z[1:-1,1:-1,1:-1] * (
            + (psi_new_z[2:,2:,1:-1] - psi_new_z[:-2,2:,1:-1] - psi_new_z[2:,:-2,1:-1] + psi_new_z[:-2,:-2,1:-1]) / (4.0*dy*dx)
            - (psi_new_y[1:-1,2:,2:] - psi_new_y[1:-1,:-2,2:] - psi_new_y[1:-1,2:,:-2] + psi_new_y[1:-1,:-2,:-2]) / (4.0*dx*dz)
        )
        - viscous_term_x[1:-1,1:-1,1:-1] * (
            + (psi_new_y[1:-1,2:,1:-1] - 2.0*psi_new_y[1:-1,1:-1,1:-1] + psi_new_y[1:-1,:-2,1:-1]) / (dx**2)
            - (psi_new_x[2:,2:,1:-1] - psi_new_x[2:,:-2,1:-1] - psi_new_x[:-2,2:,1:-1] + psi_new_x[:-2,:-2,1:-1]) / (4.0*dy*dx)
        )
        - viscous_term_y[1:-1,1:-1,1:-1] * (
            + (psi_new_y[2:,2:,1:-1] - psi_new_y[2:,:-2,1:-1] - psi_new_y[:-2,2:,1:-1] + psi_new_y[:-2,:-2,1:-1]) / (4.0*dx*dy)
            - (psi_new_x[2:,1:-1,1:-1] - 2.0*psi_new_x[1:-1,1:-1,1:-1] + psi_new_x[:-2,1:-1,1:-1]) / (dy**2)
        )
        + viscous_term_z[1:-1,1:-1,1:-1] * (
            + (psi_new_x[2:,1:-1,2:] - psi_new_x[:-2,1:-1,2:] - psi_new_x[2:,1:-1,:-2] + psi_new_x[:-2,1:-1,:-2]) / (4.0*dz*dy)
            - (psi_new_z[2:,2:,1:-1] - psi_new_z[:-2,2:,1:-1] - psi_new_z[2:,:-2,1:-1] + psi_new_z[:-2,:-2,1:-1]) / (4.0*dx*dy)
        )
        + ( (psi_new_z[2:,1:-1,1:-1] - psi_new_z[:-2,1:-1,1:-1])/(2.0*dy) - (psi_new_y[1:-1,1:-1,2:]-psi_new_y[1:-1,1:-1,:-2])/(2.0*dz) ) * (
            int_grad_x(viscous_term_z,dx)
        )
        - ( (psi_new_y[1:-1,2:,1:-1] - psi_new_y[1:-1,:-2,1:-1])/(2.0*dx) - (psi_new_x[2:,1:-1,1:-1]-psi_new_x[:-2,1:-1,1:-1])/(2.0*dy) ) * (
            int_grad_x(viscous_term_x,dx)
        )
        + ( (psi_new_y[1:-1,2:,1:-1] - psi_new_y[1:-1,:-2,1:-1])/(2.0*dx) - (psi_new_x[2:,1:-1,1:-1]-psi_new_x[:-2,1:-1,1:-1])/(2.0*dy) ) * (
            int_grad_y(viscous_term_y,dy)
        )
        - ( (psi_new_x[1:-1,1:-1,2:] - psi_new_x[1:-1,1:-1,:-2])/(2.0*dz) - (psi_new_z[1:-1,2:,1:-1]-psi_new_z[1:-1,:-2,1:-1])/(2.0*dx) ) * (
            int_grad_y(viscous_term_z,dy)
        )
    )

    return NonLinear_Conv_x, NonLinear_Conv_y, NonLinear_Conv_z
            

    