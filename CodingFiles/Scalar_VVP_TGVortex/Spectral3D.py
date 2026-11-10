import jax
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
    
    
@jit
def Spectral_Poisson(b):
    k1 = jnp.fft.fftfreq(Nx, L/(Nx*2.0*jnp.pi))      # x
    k2 = jnp.fft.fftfreq(Ny, L/(Ny*2.0*jnp.pi))      # y
    k3 = jnp.fft.rfftfreq(Nz, L/(Nz*2.0*jnp.pi))     # z
    K2, K1, K3 = jnp.meshgrid(k2, k1, k3, indexing='ij')

    kappa = K1**2 + K2**2 + K3**2
    kappa = kappa.at[0, 0, 0].set(1.0)

    b_hat = jnp.fft.rfftn(b)
    b_hat = b_hat.at[0, 0, 0].set(0.0)

    return jnp.fft.irfftn(b_hat / kappa, s=b.shape)

Nx = 32
Ny = 64
Nz = 64
L = 1.0
mesh_x = jnp.linspace(0.0, L, Nx, endpoint=False)
mesh_y = jnp.linspace(0.0, L, Ny, endpoint=False)
mesh_z = jnp.linspace(0.0, L, Nz, endpoint=False)

Y_s, X_s, Z_s = jnp.meshgrid(mesh_y, mesh_x, mesh_z, indexing='ij')
b_spectal = jnp.where(
    (X_s>0.4) & (X_s<0.6) & (Y_s>0.4) & (Y_s<0.6) & (Z_s>0.4) & (Z_s<0.6),
    1.0,
    0.0
)
u0_spectral = jnp.zeros((Ny, Nx, Nz)) 
u_spectral = Spectral_Poisson(b_spectal,)
print(u_spectral.shape)
print(u_spectral.shape[0])
print(u_spectral.shape[1])
print(u_spectral.shape[2])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
ax1 = axes[0]
im1 = ax1.contourf(mesh_y, mesh_z, u_spectral[:,12,:],cmap='plasma')
ax1.set_title('Spectral Method')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(im1, ax=ax1,)

plt.tight_layout()
plt.show()