import os, sys

PINN = "/home/milic/codes/Adora/pinn"
sys.path.insert(0, PINN); 
sys.path.insert(0, os.path.dirname(PINN))
sys.path.insert(0, os.path.join(os.path.dirname(PINN), "eos_mlp"))
#import gpu;  # We don't need gpu for this script
#gpu.use_gpu()

import numpy as np, matplotlib; matplotlib.use("Agg"); 
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
import atmosphere_field as af, losses as L, eos as eos_mod

DOCS = os.path.join(PINN, "docs")

params = af.load_params("/dat/milic/adora_pinn_develop/inverted_field.npz")
eos_params = eos_mod.load_eos("pg")
d = np.load("/dat/milic/adora_pinn_develop/test_atm_falc_128.npz", allow_pickle=True)
z_km = np.asarray(d["z_km"]); 
nx, ny, nz = 128, 128, z_km.size
coords = af.grid_coords(nx, ny, nz)                        # atmosphere grid, same as the original

# per-point T, P and dP/dz (autodiff wrt the z coordinate), batched
def dPdz_norm(c):
    return jax.vmap(jax.grad(lambda cc: af.forward(params, cc[None, :])["P"][0]))(c)[:, 2]

T = np.empty(nx*ny*nz); 
P = np.empty(nx*ny*nz); 
dPz = np.empty(nx*ny*nz)

CH = 65536
for s in range(0, coords.shape[0], CH):
    cb = coords[s:s+CH]; o = af.forward(params, cb)
    T[s:s+CH] = np.asarray(o["T"]); P[s:s+CH] = np.asarray(o["P"])
    dPz[s:s+CH] = np.asarray(dPdz_norm(cb))

dPdz_phys = dPz / float(L.DZDN)                            # Pa/m
_, nhtot = eos_mod.ne_nhtot_from_pg(eos_params, jnp.asarray(T), jnp.asarray(P))
rho = np.asarray(nhtot) * eos_params["mass_per_h"] * L.AMU
rel = (dPdz_phys + rho * L.G_SUN) / (rho * L.G_SUN)        # 0 = perfect HE
rel = rel.reshape(nx, ny, nz); 
ratio = (-dPdz_phys / (rho * L.G_SUN)).reshape(nx, ny, nz)

absmed = np.median(np.abs(rel)); 
rms = np.sqrt(np.mean(rel**2))
frac10 = np.mean(np.abs(rel) < 0.1)

print(f"overall |HE rel residual|: median={absmed:.3f}  RMS={rms:.3f}  frac(|rel|<0.1)={frac10:.2%}")
print(f"dP/dz / (-rho g): median={np.median(ratio):.3f} (1 = HE)   deep(z'=-100) median={np.median(ratio[:,:,0]):.3f}"
      f"   top(z'=1000) median={np.median(ratio[:,:,-1]):.3f}")

med = np.median(np.abs(rel), axis=(0, 1)); 
q1 = np.percentile(np.abs(rel), 25, axis=(0, 1)); 
q3 = np.percentile(np.abs(rel), 75, axis=(0, 1))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
ax[0].fill_between(z_km, q1, q3, alpha=.25); 
ax[0].plot(z_km, med, "C0-")
ax[0].set(xlabel="z' [km]", ylabel="|HE relative residual|", title="HE violation vs height (median, IQR)")
ax[0].axhline(0, color="k", lw=.5); 
ax[0].grid(alpha=.3)
iz = 5; 
m = np.percentile(np.abs(rel[:, :, iz]), 99)

im = ax[1].imshow(rel[:, :, iz].T, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m)

plt.colorbar(im, ax=ax[1]); ax[1].set(title=f"HE relative residual at z'={z_km[iz]:.0f} km")
fig.savefig(os.path.join(DOCS, "fig_he_check.png"), dpi=130); plt.close(fig)
print("saved fig_he_check.png")
