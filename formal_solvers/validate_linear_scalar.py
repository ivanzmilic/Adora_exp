'''
Validation for the piecewise-linear scalar formal solver (linear_fs).
  1. FORWARD  : scalar (unpolarised) RT on FAL-C with linear_fs vs the old nearest_fs, both against
                Lightweaver (Stokes I). Normalised by each solver's own continuum.
  2. COARSE   : same opacity code, fine-grid linear as truth -> isolates the solver.
  3. GRADIENT : jacrev(linear_fs) vs finite differences on a column with dtau in 1e-5 .. 10.
The old scalar_formal_solver.py is imported read-only for comparison; nothing old is touched.
'''
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adora_precision
import jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightweaver as lw
from lightweaver.fal import Falc82
from lineop import read_kurucz, emis_opac
from scalar_formal_solver import nearest_fs               # old, untouched (read-only)
from scalar_formal_solver_linear import linear_fs         # new

HERE = os.path.dirname(os.path.abspath(__file__))


def rt(fs, adata, wave, dz, T, ne, nh, vz, vt):
    # mirrors response_fn.lte_rt, but the formal solver `fs` is a parameter
    eta, chi = jax.vmap(
        emis_opac, in_axes=[None, None, 0, 0, 0, 0, 0]
    )(adata, wave, T, ne, nh, vz, vt)
    return fs(dz, eta, chi)


fal = Falc82()
z = fal.z[::-1]
dz = jnp.array(np.concatenate([[z[0]-z[1]], z[1:]-z[:-1]]))
T  = jnp.array(fal.temperature[::-1]); ne = jnp.array(fal.ne[::-1])
nh = jnp.array(fal.nHTot[::-1]);       vt = jnp.array(fal.vturb[::-1])
vz = jnp.zeros_like(T)
waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)
adata = read_kurucz(os.path.join(REPO, "kurucz_6301_6302.linelist"))
print(f"precision: X64={adora_precision.X64}")

# ================================================================= 1. FORWARD vs Lightweaver
rt_wave = lambda fs: jax.jit(jax.vmap(
    lambda w: rt(fs, adata, w, dz, T, ne, nh, vz, vt)))(waves)
I_lin = np.asarray(rt_wave(linear_fs))
I_near = np.asarray(rt_wave(nearest_fs))

from lightweaver.rh_atoms import H_atom, Fe23_atom
fal_lw = Falc82(); fal_lw.quadrature(3)
rad_set = lw.RadiativeSet([H_atom(), Fe23_atom()]); rad_set.set_detailed_static("Fe")
spect = rad_set.compute_wavelength_grid(); eq_pops = rad_set.compute_eq_pops(fal_lw)
ctx = lw.Context(fal_lw, spect, eq_pops)
I_lw = np.asarray(ctx.compute_rays(wavelengths=np.array(waves), mus=[1.0])).squeeze()

n_lin, n_near, n_lw = I_lin/I_lin[0], I_near/I_near[0], I_lw/I_lw[0]
print("\n1. FORWARD (max | (solver/Ic) - (Lightweaver/Ic) |):")
e_lin = np.abs(n_lin - n_lw).max(); e_near = np.abs(n_near - n_lw).max()
print(f"   I: linear={e_lin:.3e}  nearest={e_near:.3e}   "
      f"({'linear better' if e_lin <= e_near else 'nearest better'})")

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.plot(waves, n_lw, 'k-', lw=2, label="Lightweaver")
ax.plot(waves, n_near, 'C1--', label="nearest (constant)")
ax.plot(waves, n_lin, 'C0:', lw=2, label="linear")
ax.set(title="Scalar Stokes I / Ic", xlabel="wavelength [nm]"); ax.legend(fontsize=8)
fig.savefig(os.path.join(HERE, "validate_linear_scalar.png"), dpi=130)
print("   saved validate_linear_scalar.png")

# ================================================================= 2. COARSE GRID (isolates solver)
ref = n_lin
print("\n2. COARSE GRID (max |coarse/Ic - fine_linear/Ic|) -- isolates the formal solver:")
for f in (4, 8):
    idx = np.arange(0, T.shape[0], f)
    Tc, nec, nhc, vtc, vzc = T[idx], ne[idx], nh[idx], vt[idx], vz[idx]
    zc = z[idx]; dzc = jnp.array(np.concatenate([[zc[0]-zc[1]], zc[1:]-zc[:-1]]))
    rt_c = lambda fs: jax.jit(jax.vmap(lambda w: rt(fs, adata, w, dzc, Tc, nec, nhc, vzc, vtc)))(waves)
    Il = np.asarray(rt_c(linear_fs)); Inr = np.asarray(rt_c(nearest_fs))
    nl = Il/Il[0]; nn = Inr/Inr[0]
    el = np.abs(nl - ref).max(); en = np.abs(nn - ref).max()
    print(f"   skip={f} (nz={idx.size:3d}): linear={el:.3e}  nearest={en:.3e}  "
          f"({'linear better' if el <= en else 'nearest better'})")

# ================================================================= 3. GRADIENT jacrev vs FD
nz = 12
dz2 = jnp.ones(nz)
opac2 = jnp.logspace(-5, 1, nz)                            # dtau in [1e-5, 10]
emis2 = jnp.linspace(1e-6, 3e-6, nz)                       # varying source S = emis/opac
jac_emis, jac_opac = jax.jacrev(linear_fs, argnums=(1, 2))(dz2, emis2, opac2)
print("\n3. GRADIENT jacrev(linear_fs):")
print(f"   finite: d/demis={bool(jnp.isfinite(jac_emis).all())}  d/dopac={bool(jnp.isfinite(jac_opac).all())}")

dtau_col = 0.5 * (opac2[1:] + opac2[:-1]) * dz2[1:]
j_small = int(jnp.argmin(jnp.abs(dtau_col - 1e-4))) + 1
j_large = int(jnp.argmin(jnp.abs(dtau_col - 1.0))) + 1
def fd(argnum, layer, h_rel=1e-6):
    base = [dz2, emis2, opac2][argnum]
    h = h_rel * jnp.maximum(jnp.abs(base[layer]), 1e-12)
    ap = base.at[layer].add(h); am = base.at[layer].add(-h)
    if argnum == 1:
        return (linear_fs(dz2, ap, opac2) - linear_fs(dz2, am, opac2)) / (2*h)
    return (linear_fs(dz2, emis2, ap) - linear_fs(dz2, emis2, am)) / (2*h)

print(f"   FD check (tiny j={j_small}, dtau={float(dtau_col[j_small-1]):.2e} ; "
      f"thick j={j_large}, dtau={float(dtau_col[j_large-1]):.2e}):")
for tag, (arg, jac, layer) in {
    "dI/demis @tiny":  (1, jac_emis, j_small),
    "dI/dopac @tiny":  (2, jac_opac, j_small),
    "dI/demis @thick": (1, jac_emis, j_large),
    "dI/dopac @thick": (2, jac_opac, j_large),
}.items():
    ad = float(jac[layer]); fdv = float(fd(arg, layer))
    rel = abs(ad - fdv) / (abs(fdv) + 1e-30)
    print(f"     {tag}: |jacrev-FD|/|FD| = {rel:.2e}")
print("\nDONE")
