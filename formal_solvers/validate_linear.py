'''
Validation for the piecewise-linear vector formal solver (delo_linear_fs).
  1. FORWARD  : full Stokes RT on FAL-C with the linear solver vs the old constant solver,
                both against Lightweaver (the reference). Linear should track Lightweaver at
                least as well as constant. -> validate_linear.png
  2. GRADIENT : jacrev(delo_linear_fs) vs finite differences on a column whose dtau spans
                1e-5 .. 10, so the small-dtau guard is exercised; checks no NaNs and FD match.
  3. Full-RT gradient finite check (jacrev of the linear RT wrt temperature).
Nothing in the old code is touched; the old solver is imported read-only for comparison.
'''
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                                   # repo root: adora_precision, lineop, old solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this dir: the new solver
import adora_precision
import jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightweaver as lw
from lightweaver.fal import Falc82
from lineop import read_kurucz, emis_opac_polarised, planck
from vector_formal_solver import delo_constant_fs          # old, untouched (read-only)
from vector_formal_solver_linear import delo_linear_fs     # new

HERE = os.path.dirname(os.path.abspath(__file__))


def rt(fs, adata, wave, dz, T, ne, nh, vz, vt, b, gb, cb):
    # mirrors vector_response_fn.lte_polarised_rt, but the formal solver `fs` is a parameter
    eta, chi = jax.vmap(
        emis_opac_polarised, in_axes=[None, None, 0, 0, 0, 0, 0, 0, 0, 0]
    )(adata, wave, T, ne, nh, vz, vt, b, gb, cb)
    I_start = jnp.array([planck(wave, T[0]), 0.0, 0.0, 0.0])
    return fs(dz, I_start, eta, chi)


# --- FAL-C atmosphere (deep boundary first) ---
fal = Falc82()
z = fal.z[::-1]
dz = jnp.array(np.concatenate([[z[0]-z[1]], z[1:]-z[:-1]]))
T  = jnp.array(fal.temperature[::-1]); ne = jnp.array(fal.ne[::-1])
nh = jnp.array(fal.nHTot[::-1]);       vt = jnp.array(fal.vturb[::-1])
vz = jnp.zeros_like(T)
b  = jnp.ones_like(T) * 0.05
gb = jnp.ones_like(T) * 0.785
cb = jnp.zeros_like(T)
waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)
adata = read_kurucz(os.path.join(REPO, "kurucz_6301_6302.linelist"))

print(f"precision: X64={adora_precision.X64}")

# ================================================================= 1. FORWARD vs Lightweaver
rt_wave = lambda fs: jax.jit(jax.vmap(
    lambda w: rt(fs, adata, w, dz, T, ne, nh, vz, vt, b, gb, cb), out_axes=1))(waves)
I_lin = np.asarray(rt_wave(delo_linear_fs))
I_con = np.asarray(rt_wave(delo_constant_fs))

# Lightweaver reference (same setup as vector_response_fn.py __main__)
from lightweaver.rh_atoms import H_atom, Fe23_atom
fal_lw = Falc82()
fal_lw.B = np.ones(T.shape[0]) * 0.05
fal_lw.gammaB = np.ones(T.shape[0]) * 0.785
fal_lw.chiB = np.zeros(T.shape[0])
fal_lw.quadrature(3)
rad_set = lw.RadiativeSet([H_atom(), Fe23_atom()]); rad_set.set_detailed_static("Fe")
spect = rad_set.compute_wavelength_grid(); eq_pops = rad_set.compute_eq_pops(fal_lw)
ctx = lw.Context(fal_lw, spect, eq_pops)
I_lw = np.asarray(ctx.compute_rays(wavelengths=np.array(waves), mus=[1.0], stokes=True))

# the JAX RT and Lightweaver are on different absolute unit scales, so compare each solver's
# profile normalised by ITS OWN continuum (as vector_response_fn.py does).
n_lin = I_lin / I_lin[0, 0]
n_con = I_con / I_con[0, 0]
n_lw  = I_lw  / I_lw[0, 0]
names = ["I", "Q", "U", "V"]
print("\n1. FORWARD (max | (solver/Ic) - (Lightweaver/Ic) |):")
for k in range(4):
    e_lin = np.abs(n_lin[k] - n_lw[k]).max()
    e_con = np.abs(n_con[k] - n_lw[k]).max()
    flag = "linear better" if e_lin <= e_con else "constant better"
    print(f"   {names[k]}: linear={e_lin:.3e}  constant={e_con:.3e}   ({flag})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
ax[0].plot(waves, n_lw[0], 'k-', lw=2, label="Lightweaver")
ax[0].plot(waves, n_con[0], 'C1--', label="DELO constant")
ax[0].plot(waves, n_lin[0], 'C0:', lw=2, label="DELO linear")
ax[0].set(title="Stokes I / Ic", xlabel="wavelength [nm]"); ax[0].legend(fontsize=8)
ax[1].plot(waves, n_lw[3], 'k-', lw=2, label="Lightweaver")
ax[1].plot(waves, n_con[3], 'C1--', label="DELO constant")
ax[1].plot(waves, n_lin[3], 'C0:', lw=2, label="DELO linear")
ax[1].set(title="Stokes V / Ic", xlabel="wavelength [nm]"); ax[1].legend(fontsize=8)
fig.savefig(os.path.join(HERE, "validate_linear.png"), dpi=130)
print("   saved validate_linear.png")

# --- coarse-grid test: SAME opacity code, so this isolates the formal solver. Reference is the
# fine-grid linear spectrum; as the grid coarsens (thicker layers) linear should beat constant.
ref = n_lin
print("\n1b. COARSE GRID (max |coarse/Ic - fine_linear/Ic|) -- isolates the formal solver:")
for f in (4, 8):
    idx = np.arange(0, T.shape[0], f)
    Tc, nec, nhc = T[idx], ne[idx], nh[idx]
    vtc, vzc, bc, gbc, cbc = vt[idx], vz[idx], b[idx], gb[idx], cb[idx]
    zc = z[idx]
    dzc = jnp.array(np.concatenate([[zc[0]-zc[1]], zc[1:]-zc[:-1]]))
    rt_c = lambda fs: jax.jit(jax.vmap(
        lambda w: rt(fs, adata, w, dzc, Tc, nec, nhc, vzc, vtc, bc, gbc, cbc), out_axes=1))(waves)
    Il = np.asarray(rt_c(delo_linear_fs)); Icn = np.asarray(rt_c(delo_constant_fs))
    nl = Il / Il[0, 0]; ncc = Icn / Icn[0, 0]
    el = np.abs(nl - ref).max(); ec = np.abs(ncc - ref).max()
    print(f"   skip={f} (nz={idx.size:3d}): linear={el:.3e}  constant={ec:.3e}  "
          f"({'linear better' if el <= ec else 'constant better'})")

# ================================================================= 2. GRADIENT jacrev vs FD
# toy column whose dtau spans 1e-5 .. 10 (dz=1), exercising the small-dtau guard
nz = 12
dz2 = jnp.ones(nz)
chi0 = jnp.logspace(-5, 1, nz)                              # eta_I -> dtau in [1e-5, 10]
opac2 = jnp.stack([chi0, 0.05*chi0, 0.05*chi0, 0.05*chi0,
                   0.02*chi0, 0.02*chi0, 0.02*chi0], axis=1)
S0 = jnp.linspace(0.5, 1.5, nz)                             # varying source -> linear term matters
emis2 = jnp.stack([S0, 0.05*S0, 0.05*S0, 0.05*S0], axis=1)
I0 = jnp.array([1.0, 0.0, 0.0, 0.0])

jac = jax.jacrev(delo_linear_fs, argnums=(2, 3))(dz2, I0, emis2, opac2)
jac_emis, jac_opac = jac                                    # (4, nz, 4) , (4, nz, 7)
print("\n2. GRADIENT jacrev(delo_linear_fs):")
print(f"   finite: d/demis={bool(jnp.isfinite(jac_emis).all())}  d/dopac={bool(jnp.isfinite(jac_opac).all())}")

dtau_col = 0.5 * (opac2[1:, 0] + opac2[:-1, 0]) * dz2[1:]
j_small = int(jnp.argmin(jnp.abs(dtau_col - 1e-4))) + 1     # a tiny-dtau layer (guard active)
j_large = int(jnp.argmin(jnp.abs(dtau_col - 1.0))) + 1      # a thick layer
def fd(argnum, layer, comp, h_rel=1e-6):
    arr = [dz2, I0, emis2, opac2]
    base = arr[argnum]
    h = h_rel * jnp.maximum(jnp.abs(base[layer, comp]), 1.0)
    ap = base.at[layer, comp].add(h);  am = base.at[layer, comp].add(-h)
    if argnum == 2:
        fp = delo_linear_fs(dz2, I0, ap, opac2); fm = delo_linear_fs(dz2, I0, am, opac2)
    else:
        fp = delo_linear_fs(dz2, I0, emis2, ap); fm = delo_linear_fs(dz2, I0, emis2, am)
    return (fp - fm) / (2*h)                                # (4,) : dI/d(arr[layer,comp])

print(f"   FD check (tiny-dtau layer j={j_small}, dtau={float(dtau_col[j_small-1]):.2e} ; "
      f"thick layer j={j_large}, dtau={float(dtau_col[j_large-1]):.2e}):")
for tag, (arg, jname, layer, comp) in {
    "dI/demis_I @tiny":  (2, jac_emis, j_small, 0),
    "dI/dchi_I  @tiny":  (3, jac_opac, j_small, 0),
    "dI/demis_I @thick": (2, jac_emis, j_large, 0),
    "dI/dchi_I  @thick": (3, jac_opac, j_large, 0),
}.items():
    ad = jname[:, layer, comp]                              # (4,) jacrev
    fdv = fd(arg, layer, comp)
    rel = float(jnp.max(jnp.abs(ad - fdv)) / (jnp.max(jnp.abs(fdv)) + 1e-30))
    print(f"     {tag}: max|jacrev-FD|/|FD| = {rel:.2e}")

# ================================================================= 3. full-RT gradient finite
g = jax.grad(lambda t: jnp.sum(
    rt(delo_linear_fs, adata, waves[100], dz, t, ne, nh, vz, vt, b, gb, cb)))(T)
print(f"\n3. full-RT jacrev wrt T: finite={bool(np.isfinite(np.asarray(g)).all())} "
      f"max|dI/dT|={float(jnp.max(jnp.abs(g))):.3e}")
print("\nDONE")
