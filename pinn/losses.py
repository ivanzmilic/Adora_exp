"""
losses.py -- Step 6 of the Adora PINN plan (pinn/plan.md).

The three losses for the inversion, all differentiable w.r.t. the field params:
  * fit_loss      -- chi^2 of the synthesized Stokes vs the observed testset, on random pixels
                     (2a-2c: per pixel a z-grid, RT over the wavelength grid, weighted MSE).
  * he_loss       -- hydrostatic equilibrium dP/dz = -rho g in PHYSICAL units, at random collocation
                     points; rho from the MLP EOS given (T, P). dP/dz via autodiff of the field w.r.t. z
                     with the correct z_norm -> z_phys chain rule.
  * boundary_loss -- soft constraint on the gas pressure at the top boundary (z'=+1).
Plus directional_fd: a finite-difference check that grad(loss) w.r.t. the params makes sense.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "3dtesting"))
sys.path.insert(0, os.path.join(REPO, "eos_mlp"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adora_precision   # global float32/64 switch
import jax, jax.numpy as jnp
import atmosphere_field as af
import eos as eos_mod
from synth3d import build_synth_cube_eos

# physical constants / domain
G_SUN = 274.0                       # m/s^2, solar surface gravity
AMU = 1.66053906660e-27            # kg
Z_RANGE_M = (1000.0 - (-100.0)) * 1.0e3   # z' span [m]; z_norm in [-1,1] maps onto this
DZDN = Z_RANGE_M / 2.0             # d z_phys / d z_norm  [m per unit z_norm]
NX = NY = 128

# per-channel scales for the smoothness normalization = the field's own head sensitivities
# (T[K], P[log10 dex], vz[m/s], B[T], vturb[m/s]). Fixed constants, all nonzero -> the horizontal-gradient
# penalty is dimensionless, comparable across channels, and never divides by a near-zero quantity.
SMOOTH_SCALE = jnp.asarray([af.DEFAULT_SCALE[c] for c in af.CHANNELS])


def make_synth(eos_params):
    """Grad-ready pixel synth fn: (adata, waves, dz, T, Pg, vz, vturb, b, gb, cb) -> (npix,4,nwave)."""
    return build_synth_cube_eos(eos_params, forward_only=False)


def pixel_column_coords(ix, iy, nz):
    """Normalized (x,y,z) coords for the columns at pixel indices (ix,iy) -> (M*nz, 3)."""
    M = ix.shape[0]
    xn = 2.0 * ix / (NX - 1) - 1.0
    yn = 2.0 * iy / (NY - 1) - 1.0
    zn = jnp.linspace(-1.0, 1.0, nz)
    X = jnp.broadcast_to(xn[:, None], (M, nz)).reshape(-1)
    Y = jnp.broadcast_to(yn[:, None], (M, nz)).reshape(-1)
    Z = jnp.broadcast_to(zn[None, :], (M, nz)).reshape(-1)
    return jnp.stack([X, Y, Z], axis=-1)


def fit_loss(params, synth_fn, adata, obs_batch, ix, iy, nz, waves, dz, sigma):
    """chi^2 of synthesized Stokes vs obs_batch (M,4,nwave) at pixels (ix,iy). sigma: per-Stokes (4,)."""
    M = ix.shape[0]
    out = af.forward(params, pixel_column_coords(ix, iy, nz))
    r = lambda a: a.reshape(M, nz)
    b, gb, cb = af.b_to_rt_angles(out["Bx"], out["By"], out["Bz"])
    I = synth_fn(adata, waves, dz, r(out["T"]), r(out["P"]), r(out["vz"]), r(out["vturb"]),
                 r(b), r(gb), r(cb))                                    # (M,4,nwave)
    return jnp.mean(((I - obs_batch) / sigma[None, :, None]) ** 2)


def he_loss(params, eos_params, colloc):
    """Relative hydrostatic-equilibrium residual (dP/dz + rho g)/(rho g) at collocation pts (K,3)."""
    out = af.forward(params, colloc)
    T, P = out["T"], out["P"]
    dPdz_norm = jax.vmap(jax.grad(lambda c: af.forward(params, c[None, :])["P"][0]))(colloc)[:, 2]
    dPdz_phys = dPdz_norm / DZDN                                        # Pa/m
    _, nhtot = eos_mod.ne_nhtot_from_pg(eos_params, T, P)
    rho = nhtot * eos_params["mass_per_h"] * AMU                        # kg/m^3
    resid = dPdz_phys + rho * G_SUN                                     # = 0 in HE
    return jnp.mean((resid / (rho * G_SUN)) ** 2)


def boundary_loss(params, ix, iy, P_top):
    """Soft top-boundary gas-pressure constraint (log10) at z_norm=+1 for pixels (ix,iy)."""
    xn = 2.0 * ix / (NX - 1) - 1.0
    yn = 2.0 * iy / (NY - 1) - 1.0
    coords = jnp.stack([xn, yn, jnp.ones_like(xn)], axis=-1)
    P = af.forward(params, coords)["P"]
    return jnp.mean((jnp.log10(P) - jnp.log10(P_top)) ** 2)


def smooth_loss(params, colloc, w_ch=None):
    """Horizontal (x,y) smoothness of the field at collocation points (K,3).
    Penalizes (df/dx)^2 + (df/dy)^2 per channel on the PRE-TRANSFORM outputs af._raw -- so P is smoothed
    in log10 (depth-unbiased) and T, vz, B, vturb in their physical units -- each normalized by the fixed
    SMOOTH_SCALE so the per-channel terms are dimensionless and comparable. Only the in-layer (x,y)
    derivatives enter, so the vertical (z) stratification is untouched: this damps the Fourier-feature
    stripes without flattening the atmosphere. w_ch: optional per-channel weights (7,); None -> equal."""
    raw1 = lambda c: af._raw(params, c[None, :])[0]                    # (7,) pre-transform outputs at a point
    J = jax.vmap(jax.jacfwd(raw1))(colloc)                            # (K,7,3); forward-mode (d_in=3 < 7)
    gxy = J[..., :2] / SMOOTH_SCALE[None, :, None]                     # normalized in-layer derivs (K,7,2)
    per_ch = jnp.mean(gxy[..., 0] ** 2 + gxy[..., 1] ** 2, axis=0)     # (7,) mean over collocation points
    return jnp.mean(per_ch) if w_ch is None else jnp.sum(jnp.asarray(w_ch) * per_ch)


# --------------------------------------------------------------- gradient check
def directional_fd(loss_of_params, params, v, eps=1e-4):
    """Central FD of loss along direction v (a params-shaped pytree) vs the analytic grad.
    Returns (fd, analytic=<grad,v>, rel_err)."""
    add = lambda a, s: jax.tree_util.tree_map(lambda p, d: p + s * d, params, v)
    fp = loss_of_params(add(params, eps))
    fm = loss_of_params(add(params, -eps))
    fd = float((fp - fm) / (2 * eps))
    g = jax.grad(loss_of_params)(params)
    analytic = float(sum(jnp.vdot(gi, vi) for gi, vi in
                         zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(v))))
    rel = abs(fd - analytic) / (abs(analytic) + 1e-30)
    return fd, analytic, rel


def random_like(params, key, scale=1.0):
    """A params-shaped pytree of N(0,scale) entries (a random FD direction)."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    new = [scale * jax.random.normal(k, jnp.shape(l)) for k, l in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, new)
