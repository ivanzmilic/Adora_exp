"""
atmosphere_field.py -- Step 1 of the Adora PINN plan (pinn/plan.md).

A coordinate MLP with anisotropic Gaussian Fourier-feature encoding that predicts the physical
atmosphere parameters as a smooth function of position:

    f_theta : (x, y, z)  ->  (T, P, vz, Bx, By, Bz, vturb)

Mirrors the cloudtorch ParameterField (Tancik Fourier features -> SiLU MLP -> anchored head with
per-channel range transforms) but in pure JAX (params are a plain pytree, like eos_mlp/eos_model.py),
so it composes with the JAX RT/EOS for the fit and physics losses in later steps.

Design choices (agreed 2026-09-20):
  * pure JAX + optax; Cartesian B (smooth, no |B|->0 azimuth degeneracy); vturb is a free output;
  * T linear (softplus, >0), P in log10 (spans ~8 orders), vz/B identity (signed).
The head is ANCHORED: at init (small head weights) the field emits the `anchor` atmosphere everywhere,
so it starts sensible before the step-3 pretraining. Coordinates are normalized to [-1, 1] per axis;
the anisotropic `sigmas` cap how rough the field can get along (x, y) vs the strongly-stratified z.

This module only DEFINES the field (init + forward + a B->RT-angles helper). No optimiser, no data, no RT.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (adora_precision)
import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp

# physical output channels, in head-column order. units: T[K], P[Pa], vz[m/s], B[T], vturb[m/s]
CHANNELS = ("T", "P", "vz", "Bx", "By", "Bz", "vturb")
# per-channel output transform: softplus (>0), log10 (P = 10**raw), or identity (signed)
_KIND = ("softplus", "log10", "id", "id", "id", "id", "softplus")

# init anchor (a mid-photosphere point) and per-channel head sensitivity (physical units per unit
# pre-activation). Both are just starting scales -- step-3 pretraining sets the real field.
DEFAULT_ANCHOR = {"T": 6000.0, "P": 1.0e4, "vz": 0.0, "Bx": 0.0, "By": 0.0, "Bz": 0.0, "vturb": 1.0e3}
DEFAULT_SCALE  = {"T": 1.0e3,  "P": 1.0,   "vz": 3.0e3, "Bx": 0.05, "By": 0.05, "Bz": 0.05, "vturb": 1.0e3}


def _inv_softplus(y):
    # x such that softplus(x) = ln(1+e^x) = y. Stable form x = y + ln(1 - e^-y) (naive ln(e^y-1)
    # overflows for large y, e.g. y=6000 K).
    return y + jnp.log(-jnp.expm1(-y))


def _anchor_pre(kind, value):
    """Pre-activation bias that maps to the anchor `value` under the channel transform."""
    if kind == "softplus":
        return _inv_softplus(jnp.asarray(value))
    if kind == "log10":
        return jnp.log10(jnp.asarray(value))
    return jnp.asarray(value)                       # identity


def init_params(key, d_in=3, n_freq=64, sigmas=(12.0, 12.0, 8.0),
                width=192, depth=4, anchor=None, scale=None, head_std=1e-3):
    """Initialise the field's params pytree.

    sigmas : per-axis Fourier bandwidth (x, y, z); z is the stratified axis, so it gets more.
    anchor : dict of the atmosphere the untrained field emits everywhere (default mid-photosphere).
    scale  : dict of per-channel head sensitivity (physical units per unit pre-activation).
    """
    anchor = {**DEFAULT_ANCHOR, **(anchor or {})}
    scale  = {**DEFAULT_SCALE,  **(scale or {})}
    keys = jax.random.split(key, depth + 2)

    # fixed anisotropic Gaussian Fourier projection: gamma(v) = [sin(2pi B v), cos(2pi B v)]
    B = jax.random.normal(keys[0], (n_freq, d_in)) * jnp.asarray(sigmas)[None, :]
    feat_dim = 2 * n_freq

    dims = [feat_dim] + [width] * depth
    layers = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        W = jnp.sqrt(2.0 / din) * jax.random.normal(keys[i + 1], (din, dout))
        layers.append((W, jnp.zeros((dout,))))

    n_out = len(CHANNELS)
    head_W = head_std * jax.random.normal(keys[-1], (width, n_out))   # small -> start at the anchor
    head_b = jnp.zeros((n_out,))
    bias_pre = jnp.stack([_anchor_pre(k, anchor[c]) for k, c in zip(_KIND, CHANNELS)])
    scale_v  = jnp.asarray([scale[c] for c in CHANNELS])
    return {"B": B, "layers": layers, "head_W": head_W, "head_b": head_b,
            "bias_pre": bias_pre, "scale": scale_v}


def _raw(params, coords):
    """Pre-activation head output (..., 7) for normalized coords (..., 3) in [-1, 1]."""
    proj = 2.0 * jnp.pi * (coords @ params["B"].T)
    h = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
    for W, b in params["layers"]:
        h = jax.nn.silu(h @ W + b)
    return params["bias_pre"] + params["scale"] * (h @ params["head_W"] + params["head_b"])


def forward(params, coords):
    """coords: (..., 3) normalized (x, y, z) in [-1, 1] -> dict of physical params, each (...,)."""
    raw = _raw(params, coords)
    return {
        "T":     jax.nn.softplus(raw[..., 0]),   # K,   > 0
        "P":     10.0 ** raw[..., 1],            # Pa,  > 0  (predicted in log10)
        "vz":    raw[..., 2],                     # m/s, signed
        "Bx":    raw[..., 3],                     # T,   signed
        "By":    raw[..., 4],
        "Bz":    raw[..., 5],
        "vturb": jax.nn.softplus(raw[..., 6]),   # m/s, > 0
    }


def b_to_rt_angles(Bx, By, Bz, eps=1e-8):
    """Cartesian B -> RT inputs (|B|, gamma_b, chi_b) for a vertical (z) line of sight.
    gamma_b = inclination to the z-axis (LOS); chi_b = azimuth in the x-y plane.

    GRADIENT-SAFE at B=0 and along the vertical axis (both singular for the naive sqrt/arccos/arctan2,
    which matters because the inversion STARTS from B=0). Soft-norm with eps^2 under the roots keeps
    |B| and the inclination differentiable everywhere; the longitudinal term |B|cos(gamma)=Bz is still
    reproduced exactly (eps cancels), so dV/dBz stays correct at B=0. The azimuth is undefined at a
    horizontal null, so it is guarded with the double-where trick (the discarded branch never sees the
    origin) -- its contribution vanishes with the transverse field anyway."""
    bmag = jnp.sqrt(Bx * Bx + By * By + Bz * Bz + eps * eps)
    bh = jnp.sqrt(Bx * Bx + By * By + eps * eps)
    gamma_b = jnp.arctan2(bh, Bz)                       # smooth: (bh, Bz) is never (0,0)
    horiz = (Bx * Bx + By * By) > eps * eps
    Bx_s = jnp.where(horiz, Bx, 1.0)                    # guard so arctan2 never sees (0,0)
    By_s = jnp.where(horiz, By, 0.0)
    chi_b = jnp.where(horiz, jnp.arctan2(By_s, Bx_s), 0.0)
    return bmag, gamma_b, chi_b


def save_params(path, params):
    """Save the field params pytree to an .npz."""
    import numpy as np
    flat = {"B": np.asarray(params["B"]), "head_W": np.asarray(params["head_W"]),
            "head_b": np.asarray(params["head_b"]), "bias_pre": np.asarray(params["bias_pre"]),
            "scale": np.asarray(params["scale"]), "n_layers": np.int64(len(params["layers"]))}
    for i, (W, b) in enumerate(params["layers"]):
        flat[f"W{i}"] = np.asarray(W); flat[f"b{i}"] = np.asarray(b)
    np.savez(path, **flat)


def load_params(path):
    """Load a field params pytree saved by save_params."""
    import numpy as np
    d = np.load(path)
    n = int(d["n_layers"])
    return {"B": jnp.asarray(d["B"]), "head_W": jnp.asarray(d["head_W"]),
            "head_b": jnp.asarray(d["head_b"]), "bias_pre": jnp.asarray(d["bias_pre"]),
            "scale": jnp.asarray(d["scale"]),
            "layers": [(jnp.asarray(d[f"W{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(n)]}


def grid_coords(nx, ny, nz):
    """Normalized (x, y, z) coords in [-1, 1] for a regular (nx, ny, nz) grid, flattened in
    (x, y, z) order -> (nx*ny*nz, 3)."""
    ax = lambda n: jnp.zeros(1) if n == 1 else jnp.linspace(-1.0, 1.0, n)
    xx, yy, zz = jnp.meshgrid(ax(nx), ax(ny), ax(nz), indexing="ij")
    return jnp.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=-1)


if __name__ == "__main__":
    import numpy as np
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    n_par = sum(int(np.prod(w.shape)) for w, _ in params["layers"]) + \
            int(np.prod(params["head_W"].shape))
    coords = grid_coords(8, 8, 40)                       # a small (x,y,z) grid
    out = forward(params, coords)
    print(f"params ~ {n_par} weights;  coords {coords.shape}")
    for c in CHANNELS:
        a = np.asarray(out[c])
        print(f"  {c:5s} min={a.min():+.3e} max={a.max():+.3e}")
    # at init the field should be ~the anchor everywhere (small head weights)
    print(f"  anchor check: <T>={float(out['T'].mean()):.1f}K  <P>={float(out['P'].mean()):.3e}Pa "
          f"<vturb>={float(out['vturb'].mean()):.1f}m/s")
