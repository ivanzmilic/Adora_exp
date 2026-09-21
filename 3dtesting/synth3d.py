# Minimal 1.5D Stokes synthesis from a 3D atmosphere cube (column-by-column).
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eos_mlp"))   # eos: eos_mlp uses flat sibling imports (import eos_config/eos_model), so its dir must be on sys.path

import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
import numpy as np                                              # for the batched path (host-side concat)
from lineop import read_kurucz
from vector_response_fn import lte_polarised_rt
import eos                                                      # eos: eos_mlp/eos.py -- differentiable MLP EOS, (T, Pgas|rho) -> (ne, nHtot)

KEYS     = ("T", "ne", "nH", "vz", "vturb", "b", "gb", "cb")   # synth_cube arg order after dz (legacy ne/nH path)
KEYS_PG  = ("T", "Pg",  "vz", "vturb", "b", "gb", "cb")        # eos: cube keys for the Pgas head (Pg replaces ne, nH)
KEYS_RHO = ("T", "rho", "vz", "vturb", "b", "gb", "cb")        # eos: cube keys for the rho head (rho replaces ne, nH)
DEFAULT_LINELIST = os.path.join(REPO, "kurucz_6301_6302.linelist")

# one column, all wavelengths -> Stokes (4, nwave)
def synth_column(adata, waves, dz, T, ne, nH, vz, vturb, b, gb, cb):
    return jax.vmap(
        lte_polarised_rt,
        in_axes=[None, 0, None, None, None, None, None, None, None, None, None],
        out_axes=1,
    )(adata, waves, dz, T, ne, nH, vz, vturb, b, gb, cb)

# cube: atmosphere params shaped (npix, nz) -> Stokes (npix, 4, nwave)
def _synth_cube(adata, waves, dz, T, ne, nH, vz, vturb, b, gb, cb):
    return jax.vmap(
        synth_column,
        in_axes=[None, None, None, 0, 0, 0, 0, 0, 0, 0, 0],
    )(adata, waves, dz, T, ne, nH, vz, vturb, b, gb, cb)

# jit a pixel-level cube fn, optionally as a forward-only twin: stop_gradient makes the result a
# CONSTANT to any outer autodiff (torch or jax) -> no tape is ever built. Used by the CPU mp workers
# and torch inversions where the gradient is supplied elsewhere. Output is bit-identical either way.
def _finalize(cube_fn, forward_only):
    if forward_only:
        return jax.jit(lambda *a: jax.lax.stop_gradient(cube_fn(*a)))
    return jax.jit(cube_fn)

# ne/nH path: two entry points sharing one engine.
synth_cube     = _finalize(_synth_cube, forward_only=False)   # differentiable-ready (response wraps in jacrev/grad)
synth_cube_fwd = _finalize(_synth_cube, forward_only=True)    # forward-only twin


# --- EOS-driven synth: Pgas (or rho) per depth instead of free ne, nH -------------------  # eos
# The trained MLP is FIXED, so we bake it into the jitted function (closed over) rather than
# threading its params dict through jit. The EOS runs ONCE per column (above the wavelength
# vmap), and reuses synth_column below it -- so the only new physics is the one convert() line.
# Because the EOS lives inside the traced/jitted function, jacrev/grad of the returned fn gives
# d Stokes / d Pgas straight through the EOS: this is the entry point the inversion differentiates.

def build_synth_cube_eos(eos_nn, forward_only=False):                                   # eos
    convert = eos.ne_nhtot_from_pg if eos_nn["mode"] == "pg" \
        else eos.ne_nhtot_from_rho                                                          # eos: pick head by mode
    def _column(adata, waves, dz, T, P, vz, vturb, b, gb, cb):
        ne, nH = convert(eos_nn, T, P)   # eos: (T, Pgas|rho) -> (ne, nHtot), once per column, (nz,)
        return synth_column(adata, waves, dz, T, ne, nH, vz, vturb, b, gb, cb)              # reuse the wavelength engine
    cube_fn = jax.vmap(_column, in_axes=[None, None, None, 0, 0, 0, 0, 0, 0, 0])            # over pixels (7 atmos args)
    return _finalize(cube_fn, forward_only)                                                 # eos: same jit/fwd wrapper as the ne/nH path


_EOS_SYNTH_CACHE = {}   # eos: cache built (jitted) fns so repeat/inversion calls don't recompile

# Dict-cube entry point (GPU). Whole cube in one call (returns an on-device JAX array that composes
# with grad/jit), or set `batch` to chunk the columns and cap GPU memory (returns a host numpy array).
# One wrapper, two physics paths:
#   - default (eos_nn=None, eos_mode=None): legacy ne/nH cube, keys = KEYS.
#   - EOS path (set `eos_mode` or pass `eos_nn`): cube carries "Pg"/"rho" instead of "ne","nH"; the
#     MLP supplies (ne, nHtot) and the RT core is unchanged. keys = KEYS_PG / KEYS_RHO.
#   adata -> preloaded line list to reuse across calls (inversion loops); None reads `linelist`.
#   batch -> columns per chunk to bound peak GPU memory (None = whole cube at once).
#   progress -> tqdm bar over the chunks (batched path; silently skipped if tqdm is missing).

def synth_cube_map(waves, dz, cube, eos_nn=None, eos_mode=None,
                   linelist=DEFAULT_LINELIST, adata=None, forward_only=False, batch=None, progress=True):

    '''
    waves : array of wavelengths (in vacuum)
    dz : array of grid spacings along the z-axis, this is read from the header
    cube : dictionary containing atmospheric parameters
    eos_nn : trained EOS neural network, if None, then we revert to the old ne/nH way
    eos_mode : mode for EOS ("pg" or "rho")
    linelist : path to the linelist file
    adata : preloaded line list data (optional)
    forward_only : if True, only perform forward synthesis without gradients
    batch : columns to synthesise per chunk, to cap peak GPU memory. None (default) does the
            whole cube in one call and returns an on-device JAX array. When set, the columns are
            processed in chunks and pulled to host, so the return is a numpy array -- meant for
            large forward runs (there is no gradient across the chunk loop).
    progress : show a tqdm progress bar over the chunks (only used when batch is set); silently
               skipped if tqdm is not installed.
    '''

    # If I did not pass specific atomic data load whatever is default
    if adata is None:
        adata = read_kurucz(linelist)

    # Convert to JAX arrays
    waves, dz = jnp.asarray(waves), jnp.asarray(dz)

    # Determine which synthesis path to use: legacy ne/nH or EOS-based
    if eos_nn is None and eos_mode is None:                       # legacy ne/nH path
        fn, keys = (synth_cube_fwd if forward_only else synth_cube), KEYS

    else:                                                         # eos: Pgas/rho path
        if eos_nn is None:
            eos_nn = eos.load_eos(eos_mode)                       # eos: load trained MLP once
            
        cache_key = (id(eos_nn), forward_only)
        fn = _EOS_SYNTH_CACHE.get(cache_key) or _EOS_SYNTH_CACHE.setdefault(
            cache_key, build_synth_cube_eos(eos_nn, forward_only))                    # eos
        keys = KEYS_PG if eos_nn["mode"] == "pg" else KEYS_RHO                        # eos: cube keys by head

    # Flatten the (nx,ny,nz) map to a stack of columns (npix, nz) on the HOST, one array per key.
    # Keeping `flat` in numpy means GPU residency is bounded by `batch`, not the whole cube -- read
    # the shape without uploading anything.
    shp = cube["T"].shape
    map_shape = shp[:-1] if len(shp) == 3 else None              # remember (nx,ny) to restore
    nz = shp[-1]
    flat = [np.asarray(cube[k]).reshape(-1, nz) for k in keys]   # host (numpy)
    npix = flat[0].shape[0]

    # batch=None -> whole cube in one call. jit uploads the numpy inputs; returns an on-device JAX
    # array (composes with grad/jit).
    if batch is None or batch >= npix:
        I = fn(adata, waves, dz, *flat)                          # (npix, 4, nwave)

    # Otherwise chunk the columns to cap peak memory (~ batch*nwave*nz*4*4*8 bytes). `fn` is the same
    # object every chunk, so it compiles once and stays warm; only a smaller final chunk recompiles
    # once. Only the current chunk is moved to the GPU and its intensity pulled back -> result is numpy.
    else:
        starts = range(0, npix, batch)                           # first column index of each chunk
        if progress:                                             # optional tqdm bar over the chunks
            try:
                from tqdm import tqdm
                starts = tqdm(starts, total=len(starts), desc="synth", unit="chunk")
            except ImportError:
                pass
        chunks = []
        for s in starts:
            sub = [jnp.asarray(a[s:s + batch]) for a in flat]    # host slice -> GPU (only this chunk)
            chunks.append(np.asarray(fn(adata, waves, dz, *sub)))  # intensity -> host, frees the GPU
        I = np.concatenate(chunks, axis=0)                       # (npix, 4, nwave) on host

    return I.reshape(*map_shape, 4, I.shape[-1]) if map_shape is not None else I


if __name__ == "__main__":

    import time
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import lightweaver as lw
    from lightweaver.fal import Falc82

    # Read the linelist(s)
    adata = read_kurucz(os.path.join(REPO, "kurucz_6301_6302.linelist"))

    # base 1D column from FALC (top of atmosphere first), shared z-grid -> single dz
    fal = Falc82()
    nz = fal.temperature.shape[0]
    dz = jnp.array(np.concatenate([[fal.z[::-1][0] - fal.z[::-1][1]],
                                   fal.z[::-1][1:] - fal.z[::-1][:-1]]))
    T0, ne0 = jnp.array(fal.temperature[::-1]), jnp.array(fal.ne[::-1])
    nH0, vt0 = jnp.array(fal.nHTot[::-1]), jnp.array(fal.vturb[::-1])

    # toy cube: same thermodynamics everywhere; vary kinematics/field across the map
    nx, ny = 6, 6
    tile = lambda a: jnp.broadcast_to(a, (nx, ny, nz))
    T, ne, nH, vturb = tile(T0), tile(ne0), tile(nH0), tile(vt0)
    vz = jnp.broadcast_to(jnp.linspace(-4e3, 4e3, nx)[:, None, None], (nx, ny, nz))  # LOS vel ramp [m/s]
    b  = jnp.broadcast_to(jnp.linspace(0.0, 0.15, ny)[None, :, None], (nx, ny, nz))  # |B| ramp [T]
    gb = jnp.full((nx, ny, nz), 0.6)   # inclination [rad]
    cb = jnp.zeros((nx, ny, nz))       # azimuth [rad]

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 101)

    # flatten pixels -> synth -> reshape back to maps
    flat = lambda a: a.reshape(nx * ny, nz)
    t0 = time.time()
    I = synth_cube(adata, waves, dz, flat(T), flat(ne), flat(nH),
                   flat(vz), flat(vturb), flat(b), flat(gb), flat(cb))
    I.block_until_ready()
    I = np.asarray(I).reshape(nx, ny, 4, waves.shape[0])   # (nx, ny, Stokes, nwave)
    print(f"Stokes cube {I.shape} in {time.time() - t0:.1f}s (compile+run)")

    # quick check: I across the velocity ramp, V across the field ramp
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for i in range(nx):
        ax[0].plot(waves, I[i, 0, 0] / I[i, 0, 0, 0], label=f"vz={np.linspace(-4,4,nx)[i]:+.1f} km/s")
    ax[0].set(title="Stokes I (velocity ramp along x)", xlabel="wavelength [nm]")
    ax[0].legend(fontsize=7)
    for j in range(ny):
        ax[1].plot(waves, I[0, j, 3] / I[0, j, 0, 0], label=f"|B|={np.linspace(0,150,ny)[j]:.0f} G")
    ax[1].set(title="Stokes V (field ramp along y)", xlabel="wavelength [nm]")
    ax[1].legend(fontsize=7)
    fig.savefig(os.path.join(os.path.dirname(__file__), "synth3d.png"), dpi=150)
    print("saved synth3d.png")
