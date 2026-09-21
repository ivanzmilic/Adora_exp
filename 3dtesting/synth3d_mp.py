# Process-parallel 1.5D Stokes synthesis: a pool of core-pinned single-device workers.
# Scales the synth3d engine across CPU cores. This module stays JAX-free at import so
# multiprocessing "spawn" workers start with a clean JAX (see bench: pmap is a dead end).
#
# COLD-START: each call spawns a fresh pool -> every worker cold-imports JAX and JIT-compiles
# on its first tile (~60-90 s fixed for tens of workers). The pool then REUSES each worker across
# many tiles (compile once, warm thereafter), which also lets results stream back for a progress
# bar. For iterative use (inversions) a PERSISTENT pool across calls is still the real next step.
import os, sys
import numpy as np
import multiprocessing as mp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS_DIR)
KEYS     = ("T", "ne", "nH", "vz", "vturb", "b", "gb", "cb")   # synth_cube_fwd arg order after dz (legacy ne/nH path)
KEYS_PG  = ("T", "Pg",  "vz", "vturb", "b", "gb", "cb")        # eos: cube keys for the Pgas head
KEYS_RHO = ("T", "rho", "vz", "vturb", "b", "gb", "cb")        # eos: cube keys for the rho head
DEFAULT_LINELIST = os.path.join(REPO, "kurucz_6301_6302.linelist")

# per-process state, set once by the pool initializer so the pin, the JAX import and the line-list
# read happen a single time per worker and are reused across every tile it handles.
_W = {}

def _init(counter, lock, ncpu, cores_per, linelist, waves, dz, eos_mode=None):   # eos: eos_mode in {None, "pg", "rho"}
    with lock:                                     # hand out a distinct worker id -> distinct cores
        wid = counter.value
        counter.value += 1
    cores = [(wid * cores_per + j) % ncpu for j in range(cores_per)]
    os.sched_setaffinity(0, set(cores))            # pin BEFORE JAX inits its thread pool
    sys.path[:0] = [THIS_DIR, REPO]
    import jax
    import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
    import jax.numpy as jnp
    from synth3d import synth_cube_fwd, build_synth_cube_eos   # importing synth3d also puts eos_mlp on sys.path  # eos
    from lineop import read_kurucz
    if eos_mode:                                   # eos: build a Pgas/rho-driven forward synth for this worker
        import eos as eos_mod                      # eos_mlp/eos.py (its dir is on sys.path via the synth3d import above)  # eos
        eos_params = eos_mod.load_eos(eos_mode)    # eos: load trained MLP params once per worker (jax arrays -> build after JAX import)
        synth = build_synth_cube_eos(eos_params, forward_only=True)   # eos
        keys = KEYS_PG if eos_mode == "pg" else KEYS_RHO             # eos
    else:
        synth, keys = synth_cube_fwd, KEYS         # legacy ne/nH path (no EOS)
    _W.update(jnp=jnp, synth=synth, keys=keys, adata=read_kurucz(linelist),
              waves=jnp.asarray(waves), dz=jnp.asarray(dz))

# one tile: synth its columns on this (already pinned, already warm) worker. s = start column index.
def _worker(payload):
    s, chunk = payload
    jnp = _W["jnp"]
    a = [jnp.asarray(chunk[k]) for k in _W["keys"]]   # eos: keys set by _init per mode (ne/nH, Pg, or rho)
    out = _W["synth"](_W["adata"], _W["waves"], _W["dz"], *a)
    out.block_until_ready()
    return s, np.asarray(out)

# cube: dict of the 8 KEYS, each (nx,ny,nz) or (npix,nz). Returns (nx,ny,4,nwave) or (npix,4,nwave).
# nworkers=None -> one worker per available core. Each worker is an independent JAX process (~1 GB),
# so on many-core nodes pass a smaller nworkers to bound memory.
# tile     -> columns per task (default ~4 tasks/worker); smaller = smoother progress bar, a touch
#             more overhead. progress -> show a tqdm bar over the tiles (falls back to silent if
#             tqdm is missing). NOTE: uses "spawn" -- any caller script MUST run this under
#             `if __name__ == "__main__":`, or the re-imported module re-spawns recursively.
def synth_cube_mp(waves, dz, cube, nworkers=None, cores_per=1, linelist=DEFAULT_LINELIST,
                  tile=None, progress=True, eos_mode=None):   # eos: eos_mode None=legacy ne/nH, "pg"=cube has "Pg", "rho"=cube has "rho"
    ncpu = len(os.sched_getaffinity(0))
    if nworkers is None:
        nworkers = ncpu // cores_per
    waves, dz = np.asarray(waves), np.asarray(dz)
    keys = KEYS if eos_mode is None else (KEYS_PG if eos_mode == "pg" else KEYS_RHO)   # eos: which cube keys to ship to workers
    shape = np.asarray(cube["T"]).shape
    map_shape = shape[:-1] if cube["T"].ndim == 3 else None   # remember (nx,ny) to restore
    nz = shape[-1]
    flat = {k: np.ascontiguousarray(np.asarray(cube[k])).reshape(-1, nz) for k in keys}
    npix = flat["T"].shape[0]

    nworkers = max(1, min(nworkers, npix))
    # split columns into fixed-size tiles (~4 per worker) so results stream back for the bar; a
    # uniform tile size means each worker's forward synth compiles once and is then reused.
    if tile is None:
        tile = max(1, -(-npix // (nworkers * 4)))
    payloads = [(s, {k: flat[k][s:s + tile] for k in keys}) for s in range(0, npix, tile)]

    ctx = mp.get_context("spawn")
    counter, lock = ctx.Value("i", 0), ctx.Lock()
    out = np.empty((npix, 4, waves.shape[0]), dtype=np.float64)
    with ctx.Pool(nworkers, initializer=_init,
                  initargs=(counter, lock, ncpu, cores_per, linelist, waves, dz, eos_mode)) as pool:   # eos: pass mode to workers
        results = pool.imap_unordered(_worker, payloads)
        if progress:
            try:
                from tqdm import tqdm
                results = tqdm(results, total=len(payloads), desc="synth", unit="tile")
            except ImportError:
                pass
        for s, chunk_out in results:                          # arrives out of order -> place by s
            out[s:s + chunk_out.shape[0]] = chunk_out
    return out.reshape(*map_shape, 4, waves.shape[0]) if map_shape else out


if __name__ == "__main__":
    import time
    import lightweaver as lw
    from lightweaver.fal import Falc82

    # toy FALC cube (same setup as synth3d.py), varied kinematics/field across the map
    fal = Falc82()
    z = fal.z[::-1]; nz = z.shape[0]
    dz = np.concatenate([[z[0] - z[1]], z[1:] - z[:-1]])
    T0, ne0, nH0, vt0 = fal.temperature[::-1], fal.ne[::-1], fal.nHTot[::-1], fal.vturb[::-1]
    nx, ny = 40, 40
    bcast = lambda x: np.broadcast_to(x, (nx, ny, nz))
    cube = {"T": bcast(T0), "ne": bcast(ne0), "nH": bcast(nH0), "vturb": bcast(vt0),
            "vz": np.broadcast_to(np.linspace(-4e3, 4e3, nx)[:, None, None], (nx, ny, nz)),
            "b":  np.broadcast_to(np.linspace(0, 0.15, ny)[None, :, None], (nx, ny, nz)),
            "gb": np.full((nx, ny, nz), 0.6), "cb": np.zeros((nx, ny, nz))}
    waves = np.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 101)

    t0 = time.time()
    I = synth_cube_mp(waves, dz, cube, nworkers=32)   # 32 single-core workers
    dt = time.time() - t0
    print(f"Stokes cube {I.shape} in {dt:.1f}s (spawn+compile+run), "
          f"finite={np.isfinite(I).all()}, Ic~{I[0,0,0,0]:.3e}")
