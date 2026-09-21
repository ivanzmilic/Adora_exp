"""
rt_forward.py -- Step 4 of the Adora PINN plan (pinn/plan.md).

Forward model: atmosphere field -> Stokes spectra. Evaluate f_theta(x,y,z), convert its outputs to the
RT's inputs (gas pressure P feeds the MLP EOS -> ne, nHtot; Cartesian B -> |B|, gamma, chi for a vertical
LOS; vz, vturb pass through), and run the differentiable polarised RT via synth_cube_map (Pgas / EOS path,
which now uses the piecewise-linear formal solver). Reused by steps 4-6.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                                   # adora_precision
sys.path.insert(0, os.path.join(REPO, "3dtesting"))        # synth_cube_map
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pinn/
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import numpy as np
import atmosphere_field as af
from synth3d import synth_cube_map


def default_waves(n=201, lam0_nm=630.1, lam1_nm=630.3):
    """Vacuum wavelength grid across the Fe I 6301/6302 pair [nm]."""
    import lightweaver as lw
    return np.linspace(lw.air_to_vac(lam0_nm), lw.air_to_vac(lam1_nm), n)


def field_to_cube(params, coords, shape, chunk=131072):
    """Evaluate the field over `coords` (matching `shape`=(nx,ny,nz)) and build the RT cube dict
    with the KEYS_PG keys (T, Pg, vz, vturb, b, gb, cb), all (nx,ny,nz) numpy."""
    outs = {c: [] for c in af.CHANNELS}
    for s in range(0, coords.shape[0], chunk):
        o = af.forward(params, coords[s:s + chunk])
        for c in af.CHANNELS:
            outs[c].append(np.asarray(o[c]))
    out = {c: np.concatenate(outs[c]) for c in af.CHANNELS}
    b, gb, cb = af.b_to_rt_angles(out["Bx"], out["By"], out["Bz"])
    nx, ny, nz = shape
    r = lambda a: np.asarray(a).reshape(nx, ny, nz)
    return {"T": r(out["T"]), "Pg": r(out["P"]), "vz": r(out["vz"]), "vturb": r(out["vturb"]),
            "b": r(b), "gb": r(gb), "cb": r(cb)}


def synth_field(params, coords, shape, waves, dz_m, batch=2048, forward_only=True, progress=True):
    """Field -> Stokes (nx, ny, 4, nwave). dz_m: (nz,) layer thickness [m]. forward_only=True for a plain
    spectrum; set False (and drop batch) when a gradient through the RT is needed (steps 5-6)."""
    cube = field_to_cube(params, coords, shape)
    return synth_cube_map(waves, dz_m, cube, eos_mode="pg",
                          forward_only=forward_only, batch=batch, progress=progress)
