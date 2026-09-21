# Load a MURaM snapshot into an Adora synth cube.  MURaM is CGS; Adora is SI -- every
# quantity is converted here (see the *_TO_* constants).  Built on muram.py (mio) and the
# variable/axis conventions of muram_to_cube.py.

import os, sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO)          # muram.py lives in the Adora repo root
import muram as mio

# --- CGS (MURaM) -> SI (Adora) ---
CM_TO_M   = 1e-2                  # length; velocity cm/s -> m/s
GCC_TO_SI = 1e3                   # mass density  g/cm^3  -> kg/m^3
NCC_TO_SI = 1e6                   # number density cm^-3  -> m^-3
G_TO_T    = 1e-4                  # magnetic field  Gauss -> Tesla
SQRT4PI   = np.sqrt(4.0 * np.pi)  # MURaM stores B in code units: B[Gauss] = B[code]*sqrt(4pi)
AMU       = 1.66053906660e-27     # kg
BARYE_TO_PA = 0.1                 # eos: pressure  dyn/cm^2 (barye) -> Pa
KB_SI       = 1.380649e-23        # eos: J/K, ideal-gas fallback for Pgas
SUM_A       = 1.095019            # eos: sum_i n_i/n_H (matches eos_mlp/eos_config; used only in the ideal-gas fallback)

WEIGHT_PER_H = 1.4                # mean atmospheric mass per H atom [amu] (H+He; metals ~negligible)
                                  # note that this is a cheat and it will be used to perform conversion approximately. We will have a small supervised
                                  # neuralnetwork that will do this.

KEYS = ("T", "ne", "nH", "vz", "vturb", "b", "gb", "cb")   # what synth_cube_mp expects

def load_muram_cube(datapath, iteration, kind="muram",
                    nx=None, ny=None, depth=None, skip=1,
                    flip_vertical=False, vsign=1.0, vturb=0.0, eos_input="pg"):   # eos: also emit Pg/rho

    """MURaM snapshot -> (cube, dz, meta), ready for synth3d_mp.synth_cube_mp.

    cube : dict of KEYS, each (Nx,Ny,Nz) fp64 in SI; Nz = vertical/depth (last axis).
    dz   : (Nz,) layer thicknesses [m] (uniform MURaM grid).
    kind : 'muram' (MuramSnap) or 'muramsub' (MuramSubSnap).
    nx,ny: (min,max) horizontal crops, subsampled by `skip`; depth: (min,max) vertical crop.
    flip_vertical: reverse depth so index 0 is the DEEP boundary -- the RT starts from
                   I=Planck(T[...,0]) and integrates up. Density rises monotonically with
                   depth, so nH_deep must be > nH_top (check meta); flip if not.
    vsign: sign for LOS velocity convention (+1 = MURaM vertical vx as-is).
    eos_input: which thermodynamic input(s) to attach for the MLP EOS synth path --
               'pg' adds cube["Pg"] (gas pressure, Pa), 'rho' adds cube["rho"] (kg/m^3),
               'both' adds both, 'none' adds neither. ne/nH are always kept (legacy path).
    """
    snap = mio.MuramSubSnap(datapath, iteration) if kind == "muramsub" \
        else mio.MuramSnap(datapath, iteration)

    need = ("Temp", "rho", "ne", "vx", "Bx", "By", "Bz")
    missing = [v for v in need if not hasattr(snap, v)]
    if missing:
        raise ValueError(f"MURaM snapshot missing {missing}; available: {snap.available}")

    # MURaM cubes are (x=vertical, y, z).  Slice as muram_to_cube.py does, then transpose so
    # the vertical (MURaM x) ends up last -> output (Ny_h, Nz_h, Nx_vert).  Depth is not skipped.
    
    dsl = slice(*depth) if depth else slice(None)
    xsl = slice(nx[0], nx[1], skip) if nx else slice(None, None, skip)
    ysl = slice(ny[0], ny[1], skip) if ny else slice(None, None, skip)
    def pick(var):
        a = np.asarray(getattr(snap, var)[dsl, xsl, ysl]).transpose(1, 2, 0).astype(np.float64)
        return a[..., ::-1] if flip_vertical else a

    T   = pick("Temp")                        # K (no conversion)
    rho = pick("rho") * GCC_TO_SI             # kg/m^3
    ne  = pick("ne")  * NCC_TO_SI             # m^-3 (MURaM EOS electron density)
    vz  = pick("vx")  * CM_TO_M * vsign       # LOS velocity = MURaM vertical vx [m/s]
    Bx, By, Bz = pick("Bx"), pick("By"), pick("Bz")   # code units; LOS component is Bx

    bmag = np.sqrt(Bx*Bx + By*By + Bz*Bz)                         # code units
    b   = bmag * SQRT4PI * G_TO_T                                 # |B| [Tesla]
    gb  = np.arccos(np.clip(Bx / np.where(bmag > 0, bmag, 1.0), -1.0, 1.0))  # inclination to LOS
    cb  = np.arctan2(Bz, By)                                      # azimuth in the sky plane
    nH  = rho / (WEIGHT_PER_H * AMU)                              # total H number density [m^-3]

    Nz = T.shape[-1]
    dz = np.full(Nz, float(snap.Temp.dX[0]) * CM_TO_M)           # uniform vertical step [m]

    cube = {"T": T, "ne": ne, "nH": nH, "vz": vz,
            "vturb": np.full_like(T, float(vturb)), "b": b, "gb": gb, "cb": cb}

    # eos: attach the thermodynamic input the MLP EOS wants -- Pgas for the "pg" head, rho for "rho".
    if eos_input in ("pg", "both"):
        if hasattr(snap, "Pres"):
            Pg = pick("Pres") * BARYE_TO_PA        # eos: MURaM's own EOS pressure (eosP) [dyn/cm^2] -> Pgas [Pa]
        else:
            Pg = (nH * SUM_A + ne) * KB_SI * T     # eos: ideal-gas fallback when no pressure field is present
        cube["Pg"] = Pg
    if eos_input in ("rho", "both"):
        cube["rho"] = rho                          # eos: mass density [kg/m^3] for the rho head

    meta = {"shape": T.shape, "dz_m": float(dz[0]), "iteration": iteration,
            "time_s": float(snap.Temp.time),
            "T_deep": float(T[..., 0].mean()), "T_top": float(T[..., -1].mean()),
            "nH_deep": float(nH[..., 0].mean()), "nH_top": float(nH[..., -1].mean())}
    if "Pg" in cube:                               # eos
        meta["Pg_deep"] = float(cube["Pg"][..., 0].mean())
        meta["Pg_top"]  = float(cube["Pg"][..., -1].mean())
    return cube, dz, meta


if __name__ == "__main__":
   
    # CLI sanity check:  python muram_loader.py <datapath> <iteration> [muram|muramsub]
    import lightweaver as lw
    datapath, iteration = sys.argv[1], int(sys.argv[2])
    kind = sys.argv[3] if len(sys.argv) > 3 else "muram"

    cube, dz, meta = load_muram_cube(datapath, iteration, kind=kind,nx=[0,8], ny=[0,8], skip=1, eos_input="pg")  # eos: also emit Pg + rho

    print("meta:", meta)
    for k in cube:                               # print every key present (incl. eos Pg/rho)
        a = cube[k]
        print(f"  {k:5s} shape={a.shape} min={a.min():.3e} max={a.max():.3e}")

    # TODO: This feels wrong - why would I check the deep vs top nH like this (and here)
    # -----------------------------------------------------------------------------------------
    #if meta["nH_deep"] < meta["nH_top"]:
    #    print("WARNING: nH_deep < nH_top -- vertical axis inverted; pass flip_vertical=True")
    # -----------------------------------------------------------------------------------------

    # simple forward-only test: single-device synth of a few columns (no multiprocessing).
    # synth_cube_fwd wants (npix, nz) arrays -> flatten the map and take a small patch.
    # import jax.numpy as jnp
    # from synth3d import read_kurucz, synth_cube_fwd
    # adata = read_kurucz(os.path.join(REPO, "kurucz_6301_6302.linelist"))
    
    waves = np.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    #npix  = min(4, cube["T"].shape[0] * cube["T"].shape[1])          # a handful of columns
    #patch = lambda a: jnp.asarray(a.reshape(-1, a.shape[-1])[:npix])
    #I = np.asarray(synth_cube_fwd(adata, jnp.asarray(waves), jnp.asarray(dz),
    #                              *[patch(cube[k]) for k in KEYS]))   # (npix, 4, nwave)
    #Ic = I[:, 0, 0]                                                   # continuum (blue edge) per column
    #print(f"forward-only synth {I.shape}: finite={np.isfinite(I).all()}, "
    #      f"Ic={Ic.mean():.3e}, min(I/Ic)={(I[:, 0] / Ic[:, None]).min():.3f}")

    # full-map synthesis (process-parallel, forward-only): each worker runs synth_cube_fwd
    #from synth3d_mp import synth_cube_mp
    # eos: Pgas-driven synth -- ne & nH now come from the MLP EOS (not MURaM's ne / the 1.4-amu cheat).
    # Saved to a NEW file so the ne/nH reference cube (synthesized_cube_djordje.npy) is not clobbered.
    #I_full = synth_cube_mp(waves, dz, cube, nworkers=64, tile=16, eos_mode="pg")   # -> (Nx, Ny, 4, nwave)
    #np.save("/dat/milic/synthesized_cube_eos_pg.npy", I_full)

    # GPU full-map synthesis attempt
    #from synth3d import synth_cube_map

    #I_map = synth_cube_map(waves, dz, cube, eos_params={"mode": "pg"}, forward_only=True)
    #print(f"GPU full-map synth {I_map.shape}")

    # Save the GPU full-map synthesis result to a file
    n#p.save("/dat/milic/adora_synth/synthesized_cube_eos_pg_map.npy", I_map)

