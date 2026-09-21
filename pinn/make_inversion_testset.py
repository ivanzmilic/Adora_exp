"""
make_inversion_testset.py -- Step 5 of the Adora PINN plan (pinn/plan.md).

Build the mock inversion testset (the "observations" to invert): crop a prepared synthetic Stokes cube to
the first 384x384 pixels, take every 3rd -> 128x128 pixels at 48 km horizontal spacing (original 16 km).
The cube was synthesized with the same Adora RT as the forward model, so its Stokes units are directly
comparable to what the PINN produces. Wavelengths are assumed to be the standard synth grid (Fe I
6301/6302, 201 points, vacuum) -- the .npy carries no axis metadata.
"""
import os
import numpy as np

SRC = "/dat/milic/adora_synth/test.npz.npy"     # (1536,1536,4,201) Adora Stokes cube
N_FULL, STEP = 384, 3                            # first 384 px, every 3rd -> 128 px
DX_KM = DY_KM = 48.0                             # horizontal spacing after subsampling (16 km * 3)


def build():
    a = np.load(SRC, mmap_mode="r")             # memmap: avoid loading the full ~15 GB
    obs = np.ascontiguousarray(a[:N_FULL:STEP, :N_FULL:STEP])   # (128,128,4,201)
    import lightweaver as lw
    waves = np.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), obs.shape[-1])

    for base in ("/data/milic", "/dat/milic"):
        out_dir = os.path.join(base, "adora_pinn_develop")
        try:
            os.makedirs(out_dir, exist_ok=True); break
        except OSError:
            continue
    out = os.path.join(out_dir, "inversion_testset.npz")
    np.savez(out, I_obs=obs, waves=waves, dx_km=DX_KM, dy_km=DY_KM,
             source=SRC, crop=f"[0:{N_FULL}:{STEP}, 0:{N_FULL}:{STEP}] -> {obs.shape[0]}x{obs.shape[1]}",
             units="Adora RT Stokes (same scale as the forward model)")
    return out, obs, waves


if __name__ == "__main__":
    out, obs, waves = build()
    Ic = obs[..., 0, 0]
    print(f"saved {out}")
    print(f"  I_obs shape {obs.shape}  (128x128, 4 Stokes, {waves.size} waves)")
    print(f"  waves {waves[0]:.3f}..{waves[-1]:.3f} nm (vac)   spacing dx=dy={DX_KM} km")
    print(f"  continuum I: {Ic.min():.3e} .. {Ic.max():.3e}")
    print(f"  Stokes ranges: "
          + "  ".join(f"{s}=[{obs[:,:,k,:].min():+.2e},{obs[:,:,k,:].max():+.2e}]"
                      for k, s in enumerate("IQUV")))
