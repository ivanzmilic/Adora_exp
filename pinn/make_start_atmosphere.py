"""
make_start_atmosphere.py -- Step 2 of the Adora PINN plan (pinn/plan.md).

Build a starting atmosphere for the PINN warm start: take FAL-C from lightweaver, resample it to a
regular z grid z in [-100, 1000] km with 20 km steps, and clone that single column into a 128 x 128
map. Saved (SI units) as an .npz holding the 7 PINN channels (T, P, vz, Bx, By, Bz, vturb) plus the
z grid, reference ne/nHtot, and metadata.

NOTE on z: this lightweaver FAL-C has z in [0, 2342] km with z=0 at the deep sub-photosphere and tau=1
near z~100 km (NOT the usual z=0 at tau=1). We relabel z' = z_lw - Z_OFFSET_KM so the model bottom sits at
z' = -100 km, and sample FALC over lightweaver z in [0, 1100] km -- entirely WITHIN the model, so there is
NO extrapolation. tau=1 then falls near z' ~ 0 (a standard 'height above photosphere' grid). The saved
z_km is this relabeled z'.
"""
import os
import numpy as np
from scipy.interpolate import interp1d
from lightweaver.fal import Falc82

# --- config ---
Z_MIN_KM, Z_MAX_KM, DZ_KM = -100.0, 1000.0, 20.0
Z_OFFSET_KM = 100.0          # z' = z_lw - Z_OFFSET_KM: lightweaver bottom (z_lw=0) -> z' = -100 km
NX, NY = 128, 128
V_AMP = 3.0e3                # peak |vz| [m/s] of the added velocity structure
SUM_A = 1.095019            # sum_i n_i/n_H (matches eos_mlp/eos_config); for ideal-gas Pgas
KB_SI = 1.380649e-23        # J/K
OUT_NAME = "test_atm_falc_128.npz"


def build():
    fal = Falc82()
    zf = np.asarray(fal.z)                    # m, ordered top-first
    order = np.argsort(zf)                    # -> ascending in z
    zf = zf[order]
    T  = np.asarray(fal.temperature)[order]
    ne = np.asarray(fal.ne)[order]
    nh = np.asarray(fal.nHTot)[order]
    vt = np.asarray(fal.vturb)[order]
    Pg = (nh * SUM_A + ne) * KB_SI * T        # ideal-gas gas pressure [Pa] at FAL-C points

    # relabeled target grid z' in [-100,1000]; sample FALC at lightweaver z = z' + Z_OFFSET_KM, which
    # lies within [0,2342] km -> pure interpolation, no extrapolation.
    z_km = np.arange(Z_MIN_KM, Z_MAX_KM + 0.5 * DZ_KM, DZ_KM)   # -100 .. 1000, 20 km (relabeled z')
    z_lw_m = (z_km + Z_OFFSET_KM) * 1.0e3                        # actual lightweaver z sampled [m]

    lin = lambda y: interp1d(zf, y, kind="linear", fill_value="extrapolate")(z_lw_m)
    logi = lambda y: np.exp(interp1d(zf, np.log(y), kind="linear", fill_value="extrapolate")(z_lw_m))

    T_t  = lin(T)
    vt_t = np.clip(lin(vt), 0.0, None)
    P_t  = logi(Pg)                           # log-linear -> exponential extrapolation below z=0
    ne_t = logi(ne)                           # reference only (EOS derives ne from T,P downstream)
    nh_t = logi(nh)

    nz = z_km.size
    col = {"T": T_t, "P": P_t, "vz": np.zeros(nz),
           "Bx": np.zeros(nz), "By": np.zeros(nz), "Bz": np.zeros(nz), "vturb": vt_t}
    # clone the FAL-C column across the (nx, ny) map -> (nx, ny, nz)
    cube = {k: np.broadcast_to(v, (NX, NY, nz)).copy() for k, v in col.items()}

    # add smooth velocity structure so vz is not trivially zero: a horizontal 'granulation-like'
    # pattern (a few cells across the map) x a vertical envelope peaking near the photosphere (z'=0),
    # amplitude +-V_AMP. Gives the field genuine (x,y,z) structure to fit in step 3.
    XX, YY = np.meshgrid(np.linspace(0.0, 1.0, NX), np.linspace(0.0, 1.0, NY), indexing="ij")
    f_xy = np.cos(2 * np.pi * 3.0 * XX) * np.cos(2 * np.pi * 2.0 * YY)      # (NX,NY) in [-1,1]
    g_z = np.exp(-((z_km - 0.0) / 400.0) ** 2)                             # (nz,) envelope ~photosphere
    cube["vz"] = (V_AMP * f_xy[:, :, None] * g_z[None, None, :]).astype(float)

    # target dir: honour /data/milic if it exists, else fall back to the established /dat/milic
    for base in ("/data/milic", "/dat/milic"):
        out_dir = os.path.join(base, "adora_pinn_develop")
        try:
            os.makedirs(out_dir, exist_ok=True)
            break
        except OSError:
            continue
    out_path = os.path.join(out_dir, OUT_NAME)

    np.savez(
        out_path,
        z_km=z_km, dz_km=DZ_KM, z_offset_km=Z_OFFSET_KM, nx=NX, ny=NY, nz=nz,
        sum_a=SUM_A, units="SI (T[K], P[Pa], vz/vturb[m/s], B[T], z_km[km])",
        source="lightweaver Falc82, cloned",
        v_amp=V_AMP,
        note="z_km is relabeled z'=z_lw-100 (bottom->-100km, tau=1 near z'~0), no extrapolation; "
             "vz = V_AMP*cos(2pi 3x)cos(2pi 2y)*exp(-(z'/400km)^2), other channels cloned, B=0",
        ref_ne=ne_t, ref_nHtot=nh_t,
        **{k: cube[k] for k in ("T", "P", "vz", "Bx", "By", "Bz", "vturb")},
    )
    return out_path, z_km, col, ne_t, nh_t, cube


if __name__ == "__main__":
    out_path, z_km, col, ne_t, nh_t, cube = build()
    print(f"saved {out_path}")
    print(f"z: {z_km.size} points, {z_km[0]:.0f} .. {z_km[-1]:.0f} km, step {z_km[1]-z_km[0]:.0f} km")
    print(f"  T     {col['T'].min():.0f} .. {col['T'].max():.0f} K")
    print(f"  P     {col['P'].min():.3e} .. {col['P'].max():.3e} Pa")
    print(f"  vz    {cube['vz'].min():+.3e} .. {cube['vz'].max():+.3e} m/s  (structured)")
    print(f"  vturb {col['vturb'].min():.3e} .. {col['vturb'].max():.3e} m/s")
    print(f"  ref ne {ne_t.min():.3e} .. {ne_t.max():.3e} m^-3   nHtot {nh_t.min():.3e} .. {nh_t.max():.3e} m^-3")
    print(f"  cube shape (per channel): {(128, 128, z_km.size)}")
