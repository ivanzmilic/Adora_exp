"""Validate the trained MLP EOS against the Wittmann ground truth and FAL-C.

Uses lightweaver (dev-time only), so it is NOT part of the portable inference
path.  Produces eos_validation.png with:
  - relative error of the (T, Pgas) head vs Wittmann across the grid
  - relative error of the (T, rho) head vs Wittmann across the grid
  - ne(depth) for FAL-C: MLP vs the model's own ne (where LTE holds)

    python validate_eos.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eos_config as C
import eos as E

KB_CGS = C.KB_CGS


def _wittmann_pe(mode, T, second):
    from lightweaver.wittmann import Wittmann
    w = Wittmann()
    fn = w.pe_from_pg if mode == "pg" else w.pe_from_rho
    out = np.full((len(T), len(second)), np.nan)
    for i, t in enumerate(T):
        for j, x in enumerate(second):
            try:
                pe = fn(float(t), float(x))
                if np.isfinite(pe) and pe > 0:
                    out[i, j] = pe
            except Exception:
                pass
    return out


def _err_map(ax, mode, eosp, T, second, xlabel):
    pe_true = _wittmann_pe(mode, T, second)               # CGS
    TT, PP = np.meshgrid(T, second, indexing="ij")
    if mode == "pg":                                       # SI inputs to eos.py
        pe_pred = np.asarray(E.pe_from_pg(eosp, jnp.asarray(TT), jnp.asarray(PP / C.PA_TO_BARYE)))
    else:
        pe_pred = np.asarray(E.pe_from_rho(eosp, jnp.asarray(TT), jnp.asarray(PP / C.KGM3_TO_GCM3)))
    pe_pred_cgs = pe_pred * C.PA_TO_BARYE
    rel = np.abs(pe_pred_cgs - pe_true) / pe_true
    med = np.nanmedian(rel)
    mx = np.nanmax(rel)
    im = ax.pcolormesh(T, second, np.log10(rel.T + 1e-12), shading="auto",
                       cmap="magma", vmin=-5, vmax=-1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("T [K]"); ax.set_ylabel(xlabel)
    ax.set_title(f"[{mode}] log10 rel.err in Pe\nmedian={med:.1e} max={mx:.1e}")
    plt.colorbar(im, ax=ax)
    return med, mx


def main():
    eos_pg = E.load_eos("pg")
    eos_rho = E.load_eos("rho")

    T = np.logspace(np.log10(C.T_MIN), np.log10(C.T_MAX), 80)
    Pg = np.logspace(np.log10(C.PG_MIN), np.log10(C.PG_MAX), 80)
    rho = np.logspace(np.log10(C.RHO_MIN), np.log10(C.RHO_MAX), 80)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    _err_map(axs[0], "pg", eos_pg, T, Pg, "Pgas [dyn/cm^2]")
    _err_map(axs[1], "rho", eos_rho, T, rho, "rho [g/cm^3]")

    # FAL-C ne, restricted to where LTE is a fair test (below the temperature min)
    from lightweaver.fal import Falc82
    fal = Falc82()
    Tf = fal.temperature; nef = fal.ne; nHf = fal.nHTot          # SI
    Pg_f = (nHf * eos_pg["sum_a"] + nef) * C.KB_SI * Tf           # SI gas pressure
    ne_pred, nH_pred = E.ne_nhtot_from_pg(eos_pg, jnp.asarray(Tf), jnp.asarray(Pg_f))
    ne_pred = np.asarray(ne_pred)
    ax = axs[2]
    ax.plot(nef, label="FAL-C ne (NLTE model)", lw=2)
    ax.plot(ne_pred, "--", label="MLP EOS ne (LTE)", lw=2)
    ax.set_yscale("log"); ax.set_xlabel("depth index"); ax.set_ylabel("ne [m^-3]")
    ax.set_title("FAL-C: ne  (LTE vs NLTE differ in chromosphere)")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(C.HERE, "eos_validation.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
