"""Portable MLP equation-of-state: inference API for the Adora RT stack.

Pure JAX -- jittable, vmappable, differentiable, and free of any lightweaver
dependency (only eos_model.py, eos_config.py and the trained .npz travel with
it).  Takes and returns SI, matching Adora's convention (ne, nHtot in m^-3;
T in K; Pgas in Pa; rho in kg/m^3).

Typical use, replacing the two free parameters (ne, nHtot) with one physical one:

    from eos_mlp import eos
    eos_pg = eos.load_eos("pg")                     # once
    ne, nHtot = eos.ne_nhtot_from_pg(eos_pg, T, Pgas)   # arrays over depth
    stokes = lte_polarised_rt(adata, wave, dz, T, ne, nHtot, vz, vturb, b, gb, cb)

For MURaM cubes where rho is native:

    eos_rho = eos.load_eos("rho")
    ne, nHtot = eos.ne_nhtot_from_rho(eos_rho, T, rho)

All functions are safe under jax.jit / jax.vmap / jax.grad w.r.t. T, Pgas, rho.
The model was trained in CGS log10 space; conversions happen at the boundary.
"""

import numpy as np
import jax.numpy as jnp

import eos_config as C
import eos_model as M

AMU_SI = 1.66053906660e-27  # kg


def load_eos(mode="pg", path=None):
    """Load a trained head. mode in {'pg','rho'}. Returns a dict of jnp arrays."""
    if path is None:
        path = C.PARAMS_PG if mode == "pg" else C.PARAMS_RHO
    d = np.load(path, allow_pickle=True)
    return {
        "params": M.unflatten(d),
        "xmin": jnp.asarray(d["xmin"]),
        "xmax": jnp.asarray(d["xmax"]),
        "ymean": float(d["ymean"]),
        "ystd": float(d["ystd"]),
        "sum_a": float(d["sum_a"]),
        "mass_per_h": float(d["mass_per_h"]),
        "mode": str(d["mode"]),
    }


def _log_pe_cgs(eosp, T, P_cgs):
    """Model prediction: log10(Pe [dyn/cm^2]) from T [K] and P_cgs (Pgas or rho, CGS)."""
    x_raw = jnp.stack([jnp.log10(T), jnp.log10(P_cgs)], axis=-1)
    xn = 2.0 * (x_raw - eosp["xmin"]) / (eosp["xmax"] - eosp["xmin"]) - 1.0
    return M.forward(eosp["params"], xn) * eosp["ystd"] + eosp["ymean"]


# --------------------------------------------------------------- (T, Pgas) head
def pe_from_pg(eosp, T, Pgas):
    """Electron pressure Pe [Pa] from T [K], Pgas [Pa]."""
    log_pe_cgs = _log_pe_cgs(eosp, T, Pgas * C.PA_TO_BARYE)
    return (10.0 ** log_pe_cgs) / C.PA_TO_BARYE


def ne_nhtot_from_pg(eosp, T, Pgas):
    """(ne, nHtot) [m^-3] from T [K], Pgas [Pa].

    ne    = Pe / (k_B T)
    nHtot = (Pgas - Pe) / (k_B T * SUM_A)     # remaining pressure is nuclei
    """
    Pe = pe_from_pg(eosp, T, Pgas)
    kT = C.KB_SI * T
    ne = Pe / kT
    nHtot = (Pgas - Pe) / (kT * eosp["sum_a"])
    return ne, nHtot


# --------------------------------------------------------------- (T, rho) head
def pe_from_rho(eosp, T, rho):
    """Electron pressure Pe [Pa] from T [K], rho [kg/m^3]."""
    log_pe_cgs = _log_pe_cgs(eosp, T, rho * C.KGM3_TO_GCM3)
    return (10.0 ** log_pe_cgs) / C.PA_TO_BARYE


def ne_nhtot_from_rho(eosp, T, rho):
    """(ne, nHtot) [m^-3] from T [K], rho [kg/m^3].

    ne    = Pe / (k_B T)
    nHtot = rho / (MASS_PER_H * amu)          # exact mass conservation, no Pe needed
    """
    Pe = pe_from_rho(eosp, T, rho)
    ne = Pe / (C.KB_SI * T)
    nHtot = rho / (eosp["mass_per_h"] * AMU_SI)
    return ne, nHtot


# --------------------------------------------------------------- domain check
def in_domain(eosp, T, P):
    """True where (T, P) lies inside the trained log10 box (P = Pgas[Pa] or rho[kg/m^3]).

    Outside the box the MLP extrapolates and should not be trusted.
    """
    P_cgs = P * (C.PA_TO_BARYE if eosp["mode"] == "pg" else C.KGM3_TO_GCM3)
    lx = jnp.stack([jnp.log10(T), jnp.log10(P_cgs)], axis=-1)
    return jnp.all((lx >= eosp["xmin"]) & (lx <= eosp["xmax"]), axis=-1)


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)
    eos_pg = load_eos("pg")
    T = jnp.array(6000.0)
    Pg = jnp.array(1.0e4)  # Pa  (== 1e5 dyn/cm^2)
    ne, nH = ne_nhtot_from_pg(eos_pg, T, Pg)
    print(f"T={float(T):.0f} K  Pgas={float(Pg):.2e} Pa")
    print(f"  ne    = {float(ne):.4e} m^-3  ({float(ne)*1e-6:.4e} cm^-3)")
    print(f"  nHtot = {float(nH):.4e} m^-3")
    # differentiability check (grad of ne w.r.t. T)
    gT = jax.grad(lambda t: ne_nhtot_from_pg(eos_pg, t, Pg)[0])(T)
    gp = jax.grad(lambda Pg: ne_nhtot_from_pg(eos_pg, T, Pg)[0])(Pg)
    print(f"  d ne / d T = {float(gT):.4e} m^-3/K  (finite -> differentiable)")
    print(f"  d ne / d Pgas = {float(gp):.4e} m^-3/Pa  (finite -> differentiable)")
