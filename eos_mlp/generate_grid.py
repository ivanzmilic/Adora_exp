"""Tabulate the electron pressure Pe on regular (T, Pgas) and (T, rho) grids
using lightweaver's Wittmann LTE equation of state.

This is the ONLY step that needs lightweaver.  Run it once; the resulting
.npz grids feed train_eos.py, and the trained MLP (eos.py) is pure JAX and
fully portable (no lightweaver at inference time).

    python generate_grid.py            # both grids, serial
    python generate_grid.py --workers 16   # parallel over T rows

Ground truth is CGS: T [K], Pgas/Pe [dyn/cm^2], rho [g/cm^3].
Wittmann is not vectorized (numba scalar njit), so we loop.

09/09/2026 - some human comments and edits by I.Milic
"""

import argparse
import os
import numpy as np

import eos_config as C


def _abund_constants():
    """
    Use lightweaver's information on abundances and masses to calculate the sum of abundances and the mass per hydrogen atom. This is all linear 
    and *does not* involve any EOS calculations.
    From lightweaver: SUM_A = sum_i n_i/n_H and MASS_PER_H = sum_i (n_i/n_H) m_i [amu].
    """

    from lightweaver.wittmann import defaultAbundances, aMass
    ab = np.asarray(defaultAbundances, dtype=float)
    am = np.asarray(aMass, dtype=float)
    sum_a = float(ab.sum() / ab[0])
    mass_per_h = float(np.sum((ab / ab[0]) * am))
    return sum_a, mass_per_h


def _eval_row(args):
    """
    Evaluate one temperature row (used by the process pool). Returns (i, pe_row).
    i is the row index
    T_i is the temperature for this row
    second_axis is the array of Pgas or rho values for this row
    mode is either "pg" or "rho" indicating which grid is being evaluated
    """

    i, T_i, second_axis, mode = args
    # Each worker builds its own Wittmann instance (numba-compiled, cheap to reuse
    # within a process; not picklable across processes).
    
    from lightweaver.wittmann import Wittmann
    w = Wittmann()
    fn = w.pe_from_pg if mode == "pg" else w.pe_from_rho
    row = np.full(second_axis.shape, np.nan)
    for j, x in enumerate(second_axis):
        try:
            pe = fn(float(T_i), float(x))
            if np.isfinite(pe) and pe > 0.0:
                row[j] = pe
        except Exception:
            pass  # unphysical corner -> leave NaN, masked out in training
    return i, row


def build_grid(mode, T_axis, second_axis, workers=0):
    """
    Here we build a grid so that we can train on it later.
    mode='pg' -> second_axis is Pgas; mode='rho' -> second_axis is rho. CGS.
    """

    tasks = [(i, T_axis[i], second_axis, mode) for i in range(len(T_axis))]
    pe = np.full((len(T_axis), len(second_axis)), np.nan)
    if workers and workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, row in pool.imap_unordered(_eval_row, tasks):
                pe[i] = row
    else:
        for t in tasks:
            i, row = _eval_row(t)
            pe[i] = row
    return pe


def main():

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=0,
                    help="process pool size for parallel row evaluation (0/1 = serial)")
    ap.add_argument("--which", choices=["pg", "rho", "both"], default="both")
    args = ap.parse_args()

    sum_a, mass_per_h = _abund_constants()
    print(f"SUM_A (n_nuclei/n_H) = {sum_a:.6f}   MASS_PER_H = {mass_per_h:.6f} amu")

    T = np.logspace(np.log10(C.T_MIN), np.log10(C.T_MAX), C.N_T)

    if args.which in ("pg", "both"):
        Pg = np.logspace(np.log10(C.PG_MIN), np.log10(C.PG_MAX), C.N_PG)
        print(f"[pg]  grid {C.N_T} x {C.N_PG} = {C.N_T * C.N_PG} points ...")
        pe = build_grid("pg", T, Pg, workers=args.workers)
        nbad = int(np.isnan(pe).sum())
        np.savez(C.GRID_PG, T=T, pgas=Pg, pe=pe, sum_a=sum_a,
                 mass_per_h=mass_per_h, mode="pg", units="cgs")
        print(f"[pg]  saved {C.GRID_PG}  ({nbad} NaN corners masked)")

    if args.which in ("rho", "both"):
        rho = np.logspace(np.log10(C.RHO_MIN), np.log10(C.RHO_MAX), C.N_RHO)
        print(f"[rho] grid {C.N_T} x {C.N_RHO} = {C.N_T * C.N_RHO} points ...")
        pe = build_grid("rho", T, rho, workers=args.workers)
        nbad = int(np.isnan(pe).sum())
        np.savez(C.GRID_RHO, T=T, rho=rho, pe=pe, sum_a=sum_a,
                 mass_per_h=mass_per_h, mode="rho", units="cgs")
        print(f"[rho] saved {C.GRID_RHO}  ({nbad} NaN corners masked)")


if __name__ == "__main__":
    main()
