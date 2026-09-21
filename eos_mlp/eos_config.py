"""Shared configuration + physical constants for the MLP equation of state.

The EOS surrogate learns a single scalar, the electron pressure Pe, as a smooth
differentiable function of two independent thermodynamic variables.  Two "heads"
are trained from the same Wittmann (LTE) ground truth:

    (T, Pgas) -> Pe        # natural for inversions: Pgas is the free depth var
    (T, rho)  -> Pe        # natural for forward-synthesis of simulation cubes

Everything else the radiative-transfer stack needs follows algebraically from Pe
(see eos.py):

    ne    = Pe / (k_B T)
    nHtot = (Pgas - Pe) / (k_B T * SUM_A)

where SUM_A = sum_i n_i/n_H is the abundance sum relative to hydrogen.

Ground truth: lightweaver.wittmann.Wittmann (CGS).  All grids are therefore
stored/trained in CGS log10 space; the inference API in eos.py takes and returns
SI to match the rest of Adora (ne, nHtot in m^-3), converting at the boundary.
"""

import os

# ------------------------------------------------------------------ paths
HERE = os.path.dirname(os.path.abspath(__file__))
GRID_PG = os.path.join(HERE, "eos_grid_pg.npz")     # tabulated Pe on (T, Pgas)
GRID_RHO = os.path.join(HERE, "eos_grid_rho.npz")    # tabulated Pe on (T, rho)
PARAMS_PG = os.path.join(HERE, "eos_pg.npz")         # trained MLP for the Pgas head
PARAMS_RHO = os.path.join(HERE, "eos_rho.npz")       # trained MLP for the rho head

# ------------------------------------------------------------------ physical constants (CGS)
KB_CGS = 1.380649e-16       # erg / K
KB_SI = 1.380649e-23        # J / K

# SI -> CGS conversion factors
PA_TO_BARYE = 10.0          # 1 Pa            = 10 dyn/cm^2 (barye)
KGM3_TO_GCM3 = 1.0e-3       # 1 kg/m^3        = 1e-3 g/cm^3
M3_TO_CM3 = 1.0e-6          # 1 m^-3          = 1e-6 cm^-3  (number density)

# Abundance sum relative to H, SUM_A = sum_i (n_i / n_H).
# Filled from lightweaver.wittmann.defaultAbundances by generate_grid.py and
# stored in the grid/param files, so eos.py never needs lightweaver.  This value
# is the fallback / documented default (H + He + metals, Grevesse-like).
SUM_A_DEFAULT = 1.095019

# Mean atomic mass per H nucleus, MASS_PER_H = sum_i (n_i/n_H) * m_i  [amu].
# Used to get nHtot exactly from mass density (rho head): nHtot = rho/(MASS_PER_H*amu).
# (muram_loader.py uses the rounded value 1.4.)
MASS_PER_H_DEFAULT = 1.408984
AMU = 1.66053906660e-24     # g  (atomic mass unit, CGS)

# ------------------------------------------------------------------ grid definition
# Photosphere + chromosphere regime (chosen 2026-09-19).  Wittmann is an LTE
# stellar-atmosphere EOS; it is trustworthy where LTE holds (photosphere and
# below) and degrades in the NLTE chromosphere -- that is a property of the
# physics, not the fit.
T_MIN, T_MAX, N_T = 3.0e3, 5.0e4, 96          # K, log10-spaced
PG_MIN, PG_MAX, N_PG = 1.0e-3, 1.0e7, 96      # dyn/cm^2, log10-spaced
RHO_MIN, RHO_MAX, N_RHO = 1.0e-14, 1.0e-6, 96  # g/cm^3, log10-spaced

# fraction of grid points held out (random) for validation during training
VAL_FRAC = 0.1

# ------------------------------------------------------------------ model hyper-params
# Deliberately tiny: a smooth 2-D function in log-log space.  Random Fourier
# features up front help resolve the (fairly sharp) hydrogen-ionization ridge.
N_FOURIER = 32          # number of random Fourier features (0 disables them)
FOURIER_SCALE = 1.5     # std of the random frequency matrix
HIDDEN = (64, 64)       # hidden-layer widths
SEED = 0
