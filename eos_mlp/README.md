# `eos_mlp` — a tiny differentiable MLP equation of state

A small neural surrogate for the **LTE equation of state**, trained once on a
dense grid and then used as a fast, differentiable function inside the Adora
radiative-transfer stack.

## Why

The RT stack (`lineop.py`, `contop.py`) needs the electron density `ne` and the
total hydrogen density `nHtot` at every depth. Until now both were passed as
**free parameters**, which is physically redundant and degenerate (see the
project notes / regularisers). A proper EOS ties them to the real thermodynamic
state, so the inversion solves for one physical variable per depth instead of
two loosely-constrained ones.

## The idea

The EOS learns **one scalar**, the electron pressure `Pe`, as a smooth function
of two thermodynamic variables. Two "heads" are trained from the same ground
truth:

| head | input | natural use |
|------|-------|-------------|
| `pg`  | `(T, Pgas)` | **inversion** — `Pgas` is the free depth variable |
| `rho` | `(T, rho)`  | **forward synthesis** of simulation cubes (MURaM gives `rho`) |

Everything the RT needs then follows **algebraically** from `Pe`:

```
ne    = Pe / (k_B T)
nHtot = (Pgas - Pe) / (k_B T * SUM_A)     # pg head: remaining pressure is nuclei
nHtot = rho / (MASS_PER_H * amu)          # rho head: exact mass conservation
```

with `SUM_A = Σ nᵢ/n_H ≈ 1.095` and `MASS_PER_H ≈ 1.409 amu`.

## Ground truth

`lightweaver.wittmann.Wittmann` — a full multi-element LTE Saha–Boltzmann EOS
(Kurucz partition functions, metal electron donors, molecules). Validated
against FAL-C: reproduces `ne` to **< 0.1 %** in the photosphere where LTE holds.
It legitimately diverges in the upper chromosphere, where FAL-C `ne` is *NLTE*
hydrogen-ionization-enhanced — an LTE EOS can't and shouldn't match that.

**lightweaver is needed only to build the grid.** The trained model
(`eos.py` + `eos_model.py` + `eos_config.py` + the two `.npz`) is pure JAX and
portable — e.g. straight into `PINNdora` — with no lightweaver at inference time.

## Files

| file | role | needs lightweaver |
|------|------|:--:|
| `eos_config.py`   | grid ranges, unit constants, paths | – |
| `generate_grid.py`| tabulate `Pe` on `(T,Pgas)` and `(T,rho)` grids | ✅ |
| `eos_model.py`    | tiny pure-JAX MLP (Fourier features + tanh) | – |
| `train_eos.py`    | train once with optax, save params | – |
| **`eos.py`**      | **portable inference API** (SI in/out) | – |
| `validate_eos.py` | error maps vs Wittmann + FAL-C check → `eos_validation.png` | ✅ |

Artefacts written by the pipeline: `eos_grid_{pg,rho}.npz` (grids),
`eos_{pg,rho}.npz` (trained models).

## Workflow

```bash
conda activate adora-develop
cd eos_mlp
python generate_grid.py --workers 16   # one-time, ~25 s (uses lightweaver)
python train_eos.py --mode both        # one-time, ~minutes on CPU
python validate_eos.py                 # optional: sanity plots
python eos.py                          # quick demo + differentiability check
```

## Use in the RT

```python
from eos_mlp import eos          # or `import eos` if eos_mlp is on sys.path

eos_pg = eos.load_eos("pg")                          # load once
ne, nHtot = eos.ne_nhtot_from_pg(eos_pg, T, Pgas)    # arrays over depth (SI)
stokes = lte_polarised_rt(adata, wave, dz, T, ne, nHtot, vz, vturb, b, gb, cb)
```

All inference functions are safe under `jax.jit`, `jax.vmap`, and `jax.grad`
w.r.t. `T`, `Pgas`, `rho`. `eos.in_domain(eosp, T, P)` flags points outside the
trained box (where the MLP extrapolates and should not be trusted).

## Conventions / caveats

- **Units:** SI at the API boundary (`T` K, `Pgas` Pa, `rho` kg/m³, `ne`/`nHtot`
  m⁻³). Grids and the model interior are CGS log10.
- **Regime:** trained for photosphere + chromosphere (`T ∈ [3e3, 5e4] K`). Edit
  the ranges in `eos_config.py` and re-run to extend.
- **LTE only** — this is an LTE EOS; do not expect it to reproduce NLTE `ne`.
- Abundances follow lightweaver's `defaultAbundances`; change the abundance mix
  there and regenerate if you need a different composition.
