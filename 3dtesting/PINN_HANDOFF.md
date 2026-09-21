# Adora — Internal Handoff: PINN Spectropolarimetric Inversion Extension

*Written 2026-09-18. Audience: future-me. Purpose: everything needed to lift Adora's
differentiable forward model into another repo and build a PINN / neural-field inversion
on top of it. This is a **code-reading** handoff (no runs were done this session — user
asked for reading only). Companion one-pager: `adora_7_pinn_inversion.pdf`.*

Related: [[adora-project-state]] (the 3D-synthesis branch of work), [[adora-develop-env]] (env).
Prior handoff `HANDOFF.md` covers the **3D synthesis** direction; **this** doc covers the
**inversion / PINN** direction. They share the same forward model.

---

## 0. TL;DR — the one thing that matters

Adora's whole forward model — EOS-ish populations → opacities → polarised formal solution — is
**one differentiable JAX function in fp64**. Therefore **response functions and Jacobians are free**
(`jax.jacrev`/`jacfwd`/`value_and_grad`). A "PINN inversion" is just: replace the atmosphere
parameterisation (currently the fixed **nodes** of `nodes.py`) with a **coordinate MLP + Fourier
features** (`f_θ(x,y,z) → 8 physical params`), push it through the *existing* forward model, and
train `θ` with `optax` to fit observed Stokes profiles + physics regularisers. Gradients already flow
`Stokes → RT → atmosphere → θ`, so nothing in the physics needs to change.

**The single call a PINN needs** (one wavelength, one column):

```python
from vector_response_fn import lte_polarised_rt   # vector_response_fn.py:8
I = lte_polarised_rt(adata, wave, dz,
                     temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b)
# -> jnp.array([I, Q, U, V]) at this wavelength; vmap over `wave` to get a profile.
```

`adata` is an `AtomicData` pytree from `read_kurucz(...)` (built once). The **8 per-depth
parameters** the network must output are exactly:

| # | name        | unit  | notes / output transform for the NN |
|---|-------------|-------|-------------------------------------|
| 1 | temperature | K     | scaled sigmoid (bounds ~2.5e3–20e3) |
| 2 | ne          | m⁻³   | exp/softplus, work in log10 (bounds ~1e16–1e22) |
| 3 | nhtot       | m⁻³   | exp/softplus, log10 (bounds ~1e16–1e24) |
| 4 | vz          | m/s   | tanh·v_max (bounds ~±20e3) |
| 5 | vturb       | m/s   | softplus (0–~10e3) |
| 6 | b           | T     | softplus/bounded (0–~1) |
| 7 | gamma_b     | rad   | inclination to LOS, 0–π |
| 8 | chi_b       | rad   | azimuth, 0–2π |

`dz` = per-layer depth spacing [m] on the sampling grid. Bounds above are lifted verbatim from
`nodes.py:269-288` — reuse them for the NN output maps.

---

## 1. Codebase map (file → role → key symbols)

Repo root `/home/milic/codes/Adora`. Every module starts with `jax.config.update("jax_enable_x64", True)`.

### Physics core (the reusable forward model — copy these)
- **`lineop.py`** (738 lines) — *the heart*.
  - `read_kurucz(path) -> AtomicData` (`:77`): parses a Kurucz line list; precomputes the LS-coupling
    Zeeman pattern (`zeeman_alphas/strengths/shifts`) into the pytree. **Uses `lightweaver`** for
    `PeriodicTable` mass, `DefaultAtomicAbundance`, and `lightweaver.zeeman.{lande_factor,
    fraction_range, zeeman_strength}` (`:12,:226-238,:277,:280`). See gotcha §5.
  - `AtomicData` (`:33`) — `@jdc.pytree_dataclass`; plain JAX arrays after construction.
  - `emis_opac_polarised(adata, wave, T, ne, nhtot, vel, vturb, b, gamma_b, chi_b)` (`:541`) →
    `(eta[4], chi[7])` = `([η_I,η_Q,η_U,η_V], [η_I,η_Q,η_U,η_V,ρ_Q,ρ_U,ρ_V])` for one point.
    **Valid only for μ_z = 1** (comment `:586`).
  - `emis_opac(...)` (`:491`) — scalar (Stokes I) twin.
  - `fe_pops` / `fei_pop_i` (`:667`,`:694`) — Fe I/II/III via Irwin (1981) partition functions
    (`Q_FeI/II/III` `:625-665`); Boltzmann level pops. **Only Fe is implemented.**
  - `planck` (`:340`) — Planck fn in kW/(m² nm sr); the LTE source function.
  - Doppler width, damping (`γ_rad+γ_Stark·ne+γ_vdW·nHI`), thermal velocity: `:295-338`.
- **`contop.py`** (227 lines) — continuum opacity `continuum_opacity(wvl,T,ne,nhtot)` (`:145`):
  H bound-free 5 levels (Seaton/RH Gaunt `:19,:46`), H⁻ bf & ff (Gray 2021 fits `:99,:67`),
  Saha H ionisation `lte_h_ion_fracs` (`:128`). Imports astropy only (no lightweaver).
- **`voigt.py`** (92 lines) — `voigt_H(a,v)` (`:37`) Humlicek w4 rational approx → (H, F=Voigt,
  Faraday); `voigt_H_re` real part only. Differentiable as-is (the `custom_jvp` block is commented
  out `:36-49`; autodiff goes straight through the complex rational form and works).
- **`vector_formal_solver.py`** (83 lines) — `delo_constant_fs(dz, I_start, emis, opac)` (`:18`):
  polarised **DELO** (modified/constant, Janett et al. 2017 appendix), 4×4 propagation matrix
  `stokes_K` (`:6`), `fori_loop` from index 1. `I_start` is the **deep** boundary.
- **`vector_response_fn.py`** (modified, uncommitted) — `lte_polarised_rt(...)` (`:8`): vmap
  `emis_opac_polarised` over depth → `delo_constant_fs`. `I_start = [planck(wave,T[0]),0,0,0]`
  (`:14`) → **depth index 0 = deepest layer** (see §5). `__main__` validates vs Lightweaver and
  saves `lte_polarised.png`. The full polarised Jacobian block is **commented out** in the working
  tree (`:93-103`) — irrelevant for a PINN, which uses `value_and_grad` of a scalar loss.

### Scalar (Stokes I only) path — lighter alternative
- **`response_fn.py`** (modified) — `lte_rt(...)` (`:7`): scalar analogue via `nearest_fs`.
- **`scalar_formal_solver.py`** — `cumsum_fs` (cumulative-τ) and `nearest_fs` (piecewise-constant
  short characteristics, `fori_loop`).

### Inversion machinery (the baseline the PINN replaces — reference, adapt)
- **`nodes.py`** (306 lines) — **the current inversion basis** (SIR/SNAPI-style).
  - `NodeSpec` (`:11`, `@jdc.pytree_dataclass`) holds `n_*` node counts per parameter + a target
    interpolation grid size `n_interp`.
  - `reconstruct_from_nodes(nodes)` (`:112`): flat node vector → `interpax.interp1d` to a uniform
    z-grid (log10 for ne & nhtot), returns `(dz, T, ne, nhtot, vz, vturb, b, gamma_b, chi_b)` —
    i.e. exactly the `lte_polarised_rt` atmosphere tuple. **This is the function a neural field
    replaces.**
  - `compute_residual_from_nodes` (`:168`) → `compute_residual_vmap` (jit+vmap over pixels, `:174`)
    and `compute_residual_jac_vmap` (jit+vmap+`jacrev`, `:182`).
  - `__main__`: self-inversion of FALC via `scipy.optimize.least_squares` TRF with bounds
    (`:290-300`); bounds at `:269-288`. **Also imports lightweaver.**
- **`iterate_adam.py`** — **optax** loop template (optimistic Adam + reduce-on-plateau,
  `:161-188`); `value_and_grad(loss)` (`:152`). **Closest existing template for the PINN trainer.**
- **`iterate_scipy.py`** — SciPy `least_squares` TRF driven by the JIT JAX Jacobian (scalar path).
- **`iterate.py`** — Levenberg–Marquardt demo (slow gradients).

### Data / misc
- `kurucz_6301_6302.linelist` — Fe I 630.15/630.25 nm pair (the only validated line data).
- `lte_pops.py` — generic Saha-Boltzmann demo, **not used by anything**.
- `muram.py`, `muram_to_cube.py`, `3dtesting/muram_loader.py`, `3dtesting/synth3d*.py` — the **3D
  synthesis** branch (see `HANDOFF.md`), not needed for a first PINN inversion.

### Call graph (forward)
```
lte_polarised_rt (vector_response_fn.py)
  ├─ vmap over depth: emis_opac_polarised (lineop.py)
  │     ├─ continuum_opacity            (contop.py) ── lte_h_ion_fracs
  │     ├─ fei_pop_i / fe_pops          (lineop.py: Irwin Q_FeI/II/III)
  │     ├─ voigt_H                       (voigt.py)
  │     └─ Zeeman pattern (precomputed in AtomicData by read_kurucz)
  └─ delo_constant_fs (vector_formal_solver.py)  ← I_start = Planck(T[deep])
```

---

## 2. What to copy to the other repo

**Minimum viable forward model (full Stokes):**
`lineop.py`, `contop.py`, `voigt.py`, `vector_formal_solver.py`, `vector_response_fn.py`,
`kurucz_6301_6302.linelist`.

**If Stokes I only suffices at first:** `response_fn.py` + `scalar_formal_solver.py` instead of the
vector pair (still need `lineop.py`, `contop.py`, `voigt.py`).

**Reference for the inverter (adapt, don't blindly copy):** `nodes.py` (parameterisation +
residual/Jacobian pattern), `iterate_adam.py` (optax loop), `iterate_scipy.py` (TRF + JAX Jacobian).

**To be written new in the target repo:**
- `neural_field.py` — Fourier-feature encoding + MLP + per-parameter physical output transforms.
- `invert_field.py` — optax training loop through `lte_polarised_rt` + regularisers.

**Python deps to carry** (see [[adora-develop-env]] for pins): `jax` (x64), `jax_dataclasses`,
`astropy` (constants in lineop/contop), `numpy`, `optax`; `interpax` (only for `nodes.py`);
`scipy` (only for the scipy/LM inverters); **`lightweaver`** (for `read_kurucz` and `nodes.py`).

---

## 3. The PINN / neural-field plan (concrete)

Fleshes out `adora_3_plan_neuralfield.pdf`. Atmosphere = coordinate-based neural field
`f_θ(γ(x)) : (x,y,z) → 8 params` (§0 table).

1. **Encoding** `γ(x) = [sin(2π B x), cos(2π B x)]`, Gaussian random freqs `B` (Tancik et al. 2020),
   **per-axis bandwidth** — `z` needs finer bandwidth than `x,y` to represent sharp photospheric
   `T`/velocity gradients (defeats MLP spectral bias).
2. **Network** small MLP (3–5 layers, width 64–256, SiLU/tanh) → 8 outputs, each through a physical
   map (§0 table) using the `nodes.py` bounds. Densities & vturb in log/softplus space.
3. **Loss** = χ² data misfit + regularisers. For each pixel: sample the field on that column's
   **z-grid** (preserve index 0 = deep, and the `dz` spacing) → `vmap(lte_polarised_rt)` over λ →
   Stokes → compare to observed. Regularisers: smoothness in depth, a hydrostatic/EOS residual
   (ties `ne`,`nHI` to `T`,`P` — see §5 "no EOS"), optionally `∇·B = 0`. This *is* the "PINN
   regularisation" from the README TODO.
4. **Train** Adam (optax, like `iterate_adam.py`) then optionally L-BFGS; `jit + vmap` over pixels/λ,
   mini-batch pixels.
5. **Validate** single pixel → 2D map (shared `θ` gives spatial coherence — the payoff over
   per-pixel node inversions) → invert a synthetic FALC/MURaM cube; compare recovered atmosphere,
   fit quality and speed vs the `nodes.py` baseline and ground truth.

**Precision** the opacity kernel is fp64-locked (`jnp.eye(4)` in DELO, Planck/const in `lineop`).
An fp32 NN can drive it, but f32 atmosphere inputs promote right back to f64 inside the kernel — a
true mixed-precision win needs a dtype-generic kernel (deferred; see [[adora-project-state]]).

---

## 4. Conventions & gotchas that MUST travel with the code

1. **fp64 mandatory** — every module sets `jax_enable_x64` at import; without it, opacities and the
   DELO solve are wrong. Set it before the first `jax.numpy` use in any new entry point.
2. **Units (SI + spectroscopy conv.)** — `I` in kW·m⁻²·nm⁻¹·sr⁻¹; emissivity kW·m⁻³·nm⁻¹; opacity
   m⁻¹; wavelength **nm**; velocities m/s; `b` in **Tesla**; angles rad. Everything else SI.
3. **Depth orientation** — arrays are ordered **index 0 = deepest layer, last index = surface**
   (demos use `fal.<x>[::-1]`). `I_start = Planck(wave, T[0])` is the lower/deep LTE boundary; the
   DELO loop integrates upward from index 1 (`vector_formal_solver.py:55`). The NN sampler must
   honour this ordering.
4. **μ = 1 only** — polarised opacity assumes vertical rays (`lineop.py:586`); no LOS projection of
   `vz`/`B` yet. Inclined rays are a TODO.
5. **No self-consistent EOS** — `ne` and `nhtot` are **free inputs**, not solved from `(T, P)`. With
   8 free params/depth the inversion is degenerate; this is exactly what the PINN's physics
   regularisers (and a future EOS) are meant to constrain. Don't assume `ne`↔`T` consistency.
6. **lightweaver is a build-time dependency** — `read_kurucz` (and `nodes.py`) import it. Options in
   the target repo: (a) install lightweaver (see [[adora-develop-env]]: **not pip-installable here**,
   wired from a prebuilt cp312 `.so`), or (b) call `read_kurucz` once and **serialise the
   `AtomicData` pytree** (it's plain arrays) so the runtime path no longer needs lightweaver.
   Recommend (b) for portability.
7. **Only Fe partition functions** — Irwin (1981) Fe I/II/III. Other elements/species need new
   `Q(T)` fits before their lines will be correct.
8. **Line data** — validated only for the Fe I 630 nm pair (`kurucz_6301_6302.linelist`), against
   Lightweaver (`lte_polarised.png`). vdW broadening is crude (Unsöld is a TODO).
9. **Working-tree state** — `response_fn.py` & `vector_response_fn.py` have **uncommitted** diffs that
   are only timing prints / `savefig` / commenting-out the (unused) polarised Jacobian block — no
   physics change. `3dtesting/`, `muram*.py`, etc. are untracked. Nothing blocks copying.

---

## 5. Open questions / not verified this session

- **Target repo identity/structure is unknown** — mapping it is the next step after files land there.
- **Sign conventions** for `vz` (up/down) and `B` inclination/azimuth in the target data vs. Adora's
  may differ; `muram_loader.py` already had to pick conventions (see [[adora-project-state]]). Confirm
  against whatever observations/atmospheres the target repo uses before trusting inverted `vz`,
  `gamma_b`, `chi_b`.
- **No code was run** — all of the above is from reading. The Lightweaver validation figure
  (`lte_polarised.png`) is from a prior session, not re-verified today.
- The polarised Jacobian in `vector_response_fn.py` is commented out; a PINN doesn't need it
  (`value_and_grad` of the scalar loss suffices), but if a full response-function cube is wanted,
  it's `vmap(jacrev(lte_polarised_rt, argnums=(3..10)))`.
