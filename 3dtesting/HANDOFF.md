# Adora — Session Handoff (updated 2026-09-19)

**Goal** (see `instructions.md`): extend Adora — a differentiable JAX LTE Stokes RT code —
to (1) synthesize spectra from 3D cubes, (2) improve the formal solver, (3) build a PINN /
neural-field inversion.

## Done so far
- **Env `adora-develop`** created & verified (Python 3.12). Pins match the README:
  jax 0.7.1, optax 0.2.5, jaxopt 0.8.5, interpax 0.3.10, jax_dataclasses 1.6.3,
  lightweaver 0.16.1, numpy 2.3, numba 0.67, tqdm 4.70. Activate: `conda activate adora-develop`.
- **Summaries** — 8 PDFs (`.pdf`+`.tex`, sharing `adora_preamble.tex`), now all in **`docs/`**:
  `adora_1_state` … `adora_6_parallel_memory`, `adora_7_pinn_inversion`,
  `adora_8_gpu_synthesis` (2026-09-19: GPU forward synth + benchmarks).
  Build: `cd docs && pdflatex adora_N_*.tex` then rm the `.aux/.log/.out`.
- **GPU forward synthesis (2026-09-19)** — `synth3d.py` runs on the H100 **unchanged** (pure jax.jit/vmap).
  `adora-develop` JAX 0.7.1 is the **CUDA12 build**; 3× H100 80 GB on the node. Benchmarked the MURaM
  cube on one GPU: **~11.9k cols/s** (skip=8 192² in ~3.2 s warm; quarter full-res 768² in ~49 s),
  bit-identical to the CPU-mp reference (fp64 round-off). Added **`synth_cube_map`** (GPU twin of
  `synth_cube_mp`). Cubes saved to `/dat/milic/adora_synth/synthesized_cube_gpu*.npy`. See `docs/adora_8_gpu_synthesis.pdf`.
- **`synth3d.py`** — 1.5D column-by-column Stokes synthesis. `synth_cube` (differentiable) +
  **`synth_cube_fwd`** (forward-only: `stop_gradient`, no autodiff tape; bit-identical, grad=0).
- **`synth3d_mp.py`** — process-parallel synthesis with a **tqdm progress bar** (tiled, streamed).
- **`muram_loader.py`** — MURaM snapshot → SI synth cube (validated by round-tripping FALC).
- **First real cube synthesised** — `3D_full_subdomain` iter 0 (`muramsub`) → `/dat/milic/synthesized_cube.npy`
  (905 MB, `skip=4`). Ran clean; **memory peaked ~540 GB** of the 1006 GB node (see Memory below).

## Things to know
- **Env gotcha:** lightweaver is NOT pip-installable here (Cython 3.x fails on `LwMiddleLayer.pyx`).
  It's wired by reusing the prebuilt cp312 `.so` at `/home/milic/codes/Lightweaver` via setuptools
  editable-finder artifacts from the `lw` env. `environment.yml` omits it — wire separately after create.
- **Code fix (uncommitted):** `contop.py` lines 1–3 — re-enabled the `jax`/`jnp` import an uncommitted
  edit had commented out; without it all synthesis raises `NameError: jnp`. `git checkout contop.py` reverts.
- **spawn footgun:** any script calling `synth_cube_mp` MUST guard it under `if __name__=="__main__"`,
  or spawn re-imports the module and recursively re-spawns (fork-bomb).
- **MEMORY (big):** a worker synthesises a whole **tile** as ONE triply-nested `vmap`
  (columns×wavelength×depth), so peak RSS/worker ≈ ~1 GB (JAX) + `K·tile·nwave·nz` — all the opacity
  intermediates live at once. The auto default `tile≈npix/(4·nworkers)` → ~1100 cols → **~17 GB/worker**,
  ×32 → ~540 GB peak (spiky, frees between tiles). **`tile` is the memory lever** (linear; total work
  unchanged). `nwave`/`nz` are fixed by the physics. See `adora_6_parallel_memory.pdf`.
- **GPU memory / batching:** the same `(cols×nwave×nz×4×4)` opacity is live per `synth_cube` call, so the
  *whole* skip=8 cube ≈ **115 GB** — over one 80 GB H100. **Column-batch** on GPU exactly like the mp tile:
  `B ≈ budget/(nwave·nz·128)`; an 8 GB budget → `B≈2560` cols. `synth_cube_map` does NOT yet auto-batch
  (whole cube in one call) — batch manually for real cubes (recipe above / `docs/adora_8`).
- **GPU env vars (set BEFORE `import jax`):** `CUDA_VISIBLE_DEVICES=<n>` picks one card (renumbers to
  `cuda:0` in-process); `XLA_PYTHON_CLIENT_PREALLOCATE=false` stops XLA grabbing ~75% up front (polite on
  the shared node). **`synth3d_mp.py` is CPU-only** (core pinning); for GPU call `synth_cube`/`synth_cube_map`.

## How the pieces fit
- `synth3d.py`: `synth_column` = vmap `lte_polarised_rt` over wavelength; `_synth_cube` = vmap over
  pixels; `synth_cube = jit(_synth_cube)`; `synth_cube_fwd = jit(stop_gradient(_synth_cube))`.
  Flatten `(nx,ny,nz)`→`(npix,nz)`, vmap, reshape → `(nx,ny,4,nwave)`. Vertical rays / shared z-grid;
  depth index 0 = DEEP boundary (`I_start=Planck(T[0])`). **`synth_cube_map(waves, dz, cube, linelist=…,
  adata=None)`** = GPU twin of `synth_cube_mp` (same dict `cube` in, `(nx,ny,4,nwave)` out) but returns a
  device array and runs the whole cube at once (batch manually — see GPU MEMORY note).
- `synth3d_mp.py`: `synth_cube_mp(waves, dz, cube, nworkers=None, cores_per=1, linelist=…, tile=None,
  progress=True)`. `spawn` Pool + **initializer** (pin + import JAX + read linelist ONCE per worker,
  stored in module `_W`, reused across tiles); worker runs **`synth_cube_fwd`**; tiles streamed via
  `imap_unordered`, placed by start index, wrapped in tqdm. Bit-identical to single-device.
  **COLD-START** still ~60–90 s (workers cold-import JAX + compile on first tile). Module JAX-free at import.
- `muram_loader.py`: `load_muram_cube(datapath, iteration, kind, …) -> (cube, dz, meta)`.
  MURaM CGS→SI (ρ·1e3, ne·1e6, v·1e-2, B·√4π·1e-4→Tesla; 3D `MuramCube` does NOT pre-apply √4π).
  nH=ρ/(1.4·amu); gb/cb from B; vturb=0. Orientation check `nH_deep>nH_top` (T fails — corona).
  Needs MURaM EOS `ne`. **Current CLI:** `skip=2`, then `synth_cube_mp(…, nworkers=128, tile=16)` →
  saves `/dat/milic/synthesized_cube.npy`. (A single-device forward-only smoke test on a small patch is
  present but commented out just above the mp call.)

## Benchmark result (256-core node) — see `adora_4_benchmark.pdf`
- Process-parallel 64×1-core: **219 µs/px**, 100×100 in ~2.2 s/run. **`pmap` over host CPU devices is a
  DEAD END** (366 s compile). Scale by many single-core workers, not one wide XLA program.

## Next steps (resume here)
0. **Auto-batch the GPU path** — make `synth_cube_map` chunk columns from an `nwave·nz` memory budget
   (on-device analogue of the `tile` fix) so a real cube can't OOM one card. Then **multi-GPU**: shard
   columns across the 3 H100s with `jax.shard_map`/`pmap` *over GPU devices* (unlike the CPU pmap dead end,
   this is the right tool) for ~3× on full-domain runs. At full res the **cube load (I/O) now dominates** —
   cache/convert once or read only the crop.
1. **Memory-bounded default `tile`** — cap the auto default from an `nwave·nz` budget (~1–2 GB/worker) so
   an unattended full-res run can't OOM the node; keep `tile=` overridable. (Small change to `synth_cube_mp`.)
2. **Validate `muram_loader` on the REAL snapshot** — orientation & velocity/field sign conventions;
   inspect `/dat/milic/synthesized_cube.npy` (see `inspect_spectra.ipynb`); cross-check one column vs Lightweaver.
3. **Persistent worker pool** — compile once, reuse across calls; prerequisite for inversions (kills cold-start).
4. Inclined rays µ<1: project `vz` & `B` on LOS; generalise polarised opacity (assumes µz=1).
5. EOS for `ne`, `nHI` from `(T, ρ/P)`; more partition functions; Unsöld vdW. Response cube via `vmap(jacrev)`.
6. Neural field (Plan B): `neural_field.py` + `invert_field.py` (optax loop through the synth).

## Map
- Deliverables in `3dtesting/`: `environment.yml`, `synth3d.py` (`synth_cube`/`synth_cube_fwd`/`synth_cube_map`),
  `synth3d_mp.py`, `muram_loader.py`, `synth3d.png`, `benchmark_scaling.png`, `inspect_spectra.ipynb`,
  `HANDOFF.md`, `PINN_HANDOFF.md`. The **8 `.tex`/`.pdf` summaries now live in `docs/`** (`adora_1…8`).
- Core repo modules (parent dir): `lineop.py` (line opacity + Zeeman), `contop.py` (continuum),
  `voigt.py`, `vector_formal_solver.py` (DELO), `vector_response_fn.py` (`lte_polarised_rt`),
  `nodes.py` (node inversion), `response_fn.py` (scalar path), `muram.py` (MURaM reader),
  `muram_to_cube.py` (reference CGS→npy packer).
