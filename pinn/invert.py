"""
invert.py -- Step 7 of the Adora PINN plan (pinn/plan.md).

The prototype inversion loop: optimise the atmosphere-field weights to fit the observed Stokes testset,
regularised by hydrostatic equilibrium and the top-boundary pressure. FRESH random samples every step
(fit pixels + collocation points + boundary pixels); a separate FIXED monitor set gives clean loss curves.

CONTINUABLE: run_inversion takes either a params pytree (fresh start) or a checkpoint dict from a previous
call, and returns a checkpoint {params, state, key, step, hist, monitor}. Feed it back in to train more --
the Adam moments, RNG, the LR schedule (via the optimiser's step count) and the history all carry over.

SCHEDULER: optax exponential decay of the learning rate -- the standard, open-ended-continuation-friendly
choice for PINNs: lr(step) = lr * lr_decay**(step / lr_transition), floored at lr_floor.

WEIGHTS: per-Stokes fit weights are HAND-SET (sigma_stokes, x mean continuum), not data-derived, so I,Q,U,V
are not weighted by orders of magnitude.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "3dtesting"))
sys.path.insert(0, os.path.join(REPO, "eos_mlp")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adora_precision   # global float32/64 switch
import jax, jax.numpy as jnp
import numpy as np
import optax
import atmosphere_field as af
import losses as L


def run_inversion(start, eos_params, obs, waves, dz, adata, P_top, *,
                  steps=1000, m_fit=384, k_he=1024, m_bnd=256,
                  w_fit=1.0, w_he=1.0, w_bnd=1.0e2,
                  sigma_stokes=(1e-2, 5e-3, 5e-3, 5e-3),        # HAND-SET per-Stokes noise, x mean continuum
                  lr=1.0e-3, lr_decay=0.9, lr_transition=1000, lr_floor=1.0e-5,
                  nz=56, seed=0, n_monitor=40, progress=True):
    synth_fn = L.make_synth(eos_params)
    obs = jnp.asarray(obs); waves = jnp.asarray(waves); dz = jnp.asarray(dz)
    Ic = float(obs[..., 0, 0].mean())
    sigma = jnp.asarray(sigma_stokes, dtype=obs.dtype) * Ic      # per-Stokes weights, set by hand
    nx, ny = obs.shape[0], obs.shape[1]

    schedule = optax.exponential_decay(lr, lr_transition, lr_decay, end_value=lr_floor)
    opt = optax.adam(schedule)

    def losses_of(p, ix, iy, obs_b, colloc, ixb, iyb):
        f = L.fit_loss(p, synth_fn, adata, obs_b, ix, iy, nz, waves, dz, sigma)
        h = L.he_loss(p, eos_params, colloc)
        b = L.boundary_loss(p, ixb, iyb, P_top)
        return f, h, b

    def total_of(p, *a):
        f, h, b = losses_of(p, *a)
        return w_fit * f + w_he * h + w_bnd * b

    @jax.jit
    def step(p, state, ix, iy, obs_b, colloc, ixb, iyb):
        l, g = jax.value_and_grad(total_of)(p, ix, iy, obs_b, colloc, ixb, iyb)
        u, state = opt.update(g, state, p)
        return optax.apply_updates(p, u), state, l
    monitor = jax.jit(losses_of)

    # fresh start (a field params pytree) or continue (a checkpoint dict, which has a 'params' key)
    if isinstance(start, dict) and "params" in start:
        params, state, key, step0 = start["params"], start["state"], start["key"], int(start["step"])
        hist = {k: list(v) for k, v in start["hist"].items()}
        mon = start["monitor"]
    else:
        params, state, key, step0 = start, opt.init(start), jax.random.PRNGKey(seed), 0
        hist = {"step": [], "fit": [], "he": [], "bnd": [], "total": [], "lr": []}
        key, *mk = jax.random.split(key, 6)                       # FIXED monitor set (drawn once, reused)
        mix = jax.random.randint(mk[0], (m_fit,), 0, nx); miy = jax.random.randint(mk[1], (m_fit,), 0, ny)
        mon = (mix, miy, obs[mix, miy],
               jax.random.uniform(mk[2], (k_he, 3), minval=-1.0, maxval=1.0),
               jax.random.randint(mk[3], (m_bnd,), 0, nx), jax.random.randint(mk[4], (m_bnd,), 0, ny))

    it_range = range(steps); bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(it_range, total=steps, desc="invert", unit="step"); it_range = bar
        except ImportError:
            pass

    every = max(1, steps // n_monitor)
    for it in it_range:
        gstep = step0 + it
        key, *ks = jax.random.split(key, 6)                       # FRESH samples each step
        ix = jax.random.randint(ks[0], (m_fit,), 0, nx); iy = jax.random.randint(ks[1], (m_fit,), 0, ny)
        obs_b = obs[ix, iy]
        colloc = jax.random.uniform(ks[2], (k_he, 3), minval=-1.0, maxval=1.0)
        ixb = jax.random.randint(ks[3], (m_bnd,), 0, nx); iyb = jax.random.randint(ks[4], (m_bnd,), 0, ny)
        params, state, _ = step(params, state, ix, iy, obs_b, colloc, ixb, iyb)
        if it % every == 0 or it == steps - 1:
            f, h, b = monitor(params, *mon)
            hist["step"].append(gstep); hist["fit"].append(float(f)); hist["he"].append(float(h))
            hist["bnd"].append(float(b)); hist["total"].append(float(w_fit*f + w_he*h + w_bnd*b))
            hist["lr"].append(float(schedule(gstep)))
            if bar is not None:
                bar.set_postfix(total=f"{hist['total'][-1]:.2e}", fit=f"{float(f):.2e}", lr=f"{hist['lr'][-1]:.1e}")
    return {"params": params, "state": state, "key": key, "step": step0 + steps,
            "hist": {k: np.array(v) for k, v in hist.items()}, "monitor": mon}
