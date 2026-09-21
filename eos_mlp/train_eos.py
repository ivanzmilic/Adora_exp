"""Train the tiny MLP EOS on a tabulated grid (see generate_grid.py).

    python train_eos.py --mode pg     # trains the (T, Pgas) -> Pe head
    python train_eos.py --mode rho    # trains the (T, rho)  -> Pe head
    python train_eos.py --mode both

Trains in CGS log10 space; inputs/outputs are normalized internally.  Saves the
MLP params + normalization stats + metadata into eos_pg.npz / eos_rho.npz, which
eos.py loads.  The model fits log10(Pe), so the reported validation error is the
relative error in Pe (== relative error in ne, since ne = Pe/(k_B T)).

A tiny full-batch problem -- forced onto CPU by default (JAX_PLATFORMS below) to
stay off the shared GPUs; override by unsetting it if you want the GPU.
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # tiny model; be polite on shared H100s

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

import eos_config as C
import eos_model as M


def _load_grid(mode):
    g = np.load(C.GRID_PG if mode == "pg" else C.GRID_RHO)
    T = g["T"]
    second = g["pgas"] if mode == "pg" else g["rho"]
    pe = g["pe"]                              # (n_T, n_second), CGS, NaN at bad corners
    TT, PP = np.meshgrid(T, second, indexing="ij")
    ok = np.isfinite(pe) & (pe > 0)
    x_raw = np.stack([np.log10(TT[ok]), np.log10(PP[ok])], axis=-1)  # (N, 2)
    y_raw = np.log10(pe[ok])                                          # (N,)
    return x_raw, y_raw, float(g["sum_a"]), float(g["mass_per_h"])


def _normalizers(x_raw, y_raw):
    xmin = x_raw.min(0)
    xmax = x_raw.max(0)
    ymean = y_raw.mean()
    ystd = y_raw.std()
    return xmin, xmax, ymean, ystd


def train_head(mode, steps=20000, lr=3e-3, seed=None):
    seed = C.SEED if seed is None else seed
    x_raw, y_raw, sum_a, mass_per_h = _load_grid(mode)
    xmin, xmax, ymean, ystd = _normalizers(x_raw, y_raw)

    xn = 2.0 * (x_raw - xmin) / (xmax - xmin) - 1.0
    yn = (y_raw - ymean) / ystd
    xn = jnp.asarray(xn); yn = jnp.asarray(yn)

    # train / val split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(yn))
    n_val = int(C.VAL_FRAC * len(yn))
    vi, ti = perm[:n_val], perm[n_val:]

    key = jax.random.PRNGKey(seed)
    params = M.init_params(key, n_in=2, hidden=C.HIDDEN,
                           n_fourier=C.N_FOURIER, fourier_scale=C.FOURIER_SCALE)

    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    state = opt.init(params)

    def loss_fn(p, xb, yb):
        return jnp.mean((M.forward(p, xb) - yb) ** 2)

    @jax.jit
    def step(p, s, xb, yb):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        updates, s = opt.update(g, s, p)
        return optax.apply_updates(p, updates), s, l

    xt, yt = xn[ti], yn[ti]
    xv, yv = xn[vi], yn[vi]
    for it in range(steps):
        params, state, l = step(params, state, xt, yt)
        if it % 2000 == 0 or it == steps - 1:
            # de-normalize to a Pe relative error (dex -> ratio)
            resid_dex = (M.forward(params, xv) - yv) * ystd
            rel = jnp.abs(10.0 ** resid_dex - 1.0)
            print(f"[{mode}] step {it:6d}  mse(norm)={float(l):.3e}  "
                  f"val Pe rel: median={float(jnp.median(rel)):.2e} "
                  f"max={float(jnp.max(rel)):.2e}")

    out = M.flatten(params)
    out.update(dict(
        xmin=xmin, xmax=xmax, ymean=np.float64(ymean), ystd=np.float64(ystd),
        mode=mode, sum_a=np.float64(sum_a), mass_per_h=np.float64(mass_per_h),
        # record the domain the model was trained on (for clipping at inference)
        log_x_lo=xmin, log_x_hi=xmax,
    ))
    path = C.PARAMS_PG if mode == "pg" else C.PARAMS_RHO
    np.savez(path, **out)
    print(f"[{mode}] saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["pg", "rho", "both"], default="both")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    modes = ["pg", "rho"] if args.mode == "both" else [args.mode]
    for m in modes:
        train_head(m, steps=args.steps, lr=args.lr)


if __name__ == "__main__":
    main()
