"""
pretrain.py -- Step 3 of the Adora PINN plan (pinn/plan.md).

Fit the step-1 atmosphere field (atmosphere_field.py) to the step-2 starting atmosphere so the PINN
begins the inversion "warm". This is a plain supervised fit of f_theta(x,y,z) to the stored cube:
random minibatches of grid points, per-channel-balanced loss (P compared in log10), optax Adam.

No RT here -- that is step 4. Trained params are saved so step 4/5 can start from them.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (adora_precision)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # pinn/
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax, jax.numpy as jnp
import numpy as np
import optax
import atmosphere_field as af

DEFAULT_ATM = "/dat/milic/adora_pinn_develop/test_atm_falc_128.npz"
# per-channel scale for a balanced loss (physical units per unit residual); P is compared in log10.
LOSS_SCALE = {"T": 1.0e3, "vz": 3.0e3, "Bx": 0.05, "By": 0.05, "Bz": 0.05, "vturb": 1.0e3}


def load_target(path=DEFAULT_ATM):
    """Return (coords (N,3) normalized, tgt7 (N,7) in CHANNELS order, shape (nx,ny,nz), z_km)."""
    d = np.load(path, allow_pickle=True)
    nx, ny, nz = int(d["nx"]), int(d["ny"]), int(d["nz"])
    coords = af.grid_coords(nx, ny, nz)                              # matches cube (x,y,z) flatten order
    tgt7 = jnp.stack([jnp.asarray(d[c]).reshape(-1) for c in af.CHANNELS], axis=-1)  # (N,7)
    return coords, tgt7, (nx, ny, nz), np.asarray(d["z_km"])


def _residual(out, tgt7):
    """Per-channel comparable residual (P in log10, others scaled) -> (M,7)."""
    cols = []
    for j, ch in enumerate(af.CHANNELS):
        tg = tgt7[..., j]
        if ch == "P":
            cols.append(jnp.log10(out["P"]) - jnp.log10(tg))
        else:
            cols.append((out[ch] - tg) / LOSS_SCALE[ch])
    return jnp.stack(cols, axis=-1)


def loss_fn(params, coords, tgt7):
    out = af.forward(params, coords)
    return jnp.mean(_residual(out, tgt7) ** 2)


def train(params, coords, tgt7, steps=6000, batch=8192, lr=3e-3, seed=0, n_log=30, progress=True):
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    state = opt.init(params)

    @jax.jit
    def step(params, state, idx):
        cb = coords[idx]
        tb = tgt7[idx]
        l, g = jax.value_and_grad(loss_fn)(params, cb, tb)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l

    N = coords.shape[0]
    key = jax.random.PRNGKey(seed)
    hist = []
    every = max(1, steps // n_log)

    it_range = range(steps)
    bar = None
    if progress:                                    # tqdm bar to follow training (auto: notebook or text)
        try:
            from tqdm.auto import tqdm
            bar = tqdm(it_range, total=steps, desc="pretrain", unit="step")
            it_range = bar
        except ImportError:
            pass

    for it in it_range:
        key, sk = jax.random.split(key)
        idx = jax.random.randint(sk, (batch,), 0, N)
        params, state, l = step(params, state, idx)
        if it % every == 0 or it == steps - 1:
            hist.append((it, float(l)))
            if bar is not None:
                bar.set_postfix(loss=f"{hist[-1][1]:.2e}")   # loss synced only on the log cadence
    return params, np.array(hist)


def predict_grid(params, coords, chunk=131072):
    """Field prediction over all coords, batched to bound memory -> dict of (N,) arrays (host)."""
    outs = {c: [] for c in af.CHANNELS}
    for s in range(0, coords.shape[0], chunk):
        o = af.forward(params, coords[s:s + chunk])
        for c in af.CHANNELS:
            outs[c].append(np.asarray(o[c]))
    return {c: np.concatenate(outs[c]) for c in af.CHANNELS}


def channel_errors(pred, tgt7):
    """Per-channel error summary vs the target (RMS in physical units; P also as dex)."""
    rep = {}
    for j, ch in enumerate(af.CHANNELS):
        p = pred[ch]; t = np.asarray(tgt7[:, j])
        rms = float(np.sqrt(np.mean((p - t) ** 2)))
        extra = f", {float(np.sqrt(np.mean((np.log10(p) - np.log10(t))**2))):.2e} dex" if ch == "P" else ""
        rep[ch] = f"RMS={rms:.3e}{extra}"
    return rep


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    coords, tgt7, shape, z_km = load_target()
    key = jax.random.PRNGKey(0)
    params = af.init_params(key)
    print("initial loss:", float(loss_fn(params, coords, tgt7)))
    params, hist = train(params, coords, tgt7, steps=500)   # quick smoke run
    print("final loss (500 steps):", hist[-1, 1])
    pred = predict_grid(params, coords)
    for ch, s in channel_errors(pred, tgt7).items():
        print(f"  {ch:5s} {s}")
