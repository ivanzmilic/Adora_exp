"""A very small, pure-JAX MLP with random Fourier features.

Learns a scalar y = log10(Pe) from a normalized 2-vector x = norm(log10 T, log10 P).
No flax/haiku dependency -- params are a plain pytree of jnp arrays so they
pickle/savez trivially and stay portable.

    params = init_params(key)
    y = forward(params, x)          # x: (..., 2) normalized -> (...,)

Fourier features (Tancik et al. 2020) up front let the tiny MLP resolve the
sharp hydrogen-ionization ridge without needing many layers.
"""

import jax
import jax.numpy as jnp


def init_params(key, n_in=2, hidden=(64, 64), n_out=1,
                n_fourier=32, fourier_scale=1.5):
    keys = jax.random.split(key, len(hidden) + 2)

    # fixed (non-trained) random Fourier projection
    if n_fourier and n_fourier > 0:
        B = fourier_scale * jax.random.normal(keys[0], (n_in, n_fourier))
        feat_dim = 2 * n_fourier          # sin + cos
    else:
        B = None
        feat_dim = n_in

    dims = [feat_dim, *hidden, n_out]
    layers = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        scale = jnp.sqrt(2.0 / din)       # He/Glorot-ish
        W = scale * jax.random.normal(keys[i + 1], (din, dout))
        b = jnp.zeros((dout,))
        layers.append((W, b))
    return {"B": B, "layers": layers}


def _featurize(params, x):
    B = params["B"]
    if B is None:
        return x
    proj = 2.0 * jnp.pi * (x @ B)
    return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)


def forward(params, x):
    """x: (..., n_in) normalized inputs -> (...,) scalar output."""
    h = _featurize(params, x)
    layers = params["layers"]
    for W, b in layers[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = layers[-1]
    out = h @ W + b
    return out[..., 0]


# --------------------------------------------------------------- (de)serialization
def flatten(params):
    """params pytree -> flat dict of numpy arrays (npz-friendly)."""
    import numpy as np
    out = {"n_layers": np.int64(len(params["layers"])),
           "has_B": np.int64(0 if params["B"] is None else 1)}
    if params["B"] is not None:
        out["B"] = np.asarray(params["B"])
    for i, (W, b) in enumerate(params["layers"]):
        out[f"W{i}"] = np.asarray(W)
        out[f"b{i}"] = np.asarray(b)
    return out


def unflatten(d):
    """flat dict (e.g. an npz handle) -> params pytree of jnp arrays."""
    n = int(d["n_layers"])
    B = jnp.asarray(d["B"]) if int(d["has_B"]) else None
    layers = [(jnp.asarray(d[f"W{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(n)]
    return {"B": B, "layers": layers}
