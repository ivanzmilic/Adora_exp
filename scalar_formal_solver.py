import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
from jax.lax import fori_loop


def cumsum_fs(dz, emis, opac):
    dtau = opac * dz
    tau_above_layer = jnp.cumsum(dtau) - dtau[0]
    transmittance = jnp.exp(-tau_above_layer)
    source_fn = emis / (opac + 1e-15)
    local_contribution = (1.0 - jnp.exp(-dtau)) * source_fn
    outgoing_contribution = local_contribution * transmittance
    I = jnp.sum(outgoing_contribution)
    return I

def nearest_fs(dz, emis, opac):
    result = fori_loop(
        0,
        dz.shape[0],
        lambda i, x: x * jnp.exp(-opac[i] * dz[i]) + (1.0 - jnp.exp(-opac[i] * dz[i])) * emis[i] / (opac[i] + 1e-15),
        0.0
    )
    return result

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()

    n_depth = 20
    n_wave = 101
    wave = jnp.linspace(-5, 5, n_wave)
    profile = jnp.exp(-wave**2)
    eta = jnp.ones((n_wave, n_depth)) * 1e-5 * profile[:, None] + 1e-6
    chi = jnp.ones((n_wave, n_depth)) * 1e-3 * profile[:, None] + 1e-6
    # eta = jnp.ones((n_wave, n_depth)) * 1e-5 * profile[:, None]
    # chi = jnp.ones((n_wave, n_depth)) * 1e-3 * profile[:, None]
    dz = jnp.ones(n_depth) * 1e2

    fs_fn = cumsum_fs
    fs_fn = nearest_fs
    fs = jax.jit(
        jax.vmap(
            fs_fn,
            in_axes=[None, 0, 0],
            out_axes=0
        )
    )
    dfs = jax.jit(
        jax.vmap(
            jax.jacrev(
                fs_fn,
                argnums=(1, 2)
            ),
            in_axes=[None, 0, 0],
            out_axes=0,
        )
    )

    I = fs(dz, eta, chi)
    dI = dfs(dz, eta, chi)

    plt.figure()
    plt.plot(wave, I)

    plt.figure()
    plt.imshow(dI[0])
    plt.colorbar()
    plt.title("dI/deta")

    plt.figure()
    plt.imshow(dI[1])
    plt.colorbar()
    plt.title("dI/dchi")