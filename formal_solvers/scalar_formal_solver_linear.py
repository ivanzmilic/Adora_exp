import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (adora_precision)
import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
from jax.lax import fori_loop


def linear_fs(dz, emis, opac):
    def body(i, intens):
        eta_m = emis[i - 1] / (opac[i - 1] + 1e-15)     # upwind source S_{i-1}
        eta   = emis[i]     / (opac[i]     + 1e-15)     # local source  S_i
        dtau = 0.5 * (opac[i] + opac[i - 1]) * dz[i]

        # PHYSICS: linear source across the layer (Olson & Kunasz 1987) instead of the per-layer
        # constant source of nearest_fs. psi_a weights the upwind point, psi_b the local one; they
        # sum to (1 - E_k) so the constant-source limit is unchanged.
        E_k = jnp.exp(-dtau)
        w0 = -jnp.expm1(-dtau)          # 1 - E_k
        w1 = w0 - dtau * E_k            # 1 - (1 + dtau) E_k

        # w1/dtau is 0/0 as dtau -> 0 (finite value, NaN gradient): Taylor below a threshold with the
        # double-where so the discarded branch never divides by zero.
        eps = 1e-3
        safe = jnp.where(dtau < eps, 1.0, dtau)
        psi_a = jnp.where(dtau < eps, 0.5*dtau - dtau**2/3.0 + dtau**3/8.0,  w1 / safe)
        psi_b = jnp.where(dtau < eps, 0.5*dtau - dtau**2/6.0 + dtau**3/24.0, w0 - w1 / safe)

        intens = intens * E_k + psi_a * eta_m + psi_b * eta
        return intens

    # NOTE: start from I = 0 at the lower boundary (as nearest_fs does) and integrate upward from
    # index 1 (the linear step needs the upwind point i-1).
    intens = fori_loop(
        1,
        dz.shape[0],
        body,
        0.0
    )
    return intens

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_depth = 20
    n_wave = 101
    wave = jnp.linspace(-5, 5, n_wave)
    profile = jnp.exp(-wave**2)
    eta = jnp.ones((n_wave, n_depth)) * 1e-5 * profile[:, None] + 1e-6
    chi = jnp.ones((n_wave, n_depth)) * 1e-3 * profile[:, None] + 1e-6
    dz = jnp.ones(n_depth) * 1e2

    fs = jax.jit(
        jax.vmap(
            linear_fs,
            in_axes=[None, 0, 0],
            out_axes=0
        )
    )
    dfs = jax.jit(
        jax.vmap(
            jax.jacrev(
                linear_fs,
                argnums=(1, 2)
            ),
            in_axes=[None, 0, 0],
            out_axes=0,
        )
    )

    I = fs(dz, eta, chi)
    dI = dfs(dz, eta, chi)
    import numpy as np
    print(f"I finite={bool(np.isfinite(I).all())} shape={I.shape}")
    print(f"dI/deta finite={bool(np.isfinite(dI[0]).all())}  dI/dchi finite={bool(np.isfinite(dI[1]).all())}")

    plt.figure()
    plt.plot(wave, I)
    plt.savefig(os.path.join(os.path.dirname(__file__), "scalar_linear.png"), bbox_inches='tight', dpi=150)
