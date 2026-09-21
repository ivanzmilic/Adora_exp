import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
from lineop import AtomicData, read_kurucz, emis_opac
from scalar_formal_solver import nearest_fs                          # previous piecewise-constant solver
from formal_solvers.scalar_formal_solver_linear import linear_fs     # piecewise-linear solver (now used)

def lte_rt(adata: AtomicData, wave, dz, temperature, ne, nhtot, vz, vturb):
    eta, chi = jax.vmap(
        emis_opac,
        in_axes=[None, None, 0, 0, 0, 0, 0]
    )(adata, wave, temperature, ne, nhtot, vz, vturb)

    I = linear_fs(dz, eta, chi)   # was nearest_fs; swap back to revert
    return I


if __name__ == "__main__":
    import lightweaver as lw
    from lightweaver.fal import Falc82
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()

    lines = read_kurucz("kurucz_6301_6302.linelist")

    fal = Falc82()
    dz = jnp.array(
        np.concatenate(
            [
                [fal.z[::-1][0] - fal.z[::-1][1]],
                fal.z[::-1][1:] - fal.z[::-1][:-1]
            ]
        )
    )
    temperature = jnp.array(fal.temperature[::-1])
    ne = jnp.array(fal.ne[::-1])
    nhtot = jnp.array(fal.nHTot[::-1])
    vturb = jnp.array(fal.vturb[::-1])
    vz = jnp.zeros(temperature.shape[0])

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    # emis_opac_atmos = jax.vmap(
    #     emis_opac,
    #     in_axes=[None, None, 0, 0, 0, 0, 0]
    # )
    # emis_opac_wave_atmos = jax.jit(jax.vmap(
    #     emis_opac_atmos,
    #     in_axes=[None, 0, None, None, None, None, None]
    # ))

    # eta, chi = emis_opac_wave_atmos(lines, waves, temperature, ne, nhtot, vz, vturb)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(eta.T, norm='log', aspect='auto')
    # ax[1].imshow(chi.T, norm='log', aspect='auto')

    # emis_opac_response = jax.jit(jax.vmap(
    #     jax.vmap(
    #         jax.jacrev(
    #             emis_opac,
    #             argnums=(2, 3, 4, 5, 6),
    #         ),
    #         in_axes=[None, None, 0, 0, 0, 0, 0]
    #     ),
    #     in_axes=[None, 0, None, None, None, None, None]
    # ))

    # grads = emis_opac_response(lines, waves, temperature, ne, nhtot, vz, vturb)
    # detadT = grads[0][0]
    # detadne = grads[0][1]
    # detadnhtot = grads[0][2]
    # detadvz = grads[0][3]
    # detadvt = grads[0][4]
    # dchidT = grads[1][0]
    # dchidne = grads[1][1]
    # dchidnhtot = grads[1][2]
    # dchidvz = grads[1][3]
    # dchidvt = grads[1][4]

    lte_rt_wave = jax.jit(
        jax.vmap(
            lte_rt,
            in_axes=[None, 0, None, None, None, None, None, None]
        )
    )
    import time
    start = time.time()
    intens = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb)
    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")
    start = time.time()
    intens = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb)
    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")

    plt.figure()
    plt.plot(waves, intens)
    plt.savefig("spectrum_scalar.png", bbox_inches='tight', dpi=300)

    lte_rt_response = jax.jit(
        jax.vmap(
            jax.jacrev(
                lte_rt,
                argnums=(3, 4, 5, 6, 7),
            ),
            in_axes=[None, 0, None, None, None, None, None, None]
        )
    )
    resp = lte_rt_response(lines, waves, dz, temperature, ne, nhtot, vz, vturb)
    dIdT = resp[0]
    dIdne = resp[1]
    dIdnhtot = resp[2]
    dIdv = resp[3]
    dIdvt = resp[4]

    def maxabs(a):
        return max(np.abs(a).max(), a.max())

    fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(8, 8))

    mappable = ax[0, 0].imshow(dIdT.T, aspect='auto')
    ax[0, 0].set_title('dI / dT')
    fig.colorbar(mappable, ax=ax[0, 0])

    m = maxabs(dIdne)
    mappable = ax[0, 1].imshow(dIdne.T, aspect='auto', vmin=-m, vmax=m, cmap='RdBu_r')
    ax[0, 1].set_title('dI / dne')
    fig.colorbar(mappable, ax=ax[0, 1])

    m = maxabs(dIdnhtot)
    mappable = ax[1, 0].imshow(dIdnhtot.T, aspect='auto', vmin=-m, vmax=m, cmap='PuOr_r')
    ax[1, 0].set_title('dI / dnhtot')
    fig.colorbar(mappable, ax=ax[1, 0])

    m = maxabs(dIdvt)
    mappable = ax[1, 1].imshow(dIdvt.T, aspect='auto', vmin=-m, vmax=m, cmap="RdYlBu_r")
    ax[1, 1].set_title('dI / dvturb')
    fig.colorbar(mappable, ax=ax[1, 1])


