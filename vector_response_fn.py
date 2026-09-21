from time import time
import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
from lineop import AtomicData, read_kurucz, emis_opac_polarised, planck
from vector_formal_solver import delo_constant_fs                       # previous piecewise-constant solver
from formal_solvers.vector_formal_solver_linear import delo_linear_fs   # piecewise-linear solver (now used)

def lte_polarised_rt(adata: AtomicData, wave, dz, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b):
    eta, chi = jax.vmap(
        emis_opac_polarised,
        in_axes=[None, None, 0, 0, 0, 0, 0, 0, 0, 0]
    )(adata, wave, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b)

    I_start = jnp.array([planck(wave, temperature[0]), 0.0, 0.0, 0.0])
    I = delo_linear_fs(dz, I_start, eta, chi)   # was delo_constant_fs; swap back to revert
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

    fal = Falc82()
    import time
    
    lines = read_kurucz("kurucz_6301_6302.linelist")

   
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
    b = jnp.ones(temperature.shape[0]) * 0.05
    # gamma_b = jnp.zeros(temperature.shape[0])
    # ~ 45 degrees
    gamma_b = jnp.ones(temperature.shape[0]) * 0.785
    chi_b = jnp.zeros(temperature.shape[0])

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    lte_rt_wave = jax.jit(
        jax.vmap(
            lte_polarised_rt,
            in_axes=[None, 0, None, None, None, None, None, None, None, None, None],
            out_axes=1,
        )
    )
    start = time.time()
    intens = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b)
    end = time.time()
    print (jax.devices())
    print(f"Execution time: {end - start:.4f} seconds")
    
    plt.figure()
    plt.plot(waves, intens[0] / intens[0, 0], label="I")
    plt.plot(waves, intens[3] / intens[0, 0], label="V")

    from lightweaver.rh_atoms import H_atom, Fe23_atom
    fal_lw = Falc82()
    fal_lw.B = np.ones(temperature.shape[0]) * 0.05
    fal_lw.gammaB = np.ones(temperature.shape[0]) * 0.785
    fal_lw.chiB = np.zeros(temperature.shape[0])
    fal_lw.quadrature(3)

    rad_set = lw.RadiativeSet([H_atom(), Fe23_atom()])
    rad_set.set_detailed_static("Fe")
    spect = rad_set.compute_wavelength_grid()
    eq_pops = rad_set.compute_eq_pops(fal_lw)

    ctx = lw.Context(fal_lw, spect, eq_pops)
    Iquv_lw = ctx.compute_rays(wavelengths=np.array(waves), mus=[1.0], stokes=True)
    plt.plot(waves, Iquv_lw[0] / Iquv_lw[0, 0], '--', label="I Lw")
    plt.plot(waves, Iquv_lw[3] / Iquv_lw[0, 0], '--', label="V Lw")
    plt.legend()

    plt.savefig("lte_polarised.png",bbox_inches='tight', dpi=300)

    '''lte_polarised_rt_response = jax.jit(
        jax.vmap(
            jax.jacrev(
                lte_polarised_rt,
                argnums=(3, 4, 5, 6, 7, 8, 9, 10),
            ),
            in_axes=[None, 0, None, None, None, None, None, None, None, None, None],
            out_axes=1,
        )
    )
    resp = lte_polarised_rt_response(lines, waves, dz, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b)'''
    # dIdT = resp[0]
    # dIdne = resp[1]
    # dIdnhtot = resp[2]
    # dIdv = resp[3]
    # dIdvt = resp[4]

    # def maxabs(a):
    #     return max(np.abs(a).max(), a.max())

    # fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(8, 8))

    # mappable = ax[0, 0].imshow(dIdT.T, aspect='auto')
    # ax[0, 0].set_title('dI / dT')
    # fig.colorbar(mappable, ax=ax[0, 0])

    # m = maxabs(dIdne)
    # mappable = ax[0, 1].imshow(dIdne.T, aspect='auto', vmin=-m, vmax=m, cmap='RdBu_r')
    # ax[0, 1].set_title('dI / dne')
    # fig.colorbar(mappable, ax=ax[0, 1])

    # m = maxabs(dIdnhtot)
    # mappable = ax[1, 0].imshow(dIdnhtot.T, aspect='auto', vmin=-m, vmax=m, cmap='PuOr_r')
    # ax[1, 0].set_title('dI / dnhtot')
    # fig.colorbar(mappable, ax=ax[1, 0])

    # m = maxabs(dIdvt)
    # mappable = ax[1, 1].imshow(dIdvt.T, aspect='auto', vmin=-m, vmax=m, cmap="RdYlBu_r")
    # ax[1, 1].set_title('dI / dvturb')
    # fig.colorbar(mappable, ax=ax[1, 1])


