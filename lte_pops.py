import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
import astropy.constants as const

DEBROGLIE_CONST = (const.h / (2 * jnp.pi * const.k_B) * const.h / const.m_e).value
K_B_EV = const.k_B.to("eV / K").value


def lte_pops(
    energy,
    g,
    stage,
    temperature,
    ne,
    ntot,
):
    n_level = energy.shape[0]
    k_B_T = temperature * K_B_EV
    saha_term = 0.5 * ne * (DEBROGLIE_CONST / temperature)**(1.5)

    pops = jnp.empty((n_level))

    sum = 1.0
    for i in range(1, n_level):
        dE = energy[i] - energy[0]
        gi0 = g[i] / g[0]
        dZ = stage[i] - stage[0]

        dE_kBT = dE / k_B_T
        pop_i = gi0 * jnp.exp(-dE_kBT)
        pop_i /= saha_term**dZ
        sum += pop_i
        pops = pops.at[i].set(pop_i)

    pop_0 = ntot / sum
    pops = pops.at[0].set(pop_0)

    for i in range(1, n_level):
        pop_i = pops[i] * pop_0
        pops = pops.at[i].set(pop_i)
    return pops


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()
    import lightweaver as lw
    from lightweaver.rh_atoms import CaII_atom, H_6_atom
    from lightweaver.fal import Falc82
    import astropy.units as u

    Ca = CaII_atom()
    energies = jnp.array([(l.E_SI << u.Unit("J")).to("eV").value for l in Ca.levels])
    gs = jnp.array([l.g for l in Ca.levels])
    stages = jnp.array([l.stage + 1 for l in Ca.levels]) # à la crtaf/dexrt, but it won't matter here since we only look at stage differences

    fal = Falc82()

    rad_set = lw.RadiativeSet([H_6_atom(), CaII_atom()])
    eq_pops = rad_set.compute_eq_pops(fal)
    ref = eq_pops.atomicPops["Ca"].nStar

    ntot = lw.DefaultAtomicAbundance['Ca'] * fal.nHTot

    lte_pops_jit = jax.jit(jax.vmap(lte_pops, in_axes=[None, None, None, 0, 0, 0], out_axes=1))
    nstar = lte_pops_jit(energies, gs, stages, fal.temperature, fal.ne, ntot)

    dnstar_datmos = jax.jit(
        jax.vmap(
            jax.jacfwd(lte_pops, argnums=(3,4,5)),
            in_axes=[None, None, None, 0, 0, 0],
            out_axes=1,
        )
    )
    # dnstar_datmos = jax.vmap(
    #         jax.jacfwd(lte_pops, argnums=(3,4,5)),
    #         in_axes=[None, None, None, 0, 0, 0],
    #         out_axes=1,
    #     )
    nstar_response = dnstar_datmos(energies, gs, stages, fal.temperature, fal.ne, ntot)

    temperature_resp = nstar_response[0]
    ne_resp = nstar_response[1]

    fal_pert = Falc82()
    ne_pert = fal.ne * 1e-10
    temp_pert = fal.temperature * 0.01
    fal_pert.ne += ne_pert
    eq_pops_pert = rad_set.compute_eq_pops(fal_pert)
    ne_resp_fd = (eq_pops_pert['Ca'] - eq_pops['Ca']) / ne_pert

    fal_pert.ne -= ne_pert
    fal_pert.temperature += temp_pert
    eq_pops_pert = rad_set.compute_eq_pops(fal_pert)
    temperature_resp_fd = (eq_pops_pert['Ca'] - eq_pops['Ca']) / temp_pert

    plt.figure()

    for i in range(nstar.shape[0]):
        plt.plot(nstar[i])

    for i in range(nstar.shape[0]):
        plt.plot(ref[i], '--', c=f"C{i}")
    plt.yscale('log')

    plt.figure()
    plt.plot(temperature_resp.T)
    for i in range(nstar.shape[0]):
        plt.plot(temperature_resp_fd[i], '--', c=f"C{i}")
    plt.title("T response")
    plt.yscale("symlog")

    plt.figure()
    plt.plot(ne_resp.T)
    for i in range(nstar.shape[0]):
        plt.plot(ne_resp_fd[i], '--', c=f"C{i}")
    plt.title("ne response")
    plt.yscale("symlog", linthresh=1e-7)