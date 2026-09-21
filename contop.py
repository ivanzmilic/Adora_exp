import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
import astropy.constants as const
import astropy.units as u

HC = const.h.value * const.c.value
NM_TO_M = u.Unit('nm').to('m')
M_TO_NM = u.Unit('m').to('nm')
E_RYD = const.Ryd.to('J', equivalencies=u.spectral()).value
Q_ELE = u.eV.to(u.J)
EPS_0 = const.eps0.value
M_ELE = const.m_e.value
K_B = const.k_B.value
H_CROSS_SECTION_C0 = 32.0 / (3.0 * jnp.sqrt(3.0)) * (Q_ELE / jnp.sqrt(4.0 * jnp.pi * EPS_0))**2 / (M_ELE * const.c.value) * const.h.value / (2.0 * E_RYD)
N_H_CONT = 5
SAHA_CONST = ((2 * jnp.pi * const.m_e.value * const.k_B.value) / const.h.value**2)**1.5

def gaunt_bf(wvl, nEff, charge) -> float:
    '''
    Gaunt factor for bound-free transitions, from Seaton (1960), Rep. Prog.
    Phys. 23, 313, as used in RH.

    Parameters
    ----------
    wvl : float or array-like
        The wavelength at which to compute the Gaunt factor [nm].
    nEff : float
        Principal quantum number.
    charge : float
        Charge of free state.

    Returns
    -------
    result : float or array-like
        Gaunt factor for bound-free transitions.
    '''
    # /* --- M. J. Seaton (1960), Rep. Prog. Phys. 23, 313 -- ----------- */
    x = HC / (wvl * NM_TO_M) / (E_RYD * charge**2)
    x3 = x**(1.0/3.0)
    nsqx = 1.0 / (nEff**2 * x)

    return 1.0 + 0.1728 * x3 * (1.0 - 2.0 * nsqx) - 0.0496 * x3**2 \
            * (1.0 - (1.0 - nsqx) * (2.0 / 3.0) * nsqx)

def h_bf_cont(wvl, i):
    """
    wvl: float
        The wavelength in nm
    i: int
        The lower level of the continuum (n), 0-indexed
    """

    Z = 1.0
    n = i + 1
    lambda_edge = HC * n**2 / E_RYD * M_TO_NM
    alpha0 = H_CROSS_SECTION_C0 * n * gaunt_bf(lambda_edge, n, 1.0)
    gbf0 = gaunt_bf(lambda_edge, n, Z)
    gbf = gaunt_bf(wvl, n, Z)
    alpha = jnp.where(
        wvl < lambda_edge,
        alpha0 * gbf / gbf0 * (wvl / lambda_edge)**2,
        0.0
    )
    return alpha

def hminus_ff_gray(wvl, temperature, ne, nhi):
    """
    wvl: float
        The wavelength in nm
    temperature: float
        The temperature in K
    ne: float
        The electron density in m-3
    nhi: float
        The number density of neutral H in m-3

    Computes the absorption in m-1 due to H- ff

    Follows Gray p.141 (2021 online)
    https://www.cambridge.org/highereducation/books/the-observation-and-analysis-of-stellar-photospheres/67B340445C56F4421BCBA0AFFAAFDEE0#contents
    """
    wvl_a = wvl * 10.0
    x1 = jnp.log10(wvl_a)
    x2 = x1 * x1
    x3 = x2 * x1
    f0 = -2.2763 - 1.6850 * x1 + 0.76661 * x2 - 0.0533464 * x3
    f1 = 15.2827 - 9.2846 * x1 + 1.99381 * x2 - 0.142631 * x3
    f2 = -197.789 + 190.266 * x1 - 67.9775 * x2 + 10.6913 * x3 - 0.625151 * x3 * x1
    thermal_log = jnp.log10(5040.0 / temperature)
    p_e = ne * K_B * temperature * 10 # to dyn/cm2
    sigma = jnp.where(
        (wvl > 260.0) & (wvl < 11390.0),
        1e-26 * p_e * 10**(f0 + f1 * thermal_log + f2 * thermal_log**2) * (nhi * 1e-6),
        0.0
    ) # in cm-1
    return sigma * 1e2

def hminus_bf_gray(wvl, temperature, ne, nhi):
    """
    wvl: float
        The wavelength in nm
    temperature: float
        The temperature in K
    ne: float
        The electron density in m-3
    nhi: float
        The number density of neutral H in m-3

    Computes the absorption in m-1 due to H- bf

    Follows Gray p.140 (2021 online)
    https://www.cambridge.org/highereducation/books/the-observation-and-analysis-of-stellar-photospheres/67B340445C56F4421BCBA0AFFAAFDEE0#contents
    """

    wvl_a = wvl * 10.0
    alpha = 0.1199654 + (-1.18267e-6 + (2.64243e-7 + (-4.40524e-11 + (3.23992e-15 + (-1.39568e-19 + 2.78701e-24 * wvl_a) * wvl_a) * wvl_a) * wvl_a) * wvl_a) * wvl_a
    p_e = ne * K_B * temperature * 10 # dyn/cm2
    theta = 5040.0 / temperature

    sigma = jnp.where(
        (wvl > 150.0) & (wvl < 1605.0),
        4.158e-10 * alpha * 1e-17 * p_e * theta**(2.5) * 10 ** (0.754 * theta) * (nhi * 1e-6),
        0.0
    ) # in cm-1
    return sigma * 1e2

def lte_h_ion_fracs(temperature, ne, nhtot):
    """
    temperature: float
        The temperature in K
    ne: float
        The electron density in m-3
    nhtot: float
        The number density of H (I and II) in m-3

    Computes nhi and nhii for the given point
    """
    # 2 g_hii / g_hi = 1
    saha = SAHA_CONST * temperature**1.5 * jnp.exp(-E_RYD / (K_B * temperature))
    nhi = nhtot / (1.0 + saha / ne)
    nhii = nhtot - nhi
    return nhi, nhii

def continuum_opacity(wvl, temperature, ne, nhtot):
    """
    wvl: float
        The wavelength in nm
    temperature: float
        The temperature in K
    ne: float
        The electron density in m-3
    nhtot: float
        The number density of H (I and II) in m-3

    Computes the continuum opacity in m-1
    """
    stimulated_correction = (1.0 - 10 ** (-1.2398e3*5040 / (wvl * temperature)))
    nhi, nhii = lte_h_ion_fracs(temperature, ne, nhtot)

    abs_h_bf = 0.0
    for i in range(0, N_H_CONT):
        n = i + 1
        dE_kT = E_RYD / (n**2 * K_B * temperature)
        saha_const = SAHA_CONST * temperature**1.5
        saha_term = ne * nhii / saha_const * jnp.exp(dE_kT)
        pop = n**2 * saha_term
        abs_h_bf = abs_h_bf + h_bf_cont(wvl, i) * pop

    abs_hm_bf = hminus_bf_gray(wvl, temperature, ne, nhi)
    abs_hm_ff = hminus_ff_gray(wvl, temperature, ne, nhi)

    return (abs_h_bf + abs_hm_bf) * stimulated_correction + abs_hm_ff

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()

    plt.figure()

    wave = jnp.linspace(50, 2000, 1001)
    temperature = 7715.0
    pe = 10**2.5 / 10
    # temperature = 6429.0
    # pe = 10**1.77 / 10
    ne = pe / (temperature * K_B)
    nhtot = 1e3 * ne
    continuum_opacity_jit = jax.jit(jax.vmap(continuum_opacity, in_axes=[0, None, None, None]))
    cop = continuum_opacity_jit(wave, temperature, ne, nhtot)

    plt.semilogy(wave, cop)

    import lightweaver as lw
    from lightweaver.rh_atoms import H_atom, CaII_atom
    import numpy as np
    n_depth = 2
    atmos = lw.Atmosphere.make_1d(
        lw.ScaleType.Geometric,
        depthScale=np.linspace(1, 0, n_depth) * 1e3,
        temperature=np.ones(n_depth) * temperature,
        vlos=np.zeros(n_depth),
        vturb=np.ones(n_depth) * 2e3,
        ne=np.ones(n_depth) * ne,
        nHTot=np.ones(n_depth) * nhtot
    )
    atmos.quadrature(3)
    rad_set = lw.RadiativeSet([H_atom(), CaII_atom()])
    rad_set.set_active("Ca")
    eq_pops = rad_set.compute_eq_pops(atmos)
    spect = rad_set.compute_wavelength_grid(extraWavelengths=wave)

    ctx = lw.Context(atmos, spect, eq_pops)
    plt.plot(ctx.spect.wavelength, ctx.background.chi[:, 0])


    dcont_op = jax.jacrev(continuum_opacity_jit, argnums=(1, 2, 3))
    dcop = dcont_op(wave, temperature, ne, nhtot)

    plt.figure()
    plt.plot(wave, dcop[0], label='dchi/dT')
    plt.plot(wave, dcop[1], label='dchi/dne')
    plt.plot(wave, dcop[2], label='dchi/dnhtot')
    plt.yscale('symlog', linthresh=1e-30)
    plt.legend()