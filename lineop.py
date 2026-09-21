from fractions import Fraction
import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
import astropy.constants as const
import astropy.units as u
import numpy as np
from contop import lte_h_ion_fracs, continuum_opacity
from voigt import voigt_H_re as voigt, voigt_H as voigt_HF
import jax_dataclasses as jdc
import lightweaver as lw
from lightweaver.zeeman import lande_factor, fraction_range, zeeman_strength

HC = const.h.value * const.c.value
NM_TO_M = u.Unit('nm').to('m')
M_TO_NM = u.Unit('m').to('nm')
E_RYD = const.Ryd.to('J', equivalencies=u.spectral()).value
Q_ELE = u.eV.to(u.J)
EPS_0 = const.eps0.value
M_ELE = const.m_e.value
K_B = const.k_B.value
K_B_EV = const.k_B.to('eV / K').value
K_B_U = (const.k_B / const.u).value
INV_C = 1.0 / const.c.value
INV_FOURPI_C = 1.0 / (4.0 * np.pi * const.c.value)
HC_FOURPI_KJ_NM = (const.h * const.c).to('kJ nm').value / (4.0 * np.pi)
SQRT_PI = np.sqrt(np.pi)
SAHA_CONST = ((2 * jnp.pi * const.m_e.value * const.k_B.value) / const.h.value**2)**1.5
TWOHC2_NM5 = 2.0 * (const.h.to('kJ s') * const.c**2 / (1e-9**4)).value
HC_KB_NM = (const.h * const.c / const.k_B).to('K nm').value
DLAMBDA_B_CONST = (Q_ELE / (4.0 * jnp.pi * const.m_e * const.c.to('nm/s'))).value

@jdc.pytree_dataclass
class AtomicData:
    mass: jax.Array
    elem: jax.Array
    stage: jax.Array
    abund: jax.Array
    lambda0: jax.Array
    log_grad: jax.Array
    log_gs: jax.Array
    log_gw: jax.Array
    gi: jax.Array
    gj: jax.Array
    ei: jax.Array
    ej: jax.Array
    Aji: jax.Array
    zeeman_alphas: jax.Array
    zeeman_strengths: jax.Array
    zeeman_shifts: jax.Array


# https://github.com/HajimeKawahara/exojax/blob/master/src/exojax/database/atomllapi.py
def air_to_vac(wlair):
    """Convert wavelengths [AA] in air into those in vacuum.

    * See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Args:
        wlair:  wavelengthe in air [Angstrom]
        n:  Refractive Index in dry air at 1 atm pressure and 15ºC with 0.045% CO2 by volume (Birch and Downs, 1994, Metrologia, 31, 315)

    Returns:
        wlvac:  wavelength in vacuum [Angstrom]
    """
    s = 1e4 / wlair
    n = (
        1.0
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s * s)
        + 0.0001599740894897 / (38.92568793293 - s * s)
    )
    wlvac = wlair * n
    return wlvac

# Modified from https://github.com/HajimeKawahara/exojax/blob/master/src/exojax/database/atomllapi.py
def read_kurucz(kuruczf):
    """Input Kurucz line list (http://kurucz.harvard.edu/linelists/)

    Args:
        kuruczf: file path

    Returns:
        AtomicData, containing some of:
        A:  Einstein coefficient in [s-1]
        wavelength:  vacuum transition wavelength in [nm]
        elower: lower excitation potential [eV]
        eupper: upper excitation potential [eV]
        glower: lower statistical weight
        gupper: upper statistical weight
        jlower: lower J (rotational quantum number, total angular momentum)
        jupper: upper J
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly)
        gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
        gamSta: log of gamma of Stark damping (s-1 / m-3)
        gamvdW:  log of (van der Waals damping constant / neutral hydrogen number) (s-1 / m-3)
    """
    ccgs = 29979245800.0  # c in cgs
    ecgs = 4.80320450e-10  # [esu]=[dyn^0.5*cm] #elementary charge
    mecgs = 9.10938356e-28  # [g] !electron mass
    with open(kuruczf) as f:
        lines = f.readlines()
    (
        wlnmair,
        loggf,
        species,
        elower,
        jlower,
        labellower,
        eupper,
        jupper,
        labelupper,
        gamRad,
        gamSta,
        gamvdW,
        ref,
        NLTElower,
        NLTEupper,
        isonum,
        hyperfrac,
        isonumdi,
        isofrac,
        hypershiftlower,
        hypershiftupper,
        hyperFlower,
        hypernotelower,
        hyperFupper,
        hypternoteupper,
        strenclass,
        auto,
        landeglower,
        landegupper,
        isoshiftmA,
    ) = (
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.array([""] * len(lines), dtype=object),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.array([""] * len(lines), dtype=object),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.array([""] * len(lines), dtype=object),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
    )
    ielem, iion = np.zeros(len(lines), dtype=int), np.zeros(len(lines), dtype=int)

    for i, line in enumerate(lines):
        wlnmair[i] = float(line[0:11])
        loggf[i] = float(line[11:18])
        species[i] = str(line[18:24])
        ielem[i] = int(species[i].split(".")[0])
        iion[i] = int(species[i].split(".")[1]) + 1
        elower[i] = float(line[24:36])
        jlower[i] = float(line[36:41])
        labellower[i] = str(line[42:52])
        eupper[i] = float(line[52:64])
        jupper[i] = float(line[64:69])
        labelupper[i] = str(line[70:80])
        gamRad[i] = float(line[80:86])
        gamSta[i] = float(line[86:92])
        gamvdW[i] = float(line[92:98])

    L_map = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6}

    multiplicity_lower = np.array([int(l[-2:-1]) for l in labellower])
    multiplicity_upper = np.array([int(l[-2:-1]) for l in labelupper])
    L_lower = np.array([L_map[l[-1]] for l in labellower])
    L_upper = np.array([L_map[l[-1]] for l in labelupper])

    not_flip = (eupper - elower) > 0
    elower_inverted = np.where(not_flip, elower, eupper)
    eupper_inverted = np.where(not_flip, eupper, elower)
    jlower_inverted = np.where(not_flip, jlower, jupper)
    jupper_inverted = np.where(not_flip, jupper, jlower)
    L_lower_inverted = np.where(not_flip, L_lower, L_upper)
    L_upper_inverted = np.where(not_flip, L_upper, L_lower)
    multiplicity_lower_inverted = np.where(not_flip, multiplicity_lower, multiplicity_upper)
    multiplicity_upper_inverted = np.where(not_flip, multiplicity_upper, multiplicity_lower)

    elower = elower_inverted
    eupper = eupper_inverted
    J_lower = jlower_inverted
    J_upper = jupper_inverted
    L_lower = L_lower_inverted
    L_upper = L_upper_inverted
    multiplicity_lower = multiplicity_lower_inverted
    multiplicity_upper = multiplicity_upper_inverted

    zeeman_alpha_list = []
    zeeman_strength_list = []
    zeeman_shift_list = []

    for l in range(len(lines)):
        Jl = Fraction.from_float(J_lower[i]).limit_denominator(2)
        Ju = Fraction.from_float(J_upper[i]).limit_denominator(2)
        Ll = int(L_lower[i])
        Lu = int(L_upper[i])
        Sl = Fraction(int(multiplicity_lower[i]) - 1, 2)
        Su = Fraction(int(multiplicity_upper[i]) - 1, 2)

        assert (Jl <= Ll + Sl) and (Ju <= Lu + Su), f"Cannot apply LS coupling to line {l}"

        # from lightweaver
        gLl = lande_factor(Jl, Ll, Sl)
        gLu = lande_factor(Ju, Lu, Su)
        alpha = []
        strength = []
        shift = []
        norm = np.zeros(3)

        for ml in fraction_range(-Jl, Jl+1):
            for mu in fraction_range(-Ju, Ju+1):
                if abs(ml - mu) <= 1.0:
                    alpha.append(int(ml - mu))
                    shift.append(gLl*ml - gLu*mu)
                    strength.append(zeeman_strength(Ju, mu, Jl, ml))
                    norm[alpha[-1]+1] += strength[-1]
        alpha = np.array(alpha, dtype=np.int32)
        strength = np.array(strength)
        shift = np.array(shift)
        strength /= norm[alpha + 1]

        zeeman_alpha_list.append(alpha)
        zeeman_strength_list.append(strength)
        zeeman_shift_list.append(shift)
    max_zeeman_components = max([z.shape[0] for z in zeeman_alpha_list])

    zeeman_alphas = np.zeros((len(lines), max_zeeman_components), dtype=np.int32)
    zeeman_strengths = np.zeros((len(lines), max_zeeman_components))
    zeeman_shifts = np.zeros((len(lines), max_zeeman_components))
    for i, (alpha, strength, shift) in enumerate(zip(zeeman_alpha_list, zeeman_strength_list, zeeman_shift_list)):
        length = zeeman_alpha_list[i].shape[0]
        zeeman_alphas[i, :length] = alpha
        zeeman_strengths[i, :length] = strength
        zeeman_shifts[i, :length] = shift

    wlaa = np.where(wlnmair < 200, wlnmair * 10, air_to_vac(wlnmair * 10))
    wl_vac = np.where(wlnmair < 200, wlnmair, air_to_vac(wlnmair * 10) * 0.1)
    nu_lines = 1e8 / wlaa[::-1]  # [cm-1]<-[AA]
    elower = (elower << u.Unit('cm-1')).to('eV', equivalencies=u.spectral()).value
    eupper = (eupper << u.Unit('cm-1')).to('eV', equivalencies=u.spectral()).value
    glower = jlower * 2 + 1
    gupper = jupper * 2 + 1
    A = (
        10**loggf
        / gupper
        * (ccgs * nu_lines) ** 2
        * (8 * np.pi**2 * ecgs**2)
        / (mecgs * ccgs**3)
    )
    gamSta = gamSta - 6.0
    gamvdW = gamvdW - 6.0

    return AtomicData(
        mass=jnp.array([lw.PeriodicTable[lw.Element(Z=z)].mass for z in ielem]),
        elem=jnp.array(ielem),
        stage=jnp.array(iion),
        abund=jnp.array([lw.DefaultAtomicAbundance[lw.Element(Z=z)] for z in ielem]),
        lambda0=jnp.array(wl_vac),
        log_grad=jnp.array(gamRad),
        log_gs=jnp.array(gamSta),
        log_gw=jnp.array(gamvdW),
        gi=jnp.array(glower),
        gj=jnp.array(gupper),
        ei=jnp.array(elower),
        ej=jnp.array(eupper),
        Aji=jnp.array(A),
        zeeman_alphas=jnp.array(zeeman_alphas),
        zeeman_strengths=jnp.array(zeeman_strengths),
        zeeman_shifts=jnp.array(zeeman_shifts)
    )

def thermal_vel(mass, temperature):
    r"""
    /** Compute mean thermal velocity
    * \param mass [u]
    * \param temperature [K]
    * \return mean thermal velocity [m/s]
    */
    """
    return jnp.sqrt(2.0 * temperature / mass * K_B_U)

def doppler_width(lambda0, mass, temperature, vturb):
    r"""
    /** Compute doppler width
    * \param lambda0 wavelength [nm]
    * \param mass [u]
    * \param temperature [K]
    * \param vturb microturbulent velocity [m / s]
    * \return Doppler width [nm]
    */
    """
    return lambda0 * jnp.sqrt(2.0 * K_B_U * temperature / mass + vturb**2) * INV_C

def damping_from_gamma(gamma, wave, dop_width):
    r"""
    /** Compute damping coefficient for Voigt profile from gamma
    * \param gamma [rad / s]
    * \param wave [nm]
    * \param dop_width [nm]
    * \return gamma / (4 pi dnu_D) = gamma / (4 pi dlambda_D) * wave**2
    */
    """
    # NOTE(cmo): Extra 1e-9 to convert c to nm / s (all lengths here in nm)
    return INV_FOURPI_C * gamma * 1e-9 * wave**2 / dop_width

def gamma_from_broadening(log_grad, log_gs, log_gw, temperature, ne, nhi):
    """
    Compute damping gamma from the line parameters and atmospheric parameters
    """
    grad = 10**log_grad
    gstark = 10**log_gs * ne
    gvdw = 10**log_gw * nhi

    gamma = grad + gstark + gvdw
    return gamma

def planck(wave, temperature):
    """
    Compute the Planck function (in kW/(nm m2 sr))

    wave : float
        Wavelength [nm]
    temperature : float
        Temperature [K]
    """
    return TWOHC2_NM5 / (wave**5 * (jnp.exp(HC_KB_NM / (wave * temperature)) - 1.0))

def emis_opac_line(
        mass,
        abund,
        lambda0,
        log_grad,
        log_gs,
        log_gw,
        gj,
        ej,
        Aji,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
):
    """
    Compute emissivity/opacity for a single LTE line
    """
    nhi, nhii = lte_h_ion_fracs(temperature, ne, nhtot)
    dop_width = doppler_width(lambda0, mass, temperature, vturb)
    gamma = gamma_from_broadening(log_grad, log_gs, log_gw, temperature, ne, nhi)
    adamp = damping_from_gamma(gamma, wave, dop_width)
    v = ((wave - lambda0) + (vel * lambda0) * INV_C) / dop_width
    p = voigt(adamp, v) / (SQRT_PI * dop_width)

    hnu_4pi = HC_FOURPI_KJ_NM / wave
    Uji = hnu_4pi * Aji * p
    Sfn = planck(wave, temperature)
    nj = fei_pop_i(abund, temperature, ne, nhtot, gj, ej)
    eta = nj * Uji
    chi = eta / Sfn
    return eta, chi

def polarised_line_compoment(
        alpha,
        strength,
        shift,
        adamp,
        v_scalar,
        v_b,
):
    # phi_sb, phi_pi, phi_sr = 0.0, 0.0, 0.0
    # psi_sb, psi_pi, psi_sr = 0.0, 0.0, 0.0
    components = jnp.zeros((2, 3))

    vh, vf = voigt_HF(adamp, v_scalar - shift * v_b)
    components = components.at[0, alpha+1].set(strength * vh)
    components = components.at[1, alpha+1].set(strength * vf)
    return components

def emis_opac_polarised_line(
        mass,
        abund,
        lambda0,
        log_grad,
        log_gs,
        log_gw,
        gj,
        ej,
        Aji,
        zeeman_alpha,
        zeeman_strength,
        zeeman_shift,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        cos_gamma,
        sin_2chi,
        cos_2chi,
):
    mag_width = DLAMBDA_B_CONST * lambda0**2 # [nm]

    nhi, nhii = lte_h_ion_fracs(temperature, ne, nhtot)
    dop_width = doppler_width(lambda0, mass, temperature, vturb)
    gamma = gamma_from_broadening(log_grad, log_gs, log_gw, temperature, ne, nhi)
    adamp = damping_from_gamma(gamma, wave, dop_width)
    v = ((wave - lambda0) + (vel * lambda0) * INV_C) / dop_width
    v_b = mag_width * b / dop_width

    sin2_gamma = 1.0 - cos_gamma**2
    voigt_norm = 1.0 / (SQRT_PI * dop_width)

    components = jax.vmap(
        polarised_line_compoment,
        in_axes=[0, 0, 0, None, None, None]
    )(
        zeeman_alpha,
        zeeman_strength,
        zeeman_shift,
        adamp,
        v,
        v_b
    ).sum(axis=0)

    phi_sigma = components[0, 0] + components[0, 2]
    phi_delta = 0.5 * components[0, 1] - 0.25 * phi_sigma
    phi = (phi_delta * sin2_gamma + 0.5 * phi_sigma) * voigt_norm

    phi_q = phi_delta * sin2_gamma * cos_2chi * voigt_norm
    phi_u = phi_delta * sin2_gamma * sin_2chi * voigt_norm
    phi_v = 0.5 * (components[0, 2] - components[0, 0]) * cos_gamma * voigt_norm

    psi_sigma = components[1, 0] + components[1, 2]
    psi_delta = 0.5 * components[1, 1] - 0.25 * psi_sigma

    psi_q = psi_delta * sin2_gamma * cos_2chi * voigt_norm
    psi_u = psi_delta * sin2_gamma * sin_2chi * voigt_norm
    psi_v = 0.5 * (components[1, 2] - components[1, 0]) * cos_gamma * voigt_norm

    Sfn = planck(wave, temperature)
    nj = fei_pop_i(abund, temperature, ne, nhtot, gj, ej)
    hnu_4pi = HC_FOURPI_KJ_NM / wave
    eta_no_prof = nj * hnu_4pi * Aji
    chi_no_prof = eta_no_prof / Sfn

    chi = jnp.array([
        chi_no_prof * phi,
        chi_no_prof * phi_q,
        chi_no_prof * phi_u,
        chi_no_prof * phi_v,
        chi_no_prof * psi_q,
        chi_no_prof * psi_u,
        chi_no_prof * psi_v,
    ])
    eta = jnp.array([
        eta_no_prof * phi,
        eta_no_prof * phi_q,
        eta_no_prof * phi_u,
        eta_no_prof * phi_v,
    ])

    return eta, chi


def emis_opac(adata: AtomicData, wave, temperature, ne, nhtot, vel, vturb):
    """
    Compute total emissivity/opacity for a atmospheric point

    adata : AtomicData
        The dataclass containing the atomic data
    wave : float
        The wavelength at which to compute [nm]
    temperature : float
        Temperature [K]
    ne : float
        Electron density [m-3]
    nhtot : float
        Total H density [m-3]
    vel : float
        LOS velocity [m/s]
    vturb : float
        microturbulent velocity [m/s]

    Returns
    -------
    Total emissivity, total opacity, from continuum and all lines.
    """
    chi_cont = continuum_opacity(wave, temperature, ne, nhtot)
    B_planck = planck(wave, temperature)
    eta_cont = chi_cont * B_planck

    axis_spec = (*[0]*9, *[None]*6)
    line_eta, line_chi = jax.vmap(emis_opac_line, in_axes=axis_spec)(
        adata.mass,
        adata.abund,
        adata.lambda0,
        adata.log_grad,
        adata.log_gs,
        adata.log_gw,
        adata.gj,
        adata.ej,
        adata.Aji,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb
    )

    eta = eta_cont + line_eta.sum()
    chi = chi_cont + line_chi.sum()
    return eta, chi

def emis_opac_polarised(
        adata: AtomicData,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        gamma_b,
        chi_b,
    ):
    """
    Compute total emissivity/opacity for a atmospheric point

    adata : AtomicData
        The dataclass containing the atomic data
    wave : float
        The wavelength at which to compute [nm]
    temperature : float
        Temperature [K]
    ne : float
        Electron density [m-3]
    nhtot : float
        Total H density [m-3]
    vel : float
        LOS velocity [m/s]
    vturb : float
        microturbulent velocity [m/s]
    b : float
        magnetic field [T]
    gamma_b : float
        inclination of magnetic field [radians]
    chi_b : float
        azimuth of magnetic field [radians]

    Returns
    -------
    Total emissivity, total opacity, from continuum and all lines. (Stokes form)
    [eps_I, eps_Q, eps_U, eps_V], [eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V]
    """
    chi_cont = continuum_opacity(wave, temperature, ne, nhtot)
    B_planck = planck(wave, temperature)
    eta_cont = chi_cont * B_planck

    # NOTE(cmo): This is only correct in the muz=1 case
    cos_gamma = jnp.cos(gamma_b)
    cos_2chi = jnp.cos(2.0 * chi_b)
    sin_2chi = jnp.sin(2.0 * chi_b)

    axis_spec = (*[0]*12, *[None]*10)
    line_eta, line_chi = jax.vmap(emis_opac_polarised_line, in_axes=axis_spec)(
        adata.mass,
        adata.abund,
        adata.lambda0,
        adata.log_grad,
        adata.log_gs,
        adata.log_gw,
        adata.gj,
        adata.ej,
        adata.Aji,
        adata.zeeman_alphas,
        adata.zeeman_strengths,
        adata.zeeman_shifts,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        cos_gamma,
        sin_2chi,
        cos_2chi,
    )
    eta = line_eta.sum(axis=0)
    chi = line_chi.sum(axis=0)

    eta = eta.at[0].set(eta[0] + eta_cont)
    chi = chi.at[0].set(chi[0] + chi_cont)

    return eta, chi


def Q_FeI(T):
    a = jnp.array([
        -1.15609527e3,
        7.46597652e2,
        -1.92865672e2,
        2.49658410e1,
        -1.61934455e0,
        4.21182087e-2
    ])
    lnQ = 0.0
    for i in range(a.shape[0]):
        lnQ = lnQ + a[i] * jnp.log(T)**i
    return jnp.exp(lnQ)

def Q_FeII(T):
    a = jnp.array([
        2.71692895e2,
        -1.52697440e2,
        3.36119665e1,
        -3.56415427e0,
        1.80193259e-1,
        -3.38654879e-3
    ])
    lnQ = 0.0
    for i in range(a.shape[0]):
        lnQ = lnQ + a[i] * jnp.log(T)**i
    return jnp.exp(lnQ)

def Q_FeIII(T):
    a = jnp.array([
        5.42788652e2,
        -3.26170152e2,
        7.77054463e1,
        -9.12244699e0,
        5.27184053e-1,
        -1.19689432e-2
    ])
    lnQ = 0.0
    for i in range(a.shape[0]):
        lnQ = lnQ + a[i] * jnp.log(T)**i
    return jnp.exp(lnQ)

def fe_pops(abund, temperature, ne, nhtot):
    """
    Compute the ion fractions of Fe I, II and III using the partition functions of Irwin 1981

    ionisation potentials from https://srd.nist.gov/jpcrdreprint/1.555659.pdf

    abund : float
        The decimal abundance of Fe relative to H
    temperature : float
        The temperature [K]
    ne : float
        The electron density [m-3]
    nhtot : float
        Total H density [m-3]
    """
    ionpot = (7.870, 16.1879) # eV
    saha = 2.0 * SAHA_CONST * temperature**1.5 / ne
    kBT = K_B_EV * temperature
    n1_n0 = Q_FeII(temperature) / Q_FeI(temperature) * saha * jnp.exp(-ionpot[0] / kBT)
    n2_n1 = Q_FeIII(temperature) / Q_FeII(temperature) * saha * jnp.exp(-ionpot[1] / kBT)

    nfei = (abund * nhtot) / (1.0 + n1_n0 + n2_n1)
    nfeii = n1_n0 * nfei
    nfeiii = n2_n1 * nfeii

    return nfei, nfeii, nfeiii

def fei_pop_i(abund, temperature, ne, nhtot, gi, ei):
    """
    Compute the population of a level i of Fe I with statistical weight gi and energy ei

    abund : float
        The decimal abundance of Fe relative to H
    temperature : float
        The temperature [K]
    ne : float
        The electron density [m-3]
    nhtot : float
        Total H density [m-3]
    gi : float
        Statistical weight g
    ei : float
        Energy level in [eV]
    """
    nfei, nfeii, nfeiii = fe_pops(abund, temperature, ne, nhtot)
    kBT = K_B_EV * temperature
    ni = nfei * gi / Q_FeI(temperature) * jnp.exp(-ei / kBT)
    return ni



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()

    # plt.figure()
    kd = read_kurucz("kurucz_6301_6302.linelist")

    # wave = np.linspace(50, 1000.0, 100)
    # b = planck(wave, 5000)
    # import lightweaver as lw
    # b_lw = (lw.planck(5000, wave) << u.Unit('W/(m2 Hz sr)')).to('kW/(m2 nm sr)', equivalencies=u.spectral_density(wav=wave << u.nm)).value
    # plt.figure()
    # plt.plot(wave, b)
    # plt.plot(wave, b_lw, '--')

    eta_s, chi_s = emis_opac(kd, 630.324, 5000, 1e22, 1e25, 0.0, 2e3)

    eta, chi = emis_opac_polarised(kd, 630.31, 5000, 1e22, 1e25, 0.0, 2e3, 0.1, 0.0, 0.0)
