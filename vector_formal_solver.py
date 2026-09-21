import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
from jax.lax import fori_loop

def stokes_K(opac):
    eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V = opac
    # NOTE(cmo): Propagation matrix
    K = jnp.array([
        [eta_I, eta_Q, eta_U, eta_V],
        [eta_Q, eta_I, rho_V, -rho_U],
        [eta_U, -rho_V, eta_I, rho_Q],
        [eta_V, rho_U, -rho_Q, eta_I]
    ])
    return K


def delo_constant_fs(dz, I_start, emis, opac):
    def body(i, intens):
        Id = jnp.eye(4)

        # if i == 0:
        #     Km = stokes_K(opac[i])
        #     eta_m = emis[i]
        #     dtau = opac[i, 0] * dz[i]
        # else:
        #     Km = stokes_K(opac[i - 1])
        #     eta_m = emis[i - 1]
        #     dtau = 0.5 * (opac[i, 0] + opac[i - 1, 0]) * dz[i]
        Km = stokes_K(opac[i - 1])
        eta_m = emis[i - 1] / Km[0, 0]
        dtau = 0.5 * (opac[i, 0] + opac[i-1, 0]) * dz[i]


        K = stokes_K(opac[i])
        eta = emis[i] / K[0, 0]

        # DELO modified
        K_prime = K / K[0, 0] - Id
        Km_prime = Km / Km[0, 0] - Id

        # Janett 2017 appendix
        E_k = jnp.exp(-dtau)
        F_k = -jnp.expm1(-dtau)
        Phi_k = E_k * Id - 0.5 * F_k * Km_prime
        Phi_kp = Id + 0.5 * F_k * K_prime
        Psi_k = 0.5 * F_k * eta_m
        Psi_kp = 0.5 * F_k * eta

        intens = jnp.linalg.solve(Phi_kp, Phi_k @ intens + Psi_k + Psi_kp)
        return intens


    # NOTE(cmo): Loop from a starting index of 1 assuming I_start at lower boundary
    intens = fori_loop(
        1,
        dz.shape[0],
        body,
        I_start
    )
    return intens

if __name__ == "__main__":
    # Phi_{k+1} I_{k+1} = Phi_k I_k + Psi_{k+1} + Psi_k
    opac_grid = jnp.array([
        [1.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
        [1.1, 0.1, 0.1, 0.1, 0.04, 0.04, 0.04],
        [1.2, 0.1, 0.1, 0.1, 0.03, 0.03, 0.03],
        [1.3, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02],
        [1.4, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    ])

    epsilon_grid = jnp.array([
        [0.5, 0.05, 0.05, 0.05],
        [0.6, 0.06, 0.06, 0.06],
        [0.7, 0.07, 0.07, 0.07],
        [0.8, 0.08, 0.08, 0.08],
        [0.9, 0.09, 0.09, 0.09]
    ])

    dz = jnp.ones(5) * 0.5
    I_start = jnp.array([1.0, 0.0, 0.0, 0.0])

    I_final = delo_constant_fs(dz, I_start, epsilon_grid, opac_grid)