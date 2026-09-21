import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (adora_precision)
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


def delo_linear_fs(dz, I_start, emis, opac):
    def body(i, intens):
        Id = jnp.eye(4)

        Km = stokes_K(opac[i - 1])
        eta_m = emis[i - 1] / Km[0, 0]
        dtau = 0.5 * (opac[i, 0] + opac[i-1, 0]) * dz[i]


        K = stokes_K(opac[i])
        eta = emis[i] / K[0, 0]

        # DELO modified
        K_prime = K / K[0, 0] - Id
        Km_prime = Km / Km[0, 0] - Id

        # PHYSICS: linear source across the layer (Olson & Kunasz 1987) in place of the constant
        # 0.5*F_k trapezoid. psi_a weights the upwind point, psi_b the local one; both reduce to a
        # constant source (sum = 1 - E_k) but redistribute correctly as dtau grows.
        E_k = jnp.exp(-dtau)
        w0 = -jnp.expm1(-dtau)          # 1 - E_k          (= F_k)
        w1 = w0 - dtau * E_k            # 1 - (1 + dtau) E_k

        # w1/dtau is 0/0 as dtau -> 0: the value is finite but the gradient is NaN. Taylor-expand
        # below a threshold, with the double-where so the discarded branch never divides by zero.
        eps = 1e-3
        safe = jnp.where(dtau < eps, 1.0, dtau)
        psi_a = jnp.where(dtau < eps, 0.5*dtau - dtau**2/3.0 + dtau**3/8.0,  w1 / safe)
        psi_b = jnp.where(dtau < eps, 0.5*dtau - dtau**2/6.0 + dtau**3/24.0, w0 - w1 / safe)

        # Janett 2017 appendix (source/coupling weighted by the linear psi_a, psi_b)
        Phi_k  = E_k * Id - psi_a * Km_prime
        Phi_kp = Id + psi_b * K_prime
        Psi_k  = psi_a * eta_m
        Psi_kp = psi_b * eta

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
    # Phi_{k+1} I_{k+1} = Phi_k I_k + Psi_{k+1} + Psi_k   (linear source)
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

    I_final = delo_linear_fs(dz, I_start, epsilon_grid, opac_grid)
    print(I_final)
