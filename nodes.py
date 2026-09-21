import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
from interpax import interp1d
import jax_dataclasses as jdc
from lineop import AtomicData, read_kurucz, emis_opac_polarised, planck
from vector_response_fn import lte_polarised_rt
import lightweaver as lw
from scipy.optimize import least_squares

@jdc.pytree_dataclass
class NodeSpec:
    z_min: jdc.Static[float]
    z_max: jdc.Static[float]
    n_temperature: jdc.Static[int]
    n_ne: jdc.Static[int]
    n_nhtot: jdc.Static[int]
    n_vz: jdc.Static[int]
    n_vturb: jdc.Static[int]
    n_b: jdc.Static[int]
    n_gamma_b: jdc.Static[int]
    n_chi_b: jdc.Static[int]
    n_interp: jdc.Static[int]

    @property
    def n_nodes(self):
        return (
            self.n_temperature +
            self.n_ne +
            self.n_nhtot +
            self.n_vz +
            self.n_vturb +
            self.n_b +
            self.n_gamma_b +
            self.n_chi_b
        )

    def unpack_nodes(self, nodes):
        running = 0
        temperature = nodes[running:running+self.n_temperature]
        running = running + self.n_temperature
        ne = nodes[running:running+self.n_ne]
        running = running + self.n_ne
        nhtot = nodes[running:running+self.n_nhtot]
        running = running + self.n_nhtot
        vz = nodes[running:running+self.n_vz]
        running = running + self.n_vz
        vturb = nodes[running:running+self.n_vturb]
        running = running + self.n_vturb
        b = nodes[running:running+self.n_b]
        running = running + self.n_b
        gamma_b = nodes[running:running+self.n_gamma_b]
        running = running + self.n_gamma_b
        chi_b = nodes[running:running+self.n_chi_b]
        running = running + self.n_chi_b

        return (
            temperature,
            ne,
            nhtot,
            vz,
            vturb,
            b,
            gamma_b,
            chi_b
        )

    def reconstruct_atmos(
            self,
            temperature,
            ne,
            nhtot,
            vz,
            vturb,
            b,
            gamma_b,
            chi_b,
        ):

        grid = jnp.linspace(self.z_min, self.z_max, self.n_interp)
        def reconstruct_param(p, in_log=False):
            if p.shape[0] == 1:
                return jnp.ones(self.n_interp) * p[0]
            else:
                if in_log:
                    p = jnp.log10(p)
                result = interp1d(grid, jnp.linspace(self.z_min, self.z_max, p.shape[0]), p)
                if in_log:
                    result = 10**result
                return result

        temperature_full = reconstruct_param(temperature)
        ne_full = reconstruct_param(ne, in_log=True)
        nhtot_full = reconstruct_param(nhtot, in_log=True)
        vz_full = reconstruct_param(vz)
        vturb_full = reconstruct_param(vturb)
        b_full = reconstruct_param(b)
        gamma_b_full = reconstruct_param(gamma_b)
        chi_b_full = reconstruct_param(chi_b)
        return (
            jnp.ones(self.n_interp) * (grid[1] - grid[0]), # dz
            temperature_full,
            ne_full,
            nhtot_full,
            vz_full,
            vturb_full,
            b_full,
            gamma_b_full,
            chi_b_full,
        )

    def reconstruct_from_nodes(self, nodes):
        return self.reconstruct_atmos(*self.unpack_nodes(nodes))

    def interp_to_nodes(self, z, temperature, ne, nhtot, vz, vturb, b=None, gamma_b=None, chi_b=None):
        z_mask = (z >= self.z_min) & (z <= self.z_max)
        def interp_to_nodes(p, n_nodes):
            if n_nodes == 1:
                return jnp.array([jnp.mean(p[z_mask])])
            else:
                grid = jnp.linspace(self.z_min, self.z_max, n_nodes)
                return jnp.interp(grid, z, p)

        temperature_nodes = interp_to_nodes(temperature, self.n_temperature)
        ne_nodes = interp_to_nodes(ne, self.n_ne)
        nhtot_nodes = interp_to_nodes(nhtot, self.n_nhtot)
        vz_nodes = interp_to_nodes(vz, self.n_vz)
        vturb_nodes = interp_to_nodes(vturb, self.n_vturb)
        if b is None:
            b_nodes = jnp.zeros(self.n_b)
        else:
            b_nodes = interp_to_nodes(b, self.n_b)
        if gamma_b is None:
            gamma_b_nodes = jnp.zeros(self.n_gamma_b)
        else:
            gamma_b_nodes = interp_to_nodes(gamma_b, self.n_gamma_b)
        if chi_b is None:
            chi_b_nodes = jnp.zeros(self.n_chi_b)
        else:
            chi_b_nodes = interp_to_nodes(chi_b, self.n_chi_b)

        return (
            temperature_nodes,
            ne_nodes,
            nhtot_nodes,
            vz_nodes,
            vturb_nodes,
            b_nodes,
            gamma_b_nodes,
            chi_b_nodes
        )

    def pack_nodes(self, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b):
        return jnp.concatenate([
            temperature,
            ne,
            nhtot,
            vz,
            vturb,
            b,
            gamma_b,
            chi_b
        ])

    def interpolate_and_pack_nodes(self, z, temperature, ne, nhtot, vz, vturb, b=None, gamma_b=None, chi_b=None):
        return self.pack_nodes(*self.interp_to_nodes(z, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b))

def compute_residual_from_nodes(nodes, target, wave, lines, node_desc: NodeSpec):
    atmos = node_desc.reconstruct_from_nodes(nodes)

    model = lte_polarised_rt(lines, wave, *atmos)
    return model - target

compute_residual_vmap = jax.jit(
    jax.vmap(
        compute_residual_from_nodes,
        in_axes=[None, 1, 0, None, None],
        out_axes=1,
    )
)

compute_residual_jac_vmap = jax.jit(
    jax.vmap(
        jax.jacrev(
            compute_residual_from_nodes,
            argnums=0
        ),
        in_axes=[None, 1, 0, None, None],
        out_axes=1,
    )
)

if __name__ == "__main__":
    from lightweaver.fal import Falc82
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()

    fal = Falc82()

    z_start = fal.z.min()
    z_end = 1.6e6

    nodes = NodeSpec(
        z_min = fal.z.min(),
        z_max = 1.6e6,
        n_temperature=3,
        n_ne=3,
        n_nhtot=3,
        n_vz=2,
        n_vturb=1,
        n_b=1,
        n_gamma_b=1,
        n_chi_b=1,
        n_interp=30
    )

    node_sets = nodes.interp_to_nodes(
        fal.z[::-1],
        fal.temperature[::-1],
        fal.ne[::-1],
        fal.nHTot[::-1],
        fal.vz[::-1],
        fal.vturb[::-1],
    )

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
    vz = jnp.linspace(1.0, 0.0, temperature.shape[0]) * 8e3
    b = jnp.ones(temperature.shape[0]) * 0.02
    gamma_b = jnp.ones(temperature.shape[0]) * 0.7
    chi_b = jnp.zeros(temperature.shape[0])

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    lte_rt_wave = jax.jit(
        jax.vmap(
            lte_polarised_rt,
            in_axes=[None, 0, None, None, None, None, None, None, None, None, None],
            out_axes=1,
        )
    )
    lines = read_kurucz("kurucz_6301_6302.linelist")
    intens_ref = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b)

    starting_nodes = nodes.pack_nodes(*node_sets)
    recon_from_nodes = nodes.reconstruct_from_nodes(starting_nodes)

    intens_nodes = lte_rt_wave(lines, waves, *recon_from_nodes)

    plt.figure()
    plt.plot(waves, intens_ref[0] / intens_ref[0, 0], label='falc')
    plt.plot(waves, intens_ref[3] / intens_ref[0, 0], label='falc V')
    plt.plot(waves, intens_nodes[0] / intens_nodes[0, 0], label='start')

    min_params = nodes.pack_nodes(
        temperature=jnp.ones(nodes.n_temperature) * 2500.0,
        ne=jnp.ones(nodes.n_ne) * 1e16,
        nhtot=jnp.ones(nodes.n_nhtot) * 1e16,
        vz=jnp.ones(nodes.n_vz) * -20e3,
        vturb=jnp.zeros(nodes.n_vturb),
        b=jnp.zeros(nodes.n_b),
        gamma_b=jnp.zeros(nodes.n_gamma_b),
        chi_b=jnp.zeros(nodes.n_chi_b),
    )
    max_params = nodes.pack_nodes(
        temperature=jnp.ones(nodes.n_temperature) * 20e3,
        ne=jnp.ones(nodes.n_ne) * 1e22,
        nhtot=jnp.ones(nodes.n_nhtot) * 1e24,
        vz=jnp.ones(nodes.n_vz) * 20e3,
        vturb=jnp.ones(nodes.n_vturb) * 10e3,
        b=jnp.ones(nodes.n_b),
        gamma_b=jnp.ones(nodes.n_gamma_b) * jnp.pi,
        chi_b=jnp.ones(nodes.n_chi_b) * 2.0 * jnp.pi,
    )

    result = least_squares(
        jax.jit(lambda p: compute_residual_vmap(p, intens_ref, waves, lines, nodes).reshape(-1)),
        x0=starting_nodes,
        jac=jax.jit(lambda p: compute_residual_jac_vmap(p, intens_ref, waves, lines, nodes).reshape(-1, nodes.n_nodes)),
        method='trf',
        bounds=(min_params, max_params),
        verbose=2,
        ftol=1e-3,
        xtol=1e-7,
        x_scale='jac',
    )

    intens_fitted = lte_rt_wave(lines, waves, *nodes.reconstruct_from_nodes(result.x))

    plt.plot(waves, intens_fitted[0] / intens_fitted[0, 0], '--', label='fit')
    plt.plot(waves, intens_fitted[3] / intens_fitted[0, 0], '--', label='fit V')
    plt.legend()