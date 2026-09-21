import jax
import adora_precision   # global float32/64 switch (ADORA_X64; default 64-bit)
import jax.numpy as jnp
from lineop import AtomicData, read_kurucz
from response_fn import lte_rt
import optax
import time

def pack_params(temperature, ne, nhtot, vz, vturb):
    return np.stack([
        temperature,
        ne,
        nhtot,
        vz,
        vturb,
    ]).reshape(-1)

def params_to_tuple(params):
    p = params.reshape(5, -1)
    return (
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
    )

lte_rt_jit = jax.jit(
        lte_rt,
)

lte_rt_wave = jax.vmap(
    lte_rt_jit,
    in_axes=[None, 0, None, None, None, None, None, None]
)

# def compute_residuals(params, target, lines, wavelengths, dz):
#     params = params.reshape(5, -1)
#     temperature = params[0]
#     ne = params[1]
#     nhtot = params[2]
#     vz = params[3]
#     vturb = params[4]

#     lte_rt_wave = jax.vmap(
#         lte_rt_jit,
#         in_axes=[None, 0, None, None, None, None, None, None]
#     )

#     model = lte_rt_wave(lines, wavelengths, dz, temperature, ne, nhtot, vz, vturb)
#     return model - target

# def vmap_residuals(params, target, lines, wavelength, dz):
#     params = params.reshape(5, -1)
#     temperature = params[0]
#     ne = params[1]
#     nhtot = params[2]
#     vz = params[3]
#     vturb = params[4]

#     model = lte_rt_jit(lines, wavelength, dz, temperature, ne, nhtot, vz, vturb)
#     return model - target


# compute_residuals_vmap = jax.jit(
#     jax.vmap(
#         vmap_residuals,
#         in_axes=[None, 0, None, 0, None]
#     )
# )

# # NOTE(cmo): Swapping the order of the jacobian application appears to be much
# # faster as it prevents "false sharing" over wavelengths
# compute_residuals_jac_vmap = jax.jit(
#     jax.vmap(
#         jax.jacrev(
#             vmap_residuals,
#             argnums=0
#         ),
#         in_axes=[None, 0, None, 0, None]
#     )
# )


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

    temperature_target = temperature.copy()
    temperature_target = temperature_target + jnp.sin(jnp.linspace(0.0, 6.0 * jnp.pi, temperature.shape[0])) * 400
    ne_target = jnp.where(
        jnp.arange(ne.shape[0]) < 30,
        ne * 1.04,
        ne
    )
    nhtot_target = nhtot.copy()
    vturb_target = vturb.copy()
    vz_target = jnp.sin(jnp.linspace(jnp.pi / 4, 3.0 * jnp.pi, temperature.shape[0])) * 5e3

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    start = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb)
    target = lte_rt_wave(lines, waves, dz, temperature_target, ne_target, nhtot_target, vz_target, vturb_target)

    initial_params = pack_params(
        temperature=temperature,
        ne=ne,
        nhtot=nhtot,
        vz=vz,
        vturb=vturb
    )

    plt.figure()
    plt.plot(waves, start)
    plt.plot(waves, target)


    @jax.jit
    def loss(params, target):
        params = params.reshape(5, -1)
        temperature = params[0]
        ne = params[1]
        nhtot = params[2]
        vz = params[3]
        vturb = params[4]
        model = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb)
        return jnp.mean((model - target)**2)

    loss_grad_fn = jax.value_and_grad(loss)

    # https://optax.readthedocs.io/en/latest/_collections/examples/flax_example.html
    # learning_rate = 1
    # tx = optax.chain(
    #     optax.scale_by_adam(b1=0.9, b2=0.99, eps=1e-12),
    #     # optax.scale_by_lbfgs(), # weird issues
    #     optax.scale(-learning_rate)
    # )
    learning_rate = 20
    tx = optax.chain(
        optax.optimistic_adam(learning_rate=learning_rate),
        optax.contrib.reduce_on_plateau(
            patience=3,
            cooldown=10,
            rtol=5e-2,
            factor=0.5,
            accumulation_size=3,
        )
    )
    params = initial_params.copy()
    opt_state = tx.init(params)

    start = time.time()
    loss_history = []
    num_steps = 500
    for _ in range(num_steps):
        loss_val, grads = loss_grad_fn(params, target)
        loss_history.append(loss_val)

        updates, opt_state = tx.update(grads, opt_state, value=loss_val)
        params = optax.apply_updates(params, updates)
    stop = time.time()

    print(f"{num_steps} iterations in {stop - start:.2f} s")

    plt.plot(waves, lte_rt_wave(lines, waves, dz, *params_to_tuple(params)), '--')

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.xlabel("MSE")
    plt.yscale("log")

