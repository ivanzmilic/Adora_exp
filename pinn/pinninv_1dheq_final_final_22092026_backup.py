# A bit meme name to troll people

# Clone of the notebook inversion #7:

import os, sys
sys.path.insert(0, os.path.abspath('.')); 
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../3dtesting')); 
sys.path.insert(0, os.path.abspath('../eos_mlp'))
import gpu; 
dev = gpu.use_gpu()
import time, numpy as np, matplotlib.pyplot as plt, jax
import matplotlib
matplotlib.use('Agg')
import atmosphere_field as af, rt_forward as rf, invert, pretrain as pt, eos as eos_mod
from synth3d import read_kurucz, DEFAULT_LINELIST

print('CUDA_VISIBLE_DEVICES =', dev, '| jax backend:', jax.default_backend())
print("info::check if this is correct GPU and interrupt if necessary!")

# HARD SET STUFF:
P_TOP = 3.75

def misfit(I):
    return {s: float(np.sqrt(np.mean((I[:,:,k]-obs[:,:,k])**2))/mc) for k, s in enumerate('IQUV')}

def show_hist(ck):
    h = ck['hist']; 
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), layout='constrained')
    for kk in ['fit','he','bnd','total']: 
        ax[0].semilogy(h['step'], h[kk], label=kk)
    ax[0].set(xlabel='step', ylabel='loss (monitor set)', title=f"losses ({ck['step']} steps)"); 
    ax[0].legend(fontsize=8); 
    ax[0].grid(alpha=.3)
    ax[1].semilogy(h['step'], h['lr']); 
    ax[1].set(xlabel='step', ylabel='learning rate', title='LR schedule (exp decay)'); 
    ax[1].grid(alpha=.3)
    plt.savefig(f"checkpoint_{ck['step']}.png")

# Main:
if __name__ == "__main__":
    pretrain = bool(int(sys.argv[1]))

    # Quick config:
    n_freq = 128
    width = 96
    depth = 4
    sigmas = (2.0, 2.0, 2.0)

    folder_to_save = '/dat/milic/adora_pinn_develop/' + str(n_freq)+'_' + str(width) + 'x' + str(depth) + '_' + '_'.join(map(str, sigmas))+'/'
    print("info::we will be saving things to: ", folder_to_save)
    os.makedirs(folder_to_save, exist_ok=True)

    if pretrain:
        print("info::imma gonna pretrain your network real quick")
        t0 = time.time()
        coords, tgt7, shape, z = pt.load_target()
        params0 = af.init_params(jax.random.PRNGKey(0),  sigmas=sigmas, n_freq=n_freq, width=width, depth=depth)
        params, hist = pt.train(params0, coords, tgt7, steps=6000, batch=8192, lr=3e-3)
        print(f'trained 6000 steps in {time.time()-t0:.0f}s   final loss {hist[-1,1]:.3e}')
        out_w = folder_to_save + 'pretrained_field.npz'
        af.save_params(out_w, params); print('saved', out_w)

    # Either use old one or the one you just pretrained:
    params0 = af.load_params(folder_to_save + 'pretrained_field.npz')  # warm start
    eos_params = eos_mod.load_eos('pg')
    d = np.load('/dat/milic/adora_pinn_develop/inversion_testset.npz', allow_pickle=True)
    obs = np.asarray(d['I_obs']); 
    waves = np.asarray(d['waves'])
    adata = read_kurucz(DEFAULT_LINELIST)

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Prep obs, HARD-CODED for now
    nx, ny, nz = 128, 128, 56; 
    dz = np.full(nz, 20.0e3); 
    coords = af.grid_coords(nx, ny, nz); 
    mc = float(obs[..., 0, 0].mean()) # Mean continuum intensity
    print('info::obs have this shape:', obs.shape)

    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # Train 

    fresh = bool(int(sys.argv[2]))

    n_iter = int(sys.argv[3])

    if fresh:
        t0 = time.time()
        ckpt = invert.run_inversion(params0, eos_params, obs, waves, dz, adata, P_TOP,
                            steps=n_iter, m_fit=1024, k_he=4096, m_bnd=1024,
                            w_fit=1.0, w_he=20.0, w_bnd=1.0e2,
                            lr=3.0e-3, lr_decay=0.95, lr_transition=1000, nz=nz)
        print(f'{ckpt["step"]} steps in {time.time()-t0:.0f}s')
        show_hist(ckpt)

        import pickle

        ckpt_path = folder_to_save + 'inversion_ckpt_step.pkl'.format(ckpt['step'])

        with open(ckpt_path, 'wb') as f:
            pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'info::saved full checkpoint -> {ckpt_path}')

    else:

        # Load saved checkpoint
        import pickle
        ckpt_path = folder_to_save + 'inversion_ckpt_step10000.pkl'
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        print(f'info::loaded checkpoint -> {ckpt_path}')

        
        ckpt = invert.run_inversion(ckpt, eos_params, obs, waves, dz, adata, P_TOP,
                            steps=n_iter, m_fit=1024, k_he=4096, m_bnd=1024,
                            w_fit=1.0, w_he=20.0, w_bnd=1.0e2,
                            lr=1.0e-3, lr_decay=0.99, lr_transition=1000, nz=nz)
        print(f'total steps: {ckpt["step"]}')
        show_hist(ckpt)

    # Save:
    
    af.save_params(folder_to_save + 'inverted_field.npz', ckpt['params'])
    print('info::saved inverted field ->', folder_to_save + 'inverted_field.npz', ' (', ckpt['step'], 'steps )')

    # ---------------------------------------------------------------------------------------------------------------
    # Print some debugging things:
    params = ckpt['params']
    I1 = np.asarray(rf.synth_field(params, coords, (nx,ny,nz), waves, dz, batch=4096, progress=False))

    # 4x6 panel: rows = [I_obs, I_fit, V_obs, V_fit], cols = selected wavelengths in `idxs`
    idxs = np.linspace(30,60,6)
    sel = np.array(idxs, dtype=int)
    print(sel)

    I_obs = obs[:, :, 0, sel]
    V_obs = obs[:, :, 3, sel]
    I_fit = I1[:, :, 0, sel]
    V_fit = I1[:, :, 3, sel]

    # consistent scales
    vmin_I = min(I_obs.min(), I_fit.min())
    vmax_I = max(I_obs.max(), I_fit.max())
    vmax_V = max(np.abs(V_obs).max(), np.abs(V_fit).max())

    fig, ax = plt.subplots(4, 6, figsize=(16, 10), layout='constrained')

    for j, iw in enumerate(sel):
        ax[0, j].imshow(I_obs[:, :, j].T, origin='lower', cmap='gray', vmin=vmin_I, vmax=vmax_I)
        ax[1, j].imshow(I_fit[:, :, j].T, origin='lower', cmap='gray', vmin=vmin_I, vmax=vmax_I)
        ax[2, j].imshow(V_obs[:, :, j].T, origin='lower', cmap='RdBu_r', vmin=-vmax_V, vmax=vmax_V)
        ax[3, j].imshow(V_fit[:, :, j].T, origin='lower', cmap='RdBu_r', vmin=-vmax_V, vmax=vmax_V)

        ax[0, j].set_title(f"{waves[iw]:.3f} nm", fontsize=10)

    for r in range(4):
        for c in range(6):
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])

    ax[0, 0].set_ylabel("I obs")
    ax[1, 0].set_ylabel("I fit")
    ax[2, 0].set_ylabel("V obs")
    ax[3, 0].set_ylabel("V fit")

    # colorbars (one for I, one for V)
    imI = ax[0, 0].images[0]
    imV = ax[2, 0].images[0]
    fig.colorbar(imI, ax=ax[0:2, :], fraction=0.02, pad=0.01, label='Stokes I')
    fig.colorbar(imV, ax=ax[2:4, :], fraction=0.02, pad=0.01, label='Stokes V')

    plt.suptitle("Observed vs Inverted fits in Stokes I and V at selected wavelengths", y=1.02)
    plt.savefig(folder_to_save + "observed_vs_inverted_fits.png")


    # ------------------------------------------------------------------------------------------------
    # Now some atmosphere plots
    # 4x6 panel: rows = physical parameters, cols = selected depths in `izs`

    pred = pt.predict_grid(params, coords); 

    sel_iz = np.array([5,10,15,20,25,30])
    z_km = np.arange(nz)*20.0 - 100.0


    params4 = [
        ('T', 'T [K]', 1.0, 'inferno', False),
        ('vz', 'vz [km/s]', 1e-3, 'bwr', True),
        ('Bz', 'Bz [mT]', 1e3, 'PuOr', True),
        ('vturb', 'vturb [km/s]', 1e-3, 'magma', False),
    ]

    fig, ax = plt.subplots(4, 6, figsize=(16, 10), layout='constrained')

    for r, (key, label, scale, cmap, sym) in enumerate(params4):

        cube = pred[key].reshape(nx, ny, nz) * scale
        stack_r = cube[:, :, sel_iz]

        for c, iz in enumerate(sel_iz):

            ## Do 1st and 99th percentiles for color scaling
            stack_perc = np.percentile(stack_r, [1, 99], axis=(0, 1))
            #print (stack_perc.shape)
            if sym:
                vmax_r = np.max(np.abs(stack_perc[1, c]))
                vmin_r = -vmax_r
            else:
                vmin_r = stack_perc[0, c]
                vmax_r = stack_perc[1, c]
            
            imr = ax[r, c].imshow(
                cube[:, :, iz].T, origin='lower', cmap=cmap, vmin=vmin_r, vmax=vmax_r
            )
            if r == 0:
                ax[r, c].set_title(f"z = {z_km[iz]:.0f} km", fontsize=10)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])

        ax[r, 0].set_ylabel(label)
        fig.colorbar(imr, ax=ax[r, :], fraction=0.02, pad=0.01)

    plt.suptitle("Recovered physical parameters at selected depths", y=1.02)
    plt.savefig(folder_to_save + "depth_maps_debug.png",bbox_inches='tight')