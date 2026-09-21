'''
Simple python script that collects things we have done so far and calculates spectrum from a MURaM cube. 
Based on things from muram_loader.py, and synth3d.py / synth3d_mp.py
'''

import os, sys
import numpy as np

# Take care of the paths. This could break in future
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS_DIR)

sys.path.insert(0, REPO)          # muram.py lives in the Adora repo root
import muram as mio

import muram_loader as mload

import time
import numpy as np
import matplotlib
matplotlib.use("Agg") # On the server if we want to plot we can't open the device
import matplotlib.pyplot as plt

import lightweaver as lw # Might still need -> Check

from synth3d import synth_cube_map, read_kurucz
from synth3d_mp import synth_cube_mp

# ---------------------------------------------------------------------------------

# Actual script:

datapath, iteration = sys.argv[1], int(sys.argv[2])
output = sys.argv[3]
kind = sys.argv[4] if len(sys.argv) > 4 else "muram"

# LOADING THE MURaM CUBE:
# -----------------------------------------------------------------------------------------------------------------------
# Load the cube. Remember that eos_input = None is for nh,ne; pg is for pgas, rho is for rho, and both adds both of them. 

cube, dz, meta = mload.load_muram_cube(datapath, iteration, kind=kind, nx=[0,1536], ny=[0,1536], skip=1, eos_input="pg")  


# Simple debug plot of the cube properties
print("meta:", meta)
for k in cube:                               # print every key present (incl. eos Pg/rho)
    a = cube[k]
    print(f"  {k:5s} shape={a.shape} min={a.min():.3e} max={a.max():.3e}")


# ----------------------------------------------------------------------------------------------------------------------

# RADIATIVE STUFF, LINES, WAVELENGTHS ETC:
adata = read_kurucz(os.path.join(REPO, "kurucz_6301_6302.linelist"))
    
waves = np.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)


# ----------------------------------------------------------------------------------------------------------------------
# SYNTHESIS:


I_map = synth_cube_map(waves, dz, cube, eos_mode="pg", forward_only=True, batch=1024)
print(f"GPU full-map synth {I_map.shape}")

# Save the GPU full-map synthesis result to a file
np.save(output, I_map)