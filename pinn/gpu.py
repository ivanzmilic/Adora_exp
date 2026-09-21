"""
gpu.py -- polite GPU selection for the PINN notebooks/scripts.

jax-free on purpose: call use_gpu() FIRST, before importing jax or anything that imports it
(adora_precision, atmosphere_field, rt_forward), because JAX reads these env vars only at import.

    import gpu; gpu.use_gpu()      # picks the idlest visible GPU, no pre-allocation
    import atmosphere_field, rt_forward, ...
"""
import os
import subprocess


def use_gpu(device=None, preallocate=False):
    """Point JAX at a GPU. device=None picks the GPU with the most free memory (nvidia-smi);
    preallocate=False stops XLA grabbing ~75% of the card up front (polite on the shared node).
    Returns the chosen CUDA index as a string. Falls back to '0' if nvidia-smi is unavailable."""
    os.environ.pop("JAX_PLATFORMS", None)                       # undo any CPU forcing
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"
    if device is None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"], text=True)
            rows = [r.split(",") for r in out.strip().splitlines() if r.strip()]
            device = max(rows, key=lambda r: int(r[1]))[0].strip()   # most free memory
        except Exception:
            device = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    return str(device)
