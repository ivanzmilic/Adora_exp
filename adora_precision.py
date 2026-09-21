# Single source of truth for float32 / float64 across Adora.
#
# jax_enable_x64 is a PROCESS-GLOBAL flag that must be set before any jax array is
# created, so every module that used to hardcode it now does `import adora_precision`
# instead -- this reads the choice once, from the environment, and applies it.
#
# Switch with the env var (set BEFORE launching python):
#     ADORA_X64=1  -> float64 (default; the validated mode)
#     ADORA_X64=0  -> float32 (~2x less memory; validate accuracy per use)
# Or flip the "1" default below.
import os
import jax

X64 = os.environ.get("ADORA_X64", "1").lower() not in ("0", "false", "no")
jax.config.update("jax_enable_x64", X64)

import jax.numpy as jnp
DTYPE = jnp.float64 if X64 else jnp.float32
