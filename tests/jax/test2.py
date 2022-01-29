# taken from https://discord.com/channels/689900678990135345/821885387655479326/935856848542793748
# it should produce mhlo output which can be further transformed into c++ code with mlir-emitc repo or iree

from numpy import dtype
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax

f = jit(jax.nn.relu)
l = f.lower(jnp.zeros((50,), dtype=float))

# TODO write it to file
with open('example.mhlo', 'w') as f:
    print(l.compiler_ir("mhlo"), file=f)
