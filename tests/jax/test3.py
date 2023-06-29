"""

"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

@tf.function(jit_compile=True)
def test3(x, y, z):
    return tf.reduce_sum(x + y * z)

print(test3.experimental_get_compiler_ir(1.0, 2.0, 3.0)(stage='hlo'))