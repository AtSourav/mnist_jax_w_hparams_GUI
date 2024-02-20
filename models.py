import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand

from jaxtyping import Array, Float, Int, PyTree


class MLP_small(eqx.Module):
    '''
    This model is a small mlp with three fully connected layers.
    '''
    
    layers: list                  # denotes that the layers are provided as a list
    descript: str                 # syntax in eqx: necessary to specify all fields at the class level, allows it to output a PyTree

    

    def __init__(self, key):
        
        key1, key2, key3 = jrand.split(key,3)     # create three keys from the initial one (deterministic)

        self.layers = [
            jnp.ravel,                                  # for flattening
            eqx.nn.Linear(28*28, 512, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(512, 256, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(256, 10, key=key3),
            jax.nn.log_softmax,                         # cross-entropy loss would just be just a -sum[...]   
        ]

        self.descript = "This model is a small mlp with three fully connected layers."


    def __call__(self, x:Float[Array, "1 28 28"]) -> Float[Array, "10"] :                   # the type and shape of the input and output is provided
        for layer in self.layers:
            x = layer(x)
        return x
    
    def describe(self):
        print(self.descript)