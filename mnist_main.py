"""
This is a python script to train a classifier model on the MNIST dataset in JAX. This is meant to serve as an illustrative example as to how we can organise the problem. Explanatory 
notes are provided in detail.

It incorporates a GUI where we can enter certain hyperparameters (batch size, learning rate, number of epochs, and a key) and also choose a model (mlp/cnn). The training log and plots 
of the training and validation loss are saved in a sequence of directories that are named so as to indicate the values of the hyperparameters, and the files are also 
named accordingly. The loss plot is also immediately displayed. One can immediately generalise this to make it more general. 

The required conda environment should be created with conda env create -f utils/create_env.yml
The dependencies should be installed with pip install -r utils/requirements.txt
JAX should be installed manually. For MNIST the cpu version should be good enough.
The script can then be run as python mnist_main.py
We'll need to be in the correct directory first. 

It is possible to modify the script so we can run it as python mnist_main.py hyperparameter1, ..., 'model', the necessary modifications are indicated within this script. 

It is also possible to manually set the hyperparameters and model choices in a separate file called parameters.py and run the script as python mnist_main.py parameters
such that the options are directly read from that file. The necessary modifications are indicated within the script. 

"""




import os
import subprocess                       # to run a script from within .py file
# import sys                            # if we want to pass arguments (such as hyperparameters) through the command line
                                        # or through a separate .py file containing values for the arguments, and then we can pass this file as an argument through the command line


import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx

from jaxtyping import Array, Float      # so we can use these to denote the classes directly instead of having to type jnp.array

 
import torch
from torch.utils.data import Dataloader         # we'll get the dataset from torch


return_code = subprocess.run(["python", "input_gui.py"])      # capture_output is set to the default value False


