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


# training code based on 'https://github.com/patrick-kidger/equinox/blob/main/examples/mnist.ipynb'




import os
import subprocess                       # to run a script from within .py file

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import models
import optax
import torch
import torchvision

from jaxtyping import Array, Float, Int, PyTree      # so we can use these to denote the classes directly instead of having to type jnp.array
from torch.utils.data import DataLoader              # we'll get the dataset from torch



#-------------------------------------------------------------
# process to retrieve hyperparameters entered throught the gui

cwd = os.getcwd()
path_param = os.path.join(cwd,'parameters') 
os.makedirs(path_param, exist_ok=True)                              # if the folder already exists, then it does nothing
                                                                    # it's importtant to give the root path or os.makedirs does not have the required permissions
open(os.path.join(path_param,'gui_parameters_temp.py'), "w")        # gui_parameters_temp.py is the module with the values of the parameters as entered in the gui 
                                                                    # rewritten every time new parameters are entered, mnist_main.py reads parameter values from this module



return_code = subprocess.run(["python", "input_gui.py"])            # capture_output is set to the default value False

from parameters import gui_parameters_temp as ginp



#------------------------------------------------------------
# parameters

model_name = ginp.model
batch_size = ginp.batch_size
lr = ginp.lr
epochs = ginp.epochs
seed = ginp.seed

key = jrand.PRNGKey(seed)



#-----------------------------------------------------------
# alternate methods of entering hyperparameters:

# we could, for example, have a file called hyperparameters.py in the parameters directory where we enter the hyperparameters by hand
# then we could pass the file name as an argument from the terminal. we'd have to have the following in this script.

# import sys
# input_file = sys.argv[-1]               # sys.argc[0] is the name of this script by default
# minp = __import__(input_file)           # we use __import__ as the name of the module is only known during runtime
#
#       model_name = minp.model
#       batch_size = minp.batch_size
#       ...


# we could also directly enter the hyperparameters in the terminal and read them off using sys.argv as follows
# this script would need to have the following:

# import sys
# model_name = sys.argv[1]
# batch_size = sys.argv[2]
# ...

# we'd have to run the script as python mnist_main.py model batch_size ...  (model should be entered as a string as before)

#-----------------------------------------------------------
# dataset loading

normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,),(0.5,))            # mean and std for normalisation are both set to 0.5 for all axes
        ]
)

train_set = torchvision.datasets.MNIST(
        "MNIST",
        train=True,
        download=True,
        transform=normalise_data,
)

test_set = torchvision.datasets.MNIST(
        "MNIST",
        train=False,
        download=True,
        transform=normalise_data,
)

trainloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
)

testloader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True
)


# -----------------------------------------------------------------------
# let's check the shape of the data

# x, y = next(iter(trainloader))
# print(x.size())
# print(y.size())
# print(y)


# -----------------------------------------------------------------------
# the model, we'll import this from a different module where we define the models

key, subkey = jrand.split(key,2)

if model_name=='mlp small':
    model = models.MLP_small(subkey)
    model_type = model.__class__.__name__     # .__class__.__name__ gives us the type of object that model is    
    print(model.__doc__)                      # work on this bit
    # print(model.__repr__)                   # for a detailed description of the model layers
else:
    raise NotImplementedError(f'The model {model_name} has not been implemented in models.py, check if you got the name right.')



#------------------------------------------------------------------------
# define the loss function

def cross_entropy(y: Int[Array, "batch"], pred_y: Float[Array, "batch 10"]) -> Float[Array, ""]:
    '''Average cross-entropy for a batch of predictions.'''

    # pred_y for a single image is the log prob of the image belonging to the ten different classes, hence it's a 10d array

    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y,1), axis=1)                  # this takes log of the predicted probability that the image belongs to it's true class 
    return -jnp.mean(pred_y)


# @eqx.filter_jit      # we don't need to filter out the arrays here as every input is an array
@jax.jit
def batch_cross_entropy_loss(model:model_type, img_batch:Float[Array, "batch 1 28 28"], labels_batch:Int[Array, "batch"]) -> Float[Array, ""]:
    '''Average cross-entropy loss for a batch of images.'''
    
    # we'll need to use vmap to vectorize the operation across the batch

    pred_prob_batch = jax.vmap(model)(img_batch)
    return cross_entropy(labels_batch, pred_prob_batch)




#-----------------------------------------------------------------------
# Evaluation metrics: classification accuracy, loss (across an entire dataset)
# we cannot jit compile these as these are not pure functions and we are using a for loop that is argument dependent

def cross_entropy_loss(model:model_type, set:DataLoader):              
    '''
    Computes the average cross-entropy loss across the entire dataset, train or test.
    The second argument should be either trainloader or testloader.
    '''

    accumulated_loss = 0

    for images_batch, labels_batch in set:
        pred_prob_batch = model(images_batch)
        accumulated_loss += batch_cross_entropy_loss(images_batch,labels_batch)

    return accumulated_loss/len(set)   

def classification_accuracy(model:model_type, set:DataLoader): 
    '''
    Computes the classification accuracy across the entire dataset, train or test.
    The second argument should be either trainloader or testloader.
    '''
    accuracy_accumulated = 0 

    for images_batch, labels_batch in set:
        pred_prob_batch = model(images_batch)
        pred_labels_batch = jnp.argmax(pred_prob_batch, axis=1)
        batch_accuracy = jnp.sum(pred_labels_batch==labels_batch)/batch_size
        accuracy_accumulated += batch_accuracy

    return accuracy_accumulated/len(set)



#----------------------------------------------------------------------------------------
# Training


