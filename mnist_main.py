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
from tqdm import tqdm

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand

import optax
import torch
import torchvision

from jaxtyping import Array, Float, Int, PyTree      # so we can use these to denote the classes directly instead of having to type jnp.array
from models import *
from torch.utils.data import DataLoader              # we'll get the dataset from torch

from SummaryWriter_mod import SummaryWriter_mod   



#-------------------------------------------------------------
# process to retrieve parameters entered throught the gui

def enter_parameters():      # an array output of mixed type

    cwd = os.getcwd()
    path_param = os.path.join(cwd,'parameters') 
    os.makedirs(path_param, exist_ok=True)                              # if the folder already exists, then it does nothing
                                                                        # it's importtant to give the root path or os.makedirs does not have the required permissions
    open(os.path.join(path_param,'gui_parameters_temp.py'), "w")        # gui_parameters_temp.py is the module with the values of the parameters as entered in the gui 
                                                                        # rewritten every time new parameters are entered, mnist_main.py reads parameter values from this module



    return_code = subprocess.run(["python", "input_gui.py"])            # capture_output is set to the default value False

    from parameters import gui_parameters_temp as ginp

    model_name = ginp.model
    batch_size = ginp.batch_size
    lr = ginp.lr
    epochs = ginp.epochs
    seed = ginp.seed
    optimizer_name = ginp.optimizer

    parameters_dict = {'model name':model_name, 'batch size':batch_size, 'learning rate':lr, 'number of epochs':epochs, 'seed':seed, 'optimizer':optimizer_name}

    return parameters_dict



#-----------------------------------------------------------
# alternate methods of entering parameters:

# we could, for example, have a file called parameters.py in the parameters directory where we enter the parameters by hand
# then we could pass the file name as an argument from the terminal. we'd have to have the following in this script.

# import sys
# input_file = sys.argv[-1]               # sys.argc[0] is the name of this script by default
# minp = __import__(input_file)           # we use __import__ as the name of the module is only known during runtime
#
#       model_name = minp.model
#       batch_size = minp.batch_size
#       ...


# we could also directly enter the parameters in the terminal and read them off using sys.argv as follows
# this script would need to have the following:

# import sys
# model_name = sys.argv[1]
# batch_size = sys.argv[2]
# ...

# we'd have to run the script as python mnist_main.py model batch_size ...  (model should be entered as a string as before)

#-----------------------------------------------------------
# dataset loading

def dataload(batch_size: Int[Array, ""]):       # output is a array of datasets and DataLoader objects

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

    trainloader_entire = DataLoader(
            train_set, batch_size=len(train_set), shuffle=True
    )

    testloader_entire = DataLoader(
            test_set, batch_size=len(test_set), shuffle=True
    )

    return [trainloader_entire, testloader_entire, trainloader, testloader]

#t= dataload(64)
#print(t[0].size)



# -----------------------------------------------------------------------
# we can check the shape of the data as follows

# x, y = next(iter(trainloader))
# print(x.size())
# print(y.size())
# print(y)


# -----------------------------------------------------------------------
# importing the model, we'll import this from a different module where we define the models

def import_model(model_name: str, key: Int[Array, "2"]) -> eqx.Module:

    key, subkey = jrand.split(key,2)

    model = model_names_dict(subkey, model_name)
    if hasattr(model, 'descript'):
        print("\n"+model.descript+"\n")
        # model.describe()                          # a short description of the model
        # print(model.__repr__)                   # for a detailed description of the model layers
        # model_type = model.__class__.__name__     # .__class__.__name__ gives us the type of object that model is 

    return model




#------------------------------------------------------------------------
# define the loss function and accuracy

def cross_entropy(y: Int[Array, "batch"], pred_y: Float[Array, "batch 10"]) -> Float[Array, ""]:
    '''Average cross-entropy for a batch of predictions.'''

    # pred_y for a single image is the log prob of the image belonging to the ten different classes, hence it's a 10d array

    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y,1), axis=1)                  # this takes log of the predicted probability that the image belongs to it's true class 
    return -jnp.mean(pred_y)



@eqx.filter_jit                    # I'd have thought that jax.jit works too since we're not working with anything other than an array, but we get an error saying: Cannot interpret value of type <class 'jaxlib.xla_extension.PjitFunction'> as an abstract array; it does not have a dtype attribute
def batch_cross_entropy_loss(model: eqx.Module, img_batch:Float[Array, "batch 1 28 28"], labels_batch:Int[Array, "batch"]) -> Float[Array, ""]:
    '''
    Average cross-entropy loss for a batch of images.
    '''
    
    # we're not specifying the object type of the input variable 'model' as that is dependent on which model we choose to use
    # we'll need to use vmap to vectorize the operation across the batch

    pred_prob_batch = jax.vmap(model)(img_batch)
    return cross_entropy(labels_batch, pred_prob_batch)


@eqx.filter_jit
def batch_classification_accuracy(model: eqx.Module, img_batch:Float[Array, "batch 1 28 28"], labels_batch:Int[Array, "batch"]) -> Float[Array, ""]:
    '''
    Computes the classification accuracy for a batch of images.
    '''

    pred_prob = jax.vmap(model)(img_batch)
    pred_labels = jnp.argmax(pred_prob, axis=1)
    accuracy = jnp.sum(pred_labels==labels_batch)/len(labels_batch)

    return accuracy


#----------------------------------------------------------------------------------------
# Training


@eqx.filter_jit                                     
def make_step(optim: optax.GradientTransformation, model: eqx.Module, opt_state:PyTree, img_batch: Float[Array, "batch 1 28 28"], labels_batch:Int[Array, "batch"]):
    
    _, grads = eqx.filter_value_and_grad(batch_cross_entropy_loss)(model, img_batch, labels_batch)           # eqx.filter_value_and_grad(f)(x1,x2,...) computes f(x1,x2,...), df/dx1 w.r.t. the array components of x1
                                                                                                             # in this case we don't need to capture the value of the loss
    updates, opt_state = optim.update(grads, opt_state, model) 
    model = eqx.apply_updates(model, updates)

    return model, opt_state


def infinite_dataloader(dataloader: DataLoader):                                     # to loop over the training dataset any number of times
    while True:
        yield from dataloader                                  # gives a generator object that has to be iterated to get the values (the different batches in this case)


def train(model:eqx.Module, writer:SummaryWriter_mod, hparams_dict: dict, trainloader_entire:DataLoader, testloader_entire:DataLoader, trainloader: DataLoader, testloader: DataLoader, optim: optax.GradientTransformation, epochs:Int) -> eqx.Module:
    
    opt_state = optim.init(eqx.filter(model, eqx.is_array))                     # filtering out the array components of the model coz we can only train those

    steps_per_epoch = len(trainloader)

    img_train_all, labels_train_all = next(iter(trainloader_entire))            # using the infinite_dataloader function gives ValueError: too many values to unpack(expected 2)
    img_test_all, labels_test_all = next(iter(testloader_entire))               # some incompatibility with the generator object produced by yield I suppose

    img_train_all = img_train_all.numpy()
    img_test_all = img_test_all.numpy()
    labels_train_all = labels_train_all.numpy()
    labels_test_all = labels_test_all.numpy()

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        pbar = tqdm(total=steps_per_epoch, desc=f'Epoch {epoch+1}. Steps')
        for step, (img_batch, labels_batch) in zip(range(steps_per_epoch), infinite_dataloader(trainloader)):                # zip() takes iterables, aggregates them in a tuple and returns that
               
            img_batch = img_batch.numpy()
            labels_batch = labels_batch.numpy()                                                                    # converting torch tensors to numpy arrays
            model, opt_state = make_step(optim, model, opt_state, img_batch, labels_batch)

            train_loss = batch_cross_entropy_loss(model,img_train_all, labels_train_all)                           # is it better to use the vectorized function or better to do this in a loop over the batches when calculating for the entire dataset?
            test_loss = batch_cross_entropy_loss(model,img_test_all, labels_test_all)
            train_acc = batch_classification_accuracy(model,img_train_all, labels_train_all)
            test_acc = batch_classification_accuracy(model,img_test_all, labels_test_all)

            train_loss_list.append(train_loss.item())
            test_loss_list.append(test_loss.item())
            train_acc_list.append(train_acc.item())
            test_acc_list.append(test_acc.item())


            pbar.update()
        pbar.close()
        
        print(f"training loss={train_loss}, training accuracy={train_acc}")
        print(f"test set loss={test_loss}, test set accuracy={test_acc}\n")

    metric_dict = {'train loss':train_loss_list, 'test loss':test_loss_list, 'train acc':train_acc_list, 'test acc':test_acc_list}    
    writer.add_hparams_plot(hparams_dict, metric_dict)

    return model


#--------------------------------------------------------------------------------------
# Main run function

def run():

    params_dict = enter_parameters()
    hparams_dict = {key: params_dict[key] for key in params_dict.keys() & {'batch size', 'learning rate', 'number of epochs', 'seed', 'optimizer'}}
    # hparams_dict is a subset of the params_dict with all the parameters other than model name. W'll assign different models to different folders.

    params = list(params_dict.values())
    model_name = params[0]
    batch_size = params[1]
    lr = params[2]
    epochs = params[3]
    seed = params[4]
    optimizer_name = params[5]

    key = jrand.PRNGKey(seed)

    data = dataload(batch_size)
    trainloader_entire  = data[0]
    testloader_entire = data[1]
    trainloader = data[2]
    testloader = data[3]

    model = import_model(model_name, key)

    optim = getattr(optax, optimizer_name)(learning_rate=lr)                             # getattr gets the chosen optax optimizer from the optax module

    print("Training with the "+optimizer_name+f" optimizer with a learning rate of {lr} for {epochs} epochs.\n")

    cwd = os.getcwd()
    results_path = os.path.join(cwd,'results/'+ model_name)
    os.makedirs(results_path, exist_ok=True)

    writer = SummaryWriter_mod(log_dir=results_path)

    train(model, writer, hparams_dict, trainloader_entire, testloader_entire, trainloader, testloader, optim, epochs)

    writer.flush()
    writer.close()

    return model




#---------------------------------------------------------------------------------------
# Running the training from the terminal

if __name__=="__main__":                      # to prevent the run() command from being executed (as we're calling it below) if this script is imported as a module
    
    run()



   



