# MLP classifier on the MNIST dataset implemented in JAX with a GUI for entering hyperparameters.

## Summary

The python script mnist_main.py (based on JAX) trains a model of your choice for classification on the MNIST dataset, with hyperparameters (batch size, learning rate, number of epochs, seed, optimizer) that you are prompted to specify through 
a GUI. Right now, the module models.py only contains a small MLP, to add other models you simply need to make the necessary additions to this module. The dataset is loaded using torch. The training data is logged in TensorBoard using PyTorch.
The relevant logdir is results/the_name_of_the_model_that_you_train. The different runs are logged using the time stamp from the run and the value of the hyperparameters. On the TensorBoard interface, you can go to the HPARAMS tab, and the 
different runs using different values of hyperparameters will be shown. For each of these, the plots showing the evolution of training loss, training accuracy, test loss, and test accuracy during the training process are shown. This is not 
a default feature offered by the Pytorch TensorBoard SummaryWriter class but I have subclassed it and added a variation of the add_hparams function to provide this feature. 

## Detailed instructions

Create the necessary conda environment using:
```
conda cenv create -f utils/create_env.yml
```
Install the necessary dependencies using:
```
pip install -r utils/requirements.txt
```
Install JAX:
```
pip install --upgrade "jax[cpu]"             # use the relevant instructions for the gpu version if you need that
```
**Adding models to models.py**: The models in models.py are to be implemented in JAX using the Equinox library. The model that is already implemented in the module models.py is called 'mlp_small'. To add a new model, simple create a class for that model. It is recommended that you add an attribute called descript with a 
small description of the model. Then add this model_name: model_class pair key:value pair to the function model_names_dict(key, model_name) in this module. This will enable mnist_main.py to import the chosen model. 

For now let's say we want to train the model named 'mlp_small'.

**To train**: you simply run `python mnist_main.py` from within the correct conda env. This will open a GUI for entering hyperparameters. 

**Entering hyperparameters**: The GUI is based on tkinter and is implemented in input_gui.py. The box will have six fields, it's self-explanatory and the values to enter are the model name (as a string), batch size, learning rate, number of epochs to train, seed, and the optimizer name (as a string, like 'adam'). You should use one of the optimizers that are implemented in the Optax library. Then you press Save and then press Done. The hyperparameter values are written into parameters/gui_parameters_temp.py (temporarily, until they're overwritten) from where they are read off by the main script. This should start the training process and you should see a progress bar (implemented through tkdm) denoting progress through the steps for each epoch. At the end of the each epoch the training loss, training accuracy, test loss, and test accuracy will be displayed. 

**Results**: The results should be viewed through TensorBoard. The logdir into which the SummaryWriter logs the data is results/model_name. So for mlp_small it's results/mlp_small. To get the link to the interface enter
```tensorboard --logdir results/mlp_small```. Follow the link to open the interface. Go to the HPARAMS tab and you'll see a list of runs with different hyperparameters. Click on the desired one to view the training plots of training loss, training accuracy, test loss, and test accuracy. 

This feature is not offered by default through the SummaryWriter class of PyTorch tensorboard. But I have subclassed it in SummaryWriter_mod.py and then added a function called add_hparams_plot that allows me to offer this visualization. 
